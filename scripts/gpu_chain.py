"""Full GPU-era results chain. Run AFTER src.sweep has produced best.yaml.

Everything downstream of the sweep, in dependency order. Resumable: a step
whose outputs already exist is skipped unless --force is given, so the chain
can be restarted after an interruption without redoing hours of training.

Steps
  1  baselines           10 econometric/deep baselines + TFT, all folds
  2  risk variants       2x2 train-time grid isolating the RQ2 changes:
                           dist in {gaussian, student_t}
                           vol_mode in {learned, garch_hybrid}
                         (the third RQ2 change, rolling sigma calibration, is
                         an EVALUATION-time choice, so it is applied to every
                         run in step 6 rather than needing its own training)
  3  RQ3 variants        extended gate signals, per-timestep gating, and both
  4  ablations           the 6 existing ablation configs
  5  seed ensemble       N seeds of the best config + the combined ensemble
  6  evaluate            every run under both global and rolling calibration
  7  stats/tables/plots

Usage:
    python -m scripts.gpu_chain                 # everything
    python -m scripts.gpu_chain --only 1 2      # selected steps
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

RUNS = Path("runs")
EXP = Path("experiments")

# risk variants: (name, dist, vol_mode)
RISK_VARIANTS = [
    ("dramt_gpu",            "gaussian",  "learned"),       # full-capacity reference
    ("dramt_t",              "student_t", "learned"),
    ("dramt_garch",          "gaussian",  "garch_hybrid"),
    ("dramt_t_garch",        "student_t", "garch_hybrid"),
]

# RQ3 variants: (name, regime_signals, gate_per_timestep)
RQ3_VARIANTS = [
    ("dramt_gate_ext",       "extended", False),
    ("dramt_gate_time",      "basic",    True),
    ("dramt_gate_ext_time",  "extended", True),
]

ENSEMBLE_SEEDS = list(range(42, 52))   # 10 seeds


def run(args: list[str], label: str = "") -> bool:
    t0 = time.time()
    print(f"[chain] START {label or ' '.join(args)}", flush=True)
    proc = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True)
    ok = proc.returncode == 0
    tail = "\n".join(proc.stdout.splitlines()[-4:] + proc.stderr.splitlines()[-6:])
    print(f"[chain] {'OK' if ok else f'FAIL rc={proc.returncode}'} "
          f"({time.time() - t0:.0f}s) {label or ' '.join(args)}\n{tail}", flush=True)
    return ok


def done(run_name: str, n_folds: int = 4) -> bool:
    return len(list((RUNS / run_name).glob("fold*/test_predictions.npz"))) >= n_folds


def write_exp(name: str, best: dict, **overrides) -> Path:
    EXP.mkdir(exist_ok=True)
    cfg = {**best, "name": name, "dataset_suffix": "", **overrides}
    path = EXP / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-folds", type=int, default=4)
    args = ap.parse_args()

    def want(step: int) -> bool:
        return args.only is None or step in args.only

    best_path = RUNS / "sweep" / "best.yaml"
    if not best_path.exists():
        sys.exit(f"{best_path} missing - run `python -m src.sweep --folds all` first")
    best = yaml.safe_load(best_path.read_text(encoding="utf-8"))
    best.pop("name", None)
    print(f"[chain] best swept config: {best}", flush=True)

    # ---- 1. baselines ----------------------------------------------------
    if want(1):
        for model in ["arima", "garch", "garch_midas", "dcc_garch", "lstm", "gru",
                      "cnn_bilstm", "cnn_bilstm_attn", "transformer",
                      "sentiment_lstm", "tft"]:
            if not args.force and done(f"baseline_{model}", args.n_folds):
                print(f"[chain] SKIP baseline_{model} (already complete)", flush=True)
                continue
            run(["src.baselines", "--model", model, "--suffix", ""], f"baseline {model}")

    # ---- 2. risk variants ------------------------------------------------
    if want(2):
        for name, dist, vol_mode in RISK_VARIANTS:
            if not args.force and done(name, args.n_folds):
                print(f"[chain] SKIP {name} (already complete)", flush=True)
                continue
            p = write_exp(name, best, dist=dist, vol_mode=vol_mode)
            run(["src.train", "--config", str(p)], f"risk variant {name}")

    # ---- 3. RQ3 variants -------------------------------------------------
    if want(3):
        for name, signals, per_step in RQ3_VARIANTS:
            if not args.force and done(name, args.n_folds):
                print(f"[chain] SKIP {name} (already complete)", flush=True)
                continue
            p = write_exp(name, best, regime_signals=signals,
                          gate_per_timestep=per_step)
            run(["src.train", "--config", str(p)], f"RQ3 variant {name}")

    # ---- 4. ablations ----------------------------------------------------
    if want(4):
        run(["src.ablation", "--suffix", ""], "ablations")

    # ---- 5. seed ensemble ------------------------------------------------
    if want(5):
        members = []
        for seed in ENSEMBLE_SEEDS:
            name = f"dramt_seed{seed}"
            members.append(name)
            if not args.force and done(name, args.n_folds):
                print(f"[chain] SKIP {name} (already complete)", flush=True)
                continue
            p = write_exp(name, best, seed=seed)
            run(["src.train", "--config", str(p)], f"seed {seed}")
        run(["src.ensemble", "--runs", *members, "--out", "dramt_ensemble"],
            "seed ensemble")

    # ---- 6. evaluate under both calibrations ------------------------------
    if want(6):
        run(["src.evaluate", "--calibration", "global", "--suffix", "_global"],
            "evaluate (global calibration)")
        run(["src.evaluate", "--calibration", "rolling", "--suffix", "_rolling"],
            "evaluate (rolling calibration)")
        # canonical summary used by tables/plots
        run(["src.evaluate", "--calibration", "rolling"], "evaluate (canonical)")

    # ---- 7. stats, tables, figures ---------------------------------------
    if want(7):
        run(["src.stats_tests", "--reference", "dramt_ensemble"], "significance tests")
        run(["src.tables"], "LaTeX/CSV tables")
        run(["src.utils.plotting", "--run", "dramt_ensemble"], "figures")

    print("[chain] complete", flush=True)


if __name__ == "__main__":
    main()
