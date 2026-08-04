"""Full GPU-era results chain. Run AFTER src.sweep has produced best.yaml.

Everything downstream of the sweep, in dependency order. Resumable: a step
whose outputs already exist is skipped unless --force is given, so the chain
can be restarted after an interruption without redoing hours of training.

Steps (note: 9 runs between 3 and 4)
  1  baselines           13 baselines, all folds: martingale null, HAR-RV,
                         ARIMA, GARCH family, 6 deep models, TFT
  2  risk variants       train-time grid isolating the RQ2 changes:
                           dist in {gaussian, student_t}
                           vol_mode in {learned, garch_hybrid, har_hybrid}
                         (the third RQ2 change, rolling sigma calibration, is
                         an EVALUATION-time choice, so it is applied to every
                         run in step 6 rather than needing its own training)
  3  RQ3 variants        extended gate signals, per-timestep gating, and both
  9  objective variants  lambda1 rebalancing, risk-only, ranking loss
  4  ablations           the 6 existing ablation configs
  5  seed ensemble       N seeds of the best config + the combined ensemble
  6  evaluate            every run under both global and rolling calibration
  7  stats/tables/plots
  8  verify              anchor comparability + Wilcoxon coverage check

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
N_HORIZONS = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))[
    "windowing"]["horizons"]

# risk variants: (name, dist, vol_mode)
RISK_VARIANTS = [
    ("dramt_gpu",            "gaussian",  "learned"),       # full-capacity reference
    ("dramt_t",              "student_t", "learned"),
    ("dramt_garch",          "gaussian",  "garch_hybrid"),
    ("dramt_t_garch",        "student_t", "garch_hybrid"),
    ("dramt_t_har",          "student_t", "har_hybrid"),
]

# Objective variants. Motivated by a measured diagnostic: a constant-ZERO
# forecast beats the trained model on this test set (MAE 0.03030 vs 0.03070,
# R^2 = -0.026), while realized volatility has lag-1 autocorrelation 0.92.
# The composite loss spends most of its weight on the unlearnable mean and
# down-weights the learnable variance by lambda1=0.1. These variants test
# whether rebalancing recovers the volatility signal.
#   (name, lambda1, point_loss, rank_weight)
OBJECTIVE_VARIANTS = [
    ("dramt_lam1_2",      2.0,  "huber", 0.0),
    ("dramt_lam1_5",      5.0,  "huber", 0.0),
    ("dramt_riskonly",    1.0,  "none",  0.0),   # no mean term at all
    ("dramt_rank",        0.1,  "huber", 1.0),   # + cross-sectional ranking
    ("dramt_rank_only",   0.1,  "none",  1.0),   # ranking replaces the mean term
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
    """True only if the run is complete AND was produced under the CURRENT
    horizon set.

    A run left over from the CPU-era study has 6 horizons instead of 10; it
    would satisfy a naive file-count check and be skipped, silently mixing
    stale predictions into the new comparison tables. Checking the horizon
    dimension makes that impossible.
    """
    import numpy as np

    paths = sorted((RUNS / run_name).glob("fold*/test_predictions.npz"))
    if len(paths) < n_folds:
        return False
    want_h = len(N_HORIZONS)
    for p in paths:
        if np.load(p)["mu"].shape[-1] != want_h:
            print(f"[chain] {run_name} exists but has "
                  f"{np.load(p)['mu'].shape[-1]} horizons (want {want_h}) - rerunning",
                  flush=True)
            return False
    return True


def write_exp(name: str, best: dict, **overrides) -> Path:
    EXP.mkdir(exist_ok=True)
    cfg = {**best, "name": name, "dataset_suffix": "", **overrides}
    path = EXP / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def verify_comparability(n_folds: int) -> None:
    """Fail loudly if the results are not actually comparable.

    src/stats_tests silently SKIPS any run whose test anchors differ from the
    reference (logged as a warning that is easy to miss in a long chain log).
    The visible symptom would be a Wilcoxon table quietly missing rows - i.e.
    a thesis table with no baseline comparisons - so it is checked explicitly.
    """
    import csv

    import numpy as np

    print("[chain] verifying anchor comparability across runs", flush=True)
    anchors: dict[str, np.ndarray] = {}
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or d.name == "sweep":
            continue
        paths = sorted(d.glob("fold*/test_predictions.npz"))
        if len(paths) < n_folds:
            continue
        anchors[d.name] = np.concatenate([np.load(p)["anchor_dates"] for p in paths])

    if not anchors:
        print("[chain] WARNING: no completed runs found to verify", flush=True)
        return

    ref_name, ref = next(iter(anchors.items()))
    bad = [n for n, a in anchors.items()
           if a.shape != ref.shape or not np.array_equal(a, ref)]
    if bad:
        print(f"[chain] ERROR: {len(bad)} run(s) have a DIFFERENT test anchor set "
              f"than {ref_name}: {bad}. Paired significance tests against these "
              f"will be skipped. Most likely cause: they were trained at a "
              f"different window length T.", flush=True)
    else:
        print(f"[chain] OK: all {len(anchors)} runs share an identical "
              f"{len(ref)}-anchor test set", flush=True)

    wpath = Path("results/stats_wilcoxon.csv")
    if wpath.exists():
        with open(wpath, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        compared = {r["model_B"] for r in rows}
        missing = sorted(set(anchors) - compared - {rows[0]["model_A"]} if rows else [])
        print(f"[chain] Wilcoxon table: {len(rows)} comparisons"
              + (f"; MISSING: {missing}" if missing else "; all runs covered"),
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, default=None,
                    help="steps: 1 baselines, 2 risk, 3 RQ3, 9 objectives, "
                         "4 ablations, 5 seeds, 6 evaluate, 7 tables, 8 verify")
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
        for model in ["martingale", "har_rv", "arima", "garch", "garch_midas",
                      "dcc_garch", "lstm", "gru", "cnn_bilstm", "cnn_bilstm_attn",
                      "transformer", "sentiment_lstm", "tft"]:
            if not args.force and done(f"baseline_{model}", args.n_folds):
                print(f"[chain] SKIP baseline_{model} (already complete)", flush=True)
                continue
            # T must match the swept DRAM-T config, otherwise the anchor sets
            # differ and every paired significance test against this baseline
            # is silently dropped by src/stats_tests.
            run(["src.baselines", "--model", model, "--suffix", "",
                 "--T", str(best["T"])], f"baseline {model} (T={best['T']})")

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

    # ---- 3b. objective variants (loss rebalancing / ranking) --------------
    if want(9):
        for name, lam1, point_loss, rank_w in OBJECTIVE_VARIANTS:
            if not args.force and done(name, args.n_folds):
                print(f"[chain] SKIP {name} (already complete)", flush=True)
                continue
            p = write_exp(name, best, lambda1=lam1, point_loss=point_loss,
                          rank_weight=rank_w, dist="student_t")
            run(["src.train", "--config", str(p)], f"objective variant {name}")

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

    # ---- 8. consistency check --------------------------------------------
    if want(8):
        verify_comparability(args.n_folds)

    print("[chain] complete", flush=True)


if __name__ == "__main__":
    main()
