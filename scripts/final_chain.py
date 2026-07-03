"""Final results chain: run once the GDELT case-study pull is (sufficiently)
complete. Detached; log -> runs/final_chain.log.

Steps:
 1. build features WITH real sentiment (FinBERT over all cached GDELT articles)
 2. DRAM-T final all-fold run (best swept config, real sentiment)
 3. sentiment_lstm baseline all folds
 4. 6 ablation configs, all folds
 5. evaluate all runs -> eval_summary.csv
 6. significance tests -> stats_*.csv
 7. tables + figures regeneration
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import yaml


def run(args: list[str]) -> bool:
    t0 = time.time()
    print(f"[final] START {' '.join(args)}", flush=True)
    proc = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True)
    tail = "\n".join(proc.stdout.splitlines()[-3:] + proc.stderr.splitlines()[-3:])
    ok = proc.returncode == 0
    print(f"[final] {'OK' if ok else f'FAIL rc={proc.returncode}'} "
          f"({time.time() - t0:.0f}s) {' '.join(args)}\n{tail}", flush=True)
    return ok


def main() -> None:
    # final experiment config: swept best hyperparameters, REAL sentiment dataset
    best = yaml.safe_load(Path("runs/sweep/best.yaml").read_text(encoding="utf-8"))
    best["dataset_suffix"] = ""
    best["name"] = "dramt_final"
    Path("experiments/final.yaml").write_text(yaml.safe_dump(best), encoding="utf-8")

    # seed-robustness variants of the final model (same config, seeds 43/44)
    for seed in (43, 44):
        variant = dict(best, seed=seed, name=f"dramt_final_s{seed}")
        Path(f"experiments/final_s{seed}.yaml").write_text(
            yaml.safe_dump(variant), encoding="utf-8")

    steps: list[list[str]] = [
        ["src.data.build_features"],                                   # incl. FinBERT scoring
        ["src.train", "--config", "experiments/final.yaml"],
        ["src.train", "--config", "experiments/final_s43.yaml"],
        ["src.train", "--config", "experiments/final_s44.yaml"],
        ["src.baselines", "--model", "sentiment_lstm", "--suffix", ""],
        ["src.ablation", "--suffix", ""],
        ["src.evaluate"],
        ["src.stats_tests", "--reference", "dramt_final"],
        ["src.tables"],
        ["src.utils.plotting", "--run", "dramt_final"],
    ]
    for step in steps:
        if not run(step):
            print(f"[final] chain STOPPED at {step}", flush=True)
            return
    print("[final] chain complete", flush=True)


if __name__ == "__main__":
    main()
