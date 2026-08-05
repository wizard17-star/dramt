"""Fold-sensitivity analysis: how much of a conclusion rests on one test block?

Motivation. Fold 0 (2023-03 – 2023-09) was a strong low-volatility rally: the
10-day portfolio return averaged +1.83% with a worst loss of −6.07% and only
25% of windows negative. NO model breaches its 95% VaR there, and zero
breaches in 115 observations is itself improbable under a 5% rate, so every
model — including the martingale null — fails Kupiec on that fold. Any
statistic pooled as a minimum over folds is therefore dominated by a single
unusually calm block rather than by model quality.

This recomputes the headline metrics with each fold left out in turn. A
conclusion that survives every leave-one-fold-out variant is robust; one that
only holds with a particular fold included should be stated as such.

Usage:
    python -m src.sensitivity
Writes results/analysis_fold_sensitivity.csv.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _per_fold(results_dir: Path, run: str) -> list[dict] | None:
    p = results_dir / f"eval_{run}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    args = ap.parse_args()

    base = load_config("config.yaml")
    results_dir = Path(base["paths"]["results_dir"])
    summary = pd.read_csv(results_dir / "eval_summary.csv")
    run_names = args.runs or list(summary["run"])

    rows = []
    for run in run_names:
        folds = _per_fold(results_dir, run)
        if not folds:
            continue
        n = len(folds)
        mae = [f["point"]["MAE"] for f in folds]
        diracc = [f["point"]["DirAcc"] for f in folds]
        breach = [f["var_backtest"]["breach_rate"] if "var_backtest" in f else np.nan
                  for f in folds]
        for drop in [None, *range(n)]:
            keep = [i for i in range(n) if i != drop]
            rows.append({
                "run": run,
                "dropped_fold": "none" if drop is None else drop,
                "MAE": float(np.mean([mae[i] for i in keep])),
                "DirAcc": float(np.mean([diracc[i] for i in keep])),
                "VaRBreach": float(np.nanmean([breach[i] for i in keep])),
            })

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "analysis_fold_sensitivity.csv", index=False)
    logger.info("wrote %s (%d rows)", results_dir / "analysis_fold_sensitivity.csv",
                len(df))

    # ---- report: does the headline ordering survive leaving a fold out? ----
    focus = [r for r in ["baseline_martingale", "baseline_arima", "baseline_garch",
                         "baseline_har_rv", "dramt_tg_ensemble", "dramt_ensemble",
                         "dramt_t_garch", "baseline_tft"] if r in set(df["run"])]
    sub = df[df["run"].isin(focus)]
    print("\nMAE by leave-one-fold-out (lower is better):")
    print(sub.pivot(index="run", columns="dropped_fold", values="MAE")
          .round(5).to_string())
    print("\n10-day VaR breach rate by leave-one-fold-out (nominal 0.05):")
    print(sub.pivot(index="run", columns="dropped_fold", values="VaRBreach")
          .round(4).to_string())

    # is the martingale still competitive in every variant?
    piv = df.pivot(index="run", columns="dropped_fold", values="MAE")
    if "baseline_martingale" in piv.index:
        print("\nModels beating the martingale null, per variant:")
        for col in piv.columns:
            better = (piv[col] < piv.loc["baseline_martingale", col]).sum()
            print(f"  dropped_fold={col}: {better}/{len(piv)} models beat the null")


if __name__ == "__main__":
    main()
