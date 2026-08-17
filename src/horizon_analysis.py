"""Horizon-resolved point accuracy.

The headline result pools h = 1..10. That pooling could hide a real edge at
short horizons, or manufacture one, so this recomputes MAE per horizon for
every run and counts how many beat the constant-zero forecast at each h.

The null is exact: MAE of predicting zero is mean(|y|) on the same anchors,
so no separate baseline run is needed and the comparison cannot drift.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path("runs")
OUT = Path("results/analysis_horizon.csv")


def load_run(run_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Concatenate (mu, y) over folds -> arrays of shape (N, S, H)."""
    mus, ys = [], []
    for f in sorted(run_dir.glob("fold*/test_predictions.npz")):
        z = np.load(f)
        if "mu" not in z or "y_ret" not in z:
            return None
        mu, y = z["mu"], z["y_ret"]
        if mu.ndim != 3 or mu.shape != y.shape:
            return None
        mus.append(mu)
        ys.append(y)
    if not mus:
        return None
    return np.concatenate(mus), np.concatenate(ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rows = []
    truth: np.ndarray | None = None
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        got = load_run(d)
        if got is None:
            continue
        mu, y = got
        if truth is None:
            truth = y
        elif truth.shape == y.shape and not np.allclose(truth, y, atol=0, rtol=0):
            raise SystemExit(f"{d.name}: targets differ from earlier runs; "
                             "the anchors are not aligned")
        for h in range(mu.shape[2]):
            yh = y[:, :, h]
            rows.append({
                "run": d.name,
                "horizon": h + 1,
                "MAE": float(np.abs(mu[:, :, h] - yh).mean()),
                "null_MAE": float(np.abs(yh).mean()),
                # Constant equal to the realised test mean. This looks ahead and
                # is not a legitimate forecast: it is the best any constant could
                # possibly do, so it bounds how much of a model's edge over the
                # zero forecast is drift rather than conditional skill.
                "oracle_drift_MAE": float(np.abs(yh - yh.mean()).mean()),
                "mean_pred": float(mu[:, :, h].mean()),
                "mean_actual": float(yh.mean()),
                "n_obs": int(yh.size),
            })

    if not rows:
        raise SystemExit("no runs with per-horizon predictions found")

    df = pd.DataFrame(rows)
    df["beats_null"] = df.MAE < df.null_MAE
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    summary = df.groupby("horizon").agg(
        null_MAE=("null_MAE", "first"),
        best_MAE=("MAE", "min"),
        median_MAE=("MAE", "median"),
        drift_MAE=("oracle_drift_MAE", "first"),
        beating=("beats_null", "sum"),
        n_runs=("run", "nunique"),
    )
    summary["best_gain_pct"] = 100 * (summary.null_MAE - summary.best_MAE) / summary.null_MAE
    summary["drift_gain_pct"] = (
        100 * (summary.null_MAE - summary.drift_MAE) / summary.null_MAE)
    # share of the best model's edge that a look-ahead constant already explains
    summary["drift_share_pct"] = 100 * summary.drift_gain_pct / summary.best_gain_pct
    print(f"runs with per-horizon predictions: {df.run.nunique()}")
    print(summary.round(5).to_string())
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
