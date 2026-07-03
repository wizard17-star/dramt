"""Uniform evaluation across all runs: point + risk metrics per fold, aggregated.

Reads runs/<run>/fold<k>/test_predictions.npz produced by train.py / baselines.py
and writes results/eval_<run>.json + a combined results/eval_summary.csv.

Risk-metric protocol:
- Vol RMSE: predicted sigma vs realized vol, all horizons where sigma exists.
- CRPS / 95% interval coverage: normal predictive distribution N(mu, sigma).
- Portfolio VaR backtest (10-day, 95%): mu_p = w^T mu, sigma_p from D R D.
  Models WITHOUT a correlation output use the static correlation matrix of
  TRAIN-fold daily returns (no test information; documented). Models without
  any sigma output (point-only) are excluded from risk metrics.
- Kupiec POF + Christoffersen independence on the breach series.

Usage:
    python -m src.evaluate                # evaluates every run under runs/
    python -m src.evaluate --runs dramt_full baseline_garch
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.splits import walk_forward_folds
from src.utils.config import load_config
from src.utils.metrics import point_metrics
from src.utils.var_es import (
    breaches,
    christoffersen_independence,
    crps_normal,
    interval_coverage,
    kupiec_pof,
    portfolio_moments,
    var_es_normal,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _train_static_corr(daily: pd.DataFrame, portfolio: list[str],
                       anchor_dates_ns: np.ndarray, train_idx_end_date: pd.Timestamp) -> np.ndarray:
    rets = daily[[f"{t}_log_return" for t in portfolio]].loc[:train_idx_end_date]
    return np.corrcoef(rets.to_numpy().T)


def _val_sigma_scale(npz_path: Path) -> float | None:
    """Post-hoc volatility calibration factor from VALIDATION predictions:
    s = sqrt(E[(y-mu)^2] / E[sigma^2]) so that val residuals are unit-scaled
    by s*sigma. Uses no test information."""
    val_path = npz_path.parent / "val_predictions.npz"
    if not val_path.exists():
        return None
    v = np.load(val_path)
    resid2 = float(np.mean((v["y_ret"] - v["mu"]) ** 2))
    sig2 = float(np.mean(v["sigma"] ** 2))
    if sig2 <= 0:
        return None
    return float(np.sqrt(resid2 / sig2))


def evaluate_fold(npz_path: Path, horizons: list[int], weights: np.ndarray,
                  static_corr: np.ndarray | None, conf: float = 0.95) -> dict:
    z = np.load(npz_path)
    mu, y_ret, y_vol = z["mu"], z["y_ret"], z["y_vol"]
    sigma = z["sigma"] if z["sigma"].size else None
    corr = z["corr"] if z["corr"].size else None

    out: dict = {"point": point_metrics(y_ret, mu)}
    if sigma is None:
        return out

    # apply validation-fitted sigma calibration when available (documented:
    # post-hoc recalibration, no test leakage; raw sigma metrics kept too)
    s = _val_sigma_scale(npz_path)
    if s is not None:
        out["sigma_calibration"] = s
        out["risk_raw"] = {
            "VolRMSE": float(np.sqrt(np.mean((sigma - y_vol) ** 2))),
            "CRPS": crps_normal(y_ret, mu, sigma),
            "Coverage95": interval_coverage(y_ret, mu, sigma, conf),
        }
        sigma = sigma * s

    h10 = horizons.index(10)
    out["risk"] = {
        "VolRMSE": float(np.sqrt(np.mean((sigma - y_vol) ** 2))),
        "CRPS": crps_normal(y_ret, mu, sigma),
        "Coverage95": interval_coverage(y_ret, mu, sigma, conf),
    }

    # portfolio VaR backtest at the 10-day horizon
    R = corr if corr is not None else np.tile(static_corr, (len(mu), 1, 1))
    # sigma is per-day realized-vol units; 10-day cumulative sigma = sqrt(10)*per-day
    mu_h, sig_h = mu[:, :, h10], sigma[:, :, h10] * np.sqrt(10.0)
    mu_p, sigma_p = portfolio_moments(mu_h, sig_h, R, weights)
    var10, es10 = var_es_normal(mu_p, sigma_p, conf)
    realized_p = y_ret[:, :, h10] @ weights
    br = breaches(realized_p, var10)
    exceed = realized_p[br] if br.any() else np.array([])
    out["var_backtest"] = {
        "breach_rate": float(br.mean()),
        "nominal": 1.0 - conf,
        "kupiec": kupiec_pof(br, conf),
        "christoffersen": christoffersen_independence(br),
        "mean_ES_pred": float(es10.mean()),
        "mean_exceed_loss": float(-exceed.mean()) if exceed.size else np.nan,
        "var_series_mean": float(var10.mean()),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=None)
    args = parser.parse_args()

    base = load_config("config.yaml")
    runs_dir = Path(base["paths"]["runs_dir"])
    results_dir = Path(base["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    horizons = [int(h) for h in base["windowing"]["horizons"]]
    weights = np.array(base["portfolio"]["weights"], dtype=float)
    portfolio = base["portfolio"]["members"]
    daily = pd.read_parquet(Path(base["paths"]["processed_dir"]) / "aligned_daily.parquet")

    run_names = args.runs or sorted(
        d.name for d in runs_dir.iterdir()
        if d.is_dir() and list(d.glob("fold*/test_predictions.npz"))
    )

    summary_rows = []
    for run in run_names:
        fold_paths = sorted((runs_dir / run).glob("fold*/test_predictions.npz"))
        per_fold = []
        for p in fold_paths:
            z = np.load(p)
            dates = pd.to_datetime(z["anchor_dates"], unit="ns")
            # static corr from strictly-before-test data (train+gap end = first test date)
            static_corr = _train_static_corr(daily, portfolio, z["anchor_dates"],
                                             dates.min() - pd.Timedelta(days=45))
            per_fold.append(evaluate_fold(p, horizons, weights, static_corr))
        (results_dir / f"eval_{run.replace('/', '_')}.json").write_text(
            json.dumps(per_fold, indent=2, default=float))

        row: dict = {"run": run, "n_folds": len(per_fold)}
        for key in ["MAE", "MSE", "RMSE", "MAPE", "DirAcc"]:
            vals = [f["point"][key] for f in per_fold]
            row[key], row[f"{key}_std"] = float(np.mean(vals)), float(np.std(vals))
        if all("risk" in f for f in per_fold) and per_fold:
            for key in ["VolRMSE", "CRPS", "Coverage95"]:
                vals = [f["risk"][key] for f in per_fold]
                row[key], row[f"{key}_std"] = float(np.mean(vals)), float(np.std(vals))
            row["VaRBreach"] = float(np.mean([f["var_backtest"]["breach_rate"] for f in per_fold]))
            row["Kupiec_p_min"] = float(np.min([f["var_backtest"]["kupiec"]["pvalue"] for f in per_fold]))
            row["Christof_p_min"] = float(np.min(
                [f["var_backtest"]["christoffersen"]["pvalue"] for f in per_fold]))
        summary_rows.append(row)
        logger.info("%s: %s", run, {k: round(v, 4) for k, v in row.items()
                                    if isinstance(v, float) and not k.endswith("_std")})

    pd.DataFrame(summary_rows).to_csv(results_dir / "eval_summary.csv", index=False)
    logger.info("wrote %s", results_dir / "eval_summary.csv")


if __name__ == "__main__":
    main()
