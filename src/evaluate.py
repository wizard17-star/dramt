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
    acerbi_szekely_z2,
    breaches,
    christoffersen_independence,
    crps_normal,
    crps_student_t,
    interval_coverage,
    interval_coverage_t,
    kupiec_pof,
    portfolio_moments,
    student_t_std_factor,
    var_es_normal,
    var_es_student_t,
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
    by s*sigma. Uses no test information.

    This is a SINGLE constant per fold. That is exactly the weakness the
    fold-2 VaR blow-up exposed: one number cannot track a volatility regime
    that shifts inside the test block. See _rolling_sigma_scale.
    """
    val_path = npz_path.parent / "val_predictions.npz"
    if not val_path.exists():
        return None
    v = np.load(val_path)
    resid2 = float(np.mean((v["y_ret"] - v["mu"]) ** 2))
    sig2 = float(np.mean(v["sigma"] ** 2))
    if sig2 <= 0:
        return None
    return float(np.sqrt(resid2 / sig2))


def _rolling_sigma_scale(npz_path: Path, horizons: list[int], window: int = 120,
                         min_obs: int = 20) -> np.ndarray | None:
    """Regime-adaptive volatility calibration: a per-anchor, per-horizon
    multiplier s[i,h] = sqrt(mean of the last `window` squared standardized
    residuals that were ALREADY OBSERVABLE at test anchor i).

    Leakage control (the whole point of this function):
    a forecast made at anchor j for horizon h is only resolved at j+h, so when
    calibrating at anchor i we may use residual j if and only if j + h <= i.
    The pool is therefore advanced by pushing anchor i-h as we step to anchor
    i, never anchor i itself. Anchors are consecutive rows of the fold's test
    block (one trading day apart), so index arithmetic is calendar-correct.

    The pool is seeded with the validation residuals, which are all fully
    resolved before the test block starts (the walk-forward split puts val
    strictly before test, with a purge gap). Early test anchors are therefore
    calibrated on validation data and the window then rolls onto realized test
    residuals as they mature.

    Returns (n_test, 1, H) so it broadcasts over the stock axis; residuals are
    pooled across stocks, which gives S times more data per update and keeps a
    single stock's idiosyncratic jump from dominating the multiplier.
    """
    val_path = npz_path.parent / "val_predictions.npz"
    if not val_path.exists():
        return None
    v = np.load(val_path)
    z = np.load(npz_path)
    sig_v = v["sigma"]
    if sig_v.size == 0 or np.all(sig_v <= 0):
        return None

    # squared standardized residuals, (n, S, H)
    r2_val = ((v["y_ret"] - v["mu"]) / np.clip(sig_v, 1e-18, None)) ** 2
    r2_test = ((z["y_ret"] - z["mu"]) / np.clip(z["sigma"], 1e-18, None)) ** 2
    n_test, _, H = r2_test.shape

    out = np.empty((n_test, 1, H))
    for hi, h in enumerate(horizons):
        # seed with the tail of validation (per-anchor mean over stocks)
        pool = list(r2_val[:, :, hi].mean(axis=1))
        for i in range(n_test):
            j = i - h
            if j >= 0:
                pool.append(float(r2_test[j, :, hi].mean()))
            recent = pool[-window:]
            if len(recent) < min_obs:
                recent = pool
            out[i, 0, hi] = np.sqrt(max(float(np.mean(recent)), 1e-18))
    return out


def evaluate_fold(npz_path: Path, horizons: list[int], weights: np.ndarray,
                  static_corr: np.ndarray | None, conf: float = 0.95,
                  sigma_units: str = "per_day", calibration: str = "global",
                  window: int = 120, es_backtest: bool = True,
                  include_epistemic: bool = False) -> dict:
    """sigma_units: DRAM-T-family models train sigma via Gaussian NLL against
    the CUMULATIVE h-day return, so their stored sigma is in "cumulative"
    units; econometric models store per-day conditional vol ("per_day").
    All metrics below are computed in consistent units for both families:
    - VolRMSE against realized per-day vol -> per-day sigma
    - CRPS / interval coverage against cumulative h-day returns -> cumulative sigma
    - 10-day portfolio VaR -> cumulative sigma at h=10 (no extra sqrt(10))."""
    z = np.load(npz_path)
    mu, y_ret, y_vol = z["mu"], z["y_ret"], z["y_vol"]
    sigma = z["sigma"] if z["sigma"].size else None
    corr = z["corr"] if z["corr"].size else None
    # Student-t models store a per-horizon degrees-of-freedom vector; its
    # presence is what switches the whole risk block to the t distribution.
    df = z["df"] if "df" in z.files and z["df"].size else None

    out: dict = {"point": point_metrics(y_ret, mu)}

    # ---- correlation accuracy -------------------------------------------
    # The model emits a 5x5 correlation matrix and the composite loss carries
    # a Frobenius term weighted by lambda2, which was swept - but none of it
    # was ever scored. Without this the correlation head is an unverified
    # claim, and DCC-GARCH (a baseline included precisely because it forecasts
    # correlations) is never tested on the thing it exists for.
    #
    # Scored BEFORE the sigma guard so models with no volatility head still
    # appear, and only on the strict upper triangle: the diagonal is 1 by
    # construction in both prediction and target, so including it would
    # dilute every model's error toward zero identically.
    if "y_corr" in z.files and z["y_corr"].size:
        y_corr = z["y_corr"]
        S = y_corr.shape[-1]
        iu = np.triu_indices(S, k=1)
        tgt = y_corr[:, iu[0], iu[1]]
        pred = (corr[:, iu[0], iu[1]] if corr is not None
                else np.tile(static_corr, (len(mu), 1, 1))[:, iu[0], iu[1]]
                if static_corr is not None else None)
        if pred is not None:
            out["correlation"] = {
                "CorrRMSE": float(np.sqrt(np.mean((pred - tgt) ** 2))),
                "CorrMAE": float(np.mean(np.abs(pred - tgt))),
                "has_learned_corr": corr is not None,
            }
            # The honest reference: a constant correlation matrix estimated on
            # TRAIN data only. A learned head must beat this to be worth
            # having - the same logic as the martingale null for the mean.
            if static_corr is not None:
                s = np.tile(static_corr, (len(mu), 1, 1))[:, iu[0], iu[1]]
                out["correlation"]["CorrRMSE_static_baseline"] = float(
                    np.sqrt(np.mean((s - tgt) ** 2)))

    if sigma is None:
        return out

    # MC-dropout epistemic component, added as an independent variance term:
    #     sigma_total^2 = sigma_aleatoric^2 + sigma_epistemic^2
    # Applied BEFORE calibration so the post-hoc multiplier sees the same
    # quantity it will scale at prediction time.
    if include_epistemic and "sigma_epistemic" in z.files and z["sigma_epistemic"].size:
        eps_sigma = z["sigma_epistemic"]
        out["epistemic"] = {
            "mean_sigma_epistemic": float(eps_sigma.mean()),
            "mean_sigma_aleatoric": float(sigma.mean()),
            "epistemic_share": float(
                (eps_sigma ** 2).mean() / ((eps_sigma ** 2).mean() + (sigma ** 2).mean())),
        }
        sigma = np.sqrt(sigma ** 2 + eps_sigma ** 2)

    # Post-hoc sigma calibration (fitted on validation / already-resolved test
    # residuals only -- never on the anchor being calibrated; see the two
    # helpers above).
    if calibration == "rolling":
        s_roll = _rolling_sigma_scale(npz_path, horizons, window=window)
        if s_roll is not None:
            out["sigma_calibration"] = {
                "mode": "rolling", "window": window,
                "mean": float(s_roll.mean()), "min": float(s_roll.min()),
                "max": float(s_roll.max()),
            }
            sigma = sigma * s_roll
        else:
            calibration = "global"
    if calibration != "rolling":
        s = _val_sigma_scale(npz_path)
        if s is not None:
            out["sigma_calibration"] = {"mode": "global", "scale": s}
            sigma = sigma * s

    root_h = np.sqrt(np.array(horizons, dtype=float))[None, None, :]
    if sigma_units == "cumulative":
        sigma_cum, sigma_day = sigma, sigma / root_h
    else:
        sigma_day, sigma_cum = sigma, sigma * root_h

    # Under Student-t the head emits a SCALE; realized vol is a standard
    # deviation, so VolRMSE must compare like with like.
    if df is not None:
        std_factor = student_t_std_factor(df)[None, None, :]
        vol_pred_day = sigma_day * std_factor
    else:
        vol_pred_day = sigma_day

    h10 = horizons.index(10)
    nu10 = float(df[h10]) if df is not None else None
    out["risk"] = {
        "VolRMSE": float(np.sqrt(np.mean((vol_pred_day - y_vol) ** 2))),
        "CRPS": (crps_student_t(y_ret, mu, sigma_cum, df[None, None, :])
                 if df is not None else crps_normal(y_ret, mu, sigma_cum)),
        "Coverage95": (interval_coverage_t(y_ret, mu, sigma_cum, df[None, None, :], conf)
                       if df is not None else interval_coverage(y_ret, mu, sigma_cum, conf)),
    }
    if df is not None:
        out["risk"]["df_per_horizon"] = [float(x) for x in df]

    # portfolio VaR backtest at the 10-day horizon (cumulative units).
    # With a common nu across stocks the 5-stock vector is multivariate-t, so
    # w'r is exactly t_nu(w'mu, w'(D R D)w) and this aggregation is not an
    # approximation.
    R = corr if corr is not None else np.tile(static_corr, (len(mu), 1, 1))
    mu_h, sig_h = mu[:, :, h10], sigma_cum[:, :, h10]
    mu_p, sigma_p = portfolio_moments(mu_h, sig_h, R, weights)
    if nu10 is not None:
        var10, es10 = var_es_student_t(mu_p, sigma_p, nu10, conf)
    else:
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
    if es_backtest:
        out["var_backtest"]["acerbi_szekely"] = acerbi_szekely_z2(
            realized_p, var10, es10, conf, nu=nu10, mu_p=mu_p, scale_p=sigma_p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--calibration", choices=["global", "rolling"], default=None,
                        help="sigma calibration mode (default: config risk.sigma_calibration)")
    parser.add_argument("--window", type=int, default=None,
                        help="rolling calibration window in anchors")
    parser.add_argument("--no-es-backtest", action="store_true",
                        help="skip the (simulation-based) Acerbi-Szekely ES test")
    parser.add_argument("--include-epistemic", action="store_true",
                        help="add the MC-dropout epistemic variance to sigma")
    parser.add_argument("--suffix", default="",
                        help="append to output filenames, e.g. '_rolling'")
    args = parser.parse_args()

    base = load_config("config.yaml")
    runs_dir = Path(base["paths"]["runs_dir"])
    results_dir = Path(base["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    risk_cfg = base.get("risk", {})
    calibration = args.calibration or risk_cfg.get("sigma_calibration", "global")
    window = args.window or int(risk_cfg.get("calibration_window", 120))
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
        # DRAM-T family (dramt_*, ablation_*) trains sigma against cumulative
        # h-day returns; econometric baselines store per-day conditional vol.
        sigma_units = "cumulative" if run.startswith(("dramt", "ablation")) else "per_day"
        fold_paths = sorted((runs_dir / run).glob("fold*/test_predictions.npz"))
        per_fold = []
        for p in fold_paths:
            z = np.load(p)
            dates = pd.to_datetime(z["anchor_dates"], unit="ns")
            # static corr from strictly-before-test data (train+gap end = first test date)
            static_corr = _train_static_corr(daily, portfolio, z["anchor_dates"],
                                             dates.min() - pd.Timedelta(days=45))
            per_fold.append(evaluate_fold(
                p, horizons, weights, static_corr, sigma_units=sigma_units,
                calibration=calibration, window=window,
                es_backtest=not args.no_es_backtest,
                include_epistemic=args.include_epistemic))
        (results_dir / f"eval_{run.replace('/', '_')}{args.suffix}.json").write_text(
            json.dumps(per_fold, indent=2, default=float))

        row: dict = {"run": run, "n_folds": len(per_fold)}
        for key in ["MAE", "MSE", "RMSE", "MAPE", "DirAcc"]:
            vals = [f["point"][key] for f in per_fold]
            row[key], row[f"{key}_std"] = float(np.mean(vals)), float(np.std(vals))
        if all("correlation" in f for f in per_fold):
            for key in ["CorrRMSE", "CorrMAE", "CorrRMSE_static_baseline"]:
                vals = [f["correlation"].get(key) for f in per_fold]
                if all(v is not None for v in vals):
                    row[key] = float(np.mean(vals))
            row["learned_corr"] = bool(per_fold[0]["correlation"]["has_learned_corr"])
        if all("epistemic" in f for f in per_fold):
            for key in ["mean_sigma_epistemic", "mean_sigma_aleatoric",
                        "epistemic_share"]:
                row[key] = float(np.mean([f["epistemic"][key] for f in per_fold]))
        if all("risk" in f for f in per_fold) and per_fold:
            for key in ["VolRMSE", "CRPS", "Coverage95"]:
                vals = [f["risk"][key] for f in per_fold]
                row[key], row[f"{key}_std"] = float(np.mean(vals)), float(np.std(vals))
            row["VaRBreach"] = float(np.mean([f["var_backtest"]["breach_rate"] for f in per_fold]))
            row["VaRBreach_std"] = float(np.std([f["var_backtest"]["breach_rate"] for f in per_fold]))
            row["Kupiec_p_min"] = float(np.min([f["var_backtest"]["kupiec"]["pvalue"] for f in per_fold]))
            row["Christof_p_min"] = float(np.min(
                [f["var_backtest"]["christoffersen"]["pvalue"] for f in per_fold]))
            if all("acerbi_szekely" in f["var_backtest"] for f in per_fold):
                row["ES_Z2"] = float(np.mean(
                    [f["var_backtest"]["acerbi_szekely"]["Z2"] for f in per_fold]))
                row["ES_Z2_p_min"] = float(np.min(
                    [f["var_backtest"]["acerbi_szekely"]["pvalue"] for f in per_fold]))
            if "df_per_horizon" in per_fold[0].get("risk", {}):
                row["df_h10"] = float(np.mean(
                    [f["risk"]["df_per_horizon"][horizons.index(10)] for f in per_fold]))
        summary_rows.append(row)
        logger.info("%s: %s", run, {k: round(v, 4) for k, v in row.items()
                                    if isinstance(v, float) and not k.endswith("_std")})

    out_csv = results_dir / f"eval_summary{args.suffix}.csv"
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    logger.info("wrote %s (calibration=%s, window=%d)", out_csv, calibration, window)


if __name__ == "__main__":
    main()
