"""Two analyses that reinterpret the saved predictions, no retraining needed.

1. NEWS-STRATIFIED ACCURACY
   The sentiment modality is mostly absent. Measured on the aligned grid, the
   share of trading days carrying any news is NVDA 71%, AAPL 55%, MSFT 33%,
   AMZN 14%, META 12%, GOOGL 9% -- so for most stocks the sentiment features
   are the neutral zero placeholder plus has_news=0 on the large majority of
   days. A sentiment ablation averaged over all days therefore measures
   sentiment mostly where sentiment does not exist, which dilutes any real
   effect toward zero.

   This splits the SAME predictions into anchors whose window contained news
   for that stock and anchors that did not, and reports accuracy separately.
   If the sentiment modality works at all, the gap between the full model and
   the no-sentiment ablation must be larger on the news stratum.

2. ECONOMIC VALUE OF THE VOLATILITY FORECAST
   RQ2 is currently answered only in the language of backtests (Kupiec,
   Christoffersen, ES). A reader is entitled to ask what a better-calibrated
   sigma is worth. The standard answer is volatility targeting: scale exposure
   inversely to the forecast volatility,

       L_t = clip(target_vol / sigma_p(t), 0, max_leverage)

   and hold L_t of the equal-weight portfolio over the next day. A model whose
   sigma tracks realized risk delivers a realized volatility close to target
   and a better risk-adjusted return; a model whose sigma is noisy or badly
   scaled does not. Leverage is formed from information available at the
   anchor only, and applied to the NEXT day's return.

Usage:
    python -m src.analysis --calibration rolling
Writes results/analysis_news_strata.csv and results/analysis_economic.csv.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.var_es import portfolio_moments

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def _calibrated_sigma(npz_path: Path, horizons: list[int], calibration: str,
                      window: int):
    """Same post-hoc calibration evaluate.py applies, so these analyses and the
    headline risk tables describe the same quantity."""
    from src.evaluate import _rolling_sigma_scale, _val_sigma_scale

    z = np.load(npz_path)
    sigma = z["sigma"]
    if not sigma.size:
        return None, z
    if calibration == "rolling":
        s = _rolling_sigma_scale(npz_path, horizons, window=window)
        if s is not None:
            return sigma * s, z
    s = _val_sigma_scale(npz_path)
    return (sigma * s if s is not None else sigma), z


# --------------------------------------------------------------------------- #
# 1. news-stratified accuracy
# --------------------------------------------------------------------------- #

def news_mask_for_anchors(daily: pd.DataFrame, portfolio: list[str],
                          anchor_dates_ns: np.ndarray, T: int) -> np.ndarray:
    """(n_anchors, S) bool: did this stock have ANY news inside the window?

    The window for an anchor at grid position p is rows [p-T+1 .. p], i.e.
    exactly the input the model saw - no future information.
    """
    grid = daily.index
    pos = {d: i for i, d in enumerate(grid)}
    dates = pd.to_datetime(anchor_dates_ns, unit="ns")
    has = np.zeros((len(dates), len(portfolio)), dtype=bool)
    cols = {t: daily[f"{t}_has_news"].to_numpy() for t in portfolio}
    for i, d in enumerate(dates):
        p = pos[pd.Timestamp(d)]
        lo = max(p - T + 1, 0)
        for si, t in enumerate(portfolio):
            has[i, si] = bool((cols[t][lo:p + 1] > 0).any())
    return has


def stratified_accuracy(runs_dir: Path, run: str, daily: pd.DataFrame,
                        portfolio: list[str], T: int) -> list[dict]:
    rows = []
    for p in sorted((runs_dir / run).glob("fold*/test_predictions.npz")):
        z = np.load(p)
        mask = news_mask_for_anchors(daily, portfolio, z["anchor_dates"], T)  # (n,S)
        err = np.abs(z["y_ret"] - z["mu"])            # (n,S,H)
        direction = (np.sign(z["mu"]) == np.sign(z["y_ret"]))
        m3 = np.repeat(mask[:, :, None], err.shape[2], axis=2)
        for label, sel in (("news", m3), ("no_news", ~m3)):
            if not sel.any():
                continue
            rows.append({
                "run": run, "fold": int(p.parent.name.replace("fold", "")),
                "stratum": label,
                "n_obs": int(sel.sum()),
                "MAE": float(err[sel].mean()),
                "DirAcc": float(direction[sel].mean()),
            })
    return rows


# --------------------------------------------------------------------------- #
# 2. economic value: volatility targeting
# --------------------------------------------------------------------------- #

def vol_targeted_portfolio(runs_dir: Path, run: str, weights: np.ndarray,
                           horizons: list[int], calibration: str, window: int,
                           target_vol_annual: float = 0.10,
                           max_leverage: float = 3.0,
                           sigma_units: str = "cumulative") -> dict | None:
    """Realized performance of an equal-weight portfolio scaled to a constant
    volatility target using this model's 1-day-ahead sigma forecast."""
    target_daily = target_vol_annual / np.sqrt(TRADING_DAYS)
    h1 = horizons.index(1)
    rets, levs = [], []

    for p in sorted((runs_dir / run).glob("fold*/test_predictions.npz")):
        sigma, z = _calibrated_sigma(p, horizons, calibration, window)
        if sigma is None:
            return None
        # at h=1 the cumulative and per-day conventions coincide
        sig_h = sigma[:, :, h1]
        mu_h = z["mu"][:, :, h1]
        S = sig_h.shape[1]
        corr = z["corr"] if z["corr"].size else np.tile(np.eye(S), (len(sig_h), 1, 1))
        _, sigma_p = portfolio_moments(mu_h, sig_h, corr, weights)
        lev = np.clip(target_daily / np.clip(sigma_p, 1e-12, None), 0.0, max_leverage)
        realized = z["y_ret"][:, :, h1] @ weights        # next-day portfolio return
        rets.append(lev * realized)
        levs.append(lev)

    if not rets:
        return None
    r = np.concatenate(rets)
    lev = np.concatenate(levs)
    ann_ret = float(r.mean() * TRADING_DAYS)
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    equity = np.cumsum(r)                                # log-return space
    drawdown = float((equity - np.maximum.accumulate(equity)).min())
    return {
        "run": run,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "target_vol": target_vol_annual,
        # how close the realized volatility lands to the target IS the
        # economic read-out of volatility-forecast quality
        "vol_target_error": abs(ann_vol - target_vol_annual),
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        "max_drawdown": drawdown,
        "mean_leverage": float(lev.mean()),
        "leverage_turnover": float(np.abs(np.diff(lev)).mean()),
        "n_days": int(len(r)),
    }


def buy_and_hold(runs_dir: Path, reference_run: str, weights: np.ndarray,
                 horizons: list[int]) -> dict:
    """Unlevered equal-weight benchmark over the identical test anchors."""
    h1 = horizons.index(1)
    rets = [np.load(p)["y_ret"][:, :, h1] @ weights
            for p in sorted((runs_dir / reference_run).glob("fold*/test_predictions.npz"))]
    r = np.concatenate(rets)
    ann_ret = float(r.mean() * TRADING_DAYS)
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    equity = np.cumsum(r)
    return {
        "run": "buy_and_hold (unlevered)", "ann_return": ann_ret, "ann_vol": ann_vol,
        "target_vol": np.nan, "vol_target_error": np.nan,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        "max_drawdown": float((equity - np.maximum.accumulate(equity)).min()),
        "mean_leverage": 1.0, "leverage_turnover": 0.0, "n_days": int(len(r)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", choices=["global", "rolling"], default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--target-vol", type=float, default=0.10)
    ap.add_argument("--max-leverage", type=float, default=3.0)
    ap.add_argument("--runs", nargs="*", default=None)
    args = ap.parse_args()

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
        if d.is_dir() and d.name != "sweep" and list(d.glob("fold*/test_predictions.npz")))
    if not run_names:
        raise SystemExit("no completed runs found under runs/")

    # T is needed to reconstruct each anchor's input window
    T = int(yaml_best_T(runs_dir, base))

    # ---- 1. news strata --------------------------------------------------
    strata = []
    for run in run_names:
        strata.extend(stratified_accuracy(runs_dir, run, daily, portfolio, T))
    if strata:
        df = pd.DataFrame(strata)
        agg = (df.groupby(["run", "stratum"])
                 .agg(MAE=("MAE", "mean"), DirAcc=("DirAcc", "mean"),
                      n_obs=("n_obs", "sum"))
                 .reset_index())
        agg.to_csv(results_dir / "analysis_news_strata.csv", index=False)
        logger.info("wrote %s (%d rows)", results_dir / "analysis_news_strata.csv", len(agg))

    # ---- 2. economic value ----------------------------------------------
    econ = []
    for run in run_names:
        res = vol_targeted_portfolio(
            runs_dir, run, weights, horizons, calibration, window,
            target_vol_annual=args.target_vol, max_leverage=args.max_leverage)
        if res:
            econ.append(res)
    if econ:
        econ.append(buy_and_hold(runs_dir, run_names[0], weights, horizons))
        pd.DataFrame(econ).sort_values("vol_target_error").to_csv(
            results_dir / "analysis_economic.csv", index=False)
        logger.info("wrote %s (%d models)", results_dir / "analysis_economic.csv", len(econ))


def yaml_best_T(runs_dir: Path, base: dict) -> int:
    import yaml
    p = runs_dir / "sweep" / "best.yaml"
    if p.exists():
        return int(yaml.safe_load(p.read_text(encoding="utf-8")).get(
            "T", base["windowing"]["default_T"]))
    return int(base["windowing"]["default_T"])


if __name__ == "__main__":
    main()
