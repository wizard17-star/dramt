"""GARCH(1,1) conditional-volatility features for the hybrid vol head.

Why this exists
---------------
The CPU-era results identified the failure mode precisely: DRAM-T's learned
volatility "does not track regime shifts across a fold the way a recursive
GARCH filter does" (fold-0 VaR breach 0%, fold-2 20.7%). A GARCH filter
updates its conditional variance recursively from every new observation; the
transformer's vol head only sees a T-day window and has no such recursion.

The hybrid head therefore keeps the recursion where it works -- a GARCH(1,1)
filter supplies the LEVEL -- and lets the network learn a multiplicative
correction on top:

    sigma_hybrid = softplus(vol_head_output) * sigma_garch

so the network only has to learn a (mostly O(1)) adjustment reflecting
multimodal information, not the volatility level itself.

Leakage protocol (identical to the GARCH baseline in
src/models/baselines/econometric.py)
------------------------------------
- GARCH parameters are estimated ONCE per fold on TRAIN returns only
  (`last_obs=train_end`), then held fixed.
- With fixed parameters the filter is run over the whole series; the
  conditional variance at day t depends only on returns up to t-1.
- Forecasts at an anchor therefore use no information from after that anchor.

Results are cached per (T, fold, suffix) because fitting 5 univariate GARCH
models and forecasting over every anchor is slow relative to a training epoch.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCALE = 100.0  # percent units, matching src/models/baselines/econometric.py


def garch_cumulative_vol(
    returns_df: pd.DataFrame,      # (days, S) raw daily log returns
    train_end: pd.Timestamp,       # last date usable for parameter estimation
    anchors: pd.DatetimeIndex,     # anchor days to produce forecasts for
    horizons: list[int],
) -> np.ndarray:                   # (n_anchors, S, H) CUMULATIVE h-day vol, raw units
    """Per-stock GARCH(1,1) cumulative volatility forecast at each anchor.

    Cumulative (not per-day) because DRAM-T's sigma is trained against
    cumulative h-day returns: sigma_cum(h) = sqrt(sum_{j=1..h} var_{t+j}).
    """
    from arch import arch_model

    R = returns_df.dropna() * SCALE
    h_max = max(horizons)
    out = np.empty((len(anchors), R.shape[1], len(horizons)))

    for si, col in enumerate(R.columns):
        am = arch_model(R[col], mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
        res = am.fit(last_obs=train_end, disp="off")
        fixed = am.fix(res.params.to_numpy())
        fc = fixed.forecast(horizon=h_max, start=anchors[0], reindex=False)
        var_rows = fc.variance.reindex(anchors).to_numpy()      # (n, h_max)
        # forward-fill any anchor the forecast frame does not cover
        if np.isnan(var_rows).any():
            var_rows = pd.DataFrame(var_rows).ffill().bfill().to_numpy()
        for hi, h in enumerate(horizons):
            out[:, si, hi] = np.sqrt(var_rows[:, :h].sum(axis=1))
    return out / SCALE


def load_or_build(
    processed_dir: Path,
    daily: pd.DataFrame,
    portfolio: list[str],
    anchor_dates_ns: np.ndarray,   # ALL anchors of the dataset (ns since epoch)
    train_idx: np.ndarray,
    horizons: list[int],
    T: int,
    fold_k: int,
    suffix: str = "",
) -> np.ndarray:
    """(n_anchors, S, H) cumulative GARCH vol for every anchor of the dataset,
    with parameters fit on this fold's training anchors only. Cached on disk."""
    cache = processed_dir / f"garch_vol_T{T}{suffix}_fold{fold_k}.npz"
    if cache.exists():
        z = np.load(cache)
        if z["vol"].shape == (len(anchor_dates_ns), len(portfolio), len(horizons)):
            return z["vol"]
        logger.warning("stale GARCH cache %s (shape mismatch) - rebuilding", cache)

    dates = pd.to_datetime(anchor_dates_ns, unit="ns")
    train_end = pd.Timestamp(dates[train_idx][-1])
    returns_df = daily[[f"{t}_log_return" for t in portfolio]].copy()
    returns_df.columns = portfolio

    logger.info("fitting GARCH(1,1) vol features: T=%d fold=%d train_end=%s",
                T, fold_k, train_end.date())
    vol = garch_cumulative_vol(returns_df, train_end, pd.DatetimeIndex(dates), horizons)
    np.savez_compressed(cache, vol=vol)
    return vol
