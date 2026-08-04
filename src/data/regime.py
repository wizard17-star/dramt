"""Market-regime signals for the DRAM-T gating network (RQ3).

Why this is separate from the modality features
-----------------------------------------------
RQ3 asks whether INPUT-DEPENDENT dynamic weighting beats static fusion. In the
CPU-era study it did not (static fusion p=0.26, nominally better). One
plausible cause is that the gate only received a 4-dimensional signal (recent
vol, return sign, macro level, news presence) - too coarse to identify the
regimes in which one modality should dominate.

The extra signals here are therefore fed to the GATE ONLY and are deliberately
NOT added to X_num. If they were added to the modality inputs as well, a
change in results could not be attributed to better gating rather than to
simply giving the model more features, and RQ3 would be unanswerable. Keeping
them gate-only isolates the effect being tested.

Signals (all computed from information available AT or BEFORE the anchor day):
    0 recent realized volatility   (as before, portfolio mean of roll_vol)
    1 recent return sign           (as before)
    2 macro level                  (as before)
    3 news presence                (as before)
    4 VIX level                    - forward-looking implied vol
    5 VIX slope                    - 5-day change, i.e. is fear rising
    6 realized average correlation - 60-day mean pairwise correlation of the
                                     portfolio; diversification breaks down in
                                     stress, which is exactly when the risk
                                     modalities should matter more
    7 drawdown                     - equal-weight portfolio drawdown from its
                                     trailing 252-day running maximum

Leakage: every signal at anchor t uses only rows <= t. Trailing windows are
backward-looking by construction; no centred windows and no ffill from the
future. Standardization is fitted on the fold's TRAIN anchors only, in
src/train.py.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASIC_SIGNALS = ["roll_vol", "ret_sign", "macro_level", "news_presence"]
EXTENDED_SIGNALS = BASIC_SIGNALS + ["vix_level", "vix_slope", "avg_corr", "drawdown"]


def load_vix(raw_dir: Path, grid: pd.DatetimeIndex) -> pd.Series:
    """VIX close reindexed onto the trading grid (ffill only - never bfill,
    which would pull a future value backwards)."""
    path = raw_dir / "equities" / "IDX_VIX.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run download_equities(['^VIX'], ...) first"
        )
    vix = pd.read_csv(path, index_col=0, parse_dates=True)["Close"]
    return vix.reindex(grid).ffill()


def build_extended_regime(
    daily: pd.DataFrame,
    raw_dir: Path,
    portfolio: list[str],
    anchor_dates: pd.DatetimeIndex,
    corr_window: int = 60,
    drawdown_window: int = 252,
    vix_slope_days: int = 5,
) -> np.ndarray:
    """(n_anchors, 4) array of the EXTRA gate signals, in raw units.

    Returns only the four new columns; the original four are still computed
    from the standardized window tensors in src/train.py:regime_features, so
    the basic behaviour is unchanged when regime_signals='basic'.
    """
    grid = daily.index
    ret_cols = [f"{t}_log_return" for t in portfolio]
    rets = daily[ret_cols]

    vix = load_vix(raw_dir, grid)
    vix_level = vix
    vix_slope = vix.diff(vix_slope_days)

    # rolling mean pairwise correlation of the portfolio (trailing window)
    S = len(portfolio)
    iu = np.triu_indices(S, k=1)
    R = rets.to_numpy()
    avg_corr = np.full(len(grid), np.nan)
    for i in range(corr_window, len(grid)):
        w = R[i - corr_window + 1 : i + 1]
        # A zero-variance column (e.g. a halted stock) makes np.corrcoef divide
        # by zero and emit NaN; treat such a pair as uncorrelated rather than
        # letting a NaN propagate into the gate.
        if (w.std(axis=0) < 1e-12).any():
            with np.errstate(invalid="ignore", divide="ignore"):
                c = np.nan_to_num(np.corrcoef(w.T), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            c = np.corrcoef(w.T)
        avg_corr[i] = float(np.mean(c[iu]))
    avg_corr = pd.Series(avg_corr, index=grid)

    # equal-weight portfolio drawdown from a trailing running maximum
    port_ret = rets.mean(axis=1)
    log_equity = port_ret.cumsum()
    running_max = log_equity.rolling(drawdown_window, min_periods=1).max()
    drawdown = log_equity - running_max              # <= 0

    feat = pd.DataFrame({
        "vix_level": vix_level,
        "vix_slope": vix_slope,
        "avg_corr": avg_corr,
        "drawdown": drawdown,
    }, index=grid).ffill().bfill()

    out = feat.reindex(pd.DatetimeIndex(anchor_dates)).to_numpy(dtype=np.float32)
    if np.isnan(out).any():
        raise ValueError("extended regime features contain NaN after alignment")
    return out
