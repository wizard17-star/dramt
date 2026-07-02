"""Sliding-window sample construction with multi-horizon point + risk targets.

Sample at anchor day t (index into the aligned trading-day grid):
- inputs: rows [t-T+1 .. t] of each modality's feature matrix (past only)
- point targets:   y_ret[t, s, h]  = cumulative log return of stock s over
                    days t+1 .. t+h (h in `horizons`)
- volatility tgts: y_vol[t, s, h]  = realized vol of daily log returns over
                    t+1 .. t+h; for h=1 the std of a single return is undefined,
                    so |r_{t+1}| is used as the realized-vol proxy (documented).
- correlation tgt: y_corr[t]       = realized correlation matrix of the 5
                    portfolio stocks' daily returns over t+1 .. t+h_max
                    (one matrix per sample at the maximum horizon; shorter-
                    horizon correlation of <3 observations is too noisy).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WindowedDataset:
    """Arrays are aligned on the first axis (samples)."""
    anchor_dates: pd.DatetimeIndex          # (n,)
    X_num: np.ndarray                        # (n, T, F_num)
    X_macro: np.ndarray                      # (n, T, F_macro)
    X_sent: np.ndarray                       # (n, T, F_sent)
    y_ret: np.ndarray                        # (n, S, H)
    y_vol: np.ndarray                        # (n, S, H)
    y_corr: np.ndarray                       # (n, S, S)
    horizons: list[int]
    stocks: list[str]


def build_windows(
    num_feats: pd.DataFrame,
    macro_feats: pd.DataFrame,
    sent_feats: pd.DataFrame,
    stock_returns: pd.DataFrame,   # daily log returns, columns = portfolio stocks
    T: int,
    horizons: list[int],
) -> WindowedDataset:
    """Construct windowed samples. All inputs must share the same DatetimeIndex."""
    idx = num_feats.index
    assert macro_feats.index.equals(idx) and sent_feats.index.equals(idx) and stock_returns.index.equals(idx), \
        "all feature frames must share the trading-day index"
    h_max = max(horizons)
    n_days = len(idx)
    stocks = list(stock_returns.columns)
    S, H = len(stocks), len(horizons)

    Xn, Xm, Xs = num_feats.to_numpy(float), macro_feats.to_numpy(float), sent_feats.to_numpy(float)
    R = stock_returns.to_numpy(float)  # (n_days, S)

    anchors = list(range(T - 1, n_days - h_max))
    n = len(anchors)
    out_num = np.empty((n, T, Xn.shape[1]), dtype=np.float32)
    out_mac = np.empty((n, T, Xm.shape[1]), dtype=np.float32)
    out_sen = np.empty((n, T, Xs.shape[1]), dtype=np.float32)
    y_ret = np.empty((n, S, H), dtype=np.float32)
    y_vol = np.empty((n, S, H), dtype=np.float32)
    y_corr = np.empty((n, S, S), dtype=np.float32)

    for i, t in enumerate(anchors):
        out_num[i] = Xn[t - T + 1 : t + 1]
        out_mac[i] = Xm[t - T + 1 : t + 1]
        out_sen[i] = Xs[t - T + 1 : t + 1]
        fut = R[t + 1 : t + 1 + h_max]  # (h_max, S) strictly future daily returns
        for j, h in enumerate(horizons):
            seg = fut[:h]
            y_ret[i, :, j] = seg.sum(axis=0)
            if h == 1:
                y_vol[i, :, j] = np.abs(seg[0])
            else:
                y_vol[i, :, j] = seg.std(axis=0, ddof=1)
        c = np.corrcoef(fut.T)
        # guard: constant column -> NaN row/col; replace with identity structure
        if np.isnan(c).any():
            c = np.where(np.isnan(c), np.eye(S), c)
        y_corr[i] = c

    return WindowedDataset(
        anchor_dates=idx[anchors],
        X_num=out_num, X_macro=out_mac, X_sent=out_sen,
        y_ret=y_ret, y_vol=y_vol, y_corr=y_corr,
        horizons=list(horizons), stocks=stocks,
    )
