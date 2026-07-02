"""Mixed-frequency macro alignment: forward-fill to daily grid + Beta-weighted
MIDAS long-term component tau_t (GARCH-MIDAS style), with zero look-ahead.

Alignment convention (documented per spec):
- A macro observation stamped at month m (FRED stamps at period start) is treated
  as KNOWN only after the end of that period plus a publication lag of
  `publication_lag_days` calendar days. E.g. CPI for January (stamped 2023-01-01)
  covers January and is published mid-February -> it becomes available to the
  model no earlier than end-of-January + lag. We implement this by shifting each
  series' timestamps to (period_end + lag) before forward-filling onto the daily
  trading grid, guaranteeing the model never sees a value before it could have
  been published.
- Daily series (fed_funds, yield_10y) use a 1-business-day availability lag.
- tau_t = theta0 + theta1 * sum_k w_k(Beta(a,b)) * X_{t-k} over the last K
  *available* macro-frequency observations. Here we compute the Beta-weighted
  rolling aggregate as a FEATURE (theta0/theta1 learned downstream by the model;
  the raw weighted sum is what enters the feature matrix).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTHLY_SERIES = {"cpi", "unemployment", "consumer_sentiment", "industrial_production"}
DAILY_SERIES = {"fed_funds", "yield_10y"}


def beta_weights(n_lags: int, a: float = 1.0, b: float = 5.0) -> np.ndarray:
    """Normalized Beta(a,b) lag weights over k=1..K (most recent lag first)."""
    k = np.arange(1, n_lags + 1, dtype=float)
    x = k / (n_lags + 1.0)
    w = x ** (a - 1.0) * (1.0 - x) ** (b - 1.0)
    return w / w.sum()


def _availability_shift(df: pd.DataFrame, name: str, publication_lag_days: int = 15) -> pd.DataFrame:
    """Shift a raw FRED series' index to the date it becomes *available*."""
    out = df.copy()
    if name in MONTHLY_SERIES:
        # stamp at period start -> available at period end + publication lag
        period_end = out.index + pd.offsets.MonthEnd(0)
        out.index = period_end + pd.Timedelta(days=publication_lag_days)
    else:
        # daily series available next business day
        out.index = out.index + pd.offsets.BDay(1)
    return out


def align_macro_daily(
    macro: dict[str, pd.DataFrame],
    trading_days: pd.DatetimeIndex,
    publication_lag_days: int = 15,
) -> pd.DataFrame:
    """Forward-fill each macro series onto the trading-day grid, respecting
    availability (publication lag). Returns one column per series."""
    cols = {}
    for name, df in macro.items():
        shifted = _availability_shift(df, name, publication_lag_days)
        s = shifted.iloc[:, 0]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        aligned = s.reindex(s.index.union(trading_days)).ffill().reindex(trading_days)
        cols[name] = aligned
    return pd.DataFrame(cols, index=trading_days)


def midas_tau_features(
    macro: dict[str, pd.DataFrame],
    trading_days: pd.DatetimeIndex,
    n_lags: int = 12,
    beta_a: float = 1.0,
    beta_b: float = 5.0,
    publication_lag_days: int = 15,
) -> pd.DataFrame:
    """Beta-weighted MIDAS long-term component per monthly macro series.

    For each trading day t, tau feature = sum_k w_k * X_{available lag k},
    computed over the last `n_lags` *available* native-frequency observations
    (strictly before-or-at t after publication-lag shifting). Uses monthly
    growth rates (log-diff for level series) so the component is stationary.
    """
    w = beta_weights(n_lags, beta_a, beta_b)
    cols = {}
    for name, df in macro.items():
        if name not in MONTHLY_SERIES:
            continue
        shifted = _availability_shift(df, name, publication_lag_days)
        s = shifted.iloc[:, 0].dropna()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        # stationarize: log-diff for positive level series, plain diff otherwise
        if (s > 0).all() and name in {"cpi", "industrial_production", "consumer_sentiment"}:
            x = np.log(s).diff().dropna()
        else:
            x = s.diff().dropna()
        # rolling Beta-weighted sum over the last n_lags observations
        vals = x.to_numpy()
        tau = np.full(len(x), np.nan)
        for i in range(n_lags - 1, len(x)):
            window = vals[i - n_lags + 1 : i + 1][::-1]  # most recent first
            tau[i] = float(np.dot(w, window))
        tau_s = pd.Series(tau, index=x.index)
        aligned = tau_s.reindex(tau_s.index.union(trading_days)).ffill().reindex(trading_days)
        cols[f"tau_{name}"] = aligned
    return pd.DataFrame(cols, index=trading_days)


def build_macro_features(
    macro: dict[str, pd.DataFrame],
    trading_days: pd.DatetimeIndex,
    n_lags: int = 12,
    beta_a: float = 1.0,
    beta_b: float = 5.0,
    publication_lag_days: int = 15,
) -> pd.DataFrame:
    """Daily ffilled levels + MIDAS tau components, one frame."""
    daily = align_macro_daily(macro, trading_days, publication_lag_days)
    tau = midas_tau_features(macro, trading_days, n_lags, beta_a, beta_b, publication_lag_days)
    return daily.join(tau)
