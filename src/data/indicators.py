"""Technical indicators computed from daily OHLCV.

All indicators use only past/current bars (rolling / EWM with adjust=False),
so no look-ahead is introduced at feature-construction time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": macd_line - signal_line,
    })


def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    # %B: position of close within the bands (0=lower, 1=upper)
    pct_b = (close - lower) / (upper - lower)
    bandwidth = (upper - lower) / mid
    return pd.DataFrame({"boll_pct_b": pct_b, "boll_bandwidth": bandwidth})


def rolling_volatility(log_ret: pd.Series, window: int = 20) -> pd.Series:
    return log_ret.rolling(window).std()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    # normalize by close so ATR is scale-free across stocks
    return tr.ewm(alpha=1.0 / period, adjust=False).mean() / close


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    raw = (direction * volume).cumsum()
    # z-scale by rolling window so magnitude is comparable across stocks;
    # rolling stats use only past values -> no look-ahead
    mean = raw.rolling(60).mean()
    std = raw.rolling(60).std()
    return (raw - mean) / std


def momentum(close: pd.Series, window: int = 10) -> pd.Series:
    return np.log(close / close.shift(window))


def compute_indicators(ohlcv: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute the full indicator set for one ticker's OHLCV frame.

    Returns a DataFrame indexed like the input with all feature columns;
    leading NaN rows (warm-up) are kept and dropped later at alignment time.
    """
    close, high, low, volume = ohlcv["Close"], ohlcv["High"], ohlcv["Low"], ohlcv["Volume"]
    lr = log_returns(close)
    feats = pd.DataFrame(index=ohlcv.index)
    feats["log_return"] = lr
    feats["rsi"] = rsi(close, cfg["rsi_period"]) / 100.0
    feats = feats.join(macd(close, *cfg["macd"]))
    feats = feats.join(bollinger(close, cfg["bollinger"][0], cfg["bollinger"][1]))
    feats["roll_vol"] = rolling_volatility(lr, cfg["rolling_vol_window"])
    feats["atr"] = atr(high, low, close, cfg["atr_period"])
    feats["obv_z"] = obv(close, volume)
    feats["log_volume"] = np.log1p(volume)
    feats["momentum"] = momentum(close, cfg["momentum_window"])
    return feats
