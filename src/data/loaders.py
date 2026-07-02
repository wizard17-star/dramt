"""Equities (yfinance) and macro (FRED) raw data loaders with on-disk caching."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def download_equities(
    tickers: list[str],
    start: str,
    end: str | None,
    raw_dir: Path,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV for each ticker via yfinance, caching to CSV."""
    out_dir = raw_dir / "equities"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        cache_path = out_dir / f"{ticker.replace('^', 'IDX_')}.csv"
        if cache_path.exists() and not force:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            logger.info("equities: loaded cache for %s (%d rows)", ticker, len(df))
        else:
            df = yf.download(
                ticker, start=start, end=end, progress=False, auto_adjust=True
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index.name = "date"
            df.to_csv(cache_path)
            logger.info("equities: downloaded %s (%d rows) -> %s", ticker, len(df), cache_path)
        frames[ticker] = df
    return frames


def download_macro(
    fred_series: dict[str, str],
    start: str,
    end: str | None,
    raw_dir: Path,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download macro series from FRED (no API key required via pandas-datareader)."""
    import datetime as dt

    import pandas_datareader.data as web

    out_dir = raw_dir / "macro"
    out_dir.mkdir(parents=True, exist_ok=True)
    end_dt = dt.datetime.today() if end is None else pd.to_datetime(end)
    start_dt = pd.to_datetime(start)

    frames: dict[str, pd.DataFrame] = {}
    for code, name in fred_series.items():
        cache_path = out_dir / f"{name}.csv"
        if cache_path.exists() and not force:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            logger.info("macro: loaded cache for %s (%d rows)", name, len(df))
        else:
            df = web.DataReader(code, "fred", start_dt, end_dt)
            df.index.name = "date"
            df.to_csv(cache_path)
            logger.info("macro: downloaded %s (%d rows) -> %s", name, len(df), cache_path)
        frames[name] = df
    return frames
