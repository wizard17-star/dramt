"""Assemble the processed multimodal dataset from raw caches.

Usage:
    python -m src.data.build_features                 # with sentiment (needs GDELT cache)
    python -m src.data.build_features --no-sentiment  # zero sentiment placeholder

Outputs to data/processed/:
    features_T{T}.npz   - windowed arrays for each T in the sweep
    aligned_daily.parquet - the aligned daily feature frame (debug/inspection)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.indicators import compute_indicators
from src.data.loaders import download_equities, download_macro
from src.data.macro_midas import build_macro_features
from src.data.windows import build_windows
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build(no_sentiment: bool = False) -> None:
    cfg = load_config("config.yaml")
    raw_dir = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    tickers = cfg["universe"]["tickers"]
    benchmark = cfg["universe"]["benchmark"]
    portfolio = cfg["portfolio"]["members"]

    # --- load raw (from cache; will download only if missing) ---
    equities = download_equities(tickers + [benchmark], cfg["data"]["start_date"], cfg["data"]["end_date"], raw_dir)
    macro = download_macro(cfg["macro"]["fred_series"], cfg["data"]["start_date"], cfg["data"]["end_date"], raw_dir)

    # --- common trading-day grid: intersection of all equity indices ---
    grid = None
    for t, df in equities.items():
        grid = df.index if grid is None else grid.intersection(df.index)
    grid = grid.sort_values()
    logger.info("common trading days: %d (%s -> %s)", len(grid), grid.min().date(), grid.max().date())

    # --- numerical modality: indicators for every ticker + benchmark ---
    ind_cfg = cfg["technical_indicators"]
    num_frames = []
    returns_cols = {}
    for t, df in equities.items():
        df = df.reindex(grid).ffill(limit=3)
        feats = compute_indicators(df, ind_cfg)
        safe = t.replace("^", "IDX_")
        feats.columns = [f"{safe}_{c}" for c in feats.columns]
        num_frames.append(feats)
        returns_cols[t] = np.log(df["Close"] / df["Close"].shift(1))
    num_feats = pd.concat(num_frames, axis=1)
    stock_returns = pd.DataFrame({t: returns_cols[t] for t in portfolio}, index=grid)

    # --- macro modality ---
    mc = cfg["macro"]["midas"]
    macro_feats = build_macro_features(
        macro, grid, n_lags=mc["n_lags"], beta_a=mc["beta_a"], beta_b=mc["beta_b"]
    )

    # --- sentiment modality ---
    if no_sentiment:
        sent_cols = [f"{t}_{c}" for t in tickers for c in
                     ["sent_mean", "sent_polarity", "news_volume", "sent_decay", "has_news"]]
        sent_feats = pd.DataFrame(0.0, index=grid, columns=sent_cols)
        logger.warning("sentiment: using ZERO placeholder (--no-sentiment)")
    else:
        from src.data.sentiment_gdelt_finbert import build_sentiment_features
        sent_feats = build_sentiment_features(
            raw_dir, processed_dir, tickers, grid,
            model_name=cfg["sentiment"]["finbert_model"],
            decay_days=cfg["sentiment"]["decay_days"],
        )

    # --- drop warm-up edge rows (indicator/midas NaNs), ffill small interior gaps ---
    all_daily = num_feats.join(macro_feats).join(sent_feats)
    all_daily = all_daily.ffill(limit=3)
    valid_start = all_daily.dropna().index.min()
    all_daily = all_daily.loc[valid_start:]
    stock_returns = stock_returns.loc[valid_start:]
    n_bad = int(all_daily.isna().sum().sum())
    assert n_bad == 0, f"{n_bad} NaNs remain after edge drop"
    logger.info("aligned daily frame: %d days x %d features from %s", *all_daily.shape, valid_start.date())

    all_daily.to_parquet(processed_dir / "aligned_daily.parquet")

    num_cols = list(num_feats.columns)
    macro_cols = list(macro_feats.columns)
    sent_cols = list(sent_feats.columns)

    # --- windowed datasets for each T in sweep ---
    for T in cfg["windowing"]["T_sweep"]:
        ds = build_windows(
            all_daily[num_cols], all_daily[macro_cols], all_daily[sent_cols],
            stock_returns, T=T, horizons=cfg["windowing"]["horizons"],
        )
        out = processed_dir / f"features_T{T}{'_nosent' if no_sentiment else ''}.npz"
        np.savez_compressed(
            out,
            # store epoch-NANOSECONDS explicitly (pandas may hold datetime64[us])
            anchor_dates=ds.anchor_dates.astype("datetime64[ns]").astype("int64"),
            X_num=ds.X_num, X_macro=ds.X_macro, X_sent=ds.X_sent,
            y_ret=ds.y_ret, y_vol=ds.y_vol, y_corr=ds.y_corr,
            horizons=np.array(ds.horizons), stocks=np.array(ds.stocks),
            num_cols=np.array(num_cols), macro_cols=np.array(macro_cols),
            sent_cols=np.array(sent_cols),
        )
        logger.info("T=%d: %d samples -> %s", T, len(ds.anchor_dates), out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-sentiment", action="store_true")
    args = parser.parse_args()
    build(no_sentiment=args.no_sentiment)
