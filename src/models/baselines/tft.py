"""Temporal Fusion Transformer baseline (pytorch-forecasting).

Skipped in the CPU-era study for compute reasons; added here for the GPU run.

Fair-comparison protocol
------------------------
TFT is a long-format, per-series model, whereas DRAM-T consumes pre-windowed
tensors. To keep the comparison honest, every degree of freedom that matters
is matched to DRAM-T rather than to TFT's defaults:

- IDENTICAL folds and anchors. Fold indices come from the same
  walk_forward_folds call; predictions are produced only at the fold's test
  anchors and returned in that exact order.
- IDENTICAL target definition. DRAM-T predicts the CUMULATIVE h-day log
  return. TFT natively forecasts the next `max_prediction_length` steps of a
  series, so the target here is the DAILY log return and the h-day cumulative
  forecast is the cumsum of the first h predicted steps. These are the same
  quantity, so no re-definition of the metric is needed.
- IDENTICAL information set. Encoder length = T, and the per-stock features
  are the same technical + sentiment columns DRAM-T sees, plus the shared
  macro block. Nothing is exposed to TFT that DRAM-T cannot see.
- NO LEAKAGE. `time_varying_unknown_reals` for everything observed (including
  macro, which is released with a lag), so TFT may not peek at covariate
  values inside its own prediction window. Normalization is fitted by
  pytorch-forecasting on the training TimeSeriesDataSet only, and the
  validation/test sets are constructed with `from_dataset(..., predict=...)`
  so they inherit those training statistics.

Multi-stock handling: one series per portfolio member with the ticker as a
static categorical, so a single TFT is trained across the 5 stocks, matching
DRAM-T's joint modelling rather than giving TFT 5 independent models.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

PER_STOCK_FEATURES = [
    "log_return", "rsi", "macd", "macd_signal", "macd_hist", "boll_pct_b",
    "boll_bandwidth", "roll_vol", "atr", "obv_z", "log_volume", "momentum",
    "sent_mean", "sent_polarity", "news_volume", "sent_decay", "has_news",
]
MACRO_FEATURES = [
    "cpi", "unemployment", "fed_funds", "yield_10y", "consumer_sentiment",
    "industrial_production", "tau_cpi", "tau_unemployment",
    "tau_consumer_sentiment", "tau_industrial_production",
]


def build_long_frame(daily: pd.DataFrame, portfolio: list[str]) -> pd.DataFrame:
    """Wide aligned frame -> long panel (one row per stock-day) for TFT."""
    frames = []
    dates = daily.index
    time_idx = np.arange(len(dates))
    for stock in portfolio:
        cols = {f: daily[f"{stock}_{f}"].to_numpy() for f in PER_STOCK_FEATURES}
        cols.update({m: daily[m].to_numpy() for m in MACRO_FEATURES})
        frames.append(pd.DataFrame({
            "time_idx": time_idx,
            "stock": stock,
            "date": dates,
            **cols,
        }))
    out = pd.concat(frames, ignore_index=True)
    out["stock"] = out["stock"].astype("category")
    return out


def tft_forecasts(
    daily: pd.DataFrame,
    portfolio: list[str],
    anchor_dates: pd.DatetimeIndex,   # ALL dataset anchors (index-aligned to folds)
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    horizons: list[int],
    T: int,
    device: torch.device,
    max_epochs: int = 100,
    patience: int = 15,
    lr: float = 1e-3,
    hidden_size: int = 64,
    attention_head_size: int = 4,
    dropout: float = 0.2,
    batch_size: int = 64,
    seed: int = 42,
) -> np.ndarray:                      # (n_test, S, H) cumulative-return forecasts
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss

    pl.seed_everything(seed, workers=True)
    h_max = max(horizons)
    long = build_long_frame(daily, portfolio)

    # map anchor dates -> the integer time_idx of the aligned grid
    date_to_idx = {d: i for i, d in enumerate(daily.index)}
    anchor_pos = np.array([date_to_idx[pd.Timestamp(d)] for d in anchor_dates])

    # An anchor at grid position p means: encoder ends at p, decoder covers
    # p+1 .. p+h_max. TimeSeriesDataSet indexes a sample by its decoder start,
    # so training cutoffs are expressed in encoder-end terms and converted.
    train_end_pos = int(anchor_pos[train_idx][-1])
    val_end_pos = int(anchor_pos[val_idx][-1])

    unknown_reals = PER_STOCK_FEATURES + MACRO_FEATURES

    training = TimeSeriesDataSet(
        long[long.time_idx <= train_end_pos + h_max],
        time_idx="time_idx",
        target="log_return",
        group_ids=["stock"],
        max_encoder_length=T,
        min_encoder_length=T,
        max_prediction_length=h_max,
        min_prediction_length=h_max,
        static_categoricals=["stock"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=unknown_reals,
        # normalizer fitted on TRAIN rows only; val/test inherit it below
        target_normalizer=GroupNormalizer(groups=["stock"], transformation=None),
        add_relative_time_idx=True,
        add_target_scales=True,
        allow_missing_timesteps=False,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, long[long.time_idx <= val_end_pos + h_max],
        min_prediction_idx=train_end_pos + 1, stop_randomization=True,
    )

    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size * 4, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=lr,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=max(8, hidden_size // 4),
        loss=QuantileLoss(),
        log_interval=-1,
        optimizer="adam",
    )
    logger.info("TFT parameters: %.1fk", tft.size() / 1e3)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        gradient_clip_val=1.0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=patience, mode="min")],
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # ---- predict exactly at the fold's test anchors -----------------------
    test_pos = anchor_pos[test_idx]
    # decoder starts at anchor+1; keep rows whose encoder ends at a test anchor
    pred_frame = long[long.time_idx <= int(test_pos[-1]) + h_max]
    scoring = TimeSeriesDataSet.from_dataset(
        training, pred_frame, min_prediction_idx=int(test_pos[0]) + 1,
        stop_randomization=True,
    )
    loader = scoring.to_dataloader(train=False, batch_size=batch_size * 4, num_workers=0)

    raw = tft.predict(loader, mode="prediction", return_index=True,
                      trainer_kwargs={"accelerator": "gpu" if device.type == "cuda" else "cpu",
                                      "devices": 1, "logger": False,
                                      "enable_progress_bar": False})
    preds = raw.output.cpu().numpy() if hasattr(raw.output, "cpu") else np.asarray(raw.output)
    index = raw.index                                   # columns: time_idx, stock

    # index.time_idx is the DECODER START, so the anchor is time_idx - 1
    index = index.copy()
    index["anchor_pos"] = index["time_idx"] - 1
    lookup = {(int(r.anchor_pos), str(r.stock)): i for i, r in enumerate(index.itertuples())}

    out = np.full((len(test_idx), len(portfolio), len(horizons)), np.nan)
    missing = 0
    for ai, p in enumerate(test_pos):
        for si, stock in enumerate(portfolio):
            key = (int(p), stock)
            if key not in lookup:
                missing += 1
                continue
            daily_fc = preds[lookup[key]]               # (h_max,) daily returns
            cum = np.cumsum(daily_fc)
            out[ai, si, :] = [cum[h - 1] for h in horizons]
    if missing:
        raise RuntimeError(
            f"TFT produced no forecast for {missing} (anchor, stock) pairs - "
            "test anchors and prediction index are misaligned"
        )
    return out
