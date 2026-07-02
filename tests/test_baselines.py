"""M5 baseline tests: deep model shapes + econometric smoke on synthetic data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baselines.deep import make_deep_baseline
from src.models.baselines.econometric import (
    arima_forecasts,
    dcc_garch_forecasts,
    garch_forecasts,
    garch_midas_forecasts,
)

B, T, F_NUM, F_SEN, S, H = 4, 20, 84, 30, 5, 6


@pytest.mark.parametrize("name", ["lstm", "gru", "cnn_bilstm", "cnn_bilstm_attn", "transformer"])
def test_deep_unimodal_shapes(name):
    model = make_deep_baseline(name, F_NUM, F_SEN, S, H)
    x = torch.randn(B, T, F_NUM)
    assert model(x).shape == (B, S, H)


def test_sentiment_lstm_shape():
    model = make_deep_baseline("sentiment_lstm", F_NUM, F_SEN, S, H)
    out = model(torch.randn(B, T, F_NUM), torch.randn(B, T, F_SEN))
    assert out.shape == (B, S, H)


@pytest.fixture(scope="module")
def synth_returns():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2018-01-01", periods=1200)
    # GARCH-ish returns: volatility clustering via AR(1) log-vol
    logv = np.zeros(1200)
    for t in range(1, 1200):
        logv[t] = 0.95 * logv[t - 1] + 0.2 * rng.normal()
    r = np.exp(-4.5 + 0.5 * logv) * rng.normal(size=1200)
    return pd.Series(r, index=idx)


def _anchor_setup(series):
    anchors = series.index[1000:1050]
    train_end = series.index[900]
    horizons = [1, 3, 10]
    return train_end, pd.DatetimeIndex(anchors), horizons


def test_arima_smoke(synth_returns):
    train_end, anchors, horizons = _anchor_setup(synth_returns)
    mu = arima_forecasts(synth_returns, train_end, anchors, horizons)
    assert mu.shape == (50, 3)
    assert np.isfinite(mu).all()
    assert np.abs(mu).max() < 0.1  # sane scale for daily returns


def test_garch_smoke(synth_returns):
    train_end, anchors, horizons = _anchor_setup(synth_returns)
    mu, vol = garch_forecasts(synth_returns, train_end, anchors, horizons)
    assert mu.shape == vol.shape == (50, 3)
    assert (vol > 0).all() and np.isfinite(vol).all()
    # vol forecasts should track the actual scale of returns
    realized = synth_returns.iloc[1000:1060].std()
    assert 0.1 * realized < vol.mean() < 10 * realized


def test_garch_midas_smoke(synth_returns):
    train_end, anchors, horizons = _anchor_setup(synth_returns)
    mu, vol = garch_midas_forecasts(synth_returns, train_end, anchors, horizons)
    assert mu.shape == vol.shape == (50, 3)
    assert (vol > 0).all() and np.isfinite(vol).all()


def test_dcc_garch_smoke(synth_returns):
    rng = np.random.default_rng(1)
    base = synth_returns.to_numpy()
    df = pd.DataFrame(
        {f"s{i}": 0.7 * base + 0.3 * base.std() * rng.normal(size=len(base)) for i in range(5)},
        index=synth_returns.index,
    )
    train_end, anchors, horizons = _anchor_setup(synth_returns)
    mu, vol, corr = dcc_garch_forecasts(df, train_end, anchors, horizons)
    assert mu.shape == vol.shape == (50, 5, 3)
    assert corr.shape == (50, 5, 5)
    assert (vol > 0).all()
    # valid correlation forecasts
    assert np.allclose(corr[:, range(5), range(5)], 1.0, atol=1e-6)
    ev = np.linalg.eigvalsh(corr)
    assert (ev > -1e-6).all()
    # correlated synthetic series -> forecasts should detect positive correlation
    off = corr[:, 0, 1]
    assert off.mean() > 0.2
