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


# --------------------------------------------------------------------------- #
# TFT baseline (long-panel construction)
# --------------------------------------------------------------------------- #

def _fake_aligned(n_days=120, portfolio=("AAPL", "MSFT")):
    from src.models.baselines.tft import MACRO_FEATURES, PER_STOCK_FEATURES

    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2021-01-01", periods=n_days)
    data = {}
    for s in portfolio:
        for f in PER_STOCK_FEATURES:
            data[f"{s}_{f}"] = rng.normal(size=n_days)
    for m in MACRO_FEATURES:
        data[m] = rng.normal(size=n_days)
    return pd.DataFrame(data, index=idx)


def test_tft_long_frame_preserves_per_stock_values():
    """Wide -> long reshape must not cross-contaminate stocks: every row's
    features must come from that row's own ticker."""
    from src.models.baselines.tft import PER_STOCK_FEATURES, build_long_frame

    portfolio = ["AAPL", "MSFT"]
    daily = _fake_aligned(portfolio=portfolio)
    long = build_long_frame(daily, portfolio)

    assert len(long) == len(daily) * len(portfolio)
    for s in portfolio:
        sub = long[long["stock"] == s].sort_values("time_idx")
        for f in PER_STOCK_FEATURES:
            np.testing.assert_allclose(sub[f].to_numpy(), daily[f"{s}_{f}"].to_numpy())


def test_tft_long_frame_macro_shared_and_time_idx_monotonic():
    from src.models.baselines.tft import MACRO_FEATURES, build_long_frame

    portfolio = ["AAPL", "MSFT"]
    daily = _fake_aligned(portfolio=portfolio)
    long = build_long_frame(daily, portfolio)

    # macro is a shared block: identical across stocks at the same time_idx
    a = long[long["stock"] == "AAPL"].sort_values("time_idx")
    b = long[long["stock"] == "MSFT"].sort_values("time_idx")
    for m in MACRO_FEATURES:
        np.testing.assert_allclose(a[m].to_numpy(), b[m].to_numpy())
    # time_idx must index the aligned grid contiguously from 0
    np.testing.assert_array_equal(a["time_idx"].to_numpy(), np.arange(len(daily)))


# --------------------------------------------------------------------------- #
# Martingale null + HAR-RV
# --------------------------------------------------------------------------- #

def _fake_returns(n=800, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    # volatility clustering so HAR has something real to fit
    vol = np.zeros(n)
    vol[0] = 0.01
    for t in range(1, n):
        vol[t] = np.sqrt(0.000002 + 0.10 * (vol[t - 1] * rng.normal()) ** 2
                         + 0.88 * vol[t - 1] ** 2)
    return pd.Series(rng.normal(0, vol), index=idx)


def test_martingale_predicts_exactly_zero():
    from src.models.baselines.econometric import martingale_forecasts

    r = _fake_returns()
    anchors = r.index[600:700]
    mu, vol = martingale_forecasts(r, r.index[500], anchors, [1, 5, 10])
    assert (mu == 0.0).all(), "the martingale null must predict exactly zero"
    assert vol.shape == (len(anchors), 3)
    assert (vol > 0).all() and np.isfinite(vol).all()


def test_martingale_vol_uses_no_future_data():
    """Trailing volatility at an anchor must not change when the future does."""
    from src.models.baselines.econometric import martingale_forecasts

    r = _fake_returns()
    anchors = r.index[600:700]
    _, v1 = martingale_forecasts(r, r.index[500], anchors, [1, 5, 10])
    r2 = r.copy()
    r2.iloc[650:] = 99.0
    _, v2 = martingale_forecasts(r2, r.index[500], anchors, [1, 5, 10])
    np.testing.assert_allclose(v1[:50], v2[:50], rtol=1e-12)


def test_har_rv_shapes_and_positivity():
    from src.models.baselines.econometric import har_rv_forecasts

    r = _fake_returns()
    anchors = r.index[600:700]
    horizons = [1, 5, 10]
    mu, vol = har_rv_forecasts(r, r.index[500], anchors, horizons)
    assert mu.shape == (len(anchors), 3) and (mu == 0.0).all()
    assert vol.shape == (len(anchors), 3)
    assert (vol > 0).all() and np.isfinite(vol).all()


def _clustered_returns(n=1200, seed=5):
    """GARCH(1,1) returns: volatility is genuinely persistent, so HAR has a
    real relationship to learn. (On homoskedastic noise there is nothing to
    predict and a correctly-fitted HAR SHOULD be flat.)"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    vol = np.zeros(n)
    vol[0] = 0.01
    prev_e = 0.0
    for t in range(1, n):
        prev_e = vol[t - 1] * rng.normal()
        vol[t] = np.sqrt(2e-6 + 0.10 * prev_e ** 2 + 0.88 * vol[t - 1] ** 2)
    return pd.Series(rng.normal(0, vol), index=idx), vol


def test_har_rv_forecasts_track_future_realized_volatility():
    """HAR's forecast must correlate positively with the volatility actually
    realized over the following h days - the property that makes it a useful
    risk model rather than a constant."""
    from src.models.baselines.econometric import har_rv_forecasts

    r, _ = _clustered_returns()
    idx = r.index
    anchors = idx[900:1150]
    _, vol = har_rv_forecasts(r, idx[850], anchors, [5, 10])

    pos = {d: i for i, d in enumerate(idx)}
    ra = r.to_numpy()
    for hi, h in enumerate([5, 10]):
        realized = np.array([ra[pos[a] + 1: pos[a] + 1 + h].std() for a in anchors])
        assert np.corrcoef(vol[:, hi], realized)[0, 1] > 0.3


def test_har_rv_forecast_level_is_sane():
    """Log-space fit + Jensen correction must land near the true vol level,
    not orders of magnitude off (the failure mode of a levels-space fit)."""
    from src.models.baselines.econometric import har_rv_forecasts

    r, true_vol = _clustered_returns()
    idx = r.index
    anchors = idx[900:1150]
    _, vol = har_rv_forecasts(r, idx[850], anchors, [5])
    ratio = vol[:, 0].mean() / true_vol[900:1150].mean()
    assert 0.5 < ratio < 2.0, f"HAR vol level off by {ratio:.2f}x"
