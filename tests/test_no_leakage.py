"""M2 leakage and correctness tests: indicators, MIDAS alignment, windows, splits."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.indicators import compute_indicators
from src.data.macro_midas import align_macro_daily, beta_weights, midas_tau_features
from src.data.splits import Standardizer, walk_forward_folds
from src.data.windows import build_windows

IND_CFG = {
    "rsi_period": 14, "macd": [12, 26, 9], "bollinger": [20, 2],
    "rolling_vol_window": 20, "atr_period": 14, "momentum_window": 10,
}


def _fake_ohlcv(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    vol = rng.integers(1e6, 5e6, n).astype(float)
    return pd.DataFrame({"Close": close, "High": high, "Low": low,
                         "Open": close, "Volume": vol}, index=idx)


def test_indicators_no_lookahead():
    """Truncating the future must not change past indicator values."""
    df = _fake_ohlcv(300)
    full = compute_indicators(df, IND_CFG)
    trunc = compute_indicators(df.iloc[:200], IND_CFG)
    pd.testing.assert_frame_equal(full.iloc[:200], trunc, check_exact=False, atol=1e-10)


def test_midas_no_lookahead():
    """Removing future macro rows must not change tau/aligned values in the past."""
    rng = np.random.default_rng(1)
    m_idx = pd.date_range("2019-01-01", periods=60, freq="MS")
    macro = {"cpi": pd.DataFrame({"CPIAUCSL": 100 + np.cumsum(rng.normal(0.2, 0.1, 60))}, index=m_idx)}
    days = pd.bdate_range("2019-06-01", "2023-06-01")

    full_tau = midas_tau_features(macro, days, n_lags=12)
    full_daily = align_macro_daily(macro, days)

    macro_trunc = {"cpi": macro["cpi"].iloc[:48]}  # drop last 12 months
    cutoff = pd.Timestamp("2022-06-01")
    days_past = days[days < cutoff]
    trunc_tau = midas_tau_features(macro_trunc, days_past, n_lags=12)
    trunc_daily = align_macro_daily(macro_trunc, days_past)

    pd.testing.assert_frame_equal(full_tau.loc[days_past], trunc_tau, atol=1e-12, check_exact=False)
    pd.testing.assert_frame_equal(full_daily.loc[days_past], trunc_daily, atol=1e-12, check_exact=False)


def test_midas_publication_lag():
    """A monthly value stamped Jan-01 must not be visible before Feb (period end + lag)."""
    m_idx = pd.to_datetime(["2022-01-01"])
    macro = {"cpi": pd.DataFrame({"CPIAUCSL": [100.0]}, index=m_idx)}
    days = pd.bdate_range("2022-01-01", "2022-03-01")
    aligned = align_macro_daily(macro, days, publication_lag_days=15)
    # available only from Jan-31 + 15d = Feb-15 onward
    assert aligned.loc[days[days < "2022-02-15"], "cpi"].isna().all()
    assert (aligned.loc[days[days >= "2022-02-16"], "cpi"] == 100.0).all()


def test_beta_weights():
    w = beta_weights(12, 1.0, 5.0)
    assert np.isclose(w.sum(), 1.0)
    assert (w > 0).all()
    assert w[0] > w[-1]  # decaying: recent lag weighted more


def _fake_frames(n_days=200, T=20):
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2021-01-01", periods=n_days)
    num = pd.DataFrame(rng.normal(size=(n_days, 4)), index=idx, columns=list("abcd"))
    mac = pd.DataFrame(rng.normal(size=(n_days, 2)), index=idx, columns=["m1", "m2"])
    sen = pd.DataFrame(rng.normal(size=(n_days, 3)), index=idx, columns=["s1", "s2", "s3"])
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(n_days, 5)), index=idx,
                        columns=["AAPL", "GOOGL", "MSFT", "AMZN", "META"])
    return num, mac, sen, rets


def test_windows_targets_correct():
    num, mac, sen, rets = _fake_frames()
    T, horizons = 20, [1, 3, 5]
    ds = build_windows(num, mac, sen, rets, T=T, horizons=horizons)

    i = 10  # arbitrary sample
    t = num.index.get_loc(ds.anchor_dates[i])
    # input window is rows [t-T+1 .. t]
    np.testing.assert_allclose(ds.X_num[i], num.iloc[t - T + 1 : t + 1].to_numpy(), rtol=1e-6)
    # point target: cumulative future returns, strictly after t
    r = rets.to_numpy()
    for j, h in enumerate(horizons):
        np.testing.assert_allclose(ds.y_ret[i, :, j], r[t + 1 : t + 1 + h].sum(axis=0), rtol=1e-5)
    # vol target for h=3 is std of the 3 future daily returns
    np.testing.assert_allclose(ds.y_vol[i, :, 1], r[t + 1 : t + 4].std(axis=0, ddof=1), rtol=1e-5)
    # correlation target is over the max horizon
    c = np.corrcoef(r[t + 1 : t + 6].T)
    np.testing.assert_allclose(ds.y_corr[i], c, rtol=1e-4)
    # correlation matrix validity
    assert np.allclose(np.diag(ds.y_corr[i]), 1.0, atol=1e-5)
    assert np.allclose(ds.y_corr[i], ds.y_corr[i].T, atol=1e-6)


def test_windows_no_future_in_inputs():
    """Changing FUTURE feature rows must not change a sample's inputs."""
    num, mac, sen, rets = _fake_frames()
    ds1 = build_windows(num, mac, sen, rets, T=20, horizons=[1, 5])
    num2 = num.copy()
    num2.iloc[150:] = 999.0  # poison the future
    ds2 = build_windows(num2, mac, sen, rets, T=20, horizons=[1, 5])
    # all samples anchored before day 150 must be identical
    anchor_pos = np.array([num.index.get_loc(d) for d in ds1.anchor_dates])
    safe = anchor_pos < 150
    np.testing.assert_array_equal(ds1.X_num[safe], ds2.X_num[safe])


def test_walk_forward_folds():
    folds = walk_forward_folds(1000, n_folds=4, eval_frac=0.4, purge_gap=30)
    assert len(folds) == 4
    prev_train_len = 0
    for f in folds:
        # ordering: train < (gap) < val < test
        assert f.train_idx.max() < f.val_idx.min() - 29
        assert f.val_idx.max() < f.test_idx.min()
        # expanding train
        assert len(f.train_idx) > prev_train_len
        prev_train_len = len(f.train_idx)
        # no overlap of any kind
        assert not (set(f.train_idx) & set(f.val_idx))
        assert not (set(f.val_idx) & set(f.test_idx))
    # successive test blocks are disjoint and ordered
    assert folds[0].test_idx.max() < folds[1].val_idx.min() + len(folds[1].val_idx)


def test_standardizer_train_only():
    rng = np.random.default_rng(3)
    X = rng.normal(5.0, 2.0, size=(100, 10, 4))
    std = Standardizer().fit(X[:60])
    Z = std.transform(X[:60])
    # train part is ~z-scored
    assert abs(Z.mean()) < 0.05 and abs(Z.std() - 1.0) < 0.05
    # stats came from train only: transforming shifted test data leaves shift visible
    Z_test = std.transform(X[60:] + 10.0)
    assert Z_test.mean() > 3.0
