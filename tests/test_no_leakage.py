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


def _write_fold_npz(tmpdir: Path, n_val=150, n_test=120, S=5, H=3,
                    horizons=(1, 5, 10), test_sigma_boost=1.0, seed=7):
    """Minimal val/test prediction pair in the on-disk format evaluate.py reads."""
    rng = np.random.default_rng(seed)
    hs = np.array(horizons)
    sig_v = np.full((n_val, S, H), 0.02)
    y_v = rng.normal(0, 0.02, (n_val, S, H))
    np.savez_compressed(tmpdir / "val_predictions.npz", mu=np.zeros((n_val, S, H)),
                        sigma=sig_v, df=np.array([]), y_ret=y_v)
    sig_t = np.full((n_test, S, H), 0.02)
    y_t = rng.normal(0, 0.02 * test_sigma_boost, (n_test, S, H))
    np.savez_compressed(tmpdir / "test_predictions.npz", mu=np.zeros((n_test, S, H)),
                        sigma=sig_t, corr=np.tile(np.eye(S), (n_test, 1, 1)),
                        df=np.array([]), y_ret=y_t,
                        y_vol=np.full((n_test, S, H), 0.02),
                        y_corr=np.tile(np.eye(S), (n_test, 1, 1)),
                        test_idx=np.arange(n_test),
                        anchor_dates=np.arange(n_test))
    return tmpdir / "test_predictions.npz", hs


def test_rolling_sigma_calibration_no_lookahead(tmp_path):
    """The multiplier at test anchor i may only use residuals resolved by i.

    Poisoning every test residual from anchor k onward must leave the
    multipliers at anchors <= k untouched: a forecast made at anchor j for
    horizon h is only observable at j+h, so anchor i can at most have seen
    anchor i-h.
    """
    from src.evaluate import _rolling_sigma_scale

    horizons = [1, 5, 10]
    a = tmp_path / "a"
    a.mkdir()
    p_a, _ = _write_fold_npz(a, horizons=horizons)
    s_a = _rolling_sigma_scale(p_a, horizons, window=60)

    # same data, but the future (anchors >= k) is replaced with garbage
    k = 60
    b = tmp_path / "b"
    b.mkdir()
    p_b, _ = _write_fold_npz(b, horizons=horizons)
    z = dict(np.load(p_b))
    z["y_ret"][k:] = 99.0
    np.savez_compressed(p_b, **z)
    s_b = _rolling_sigma_scale(p_b, horizons, window=60)

    for hi, h in enumerate(horizons):
        # anchor i can legitimately have seen up to anchor i-h, so anchors
        # up to k+h-1 must be unaffected by poisoning from k onward
        safe = k + h - 1
        np.testing.assert_allclose(s_a[:safe, 0, hi], s_b[:safe, 0, hi], rtol=1e-12)
    # and the poison must actually show up later, otherwise the test is vacuous
    assert not np.allclose(s_a[-1, 0, 0], s_b[-1, 0, 0])


def test_rolling_sigma_calibration_tracks_regime_shift(tmp_path):
    """A test block twice as volatile as validation must push the rolling
    multiplier up toward 2, whereas the global val-fitted constant stays ~1."""
    from src.evaluate import _rolling_sigma_scale, _val_sigma_scale

    horizons = [1, 5, 10]
    d = tmp_path / "shift"
    d.mkdir()
    p, _ = _write_fold_npz(d, n_test=300, horizons=horizons, test_sigma_boost=2.0)
    s_roll = _rolling_sigma_scale(p, horizons, window=60)
    s_glob = _val_sigma_scale(p)

    assert abs(s_glob - 1.0) < 0.15          # fitted on val: blind to the shift
    assert s_roll[0, 0, 0] < 1.3             # starts from the val-seeded pool
    assert s_roll[-1, 0, 0] > 1.7            # adapts to the 2x regime


def test_extended_regime_signals_no_lookahead(tmp_path):
    """Every extended gate signal at anchor t must use only rows <= t, so
    poisoning the future must not change any earlier value."""
    from src.data.regime import build_extended_regime

    rng = np.random.default_rng(11)
    n = 600
    idx = pd.bdate_range("2020-01-01", periods=n)
    portfolio = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
    daily = pd.DataFrame(
        {f"{t}_log_return": rng.normal(0, 0.01, n) for t in portfolio}, index=idx)

    raw = tmp_path / "raw" / "equities"
    raw.mkdir(parents=True)
    vix = pd.DataFrame({"Close": 15 + rng.normal(0, 2, n)}, index=idx)
    vix.index.name = "date"
    vix.to_csv(raw / "IDX_VIX.csv")

    a = build_extended_regime(daily, tmp_path / "raw", portfolio, idx)

    # poison the future
    k = 400
    daily2 = daily.copy()
    daily2.iloc[k:] = 99.0
    vix2 = vix.copy()
    vix2.iloc[k:] = 999.0
    vix2.to_csv(raw / "IDX_VIX.csv")
    b = build_extended_regime(daily2, tmp_path / "raw", portfolio, idx)

    np.testing.assert_allclose(a[:k], b[:k], rtol=1e-10, atol=1e-10)
    # and the poison must be visible afterwards, else the test proves nothing
    assert not np.allclose(a[k:], b[k:])


def test_extended_regime_drawdown_is_non_positive(tmp_path):
    """Drawdown is measured from a trailing running maximum, so it can never
    be positive - a positive value would mean the running max saw the future."""
    from src.data.regime import build_extended_regime

    rng = np.random.default_rng(12)
    n = 400
    idx = pd.bdate_range("2020-01-01", periods=n)
    portfolio = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
    daily = pd.DataFrame(
        {f"{t}_log_return": rng.normal(0.001, 0.01, n) for t in portfolio}, index=idx)
    raw = tmp_path / "raw" / "equities"
    raw.mkdir(parents=True)
    v = pd.DataFrame({"Close": 15 + rng.normal(0, 2, n)}, index=idx)
    v.index.name = "date"
    v.to_csv(raw / "IDX_VIX.csv")

    out = build_extended_regime(daily, tmp_path / "raw", portfolio, idx)
    drawdown = out[:, 3]
    assert (drawdown <= 1e-9).all()


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
