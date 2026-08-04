"""News-stratified accuracy and volatility-targeting economic evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import (
    buy_and_hold,
    news_mask_for_anchors,
    stratified_accuracy,
    vol_targeted_portfolio,
)

PORTFOLIO = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
HORIZONS = [1, 5, 10]


def _daily(n=300, news_from=100, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    data = {}
    for i, t in enumerate(PORTFOLIO):
        data[f"{t}_log_return"] = rng.normal(0, 0.01, n)
        h = np.zeros(n)
        h[news_from + i * 10:] = 1.0        # each stock starts having news later
        data[f"{t}_has_news"] = h
    return pd.DataFrame(data, index=idx)


def _write_run(runs: Path, name: str, daily, anchors_pos, T=20, seed=0,
               sigma_scale=1.0, n_folds=1):
    rng = np.random.default_rng(seed)
    S, H = len(PORTFOLIO), len(HORIZONS)
    for k in range(n_folds):
        d = runs / name / f"fold{k}"
        d.mkdir(parents=True, exist_ok=True)
        n = len(anchors_pos)
        dates = daily.index[anchors_pos]
        y = rng.normal(0, 0.02, (n, S, H))
        np.savez_compressed(
            d / "test_predictions.npz",
            mu=rng.normal(0, 0.002, (n, S, H)),
            sigma=np.full((n, S, H), 0.02 * sigma_scale),
            df=np.array([]), corr=np.tile(np.eye(S), (n, 1, 1)),
            weights=rng.random((n, 3)), y_ret=y,
            y_vol=np.full((n, S, H), 0.02),
            y_corr=np.tile(np.eye(S), (n, 1, 1)),
            test_idx=np.arange(n),
            anchor_dates=dates.values.astype("datetime64[ns]").astype("int64"),
        )


# --------------------------------------------------------------------------- #
# news stratification
# --------------------------------------------------------------------------- #

def test_news_mask_uses_only_the_input_window():
    """An anchor may only be marked 'has news' from days inside its own
    T-day input window - never from the future."""
    daily = _daily(n=300, news_from=200)
    T = 20
    anchors = np.arange(150, 260)
    ns = daily.index[anchors].values.astype("datetime64[ns]").astype("int64")
    mask = news_mask_for_anchors(daily, PORTFOLIO, ns, T)

    # AAPL news starts at grid position 200; an anchor at p sees [p-19, p]
    # so the first anchor that can see news is p = 200
    aapl = mask[:, 0]
    first_true = anchors[np.argmax(aapl)] if aapl.any() else None
    assert first_true == 200, f"expected first news anchor at 200, got {first_true}"
    # nothing before that
    assert not aapl[anchors < 200].any()


def test_news_mask_window_length_matters():
    """A longer window can see news an earlier-starting shorter window cannot."""
    daily = _daily(n=300, news_from=200)
    anchors = np.arange(150, 260)
    ns = daily.index[anchors].values.astype("datetime64[ns]").astype("int64")
    short = news_mask_for_anchors(daily, PORTFOLIO, ns, 5)
    long_ = news_mask_for_anchors(daily, PORTFOLIO, ns, 40)
    # a 40-day window sees news at least as often as a 5-day one
    assert long_[:, 0].sum() >= short[:, 0].sum()


def test_stratified_accuracy_splits_all_observations(tmp_path):
    daily = _daily(n=300, news_from=150)
    anchors = np.arange(200, 280)
    runs = tmp_path / "runs"
    _write_run(runs, "m", daily, anchors)
    rows = stratified_accuracy(runs, "m", daily, PORTFOLIO, T=20)
    assert rows
    total = sum(r["n_obs"] for r in rows)
    assert total == len(anchors) * len(PORTFOLIO) * len(HORIZONS)
    assert {r["stratum"] for r in rows} <= {"news", "no_news"}


# --------------------------------------------------------------------------- #
# economic evaluation
# --------------------------------------------------------------------------- #

def test_vol_targeting_hits_the_target_with_a_correct_sigma(tmp_path):
    """If sigma matches the data's true volatility, the levered portfolio's
    realized volatility must land near the target. This is the whole point of
    the metric."""
    rng = np.random.default_rng(3)
    n, S, H = 800, len(PORTFOLIO), len(HORIZONS)
    true_daily_vol = 0.02
    runs = tmp_path / "runs"
    d = runs / "good" / "fold0"
    d.mkdir(parents=True)
    y = rng.normal(0, true_daily_vol, (n, S, H))
    np.savez_compressed(
        d / "test_predictions.npz",
        mu=np.zeros((n, S, H)), sigma=np.full((n, S, H), true_daily_vol),
        df=np.array([]), corr=np.tile(np.eye(S), (n, 1, 1)),
        weights=rng.random((n, 3)), y_ret=y, y_vol=np.full((n, S, H), true_daily_vol),
        y_corr=np.tile(np.eye(S), (n, 1, 1)), test_idx=np.arange(n),
        anchor_dates=np.arange(n))

    w = np.full(S, 0.2)
    res = vol_targeted_portfolio(runs, "good", w, HORIZONS, "global", 120,
                                 target_vol_annual=0.10, max_leverage=10.0)
    assert res is not None
    assert abs(res["ann_vol"] - 0.10) < 0.02, res["ann_vol"]


def test_vol_targeting_penalises_a_miscalibrated_sigma(tmp_path):
    """A model whose sigma is 3x too small over-levers and overshoots the
    volatility target, so vol_target_error must be worse than a correct model."""
    rng = np.random.default_rng(4)
    n, S, H = 800, len(PORTFOLIO), len(HORIZONS)
    true_vol = 0.02
    runs = tmp_path / "runs"
    for name, scale in (("good", 1.0), ("under", 1 / 3)):
        d = runs / name / "fold0"
        d.mkdir(parents=True)
        y = rng.normal(0, true_vol, (n, S, H))
        np.savez_compressed(
            d / "test_predictions.npz",
            mu=np.zeros((n, S, H)), sigma=np.full((n, S, H), true_vol * scale),
            df=np.array([]), corr=np.tile(np.eye(S), (n, 1, 1)),
            weights=rng.random((n, 3)), y_ret=y,
            y_vol=np.full((n, S, H), true_vol),
            y_corr=np.tile(np.eye(S), (n, 1, 1)), test_idx=np.arange(n),
            anchor_dates=np.arange(n))
    w = np.full(S, 0.2)
    good = vol_targeted_portfolio(runs, "good", w, HORIZONS, "global", 120,
                                  target_vol_annual=0.10, max_leverage=10.0)
    under = vol_targeted_portfolio(runs, "under", w, HORIZONS, "global", 120,
                                   target_vol_annual=0.10, max_leverage=10.0)
    assert under["ann_vol"] > good["ann_vol"]
    assert under["vol_target_error"] > good["vol_target_error"]


def test_leverage_is_capped(tmp_path):
    rng = np.random.default_rng(5)
    n, S, H = 200, len(PORTFOLIO), len(HORIZONS)
    runs = tmp_path / "runs"
    d = runs / "tiny_sigma" / "fold0"
    d.mkdir(parents=True)
    np.savez_compressed(
        d / "test_predictions.npz",
        mu=np.zeros((n, S, H)), sigma=np.full((n, S, H), 1e-6),   # absurdly confident
        df=np.array([]), corr=np.tile(np.eye(S), (n, 1, 1)),
        weights=rng.random((n, 3)), y_ret=rng.normal(0, 0.02, (n, S, H)),
        y_vol=np.full((n, S, H), 0.02), y_corr=np.tile(np.eye(S), (n, 1, 1)),
        test_idx=np.arange(n), anchor_dates=np.arange(n))
    res = vol_targeted_portfolio(runs, "tiny_sigma", np.full(S, 0.2), HORIZONS,
                                 "global", 120, max_leverage=3.0)
    assert res["mean_leverage"] <= 3.0 + 1e-9


def test_buy_and_hold_is_unlevered(tmp_path):
    daily = _daily(n=300)
    anchors = np.arange(200, 280)
    runs = tmp_path / "runs"
    _write_run(runs, "m", daily, anchors)
    res = buy_and_hold(runs, "m", np.full(len(PORTFOLIO), 0.2), HORIZONS)
    assert res["mean_leverage"] == 1.0
    assert res["n_days"] == len(anchors)
