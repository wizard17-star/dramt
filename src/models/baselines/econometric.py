"""Econometric baselines: ARIMA, GARCH(1,1), GARCH-MIDAS, DCC-GARCH.

Protocol (identical information set to the deep models, no refitting per day):
- Parameters are estimated ONCE per fold on the training returns only.
- Forecasts at each test anchor t use fixed parameters + data up to t
  (expanding information, filtered — the standard walk-forward evaluation
  for econometric models).
- Returns are modeled in PERCENT units (x100) for optimizer stability
  (standard for the `arch` package); outputs are converted back.

Simplifications (documented honestly):
- GARCH-MIDAS uses the common two-step approach: long-run component tau from a
  Beta-weighted moving average of monthly realized variance (fixed Beta(1,5)
  weights as in config), then GARCH(1,1) fit on tau-standardized returns.
  Joint MLE of the MIDAS parameters is not performed.
- DCC(1,1) is estimated by two-step Gaussian QMLE with correlation targeting;
  the h-step correlation forecast uses the standard mean-reversion
  approximation R_{t+h} ~ (1-(a+b)^h)*Rbar + (a+b)^h * R_t.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

SCALE = 100.0  # percent units


# --------------------------------------------------------------------------- #
# ARIMA
# --------------------------------------------------------------------------- #

def arima_forecasts(
    returns: pd.Series,           # full daily log-return series (raw units)
    train_end: pd.Timestamp,      # last date usable for parameter estimation
    anchors: pd.DatetimeIndex,    # test anchor days
    horizons: list[int],
    order: tuple[int, int, int] = (1, 0, 1),
) -> np.ndarray:                  # (n_anchors, H) cumulative-return forecasts
    from statsmodels.tsa.arima.model import ARIMA

    r = returns.dropna() * SCALE
    train = r.loc[:train_end]
    model = ARIMA(train.to_numpy(), order=order,
                  enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(method_kwargs={"maxiter": 200})

    h_max = max(horizons)
    out = np.empty((len(anchors), len(horizons)))
    for i, t in enumerate(anchors):
        hist = r.loc[:t].to_numpy()
        filtered = res.apply(hist, refit=False)
        fc = filtered.forecast(steps=h_max)           # daily-return forecasts
        cum = np.cumsum(fc)
        out[i] = [cum[h - 1] for h in horizons]
    return out / SCALE


# --------------------------------------------------------------------------- #
# GARCH(1,1)
# --------------------------------------------------------------------------- #

def garch_forecasts(
    returns: pd.Series,
    train_end: pd.Timestamp,
    anchors: pd.DatetimeIndex,
    horizons: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mu (n,H) cumulative, vol (n,H) per-day realized-vol forecasts)."""
    from arch import arch_model

    r = (returns.dropna() * SCALE)
    am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(last_obs=train_end, disp="off")
    fixed = am.fix(res.params.to_numpy())

    h_max = max(horizons)
    fc = fixed.forecast(horizon=h_max, start=anchors[0], reindex=False)
    var_fc = fc.variance                     # rows indexed by forecast origin
    mean_fc = fc.mean

    var_rows = var_fc.reindex(anchors).to_numpy()   # (n, h_max) sigma^2_{t+1..t+h}
    mu_daily = mean_fc.reindex(anchors).to_numpy()[:, 0]  # constant mean per day

    mu = np.stack([mu_daily * h for h in horizons], axis=1) / SCALE
    vol = np.stack(
        [np.sqrt(var_rows[:, :h].mean(axis=1)) for h in horizons], axis=1
    ) / SCALE
    return mu, vol


# --------------------------------------------------------------------------- #
# GARCH-MIDAS (two-step)
# --------------------------------------------------------------------------- #

def _beta_weights(n_lags: int, a: float = 1.0, b: float = 5.0) -> np.ndarray:
    k = np.arange(1, n_lags + 1, dtype=float)
    x = k / (n_lags + 1.0)
    w = x ** (a - 1.0) * (1.0 - x) ** (b - 1.0)
    return w / w.sum()


def garch_midas_forecasts(
    returns: pd.Series,
    train_end: pd.Timestamp,
    anchors: pd.DatetimeIndex,
    horizons: list[int],
    n_lags: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-step GARCH-MIDAS: tau from Beta-weighted monthly realized variance;
    GARCH(1,1) on tau-standardized returns. Returns (mu, vol) like garch_forecasts."""
    from arch import arch_model

    r = returns.dropna() * SCALE
    rv_m = r.groupby(pd.Grouper(freq="ME")).apply(lambda s: (s ** 2).sum() / max(len(s), 1))
    w = _beta_weights(n_lags)

    # tau available at day t = weighted sum of the last n_lags COMPLETED months
    tau_m = pd.Series(index=rv_m.index, dtype=float)
    vals = rv_m.to_numpy()
    for i in range(n_lags, len(rv_m)):
        tau_m.iloc[i] = float(np.dot(w, vals[i - n_lags : i][::-1]))
    # month m's tau applies to days in month m+1 (uses only completed months)
    tau_daily = tau_m.reindex(r.index, method="ffill").shift(1).ffill().bfill()
    tau_daily = tau_daily.clip(lower=1e-8)

    z = r / np.sqrt(tau_daily)
    am = arch_model(z, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(last_obs=train_end, disp="off")
    fixed = am.fix(res.params.to_numpy())
    h_max = max(horizons)
    fc = fixed.forecast(horizon=h_max, start=anchors[0], reindex=False)
    g_rows = fc.variance.reindex(anchors).to_numpy()          # short-run component
    mu_daily = fc.mean.reindex(anchors).to_numpy()[:, 0]

    tau_anchor = tau_daily.reindex(anchors).to_numpy()        # (n,)
    var_rows = g_rows * tau_anchor[:, None]                    # total variance
    mu = np.stack([mu_daily * np.sqrt(tau_anchor) * h for h in horizons], axis=1) / SCALE
    vol = np.stack(
        [np.sqrt(var_rows[:, :h].mean(axis=1)) for h in horizons], axis=1
    ) / SCALE
    return mu, vol


# --------------------------------------------------------------------------- #
# DCC-GARCH (two-step QMLE with correlation targeting)
# --------------------------------------------------------------------------- #

def dcc_garch_forecasts(
    returns_df: pd.DataFrame,     # (days, S) raw log returns
    train_end: pd.Timestamp,
    anchors: pd.DatetimeIndex,
    horizons: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (mu (n,S,H), vol (n,S,H), corr (n,S,S) at max horizon)."""
    from arch import arch_model

    R = returns_df.dropna() * SCALE
    S = R.shape[1]

    # step 1: univariate GARCH(1,1) per stock, filtered over the full series
    sig = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
    mus, vols = [], []
    for c in R.columns:
        am = arch_model(R[c], mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
        res = am.fit(last_obs=train_end, disp="off")
        fixed = am.fix(res.params.to_numpy())
        sig[c] = np.sqrt(fixed.conditional_volatility ** 2)
        h_max = max(horizons)
        fc = fixed.forecast(horizon=h_max, start=anchors[0], reindex=False)
        var_rows = fc.variance.reindex(anchors).to_numpy()
        mu_daily = fc.mean.reindex(anchors).to_numpy()[:, 0]
        mus.append(np.stack([mu_daily * h for h in horizons], axis=1))
        vols.append(np.stack(
            [np.sqrt(var_rows[:, :h].mean(axis=1)) for h in horizons], axis=1
        ))
    mu = np.stack(mus, axis=1) / SCALE                     # (n,S,H)
    vol = np.stack(vols, axis=1) / SCALE

    eps = (R / sig).dropna()
    eps_train = eps.loc[:train_end].to_numpy()
    Qbar = np.corrcoef(eps_train.T)

    def dcc_negll(params: np.ndarray, E: np.ndarray) -> float:
        a, b = params
        if a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        Q = Qbar.copy()
        nll = 0.0
        for t in range(1, len(E)):
            e_prev = E[t - 1][:, None]
            Q = (1 - a - b) * Qbar + a * (e_prev @ e_prev.T) + b * Q
            d = np.sqrt(np.clip(np.diag(Q), 1e-12, None))
            Rt = Q / np.outer(d, d)
            try:
                Rinv = np.linalg.inv(Rt)
                _, logdet = np.linalg.slogdet(Rt)
            except np.linalg.LinAlgError:
                return 1e10
            e = E[t]
            nll += 0.5 * (logdet + e @ Rinv @ e - e @ e)
        return nll

    res_opt = minimize(dcc_negll, x0=np.array([0.02, 0.95]), args=(eps_train,),
                       method="Nelder-Mead", options={"maxiter": 200, "xatol": 1e-4})
    a, b = res_opt.x
    logger.info("DCC params: a=%.4f b=%.4f (nll=%.1f, converged=%s)",
                a, b, res_opt.fun, res_opt.success)

    # filter Q_t through the full sample, record R forecast at each anchor
    E_full = eps.to_numpy()
    dates = eps.index
    h_max = max(horizons)
    persist = (a + b) ** h_max
    anchor_set = {pd.Timestamp(t): i for i, t in enumerate(anchors)}
    corr = np.empty((len(anchors), S, S))
    Q = Qbar.copy()
    for t in range(1, len(E_full)):
        e_prev = E_full[t - 1][:, None]
        Q = (1 - a - b) * Qbar + a * (e_prev @ e_prev.T) + b * Q
        ts = dates[t]
        if ts in anchor_set:
            d = np.sqrt(np.clip(np.diag(Q), 1e-12, None))
            Rt = Q / np.outer(d, d)
            corr[anchor_set[ts]] = (1 - persist) * Qbar + persist * Rt
    return mu, vol, corr
