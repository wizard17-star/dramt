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
# Martingale / random walk (the honest null)
# --------------------------------------------------------------------------- #

def martingale_forecasts(
    returns: pd.Series,
    train_end: pd.Timestamp,
    anchors: pd.DatetimeIndex,
    horizons: list[int],
    vol_window: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-drift random walk: mu = 0 at every horizon.

    Why this baseline matters
    -------------------------
    Under the efficient-market null the best forecast of a future log return is
    zero (up to a tiny drift). Measured on this project's own test set, a
    constant-zero forecast scores MAE 0.03030, while ARIMA scores 0.03006 and
    DRAM-T 0.03070 -- i.e. ARIMA's "significant win" is a 0.8% improvement on
    predicting nothing, and the deep model is actually WORSE than nothing
    (R^2 = -0.026 against the zero forecast).

    Without this baseline in the table, a reader cannot tell whether ARIMA is
    a good forecaster or merely the model that shrinks hardest toward zero.
    Including it turns an apparently damning result ("ARIMA beats the proposed
    architecture") into the correct and defensible one ("mean returns are not
    predictable at these horizons; nothing beats the null by a meaningful
    margin").

    The volatility leg is a matching null: a trailing `vol_window`-day sample
    standard deviation using only returns up to and including the anchor. This
    gives the risk tables a naive reference too (the classic "historical
    volatility" benchmark), so GARCH's value can be judged against something.
    """
    r = returns.dropna()
    mu = np.zeros((len(anchors), len(horizons)))

    trailing = r.rolling(vol_window, min_periods=max(10, vol_window // 4)).std()
    # .loc on the anchor date uses data up to and including that day only
    vol_day = trailing.reindex(anchors).ffill().to_numpy()
    if np.isnan(vol_day).any():                      # very early anchors
        vol_day = np.nan_to_num(vol_day, nan=float(r.loc[:train_end].std()))
    vol = np.repeat(vol_day[:, None], len(horizons), axis=1)
    return mu, vol


# --------------------------------------------------------------------------- #
# HAR-RV (Corsi 2009) -- heterogeneous autoregression on realized variance
# --------------------------------------------------------------------------- #

def har_rv_forecasts(
    returns: pd.Series,
    train_end: pd.Timestamp,
    anchors: pd.DatetimeIndex,
    horizons: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """HAR-RV volatility forecasts; mu = 0 (HAR models variance, not the mean).

    Corsi's HAR regresses future realized variance on daily, weekly and
    monthly averages of past realized variance, capturing the long-memory of
    volatility with three terms. It is the standard modern benchmark and is
    reported to beat GARCH(1,1) by a wide margin on equity indices.

    Two honest deviations from the canonical specification, both forced by the
    data available here:
    - No intraday data, so realized variance is proxied by the squared daily
      log return (the standard low-frequency substitute). This proxy is noisy,
      which handicaps HAR relative to a true high-frequency RV implementation.
    - DIRECT multi-horizon estimation: for each horizon h a separate
      regression targets the CUMULATIVE variance over the next h days, rather
      than iterating a one-step model forward. Direct estimation avoids
      compounding one-step errors and matches how the other models here are
      evaluated.

    The regression is run in LOG variance space, as is standard in the HAR
    literature. This is not cosmetic: an OLS fit on variance LEVELS is not
    constrained to predict a positive variance, and extrapolating a
    calm-period fit to a burst an order of magnitude larger can drive the
    prediction negative, at which point clipping floors it and the model
    reports near-zero volatility exactly when volatility spikes. (Reproduced
    on synthetic data during development.) Log space makes forecasts positive
    by construction and linearises the multiplicative dynamics of volatility.
    Back-transformation uses the standard Jensen correction exp(m + s^2/2).

    Leakage: the design matrix at day t uses only RV up to t, and the OLS
    coefficients are estimated on rows whose ENTIRE target window lies at or
    before train_end.
    """
    r = returns.dropna()
    rv = r ** 2                                        # daily realized variance proxy
    # floor zero-return days so the log is finite; scaled to the sample so it
    # is negligible relative to typical variance
    floor = max(float(rv[rv > 0].quantile(0.01)) if (rv > 0).any() else 1e-12, 1e-12)
    rv = rv.clip(lower=floor)

    # min_periods=1: the weekly/monthly averages are computed from however many
    # observations exist so far rather than being NaN at the start of the
    # sample. This is strictly backward-looking. The alternative -- leaving
    # leading NaNs and back-filling them later -- would pull a FUTURE value
    # backwards, which is leakage even though it would only touch the first
    # couple of anchors.
    X = pd.concat([np.log(rv),
                   np.log(rv.rolling(5, min_periods=1).mean()),
                   np.log(rv.rolling(22, min_periods=1).mean())], axis=1)
    X.columns = ["rv_d", "rv_w", "rv_m"]
    X = X.dropna()

    n = len(anchors)
    mu = np.zeros((n, len(horizons)))
    vol = np.empty((n, len(horizons)))

    for hi, h in enumerate(horizons):
        # target: log cumulative variance over the next h days (strictly future)
        target = np.log(rv.rolling(h).sum().shift(-h).clip(lower=floor)).reindex(X.index)
        design = X.copy()
        design["y"] = target

        # a training row is usable only if its whole target window ends by train_end
        cutoff = train_end - pd.Timedelta(days=int(h * 1.6))
        train = design.loc[design.index <= cutoff].dropna()
        if len(train) < 50:
            train = design.loc[design.index <= train_end].dropna()

        A = np.column_stack([np.ones(len(train)), train[["rv_d", "rv_w", "rv_m"]].to_numpy()])
        y = train["y"].to_numpy()
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid_var = float(np.var(y - A @ beta))        # for the Jensen correction

        Xa = X.reindex(anchors).ffill()
        Aa = np.column_stack([np.ones(len(Xa)), Xa[["rv_d", "rv_w", "rv_m"]].to_numpy()])
        var_cum = np.exp(np.clip(Aa @ beta + 0.5 * resid_var, -50, 50))
        vol[:, hi] = np.sqrt(var_cum / h)              # per-day vol, matching the others

    return mu, vol


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
