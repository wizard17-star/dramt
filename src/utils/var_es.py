"""Portfolio VaR / Expected Shortfall utilities + coverage backtests.

Conventions:
- Portfolio h-day return: r_p = w^T r  (r = per-stock cumulative log returns,
  treated as arithmetic for small-h approximation; documented simplification).
- VaR_alpha (loss quantile, positive number): VaR = -(mu_p + sigma_p * z_alpha)
  with z_alpha = Phi^{-1}(1 - alpha_conf)... concretely for 95% confidence,
  z = Phi^{-1}(0.05) = -1.645, VaR = -(mu_p - 1.645 sigma_p).
- A BREACH at anchor t: realized portfolio return < -VaR_t.
- ES_alpha under normality: ES = -(mu_p - sigma_p * phi(z)/0.05) for 95%.
"""
from __future__ import annotations

import numpy as np
from scipy import special, stats


def portfolio_moments(
    mu: np.ndarray,       # (n, S) per-stock cumulative-return forecasts at horizon h
    sigma: np.ndarray,    # (n, S) per-stock vol forecasts at horizon h (same units as mu)
    corr: np.ndarray,     # (n, S, S) correlation forecasts
    weights: np.ndarray,  # (S,)
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mu_p (n,), sigma_p (n,)) via Sigma = D R D."""
    mu_p = mu @ weights
    D = sigma[:, :, None] * np.eye(sigma.shape[1])[None]
    Sigma = D @ corr @ D
    var_p = np.einsum("i,nij,j->n", weights, Sigma, weights)
    return mu_p, np.sqrt(np.clip(var_p, 1e-18, None))


def var_es_normal(mu_p: np.ndarray, sigma_p: np.ndarray, conf: float = 0.95):
    """Parametric (normal) VaR and ES at `conf`; positive = loss magnitude."""
    z = stats.norm.ppf(1.0 - conf)                    # e.g. -1.645
    var = -(mu_p + sigma_p * z)
    es = -(mu_p - sigma_p * stats.norm.pdf(z) / (1.0 - conf))
    return var, es


def student_t_std_factor(nu: np.ndarray | float) -> np.ndarray:
    """std / scale for a Student-t with nu>2 degrees of freedom.

    A t_nu variable with SCALE s has standard deviation s*sqrt(nu/(nu-2)).
    Everything the model emits is a scale; realized volatility is a standard
    deviation, so the two must be reconciled through this factor before any
    VolRMSE-style comparison.
    """
    nu = np.asarray(nu, dtype=float)
    return np.sqrt(nu / np.clip(nu - 2.0, 1e-6, None))


def var_es_student_t(mu_p: np.ndarray, scale_p: np.ndarray,
                     nu: np.ndarray | float, conf: float = 0.95):
    """Parametric Student-t VaR and ES at `conf`; positive = loss magnitude.

    `scale_p` is the t SCALE (not the standard deviation). Sign conventions
    match var_es_normal. For X ~ t_nu(mu, s):
        VaR = -(mu + s * t_nu^{-1}(alpha))
        ES  = -(mu - s * f_nu(t_alpha)/alpha * (nu + t_alpha^2)/(nu - 1))
    (McNeil/Frey/Embrechts closed form; requires nu > 1.)
    """
    alpha = 1.0 - conf
    nu = np.clip(np.asarray(nu, dtype=float), 1.0 + 1e-6, None)
    t_a = stats.t.ppf(alpha, nu)
    var = -(mu_p + scale_p * t_a)
    es_factor = stats.t.pdf(t_a, nu) / alpha * (nu + t_a ** 2) / (nu - 1.0)
    es = -(mu_p - scale_p * es_factor)
    return var, es


def interval_coverage_t(y: np.ndarray, mu: np.ndarray, scale: np.ndarray,
                        nu: np.ndarray | float, conf: float = 0.95) -> float:
    """Fraction of realizations inside the central Student-t `conf` interval."""
    q = stats.t.ppf(0.5 + conf / 2.0, np.asarray(nu, dtype=float))
    inside = (y >= mu - q * scale) & (y <= mu + q * scale)
    return float(inside.mean())


def crps_student_t(y: np.ndarray, mu: np.ndarray, scale: np.ndarray,
                   nu: np.ndarray | float) -> float:
    """Mean CRPS for Student-t predictive distributions (closed form).

    For standardized z ~ t_nu (nu > 1), Jordan/Kruger/Lerch (2019):
        CRPS(F_nu, z) = z(2F_nu(z) - 1)
                        + 2 f_nu(z) (nu + z^2)/(nu - 1)
                        - (2 sqrt(nu)/(nu - 1)) * B(1/2, nu - 1/2)/B(1/2, nu/2)^2
    and CRPS for location-scale (mu, s) is s * CRPS(F_nu, (y-mu)/s).
    """
    scale = np.clip(scale, 1e-12, None)
    nu = np.clip(np.asarray(nu, dtype=float), 1.0 + 1e-6, None)
    z = (y - mu) / scale
    term1 = z * (2.0 * stats.t.cdf(z, nu) - 1.0)
    term2 = 2.0 * stats.t.pdf(z, nu) * (nu + z ** 2) / (nu - 1.0)
    const = (2.0 * np.sqrt(nu) / (nu - 1.0)) * (
        special.beta(0.5, nu - 0.5) / special.beta(0.5, nu / 2.0) ** 2
    )
    return float((scale * (term1 + term2 - const)).mean())


def breaches(realized_p: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Boolean breach indicator: realized loss exceeded VaR."""
    return realized_p < -var


def acerbi_szekely_z2(realized_p: np.ndarray, var: np.ndarray, es: np.ndarray,
                      conf: float = 0.95, n_sim: int = 10000,
                      nu: np.ndarray | float | None = None,
                      mu_p: np.ndarray | None = None,
                      scale_p: np.ndarray | None = None,
                      seed: int = 0) -> dict[str, float]:
    """Acerbi-Szekely (2014) Test 2 for Expected Shortfall.

        Z2 = (1/(N*alpha)) * sum_t [ X_t * 1{X_t < -VaR_t} / ES_t ] + 1

    Under a correct model E[X_t 1{breach}] = -alpha * ES_t, so E[Z2] = 0.
    Z2 < 0 means realized tail losses are LARGER than the predicted ES, i.e.
    the model understates tail risk (the direction Kupiec already flags for
    DRAM-T); Z2 > 0 means it overstates it.

    Unlike Kupiec, this scores the SEVERITY of the exceedances, not just how
    many there are, so it is the test that actually discriminates between two
    models with the same breach count.

    The null distribution has no closed form: it is obtained by simulating
    n_sim replications from the model's own predictive distribution (Student-t
    when `nu` is given, otherwise normal), recomputing Z2 on each, and taking
    the one-sided p-value P(Z2_sim <= Z2_obs). Requires the predictive moments
    (`mu_p`, `scale_p`) to simulate; without them only the statistic is
    returned.
    """
    alpha = 1.0 - conf
    n = len(realized_p)
    if n == 0:
        return {"Z2": np.nan, "pvalue": np.nan, "n": 0, "breaches": 0}

    def _z2(x: np.ndarray) -> float:
        ind = x < -var
        if not ind.any():
            # no exceedances at all -> statistic degenerates to its upper bound
            return 1.0
        return float((x * ind / np.clip(es, 1e-18, None)).sum() / (n * alpha) + 1.0)

    z2_obs = _z2(realized_p)
    out = {"Z2": z2_obs, "n": n, "breaches": int((realized_p < -var).sum())}

    if mu_p is None or scale_p is None:
        out["pvalue"] = np.nan
        return out

    rng = np.random.default_rng(seed)
    sims = np.empty(n_sim)
    for i in range(n_sim):
        if nu is None:
            x = rng.normal(mu_p, scale_p)
        else:
            nu_arr = np.broadcast_to(np.asarray(nu, dtype=float), (n,))
            x = mu_p + scale_p * rng.standard_t(nu_arr)
        sims[i] = _z2(x)
    out["pvalue"] = float((sims <= z2_obs).mean())
    return out


def kupiec_pof(breach: np.ndarray, conf: float = 0.95) -> dict[str, float]:
    """Kupiec proportion-of-failures LR test. H0: breach rate == 1-conf."""
    n = len(breach)
    x = int(breach.sum())
    p = 1.0 - conf
    pi_hat = x / n if n else 0.0
    if n == 0:
        return {"n": 0, "breaches": 0, "rate": np.nan, "LR": np.nan, "pvalue": np.nan}
    # log-likelihood ratio (guard 0*log0)
    def _ll(pi: float) -> float:
        if pi <= 0.0 or pi >= 1.0:
            return -np.inf if 0 < x < n else 0.0
        return (n - x) * np.log(1 - pi) + x * np.log(pi)
    lr = -2.0 * (_ll(p) - _ll(pi_hat))
    pval = 1.0 - stats.chi2.cdf(lr, df=1)
    return {"n": n, "breaches": x, "rate": pi_hat, "LR": float(lr), "pvalue": float(pval)}


def christoffersen_independence(breach: np.ndarray) -> dict[str, float]:
    """Christoffersen independence LR test. H0: breaches are serially independent."""
    b = breach.astype(int)
    if len(b) < 2:
        return {"LR": np.nan, "pvalue": np.nan}
    pairs = np.stack([b[:-1], b[1:]], axis=1)
    n00 = int(((pairs[:, 0] == 0) & (pairs[:, 1] == 0)).sum())
    n01 = int(((pairs[:, 0] == 0) & (pairs[:, 1] == 1)).sum())
    n10 = int(((pairs[:, 0] == 1) & (pairs[:, 1] == 0)).sum())
    n11 = int(((pairs[:, 0] == 1) & (pairs[:, 1] == 1)).sum())
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def _ll(p01: float, p11: float) -> float:
        ll = 0.0
        for n_, p_ in ((n00, 1 - p01), (n01, p01), (n10, 1 - p11), (n11, p11)):
            if n_ > 0:
                if p_ <= 0:
                    return -np.inf
                ll += n_ * np.log(p_)
        return ll

    lr = -2.0 * (_ll(pi, pi) - _ll(pi01, pi11))
    pval = 1.0 - stats.chi2.cdf(lr, df=1)
    return {"LR": float(lr), "pvalue": float(pval)}


def interval_coverage(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                      conf: float = 0.95) -> float:
    """Fraction of realizations inside the central normal `conf` interval."""
    z = stats.norm.ppf(0.5 + conf / 2.0)
    inside = (y >= mu - z * sigma) & (y <= mu + z * sigma)
    return float(inside.mean())


def crps_normal(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Mean CRPS for normal predictive distributions (closed form)."""
    sigma = np.clip(sigma, 1e-12, None)
    z = (y - mu) / sigma
    crps = sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(crps.mean())
