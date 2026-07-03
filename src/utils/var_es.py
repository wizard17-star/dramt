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
from scipy import stats


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


def breaches(realized_p: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Boolean breach indicator: realized loss exceeded VaR."""
    return realized_p < -var


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
