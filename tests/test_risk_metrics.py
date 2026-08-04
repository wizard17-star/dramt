"""M7 utility tests: VaR/ES, coverage tests, CRPS, bootstrap, Wilcoxon."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stats_tests import bootstrap_ci, wilcoxon_paired
from src.utils.var_es import (
    acerbi_szekely_z2,
    breaches,
    christoffersen_independence,
    crps_normal,
    crps_student_t,
    interval_coverage,
    interval_coverage_t,
    kupiec_pof,
    portfolio_moments,
    student_t_std_factor,
    var_es_normal,
    var_es_student_t,
)

RNG = np.random.default_rng(0)


def test_portfolio_moments_identity_corr():
    n, S = 100, 5
    mu = np.full((n, S), 0.01)
    sigma = np.full((n, S), 0.02)
    corr = np.tile(np.eye(S), (n, 1, 1))
    w = np.full(S, 0.2)
    mu_p, sigma_p = portfolio_moments(mu, sigma, corr, w)
    assert np.allclose(mu_p, 0.01)
    # independent: sigma_p = sqrt(sum (w*sigma)^2) = sqrt(5*(0.2*0.02)^2)
    assert np.allclose(sigma_p, np.sqrt(5 * (0.2 * 0.02) ** 2))


def test_var_es_ordering():
    mu_p = np.zeros(10)
    sigma_p = np.full(10, 0.03)
    var, es = var_es_normal(mu_p, sigma_p, conf=0.95)
    assert np.allclose(var, 1.6449 * 0.03, atol=1e-3)
    assert (es > var).all()  # ES always beyond VaR


def test_kupiec_calibrated_vs_not():
    n = 2000
    good = RNG.random(n) < 0.05    # calibrated 5% breach process
    bad = RNG.random(n) < 0.15     # 3x too many breaches
    assert kupiec_pof(good, 0.95)["pvalue"] > 0.05
    assert kupiec_pof(bad, 0.95)["pvalue"] < 0.01


def test_christoffersen_detects_clustering():
    n = 2000
    indep = RNG.random(n) < 0.05
    # clustered: breaches come in runs (Markov persistence)
    clustered = np.zeros(n, dtype=bool)
    state = False
    for t in range(n):
        p = 0.6 if state else 0.02
        state = RNG.random() < p
        clustered[t] = state
    assert christoffersen_independence(indep)["pvalue"] > 0.05
    assert christoffersen_independence(clustered)["pvalue"] < 0.01


def test_breaches_and_coverage():
    n = 5000
    y = RNG.normal(0, 0.02, n)
    var, _ = var_es_normal(np.zeros(n), np.full(n, 0.02), 0.95)
    rate = breaches(y, var).mean()
    assert abs(rate - 0.05) < 0.01
    cov = interval_coverage(y, np.zeros(n), np.full(n, 0.02), 0.95)
    assert abs(cov - 0.95) < 0.01


def test_crps_closed_form_and_ordering():
    n = 20000
    y = RNG.normal(0, 1, n)
    # known value: E[CRPS] of N(0,1) forecast for N(0,1) data = 2/sqrt(pi)-...  ~ 0.5642/...
    good = crps_normal(y, np.zeros(n), np.ones(n))
    bad = crps_normal(y, np.zeros(n), np.full(n, 3.0))     # overdispersed
    biased = crps_normal(y, np.full(n, 2.0), np.ones(n))   # biased mean
    assert good < bad < biased
    # perfect deterministic-ish forecast: tiny sigma at the true mean is best possible
    exact = crps_normal(np.zeros(n), np.zeros(n), np.full(n, 1e-6))
    assert exact < 1e-4


# --------------------------------------------------------------------------- #
# Student-t risk layer
# --------------------------------------------------------------------------- #

def test_student_t_converges_to_normal_as_df_grows():
    """t_nu -> N(0,1) as nu -> inf, so every t routine must reproduce its
    Gaussian counterpart in the limit."""
    mu, s = np.zeros(1), np.ones(1)
    var_t, es_t = var_es_student_t(mu, s, 1e7, 0.95)
    var_n, es_n = var_es_normal(mu, s, 0.95)
    assert np.allclose(var_t, var_n, atol=1e-4)
    assert np.allclose(es_t, es_n, atol=1e-4)
    y = np.array([0.3])
    assert abs(crps_student_t(y, mu, s, 1e6) - crps_normal(y, mu, s)) < 1e-4


def test_student_t_var_es_match_empirical_quantiles():
    """Closed forms vs the empirical tail of a large t sample."""
    for nu in (3.0, 5.0):
        var, es = var_es_student_t(np.zeros(1), np.ones(1), nu, 0.95)
        x = RNG.standard_t(nu, 2_000_000)
        assert abs(var[0] - (-np.quantile(x, 0.05))) < 0.02
        assert abs(es[0] - (-x[x < -var[0]].mean())) < 0.05


def test_student_t_has_fatter_tails_than_normal():
    """The whole point of switching: at the same scale, t must put more mass
    beyond the 95% VaR, which is what should pull the breach rate down."""
    var_t, es_t = var_es_student_t(np.zeros(1), np.ones(1), 4.0, 0.95)
    var_n, es_n = var_es_normal(np.zeros(1), np.ones(1), 0.95)
    assert var_t[0] > var_n[0]
    assert es_t[0] > es_n[0]


def test_student_t_crps_ordering():
    nu = 5.0
    y = RNG.standard_t(nu, 20000)
    n = len(y)
    good = crps_student_t(y, np.zeros(n), np.ones(n), nu)
    over = crps_student_t(y, np.zeros(n), np.full(n, 3.0), nu)
    biased = crps_student_t(y, np.full(n, 2.0), np.ones(n), nu)
    assert good < over
    assert good < biased


def test_student_t_std_factor():
    assert np.isclose(student_t_std_factor(5.0), np.sqrt(5 / 3))
    # nu -> inf: scale and std coincide
    assert np.isclose(student_t_std_factor(1e8), 1.0, atol=1e-6)


def test_interval_coverage_t_is_calibrated():
    nu, n = 5.0, 200000
    y = RNG.standard_t(nu, n)
    cov = interval_coverage_t(y, np.zeros(n), np.ones(n), nu, 0.95)
    assert abs(cov - 0.95) < 0.01
    # scoring t data with a normal interval of the same scale under-covers
    cov_wrong = interval_coverage(y, np.zeros(n), np.ones(n), 0.95)
    assert cov_wrong < cov


# --------------------------------------------------------------------------- #
# Acerbi-Szekely ES backtest
# --------------------------------------------------------------------------- #

def test_acerbi_szekely_accepts_correct_model():
    n = 3000
    mu_p, sig_p = np.zeros(n), np.full(n, 0.02)
    var, es = var_es_normal(mu_p, sig_p, 0.95)
    realized = RNG.normal(0, 0.02, n)
    r = acerbi_szekely_z2(realized, var, es, 0.95, n_sim=500,
                          mu_p=mu_p, scale_p=sig_p)
    assert abs(r["Z2"]) < 0.25          # near zero under a correct model
    assert r["pvalue"] > 0.05           # not rejected


def test_acerbi_szekely_flags_understated_tail_risk():
    """Data is 2x more volatile than the model claims: realized exceedances
    are far worse than the predicted ES, so Z2 must go clearly negative."""
    n = 3000
    mu_p, sig_p = np.zeros(n), np.full(n, 0.02)
    var, es = var_es_normal(mu_p, sig_p, 0.95)
    realized = RNG.normal(0, 0.04, n)
    r = acerbi_szekely_z2(realized, var, es, 0.95, n_sim=500,
                          mu_p=mu_p, scale_p=sig_p)
    assert r["Z2"] < -0.5
    assert r["pvalue"] < 0.05


def test_bootstrap_ci_contains_mean():
    x = RNG.normal(5.0, 1.0, 500)
    ci = bootstrap_ci(x, np.mean, n_resamples=500)
    assert ci["lo"] < 5.0 < ci["hi"]
    assert ci["lo"] < ci["stat"] < ci["hi"]


def test_wilcoxon_detects_better_model():
    n = 300
    base_err = np.abs(RNG.normal(0, 1.0, n))
    better = base_err * 0.7 + np.abs(RNG.normal(0, 0.05, n))   # clearly smaller errors
    same = base_err + RNG.normal(0, 0.01, n)                    # noise-level difference
    r_better = wilcoxon_paired(better, base_err)
    assert r_better["pvalue"] < 0.01 and r_better["median_diff"] < 0
    r_same = wilcoxon_paired(same, base_err)
    assert r_same["pvalue"] > 0.01


# --------------------------------------------------------------------------- #
# Diebold-Mariano and multiple-comparison corrections
# --------------------------------------------------------------------------- #

def test_dm_detects_a_genuinely_better_model():
    from src.stats_tests import diebold_mariano

    n = 1000
    err_b = np.abs(RNG.normal(0, 1, n))
    err_a = err_b * 0.7                       # A clearly better
    r = diebold_mariano(err_a, err_b, horizon=10)
    assert r["DM"] < 0 and r["pvalue"] < 0.01


def test_dm_accepts_equivalent_models():
    from src.stats_tests import diebold_mariano

    n = 1000
    err_a = np.abs(RNG.normal(0, 1, n))
    err_b = np.abs(RNG.normal(0, 1, n))
    assert diebold_mariano(err_a, err_b, horizon=10)["pvalue"] > 0.05


def test_dm_is_more_conservative_than_wilcoxon_under_autocorrelation():
    """The reason DM was added.

    Build a loss differential with a small mean but strong serial correlation,
    exactly what overlapping 10-day forecasts produce. Wilcoxon treats the
    samples as independent and returns a tiny p-value; DM's HAC variance
    accounts for the dependence and is far less certain.
    """
    from src.stats_tests import diebold_mariano, wilcoxon_paired

    n = 2000
    rng = np.random.default_rng(7)
    # AR(1) with high persistence -> effective sample size far below n
    d = np.zeros(n)
    for t in range(1, n):
        d[t] = 0.95 * d[t - 1] + rng.normal(0, 0.1)
    d = d + 0.05                                   # small positive mean
    err_b = np.abs(rng.normal(0, 1, n))
    err_a = err_b + d

    w = wilcoxon_paired(err_a, err_b)
    dm = diebold_mariano(err_a, err_b, horizon=10)
    assert dm["pvalue"] > w["pvalue"], (
        f"DM ({dm['pvalue']:.3g}) must be more conservative than "
        f"Wilcoxon ({w['pvalue']:.3g}) under serial correlation")


def test_holm_bonferroni_properties():
    from src.stats_tests import holm_bonferroni

    p = [0.001, 0.01, 0.04, 0.5]
    adj = holm_bonferroni(p)
    assert all(a >= b for a, b in zip(adj, p))     # never decreases a p-value
    assert adj == sorted(adj)                       # monotone in sorted input
    assert all(a <= 1.0 for a in adj)
    # a single test is unchanged
    assert np.isclose(holm_bonferroni([0.03])[0], 0.03)


def test_benjamini_hochberg_is_less_strict_than_holm():
    from src.stats_tests import benjamini_hochberg, holm_bonferroni

    p = [0.001, 0.008, 0.02, 0.03, 0.2, 0.6]
    bh = benjamini_hochberg(p)
    holm = holm_bonferroni(p)
    assert all(b <= h + 1e-12 for b, h in zip(bh, holm))
    assert all(b >= x - 1e-12 for b, x in zip(bh, p))


def test_corrections_handle_nan_pvalues():
    from src.stats_tests import benjamini_hochberg, holm_bonferroni

    p = [0.01, np.nan, 0.3]
    for fn in (holm_bonferroni, benjamini_hochberg):
        adj = fn(p)
        assert np.isnan(adj[1])
        assert np.isfinite(adj[0]) and np.isfinite(adj[2])
