"""M7 utility tests: VaR/ES, coverage tests, CRPS, bootstrap, Wilcoxon."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stats_tests import bootstrap_ci, wilcoxon_paired
from src.utils.var_es import (
    breaches,
    christoffersen_independence,
    crps_normal,
    interval_coverage,
    kupiec_pof,
    portfolio_moments,
    var_es_normal,
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
