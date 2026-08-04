"""Seed-ensemble combination tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ensemble import _combine, _renormalise_corr, build_ensemble

RNG = np.random.default_rng(0)


def _member(mu, sigma, df=np.array([])):
    return {"mu": mu, "sigma": sigma, "df": df}


def test_ensemble_mu_is_mean():
    n, S, H = 20, 5, 3
    mus = [RNG.normal(size=(n, S, H)) for _ in range(4)]
    members = [_member(m, np.full((n, S, H), 0.02)) for m in mus]
    out = _combine(members)
    np.testing.assert_allclose(out["mu"], np.mean(mus, axis=0))


def test_ensemble_sigma_uses_law_of_total_variance():
    """sigma_ens^2 must equal mean(sigma^2) + var(mu), NOT mean(sigma).

    Averaging sigma would discard between-seed disagreement and leave the
    ensemble over-confident.
    """
    n, S, H = 50, 5, 3
    mus = [RNG.normal(size=(n, S, H)) for _ in range(5)]
    sigs = [np.full((n, S, H), s) for s in (0.01, 0.02, 0.03, 0.04, 0.05)]
    out = _combine([_member(m, s) for m, s in zip(mus, sigs)])

    want = np.mean(np.stack(sigs) ** 2, axis=0) + np.var(np.stack(mus), axis=0, ddof=1)
    np.testing.assert_allclose(out["sigma"] ** 2, want, rtol=1e-10)

    # strictly wider than the naive mean-of-sigmas whenever seeds disagree
    assert (out["sigma"] > np.mean(np.stack(sigs), axis=0)).all()


def test_ensemble_sigma_reduces_to_single_member():
    n, S, H = 10, 5, 3
    mu = RNG.normal(size=(n, S, H))
    sig = np.full((n, S, H), 0.02)
    out = _combine([_member(mu, sig)])
    np.testing.assert_allclose(out["mu"], mu)
    np.testing.assert_allclose(out["sigma"], sig)


def test_ensemble_student_t_combines_in_variance_space():
    """Under Student-t, sigma is a SCALE; the variance identity must be applied
    to scale^2 * nu/(nu-2) and converted back, not to the scale directly."""
    n, S, H = 30, 5, 3
    nu_a = np.full(H, 4.0)
    nu_b = np.full(H, 8.0)
    mu_a = np.zeros((n, S, H))
    mu_b = np.zeros((n, S, H))            # identical mu -> zero disagreement
    s_a = np.full((n, S, H), 0.02)
    s_b = np.full((n, S, H), 0.02)

    out = _combine([_member(mu_a, s_a, nu_a), _member(mu_b, s_b, nu_b)])
    nu_ens = out["df"]
    np.testing.assert_allclose(nu_ens, np.full(H, 6.0))

    f = lambda v: v / (v - 2.0)           # noqa: E731  variance factor
    want_var = 0.5 * (0.02 ** 2 * f(4.0) + 0.02 ** 2 * f(8.0))
    got_var = out["sigma"] ** 2 * f(nu_ens)[None, None, :]
    np.testing.assert_allclose(got_var, np.full((n, S, H), want_var), rtol=1e-10)


def test_renormalise_corr_gives_unit_diagonal():
    n, S = 12, 5
    a = RNG.normal(size=(n, S, S))
    psd = a @ a.transpose(0, 2, 1)
    out = _renormalise_corr(psd)
    np.testing.assert_allclose(np.diagonal(out, axis1=1, axis2=2), np.ones((n, S)), atol=1e-10)
    np.testing.assert_allclose(out, out.transpose(0, 2, 1), atol=1e-10)
    assert (np.linalg.eigvalsh(out) > -1e-8).all()


def _write_run(runs: Path, name: str, k: int, n=20, S=5, H=3, seed=0):
    rng = np.random.default_rng(seed)
    d = runs / name / f"fold{k}"
    d.mkdir(parents=True, exist_ok=True)
    anchors = np.arange(n)
    np.savez_compressed(
        d / "test_predictions.npz",
        mu=rng.normal(size=(n, S, H)), sigma=np.full((n, S, H), 0.02),
        df=np.array([]), corr=np.tile(np.eye(S), (n, 1, 1)),
        weights=rng.random((n, 3)), y_ret=rng.normal(size=(n, S, H)),
        y_vol=np.full((n, S, H), 0.02), y_corr=np.tile(np.eye(S), (n, 1, 1)),
        test_idx=anchors, anchor_dates=anchors,
    )
    return d


def test_build_ensemble_rejects_mismatched_anchors(tmp_path):
    """Members trained on different folds must not be silently averaged."""
    runs = tmp_path / "runs"
    _write_run(runs, "a", 0, seed=1)
    d = _write_run(runs, "b", 0, seed=2)
    z = dict(np.load(d / "test_predictions.npz"))
    z["anchor_dates"] = z["anchor_dates"] + 999      # different anchors
    np.savez_compressed(d / "test_predictions.npz", **z)

    with pytest.raises(ValueError, match="test anchors"):
        build_ensemble(runs, ["a", "b"], "ens", n_folds=1)


def test_build_ensemble_writes_evaluable_run(tmp_path):
    runs = tmp_path / "runs"
    for i, name in enumerate(["a", "b", "c"]):
        _write_run(runs, name, 0, seed=i)
    build_ensemble(runs, ["a", "b", "c"], "ens", n_folds=1)

    out = runs / "ens" / "fold0" / "test_predictions.npz"
    assert out.exists()
    z = np.load(out)
    # same on-disk contract as any other run, so evaluate.py needs no special case
    for key in ("mu", "sigma", "corr", "y_ret", "y_vol", "y_corr", "anchor_dates"):
        assert key in z.files
    assert z["mu"].shape == (20, 5, 3)
