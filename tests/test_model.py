"""M3 model tests: shapes, constraint validity, ablation flags, overfit-one-batch."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.dramt import DRAMT
from src.utils.seed import set_seed

B, T, F_NUM, F_MAC, F_SEN, N_REG, S, H = 8, 20, 63, 10, 30, 4, 5, 6


def _make_model(**kw) -> DRAMT:
    set_seed(0)
    return DRAMT(
        n_num_features=F_NUM, n_macro_features=F_MAC, n_sent_features=F_SEN,
        n_regime_features=N_REG, n_stocks=S, n_horizons=H,
        d_model=32, n_heads=4, n_layers=2, dropout=0.1, **kw,
    )


def _make_inputs(batch: int = B):
    g = torch.Generator().manual_seed(1)
    return (
        torch.randn(batch, T, F_NUM, generator=g),
        torch.randn(batch, T, F_MAC, generator=g),
        torch.randn(batch, T, F_SEN, generator=g),
        torch.randn(batch, N_REG, generator=g),
    )


def test_forward_shapes():
    model = _make_model()
    out = model(*_make_inputs())
    assert out["mu"].shape == (B, S, H)
    assert out["sigma"].shape == (B, S, H)
    assert out["corr"].shape == (B, S, S)
    assert out["modality_weights"].shape == (B, 3)


def test_sigma_positive():
    model = _make_model()
    out = model(*_make_inputs())
    assert (out["sigma"] > 0).all()


def test_corr_valid():
    model = _make_model()
    out = model(*_make_inputs())
    R = out["corr"]
    assert torch.allclose(torch.diagonal(R, dim1=1, dim2=2), torch.ones(B, S), atol=1e-4)
    assert torch.allclose(R, R.transpose(1, 2), atol=1e-6)
    assert (R >= -1.0 - 1e-5).all() and (R <= 1.0 + 1e-5).all()
    eigvals = torch.linalg.eigvalsh(R)
    assert (eigvals > -1e-5).all(), "correlation matrix must be PSD"


def test_modality_weights_softmax():
    model = _make_model()
    out = model(*_make_inputs())
    w = out["modality_weights"]
    assert torch.allclose(w.sum(dim=-1), torch.ones(B), atol=1e-5)
    assert (w > 0).all()
    # dynamic: weights must vary with the regime input
    x = _make_inputs()
    out2 = model(x[0], x[1], x[2], x[3] + 5.0)
    assert not torch.allclose(out["modality_weights"], out2["modality_weights"])


@pytest.mark.parametrize("kw,n_mod", [
    ({"use_sentiment": False}, 2),
    ({"use_macro": False}, 2),
    ({"use_sentiment": False, "use_macro": False}, 1),
    ({"dynamic_weighting": False}, 3),
])
def test_ablation_flags(kw, n_mod):
    model = _make_model(**kw)
    out = model(*_make_inputs())
    assert out["mu"].shape == (B, S, H)
    assert out["modality_weights"].shape == (B, n_mod)
    if not kw.get("dynamic_weighting", True):
        # static fusion -> equal fixed weights
        assert torch.allclose(out["modality_weights"], torch.full((B, n_mod), 1.0 / n_mod))


def test_student_t_head_shapes_and_constraints():
    model = _make_model(dist="student_t")
    out = model(*_make_inputs())
    assert "df" in out
    assert out["df"].shape == (1, 1, H)              # broadcasts over (B,S,H)
    assert (out["df"] > 2.0).all(), "nu must stay > 2 so the variance is finite"


def test_gaussian_head_has_no_df():
    out = _make_model(dist="gaussian")(*_make_inputs())
    assert "df" not in out


def test_student_t_nll_matches_scipy():
    """The hand-written likelihood must agree with scipy's t log-pdf."""
    from scipy import stats as sps

    from src.losses import student_t_nll

    y = torch.tensor([[[0.4, -1.2]]])
    mu = torch.tensor([[[0.1, 0.3]]])
    s = torch.tensor([[[0.8, 1.5]]])
    nu = torch.tensor([[[4.0, 7.0]]])
    got = student_t_nll(y, mu, s, nu).numpy().ravel()
    want = -sps.t.logpdf(y.numpy().ravel(), df=nu.numpy().ravel(),
                         loc=mu.numpy().ravel(), scale=s.numpy().ravel())
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_student_t_df_is_learnable():
    """nu must actually receive gradient - otherwise the 'learnable df' claim
    is false and the head silently stays at its initial value."""
    from src.losses import CompositeLoss

    model = _make_model(dist="student_t")
    out = model(*_make_inputs())
    y_ret = torch.randn(B, S, H) * 2.0
    y_corr = torch.eye(S).expand(B, S, S)
    CompositeLoss(0.1, 0.1, dist="student_t")(out, y_ret, y_corr)["loss"].backward()
    assert model.df_raw.grad is not None
    assert model.df_raw.grad.abs().sum() > 0


def test_student_t_nll_prefers_heavy_tails_on_heavy_tailed_data():
    """On t(3) data a t likelihood must score better (lower NLL) than a
    Gaussian one - the premise of the whole RQ2 change.

    gaussian_nll drops the 0.5*log(2*pi) constant while student_t_nll is a
    complete NLL, so the constant must be restored before comparing.
    """
    from src.losses import GAUSSIAN_NLL_CONST, gaussian_nll, student_t_nll

    torch.manual_seed(3)
    y = torch.distributions.StudentT(3.0).sample((20000,))
    mu = torch.zeros_like(y)
    s = torch.ones_like(y)
    t_nll = student_t_nll(y, mu, s, torch.full_like(y, 3.0)).mean()
    n_nll = gaussian_nll(y, mu, s).mean() + GAUSSIAN_NLL_CONST
    assert t_nll < n_nll

    # ...and the best-fit Gaussian (scale = sample std) is still worse
    n_nll_best = (gaussian_nll(y, mu, torch.full_like(y, float(y.std()))).mean()
                  + GAUSSIAN_NLL_CONST)
    assert t_nll < n_nll_best


def test_gaussian_nll_constant_matches_scipy():
    """gaussian_nll + GAUSSIAN_NLL_CONST must equal the true Gaussian NLL."""
    from scipy import stats as sps

    from src.losses import GAUSSIAN_NLL_CONST, gaussian_nll

    y = torch.tensor([0.4, -1.2])
    mu = torch.tensor([0.1, 0.3])
    s = torch.tensor([0.8, 1.5])
    got = (gaussian_nll(y, mu, s) + GAUSSIAN_NLL_CONST).numpy()
    want = -sps.norm.logpdf(y.numpy(), loc=mu.numpy(), scale=s.numpy())
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_garch_hybrid_starts_as_identity_on_garch_vol():
    """At initialisation the hybrid head must reproduce the GARCH forecast
    exactly (multiplier == 1), so any later deviation is something the network
    actually learned rather than an arbitrary starting point."""
    model = _make_model(vol_mode="garch_hybrid")
    x = _make_inputs()
    gv = torch.rand(B, S, H) * 2.0 + 0.5
    out = model(*x, gv)
    torch.testing.assert_close(out["sigma"], gv, rtol=1e-4, atol=1e-5)


def test_garch_hybrid_scales_with_garch_vol():
    """Doubling the GARCH input must double sigma - the level comes from the
    filter, not from the network."""
    model = _make_model(vol_mode="garch_hybrid")
    x = _make_inputs()
    gv = torch.rand(B, S, H) + 0.5
    s1 = model(*x, gv)["sigma"]
    s2 = model(*x, gv * 2.0)["sigma"]
    torch.testing.assert_close(s2, s1 * 2.0, rtol=1e-3, atol=1e-5)


def test_garch_hybrid_requires_garch_input():
    model = _make_model(vol_mode="garch_hybrid")
    with pytest.raises(ValueError, match="garch_vol"):
        model(*_make_inputs())


def test_learned_vol_mode_ignores_garch_input():
    """Default mode must be unaffected by the extra tensor, so the batch
    layout change cannot silently alter non-hybrid runs."""
    model = _make_model(vol_mode="learned")
    model.eval()   # otherwise dropout, not the garch input, drives the difference
    x = _make_inputs()
    with torch.no_grad():
        a = model(*x, torch.rand(B, S, H))["sigma"]
        b = model(*x, None)["sigma"]
    torch.testing.assert_close(a, b)


def test_garch_hybrid_multiplier_is_trainable():
    model = _make_model(vol_mode="garch_hybrid")
    gv = torch.rand(B, S, H) + 0.5
    out = model(*_make_inputs(), gv)
    out["sigma"].sum().backward()
    assert model.vol_head.proj.weight.grad is not None
    assert model.vol_head.proj.weight.grad.abs().sum() > 0


def test_mc_dropout_enables_only_dropout_layers():
    """LayerNorm must stay in eval mode: a blanket model.train() would change
    normalization statistics too, and the spread would no longer be a pure
    dropout posterior sample."""
    from src.train import _enable_mc_dropout

    model = _make_model()
    _enable_mc_dropout(model)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            assert m.training
        elif isinstance(m, torch.nn.LayerNorm):
            assert not m.training


def test_mc_dropout_produces_nonzero_epistemic_spread():
    """Stochastic passes must actually differ, else 'epistemic uncertainty'
    would silently be a column of zeros."""
    from src.train import _enable_mc_dropout

    model = _make_model()
    x = _make_inputs()
    _enable_mc_dropout(model)
    with torch.no_grad():
        a = model(*x)["mu"]
        b = model(*x)["mu"]
    assert not torch.allclose(a, b)
    # ...and deterministic eval mode must NOT differ
    model.eval()
    with torch.no_grad():
        c = model(*x)["mu"]
        d = model(*x)["mu"]
    torch.testing.assert_close(c, d)


def test_extended_regime_gate_accepts_8_signals():
    model = DRAMT(
        n_num_features=F_NUM, n_macro_features=F_MAC, n_sent_features=F_SEN,
        n_regime_features=8, n_stocks=S, n_horizons=H,
        d_model=32, n_heads=4, n_layers=2, dropout=0.1,
    )
    x = _make_inputs()
    out = model(x[0], x[1], x[2], torch.randn(B, 8))
    assert out["modality_weights"].shape == (B, 3)


def test_per_timestep_gate_shapes_and_variation():
    """Per-timestep gating must produce genuinely different weights at
    different steps; otherwise it is just the per-window gate with extra cost.
    """
    from src.models.dramt import GatingMLP

    gate = GatingMLP(n_regime=8, n_modalities=3, per_timestep=True, d_model=32)
    regime = torch.randn(B, 8)
    tokens = torch.randn(B, 3, T, 32)
    w = gate(regime, tokens)
    assert w.shape == (B, T, 3)
    torch.testing.assert_close(w.sum(-1), torch.ones(B, T), rtol=1e-5, atol=1e-5)
    # weights must vary across time steps
    assert w.std(dim=1).max() > 1e-4


def test_per_timestep_gate_reports_window_average_weights():
    model = _make_model(gate_per_timestep=True)
    out = model(*_make_inputs())
    w = out["modality_weights"]
    assert w.shape == (B, 3)                       # same shape as per-window mode
    torch.testing.assert_close(w.sum(-1), torch.ones(B), rtol=1e-5, atol=1e-5)


def test_per_timestep_gate_responds_to_regime():
    model = _make_model(gate_per_timestep=True)
    model.eval()
    x = _make_inputs()
    with torch.no_grad():
        a = model(*x)["modality_weights"]
        b = model(x[0], x[1], x[2], x[3] + 5.0)["modality_weights"]
    assert not torch.allclose(a, b)


def test_gradients_flow_to_all_modalities():
    model = _make_model()
    x = _make_inputs()
    inputs = [t.requires_grad_(True) for t in x]
    out = model(*inputs)
    loss = out["mu"].square().mean() + out["sigma"].mean() + out["corr"].square().mean()
    loss.backward()
    for i, t in enumerate(inputs):
        assert t.grad is not None and t.grad.abs().sum() > 0, f"no gradient into input {i}"


def test_overfit_one_batch():
    """A small DRAM-T must be able to (near-)memorize a single batch.

    Dropout is disabled: the check verifies capacity and gradient plumbing,
    and stochastic masking specifically prevents memorizing noise targets.

    Targets are in TRAINING units: raw log returns (~0.02 std) are badly
    conditioned for MSE optimization (verified empirically: mse/var plateaus
    at ~0.8), so the training pipeline scales returns by `target_scale`=100
    (percent units). This test uses the same convention.
    """
    set_seed(0)
    model = DRAMT(
        n_num_features=F_NUM, n_macro_features=F_MAC, n_sent_features=F_SEN,
        n_regime_features=N_REG, n_stocks=S, n_horizons=H,
        d_model=32, n_heads=4, n_layers=2, dropout=0.0,
    )
    x_num, x_mac, x_sen, reg = _make_inputs(16)
    g = torch.Generator().manual_seed(7)
    y_ret = torch.randn(16, S, H, generator=g) * 2.0   # percent units (0.02 * 100)
    y_vol = torch.rand(16, S, H, generator=g) * 2.0 + 0.5
    a = torch.randn(16, S, S, generator=g)
    y_corr = torch.eye(S) * 0.5 + 0.5 * torch.softmax(a @ a.transpose(1, 2), dim=-1)
    y_corr = 0.5 * (y_corr + y_corr.transpose(1, 2))

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    first = None
    for step in range(500):
        out = model(x_num, x_mac, x_sen, reg)
        nll = 0.5 * (torch.log(out["sigma"] ** 2) + (y_ret - out["mu"]) ** 2 / out["sigma"] ** 2)
        loss = ((out["mu"] - y_ret) ** 2).mean() + 0.1 * nll.mean() \
            + 0.1 * ((out["corr"] - y_corr) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    model.eval()
    with torch.no_grad():
        mse = ((model(x_num, x_mac, x_sen, reg)["mu"] - y_ret) ** 2).mean().item()
    var = y_ret.var().item()
    assert mse < 0.2 * var, f"overfit check failed: mse={mse:.6f} vs var={var:.6f}"


# --------------------------------------------------------------------------- #
# Objective variants: ranking loss, point_loss='none', HAR hybrid
# --------------------------------------------------------------------------- #

def test_ranking_loss_removes_the_shrinkage_incentive():
    """The motivation for a ranking objective.

    When predictions carry no information -- the empirical situation for daily
    returns here, where a constant-zero forecast beats the trained model
    (MAE 0.03030 vs 0.03070) -- a pointwise loss is minimised by collapsing
    toward zero. A ranking loss scores only ORDER, so scaling predictions
    barely moves it and the collapse incentive disappears.
    """
    from src.losses import pairwise_ranking_loss

    huber = torch.nn.HuberLoss(delta=1.0)
    torch.manual_seed(0)
    y = torch.randn(16, 5, 4) * 0.03            # realistic return scale
    mu = torch.randn(16, 5, 4) * 0.03           # deliberately UNINFORMATIVE

    h_full, h_zero = float(huber(mu, y)), float(huber(mu * 0.0, y))
    r_full, r_zero = float(pairwise_ranking_loss(mu, y)), float(pairwise_ranking_loss(mu * 0.0, y))

    # Huber strongly prefers predicting nothing
    assert h_zero < h_full / 2.0
    # the ranking loss is essentially indifferent to scale
    assert abs(r_zero - r_full) / r_full < 0.05


def test_ranking_loss_penalises_shrinkage_when_order_is_informative():
    """With an informative ordering, shrinking must strictly INCREASE the
    ranking loss, i.e. it actively defends the spread it has earned."""
    from src.losses import pairwise_ranking_loss

    torch.manual_seed(0)
    y = torch.randn(16, 5, 4)
    mu = y + 0.1 * torch.randn(16, 5, 4)        # correct ordering, noisy
    losses = [float(pairwise_ranking_loss(mu * s, y)) for s in (1.0, 0.5, 0.1, 0.0)]
    assert losses == sorted(losses), f"must increase monotonically as we shrink: {losses}"


def test_ranking_loss_rewards_correct_order():
    from src.losses import pairwise_ranking_loss

    torch.manual_seed(1)
    y = torch.randn(32, 5, 3)
    good = pairwise_ranking_loss(y, y)              # perfect ordering
    bad = pairwise_ranking_loss(-y, y)              # exactly inverted
    assert good < bad


def test_point_loss_none_drops_mean_term():
    from src.losses import CompositeLoss

    model = _make_model()
    out = model(*_make_inputs())
    y_ret = torch.randn(B, S, H) * 2.0
    y_corr = torch.eye(S).expand(B, S, S)
    res = CompositeLoss(0.5, 0.1, point_loss="none")(out, y_ret, y_corr)
    assert float(res["l_point"]) == 0.0
    assert float(res["l_vol"]) != 0.0               # risk terms still active


def test_har_hybrid_head_behaves_like_garch_hybrid():
    model = _make_model(vol_mode="har_hybrid")
    x = _make_inputs()
    gv = torch.rand(B, S, H) * 2.0 + 0.5
    out = model(*x, gv)
    torch.testing.assert_close(out["sigma"], gv, rtol=1e-4, atol=1e-5)


def test_unknown_vol_mode_rejected():
    with pytest.raises(ValueError, match="vol mode"):
        _make_model(vol_mode="nonsense")
