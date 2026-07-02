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
