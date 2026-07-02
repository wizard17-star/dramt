"""M0 smoke test: env sanity, config loads, seeding is deterministic, device resolves."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.seed import get_device, set_seed

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_config_loads():
    cfg = load_config(CONFIG_PATH)
    assert cfg["seed"] == 42
    assert cfg["universe"]["tickers"] == ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA"]
    assert len(cfg["portfolio"]["members"]) == 5
    assert abs(sum(cfg["portfolio"]["weights"]) - 1.0) < 1e-9


def test_seed_determinism():
    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.allclose(a, b)


def test_device_resolves():
    cfg = load_config(CONFIG_PATH)
    device = get_device(cfg["device"]["prefer_cuda"])
    assert device.type in ("cuda", "cpu")


def test_torch_forward_smoke():
    set_seed(0)
    x = torch.randn(4, 10, 8)
    linear = torch.nn.Linear(8, 16)
    y = linear(x)
    assert y.shape == (4, 10, 16)
