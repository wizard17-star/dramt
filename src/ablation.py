"""M7 ablations: 6 configurations, all folds, best swept hyperparameters.

Configs (one table row each):
  full            - all modalities, dynamic weighting, risk-aware heads
  no_sentiment    - sentiment modality removed
  no_macro        - macro modality removed
  static_fusion   - dynamic gating replaced by fixed equal-weight fusion
  point_only      - risk heads detached from the loss (lambda1=lambda2=0)
  numerical_only  - OHLCV/indicators only (both other modalities removed)

Usage:
    python -m src.ablation [--suffix _nosent] [--configs full no_macro ...]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from src.data.splits import walk_forward_folds
from src.train import load_dataset, resolve_config, train_fold
from src.utils.config import load_config
from src.utils.seed import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ABLATIONS: dict[str, dict] = {
    "full": {},
    "no_sentiment": {"use_sentiment": False},
    "no_macro": {"use_macro": False},
    "static_fusion": {"dynamic_weighting": False},
    "point_only": {"risk_head": False},
    "numerical_only": {"use_sentiment": False, "use_macro": False},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="_nosent")
    parser.add_argument("--configs", nargs="*", default=list(ABLATIONS))
    args = parser.parse_args()

    base = load_config("config.yaml")
    device = get_device(base["device"]["prefer_cuda"])
    best = yaml.safe_load(Path("runs/sweep/best.yaml").read_text(encoding="utf-8"))

    for cfg_name in args.configs:
        flags = ABLATIONS[cfg_name]
        exp = {**best, **flags, "name": f"ablation_{cfg_name}", "dataset_suffix": args.suffix}
        ecfg = resolve_config(base, exp)
        data = load_dataset(Path(base["paths"]["processed_dir"]), ecfg.T, args.suffix)
        folds = walk_forward_folds(
            len(data["anchor_dates"]), n_folds=base["splits"]["n_folds"],
            val_frac_of_fold=base["splits"]["val_frac_of_fold"],
            purge_gap=ecfg.T + int(max(data["horizons"])),
        )
        run_dir = Path(base["paths"]["runs_dir"]) / ecfg.name
        run_dir.mkdir(parents=True, exist_ok=True)
        results = [train_fold(base, ecfg, data, fold, run_dir, device) for fold in folds]
        (run_dir / "results.json").write_text(json.dumps(results, indent=2))
        logger.info("ablation %s: %d folds done", cfg_name, len(results))


if __name__ == "__main__":
    main()
