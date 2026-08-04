"""M7 ablations: 6 configurations, all folds, best swept hyperparameters.

Configs (one table row each):
  full            - all modalities, dynamic weighting, risk-aware heads
  no_sentiment    - sentiment modality removed
  no_macro        - macro modality removed
  static_fusion   - dynamic gating replaced by fixed equal-weight fusion
  point_only      - risk heads detached from the loss (lambda1=lambda2=0)
  numerical_only  - OHLCV/indicators only (both other modalities removed)

MULTIPLE SEEDS PER CONFIG (--seeds) because the single-seed ablation table is
not interpretable. Measured on the CPU-era runs, the ablation effects were:

    -sentiment      +0.00076 MAE
    -macro          +0.00035
    -static fusion  -0.00040
    numerical only  +0.00021
    -risk head      -0.00002

while the seed-to-seed spread of the SAME configuration was 0.00053. Four of
the five "effects" are smaller than the noise floor, so any claim drawn from
them is unsupported. Running each configuration over several seeds and
reporting mean +/- std is what makes the table evidence rather than anecdote.

Runs are written as runs/ablation_<config>_s<seed>; src/tables.py groups them.

Usage:
    python -m src.ablation --suffix "" --seeds 42 43 44 45 46
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
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="seeds per configuration (default: config seed only). "
                             "Several seeds are needed for the effects to be "
                             "distinguishable from seed noise.")
    args = parser.parse_args()

    base = load_config("config.yaml")
    device = get_device(base["device"]["prefer_cuda"])
    best = yaml.safe_load(Path("runs/sweep/best.yaml").read_text(encoding="utf-8"))
    best.pop("name", None)
    seeds = args.seeds or [base["seed"]]

    dataset_cache: dict[int, dict] = {}
    for cfg_name in args.configs:
        flags = ABLATIONS[cfg_name]
        for seed in seeds:
            name = (f"ablation_{cfg_name}_s{seed}" if len(seeds) > 1
                    else f"ablation_{cfg_name}")
            exp = {**best, **flags, "name": name,
                   "dataset_suffix": args.suffix, "seed": seed}
            ecfg = resolve_config(base, exp)
            run_dir = Path(base["paths"]["runs_dir"]) / ecfg.name
            if len(list(run_dir.glob("fold*/test_predictions.npz"))) >= base["splits"]["n_folds"]:
                logger.info("skip %s (already complete)", name)
                continue
            if ecfg.T not in dataset_cache:
                dataset_cache[ecfg.T] = load_dataset(
                    Path(base["paths"]["processed_dir"]), ecfg.T, args.suffix)
            data = dataset_cache[ecfg.T]
            folds = walk_forward_folds(
                len(data["anchor_dates"]), n_folds=base["splits"]["n_folds"],
                val_frac_of_fold=base["splits"]["val_frac_of_fold"],
                purge_gap=ecfg.T + int(max(data["horizons"])),
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            results = [train_fold(base, ecfg, data, fold, run_dir, device) for fold in folds]
            (run_dir / "results.json").write_text(json.dumps(results, indent=2))
            logger.info("ablation %s seed %d: %d folds done", cfg_name, seed, len(results))


if __name__ == "__main__":
    main()
