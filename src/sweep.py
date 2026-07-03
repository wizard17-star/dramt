"""Reduced hyperparameter sweep for DRAM-T (CPU budget).

Grid (from config.yaml sweep lists): d_model x lambda1 x lambda2 x lr x T.
Selection protocol: each config is trained on FOLD 0 ONLY and scored by its
best composite validation loss (covers point error + vol NLL + corr Frobenius,
i.e. both accuracy and risk calibration); the winning config is then trained
on all folds by src/train.py. Sweeping on a single fold keeps the CPU cost
tractable and never touches any test data.

Usage:
    python -m src.sweep [--suffix _nosent]
Writes runs/sweep/results.csv and runs/sweep/best.yaml.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import time
from pathlib import Path

import yaml

from src.data.splits import walk_forward_folds
from src.train import load_dataset, resolve_config, train_fold
from src.utils.config import load_config
from src.utils.seed import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="_nosent")
    parser.add_argument("--max-epochs", type=int, default=20, help="epoch cap during sweep")
    args = parser.parse_args()

    base = load_config("config.yaml")
    device = get_device(base["device"]["prefer_cuda"])
    grid = list(itertools.product(
        base["windowing"]["T_sweep"],
        base["model"]["d_model_sweep"],
        base["loss"]["lambda1_sweep"],
        base["loss"]["lambda2_sweep"],
        [float(lr) for lr in base["training"]["lr_sweep"]],
    ))
    logger.info("sweep: %d configs on fold 0 (device=%s, suffix=%r)", len(grid), device, args.suffix)

    sweep_dir = Path(base["paths"]["runs_dir"]) / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (T, d_model, lam1, lam2, lr) in enumerate(grid):
        exp = {
            "name": f"sweep/cfg{i:02d}", "T": T, "d_model": d_model,
            "lambda1": lam1, "lambda2": lam2, "lr": lr,
            "dataset_suffix": args.suffix, "max_epochs": args.max_epochs,
        }
        ecfg = resolve_config(base, exp)
        data = load_dataset(Path(base["paths"]["processed_dir"]), T, args.suffix)
        folds = walk_forward_folds(
            len(data["anchor_dates"]), n_folds=base["splits"]["n_folds"],
            val_frac_of_fold=base["splits"]["val_frac_of_fold"],
            purge_gap=T + int(max(data["horizons"])),
        )
        run_dir = Path(base["paths"]["runs_dir"]) / ecfg.name
        t0 = time.time()
        result = train_fold(base, ecfg, data, folds[0], run_dir, device)
        # Composite val losses are NOT comparable across configs with different
        # lambdas (bigger lambda => mechanically bigger total). Score every
        # config with the same FIXED reference weights on its per-component
        # validation losses at the best epoch.
        vc = result["best_val_components"]
        score = vc["l_point"] + 0.1 * vc["l_vol"] + 0.1 * vc["l_corr"]
        row = {
            "cfg": i, "T": T, "d_model": d_model, "lambda1": lam1, "lambda2": lam2,
            "lr": lr, "score": score,
            "val_point": vc["l_point"], "val_vol": vc["l_vol"], "val_corr": vc["l_corr"],
            "best_epoch": result["best_epoch"],
            "seconds": round(time.time() - t0, 1),
        }
        rows.append(row)
        logger.info("cfg %02d/%d done: score=%.4f point=%.4f (%.0fs)", i, len(grid) - 1,
                    score, vc["l_point"], row["seconds"])

    rows.sort(key=lambda r: r["score"])
    with open(sweep_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    best_yaml = {
        "name": "dramt_full", "T": best["T"], "d_model": best["d_model"],
        "lambda1": best["lambda1"], "lambda2": best["lambda2"], "lr": best["lr"],
        "dataset_suffix": args.suffix,
    }
    (sweep_dir / "best.yaml").write_text(yaml.safe_dump(best_yaml), encoding="utf-8")
    logger.info("sweep done. best: %s", json.dumps(best))


if __name__ == "__main__":
    main()
