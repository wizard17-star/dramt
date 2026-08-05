"""Full-capacity hyperparameter sweep for DRAM-T (GPU budget).

Grid (from config.yaml sweep lists):
    T x d_model x n_layers x lambda1 x lambda2 x lr

Selection protocol
------------------
Each config is trained and scored by its best VALIDATION loss; no test data is
ever touched. Two selection modes:

  --folds 0      score on fold 0 only (the CPU-era protocol, kept for
                 reproducibility of the earlier results)
  --folds all    train every config on all K folds and score by the MEAN
                 validation score across folds (more robust config choice:
                 it cannot latch onto a single fold's regime)

Because composite validation losses are NOT comparable across configs with
different lambdas (a bigger lambda mechanically inflates the total), every
config is scored with the SAME fixed reference weights applied to its
per-component validation losses at its best epoch:

    score = l_point + 0.1 * l_vol + 0.1 * l_corr        (--select-by composite)

Measured caveat on that default: across this project's 1,920 trials, 101.6% of
the score variance comes from l_point, the objective the study finds to be
unlearnable (l_vol 0.3%, l_corr 1.1%), and fold 0 supplies 46% of the
between-config variance because its loss scale is twice the others. --select-by
risk|balanced standardise each component within fold and reweight toward the
learnable objectives. Empirically the resulting config performs inside the seed
range of the composite choice, which is consistent with selection being
noise-driven either way.

Cost control
------------
--stage1-fold-0 --topk K runs the full grid on fold 0, then promotes only the
K best configs to the remaining folds and re-scores them on the mean across
all folds. This is a successive-halving style budget saver; it still selects
on all-fold validation performance, just without paying for the whole grid on
every fold. Use --folds all for the exhaustive version.

The run is RESUMABLE: completed (config, fold) pairs are appended to
runs/sweep/trials.csv and skipped on restart.

Usage:
    python -m src.sweep --folds all
    python -m src.sweep --stage1-fold-0 --topk 12
Writes runs/sweep/trials.csv, runs/sweep/results.csv and runs/sweep/best.yaml.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from src.data.splits import walk_forward_folds
from src.train import load_dataset, resolve_config, train_fold
from src.utils.config import load_config
from src.utils.seed import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRIAL_FIELDS = [
    "cfg", "fold", "T", "d_model", "n_layers", "lambda1", "lambda2", "lr",
    "score", "val_point", "val_vol", "val_corr", "best_epoch", "seconds",
]


def build_grid(base: dict) -> list[dict]:
    combos = itertools.product(
        base["windowing"]["T_sweep"],
        base["model"]["d_model_sweep"],
        base["model"]["n_layers_sweep"],
        base["loss"]["lambda1_sweep"],
        base["loss"]["lambda2_sweep"],
        [float(lr) for lr in base["training"]["lr_sweep"]],
    )
    return [
        {"cfg": i, "T": T, "d_model": d, "n_layers": L,
         "lambda1": l1, "lambda2": l2, "lr": lr}
        for i, (T, d, L, l1, l2, lr) in enumerate(combos)
    ]


def trial_files(sweep_dir: Path) -> list[Path]:
    """All trial logs: the single-worker file plus any per-shard files."""
    return sorted(set(list(sweep_dir.glob("trials.csv")) + list(sweep_dir.glob("trials_shard*.csv"))))


def read_trials(sweep_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for p in trial_files(sweep_dir):
        with open(p, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def load_done(sweep_dir: Path) -> set[tuple[int, int]]:
    """Completed (cfg, fold) pairs across every shard, so a long sweep can
    resume after a crash and shards never duplicate each other's work."""
    return {(int(r["cfg"]), int(r["fold"])) for r in read_trials(sweep_dir)}


def append_trial(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row[k] for k in TRIAL_FIELDS})


def run_trial(base: dict, spec: dict, fold_k: int, suffix: str,
              max_epochs: int | None, device, dataset_cache: dict) -> dict:
    exp = {
        "name": f"sweep/cfg{spec['cfg']:03d}", "T": spec["T"], "d_model": spec["d_model"],
        "n_layers": spec["n_layers"], "lambda1": spec["lambda1"],
        "lambda2": spec["lambda2"], "lr": spec["lr"], "dataset_suffix": suffix,
    }
    if max_epochs is not None:
        exp["max_epochs"] = max_epochs
    ecfg = resolve_config(base, exp)

    T = spec["T"]
    if T not in dataset_cache:
        dataset_cache[T] = load_dataset(Path(base["paths"]["processed_dir"]), T, suffix)
    data = dataset_cache[T]
    folds = walk_forward_folds(
        len(data["anchor_dates"]), n_folds=base["splits"]["n_folds"],
        val_frac_of_fold=base["splits"]["val_frac_of_fold"],
        purge_gap=T + int(max(data["horizons"])),
    )
    run_dir = Path(base["paths"]["runs_dir"]) / ecfg.name
    t0 = time.time()
    result = train_fold(base, ecfg, data, folds[fold_k], run_dir, device)
    vc = result["best_val_components"]
    return {
        **spec, "fold": fold_k,
        "score": vc["l_point"] + 0.1 * vc["l_vol"] + 0.1 * vc["l_corr"],
        "val_point": vc["l_point"], "val_vol": vc["l_vol"], "val_corr": vc["l_corr"],
        "best_epoch": result["best_epoch"], "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="", help="dataset suffix ('' = with sentiment)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="epoch cap during sweep (default: config max_epochs)")
    parser.add_argument("--folds", default="all", help="'all' or a fold index, e.g. '0'")
    parser.add_argument("--stage1-fold-0", action="store_true",
                        help="full grid on fold 0, then promote --topk configs to all folds")
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None,
                        help="debug: only run the first N configs")
    parser.add_argument("--shard", type=int, default=None,
                        help="this worker's index (0-based); requires --nshards")
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--no-aggregate", action="store_true",
                        help="skip writing results.csv/best.yaml (for shard workers)")
    parser.add_argument("--select-by", default="composite",
                        choices=["composite", "risk", "balanced"],
                        help="which validation criterion ranks configurations")
    args = parser.parse_args()

    base = load_config("config.yaml")
    device = get_device(base["device"]["prefer_cuda"])
    n_folds = base["splits"]["n_folds"]
    grid = build_grid(base)
    if args.limit:
        grid = grid[: args.limit]

    sweep_dir = Path(base["paths"]["runs_dir"]) / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    trials_path = (sweep_dir / f"trials_shard{args.shard}.csv" if args.shard is not None
                   else sweep_dir / "trials.csv")
    done = load_done(sweep_dir)
    cache: dict[int, dict] = {}

    if args.stage1_fold_0:
        plan = [(s, 0) for s in grid]
        logger.info("sweep stage 1: %d configs on fold 0 (device=%s)", len(grid), device)
    elif args.folds == "all":
        plan = [(s, k) for s in grid for k in range(n_folds)]
        logger.info("sweep: %d configs x %d folds = %d trials (device=%s)",
                    len(grid), n_folds, len(plan), device)
    else:
        k = int(args.folds)
        plan = [(s, k) for s in grid]
        logger.info("sweep: %d configs on fold %d (device=%s)", len(grid), k, device)

    if args.shard is not None:
        # Deterministic disjoint split so parallel workers never duplicate work.
        #
        # Shard by CONFIG, not by position in the (config, fold) plan. Folds
        # differ a lot in cost -- the walk-forward window expands, so fold 3
        # trains on roughly twice the data of fold 0 -- and round-robin over
        # plan positions with nshards == n_folds degenerates into "one fold per
        # shard" (measured: 8.3h vs 15.5h ETA for shards 0 and 3). Giving each
        # shard every fold of a disjoint config subset balances by construction.
        plan = [(s, k) for s, k in plan if s["cfg"] % args.nshards == args.shard]
        logger.info("shard %d/%d -> %d trials", args.shard, args.nshards, len(plan))

    def execute(plan_items) -> None:
        todo = [(s, k) for s, k in plan_items if (s["cfg"], k) not in done]
        logger.info("%d trials to run (%d already done, resuming)",
                    len(todo), len(plan_items) - len(todo))
        t_start = time.time()
        for i, (spec, k) in enumerate(todo):
            row = run_trial(base, spec, k, args.suffix, args.max_epochs, device, cache)
            append_trial(trials_path, row)
            done.add((spec["cfg"], k))
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(todo) - i - 1)
            logger.info(
                "trial %d/%d cfg%03d fold%d (T=%d d=%d L=%d lr=%.0e) score=%.4f "
                "%.0fs | elapsed %.1fh eta %.1fh",
                i + 1, len(todo), spec["cfg"], k, spec["T"], spec["d_model"],
                spec["n_layers"], spec["lr"], row["score"], row["seconds"],
                elapsed / 3600, eta / 3600,
            )

    execute(plan)

    if args.stage1_fold_0:
        assert args.shard is None, "--stage1-fold-0 does not support sharding"
        fold0 = [r for r in read_trials(sweep_dir) if int(r["fold"]) == 0]
        fold0.sort(key=lambda r: float(r["score"]))
        promoted = {int(r["cfg"]) for r in fold0[: args.topk]}
        logger.info("stage 2: promoting %d configs to folds 1..%d: %s",
                    len(promoted), n_folds - 1, sorted(promoted))
        execute([(s, k) for s in grid if s["cfg"] in promoted
                 for k in range(1, n_folds)])

    if args.no_aggregate:
        logger.info("shard %s finished; skipping aggregation", args.shard)
        return

    # ---- aggregate: mean validation score across the folds each config ran on
    rows = read_trials(sweep_dir)
    by_cfg: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_cfg[int(r["cfg"])].append(r)

    spec_by_cfg = {s["cfg"]: s for s in grid}
    agg = []
    for cfg, rs in by_cfg.items():
        if cfg not in spec_by_cfg:
            continue
        scores = [float(r["score"]) for r in rs]
        agg.append({
            **spec_by_cfg[cfg],
            "n_folds_scored": len(rs),
            "score_mean": sum(scores) / len(scores),
            "score_max": max(scores),
            "val_point_mean": sum(float(r["val_point"]) for r in rs) / len(rs),
            "val_vol_mean": sum(float(r["val_vol"]) for r in rs) / len(rs),
            "val_corr_mean": sum(float(r["val_corr"]) for r in rs) / len(rs),
            "seconds_total": round(sum(float(r["seconds"]) for r in rs), 1),
        })

    # ---- selection criterion -------------------------------------------
    # Measured on this project's own 1,920 trials, the default "composite"
    # criterion (l_point + 0.1*l_vol + 0.1*l_corr, averaged raw over folds)
    # has two properties worth knowing:
    #   * 101.6% of its variance across configurations comes from l_point,
    #     the objective this study finds to be unlearnable; l_vol contributes
    #     0.3% and l_corr 1.1%. Configurations are ranked almost entirely on
    #     noise, and the sweep duly selected the LOWEST correlation weight for
    #     the head that turns out to work best.
    #   * fold 0's loss scale is roughly twice the others', so a raw mean
    #     lets that single (anomalous, low-volatility) fold supply 46% of the
    #     between-configuration variance.
    # "risk" and "balanced" standardise each component within fold before
    # averaging, which removes the scale imbalance, and reweight toward the
    # learnable objectives. Empirically the resulting configuration performs
    # inside the seed range of the composite choice - consistent with
    # selection being noise-driven either way - but the criterion is exposed
    # so the choice is explicit rather than accidental.
    max_folds = max(a["n_folds_scored"] for a in agg)
    pool = [a for a in agg if a["n_folds_scored"] == max_folds]

    if args.select_by == "composite":
        ranked = sorted(pool, key=lambda a: a["score_mean"])
    else:
        by_fold: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            by_fold[int(r["fold"])].append(r)
        stats: dict[int, dict[str, tuple[float, float]]] = {}
        for k, rs in by_fold.items():
            stats[k] = {}
            for comp in ("val_point", "val_vol", "val_corr"):
                v = np.array([float(r[comp]) for r in rs])
                stats[k][comp] = (float(v.mean()), float(v.std()) or 1.0)

        weights = ({"val_vol": 1.0, "val_corr": 1.0} if args.select_by == "risk"
                   else {"val_point": 1.0, "val_vol": 1.0, "val_corr": 1.0})
        for a in pool:
            zs = []
            for r in by_cfg[a["cfg"]]:
                k = int(r["fold"])
                zs.append(sum(
                    w * (float(r[c]) - stats[k][c][0]) / stats[k][c][1]
                    for c, w in weights.items()))
            a["score_selected"] = float(np.mean(zs))
        ranked = sorted(pool, key=lambda a: a["score_selected"])
    logger.info("selection criterion: %s", args.select_by)
    with open(sweep_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(sorted(agg, key=lambda a: (-a["n_folds_scored"], a["score_mean"])))

    best = ranked[0]
    best_yaml = {
        "name": "dramt_full", "T": int(best["T"]), "d_model": int(best["d_model"]),
        "n_layers": int(best["n_layers"]), "lambda1": float(best["lambda1"]),
        "lambda2": float(best["lambda2"]), "lr": float(best["lr"]),
        "dataset_suffix": args.suffix,
    }
    (sweep_dir / "best.yaml").write_text(yaml.safe_dump(best_yaml), encoding="utf-8")
    logger.info("sweep done. best (over %d folds): %s", max_folds, json.dumps(best))


if __name__ == "__main__":
    main()
