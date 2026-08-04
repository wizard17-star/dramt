"""Train/evaluate baselines on the identical walk-forward folds as DRAM-T.

Usage:
    python -m src.baselines --model lstm --fold 0
    python -m src.baselines --model arima            # all folds
    python -m src.baselines --all                     # every baseline, all folds

Deep baselines:   lstm, gru, cnn_bilstm, cnn_bilstm_attn, transformer, sentiment_lstm
Econometric:      arima, garch, garch_midas, dcc_garch
(Temporal Fusion Transformer is NOT implemented: under the CPU-only compute
budget its cost is disproportionate; documented as out of scope in results.md.)

Artifacts mirror src/train.py: runs/baseline_<model>/fold<k>/{test_predictions.npz,result.json}
Econometric models store vol/corr forecasts where the model family provides them.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.splits import Fold, walk_forward_folds
from src.models.baselines.deep import make_deep_baseline
from src.models.baselines.econometric import (
    arima_forecasts,
    dcc_garch_forecasts,
    garch_forecasts,
    garch_midas_forecasts,
)
from src.train import amp_context, load_dataset, prepare_fold
from src.utils.config import load_config
from src.utils.metrics import point_metrics
from src.utils.seed import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEEP_MODELS = ["lstm", "gru", "cnn_bilstm", "cnn_bilstm_attn", "transformer", "sentiment_lstm"]
ECONOMETRIC_MODELS = ["arima", "garch", "garch_midas", "dcc_garch"]
# TFT is handled separately: pytorch-forecasting needs a long-format panel
# rather than the pre-windowed tensors the other deep baselines consume.
SPECIAL_MODELS = ["tft"]


def _save_fold_result(fold_dir: Path, fold: Fold, data: dict, mu: np.ndarray,
                      sigma: np.ndarray | None, corr: np.ndarray | None,
                      extra: dict | None = None) -> dict:
    fold_dir.mkdir(parents=True, exist_ok=True)
    y_true = data["y_ret"][fold.test_idx]
    pm = point_metrics(y_true, mu)
    result = {"fold": fold.k, "test_point_metrics": pm,
              "n_train": len(fold.train_idx), "n_test": len(fold.test_idx)}
    if extra:
        result.update(extra)
    np.savez_compressed(
        fold_dir / "test_predictions.npz",
        mu=mu,
        sigma=sigma if sigma is not None else np.array([]),
        corr=corr if corr is not None else np.array([]),
        y_ret=y_true, y_vol=data["y_vol"][fold.test_idx],
        y_corr=data["y_corr"][fold.test_idx],
        test_idx=fold.test_idx, anchor_dates=data["anchor_dates"][fold.test_idx],
    )
    (fold_dir / "result.json").write_text(json.dumps(result, indent=2))
    logger.info("fold %d: %s", fold.k, {k: round(v, 6) for k, v in pm.items()})
    return result


# --------------------------------------------------------------------------- #
# deep baselines
# --------------------------------------------------------------------------- #

def train_deep_baseline(name: str, base: dict, data: dict, fold: Fold,
                        run_dir: Path, device: torch.device) -> dict:
    t_cfg = base["training"]
    target_scale = float(t_cfg["target_scale"])
    set_seed(base["seed"] + fold.k)
    parts, _ = prepare_fold(data, fold, target_scale)
    g = torch.Generator().manual_seed(base["seed"] + fold.k)
    loaders = {
        "train": DataLoader(parts["train"], batch_size=t_cfg["batch_size"], shuffle=True, generator=g),
        "val": DataLoader(parts["val"], batch_size=256),
        "test": DataLoader(parts["test"], batch_size=256),
    }
    model = make_deep_baseline(
        name, data["X_num"].shape[-1], data["X_sent"].shape[-1],
        data["y_ret"].shape[1], data["y_ret"].shape[2],
    ).to(device)
    loss_fn = torch.nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(t_cfg["default_lr"]),
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    amp = bool(base["device"].get("amp", False))

    def fwd(batch):
        xn, xm, xs, reg, gv, y_ret, y_corr = [b.to(device) for b in batch]
        with amp_context(device, amp):
            pred = model(xn, xs) if name == "sentiment_lstm" else model(xn)
        return loss_fn(pred.float(), y_ret)

    best_val, bad, best_state = float("inf"), 0, None
    for epoch in range(t_cfg["max_epochs"]):
        t0 = time.time()
        model.train()
        for batch in loaders["train"]:
            loss = fwd(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg["grad_clip_norm"])
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val = float(np.mean([fwd(b).item() for b in loaders["val"]]))
        scheduler.step(val)
        logger.info("%s fold %d epoch %02d val %.4f (%.1fs)", name, fold.k, epoch, val, time.time() - t0)
        if val < best_val - 1e-5:
            best_val, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= t_cfg["early_stopping_patience"]:
                break

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loaders["test"]:
            xn, xm, xs, *_ = [b.to(device) for b in batch]
            with amp_context(device, amp):
                p = model(xn, xs) if name == "sentiment_lstm" else model(xn)
            preds.append(p.float().cpu().numpy())
    mu = np.concatenate(preds) / target_scale
    return _save_fold_result(run_dir / f"fold{fold.k}", fold, data, mu, None, None,
                             {"best_val_loss": best_val})


# --------------------------------------------------------------------------- #
# econometric baselines
# --------------------------------------------------------------------------- #

def run_econometric_baseline(name: str, base: dict, data: dict, fold: Fold,
                             run_dir: Path, daily: pd.DataFrame) -> dict:
    portfolio = base["portfolio"]["members"]
    horizons = [int(h) for h in data["horizons"]]
    dates = pd.to_datetime(data["anchor_dates"], unit="ns")
    anchors = pd.DatetimeIndex(dates[fold.test_idx])
    train_end = pd.Timestamp(dates[fold.train_idx][-1])
    returns_df = daily[[f"{t}_log_return" for t in portfolio]].copy()
    returns_df.columns = portfolio

    n, S, H = len(anchors), len(portfolio), len(horizons)
    if name == "dcc_garch":
        mu, vol, corr = dcc_garch_forecasts(returns_df, train_end, anchors, horizons)
        return _save_fold_result(run_dir / f"fold{fold.k}", fold, data, mu, vol, corr)

    mu = np.empty((n, S, H))
    vol = np.full((n, S, H), np.nan)
    for si, stock in enumerate(portfolio):
        r = returns_df[stock]
        if name == "arima":
            mu[:, si, :] = arima_forecasts(r, train_end, anchors, horizons)
        elif name == "garch":
            mu[:, si, :], vol[:, si, :] = garch_forecasts(r, train_end, anchors, horizons)
        elif name == "garch_midas":
            mu[:, si, :], vol[:, si, :] = garch_midas_forecasts(r, train_end, anchors, horizons)
        else:
            raise ValueError(name)
        logger.info("%s fold %d: %s done", name, fold.k, stock)
    sigma = None if np.isnan(vol).all() else vol
    return _save_fold_result(run_dir / f"fold{fold.k}", fold, data, mu, sigma, None)


# --------------------------------------------------------------------------- #
# Temporal Fusion Transformer
# --------------------------------------------------------------------------- #

def run_tft_baseline(base: dict, data: dict, fold: Fold, run_dir: Path,
                     daily: pd.DataFrame, device: torch.device, T: int) -> dict:
    from src.models.baselines.tft import tft_forecasts

    t_cfg = base["training"]
    mu = tft_forecasts(
        daily=daily,
        portfolio=base["portfolio"]["members"],
        anchor_dates=pd.to_datetime(data["anchor_dates"], unit="ns"),
        train_idx=fold.train_idx, val_idx=fold.val_idx, test_idx=fold.test_idx,
        horizons=[int(h) for h in data["horizons"]],
        T=T, device=device,
        max_epochs=t_cfg["max_epochs"], patience=t_cfg["early_stopping_patience"],
        lr=float(t_cfg["default_lr"]), dropout=base["model"]["dropout"],
        batch_size=t_cfg["batch_size"], seed=base["seed"] + fold.k,
    )
    return _save_fold_result(run_dir / f"fold{fold.k}", fold, data, mu, None, None)


# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=DEEP_MODELS + ECONOMETRIC_MODELS + SPECIAL_MODELS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--suffix", default="_nosent", help="dataset suffix ('' = with sentiment)")
    args = parser.parse_args()

    base = load_config("config.yaml")
    device = get_device(base["device"]["prefer_cuda"])
    T = base["windowing"]["default_T"]
    data = load_dataset(Path(base["paths"]["processed_dir"]), T, args.suffix)
    daily = pd.read_parquet(Path(base["paths"]["processed_dir"]) / "aligned_daily.parquet")

    folds = walk_forward_folds(
        len(data["anchor_dates"]), n_folds=base["splits"]["n_folds"],
        val_frac_of_fold=base["splits"]["val_frac_of_fold"],
        purge_gap=T + int(max(data["horizons"])),
    )
    models = DEEP_MODELS + ECONOMETRIC_MODELS + SPECIAL_MODELS if args.all else [args.model]
    assert models[0] is not None, "--model or --all required"

    for name in models:
        run_dir = Path(base["paths"]["runs_dir"]) / f"baseline_{name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for fold in folds:
            if args.fold is not None and fold.k != args.fold:
                continue
            if name in DEEP_MODELS:
                results.append(train_deep_baseline(name, base, data, fold, run_dir, device))
            elif name == "tft":
                results.append(run_tft_baseline(base, data, fold, run_dir, daily, device, T))
            else:
                results.append(run_econometric_baseline(name, base, data, fold, run_dir, daily))
        (run_dir / "results.json").write_text(json.dumps(results, indent=2))
        logger.info("%s: %d fold(s) done", name, len(results))


if __name__ == "__main__":
    main()
