"""DRAM-T training: fold preparation, composite-loss training loop, checkpointing.

Usage:
    python -m src.train --config experiments/full.yaml --fold 0
    python -m src.train --config experiments/full.yaml            # all folds
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.data.splits import Fold, Standardizer, walk_forward_folds
from src.losses import CompositeLoss
from src.models.dramt import DRAMT
from src.utils.config import load_config
from src.utils.metrics import point_metrics
from src.utils.seed import get_device, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Resolved experiment settings (config.yaml defaults + experiment overrides)."""
    name: str
    T: int
    d_model: int
    n_layers: int
    n_heads: int
    dropout: float
    ffn_mult: int
    lr: float
    batch_size: int
    max_epochs: int
    patience: int
    grad_clip: float
    lambda1: float
    lambda2: float
    weight_decay: float
    target_scale: float
    use_sentiment: bool
    use_macro: bool
    dynamic_weighting: bool
    risk_head: bool
    dataset_suffix: str          # "" or "_nosent"
    seed: int


def resolve_config(base: dict, exp: dict) -> ExperimentConfig:
    m, t, l, w = base["model"], base["training"], base["loss"], base["windowing"]
    return ExperimentConfig(
        name=exp.get("name", "dramt"),
        T=exp.get("T", w["default_T"]),
        d_model=exp.get("d_model", m["default_d_model"]),
        n_layers=exp.get("n_layers", m["default_n_layers"]),
        n_heads=exp.get("n_heads", m["n_heads"]),
        dropout=exp.get("dropout", m["dropout"]),
        ffn_mult=exp.get("ffn_mult", m["ffn_mult"]),
        lr=float(exp.get("lr", t["default_lr"])),
        batch_size=exp.get("batch_size", t["batch_size"]),
        max_epochs=exp.get("max_epochs", t["max_epochs"]),
        patience=exp.get("patience", t["early_stopping_patience"]),
        grad_clip=t["grad_clip_norm"],
        lambda1=float(exp.get("lambda1", 0.1)),
        lambda2=float(exp.get("lambda2", 0.1)),
        weight_decay=float(base["loss"]["lambda3_weight_decay"]),
        target_scale=float(t["target_scale"]),
        use_sentiment=exp.get("use_sentiment", True),
        use_macro=exp.get("use_macro", True),
        dynamic_weighting=exp.get("dynamic_weighting", True),
        risk_head=exp.get("risk_head", True),
        dataset_suffix=exp.get("dataset_suffix", ""),
        seed=int(exp.get("seed", base["seed"])),
    )


def load_dataset(processed_dir: Path, T: int, suffix: str = "") -> dict[str, np.ndarray]:
    path = processed_dir / f"features_T{T}{suffix}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run `python -m src.data.build_features` first"
        )
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def regime_features(X_num: np.ndarray, X_macro: np.ndarray, X_sent: np.ndarray,
                    num_cols: list[str], macro_cols: list[str], sent_cols: list[str]) -> np.ndarray:
    """Market-state signal for the gating MLP, from the window's LAST timestep
    (standardized units): recent realized vol, recent return sign, macro level,
    news presence. Uses only in-window information -> no leakage."""
    def col_idx(cols: list[str], key: str) -> list[int]:
        return [i for i, c in enumerate(cols) if key in c]

    last_num = X_num[:, -1, :]
    last_mac = X_macro[:, -1, :]
    last_sen = X_sent[:, -1, :]
    vol_i = col_idx(num_cols, "roll_vol")
    ret_i = col_idx(num_cols, "log_return")
    news_i = col_idx(sent_cols, "has_news")
    reg = np.stack([
        last_num[:, vol_i].mean(axis=1),
        np.sign(last_num[:, ret_i]).mean(axis=1),
        last_mac.mean(axis=1),
        last_sen[:, news_i].mean(axis=1) if news_i else np.zeros(len(last_num)),
    ], axis=1)
    return reg.astype(np.float32)


def prepare_fold(data: dict[str, np.ndarray], fold: Fold, target_scale: float):
    """Standardize on train only; scale targets; build regime features."""
    num_cols = [str(c) for c in data["num_cols"]]
    macro_cols = [str(c) for c in data["macro_cols"]]
    sent_cols = [str(c) for c in data["sent_cols"]]

    scalers = {k: Standardizer().fit(data[k][fold.train_idx]) for k in ("X_num", "X_macro", "X_sent")}
    parts = {}
    for split, idx in (("train", fold.train_idx), ("val", fold.val_idx), ("test", fold.test_idx)):
        Xn = scalers["X_num"].transform(data["X_num"][idx])
        Xm = scalers["X_macro"].transform(data["X_macro"][idx])
        Xs = scalers["X_sent"].transform(data["X_sent"][idx])
        reg = regime_features(Xn, Xm, Xs, num_cols, macro_cols, sent_cols)
        parts[split] = TensorDataset(
            torch.from_numpy(Xn), torch.from_numpy(Xm), torch.from_numpy(Xs),
            torch.from_numpy(reg),
            torch.from_numpy(data["y_ret"][idx] * target_scale),
            torch.from_numpy(data["y_corr"][idx]),
        )
    return parts, scalers


def amp_context(device: torch.device, enabled: bool):
    """bfloat16 autocast over the FORWARD pass only (CUDA). bf16 has the same
    exponent range as fp32, so no GradScaler is needed and the composite loss
    (log sigma^2, Gaussian NLL, Cholesky) stays numerically safe once outputs
    are cast back to fp32 before the loss."""
    use = bool(enabled) and device.type == "cuda"
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=use)


def run_epoch(model, loader, loss_fn, device, optimizer=None, grad_clip=1.0, amp=False):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "l_point": 0.0, "l_vol": 0.0, "l_corr": 0.0}
    n = 0
    with torch.set_grad_enabled(training):
        for xn, xm, xs, reg, y_ret, y_corr in loader:
            xn, xm, xs, reg = xn.to(device), xm.to(device), xs.to(device), reg.to(device)
            y_ret, y_corr = y_ret.to(device), y_corr.to(device)
            with amp_context(device, amp):
                out = model(xn, xm, xs, reg)
            out = {k: v.float() for k, v in out.items()}   # loss always in fp32
            losses = loss_fn(out, y_ret, y_corr)
            if training:
                optimizer.zero_grad()
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            bs = xn.shape[0]
            n += bs
            for k in totals:
                totals[k] += losses[k].item() * bs
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def predict(model, loader, device, target_scale: float, amp: bool = False):
    """Returns raw-unit predictions: mu, sigma (n,S,H); corr (n,S,S); weights (n,M)."""
    model.eval()
    mus, sigmas, corrs, weights = [], [], [], []
    for xn, xm, xs, reg, *_ in loader:
        with amp_context(device, amp):
            out = model(xn.to(device), xm.to(device), xs.to(device), reg.to(device))
        out = {k: v.float() for k, v in out.items()}
        mus.append(out["mu"].cpu().numpy())
        sigmas.append(out["sigma"].cpu().numpy())
        corrs.append(out["corr"].cpu().numpy())
        weights.append(out["modality_weights"].cpu().numpy())
    return {
        "mu": np.concatenate(mus) / target_scale,
        "sigma": np.concatenate(sigmas) / target_scale,
        "corr": np.concatenate(corrs),
        "weights": np.concatenate(weights),
    }


def train_fold(base_cfg: dict, ecfg: ExperimentConfig, data: dict[str, np.ndarray],
               fold: Fold, run_dir: Path, device: torch.device) -> dict:
    amp = bool(base_cfg["device"].get("amp", False))
    set_seed(ecfg.seed + fold.k)
    parts, _ = prepare_fold(data, fold, ecfg.target_scale)
    g = torch.Generator().manual_seed(ecfg.seed + fold.k)
    loaders = {
        "train": DataLoader(parts["train"], batch_size=ecfg.batch_size, shuffle=True, generator=g),
        "val": DataLoader(parts["val"], batch_size=256),
        "test": DataLoader(parts["test"], batch_size=256),
    }

    model = DRAMT(
        n_num_features=data["X_num"].shape[-1],
        n_macro_features=data["X_macro"].shape[-1],
        n_sent_features=data["X_sent"].shape[-1],
        n_regime_features=4,
        n_stocks=data["y_ret"].shape[1],
        n_horizons=data["y_ret"].shape[2],
        d_model=ecfg.d_model, n_heads=ecfg.n_heads, n_layers=ecfg.n_layers,
        ffn_mult=ecfg.ffn_mult, dropout=ecfg.dropout,
        use_sentiment=ecfg.use_sentiment, use_macro=ecfg.use_macro,
        dynamic_weighting=ecfg.dynamic_weighting, risk_head=ecfg.risk_head,
    ).to(device)
    loss_fn = CompositeLoss(ecfg.lambda1, ecfg.lambda2, risk_head=ecfg.risk_head)
    optimizer = torch.optim.Adam(model.parameters(), lr=ecfg.lr, weight_decay=ecfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    fold_dir = run_dir / f"fold{fold.k}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    log_path = fold_dir / "epochs.csv"
    best_val, best_epoch, bad_epochs = float("inf"), -1, 0
    best_val_components: dict[str, float] = {}

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_point", "train_vol", "train_corr",
                         "val_loss", "val_point", "lr", "seconds"])
        for epoch in range(ecfg.max_epochs):
            t0 = time.time()
            tr = run_epoch(model, loaders["train"], loss_fn, device, optimizer,
                           ecfg.grad_clip, amp=amp)
            va = run_epoch(model, loaders["val"], loss_fn, device, amp=amp)
            scheduler.step(va["loss"])
            dt = time.time() - t0
            writer.writerow([epoch, f"{tr['loss']:.6f}", f"{tr['l_point']:.6f}",
                             f"{tr['l_vol']:.6f}", f"{tr['l_corr']:.6f}",
                             f"{va['loss']:.6f}", f"{va['l_point']:.6f}",
                             optimizer.param_groups[0]["lr"], f"{dt:.1f}"])
            f.flush()
            logger.info("fold %d epoch %02d train %.4f val %.4f (%.1fs)",
                        fold.k, epoch, tr["loss"], va["loss"], dt)
            if va["loss"] < best_val - 1e-5:
                best_val, best_epoch, bad_epochs = va["loss"], epoch, 0
                best_val_components = dict(va)
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "val_loss": best_val, "config": ecfg.__dict__},
                           fold_dir / "best.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= ecfg.patience:
                    logger.info("fold %d: early stop at epoch %d (best %d)", fold.k, epoch, best_epoch)
                    break

    # restore best & evaluate on test (plus val predictions for post-hoc
    # sigma calibration downstream — validation only, never test)
    ckpt = torch.load(fold_dir / "best.pt", weights_only=False)
    model.load_state_dict(ckpt["model"])
    val_preds = predict(model, loaders["val"], device, ecfg.target_scale, amp=amp)
    np.savez_compressed(fold_dir / "val_predictions.npz",
                        mu=val_preds["mu"], sigma=val_preds["sigma"],
                        y_ret=data["y_ret"][fold.val_idx])
    preds = predict(model, loaders["test"], device, ecfg.target_scale, amp=amp)
    y_true = data["y_ret"][fold.test_idx]
    pm = point_metrics(y_true, preds["mu"])
    result = {
        "fold": fold.k, "best_epoch": best_epoch, "best_val_loss": best_val,
        "best_val_components": best_val_components,
        "test_point_metrics": pm,
        "n_train": len(fold.train_idx), "n_val": len(fold.val_idx), "n_test": len(fold.test_idx),
    }
    np.savez_compressed(fold_dir / "test_predictions.npz",
                        mu=preds["mu"], sigma=preds["sigma"], corr=preds["corr"],
                        weights=preds["weights"], y_ret=y_true,
                        y_vol=data["y_vol"][fold.test_idx],
                        y_corr=data["y_corr"][fold.test_idx],
                        test_idx=fold.test_idx,
                        anchor_dates=data["anchor_dates"][fold.test_idx])
    (fold_dir / "result.json").write_text(json.dumps(result, indent=2))
    logger.info("fold %d test: %s", fold.k, {k: round(v, 6) for k, v in pm.items()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/full.yaml")
    parser.add_argument("--fold", type=int, default=None, help="train a single fold")
    args = parser.parse_args()

    base = load_config("config.yaml")
    with open(args.config, encoding="utf-8") as f:
        exp = yaml.safe_load(f) or {}
    ecfg = resolve_config(base, exp)

    device = get_device(base["device"]["prefer_cuda"])
    logger.info("experiment %r on %s | T=%d d=%d L=%d lam1=%.2f lam2=%.2f suffix=%r",
                ecfg.name, device, ecfg.T, ecfg.d_model, ecfg.n_layers,
                ecfg.lambda1, ecfg.lambda2, ecfg.dataset_suffix)

    data = load_dataset(Path(base["paths"]["processed_dir"]), ecfg.T, ecfg.dataset_suffix)
    n = len(data["anchor_dates"])
    folds = walk_forward_folds(
        n, n_folds=base["splits"]["n_folds"],
        val_frac_of_fold=base["splits"]["val_frac_of_fold"],
        purge_gap=ecfg.T + int(max(data["horizons"])),
    )
    run_dir = Path(base["paths"]["runs_dir"]) / ecfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment.json").write_text(json.dumps(ecfg.__dict__, indent=2))

    results = []
    for fold in folds:
        if args.fold is not None and fold.k != args.fold:
            continue
        results.append(train_fold(base, ecfg, data, fold, run_dir, device))
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    logger.info("done: %d fold(s) -> %s", len(results), run_dir)


if __name__ == "__main__":
    main()
