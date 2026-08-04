"""Thesis figures (vector PDF + PNG). Entry point: python -m src.utils.plotting

Reads runs/<run>/fold*/{epochs.csv,test_predictions.npz} and writes to
results/figures/. Every figure regenerates identically when re-run on the
final artifacts.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config import load_config
from src.utils.var_es import (
    breaches,
    portfolio_moments,
    var_es_normal,
    var_es_student_t,
)

logger = logging.getLogger(__name__)

FIG_KW = {"dpi": 150, "bbox_inches": "tight"}


def _calibrated_sigma(npz_path: Path, horizons: list[int], calibration: str,
                      window: int):
    """sigma with the SAME post-hoc calibration src/evaluate.py applies.

    Without this the figures silently disagree with the tables: the plotted
    VaR line would be built from raw sigma while the reported breach rate
    comes from calibrated sigma, so a reader could count breaches in the
    figure and get a different number than the table states.
    """
    from src.evaluate import _rolling_sigma_scale, _val_sigma_scale

    z = np.load(npz_path)
    sigma = z["sigma"]
    if not sigma.size:
        return None, None
    if calibration == "rolling":
        s = _rolling_sigma_scale(npz_path, horizons, window=window)
        if s is not None:
            return sigma * s, z
    s = _val_sigma_scale(npz_path)
    return (sigma * s if s is not None else sigma), z


def _df_of(z) -> np.ndarray | None:
    return z["df"] if "df" in z.files and z["df"].size else None


def _save(fig, figures_dir: Path, name: str) -> None:
    fig.savefig(figures_dir / f"{name}.png", **FIG_KW)
    fig.savefig(figures_dir / f"{name}.pdf", **FIG_KW)
    plt.close(fig)
    logger.info("figure -> %s.{png,pdf}", figures_dir / name)


def loss_curves(run_dir: Path, figures_dir: Path, run_label: str) -> None:
    folds = sorted(run_dir.glob("fold*/epochs.csv"))
    if not folds:
        # e.g. the seed ensemble, which is assembled from member runs and was
        # never itself trained -> no epoch log. Skip rather than emit an empty
        # axes with a legend warning.
        logger.info("no epochs.csv under %s - skipping loss curves", run_dir)
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for p in folds:
        df = pd.read_csv(p)
        k = p.parent.name.replace("fold", "")
        ax.plot(df["epoch"], df["train_loss"], alpha=0.7, label=f"fold {k} train")
        ax.plot(df["epoch"], df["val_loss"], alpha=0.7, linestyle="--", label=f"fold {k} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Composite loss")
    ax.set_title(f"Training / validation loss ({run_label})")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, figures_dir, "loss_curves")


def predicted_vs_actual(run_dir: Path, figures_dir: Path, horizons: list[int],
                        stocks: list[str], h: int = 1, stock_idx: int = 0,
                        n_days: int = 120) -> None:
    p = sorted(run_dir.glob("fold*/test_predictions.npz"))[-1]   # latest fold
    z = np.load(p)
    hi = horizons.index(h)
    dates = pd.to_datetime(z["anchor_dates"], unit="ns")[-n_days:]
    y = z["y_ret"][-n_days:, stock_idx, hi]
    mu = z["mu"][-n_days:, stock_idx, hi]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, y, label="actual", lw=1.2)
    ax.plot(dates, mu, label="predicted", lw=1.2)
    if z["sigma"].size:
        s = z["sigma"][-n_days:, stock_idx, hi]
        ax.fill_between(dates, mu - 1.96 * s, mu + 1.96 * s, alpha=0.2, label="95% interval")
    ax.set_title(f"{stocks[stock_idx]} {h}-day log return: predicted vs actual (last test fold)")
    ax.set_ylabel("log return")
    ax.legend()
    _save(fig, figures_dir, "predicted_vs_actual")


def var_backtest_plot(run_dir: Path, figures_dir: Path, horizons: list[int],
                      weights: np.ndarray, conf: float = 0.95,
                      calibration: str = "rolling", window: int = 120) -> None:
    h10 = horizons.index(10)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    n_breach = n_total = 0
    for p in sorted(run_dir.glob("fold*/test_predictions.npz")):
        sigma, z = _calibrated_sigma(p, horizons, calibration, window)
        if sigma is None or not z["corr"].size:
            continue
        dates = pd.to_datetime(z["anchor_dates"], unit="ns")
        mu_h = z["mu"][:, :, h10]
        # DRAM-T sigma is trained against cumulative h-day returns -> already
        # in 10-day units at h10 (this plot is only produced for DRAM-T runs)
        sig_h = sigma[:, :, h10]
        mu_p, sigma_p = portfolio_moments(mu_h, sig_h, z["corr"], weights)
        df = _df_of(z)
        # must match evaluate.py: a Student-t run has to use the t quantile,
        # otherwise the plotted VaR line is not the one that was backtested
        if df is not None:
            var10, _ = var_es_student_t(mu_p, sigma_p, float(df[h10]), conf)
        else:
            var10, _ = var_es_normal(mu_p, sigma_p, conf)
        realized = z["y_ret"][:, :, h10] @ weights
        br = breaches(realized, var10)
        n_breach += int(br.sum())
        n_total += len(br)
        ax.plot(dates, realized, color="steelblue", lw=0.8)
        ax.plot(dates, -var10, color="firebrick", lw=1.0)
        if br.any():
            ax.scatter(dates[br], realized[br], color="red", zorder=5, s=18)
    ax.plot([], [], color="steelblue", label="realized 10-day portfolio return")
    ax.plot([], [], color="firebrick", label=f"-VaR {int(conf*100)}%")
    ax.scatter([], [], color="red", label="breach")
    rate = 100 * n_breach / n_total if n_total else float("nan")
    ax.set_title(f"Portfolio 10-day VaR backtest, all test folds "
                 f"({calibration} calibration; {n_breach}/{n_total} breaches "
                 f"= {rate:.1f}%, nominal 5%)")
    ax.legend()
    _save(fig, figures_dir, "var_backtest")


def reliability_plot(run_dir: Path, figures_dir: Path, horizons: list[int],
                     calibration: str = "rolling", window: int = 120) -> None:
    """Empirical vs nominal central-interval coverage (PIT-based reliability).

    The PIT must use the model's OWN predictive distribution and the SAME
    calibrated sigma as the tables. Pushing Student-t residuals through a
    normal CDF would manufacture an apparent miscalibration that the model
    does not have.
    """
    qs = np.linspace(0.05, 0.95, 19)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    pits = []
    for p in sorted(run_dir.glob("fold*/test_predictions.npz")):
        sigma, z = _calibrated_sigma(p, horizons, calibration, window)
        if sigma is None:
            continue
        resid = (z["y_ret"] - z["mu"]) / np.clip(sigma, 1e-12, None)
        df = _df_of(z)
        pit = (stats.t.cdf(resid, df[None, None, :]) if df is not None
               else stats.norm.cdf(resid))
        pits.append(pit.ravel())
    if not pits:
        return
    pit = np.concatenate(pits)
    emp = [np.mean((pit > 0.5 - q / 2) & (pit < 0.5 + q / 2)) for q in qs]
    ax.plot(qs, emp, marker="o", ms=3, label="model")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    ax.set_xlabel("Nominal central coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Calibration (reliability) of predictive intervals")
    ax.legend()
    _save(fig, figures_dir, "reliability")


def modality_weight_heatmap(run_dir: Path, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ws, dates_all = [], []
    for p in sorted(run_dir.glob("fold*/test_predictions.npz")):
        z = np.load(p)
        if "weights" not in z or z["weights"].ndim != 2 or z["weights"].shape[1] < 2:
            continue
        ws.append(z["weights"])
        dates_all.append(pd.to_datetime(z["anchor_dates"], unit="ns"))
    if not ws:
        return
    W = np.concatenate(ws)             # (n, M)
    dates = dates_all[0].append(dates_all[1:]) if len(dates_all) > 1 else dates_all[0]
    im = ax.imshow(W.T, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                   extent=[0, len(W), -0.5, W.shape[1] - 0.5])
    labels = ["numerical", "macro", "sentiment"][: W.shape[1]]
    ax.set_yticks(range(W.shape[1]), labels)
    step = max(len(dates) // 8, 1)
    ax.set_xticks(range(0, len(dates), step),
                  [d.strftime("%Y-%m") for d in dates[::step]], rotation=30)
    ax.set_title("Dynamic modality weights over test periods")
    fig.colorbar(im, ax=ax, label="weight")
    _save(fig, figures_dir, "modality_weights")


def fold_boxplots(results_dir: Path, figures_dir: Path) -> None:
    """Per-fold MAE and DirAcc box plots across models (from eval_*.json)."""
    import json
    import re

    rows = []
    for p in sorted(results_dir.glob("eval_*.json")):
        run = p.stem.replace("eval_", "")
        # The chain writes eval_<run>.json plus eval_<run>_global.json and
        # eval_<run>_rolling.json. Those are the SAME point predictions scored
        # under different sigma calibrations, so including all three would put
        # every model in the plot three times with identical MAE boxes.
        if run.endswith(("_global", "_rolling")):
            continue
        # 10 individual seed runs would swamp the axis; the ensemble stands in
        if re.match(r"^dramt_seed\d+$", run):
            continue
        for k, fold in enumerate(json.loads(p.read_text())):
            rows.append({"run": run, "fold": k, "MAE": fold["point"]["MAE"],
                         "DirAcc": fold["point"]["DirAcc"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    for metric in ["MAE", "DirAcc"]:
        order = df.groupby("run")[metric].mean().sort_values().index
        fig, ax = plt.subplots(figsize=(10, 4.5))
        data = [df[df["run"] == r][metric].to_numpy() for r in order]
        ax.boxplot(data, tick_labels=[r.replace("baseline_", "") for r in order])
        ax.set_ylabel(metric)
        ax.set_title(f"Per-fold {metric} across models")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        _save(fig, figures_dir, f"fold_box_{metric.lower()}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="dramt_full")
    parser.add_argument("--calibration", choices=["global", "rolling"], default=None,
                        help="must match the calibration used for the tables")
    args = parser.parse_args()

    base = load_config("config.yaml")
    run_dir = Path(base["paths"]["runs_dir"]) / args.run
    figures_dir = Path(base["paths"]["figures_dir"])
    results_dir = Path(base["paths"]["results_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    horizons = [int(h) for h in base["windowing"]["horizons"]]
    weights = np.array(base["portfolio"]["weights"], dtype=float)
    stocks = base["portfolio"]["members"]
    risk_cfg = base.get("risk", {})
    calibration = args.calibration or risk_cfg.get("sigma_calibration", "global")
    window = int(risk_cfg.get("calibration_window", 120))
    logger.info("figures for %s using %s calibration", args.run, calibration)

    loss_curves(run_dir, figures_dir, args.run)
    predicted_vs_actual(run_dir, figures_dir, horizons, stocks)
    var_backtest_plot(run_dir, figures_dir, horizons, weights,
                      calibration=calibration, window=window)
    reliability_plot(run_dir, figures_dir, horizons, calibration, window)
    modality_weight_heatmap(run_dir, figures_dir)
    fold_boxplots(results_dir, figures_dir)
    logger.info("all figures written to %s", figures_dir)


if __name__ == "__main__":
    main()
