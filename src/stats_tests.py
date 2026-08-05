"""Statistical significance: bootstrap CIs, Wilcoxon, and Diebold-Mariano.

Run as a module to produce the significance tables (M7 deliverable):
    python -m src.stats_tests [--reference dramt_ensemble]
Writes results/stats_wilcoxon.csv/.tex and results/stats_bootstrap.csv.
Pairing: per-sample absolute errors over the SAME test anchors (all folds
concatenated); models with different anchor sets are skipped with a warning.

Two corrections to the original M7 protocol, both of which make the reported
p-values LARGER (more conservative):

1. Serial correlation. The Wilcoxon signed-rank test assumes independent
   pairs. These are OVERLAPPING multi-horizon forecasts -- an anchor's 10-day
   return shares 9 of its 10 days with the next anchor's -- so consecutive
   loss differentials are strongly autocorrelated and Wilcoxon p-values are
   anti-conservative (too many "significant" results). The Diebold-Mariano
   test with a Newey-West HAC variance is the standard remedy and is reported
   alongside, with the Harvey-Leybourne-Newbold small-sample adjustment.

2. Multiple comparisons. The reference model is tested against every other
   run; at ~34 comparisons and alpha=0.05 roughly two false positives are
   expected by chance. Holm-Bonferroni (family-wise) and Benjamini-Hochberg
   (false-discovery-rate) adjusted p-values are added.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(
    values: np.ndarray,
    stat_fn=np.mean,
    n_resamples: int = 1000,
    conf: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile bootstrap CI for a statistic of per-sample values."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values).ravel()
    n = len(values)
    boot = np.array([stat_fn(values[rng.integers(0, n, n)]) for _ in range(n_resamples)])
    lo, hi = np.percentile(boot, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100])
    return {"stat": float(stat_fn(values)), "lo": float(lo), "hi": float(hi)}


def wilcoxon_paired(err_a: np.ndarray, err_b: np.ndarray) -> dict[str, float]:
    """Two-sided Wilcoxon signed-rank on paired per-sample errors (A vs B).

    Inputs are per-sample loss values (e.g. |error|); a small p-value with
    median(err_a - err_b) < 0 means model A significantly better.
    """
    a, b = np.asarray(err_a).ravel(), np.asarray(err_b).ravel()
    assert a.shape == b.shape, "paired test requires identical sample sets"
    diff = a - b
    nz = diff[diff != 0]
    if len(nz) < 10:
        return {"statistic": np.nan, "pvalue": np.nan, "median_diff": float(np.median(diff))}
    res = stats.wilcoxon(nz, alternative="two-sided")
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue),
            "median_diff": float(np.median(diff))}


def diebold_mariano(err_a: np.ndarray, err_b: np.ndarray,
                    horizon: int = 10, hln: bool = True) -> dict[str, float]:
    """Diebold-Mariano test on a loss differential, robust to serial correlation.

    d_t = loss_A(t) - loss_B(t);  H0: E[d] = 0.

        DM = mean(d) / sqrt(LRV(d) / n)

    where LRV is the Newey-West long-run variance with Bartlett weights and a
    lag truncation of `horizon - 1`. That truncation is the standard choice
    for h-step-ahead forecasts: an h-step forecast error follows an MA(h-1)
    process, so autocovariances up to lag h-1 must be included. Ignoring them
    (as a Wilcoxon test implicitly does) understates the variance and inflates
    significance.

    With `hln`, the Harvey-Leybourne-Newbold small-sample correction is
    applied and the statistic is referred to a t distribution with n-1 degrees
    of freedom rather than the standard normal.

    A negative statistic with a small p-value means model A has significantly
    LOWER loss than B.
    """
    a, b = np.asarray(err_a).ravel(), np.asarray(err_b).ravel()
    assert a.shape == b.shape, "paired test requires identical sample sets"
    d = a - b
    n = len(d)
    if n < 20 or np.allclose(d, 0):
        return {"DM": np.nan, "pvalue": np.nan, "mean_diff": float(np.mean(d))}

    d_bar = float(np.mean(d))
    dc = d - d_bar
    lag = max(int(horizon) - 1, 0)
    gamma0 = float(np.dot(dc, dc) / n)
    lrv = gamma0
    for k in range(1, lag + 1):
        if k >= n:
            break
        gk = float(np.dot(dc[k:], dc[:-k]) / n)
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * gk        # Bartlett weight
    if lrv <= 0:                                          # can happen numerically
        lrv = gamma0 if gamma0 > 0 else np.nan
    if not np.isfinite(lrv) or lrv <= 0:
        return {"DM": np.nan, "pvalue": np.nan, "mean_diff": d_bar}

    dm = d_bar / np.sqrt(lrv / n)
    h = int(horizon)
    if hln:
        adj = (n + 1 - 2 * h + h * (h - 1) / n) / n
        dm = dm * np.sqrt(max(adj, 1e-12))
        pval = 2.0 * (1.0 - stats.t.cdf(abs(dm), df=n - 1))
    else:
        pval = 2.0 * (1.0 - stats.norm.cdf(abs(dm)))
    return {"DM": float(dm), "pvalue": float(pval), "mean_diff": d_bar}


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (family-wise error rate)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    out = np.full(n, np.nan)
    finite = np.where(np.isfinite(p))[0]
    if len(finite) == 0:
        return out.tolist()
    order = finite[np.argsort(p[finite])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)                       # enforce monotonicity
        out[idx] = min(running, 1.0)
    return out.tolist()


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (false discovery rate)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    out = np.full(n, np.nan)
    finite = np.where(np.isfinite(p))[0]
    if len(finite) == 0:
        return out.tolist()
    order = finite[np.argsort(p[finite])]
    m = len(order)
    running = 1.0
    for rank in range(m - 1, -1, -1):                     # step-up
        idx = order[rank]
        val = m / (rank + 1) * p[idx]
        running = min(running, val)
        out[idx] = min(running, 1.0)
    return out.tolist()


def _block_bootstrap_indices(n: int, n_boot: int, block: int, seed: int) -> np.ndarray:
    """(n_boot, n) circular block-bootstrap index matrix.

    Blocks, not i.i.d. resampling: the loss series of overlapping h-day
    forecasts is strongly autocorrelated, and an i.i.d. bootstrap would
    destroy that dependence and understate the variance of the mean loss -
    exactly the error the Diebold-Mariano HAC variance exists to avoid.
    """
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n   # circular
    return idx.reshape(n_boot, -1)[:, :n]


def model_confidence_set(losses: dict[str, np.ndarray], alpha: float = 0.10,
                         n_boot: int = 1000, block: int = 10,
                         seed: int = 0) -> dict:
    """Hansen-Lunde-Nason (2011) Model Confidence Set, T_max variant.

    Answers the question the pairwise tests cannot: *which* models are, as a
    group, statistically indistinguishable from the best? A table of
    "nothing is significant" says only that no single comparison survives
    correction; the MCS turns that into a positive statement - a set that
    contains the best model with probability 1-alpha.

    Procedure: with M the surviving set, let d_i be model i's mean loss minus
    the average mean loss over M, and t_i = d_i / se*(d_i) with se* from a
    block bootstrap. The equal-predictive-ability null is tested with
    T_max = max_i t_i against its bootstrap distribution; if rejected, the
    single worst model (argmax t_i) is eliminated and the test repeats. The
    procedure stops at the first non-rejection, and everything still standing
    is the MCS.

    Returns the surviving set, the elimination order, and each model's MCS
    p-value (the level at which it would drop out).
    """
    names = list(losses)
    L = np.stack([np.asarray(losses[k], dtype=float).ravel() for k in names])  # (m, n)
    m, n = L.shape
    if m < 2:
        return {"mcs": names, "eliminated": [], "pvalues": {names[0]: 1.0} if names else {}}

    idx = _block_bootstrap_indices(n, n_boot, block, seed)
    # bootstrap mean loss per (replicate, model); computed once and reused
    boot_means = np.empty((n_boot, m))
    for b in range(n_boot):
        boot_means[b] = L[:, idx[b]].mean(axis=1)
    mean_loss = L.mean(axis=1)

    alive = list(range(m))
    eliminated: list[tuple[str, float]] = []
    pvals: dict[str, float] = {}
    running_p = 0.0

    while len(alive) > 1:
        ml = mean_loss[alive]
        bm = boot_means[:, alive]
        d = ml - ml.mean()                              # (k,)
        d_boot = bm - bm.mean(axis=1, keepdims=True)    # (n_boot, k)
        var = ((d_boot - d) ** 2).mean(axis=0)
        se = np.sqrt(np.clip(var, 1e-30, None))
        t = d / se
        t_boot = (d_boot - d) / se

        T_obs = float(t.max())
        T_boot = t_boot.max(axis=1)
        p = float((T_boot > T_obs).mean())
        # MCS p-values are monotone by construction: a model cannot be
        # eliminated at a level below one at which an earlier model went
        running_p = max(running_p, p)

        if p >= alpha:
            break
        worst_local = int(np.argmax(t))
        worst = alive[worst_local]
        pvals[names[worst]] = running_p
        eliminated.append((names[worst], running_p))
        alive.pop(worst_local)

    for i in alive:
        pvals[names[i]] = max(running_p, alpha) if eliminated else 1.0
    return {
        "mcs": [names[i] for i in alive],
        "eliminated": eliminated,
        "pvalues": pvals,
        "alpha": alpha,
        "n_boot": n_boot,
        "block": block,
    }


# --------------------------------------------------------------------------- #
# module entry point: significance tables over saved runs
# --------------------------------------------------------------------------- #

def _load_abs_errors(run_dir) -> tuple[np.ndarray, np.ndarray] | None:
    """Concatenate per-sample |error| (mean over stocks & horizons) across folds."""
    from pathlib import Path
    paths = sorted(Path(run_dir).glob("fold*/test_predictions.npz"))
    if not paths:
        return None
    errs, anchors = [], []
    for p in paths:
        z = np.load(p)
        errs.append(np.abs(z["y_ret"] - z["mu"]).mean(axis=(1, 2)))
        anchors.append(z["anchor_dates"])
    return np.concatenate(errs), np.concatenate(anchors)


def _load_corr_errors(run_dir) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-anchor mean |correlation error| over the strict upper triangle.

    Same per-anchor loss shape as _load_abs_errors, so the correlation
    forecasts can go through exactly the same Diebold-Mariano, Holm and Model
    Confidence Set machinery as the point forecasts. Only runs carrying a
    learned correlation matrix are returned; a run without one has nothing of
    its own to test.
    """
    from pathlib import Path
    paths = sorted(Path(run_dir).glob("fold*/test_predictions.npz"))
    if not paths:
        return None
    errs, anchors = [], []
    for p in paths:
        z = np.load(p)
        if "corr" not in z.files or not z["corr"].size or not z["y_corr"].size:
            return None
        S = z["y_corr"].shape[-1]
        iu = np.triu_indices(S, k=1)
        d = z["corr"][:, iu[0], iu[1]] - z["y_corr"][:, iu[0], iu[1]]
        errs.append(np.abs(d).mean(axis=1))
        anchors.append(z["anchor_dates"])
    return np.concatenate(errs), np.concatenate(anchors)


def main() -> None:
    import argparse
    import logging
    from pathlib import Path

    import pandas as pd

    from src.utils.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="dramt_full")
    parser.add_argument("--mcs-alpha", type=float, default=0.10,
                        help="Model Confidence Set confidence level (1-alpha)")
    parser.add_argument("--mcs-boot", type=int, default=1000)
    args = parser.parse_args()

    base = load_config("config.yaml")
    # h-step forecast errors follow an MA(h-1) process; the DM long-run
    # variance must cover autocovariances out to that lag
    h_max = max(int(h) for h in base["windowing"]["horizons"])
    runs_dir = Path(base["paths"]["runs_dir"])
    results_dir = Path(base["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    ref = _load_abs_errors(runs_dir / args.reference)
    assert ref is not None, f"reference run {args.reference} not found"
    ref_err, ref_anchor = ref

    all_losses: dict[str, np.ndarray] = {}
    wilcoxon_rows, boot_rows = [], []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or not list(d.glob("fold*/test_predictions.npz")):
            continue
        loaded = _load_abs_errors(d)
        if loaded is None:
            continue
        err, anchor = loaded
        ci = bootstrap_ci(err, np.mean, n_resamples=1000)
        boot_rows.append({"run": d.name, "mean_abs_err": ci["stat"],
                          "ci_lo": ci["lo"], "ci_hi": ci["hi"], "n": len(err)})
        if d.name == args.reference:
            continue
        if len(err) != len(ref_err) or not np.array_equal(anchor, ref_anchor):
            logger.warning("skip %s: anchor set differs from reference", d.name)
            continue
        all_losses[d.name] = err
        w = wilcoxon_paired(ref_err, err)
        dm = diebold_mariano(ref_err, err, horizon=h_max)
        wilcoxon_rows.append({
            "model_A": args.reference, "model_B": d.name,
            "median_diff_A_minus_B": w["median_diff"],
            "mean_diff_A_minus_B": dm["mean_diff"],
            "wilcoxon_p": w["pvalue"],
            "dm_stat": dm["DM"],
            "dm_p": dm["pvalue"],
            "A_better": bool(w["median_diff"] < 0),
        })

    # the reference is `continue`d out of the pairwise loop before the losses
    # are collected, but it must be a candidate in its own confidence set
    all_losses[args.reference] = ref_err

    # ---- Model Confidence Set ------------------------------------------
    mcs = model_confidence_set(all_losses, alpha=args.mcs_alpha,
                               n_boot=args.mcs_boot, block=h_max,
                               seed=base["seed"])
    mcs_rows = [{"run": r,
                 "in_mcs": r in set(mcs["mcs"]),
                 "mcs_pvalue": mcs["pvalues"].get(r, np.nan),
                 "mean_abs_err": float(np.mean(all_losses[r]))}
                for r in sorted(all_losses)]
    pd.DataFrame(mcs_rows).sort_values("mean_abs_err").to_csv(
        results_dir / "stats_mcs.csv", index=False)
    logger.info("Model Confidence Set (alpha=%.2f, block bootstrap len=%d): "
                "%d of %d models survive; reference %s in MCS: %s",
                args.mcs_alpha, h_max, len(mcs["mcs"]), len(all_losses),
                args.reference, args.reference in set(mcs["mcs"]))
    if mcs["eliminated"]:
        logger.info("  eliminated (worst first): %s",
                    [n for n, _ in mcs["eliminated"]][:12])

    # ---- correlation forecasts: same tests, separate family --------------
    # The correlation head is a distinct claim from the mean forecast and
    # deserves the same treatment rather than an eyeballed table of RMSEs.
    corr_losses: dict[str, np.ndarray] = {}
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or d.name == "sweep":
            continue
        loaded = _load_corr_errors(d)
        if loaded is None:
            continue
        err, anchor = loaded
        if len(err) == len(ref_err) and np.array_equal(anchor, ref_anchor):
            corr_losses[d.name] = err
    if len(corr_losses) >= 2:
        ref_corr_name = (args.reference if args.reference in corr_losses
                         else min(corr_losses, key=lambda k: corr_losses[k].mean()))
        ref_c = corr_losses[ref_corr_name]
        rows = []
        for name, err in corr_losses.items():
            if name == ref_corr_name:
                continue
            dm = diebold_mariano(ref_c, err, horizon=h_max)
            rows.append({"model_A": ref_corr_name, "model_B": name,
                         "mean_diff_A_minus_B": dm["mean_diff"],
                         "dm_stat": dm["DM"], "dm_p": dm["pvalue"]})
        dfc = pd.DataFrame(rows)
        dfc["dm_p_holm"] = holm_bonferroni(dfc["dm_p"].tolist())
        dfc.to_csv(results_dir / "stats_corr_wilcoxon.csv", index=False)

        mcs_c = model_confidence_set(corr_losses, alpha=args.mcs_alpha,
                                     n_boot=args.mcs_boot, block=h_max,
                                     seed=base["seed"])
        pd.DataFrame([{"run": r, "in_mcs": r in set(mcs_c["mcs"]),
                       "mcs_pvalue": mcs_c["pvalues"].get(r, np.nan),
                       "mean_corr_err": float(np.mean(corr_losses[r]))}
                      for r in sorted(corr_losses)]).sort_values(
            "mean_corr_err").to_csv(results_dir / "stats_corr_mcs.csv", index=False)
        logger.info("correlation forecasts: reference %s | DM+Holm significant "
                    "%d/%d | MCS retains %d of %d at alpha=%.2f",
                    ref_corr_name, int((dfc["dm_p_holm"] < 0.05).sum()), len(dfc),
                    len(mcs_c["mcs"]), len(corr_losses), args.mcs_alpha)

    dfw = pd.DataFrame(wilcoxon_rows)
    if len(dfw):
        # Multiple-comparison correction across the whole family of tests.
        # Reported for the Diebold-Mariano p-values, which are the ones that
        # account for the overlapping-forecast autocorrelation.
        dfw["dm_p_holm"] = holm_bonferroni(dfw["dm_p"].tolist())
        dfw["dm_p_bh"] = benjamini_hochberg(dfw["dm_p"].tolist())
        dfw["wilcoxon_p_holm"] = holm_bonferroni(dfw["wilcoxon_p"].tolist())
        dfw["significant_5pct_dm_holm"] = dfw["dm_p_holm"] < 0.05
        dfw["significant_5pct_wilcoxon_raw"] = dfw["wilcoxon_p"] < 0.05
        n_raw = int(dfw["significant_5pct_wilcoxon_raw"].sum())
        n_adj = int(dfw["significant_5pct_dm_holm"].sum())
        logger.info(
            "significant at 5%%: %d/%d by raw Wilcoxon, %d/%d by "
            "Diebold-Mariano + Holm (autocorrelation- and multiplicity-adjusted)",
            n_raw, len(dfw), n_adj, len(dfw))
    dfb = pd.DataFrame(boot_rows)
    dfw.to_csv(results_dir / "stats_wilcoxon.csv", index=False)
    dfb.to_csv(results_dir / "stats_bootstrap.csv", index=False)
    if len(dfw):
        tex_cols = ["model_B", "mean_diff_A_minus_B", "wilcoxon_p", "dm_stat",
                    "dm_p", "dm_p_holm"]
        tex = dfw[tex_cols].to_latex(
            index=False, float_format="%.4g",
            caption=(f"Forecast-accuracy comparisons against {args.reference}. "
                     "Wilcoxon assumes independent pairs; because these are "
                     "overlapping multi-horizon forecasts, the "
                     "Diebold-Mariano statistic with a Newey-West HAC variance "
                     f"(lag {h_max - 1}) is the appropriate test, and "
                     "\\texttt{dm\\_p\\_holm} additionally corrects for the "
                     "number of comparisons. A negative statistic favours the "
                     "reference model."),
            label="tab:wilcoxon")
        (results_dir / "stats_wilcoxon.tex").write_text(tex, encoding="utf-8")
    logger.info("wrote stats tables (%d wilcoxon rows, %d bootstrap rows)", len(dfw), len(dfb))


if __name__ == "__main__":
    main()
