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
