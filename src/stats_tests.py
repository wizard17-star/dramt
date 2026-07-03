"""Statistical significance: bootstrap CIs and Wilcoxon signed-rank tests.

Run as a module to produce the significance tables (M7 deliverable):
    python -m src.stats_tests [--reference dramt_full]
Writes results/stats_wilcoxon.csv/.tex and results/stats_bootstrap.csv.
Pairing: per-sample absolute errors over the SAME test anchors (all folds
concatenated); models with different anchor sets are skipped with a warning.
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
        wilcoxon_rows.append({
            "model_A": args.reference, "model_B": d.name,
            "median_diff_A_minus_B": w["median_diff"],
            "pvalue": w["pvalue"],
            "A_better": bool(w["median_diff"] < 0),
            "significant_5pct": bool(w["pvalue"] < 0.05) if not np.isnan(w["pvalue"]) else False,
        })

    dfw = pd.DataFrame(wilcoxon_rows)
    dfb = pd.DataFrame(boot_rows)
    dfw.to_csv(results_dir / "stats_wilcoxon.csv", index=False)
    dfb.to_csv(results_dir / "stats_bootstrap.csv", index=False)
    if len(dfw):
        tex = dfw.to_latex(index=False, float_format="%.4g",
                           caption="Two-sided Wilcoxon signed-rank tests on paired per-sample "
                                   "absolute errors (reference vs each model).",
                           label="tab:wilcoxon")
        (results_dir / "stats_wilcoxon.tex").write_text(tex, encoding="utf-8")
    logger.info("wrote stats tables (%d wilcoxon rows, %d bootstrap rows)", len(dfw), len(dfb))


if __name__ == "__main__":
    main()
