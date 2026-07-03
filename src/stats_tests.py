"""Statistical significance utilities: bootstrap CIs and Wilcoxon signed-rank."""
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
