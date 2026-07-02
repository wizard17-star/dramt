"""Expanding walk-forward splits with purge gap, and train-only standardization.

Scheme (n_folds=K over n samples ordered in time):
- The last `eval_frac` of samples is divided into K equal contiguous blocks.
- Fold k: block k is split into a validation half then a test half
  (val first, test after -> test is always strictly later than val).
- Train = all samples ending `purge_gap` BEFORE the block start (expanding
  window: later folds have strictly more training data).
- `purge_gap` >= T + h_max samples removes label/window overlap between the
  end of train and the start of val (no shared future days -> no leakage).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Fold:
    k: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def walk_forward_folds(
    n_samples: int,
    n_folds: int = 4,
    eval_frac: float = 0.4,
    val_frac_of_fold: float = 0.5,
    purge_gap: int = 30,
) -> list[Fold]:
    eval_len = int(n_samples * eval_frac)
    block_len = eval_len // n_folds
    eval_start = n_samples - block_len * n_folds
    folds: list[Fold] = []
    for k in range(n_folds):
        block_start = eval_start + k * block_len
        val_len = int(block_len * val_frac_of_fold)
        val_idx = np.arange(block_start, block_start + val_len)
        test_idx = np.arange(block_start + val_len, block_start + block_len)
        train_end = max(0, block_start - purge_gap)
        train_idx = np.arange(0, train_end)
        assert len(train_idx) > 0, f"fold {k}: empty train set"
        folds.append(Fold(k=k, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx))
    return folds


class Standardizer:
    """Per-feature z-score fit on TRAIN ONLY; applied to all splits."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        """X: (n, T, F) — statistics over samples*time per feature."""
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = np.nanmean(flat, axis=0)
        self.std_ = np.nanstd(flat, axis=0)
        self.std_ = np.where(self.std_ < 1e-12, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "fit before transform"
        return ((X - self.mean_) / self.std_).astype(np.float32)
