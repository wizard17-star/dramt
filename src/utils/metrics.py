"""Point-forecast metrics. All operate on numpy arrays in RAW return units.

Shapes: y_true, y_pred are (n, S, H) unless noted. Metrics are averaged over
all entries by default; use `per_horizon=True` to keep the horizon axis.
"""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-4) -> float:
    """MAPE on returns is unstable near zero; entries with |y|<eps are excluded
    (standard practice for return series, documented in results notes)."""
    mask = np.abs(y_true) >= eps
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100.0


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "DirAcc": directional_accuracy(y_true, y_pred),
    }


def vol_rmse(vol_true: np.ndarray, vol_pred: np.ndarray) -> float:
    return rmse(vol_true, vol_pred)
