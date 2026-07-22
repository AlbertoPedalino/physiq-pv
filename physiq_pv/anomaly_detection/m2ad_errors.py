"""Residual/discrepancy transforms used by M2AD."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


_EPS = 1e-12


def _ewma(values: np.ndarray, com: Optional[float]) -> np.ndarray:
    if com is None:
        return np.asarray(values, dtype=np.float64)
    if com < 0:
        raise ValueError("ewma_com must be >= 0 or None")
    return pd.DataFrame(values).ewm(com=com, adjust=True).mean().to_numpy(dtype=np.float64)


def _area_average(values: np.ndarray, half_window: int) -> np.ndarray:
    if half_window < 1:
        raise ValueError("area_half_window must be >= 1")
    width = 2 * half_window + 1

    def average_integral(window: np.ndarray) -> float:
        if window.size == 1:
            return float(window[0])
        return float(np.trapezoid(window) / (window.size - 1))

    result = np.empty_like(values, dtype=np.float64)
    for sensor in range(values.shape[1]):
        result[:, sensor] = (
            pd.Series(values[:, sensor])
            .rolling(width, center=True, min_periods=1)
            .apply(average_integral, raw=True)
            .to_numpy(dtype=np.float64)
        )
    return result


def compute_discrepancy(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    error: str = "area",
    area_half_window: int = 2,
    ewma_com: Optional[float] = 10.0,
) -> np.ndarray:
    """Compute point or signed-area discrepancy for one contiguous segment."""
    y = np.asarray(observed, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    if y.shape != pred.shape or y.ndim != 2:
        raise ValueError(
            "observed/predicted must share shape (time, sensors), "
            f"got {y.shape}/{pred.shape}"
        )
    if error == "point":
        raw = np.abs(y - pred)
    elif error == "area":
        raw = _area_average(y, area_half_window) - _area_average(pred, area_half_window)
    else:
        raise ValueError("error must be either 'point' or 'area'")
    return _ewma(raw, ewma_com)


class M2ADDiscrepancy:
    """Apply discrepancies per segment and own the train-fitted area scaling."""

    def __init__(
        self,
        *,
        error: str = "area",
        area_half_window: int = 2,
        ewma_com: Optional[float] = 10.0,
    ) -> None:
        if error not in {"point", "area"}:
            raise ValueError("error must be either 'point' or 'area'")
        self.error = error
        self.area_half_window = int(area_half_window)
        self.ewma_com = ewma_com
        self.is_fitted = False

    def _raw_by_segment(
        self, observed: np.ndarray, predicted: np.ndarray, segment_ids: np.ndarray
    ) -> np.ndarray:
        observed = np.asarray(observed, dtype=np.float64)
        predicted = np.asarray(predicted, dtype=np.float64)
        segment_ids = np.asarray(segment_ids)
        if observed.shape != predicted.shape or observed.ndim != 2:
            raise ValueError("observed and predicted must share shape (time, sensors)")
        if segment_ids.ndim != 1 or len(segment_ids) != len(observed):
            raise ValueError("segment_ids must contain one id per observation")
        errors = np.empty_like(observed, dtype=np.float64)
        for segment_id in np.unique(segment_ids):
            mask = segment_ids == segment_id
            errors[mask] = compute_discrepancy(
                observed[mask],
                predicted[mask],
                error=self.error,
                area_half_window=self.area_half_window,
                ewma_com=self.ewma_com,
            )
        return errors

    def fit_transform(
        self, observed: np.ndarray, predicted: np.ndarray, segment_ids: np.ndarray
    ) -> np.ndarray:
        errors = self._raw_by_segment(observed, predicted, segment_ids)
        if self.error == "area":
            self.area_mean = float(np.mean(errors))
            self.area_scale = float(np.std(errors))
            if self.area_scale < _EPS:
                self.area_scale = 1.0
            errors = (errors - self.area_mean) / self.area_scale
        self.is_fitted = True
        return errors

    def transform(
        self, observed: np.ndarray, predicted: np.ndarray, segment_ids: np.ndarray
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("call fit_transform before transform")
        errors = self._raw_by_segment(observed, predicted, segment_ids)
        if self.error == "area":
            errors = (errors - self.area_mean) / self.area_scale
        return errors
