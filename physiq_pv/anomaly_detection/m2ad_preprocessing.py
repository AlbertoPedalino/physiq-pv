"""Leakage-safe interpolation and min-max scaling for M2AD."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def ensure_segments(
    segments: Sequence[np.ndarray], n_sensors: Optional[int] = None
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for index, segment in enumerate(segments):
        arr = np.asarray(segment, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                f"segment {index} must have shape (time, sensors), got {arr.shape}"
            )
        if n_sensors is not None and arr.shape[1] != n_sensors:
            raise ValueError(
                f"segment {index} has {arr.shape[1]} sensors; expected {n_sensors}"
            )
        if arr.shape[0] == 0:
            raise ValueError(f"segment {index} is empty")
        out.append(arr)
    if not out:
        raise ValueError("at least one time-series segment is required")
    return out


class M2ADPreprocessor:
    """Fit imputation/scaling on training once and reuse them at inference."""

    def __init__(
        self, n_sensors: int, feature_range: tuple[float, float] = (-1.0, 1.0)
    ) -> None:
        self.n_sensors = int(n_sensors)
        self.feature_range = tuple(float(value) for value in feature_range)
        if self.feature_range[0] >= self.feature_range[1]:
            raise ValueError("feature_range must be increasing")
        self.is_fitted = False

    def _impute(self, segment: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(np.asarray(segment, dtype=np.float64))
        values = (
            frame.interpolate(method="linear", limit_direction="both")
            .to_numpy(dtype=np.float64)
        )
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.take(self.impute_values, np.nonzero(missing)[1])
        return values

    def _scale(self, segment: np.ndarray) -> np.ndarray:
        low, high = self.feature_range
        unit = (np.asarray(segment, dtype=np.float64) - self.data_min) / self.data_scale
        return (low + unit * (high - low)).astype(np.float32)

    def fit_transform(self, segments: Sequence[np.ndarray]) -> list[np.ndarray]:
        raw = ensure_segments(segments, self.n_sensors)
        stacked = np.concatenate(raw, axis=0)
        self.impute_values = np.nanmedian(stacked, axis=0)
        if not np.isfinite(self.impute_values).all():
            bad = np.flatnonzero(~np.isfinite(self.impute_values)).tolist()
            raise ValueError(f"training sensors contain no finite values at indices: {bad}")
        filled = [self._impute(segment) for segment in raw]
        fit_values = np.concatenate(filled, axis=0)
        self.data_min = np.min(fit_values, axis=0)
        self.data_max = np.max(fit_values, axis=0)
        self.data_scale = np.where(
            self.data_max > self.data_min, self.data_max - self.data_min, 1.0
        )
        self.is_fitted = True
        return [self._scale(segment) for segment in filled]

    def transform(self, segments: Sequence[np.ndarray]) -> list[np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("call fit_transform before transform")
        raw = ensure_segments(segments, self.n_sensors)
        return [self._scale(self._impute(segment)) for segment in raw]

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("preprocessor is not fitted")
        low, high = self.feature_range
        unit = (np.asarray(values, dtype=np.float64) - low) / (high - low)
        return unit * self.data_scale + self.data_min
