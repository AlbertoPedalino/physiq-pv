"""Leakage-safe preprocessing and window datasets for CATCH."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class CATCHPreprocessor:
    """Train-only interpolation and standardisation for multivariate segments."""

    def __init__(self, n_channels: int, *, eps: float = 1e-6) -> None:
        self.n_channels = int(n_channels)
        self.eps = float(eps)
        self.is_fitted = False

    def _validate_and_interpolate(self, segment: np.ndarray) -> np.ndarray:
        values = np.asarray(segment, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_channels:
            raise ValueError(
                f"each segment must have shape (time, {self.n_channels})"
            )
        if not len(values):
            raise ValueError("segments must not be empty")
        result = values.copy()
        index = np.arange(len(result))
        for channel in range(self.n_channels):
            finite = np.isfinite(result[:, channel])
            if not finite.any():
                raise ValueError(f"channel {channel} contains no finite training values")
            if not finite.all():
                result[:, channel] = np.interp(
                    index, index[finite], result[finite, channel]
                )
        return result

    def fit(self, segments: Sequence[np.ndarray]) -> "CATCHPreprocessor":
        prepared = [self._validate_and_interpolate(segment) for segment in segments]
        if not prepared:
            raise ValueError("at least one training segment is required")
        stacked = np.concatenate(prepared, axis=0)
        self.mean_ = stacked.mean(axis=0)
        self.scale_ = stacked.std(axis=0)
        self.scale_ = np.where(self.scale_ < self.eps, 1.0, self.scale_)
        self.is_fitted = True
        return self

    def transform(self, segments: Sequence[np.ndarray]) -> list[np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("call fit before transform")
        return [
            ((self._validate_and_interpolate(segment) - self.mean_) / self.scale_)
            .astype(np.float32)
            for segment in segments
        ]

    def fit_transform(self, segments: Sequence[np.ndarray]) -> list[np.ndarray]:
        return self.fit(segments).transform(segments)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("call fit before inverse_transform")
        return np.asarray(values) * self.scale_ + self.mean_


def temporal_train_validation_split(
    segments: Sequence[np.ndarray],
    *,
    validation_split: float,
    min_length: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split each segment chronologically without overlapping train/validation."""

    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation_split must be in [0, 1)")
    train_segments: list[np.ndarray] = []
    validation_segments: list[np.ndarray] = []
    for segment_id, segment in enumerate(segments):
        values = np.asarray(segment)
        if len(values) < min_length:
            raise ValueError(
                f"segment {segment_id} has {len(values)} points, fewer than {min_length}"
            )
        if validation_split == 0:
            train_segments.append(values)
            continue
        boundary = int(np.floor(len(values) * (1.0 - validation_split)))
        if boundary < min_length or len(values) - boundary < min_length:
            raise ValueError(
                f"segment {segment_id} cannot provide disjoint train/validation "
                f"parts of at least {min_length} points"
            )
        train_segments.append(values[:boundary])
        validation_segments.append(values[boundary:])
    return train_segments, validation_segments


class CATCHWindowDataset(Dataset):
    """Sliding windows that never cross segment/year boundaries."""

    def __init__(
        self,
        segments: Sequence[np.ndarray],
        seq_len: int,
        *,
        stride: int = 1,
    ) -> None:
        if seq_len < 2 or stride < 1:
            raise ValueError("seq_len must be >= 2 and stride must be >= 1")
        self.segments = [np.asarray(segment, dtype=np.float32) for segment in segments]
        self.seq_len = int(seq_len)
        self.entries: list[tuple[int, int]] = []
        for segment_id, segment in enumerate(self.segments):
            if segment.ndim != 2:
                raise ValueError("segments must have shape (time, channels)")
            if len(segment) < seq_len:
                raise ValueError(
                    f"segment {segment_id} has {len(segment)} points, fewer than seq_len={seq_len}"
                )
            starts = list(range(0, len(segment) - seq_len + 1, stride))
            final_start = len(segment) - seq_len
            if starts[-1] != final_start:
                starts.append(final_start)
            self.entries.extend((segment_id, start) for start in starts)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        segment_id, start = self.entries[index]
        window = self.segments[segment_id][start : start + self.seq_len]
        return torch.from_numpy(window), segment_id, start
