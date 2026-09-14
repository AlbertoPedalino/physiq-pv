"""Aligned PVGIS cubes and paper-style per-location STGAN windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..common import chronological_frame, detector_features, validate_numeric_features
from .grid import SpatialGrid


@dataclass
class AlignedPVGISCubes:
    train: np.ndarray
    test: np.ndarray
    train_timestamps: pd.DatetimeIndex
    test_timestamps: pd.DatetimeIndex
    location_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    latitudes: np.ndarray
    longitudes: np.ndarray

    def close(self):
        for array in (self.train, self.test):
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()


def prepend_training_context_to_test(
    train: np.ndarray,
    test: np.ndarray,
    *,
    train_timestamps: pd.DatetimeIndex,
    test_timestamps: pd.DatetimeIndex,
    context_steps: int,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Prefix test data with contiguous training history.

    The prefix is input context only: targets still start at the first test
    timestamp. This mirrors the official STGAN split, where the first test
    target can consume observations immediately preceding the split.
    """
    if context_steps < 1:
        raise ValueError("context_steps must be positive.")
    if train.ndim != 3 or test.ndim != 3 or train.shape[1:] != test.shape[1:]:
        raise ValueError("STGAN train/test context arrays are incompatible.")
    if len(train_timestamps) != train.shape[0] or len(test_timestamps) != test.shape[0]:
        raise ValueError("STGAN context timestamps do not match their arrays.")
    if len(train) < context_steps or len(test) == 0:
        raise ValueError("STGAN split is too short to construct test context.")

    context_times = train_timestamps[-context_steps:]
    combined_times = context_times.append(test_timestamps)
    deltas = np.diff(combined_times.asi8)
    if deltas.size == 0 or np.any(deltas <= 0):
        raise ValueError("STGAN train/test timestamps must be strictly increasing.")
    unique, counts = np.unique(deltas, return_counts=True)
    cadence = unique[np.argmax(counts)]
    if np.any(deltas != cadence):
        boundary = test_timestamps[0] - context_times[-1]
        raise ValueError(
            "STGAN requires contiguous regular history across the train/test "
            f"boundary; observed boundary delta {boundary}."
        )

    combined = np.concatenate(
        (np.asarray(train[-context_steps:]), np.asarray(test)), axis=0
    )
    return combined, combined_times


def _read_detector_frame(path: str | Path) -> pd.DataFrame:
    return chronological_frame(pd.read_csv(path))


def _write_split_cube(
    manifest: pd.DataFrame,
    *,
    path_column: str,
    feature_names: tuple[str, ...],
    output_path: Path,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    first = _read_detector_frame(manifest.iloc[0][path_column])
    missing = sorted(set(feature_names) - set(first.columns))
    if missing:
        raise ValueError(f"STGAN split is missing features: {missing}")
    validate_numeric_features(first, list(feature_names), method="STGAN")
    reference_times = pd.DatetimeIndex(first["timestamp"])
    cube = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(reference_times), len(manifest), len(feature_names)),
    )

    try:
        for location_index, row in enumerate(manifest.itertuples(index=False)):
            frame = first if location_index == 0 else _read_detector_frame(
                getattr(row, path_column)
            )
            missing = sorted(set(feature_names) - set(frame.columns))
            if missing:
                raise ValueError(
                    f"{getattr(row, 'site_key', location_index)} is missing STGAN features: {missing}"
                )
            validate_numeric_features(frame, list(feature_names), method="STGAN")
            times = pd.DatetimeIndex(frame["timestamp"])
            if not times.equals(reference_times):
                raise ValueError(
                    "STGAN requires identical synchronized timestamps at every location."
                )
            cube[:, location_index, :] = frame.loc[:, feature_names].to_numpy(
                dtype=np.float32
            )
        cube.flush()
    finally:
        cube._mmap.close()
    return np.load(output_path, mmap_mode="r"), reference_times


def load_aligned_manifest_cubes(
    manifest: pd.DataFrame,
    *,
    cache_dir: str | Path,
) -> AlignedPVGISCubes:
    """Materialise disk-backed ``[time, location, feature]`` arrays.

    The per-location CSV contract remains compatible with the MTGFlow
    preparation step while avoiding a second in-memory copy of the full fleet.
    """
    required = {
        "location",
        "site_key",
        "train_csv",
        "test_csv",
        "latitude",
        "longitude",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(
            "STGAN manifest is missing required columns: "
            f"{missing}. Regenerate it with prepare_pvgis_stgan.py."
        )
    if manifest.empty:
        raise ValueError("STGAN manifest contains no locations.")
    if manifest["location"].astype(str).duplicated().any():
        raise ValueError("STGAN manifest contains duplicate locations.")

    manifest = manifest.reset_index(drop=True)
    first_train = _read_detector_frame(manifest.iloc[0]["train_csv"])
    feature_names = tuple(detector_features(first_train))
    cache_root = Path(cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    train, train_times = _write_split_cube(
        manifest,
        path_column="train_csv",
        feature_names=feature_names,
        output_path=cache_root / "train_cube.npy",
    )
    try:
        test, test_times = _write_split_cube(
            manifest,
            path_column="test_csv",
            feature_names=feature_names,
            output_path=cache_root / "test_cube.npy",
        )
    except Exception:
        train._mmap.close()
        raise
    return AlignedPVGISCubes(
        train=train,
        test=test,
        train_timestamps=train_times,
        test_timestamps=test_times,
        location_names=tuple(manifest["location"].astype(str)),
        feature_names=feature_names,
        latitudes=manifest["latitude"].to_numpy(dtype=np.float64),
        longitudes=manifest["longitude"].to_numpy(dtype=np.float64),
    )


def regular_target_indices(
    timestamps: pd.DatetimeIndex,
    *,
    context_steps: int,
    stride: int,
) -> np.ndarray:
    """Return target indices whose entire context has the modal cadence."""
    if context_steps < 1 or stride < 1:
        raise ValueError("context_steps and stride must be positive.")
    candidates = np.arange(context_steps, len(timestamps), stride, dtype=np.int64)
    if candidates.size == 0:
        return candidates
    deltas = np.diff(timestamps.asi8)
    positive = deltas[deltas > 0]
    if positive.size == 0:
        raise ValueError("Cannot infer a positive STGAN sampling cadence.")
    unique, counts = np.unique(positive, return_counts=True)
    cadence = unique[np.argmax(counts)]
    irregular = deltas != cadence
    prefix = np.concatenate(([0], np.cumsum(irregular, dtype=np.int64)))
    return candidates[prefix[candidates] == prefix[candidates - context_steps]]


def calendar_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Paper-compatible weekday (7) plus hour-of-day (24) one-hot features."""
    result = np.zeros((len(timestamps), 31), dtype=np.float32)
    rows = np.arange(len(timestamps))
    result[rows, timestamps.dayofweek.to_numpy()] = 1.0
    result[rows, 7 + timestamps.hour.to_numpy()] = 1.0
    return result


class STGANWindowDataset(Dataset):
    """Flatten target time and target location into paper-style samples."""

    def __init__(
        self,
        data: np.ndarray,
        timestamps: pd.DatetimeIndex,
        grid: SpatialGrid,
        *,
        feature_minimum: np.ndarray,
        feature_scale: np.ndarray,
        recent_steps: int,
        trend_steps: int,
        stride: int,
    ):
        if data.ndim != 3:
            raise ValueError("STGAN data must be [time,location,feature].")
        if data.shape[0] != len(timestamps):
            raise ValueError("STGAN timestamps do not match the data time axis.")
        if data.shape[1] != grid.n_locations:
            raise ValueError("STGAN grid does not match the data location axis.")
        if recent_steps < 1 or trend_steps < recent_steps:
            raise ValueError("Require trend_steps >= recent_steps >= 1.")
        self.data = data
        self.timestamps = timestamps
        self.grid = grid
        self.minimum = np.asarray(feature_minimum, dtype=np.float32)
        self.scale = np.asarray(feature_scale, dtype=np.float32)
        self.recent_steps = int(recent_steps)
        self.trend_steps = int(trend_steps)
        self.targets = regular_target_indices(
            timestamps, context_steps=trend_steps, stride=stride
        )
        self.time_features = calendar_features(timestamps)
        self.n_locations = data.shape[1]

    def __len__(self) -> int:
        return len(self.targets) * self.n_locations

    @property
    def target_timestamps(self) -> pd.DatetimeIndex:
        return self.timestamps[self.targets]

    def _normalise(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.minimum) / self.scale * 2.0 - 1.0).astype(
            np.float32, copy=False
        )

    def __getitem__(self, item: int):
        target_position = item // self.n_locations
        location_index = item % self.n_locations
        target_time = int(self.targets[target_position])
        nodes = self.grid.node_indices[location_index]
        valid = self.grid.valid_mask[location_index]
        safe_nodes = np.maximum(nodes, 0).ravel()
        size = self.grid.patch_size
        recent = self._normalise(
            np.asarray(
                self.data[target_time - self.recent_steps : target_time, safe_nodes, :]
            )
        )
        recent = recent.reshape(self.recent_steps, size, size, -1).transpose(0, 3, 1, 2)
        recent = np.where(valid[None, None], recent, 0.0)
        trend = self._normalise(
            np.asarray(
                self.data[
                    target_time - self.trend_steps : target_time,
                    location_index,
                    :,
                ]
            )
        )
        observed = self._normalise(np.asarray(self.data[target_time, safe_nodes, :]))
        observed = observed.reshape(size, size, -1).transpose(2, 0, 1)
        observed = np.where(valid[None], observed, 0.0)
        mask = valid[None].astype(np.float32)
        return (
            torch.from_numpy(np.ascontiguousarray(recent)),
            torch.from_numpy(np.ascontiguousarray(trend)),
            torch.from_numpy(np.ascontiguousarray(mask)),
            torch.from_numpy(np.ascontiguousarray(self.time_features[target_time])),
            torch.from_numpy(np.ascontiguousarray(observed)),
            target_position,
            location_index,
        )
