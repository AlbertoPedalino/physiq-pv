"""Per-location seasonal standardisation: remove the mean annual and diurnal cycles.

For every location, feature, time of day and day of year, the training period
gives a mean and a standard deviation pooled over a circular window of days.
Data become (value - mean) / std: how unusual a value is for that place,
season and hour. Statistics never use calibration or test data.

The climate drifts: years after the training ones are, as a whole, away from
that normal. A line in time through the yearly means of the standardised
training data, one for each feature, can be removed from every period, so that
a value is unusual for the climate of its own time. The line is fitted on the
training years alone and extended to the others.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DAYS = 366  # Day-of-year bins; day 366 only receives leap-year samples.


class FourthRoot:
    """An array [time, location, feature] read with the fourth root of some of its features.

    For a feature that is 0 almost always with rare large values, as the precipitation: a
    standard deviation describes its fourth root, which it does not the values themselves.
    Values below 0 count as 0. Only time is indexed; the features stay on the last axis.
    """
    def __init__(self, data, columns):
        self.data, self.columns = data, list(columns)
        self.shape, self.ndim, self.dtype = data.shape, 3, np.dtype(np.float64)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        block = np.array(self.data[key], dtype=np.float64)
        block[..., self.columns] = np.sqrt(np.sqrt(np.maximum(block[..., self.columns], 0.0)))
        return block


@dataclass
class SeasonalClimatology:
    slots: np.ndarray  # Seconds since midnight of each training time of day.
    mean: np.ndarray  # [slot, day, location, feature]
    std: np.ndarray
    metadata: dict

    def close(self):
        for array in (self.mean, self.std):
            if getattr(array, "_mmap", None) is not None:
                array._mmap.close()


def _bins(timestamps: pd.DatetimeIndex, slots: np.ndarray):
    seconds = (timestamps.hour.to_numpy() * 3600 + timestamps.minute.to_numpy() * 60
               + timestamps.second.to_numpy())
    slot = np.minimum(np.searchsorted(slots, seconds), len(slots) - 1)
    if np.any(slots[slot] != seconds):
        raise ValueError("Timestamps contain a time of day absent from the training climatology.")
    return timestamps.dayofyear.to_numpy() - 1, slot


def _window_sum(values: np.ndarray, half: int) -> np.ndarray:
    """Circular sum over 2*half+1 consecutive day bins."""
    if not half:
        return values
    padded = np.concatenate((values[-half:], values, values[:half]))
    cumulative = np.concatenate((np.zeros_like(padded[:1]), np.cumsum(padded, axis=0)))
    return cumulative[2 * half + 1:] - cumulative[:-(2 * half + 1)]


def fit_seasonal_climatology(train, timestamps: pd.DatetimeIndex, directory, *,
                             window_days: int = 15, std_floor: float = 0.01,
                             buffer_bytes: int = 64 * 1024**2) -> SeasonalClimatology:
    """Fit on disk, with one pass over the training data and O(day bins) RAM.

    ``std_floor`` is a fraction of each feature's overall training standard
    deviation: it bounds the division where a feature barely varies (night-time
    solar radiation, dry-season precipitation).
    """
    if type(window_days) is not int or not 0 <= window_days <= DAYS // 2:
        raise ValueError("window_days must be an integer between 0 and 183.")
    if not np.isfinite(std_floor) or std_floor <= 0:
        raise ValueError("std_floor must be finite and positive.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seconds = (timestamps.hour.to_numpy() * 3600 + timestamps.minute.to_numpy() * 60
               + timestamps.second.to_numpy())
    slots = np.unique(seconds)
    day, slot = _bins(timestamps, slots)
    shape = (len(slots), DAYS, *train.shape[1:])
    paths = [directory / name for name in ("_seasonal_sum.npy", "_seasonal_square.npy")]
    total, square = (np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=shape)
                     for path in paths)
    counts = np.zeros(shape[:2])
    try:
        rows = max(1, buffer_bytes // (int(np.prod(shape[2:])) * 8))
        for start in range(0, len(train), rows):
            block = np.asarray(train[start:start + rows], dtype=np.float64)
            for offset, values in enumerate(block):
                key = slot[start + offset], day[start + offset]
                total[key] += values
                square[key] += values * values
                counts[key] += 1
        samples = counts.sum() * shape[2]
        feature_mean = sum(np.asarray(total[index]).sum(axis=(0, 1)) for index in range(len(slots))) / samples
        feature_square = sum(np.asarray(square[index]).sum(axis=(0, 1)) for index in range(len(slots))) / samples
        floor = std_floor * np.sqrt(np.maximum(feature_square - feature_mean**2, 0.0))
        floor[floor <= 0] = 1.0  # Constant feature: standardised values stay zero.
        mean, std = (np.lib.format.open_memmap(directory / name, mode="w+", dtype=np.float32, shape=shape)
                     for name in ("seasonal_mean.npy", "seasonal_std.npy"))
        try:
            for index in range(len(slots)):
                pooled = _window_sum(counts[index], window_days)[:, None, None]
                if pooled.min() < 2:
                    raise ValueError("Too few training samples per seasonal bin; widen window_days.")
                day_total = np.asarray(total[index])
                average = _window_sum(day_total, window_days) / pooled
                # Spread around each day's own smoothed mean: pooling raw values would
                # count the drift of the seasonal cycle inside the window as variability.
                residual = (np.asarray(square[index]) - 2.0 * average * day_total
                            + counts[index][:, None, None] * average**2)
                variance = _window_sum(residual, window_days) / pooled
                mean[index] = average
                std[index] = np.maximum(np.sqrt(np.maximum(variance, 0.0)), floor)
            mean.flush()
            std.flush()
        finally:
            mean._mmap.close()
            std._mmap.close()
    finally:
        total._mmap.close()
        square._mmap.close()
        for path in paths:
            path.unlink(missing_ok=True)
    return SeasonalClimatology(
        slots, np.load(directory / "seasonal_mean.npy", mmap_mode="r"),
        np.load(directory / "seasonal_std.npy", mmap_mode="r"),
        {"kind": "per_location_day_of_year_and_time_of_day_mean_std",
         "fit_period": {"start": str(timestamps[0]), "end": str(timestamps[-1]),
                        "timestamps": len(timestamps)},
         "window_days_each_side": window_days, "day_bins": DAYS,
         "time_of_day_slots_seconds": slots.tolist(),
         "std_floor_fraction_of_feature_std": std_floor,
         "min_samples_per_bin": int(min(_window_sum(counts[index], window_days).min()
                                        for index in range(len(slots)))),
         "mean_file": str(directory / "seasonal_mean.npy"),
         "std_file": str(directory / "seasonal_std.npy")})


def _years(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Time in years, with the fraction of the year."""
    return timestamps.year.to_numpy() + (timestamps.dayofyear.to_numpy() - 1) / 365.25


def fit_seasonal_trend(data, timestamps: pd.DatetimeIndex, climatology: SeasonalClimatology, *,
                       stride: int = 9) -> dict:
    """Line in time of every feature of the standardised training data.

    The line goes through the yearly means over all locations; its slope is in standard
    deviations by year and it is 0 at the middle of the training years, where the standardised
    data have zero mean. One timestamp every ``stride`` is read: an odd number, so that every time
    of day is.
    """
    if type(stride) is not int or stride < 1 or stride % 2 == 0:
        raise ValueError("stride must be an odd positive integer.")
    positions = np.arange(0, len(data), stride)
    sampled = timestamps[positions]
    day, slot = _bins(sampled, climatology.slots)
    years = sampled.year.to_numpy()
    labels = np.unique(years)
    if len(labels) < 2:
        raise ValueError("A trend needs at least two training years.")
    totals, counts = np.zeros((len(labels), data.shape[2])), np.zeros(len(labels))
    for index, position in enumerate(positions):
        key = slot[index], day[index]
        values = (np.asarray(data[position:position + 1], dtype=np.float64)[0] - climatology.mean[key]) / climatology.std[key]
        row = np.searchsorted(labels, years[index])
        totals[row] += values.mean(axis=0)
        counts[row] += 1
    yearly = totals / counts[:, None]
    time = labels + 0.5
    centred = time - time.mean()
    slope = (centred[:, None] * (yearly - yearly.mean(axis=0))).sum(axis=0) / (centred**2).sum()
    return {"kind": "one_line_in_time_per_feature_through_the_yearly_means_of_the_standardised_training_data",
            "slope_per_year": [float(value) for value in slope], "zero_at_year": float(time.mean()),
            "fit_years": [int(labels[0]), int(labels[-1])], "timestamp_stride": stride}


def seasonal_memmap(data, timestamps: pd.DatetimeIndex, climatology: SeasonalClimatology, path, *,
                    trend: dict | None = None, buffer_bytes: int = 16 * 1024**2):
    """Standardise once to a disk cache, in bounded time chunks; ``trend`` (of ``fit_seasonal_trend``) is removed."""
    path = Path(path)
    if len(timestamps) != len(data):
        raise ValueError("Seasonal standardisation timestamps do not match the data.")
    day, slot = _bins(timestamps, climatology.slots)
    line = None
    if trend is not None:
        slope = np.asarray(trend["slope_per_year"], dtype=np.float64)
        if slope.shape != (data.shape[2],):
            raise ValueError("The trend does not have one slope per feature.")
        line = (_years(timestamps) - trend["zero_at_year"])[:, None, None] * slope
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=data.shape)
    rows = max(1, buffer_bytes // (int(np.prod(data.shape[1:])) * 8))
    try:
        for start in range(0, len(data), rows):
            key = slot[start:start + rows], day[start:start + rows]
            block = (np.asarray(data[start:start + rows], dtype=np.float64) - climatology.mean[key]) / climatology.std[key]
            output[start:start + rows] = block if line is None else block - line[start:start + rows]
        output.flush()
    finally:
        output._mmap.close()
    return np.load(path, mmap_mode="r")
