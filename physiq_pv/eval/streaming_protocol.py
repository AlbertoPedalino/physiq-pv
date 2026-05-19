"""
Walk-forward streaming protocol for continual learning evaluation.

Splits a multi-year xr.Dataset into four disjoint regions:
  - train       (offline baseline fit)
  - calibration (held aside, used by Conformal Prediction)
  - holdout     (frozen reference for Backward Transfer)
  - stream      (sliding online windows, broken into ordered tasks)

The protocol is loader-agnostic: it operates on whatever xr.Dataset the
caller passes in. It does not load Sentinel / PVGIS itself.

Typical usage:

    split = WalkForwardSplit(
        ds,
        train_end="2019-12-31",
        calibration_months=2,
        holdout_months=1,
        stream_stride_hours=168,
        stream_window_hours=720,
    )

    base_model = train_offline(split.train_ds)
    cp = fit_conformal(split.calibration_ds)
    holdout = split.holdout_ds  # never updated; used by CL metrics

    for task_id, ds_window in split.stream_tasks():
        report = run_online_step(ds_window, model, ...)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import xarray as xr


_HOURS_PER_MONTH = 24 * 30  # approximate, used only for default window sizing


@dataclass(frozen=True)
class TaskSlice:
    """One stream task: an ordered time slice of the dataset."""
    task_id: int
    t_start: int
    t_end: int            # exclusive
    timestamp_start: pd.Timestamp
    timestamp_end: pd.Timestamp


class WalkForwardSplit:
    """
    Defines the temporal partition of a dataset for CL evaluation.

    Layout on the time axis:

        |----- train -----|--- calib ---|--- holdout ---|----- stream -----|
                                                       ^                 ^
                                                       stream_start      end

    holdout sits between calibration and stream so the stream cannot
    accidentally include holdout timestamps. The hold-out is frozen and used
    only as a BWT reference (replayed for evaluation, never updated on).
    """

    def __init__(
        self,
        ds: xr.Dataset,
        train_end: str | pd.Timestamp,
        calibration_months: float = 2.0,
        holdout_months: float = 1.0,
        stream_stride_hours: int = 168,    # 1 week
        stream_window_hours: int = 720,    # 30 days
        time_coord: str = "time",
    ):
        if time_coord not in ds.coords:
            raise KeyError(f"dataset missing '{time_coord}' coordinate")

        self.ds = ds
        self.time_coord = time_coord
        self.stream_stride = int(stream_stride_hours)
        self.stream_window = int(stream_window_hours)

        times = pd.DatetimeIndex(ds.coords[time_coord].values)
        self.times = times

        train_end_ts = pd.Timestamp(train_end)
        calib_hours = int(round(calibration_months * _HOURS_PER_MONTH))
        holdout_hours = int(round(holdout_months * _HOURS_PER_MONTH))

        train_end_idx = int(times.searchsorted(train_end_ts, side="right"))
        if train_end_idx <= 0:
            raise ValueError(f"train_end {train_end_ts} precedes dataset start {times[0]}")
        if train_end_idx >= len(times):
            raise ValueError(f"train_end {train_end_ts} after dataset end {times[-1]}")

        calib_end_idx = min(train_end_idx + calib_hours, len(times))
        holdout_end_idx = min(calib_end_idx + holdout_hours, len(times))
        stream_start_idx = holdout_end_idx
        stream_end_idx = len(times)

        if stream_end_idx - stream_start_idx < self.stream_window:
            raise ValueError(
                f"insufficient data after train+calib+holdout for at least "
                f"one stream window (need {self.stream_window} hours, "
                f"have {stream_end_idx - stream_start_idx})"
            )

        self.train_slice = slice(0, train_end_idx)
        self.calibration_slice = slice(train_end_idx, calib_end_idx)
        self.holdout_slice = slice(calib_end_idx, holdout_end_idx)
        self.stream_slice = slice(stream_start_idx, stream_end_idx)

        self._stream_start_idx = stream_start_idx
        self._stream_end_idx = stream_end_idx

    # ------------------------------------------------------------------ #
    # Slice accessors
    # ------------------------------------------------------------------ #

    @property
    def train_ds(self) -> xr.Dataset:
        return self.ds.isel({self.time_coord: self.train_slice})

    @property
    def calibration_ds(self) -> xr.Dataset:
        return self.ds.isel({self.time_coord: self.calibration_slice})

    @property
    def holdout_ds(self) -> xr.Dataset:
        return self.ds.isel({self.time_coord: self.holdout_slice})

    @property
    def stream_ds(self) -> xr.Dataset:
        return self.ds.isel({self.time_coord: self.stream_slice})

    # ------------------------------------------------------------------ #
    # Task iteration
    # ------------------------------------------------------------------ #

    def stream_tasks(self) -> Iterator[tuple[TaskSlice, xr.Dataset]]:
        """
        Iterate ordered stream tasks. Each task is a sliding window of
        `stream_window` hours, advanced by `stream_stride`. The window is
        absolute (anchored on dataset time axis), not cumulative.
        """
        task_id = 0
        cursor = self._stream_start_idx
        while cursor + self.stream_window <= self._stream_end_idx:
            t_start = cursor
            t_end = cursor + self.stream_window
            slice_obj = TaskSlice(
                task_id=task_id,
                t_start=t_start,
                t_end=t_end,
                timestamp_start=self.times[t_start],
                timestamp_end=self.times[t_end - 1],
            )
            ds_window = self.ds.isel({self.time_coord: slice(t_start, t_end)})
            yield slice_obj, ds_window
            task_id += 1
            cursor += self.stream_stride

    def n_tasks(self) -> int:
        n_steps = self._stream_end_idx - self._stream_start_idx
        if n_steps < self.stream_window:
            return 0
        return 1 + (n_steps - self.stream_window) // self.stream_stride

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        return {
            "n_total_hours": len(self.times),
            "train_hours":   self.train_slice.stop - self.train_slice.start,
            "calib_hours":   self.calibration_slice.stop - self.calibration_slice.start,
            "holdout_hours": self.holdout_slice.stop - self.holdout_slice.start,
            "stream_hours":  self.stream_slice.stop - self.stream_slice.start,
            "n_stream_tasks": self.n_tasks(),
            "stream_window": self.stream_window,
            "stream_stride": self.stream_stride,
            "train_range":   (self.times[self.train_slice.start], self.times[self.train_slice.stop - 1]),
            "calib_range":   (self.times[self.calibration_slice.start], self.times[self.calibration_slice.stop - 1]),
            "holdout_range": (self.times[self.holdout_slice.start], self.times[self.holdout_slice.stop - 1]),
            "stream_range":  (self.times[self.stream_slice.start], self.times[self.stream_slice.stop - 1]),
        }
