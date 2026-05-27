from __future__ import annotations

from typing import Iterator

import pandas as pd
import xarray as xr
from dateutil.relativedelta import relativedelta


class TemporalStream:
    """
    Date-based temporal windowing over an xr.Dataset for continual adaptation.

    Splits the time axis into:
      1. An initial training region [initial_train_start, initial_train_end].
      2. Successive non-overlapping windows of window_months duration.

    Date boundaries are inclusive: --initial-train-end 2023-03-31 includes
    all hours of 2023-03-31.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        initial_train_start: str | pd.Timestamp,
        initial_train_end: str | pd.Timestamp,
        window_months: int = 1,
        max_windows: int | None = None,
        time_coord: str = "time",
    ):
        self.ds = ds
        self.train_start = pd.Timestamp(initial_train_start)
        self.train_end_exclusive = pd.Timestamp(initial_train_end) + pd.Timedelta(days=1)
        self.window_months = int(window_months)
        self.max_windows = max_windows
        self.time_coord = time_coord
        self.times = pd.DatetimeIndex(ds.coords[time_coord].values)

    def initial_train_ds(self) -> xr.Dataset:
        mask = (self.times >= self.train_start) & (self.times < self.train_end_exclusive)
        return self.ds.isel({self.time_coord: mask})

    def stream_windows(
        self,
    ) -> Iterator[tuple[int, pd.Timestamp, pd.Timestamp, xr.Dataset]]:
        """Yield (window_id, window_start, window_end, ds_window).

        Empty windows (e.g. June/August gaps in Sentinel 2019 data) are
        silently skipped — the cursor advances but no window is yielded.
        The loop terminates only when the cursor passes the last timestamp
        in the dataset.
        """
        cursor = self.train_end_exclusive
        dataset_end = self.times[-1]
        window_id = 0

        while cursor <= dataset_end:
            if self.max_windows is not None and window_id >= self.max_windows:
                break

            next_cursor = cursor + relativedelta(months=self.window_months)
            mask = (self.times >= cursor) & (self.times < next_cursor)
            ds_window = self.ds.isel({self.time_coord: mask})

            if ds_window.sizes[self.time_coord] == 0:
                cursor = next_cursor
                continue

            yield window_id, cursor, next_cursor - pd.Timedelta(hours=1), ds_window
            cursor = next_cursor
            window_id += 1

    def n_windows(self) -> int:
        return sum(1 for _ in self.stream_windows())
