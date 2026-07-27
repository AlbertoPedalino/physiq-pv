"""PVGIS adapters for the label-unaware M2AD detector."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from physiq_pv.data.pvgis_irradiance import with_effective_poa


# PVGIS power is a theoretical target derived from the same irradiance and
# weather inputs. Keep detector labels independent from the SDE forecast target.
DEFAULT_M2AD_SENSORS = [
    "solar_irradiance_poa",
    "temperature_2m",
    "wind_speed_10m",
]


def prepare_pvgis_years(
    datasets: Dict[int, xr.Dataset], sensor_names: Sequence[str]
) -> Dict[int, xr.Dataset]:
    """Validate sensors and a common, ordered location set for all years."""
    if not datasets:
        raise ValueError("at least one PVGIS year is required")
    prepared = {year: with_effective_poa(ds) for year, ds in sorted(datasets.items())}
    first_year = next(iter(prepared))
    first = prepared[first_year]
    if "location" not in first.dims or "time" not in first.dims:
        raise ValueError("PVGIS datasets must have 'location' and 'time' dimensions")
    locations = np.asarray(first["location"].values)
    location_keys = locations.astype(str)
    for year, ds in prepared.items():
        missing = [sensor for sensor in sensor_names if sensor not in ds]
        if missing:
            raise ValueError(f"PVGIS year {year} is missing M2AD sensors: {missing}")
        current = np.asarray(ds["location"].values)
        if not np.array_equal(current.astype(str), location_keys):
            raise ValueError(
                f"PVGIS location ordering differs in {year}; all years must use "
                "the same assets in the same order"
            )
    return prepared


def extract_location_segments(
    datasets: Dict[int, xr.Dataset],
    location,
    sensor_names: Sequence[str],
) -> tuple[list[np.ndarray], list[pd.DatetimeIndex]]:
    """Extract one (time, sensors) segment per year for a PVGIS location."""
    segments: list[np.ndarray] = []
    timestamps: list[pd.DatetimeIndex] = []
    for year, ds in sorted(datasets.items()):
        arrays: list[np.ndarray] = []
        for sensor in sensor_names:
            da = ds[sensor].sel(location=location)
            extra_dims = [dim for dim in da.dims if dim != "time"]
            if extra_dims:
                raise ValueError(
                    f"sensor {sensor!r} in year {year} is not scalar per location; "
                    f"remaining dimensions are {extra_dims}"
                )
            arrays.append(np.asarray(da.transpose("time").values, dtype=np.float64))
        segment = np.column_stack(arrays)
        times = pd.DatetimeIndex(ds["time"].values)
        if len(times) != len(segment):
            raise ValueError(f"time length mismatch in PVGIS year {year}")
        segments.append(segment)
        timestamps.append(times)
    return segments, timestamps


def available_locations(datasets: Dict[int, xr.Dataset]) -> np.ndarray:
    first = datasets[next(iter(sorted(datasets)))]
    return np.asarray(first["location"].values)
