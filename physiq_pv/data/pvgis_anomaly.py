"""Detector-neutral PVGIS adapters for multivariate anomaly detection."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from physiq_pv.data.pvgis_irradiance import with_effective_poa


DEFAULT_ANOMALY_SENSORS = [
    "pv_power_output",
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
    prepared = {
        year: with_effective_poa(dataset)
        for year, dataset in sorted(datasets.items())
    }
    first_year = next(iter(prepared))
    first = prepared[first_year]
    if "location" not in first.dims or "time" not in first.dims:
        raise ValueError("PVGIS datasets must have 'location' and 'time' dimensions")
    locations = np.asarray(first["location"].values)
    location_keys = locations.astype(str)
    for year, dataset in prepared.items():
        missing = [sensor for sensor in sensor_names if sensor not in dataset]
        if missing:
            raise ValueError(f"PVGIS year {year} is missing sensors: {missing}")
        current = np.asarray(dataset["location"].values)
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
    """Extract one ``(time, sensors)`` segment per year and location."""

    segments: list[np.ndarray] = []
    timestamps: list[pd.DatetimeIndex] = []
    for year, dataset in sorted(datasets.items()):
        arrays: list[np.ndarray] = []
        for sensor in sensor_names:
            values = dataset[sensor].sel(location=location)
            extra_dims = [dimension for dimension in values.dims if dimension != "time"]
            if extra_dims:
                raise ValueError(
                    f"sensor {sensor!r} in year {year} is not scalar per location; "
                    f"remaining dimensions are {extra_dims}"
                )
            arrays.append(
                np.asarray(values.transpose("time").values, dtype=np.float64)
            )
        segment = np.column_stack(arrays)
        times = pd.DatetimeIndex(dataset["time"].values)
        if len(times) != len(segment):
            raise ValueError(f"time length mismatch in PVGIS year {year}")
        segments.append(segment)
        timestamps.append(times)
    return segments, timestamps


def available_locations(datasets: Dict[int, xr.Dataset]) -> np.ndarray:
    if not datasets:
        raise ValueError("at least one PVGIS year is required")
    first = datasets[next(iter(sorted(datasets)))]
    return np.asarray(first["location"].values)
