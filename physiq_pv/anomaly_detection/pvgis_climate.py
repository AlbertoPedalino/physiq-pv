"""Prepare leakage-safe, per-location PVGIS climate time series.

The detector never consumes plant diagnostics or the future PV regression
target. It models meteorological and solar conditions only. The default
protocol is train=2005--2018 and target/test=2019; an optional validation year
can be prepared separately.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from physiq_pv.data.pvgis_dataset import build_year_raw, load_pvgis_years


CLIMATE_FEATURES = (
    "solar_irradiance_poa",
    "temperature_2m",
    "wind_speed_10m",
)

_SEASONALLY_NORMALIZED = CLIMATE_FEATURES
_MAD_TO_STD = 1.4826
_EPS = 1e-6


def _safe_key(value: object, index: int) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return f"loc_{index:04d}_{text or 'unknown'}"


def raw_to_climate_frame(raw: dict, location_column: int = 0) -> pd.DataFrame:
    """Convert one column of ``build_year_raw`` output to a climate frame."""
    n_locations = raw["temp"].shape[1]
    if not 0 <= location_column < n_locations:
        raise IndexError(
            f"location_column={location_column} outside raw location count {n_locations}."
        )
    times = pd.DatetimeIndex(raw["times"])
    return pd.DataFrame(
        {
            "timestamp": times,
            "solar_irradiance_poa": raw["solar_wm2"][:, location_column],
            "temperature_2m": raw["temp"][:, location_column],
            "wind_speed_10m": raw["wind"][:, location_column],
            "is_daytime": raw["day"][:, location_column].astype(bool),
        }
    )


@dataclass
class SeasonalRobustScaler:
    """Month-hour median/MAD scaler fitted exclusively on training history."""

    features: tuple[str, ...]
    global_median: dict[str, float]
    global_scale: dict[str, float]
    bucket_median: dict[str, list[float]]
    bucket_scale: dict[str, list[float]]

    @classmethod
    def fit(cls, frame: pd.DataFrame, features=_SEASONALLY_NORMALIZED):
        times = pd.DatetimeIndex(frame["timestamp"])
        buckets = (times.month.to_numpy() - 1) * 24 + times.hour.to_numpy()
        global_median: dict[str, float] = {}
        global_scale: dict[str, float] = {}
        bucket_median: dict[str, list[float]] = {}
        bucket_scale: dict[str, list[float]] = {}
        for feature in features:
            values = frame[feature].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                raise ValueError(f"Feature {feature!r} has no finite training values.")
            gmed = float(np.median(finite))
            gscale = float(_MAD_TO_STD * np.median(np.abs(finite - gmed)))
            if not np.isfinite(gscale) or gscale < _EPS:
                gscale = float(np.std(finite))
            if not np.isfinite(gscale) or gscale < _EPS:
                gscale = 1.0
            meds = np.full(288, gmed, dtype=np.float64)
            scales = np.full(288, gscale, dtype=np.float64)
            for bucket in range(288):
                selected = values[(buckets == bucket) & np.isfinite(values)]
                if selected.size < 3:
                    continue
                med = float(np.median(selected))
                scale = float(_MAD_TO_STD * np.median(np.abs(selected - med)))
                meds[bucket] = med
                scales[bucket] = scale if np.isfinite(scale) and scale >= _EPS else gscale
            global_median[feature] = gmed
            global_scale[feature] = gscale
            bucket_median[feature] = meds.tolist()
            bucket_scale[feature] = scales.tolist()
        return cls(tuple(features), global_median, global_scale, bucket_median, bucket_scale)

    def transform(self, frame: pd.DataFrame, clip: float = 12.0) -> pd.DataFrame:
        out = frame.copy()
        times = pd.DatetimeIndex(out["timestamp"])
        buckets = (times.month.to_numpy() - 1) * 24 + times.hour.to_numpy()
        for feature in self.features:
            values = out[feature].to_numpy(dtype=np.float64)
            med = np.asarray(self.bucket_median[feature])[buckets]
            scale = np.asarray(self.bucket_scale[feature])[buckets]
            out[feature] = np.clip((values - med) / scale, -clip, clip).astype(np.float32)
        return out

    def to_dict(self) -> dict:
        return {
            "kind": "month_hour_median_mad",
            "fit_period": "training_only",
            "features": list(self.features),
            "global_median": self.global_median,
            "global_scale": self.global_scale,
            "bucket_median": self.bucket_median,
            "bucket_scale": self.bucket_scale,
        }


def _assert_same_locations(year_map: dict[int, object]) -> np.ndarray:
    first_year = min(year_map)
    reference = np.asarray(year_map[first_year]["location"].values).astype(str)
    for year, ds in year_map.items():
        current = np.asarray(ds["location"].values).astype(str)
        if current.shape != reference.shape or not np.array_equal(current, reference):
            raise ValueError(f"PVGIS locations/order in {year} differ from {first_year}.")
    return reference


def prepare_pvgis_climate_data(
    *,
    pvgis_dir: str,
    out_dir: str,
    train_start: int = 2005,
    train_end: int = 2018,
    validation_year: int | None = None,
    test_year: int = 2019,
    file_template: str = "piedmont_pvgis_{year}.nc",
    target_variable: str = "pv_power_output",
    shard_index: int = 0,
    num_shards: int = 1,
    max_locations: int | None = None,
    location_batch_size: int = 8,
    seasonal_normalization: bool = False,
) -> pd.DataFrame:
    """Write per-location train/validation/test CSVs and return their manifest."""
    if not 0 <= shard_index < num_shards:
        raise ValueError("Require 0 <= shard_index < num_shards.")
    if location_batch_size < 1:
        raise ValueError("location_batch_size must be >= 1.")
    if train_start > train_end:
        raise ValueError("train_start must be <= train_end.")
    train_years = list(range(train_start, train_end + 1))
    if test_year in train_years:
        raise ValueError("test_year must not overlap the training period.")
    if validation_year is not None and (
        validation_year in train_years or validation_year == test_year
    ):
        raise ValueError("validation_year must be distinct from train and test years.")
    years = train_years + ([] if validation_year is None else [validation_year]) + [test_year]
    year_map = load_pvgis_years(pvgis_dir, years, file_template=file_template)
    missing = sorted(set(years) - set(year_map))
    if missing:
        raise FileNotFoundError(f"Missing required PVGIS years: {missing}.")
    locations = _assert_same_locations(year_map)
    coordinate_source = year_map[min(year_map)]
    latitudes = np.asarray(coordinate_source["lat"].values, dtype=np.float64)
    longitudes = np.asarray(coordinate_source["lon"].values, dtype=np.float64)
    if latitudes.shape != locations.shape or longitudes.shape != locations.shape:
        raise ValueError("PVGIS latitude/longitude coordinates do not match locations.")
    selected = [i for i in range(len(locations)) if i % num_shards == shard_index]
    if max_locations is not None:
        selected = selected[:max_locations]

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    try:
        for batch_start in range(0, len(selected), location_batch_size):
            batch_indices = selected[batch_start : batch_start + location_batch_size]
            site_frames: dict[int, dict[int, pd.DataFrame]] = {
                index: {} for index in batch_indices
            }
            for year in years:
                batch_ds = year_map[year].isel(location=batch_indices)
                batch_raw = build_year_raw(batch_ds, target_variable=target_variable)
                for column, index in enumerate(batch_indices):
                    site_frames[index][year] = raw_to_climate_frame(
                        batch_raw, location_column=column
                    )
            for index in batch_indices:
                key = _safe_key(locations[index], index)
                frames = site_frames[index]
                train = pd.concat(
                    [frames[y] for y in range(train_start, train_end + 1)], ignore_index=True
                )
                validation = (
                    None
                    if validation_year is None
                    else frames[validation_year].reset_index(drop=True)
                )
                test = frames[test_year].reset_index(drop=True)
                scaler = None
                if seasonal_normalization:
                    scaler = SeasonalRobustScaler.fit(train)
                    train = scaler.transform(train)
                    if validation is not None:
                        validation = scaler.transform(validation)
                    test = scaler.transform(test)

                site_dir = root / key
                site_dir.mkdir(parents=True, exist_ok=True)
                train.to_csv(site_dir / "train.csv", index=False)
                validation_path = None
                if validation is not None:
                    validation_path = site_dir / "validation.csv"
                    validation.to_csv(validation_path, index=False)
                test.to_csv(site_dir / "test.csv", index=False)
                metadata = {
                    "location": str(locations[index]),
                    "location_index": index,
                    "latitude": float(latitudes[index]),
                    "longitude": float(longitudes[index]),
                    "train_years": [train_start, train_end],
                    "validation_year": validation_year,
                    "test_year": test_year,
                    "features": list(CLIMATE_FEATURES),
                    "target_excluded": True,
                    "seasonal_normalization": seasonal_normalization,
                    "normalizer": None if scaler is None else scaler.to_dict(),
                }
                (site_dir / "metadata.json").write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8"
                )
                rows.append(
                    {
                        "location": str(locations[index]),
                        "location_index": index,
                        "latitude": float(latitudes[index]),
                        "longitude": float(longitudes[index]),
                        "site_key": key,
                        "train_csv": str((site_dir / "train.csv").resolve()),
                        "validation_csv": (
                            None if validation_path is None else str(validation_path.resolve())
                        ),
                        "test_csv": str((site_dir / "test.csv").resolve()),
                        "seasonal_normalization": seasonal_normalization,
                    }
                )
    finally:
        for ds in year_map.values():
            ds.close()
    manifest = pd.DataFrame(rows)
    manifest.to_csv(root / f"manifest_shard_{shard_index:04d}.csv", index=False)
    return manifest
