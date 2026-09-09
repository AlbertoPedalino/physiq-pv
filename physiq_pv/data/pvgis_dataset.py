"""PVGIS-only dataset builders for the ST-GNN forecasting run."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pvlib
import torch
import xarray as xr
from torch.utils.data import Dataset

from physiq_pv.data.pvgis_irradiance import (
    DIFFUSE_TILTED_VAR,
    DIRECT_TILTED_VAR,
    with_effective_poa,
)
from physiq_pv.model.st_gnn import STGNN

PVGIS_STGNN_FEATURES: List[str] = [
    "temperature_2m",
    "solar_irradiance_poa",
    "wind_speed_10m",
    "sin_elev",
    "cos_elev",
    "kt_poa",
    "kt_poa_std_3h",
    "dpoa_dt",
    "direct_irradiance_tilted",
    "diffuse_irradiance_tilted",
    "pv_lag_pvgis",
]
N_FEATURES = len(PVGIS_STGNN_FEATURES)
DEFAULT_TARGET_VARIABLE = "pv_power_output"
DAYTIME_IRRADIANCE_THRESHOLD_WM2 = 10.0
_REQUIRED_VARS = ["temperature_2m", "wind_speed_10m"]

# --------------------------------------------------------------------------- #
# Feature-set ablation
# --------------------------------------------------------------------------- #
# Each set is a subset of PVGIS_STGNN_FEATURES. resolve_feature_set() always
# returns the selected names in the canonical PVGIS_STGNN_FEATURES order so the
# channel subsetting in build_datasets stays consistent. n_features is then
# len(selected) — never hardcode 11 when a feature set is in play.
FEATURE_SETS: Dict[str, List[str]] = {
    "full": list(PVGIS_STGNN_FEATURES),
    "no_pv_lag": [f for f in PVGIS_STGNN_FEATURES if f != "pv_lag_pvgis"],
    "meteo_only": [
        "temperature_2m", "solar_irradiance_poa", "wind_speed_10m",
        "sin_elev", "cos_elev",
    ],
    "irradiance_only": [
        "solar_irradiance_poa", "sin_elev", "cos_elev",
        "kt_poa", "kt_poa_std_3h", "dpoa_dt",
        "direct_irradiance_tilted", "diffuse_irradiance_tilted",
    ],
    "no_derived_irradiance": [
        "temperature_2m", "solar_irradiance_poa", "wind_speed_10m",
        "sin_elev", "cos_elev", "pv_lag_pvgis",
    ],
}


def resolve_feature_set(name: str) -> List[str]:
    """Return the selected feature names in canonical PVGIS_STGNN_FEATURES order."""
    if name not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature_set '{name}'. Available: {sorted(FEATURE_SETS)}."
        )
    selected = set(FEATURE_SETS[name])
    return [f for f in PVGIS_STGNN_FEATURES if f in selected]

SPECIFIC_ANOMALY_LABELS = [
    "unusually_low_solar_potential",
    "unusually_high_solar_potential",
    "extreme_temperature_condition",
    "extreme_wind_condition",
]
GROUP_NORMAL = "normal"
GROUP_RARE = "rare_or_extreme"
DEFAULT_EVENT_SPATIAL_QUANTILE = 0.99
DEFAULT_EVENT_TAIL_QUANTILE = 0.975
DEFAULT_DETECTOR_REGIONAL_QUANTILE = 0.975
DEFAULT_DETECTOR_MIN_TEMPORAL_COVERAGE = 0.95
ANOMALY_SOURCES = ("climatology", "detector")
SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")
MONTH_TO_SEASON = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}
# kt_poa_std_3h at the first input step depends on the two preceding timestamps.
DERIVED_FEATURE_LOOKBACK_HOURS = 2


def _validate_event_quantile(value: float, *, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.5 < value < 1.0:
        raise ValueError(f"{name} must be finite and in (0.5, 1.0), got {value}.")
    return value


def _validate_unit_fraction(value: float, *, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1], got {value}.")
    return value


def _normalise_event_times(values: pd.Series) -> pd.Series:
    times = pd.to_datetime(values)
    if times.dt.tz is not None:
        times = times.dt.tz_convert("UTC").dt.tz_localize(None)
    return times


def build_detector_event_protocol(
    detector_scores: pd.DataFrame,
    times_by_year: Dict[int, pd.DatetimeIndex],
    loc_ids: np.ndarray,
    *,
    regional_quantile: float = DEFAULT_DETECTOR_REGIONAL_QUANTILE,
    seasonal_thresholds: Optional[Dict[str, float]] = None,
    min_temporal_coverage: float = DEFAULT_DETECTOR_MIN_TEMPORAL_COVERAGE,
) -> dict:
    """Aggregate detector decisions into graph-wide normal/rare events.

    Detector thresholds are already calibrated by MTGFlow/CATCH/M2AD.  This
    adapter consumes their boolean ``is_anomaly`` decisions directly at node
    level.  It then computes the anomalous-node fraction at every timestamp.
    Training data fit one seasonal quantile of that regional fraction; frozen
    thresholds are passed back through ``seasonal_thresholds`` for test data.

    Dense hourly scores are required for the forecasting protocol.  Small
    boundary gaps caused by a detector lookback window are allowed, but a
    sparse score stride is rejected through ``min_temporal_coverage``.
    """
    regional_quantile = _validate_event_quantile(
        regional_quantile, name="detector regional quantile"
    )
    min_temporal_coverage = _validate_unit_fraction(
        min_temporal_coverage, name="detector minimum temporal coverage"
    )
    required = {"location", "timestamp", "is_anomaly"}
    missing = required - set(detector_scores.columns)
    if missing:
        raise ValueError(
            "Detector event filtering requires columns "
            f"{sorted(required)}; missing {sorted(missing)}."
        )
    if detector_scores.empty:
        raise ValueError("Detector event filtering requires non-empty scores.")

    optional = [
        name
        for name in ("anomaly_score", "threshold", "detector", "method", "seed")
        if name in detector_scores.columns
    ]
    scores = detector_scores[[*required, *optional]].copy()
    scores["location"] = scores["location"].astype(str)
    scores["timestamp"] = _normalise_event_times(scores["timestamp"])
    if scores[["location", "timestamp"]].duplicated().any():
        raise ValueError(
            "Detector scores must contain one decision per location/timestamp. "
            "Select one seed before using an ensemble output."
        )
    if scores["is_anomaly"].isna().any():
        raise ValueError("Detector scores contain missing is_anomaly decisions.")
    if not pd.api.types.is_bool_dtype(scores["is_anomaly"]):
        text = scores["is_anomaly"].astype(str).str.strip().str.lower()
        mapping = {"true": True, "false": False, "1": True, "0": False}
        invalid = ~text.isin(mapping)
        if invalid.any():
            examples = sorted(text[invalid].unique().tolist())[:5]
            raise ValueError(
                "Detector is_anomaly must be boolean; invalid values "
                f"include {examples}."
            )
        scores["is_anomaly"] = text.map(mapping).astype(bool)

    loc_text = np.asarray([str(value) for value in loc_ids])
    if len(np.unique(loc_text)) != len(loc_text):
        raise ValueError("Detector aggregation requires unique graph location IDs.")
    known_locations = set(loc_text)
    scores = scores[scores["location"].isin(known_locations)].copy()
    if scores.empty:
        raise ValueError("No detector rows match the supplied graph locations.")

    detector_names = []
    for column in ("detector", "method"):
        if column in scores:
            detector_names.extend(scores[column].dropna().astype(str).unique().tolist())
    detector_names = sorted(set(detector_names))
    detector_name = detector_names[0] if len(detector_names) == 1 else "detector"

    rare_by_year: Dict[int, np.ndarray] = {}
    labels_by_year: Dict[int, pd.DataFrame] = {}
    severity_by_year: Dict[int, pd.DataFrame] = {}
    coverage_by_year: Dict[int, dict] = {}
    matched_rows = 0
    calibration_rows = []

    for year, raw_times in times_by_year.items():
        times = pd.DatetimeIndex(raw_times)
        if times.tz is not None:
            times = times.tz_convert("UTC").tz_localize(None)
        validate_hourly_grid(times, label=f"PVGIS detector event grid {year}")
        year_scores = scores[
            (scores["timestamp"] >= times[0])
            & (scores["timestamp"] <= times[-1])
        ].copy()
        year_scores = year_scores[year_scores["timestamp"].isin(times)]
        matched_rows += len(year_scores)

        grouped = year_scores.groupby("timestamp", sort=False)["is_anomaly"].agg(
            n_scored="size", n_anomaly="sum"
        )
        n_scored = grouped["n_scored"].reindex(times, fill_value=0).to_numpy(int)
        n_anomaly = grouped["n_anomaly"].reindex(times, fill_value=0).to_numpy(int)
        spatial_coverage = n_scored / float(len(loc_text))
        scored_timestamps = n_scored > 0
        temporal_coverage = float(scored_timestamps.mean())
        if temporal_coverage < min_temporal_coverage:
            raise ValueError(
                f"Detector scores cover only {temporal_coverage:.3%} of hourly "
                f"timestamps in {year}; require >= {min_temporal_coverage:.3%}. "
                "Use dense detector scoring (for MTGFlow: --score-stride 1)."
            )
        if scored_timestamps.any() and float(spatial_coverage[scored_timestamps].min()) < 0.99:
            raise ValueError(
                f"Detector spatial coverage falls below 99% in {year}; the CSV "
                "must score the same PVGIS locations used by the graph."
            )

        anomaly_fraction = n_anomaly / float(len(loc_text))
        severity = pd.DataFrame(
            {
                "anomaly_fraction": anomaly_fraction,
                "spatial_coverage": spatial_coverage,
                "is_scored": scored_timestamps,
                "season": [MONTH_TO_SEASON[month] for month in times.month],
            },
            index=times,
        )
        severity_by_year[int(year)] = severity
        calibration_rows.append(
            severity.loc[severity["is_scored"], ["season", "anomaly_fraction"]]
        )
        coverage_by_year[int(year)] = {
            "temporal": temporal_coverage,
            "minimum_spatial_on_scored_timestamps": (
                float(spatial_coverage[scored_timestamps].min())
                if scored_timestamps.any()
                else 0.0
            ),
            "unscored_timestamps": int((~scored_timestamps).sum()),
        }

    if matched_rows == 0:
        raise ValueError(
            "No detector rows matched the supplied PVGIS timestamps/locations."
        )
    required_seasons = {
        season
        for severity in severity_by_year.values()
        for season in severity["season"].unique()
    }
    fitted_on_input = seasonal_thresholds is None
    if fitted_on_input:
        calibration = pd.concat(calibration_rows, ignore_index=True)
        thresholds = (
            calibration.groupby("season", observed=True)["anomaly_fraction"]
            .quantile(regional_quantile)
            .to_dict()
        )
    else:
        thresholds = {str(key): float(value) for key, value in seasonal_thresholds.items()}
    missing_seasons = required_seasons - set(thresholds)
    if missing_seasons:
        raise ValueError(
            "Detector seasonal thresholds are missing "
            f"{sorted(missing_seasons)}."
        )
    invalid_thresholds = {
        season: value
        for season, value in thresholds.items()
        if not np.isfinite(value) or not 0.0 <= value <= 1.0
    }
    if invalid_thresholds:
        raise ValueError(
            "Detector seasonal thresholds must be finite fractions in [0, 1]; "
            f"invalid values: {invalid_thresholds}."
        )
    thresholds = {
        season: float(thresholds[season])
        for season in SEASON_ORDER
        if season in thresholds
    }

    for year, severity in severity_by_year.items():
        threshold_at_time = severity["season"].map(thresholds).to_numpy(float)
        anomaly_fraction = severity["anomaly_fraction"].to_numpy(float)
        scored = severity["is_scored"].to_numpy(bool)
        # Match the audited >= P97.5 rule.  A zero quantile is treated as
        # "at least one anomalous node" so a zero-valued normal timestamp is
        # never labelled rare merely because the detector is sparse.
        rare = scored & np.where(
            threshold_at_time > 0.0,
            anomaly_fraction >= threshold_at_time,
            anomaly_fraction > 0.0,
        )
        denominator = np.maximum(threshold_at_time, 1.0 / float(len(loc_text)))
        event_score = anomaly_fraction / denominator
        rare_by_year[year] = rare
        labels_by_year[year] = pd.DataFrame(
            {
                "timestamp": severity.index,
                "event_group": np.where(rare, GROUP_RARE, GROUP_NORMAL),
                "event_score": event_score,
                "event_driver": detector_name,
            }
        )

    return {
        "source": "detector",
        "detector": detector_name,
        "spatial_quantile": None,
        "event_quantile": regional_quantile,
        "regional_quantile": regional_quantile,
        "seasonal_thresholds": thresholds,
        "thresholds_fitted_on_input": fitted_on_input,
        "min_temporal_coverage": min_temporal_coverage,
        "variables": [detector_name],
        "inactive_variables": [],
        "thresholds": {
            "anomalous_location_fraction_by_season": thresholds,
        },
        "severity_by_year": severity_by_year,
        "rare_by_year": rare_by_year,
        "labels_by_year": labels_by_year,
        "coverage_by_year": coverage_by_year,
        "matched_score_rows": matched_rows,
    }


def build_regional_event_protocol(
    anomaly_scores: pd.DataFrame,
    times_by_year: Dict[int, pd.DatetimeIndex],
    loc_ids: np.ndarray,
    *,
    fit_years: Optional[List[int]] = None,
    spatial_quantile: float = DEFAULT_EVENT_SPATIAL_QUANTILE,
    event_quantile: float = DEFAULT_EVENT_TAIL_QUANTILE,
    thresholds: Optional[Dict[str, float]] = None,
) -> dict:
    """Aggregate local climatology anomalies into graph-wide event labels.

    Each timestamp is the analogue of one full precipitation map in Monaco
    et al.: local anomaly scores are zero-filled for unflagged nodes, reduced
    with a robust spatial upper quantile per variable, and then compared with
    a temporal threshold fitted on the effective training years only.

    The returned labels are evaluation/filtering metadata. They are never
    exposed as model features or prediction targets.
    """
    spatial_quantile = _validate_event_quantile(
        spatial_quantile, name="event spatial quantile"
    )
    event_quantile = _validate_event_quantile(
        event_quantile, name="event tail quantile"
    )
    required = {"location", "timestamp", "variable", "anomaly_score"}
    missing = required - set(anomaly_scores.columns)
    if missing:
        raise ValueError(
            "Event-level filtering requires anomaly score columns "
            f"{sorted(required)}; missing {sorted(missing)}."
        )
    if anomaly_scores.empty and thresholds is None:
        raise ValueError("Event-level filtering requires non-empty anomaly scores.")

    scores = anomaly_scores[list(required)].copy()
    scores["location"] = scores["location"].astype(str)
    scores["timestamp"] = pd.to_datetime(scores["timestamp"])
    if scores["timestamp"].dt.tz is not None:
        scores["timestamp"] = (
            scores["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
        )
    scores["variable"] = scores["variable"].astype(str)
    scores["anomaly_score"] = pd.to_numeric(
        scores["anomaly_score"], errors="coerce"
    )
    if not np.isfinite(scores["anomaly_score"].to_numpy(dtype=float)).all():
        raise ValueError("Anomaly scores contain non-finite anomaly_score values.")

    loc_text = np.asarray([str(value) for value in loc_ids])
    loc_position = {value: i for i, value in enumerate(loc_text)}
    if len(loc_position) != len(loc_text):
        raise ValueError("Regional event aggregation requires unique location IDs.")

    if thresholds is None:
        if not fit_years:
            raise ValueError("fit_years are required when event thresholds are fitted.")
        fit_scores = scores[scores["timestamp"].dt.year.isin(fit_years)]
        variables = sorted(fit_scores["variable"].unique().tolist())
    else:
        variables = sorted(str(name) for name in thresholds)
    if not variables:
        raise ValueError("No anomaly variables are available for event aggregation.")

    severity_by_year: Dict[int, pd.DataFrame] = {}
    matched_rows = 0
    for year, raw_times in times_by_year.items():
        times = pd.DatetimeIndex(raw_times)
        if times.tz is not None:
            times = times.tz_convert("UTC").tz_localize(None)
        validate_hourly_grid(times, label=f"PVGIS event grid {year}")
        year_scores = scores[
            (scores["timestamp"] >= times[0])
            & (scores["timestamp"] <= times[-1])
        ]
        frame = pd.DataFrame(0.0, index=times, columns=variables, dtype=np.float32)
        for variable in variables:
            rows = year_scores[year_scores["variable"] == variable]
            if rows.empty:
                continue
            time_pos = times.get_indexer(pd.DatetimeIndex(rows["timestamp"]))
            loc_pos = rows["location"].map(loc_position).fillna(-1).to_numpy(dtype=np.int64)
            valid = (time_pos >= 0) & (loc_pos >= 0)
            matched_rows += int(valid.sum())
            if not bool(valid.any()):
                continue
            local = np.zeros((len(times), len(loc_ids)), dtype=np.float32)
            np.maximum.at(
                local,
                (time_pos[valid], loc_pos[valid]),
                np.abs(rows["anomaly_score"].to_numpy(dtype=np.float32)[valid]),
            )
            frame[variable] = np.quantile(
                local, spatial_quantile, axis=1
            ).astype(np.float32)
        severity_by_year[int(year)] = frame

    if matched_rows == 0 and thresholds is None:
        raise ValueError(
            "No anomaly-score rows matched the supplied PVGIS timestamps/locations."
        )

    inactive_variables: List[str] = []
    if thresholds is None:
        missing_years = sorted(set(fit_years) - set(severity_by_year))
        if missing_years:
            raise ValueError(
                f"Event threshold fit years are missing from the data: {missing_years}."
            )
        fitted_thresholds: Dict[str, float] = {}
        for variable in variables:
            values = np.concatenate(
                [
                    severity_by_year[year][variable].to_numpy(dtype=float)
                    for year in fit_years
                ]
            )
            threshold = float(np.quantile(values, event_quantile))
            if not np.isfinite(threshold):
                raise ValueError(
                    f"Regional event threshold for {variable!r} is non-finite."
                )
            if threshold <= 0.0:
                inactive_variables.append(variable)
                continue
            fitted_thresholds[variable] = threshold
        if not fitted_thresholds:
            raise ValueError(
                "No variable has a positive regional event threshold; lower "
                "event_spatial_quantile or inspect the anomaly scores."
            )
        variables = sorted(fitted_thresholds)
    else:
        fitted_thresholds = {str(k): float(v) for k, v in thresholds.items()}
        invalid = {
            key: value
            for key, value in fitted_thresholds.items()
            if not np.isfinite(value) or value <= 0.0
        }
        if invalid:
            raise ValueError(f"Event thresholds must be finite and positive: {invalid}.")

    rare_by_year: Dict[int, np.ndarray] = {}
    labels_by_year: Dict[int, pd.DataFrame] = {}
    for year, severity in severity_by_year.items():
        threshold_row = np.asarray(
            [fitted_thresholds[name] for name in variables], dtype=np.float64
        )
        relative = severity[variables].to_numpy(dtype=np.float64) / threshold_row
        driver_idx = np.argmax(relative, axis=1)
        event_score = relative[np.arange(len(relative)), driver_idx]
        rare = np.any(relative > 1.0, axis=1)
        rare_by_year[year] = rare
        labels_by_year[year] = pd.DataFrame(
            {
                "timestamp": severity.index,
                "event_group": np.where(rare, GROUP_RARE, GROUP_NORMAL),
                "event_score": event_score,
                "event_driver": np.asarray(variables, dtype=object)[driver_idx],
            }
        )

    return {
        "source": "climatology",
        "detector": None,
        "spatial_quantile": spatial_quantile,
        "event_quantile": event_quantile,
        "variables": variables,
        "inactive_variables": inactive_variables,
        "thresholds": fitted_thresholds,
        "severity_by_year": severity_by_year,
        "rare_by_year": rare_by_year,
        "labels_by_year": labels_by_year,
        "matched_score_rows": matched_rows,
    }


def normal_event_window_mask(
    rare_at_time: np.ndarray,
    *,
    seq_len: int,
    horizon: int,
    forecast_horizons: Optional[tuple[int, ...]] = None,
    feature_lookback: int = DERIVED_FEATURE_LOOKBACK_HOURS,
) -> np.ndarray:
    """Return windows with normal feature context, input history and target."""
    rare = np.asarray(rare_at_time, dtype=bool)
    horizons = _normalise_forecast_horizons(horizon, forecast_horizons)
    if feature_lookback < 0:
        raise ValueError("feature_lookback must be >= 0.")
    n_windows = len(rare) - seq_len - max(horizons) + 1
    if n_windows <= 0:
        return np.zeros(0, dtype=bool)
    starts = np.arange(n_windows, dtype=np.int64)
    prefix = np.concatenate(([0], np.cumsum(rare, dtype=np.int64)))
    context_starts = np.maximum(starts - feature_lookback, 0)
    history_rare = prefix[starts + seq_len] > prefix[context_starts]
    target_rare = np.column_stack([
        rare[starts + seq_len + target_horizon - 1]
        for target_horizon in horizons
    ]).any(axis=1)
    return ~(history_rare | target_rare)


def normal_event_timestamp_mask(
    rare_at_time: np.ndarray,
    *,
    seq_len: int,
    horizon: int,
    forecast_horizons: Optional[tuple[int, ...]] = None,
    feature_lookback: int = DERIVED_FEATURE_LOOKBACK_HOURS,
) -> np.ndarray:
    """Mark raw timestamps actually used by retained normal-only windows."""
    rare = np.asarray(rare_at_time, dtype=bool)
    horizons = _normalise_forecast_horizons(horizon, forecast_horizons)
    max_horizon = max(horizons)
    keep = normal_event_window_mask(
        rare,
        seq_len=seq_len,
        horizon=max_horizon,
        forecast_horizons=horizons,
        feature_lookback=feature_lookback,
    )
    starts = np.flatnonzero(keep)
    used = np.zeros(len(rare), dtype=bool)
    if starts.size == 0:
        return used
    delta = np.zeros(len(rare) + 1, dtype=np.int64)
    np.add.at(delta, starts, 1)
    np.add.at(delta, starts + seq_len, -1)
    used |= np.cumsum(delta[:-1]) > 0
    for target_horizon in horizons:
        used[starts + seq_len + target_horizon - 1] = True
    return used


def _normalise_forecast_horizons(
    horizon: int,
    forecast_horizons: Optional[tuple[int, ...]],
) -> tuple[int, ...]:
    """Return a validated, ordered tuple of direct forecast horizons."""
    values = (int(horizon),) if forecast_horizons is None else tuple(
        int(value) for value in forecast_horizons
    )
    if not values:
        raise ValueError("forecast_horizons must contain at least one horizon.")
    if any(value < 1 for value in values):
        raise ValueError(
            f"Every forecast horizon must be >= 1 hour, got {values}."
        )
    if len(set(values)) != len(values):
        raise ValueError(f"forecast_horizons contains duplicates: {values}.")
    if tuple(sorted(values)) != values:
        raise ValueError(f"forecast_horizons must be strictly increasing: {values}.")
    return values


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_pvgis_year(path: str) -> xr.Dataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PVGIS file not found: {p}")
    return xr.open_dataset(p)


def load_pvgis_years(
    pvgis_dir: str, years: List[int], file_template: str = "piedmont_pvgis_{year}.nc"
) -> Dict[int, xr.Dataset]:
    d = Path(pvgis_dir)
    if not d.is_dir():
        raise NotADirectoryError(f"PVGIS directory not found: {d}")
    out: Dict[int, xr.Dataset] = {}
    for year in years:
        fp = d / file_template.format(year=year)
        if not fp.exists():
            print(f"  [skip] missing PVGIS file for {year}: {fp}")
            continue
        out[year] = xr.open_dataset(fp)
        print(f"  [ok]   loaded PVGIS year {year}: {fp.name}")
    if not out:
        raise FileNotFoundError(f"No PVGIS files found in {d} for years {years}.")
    return out


# --------------------------------------------------------------------------- #
# Solar geometry + raw channels (PVGIS-only)
# --------------------------------------------------------------------------- #
def validate_hourly_grid(times: pd.DatetimeIndex, *, label: str) -> None:
    """Require a unique, increasing time axis with exactly one-hour steps."""
    if len(times) < 2:
        raise ValueError(f"{label}: at least two timestamps are required.")
    if times.has_duplicates:
        raise ValueError(f"{label}: duplicate timestamps are not allowed.")
    if not times.is_monotonic_increasing:
        raise ValueError(f"{label}: timestamps must be strictly increasing.")
    deltas = np.diff(
        times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    )
    expected = pd.Timedelta(hours=1).value
    bad = np.flatnonzero(deltas != expected)
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"{label}: non-hourly step between {times[i]} and {times[i + 1]} "
            f"({pd.Timedelta(int(deltas[i]), unit='ns')})."
        )


def _solar_geometry_and_clearsky_poa(
    times: pd.DatetimeIndex,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    surface_tilt: float,
    surface_azimuth: float,
) -> tuple:
    """Return solar elevation channels and clear-sky inclined POA [kW/m²]."""
    T, N = len(times), len(lats)
    times_utc = times.tz_localize("UTC") if times.tzinfo is None else times
    sin_elev = np.zeros((T, N), dtype=np.float32)
    cos_elev = np.zeros((T, N), dtype=np.float32)
    poa_cs = np.zeros((T, N), dtype=np.float32)
    fleet_lat, fleet_lon = float(np.nanmean(lats)), float(np.nanmean(lons))
    for p in range(N):
        lat_p = float(lats[p]) if np.isfinite(lats[p]) else fleet_lat
        lon_p = float(lons[p]) if np.isfinite(lons[p]) else fleet_lon
        loc = pvlib.location.Location(lat_p, lon_p, tz="UTC")
        sp = loc.get_solarposition(times_utc)
        elev = np.clip(sp["apparent_elevation"].values, 0.0, 90.0).astype(np.float32)
        sin_elev[:, p] = np.sin(np.radians(elev))
        cos_elev[:, p] = np.cos(np.radians(elev))
        try:
            cs = loc.get_clearsky(times_utc, model="ineichen")
        except Exception:
            cs = loc.get_clearsky(times_utc, model="simplified_solis")
        total = pvlib.irradiance.get_total_irradiance(
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            solar_zenith=sp["apparent_zenith"].to_numpy(),
            solar_azimuth=sp["azimuth"].to_numpy(),
            dni=cs["dni"].to_numpy(),
            ghi=cs["ghi"].to_numpy(),
            dhi=cs["dhi"].to_numpy(),
            albedo=0.0,
        )
        poa_cs[:, p] = np.clip(
            np.nan_to_num(np.asarray(total["poa_global"]), nan=0.0) / 1000.0,
            0.0,
            None,
        )
    return sin_elev, cos_elev, poa_cs


def build_year_raw(
    ds: xr.Dataset,
    target_variable: str,
    loc_dim: str = "location",
    kt_poa_max: float = 1.6,
) -> dict:
    """Build raw (un-normalised) per-(time, location) channels for one PVGIS year."""
    ds = with_effective_poa(ds)
    if not np.isfinite(kt_poa_max) or kt_poa_max <= 0:
        raise ValueError("kt_poa_max must be finite and positive.")
    for v in _REQUIRED_VARS + [target_variable]:
        if v not in ds:
            raise ValueError(f"Required PVGIS variable '{v}' missing from dataset.")

    times = pd.DatetimeIndex(ds["time"].values)
    validate_hourly_grid(times, label="PVGIS year")
    lats = np.asarray(ds["lat"].values, dtype=float)
    lons = np.asarray(ds["lon"].values, dtype=float)
    if not np.isfinite(lats).all() or not np.isfinite(lons).all():
        raise ValueError("PVGIS locations require finite latitude/longitude.")

    def col(v):  # (T, N)
        return np.asarray(ds[v].transpose(loc_dim, "time").values, dtype=np.float32).T

    temp = col("temperature_2m")
    solar_wm2 = col("solar_irradiance_poa")
    direct_wm2 = col(DIRECT_TILTED_VAR)
    diffuse_wm2 = col(DIFFUSE_TILTED_VAR)
    wind = col("wind_speed_10m")
    pv = col(target_variable)
    for name, values in (
        ("temperature_2m", temp),
        ("solar_irradiance_poa", solar_wm2),
        (DIRECT_TILTED_VAR, direct_wm2),
        (DIFFUSE_TILTED_VAR, diffuse_wm2),
        ("wind_speed_10m", wind),
        (target_variable, pv),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"PVGIS variable {name!r} contains non-finite values.")
    solar_kwm2 = np.clip(solar_wm2 / 1000.0, 0.0, None)

    surface_tilt = float(
        ds.attrs.get("tilt_angle", ds.attrs.get("pvgis_tilt_angle", 30.0))
    )
    surface_azimuth = float(
        ds.attrs.get("azimuth_angle", ds.attrs.get("pvgis_azimuth_angle", 180.0))
    )
    sin_elev, cos_elev, poa_cs = _solar_geometry_and_clearsky_poa(
        times,
        lats,
        lons,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
    )
    day = (sin_elev > 0.05) & (solar_kwm2 > 0.03)

    kt_poa = np.where(poa_cs > 0.1, solar_kwm2 / (poa_cs + 1e-6), 0.0)
    kt_poa = np.clip(kt_poa, 0.0, kt_poa_max).astype(np.float32)

    kt_poa_std = np.zeros_like(kt_poa)
    for p in range(kt_poa.shape[1]):
        kt_poa_std[:, p] = (
            pd.Series(kt_poa[:, p])
            .rolling(3, min_periods=1)
            .std()
            .fillna(0.0)
            .to_numpy()
        )

    dpoa = np.zeros_like(solar_kwm2)
    dpoa[1:, :] = solar_kwm2[1:, :] - solar_kwm2[:-1, :]
    direct_tilted = np.clip(direct_wm2 / 1000.0, 0.0, None).astype(np.float32)
    diffuse_tilted = np.clip(diffuse_wm2 / 1000.0, 0.0, None).astype(np.float32)

    return {
        "times": times,
        "lats": lats,
        "lons": lons,
        "temp": temp,
        "solar_wm2": solar_wm2,
        "wind": wind,
        "sin": sin_elev,
        "cos": cos_elev,
        "poa_cs": poa_cs,
        "kt_poa": kt_poa,
        "kt_poa_std": kt_poa_std,
        "dpoa": dpoa,
        "direct_tilted": direct_tilted,
        "diffuse_tilted": diffuse_tilted,
        "pv": pv,
        "day": day,
    }


def fit_normalization(
    train_raws: List[dict],
    time_masks: Optional[List[np.ndarray]] = None,
) -> dict:
    """Per-location pv_scale (p99 daytime pv) and global z-score stats from TRAIN only."""
    if time_masks is None:
        time_masks = [
            np.ones(raw["pv"].shape[0], dtype=bool) for raw in train_raws
        ]
    if len(time_masks) != len(train_raws):
        raise ValueError("time_masks must have one entry per training raw dataset.")
    checked_masks: List[np.ndarray] = []
    for raw, mask in zip(train_raws, time_masks):
        checked = np.asarray(mask, dtype=bool)
        if checked.shape != (raw["pv"].shape[0],):
            raise ValueError(
                "Each normalization time mask must match its raw time dimension."
            )
        checked_masks.append(checked)
    if not any(bool(mask.any()) for mask in checked_masks):
        raise ValueError("No normal training timestamps remain for normalization.")

    n_loc = train_raws[0]["pv"].shape[1]
    # per-location p99 of daytime, positive pv (target scale)
    pv_stack = np.concatenate(
        [raw["pv"][mask] for raw, mask in zip(train_raws, checked_masks)], axis=0
    )
    day_stack = np.concatenate(
        [raw["day"][mask] for raw, mask in zip(train_raws, checked_masks)], axis=0
    )
    pv_scale = np.full(n_loc, np.nan, dtype=np.float64)
    for p in range(n_loc):
        vals = pv_stack[day_stack[:, p], p]
        vals = vals[vals > 0]
        if len(vals) > 10:
            pv_scale[p] = float(np.percentile(vals, 99)) + 1e-6
    fitted = np.isfinite(pv_scale) & (pv_scale > 0)
    if not fitted.any():
        raise ValueError("Training data has no usable positive daytime PV values.")
    fallback_scale = float(np.median(pv_scale[fitted]))
    pv_scale[~fitted] = fallback_scale

    def gstats(key):
        arr = np.concatenate(
            [raw[key][mask] for raw, mask in zip(train_raws, checked_masks)],
            axis=0,
        )
        return float(np.nanmean(arr)), float(np.nanstd(arr) + 1e-6)

    z = {k: gstats(k) for k in ("temp", "solar_wm2", "wind", "dpoa")}
    return {
        "pv_scale": pv_scale,
        "z": z,
        "pv_scale_fallback": fallback_scale,
        "pv_scale_fallback_count": int((~fitted).sum()),
        "fit_timestamp_count": int(sum(mask.sum() for mask in checked_masks)),
    }


def assemble_feats(
    raw: dict,
    norm: dict,
    pv_target_clip_max: Optional[float] = None,
) -> tuple:
    """Return (feats (T,N,11), pv_norm (T,N), pv_raw (T,N)) using train normalisation."""
    z = norm["z"]
    pv_scale = norm["pv_scale"][None, :]

    def zc(key):
        mu, sd = z[key]
        return (raw[key] - mu) / sd

    pv_scaled = raw["pv"] / pv_scale
    if pv_target_clip_max is None:
        pv_norm = np.maximum(pv_scaled, 0.0).astype(np.float32)
    else:
        pv_norm = np.clip(
            pv_scaled, 0.0, pv_target_clip_max
        ).astype(np.float32)
    channels = [
        zc("temp"),
        zc("solar_wm2"),
        zc("wind"),
        raw["sin"],
        raw["cos"],
        raw["kt_poa"],
        raw["kt_poa_std"],
        zc("dpoa"),
        raw["direct_tilted"],
        raw["diffuse_tilted"],
        pv_norm,  # pv_lag_pvgis (causal lag once sliced inside the window)
    ]
    feats = np.stack(channels, axis=-1).astype(np.float32)  # (T, N, 11)
    return feats, pv_norm, raw["pv"].astype(np.float32)


# --------------------------------------------------------------------------- #
# Window dataset
# --------------------------------------------------------------------------- #
class PVGISWindowDataset(Dataset):
    """
    Sliding windows over one or more PVGIS years (no cross-year windows).

    __getitem__ -> (x, y_norm, k) with
        x      (N, seq_len, n_features)
        y_norm (N,) or (N, H)             normalised direct PV target(s)
        k      int                        global sample index (for eval lookups)
    """

    def __init__(
        self,
        feats_by_year: Dict[int, np.ndarray],
        pvnorm_by_year: Dict[int, np.ndarray],
        pvraw_by_year: Dict[int, np.ndarray],
        solarraw_by_year: Optional[Dict[int, np.ndarray]],
        times_by_year: Dict[int, pd.DatetimeIndex],
        seq_len: int,
        horizon: int,
        pv_scale: np.ndarray,
        loc_ids: np.ndarray,
        kt_poa_by_year: Optional[Dict[int, np.ndarray]] = None,
        event_rare_by_year: Optional[Dict[int, np.ndarray]] = None,
        forecast_horizons: Optional[tuple[int, ...]] = None,
    ):
        self.feats_by_year = feats_by_year
        self.times_by_year = times_by_year
        self.seq_len = seq_len
        self.forecast_horizons = _normalise_forecast_horizons(
            horizon, forecast_horizons
        )
        self.horizon = max(self.forecast_horizons)
        self.n_horizons = len(self.forecast_horizons)
        self.pv_scale = pv_scale.astype(np.float32)
        self.loc_ids = loc_ids
        self.n_nodes = len(loc_ids)

        samples: List[tuple] = []
        y_norm_rows: List[np.ndarray] = []
        y_true_rows: List[np.ndarray] = []
        solar_target_rows: List[np.ndarray] = []
        kt_target_rows: List[np.ndarray] = []
        times_rows: List[np.ndarray] = []
        issue_times_rows: List[np.datetime64] = []
        event_target_rows: List[bool] = []
        event_history_rows: List[bool] = []
        for year, feats in feats_by_year.items():
            T = feats.shape[0]
            n_windows = T - seq_len - self.horizon + 1
            if n_windows <= 0:
                continue
            target_offsets = np.asarray(
                [seq_len + value - 1 for value in self.forecast_horizons],
                dtype=np.int64,
            )
            pvn = pvnorm_by_year[year]
            pvr = pvraw_by_year[year]
            solar = solarraw_by_year[year] if solarraw_by_year is not None else None
            kt_poa = (
                kt_poa_by_year[year] if kt_poa_by_year is not None else None
            )
            ts = times_by_year[year]
            event_rare = (
                np.asarray(event_rare_by_year[year], dtype=bool)
                if event_rare_by_year is not None
                else np.zeros(T, dtype=bool)
            )
            if event_rare.shape != (T,):
                raise ValueError(
                    f"Event labels for {year} have shape {event_rare.shape}, "
                    f"expected {(T,)}."
                )
            rare_prefix = np.concatenate(
                ([0], np.cumsum(event_rare, dtype=np.int64))
            )
            for i in range(n_windows):
                samples.append((year, i))
                target_positions = i + target_offsets
                # Arrays are stored node-major: (N, H).  This matches model
                # outputs and avoids copies in the training loop.
                y_norm_rows.append(pvn[target_positions].T)
                y_true_rows.append(pvr[target_positions].T)
                if solar is not None:
                    solar_target_rows.append(solar[target_positions].T)
                if kt_poa is not None:
                    kt_target_rows.append(kt_poa[target_positions].T)
                times_rows.append(ts.values[target_positions])
                issue_times_rows.append(ts.values[i + self.seq_len - 1])
                event_target_rows.append(bool(event_rare[target_positions].any()))
                context_start = max(i - DERIVED_FEATURE_LOOKBACK_HOURS, 0)
                event_history_rows.append(
                    bool(
                        rare_prefix[i + self.seq_len]
                        > rare_prefix[context_start]
                    )
                )
        if not samples:
            raise ValueError("No supervised windows could be built (year too short?).")

        self.samples = samples
        self.y_norm_all = np.stack(y_norm_rows)  # (samples, N, H)
        self.y_true_all = np.stack(y_true_rows)  # (samples, N, H)
        self.solar_irradiance_poa_target_all = (
            np.stack(solar_target_rows) if solar_target_rows else None
        )
        # Target-time clear-sky index (kt): supervision target for the optional
        # auxiliary irradiance loss (use_irradiance_loss in train_model).
        self.kt_poa_target_all = (
            np.stack(kt_target_rows) if kt_target_rows else None
        )
        target_times = np.stack(times_rows)  # (samples, H)
        self.issue_time_all = pd.DatetimeIndex(issue_times_rows)
        if self.n_horizons == 1:
            # Preserve the established single-horizon public interface.
            self.y_norm_all = self.y_norm_all[..., 0]
            self.y_true_all = self.y_true_all[..., 0]
            if self.solar_irradiance_poa_target_all is not None:
                self.solar_irradiance_poa_target_all = (
                    self.solar_irradiance_poa_target_all[..., 0]
                )
            if self.kt_poa_target_all is not None:
                self.kt_poa_target_all = self.kt_poa_target_all[..., 0]
            self.target_time_all = pd.DatetimeIndex(target_times[:, 0])
        else:
            self.target_time_all = target_times
        self.event_rare_target_all = np.asarray(event_target_rows, dtype=bool)
        self.event_rare_history_all = np.asarray(event_history_rows, dtype=bool)
        self.event_labels_attached = event_rare_by_year is not None
        self.event_filter_applied = False
        # Target and input-history anomaly masks for normal-only training.
        self.anomaly_mask_all: Optional[np.ndarray] = None
        self.anomaly_history_mask_all: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, k: int):
        year, i = self.samples[k]
        win = self.feats_by_year[year][i : i + self.seq_len]  # (seq_len, N, C)
        x = torch.from_numpy(np.ascontiguousarray(win.transpose(1, 0, 2)))  # (N, seq_len, C)
        y = torch.from_numpy(self.y_norm_all[k])  # (N,)
        return x, y, k

    def _select_indices(self, keep: np.ndarray) -> None:
        self.samples = [self.samples[i] for i in keep]
        self.y_norm_all = self.y_norm_all[keep]
        self.y_true_all = self.y_true_all[keep]
        if self.solar_irradiance_poa_target_all is not None:
            self.solar_irradiance_poa_target_all = (
                self.solar_irradiance_poa_target_all[keep]
            )
        if self.kt_poa_target_all is not None:
            self.kt_poa_target_all = self.kt_poa_target_all[keep]
        if self.anomaly_mask_all is not None:
            self.anomaly_mask_all = self.anomaly_mask_all[keep]
        if self.anomaly_history_mask_all is not None:
            self.anomaly_history_mask_all = self.anomaly_history_mask_all[keep]
        self.event_rare_target_all = self.event_rare_target_all[keep]
        self.event_rare_history_all = self.event_rare_history_all[keep]
        self.target_time_all = self.target_time_all[keep]

    def subsample(self, max_samples: Optional[int], seed: int = 0) -> "PVGISWindowDataset":
        """Randomly keep at most `max_samples` windows (in place); returns self."""
        if max_samples is None or len(self.samples) <= max_samples:
            return self
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(self.samples), size=max_samples, replace=False))
        self._select_indices(keep)
        return self

    def drop_rare_event_windows(self) -> dict:
        """Physically remove windows with a rare history or rare target event."""
        if not self.event_labels_attached:
            raise ValueError(
                "Event-level filtering requires regional event labels on the dataset."
            )
        before = len(self.samples)
        target_rare = int(self.event_rare_target_all.sum())
        history_rare = int(self.event_rare_history_all.sum())
        keep = ~(self.event_rare_target_all | self.event_rare_history_all)
        if not bool(keep.any()):
            raise ValueError("Event-level filtering removed every supervised window.")
        self._select_indices(np.flatnonzero(keep))
        self.event_filter_applied = True
        return {
            "before": before,
            "after": len(self.samples),
            "dropped": int(before - len(self.samples)),
            "target_rare_windows": target_rare,
            "history_rare_windows": history_rare,
        }

    def attach_anomaly_mask(self, anomaly_scores: Optional[pd.DataFrame]) -> int:
        """Attach target and input-history anomaly masks; return target count."""
        n_samples = len(self.samples)
        mask = np.zeros((n_samples, self.n_nodes), dtype=bool)
        history_mask = np.zeros((n_samples, self.n_nodes), dtype=bool)
        self.anomaly_mask_all = mask
        self.anomaly_history_mask_all = history_mask
        if anomaly_scores is None or anomaly_scores.empty:
            return 0
        scores = anomaly_scores[["location", "timestamp"]].copy()
        scores["location"] = scores["location"].astype(str)
        ts_int = pd.to_datetime(scores["timestamp"]).to_numpy("datetime64[ns]").astype("int64")
        scores["ts_int"] = ts_int
        by_loc = scores.groupby("location")["ts_int"].apply(
            lambda s: np.unique(s.to_numpy())
        ).to_dict()
        target_values = np.asarray(self.target_time_all, dtype="datetime64[ns]")
        if target_values.ndim == 1:
            target_values = target_values[:, None]
        target_int = target_values.astype("int64")
        sample_years = np.fromiter((year for year, _ in self.samples), dtype=np.int64,
                                    count=n_samples)
        sample_starts = np.fromiter((start for _, start in self.samples), dtype=np.int64,
                                      count=n_samples)
        sample_idx_by_year = {
            year: np.flatnonzero(sample_years == year)
            for year in self.times_by_year
        }
        time_int_by_year = {
            year: pd.DatetimeIndex(times).to_numpy("datetime64[ns]").astype("int64")
            for year, times in self.times_by_year.items()
        }
        for n, loc in enumerate(self.loc_ids):
            ts_arr = by_loc.get(str(loc))
            if ts_arr is None or ts_arr.size == 0:
                continue
            mask[:, n] = np.isin(target_int, ts_arr).any(axis=1)
            # A prefix sum marks all sliding input windows containing at least
            # one labelled timestamp for this node, without iterating windows.
            for year, sample_idx in sample_idx_by_year.items():
                if sample_idx.size == 0:
                    continue
                is_rare_at_time = np.isin(time_int_by_year[year], ts_arr)
                prefix = np.concatenate(([0], np.cumsum(is_rare_at_time, dtype=np.int64)))
                starts = sample_starts[sample_idx]
                history_mask[sample_idx, n] = (
                    prefix[starts + self.seq_len] > prefix[starts]
                )
        return int(mask.sum())

    def filter_normal_only_windows(self) -> tuple[int, int]:
        """Drop windows containing any target/history anomaly in any node."""
        mask_all = self.anomaly_mask_all
        history_mask_all = self.anomaly_history_mask_all
        if mask_all is None or history_mask_all is None:
            raise ValueError(
                "filter_normal_only_windows requires anomaly masks; call "
                "attach_anomaly_mask(train_scores) first."
            )
        if history_mask_all.shape != mask_all.shape:
            raise ValueError(
                "anomaly_history_mask_all must match anomaly_mask_all shape; "
                f"got {history_mask_all.shape} vs {mask_all.shape}."
            )
        before = len(self.samples)
        keep_window = ~np.any(mask_all | history_mask_all, axis=1)
        kept = int(keep_window.sum())
        if kept == 0:
            raise ValueError(
                "normal-only window filtering removed every training window; "
                "relax the anomaly labels/window definition or disable "
                "--train-normal-only."
            )
        self._select_indices(np.flatnonzero(keep_window))
        return kept, before


def _align_locations(
    ds: xr.Dataset,
    canonical_ids: np.ndarray,
    canonical_lats: np.ndarray,
    canonical_lons: np.ndarray,
    *,
    label: str,
    loc_dim: str = "location",
) -> xr.Dataset:
    """Validate the node set and reorder it to the canonical training order."""
    if loc_dim not in ds.dims or loc_dim not in ds.coords:
        raise ValueError(f"{label}: missing {loc_dim!r} dimension/coordinate.")
    for name in ("lat", "lon"):
        if name not in ds:
            raise ValueError(f"{label}: missing location coordinate {name!r}.")

    ids = np.asarray(ds[loc_dim].values)
    ids_text = np.asarray([str(v) for v in ids])
    canonical_text = np.asarray([str(v) for v in canonical_ids])
    if len(np.unique(ids_text)) != len(ids_text):
        raise ValueError(f"{label}: duplicate location IDs are not allowed.")
    if set(ids_text) != set(canonical_text):
        missing = sorted(set(canonical_text) - set(ids_text))
        extra = sorted(set(ids_text) - set(canonical_text))
        raise ValueError(
            f"{label}: location IDs differ from training nodes; "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )

    position = {value: i for i, value in enumerate(ids_text)}
    order = np.asarray([position[value] for value in canonical_text], dtype=np.int64)
    aligned = ds.isel({loc_dim: order})
    lats = np.asarray(aligned["lat"].values, dtype=float)
    lons = np.asarray(aligned["lon"].values, dtype=float)
    if not np.allclose(lats, canonical_lats, rtol=0.0, atol=1e-6):
        raise ValueError(f"{label}: latitude changed for one or more location IDs.")
    if not np.allclose(lons, canonical_lons, rtol=0.0, atol=1e-6):
        raise ValueError(f"{label}: longitude changed for one or more location IDs.")
    return aligned


def build_datasets(
    train_ds_map: Dict[int, xr.Dataset],
    test_ds: xr.Dataset,
    seq_len: int,
    horizon: int,
    target_variable: str = DEFAULT_TARGET_VARIABLE,
    feature_names: Optional[List[str]] = None,
    pv_target_clip_max: Optional[float] = None,
    validation_year: Optional[int] = None,
    kt_poa_max: float = 1.6,
    train_normal_only: bool = False,
    train_anomaly_scores: Optional[pd.DataFrame] = None,
    test_anomaly_scores: Optional[pd.DataFrame] = None,
    anomaly_source: str = "climatology",
    event_spatial_quantile: float = DEFAULT_EVENT_SPATIAL_QUANTILE,
    event_tail_quantile: float = DEFAULT_EVENT_TAIL_QUANTILE,
    detector_regional_quantile: float = DEFAULT_DETECTOR_REGIONAL_QUANTILE,
    detector_min_temporal_coverage: float = DEFAULT_DETECTOR_MIN_TEMPORAL_COVERAGE,
    forecast_horizons: Optional[tuple[int, ...]] = None,
) -> dict:
    """Build disjoint train/validation/test datasets with train-only fitting."""
    horizons = _normalise_forecast_horizons(horizon, forecast_horizons)
    if len(train_ds_map) < 2:
        raise ValueError(
            "At least two training years are required: the latest (or "
            "validation_year) is held out for validation and is never used to "
            "fit preprocessing."
        )
    if pv_target_clip_max is not None and pv_target_clip_max <= 0:
        raise ValueError("pv_target_clip_max must be positive or None.")
    anomaly_source = str(anomaly_source).strip().lower()
    if anomaly_source not in ANOMALY_SOURCES:
        raise ValueError(
            f"Unknown anomaly_source {anomaly_source!r}; expected {ANOMALY_SOURCES}."
        )
    selected = (
        list(feature_names)
        if feature_names is not None
        else list(PVGIS_STGNN_FEATURES)
    )
    unknown = [f for f in selected if f not in PVGIS_STGNN_FEATURES]
    if unknown:
        raise ValueError(f"Unknown feature(s) {unknown}; valid: {PVGIS_STGNN_FEATURES}.")
    if not selected:
        raise ValueError("feature_names selected an empty feature set.")
    keep_idx = [PVGIS_STGNN_FEATURES.index(f) for f in selected]

    years = sorted(train_ds_map)
    validation_year = max(years) if validation_year is None else validation_year
    if validation_year not in train_ds_map:
        raise ValueError(
            f"validation_year={validation_year} is not among training years {years}."
        )
    fit_years = [year for year in years if year != validation_year]
    if not fit_years:
        raise ValueError("Validation split leaves no year available for training.")

    canonical_ds = train_ds_map[fit_years[0]]
    loc_ids = np.asarray(canonical_ds["location"].values)
    canonical_lats = np.asarray(canonical_ds["lat"].values, dtype=float)
    canonical_lons = np.asarray(canonical_ds["lon"].values, dtype=float)
    if len(np.unique(np.asarray([str(v) for v in loc_ids]))) != len(loc_ids):
        raise ValueError("Canonical training year contains duplicate location IDs.")
    if not np.isfinite(canonical_lats).all() or not np.isfinite(canonical_lons).all():
        raise ValueError("Canonical training locations require finite coordinates.")

    aligned_train = {
        year: _align_locations(
            train_ds_map[year],
            loc_ids,
            canonical_lats,
            canonical_lons,
            label=f"PVGIS {year}",
        )
        for year in years
    }
    aligned_test = _align_locations(
        test_ds,
        loc_ids,
        canonical_lats,
        canonical_lons,
        label="PVGIS test year",
    )

    raws = {
        year: build_year_raw(
            aligned_train[year], target_variable, kt_poa_max=kt_poa_max
        )
        for year in years
    }
    test_raw = build_year_raw(
        aligned_test, target_variable, kt_poa_max=kt_poa_max
    )
    event_protocol = None
    event_rare_by_year = None
    normalization_masks = None
    if anomaly_source == "detector":
        if train_anomaly_scores is None:
            raise ValueError(
                "Detector regional P97.5 calibration requires "
                "train_anomaly_scores before dataset construction."
            )
        event_protocol = build_detector_event_protocol(
            train_anomaly_scores,
            {year: raws[year]["times"] for year in years},
            loc_ids,
            regional_quantile=detector_regional_quantile,
            min_temporal_coverage=detector_min_temporal_coverage,
        )
    elif train_normal_only:
        if train_anomaly_scores is None:
            raise ValueError(
                "train_normal_only=True requires train_anomaly_scores before "
                "dataset construction."
            )
        event_protocol = build_regional_event_protocol(
            train_anomaly_scores,
            {year: raws[year]["times"] for year in years},
            loc_ids,
            fit_years=fit_years,
            spatial_quantile=event_spatial_quantile,
            event_quantile=event_tail_quantile,
        )
    if train_normal_only:
        event_rare_by_year = event_protocol["rare_by_year"]
        normalization_masks = [
            normal_event_timestamp_mask(
                event_rare_by_year[year],
                seq_len=seq_len,
                horizon=max(horizons),
                forecast_horizons=horizons,
            )
            for year in fit_years
        ]
    norm = fit_normalization(
        [raws[year] for year in fit_years],
        time_masks=normalization_masks,
    )

    def _select(feats: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(feats[:, :, keep_idx])

    def _make_dataset(
        raw_map: Dict[int, dict],
        *,
        include_physical_targets: bool,
        event_map: Optional[Dict[int, np.ndarray]] = None,
        filter_rare_events: bool = False,
    ) -> PVGISWindowDataset:
        feats, pvn, pvr, solar, kt_poa, times = {}, {}, {}, {}, {}, {}
        for year, raw in raw_map.items():
            feature_array, pv_norm, pv_raw = assemble_feats(
                raw, norm, pv_target_clip_max
            )
            feats[year] = _select(feature_array)
            pvn[year] = pv_norm
            pvr[year] = pv_raw
            if include_physical_targets:
                solar[year] = raw["solar_wm2"]
            kt_poa[year] = raw["kt_poa"]
            times[year] = raw["times"]
        dataset = PVGISWindowDataset(
            feats,
            pvn,
            pvr,
            solar if include_physical_targets else None,
            times,
            seq_len,
            horizon,
            norm["pv_scale"],
            loc_ids,
            kt_poa_by_year=kt_poa,
            event_rare_by_year=event_map,
            forecast_horizons=horizons,
        )
        if filter_rare_events:
            dataset.event_filter_stats = dataset.drop_rare_event_windows()
        else:
            dataset.event_filter_stats = None
        return dataset

    train_dataset = _make_dataset(
        {year: raws[year] for year in fit_years},
        include_physical_targets=False,
        event_map=(
            {year: event_rare_by_year[year] for year in fit_years}
            if event_rare_by_year is not None
            else None
        ),
        filter_rare_events=train_normal_only,
    )
    validation_dataset = _make_dataset(
        {validation_year: raws[validation_year]},
        include_physical_targets=True,
        event_map=(
            {validation_year: event_rare_by_year[validation_year]}
            if event_rare_by_year is not None
            else None
        ),
        filter_rare_events=train_normal_only,
    )
    test_year = int(test_raw["times"][0].year)
    test_event_protocol = None
    test_event_map = None
    test_event_labels = None
    if test_anomaly_scores is not None and (
        event_protocol is not None or anomaly_source == "detector"
    ):
        if anomaly_source == "detector":
            test_event_protocol = build_detector_event_protocol(
                test_anomaly_scores,
                {test_year: test_raw["times"]},
                loc_ids,
                regional_quantile=detector_regional_quantile,
                seasonal_thresholds=event_protocol["seasonal_thresholds"],
                min_temporal_coverage=detector_min_temporal_coverage,
            )
        else:
            test_event_protocol = build_regional_event_protocol(
                test_anomaly_scores,
                {test_year: test_raw["times"]},
                loc_ids,
                spatial_quantile=event_spatial_quantile,
                event_quantile=event_tail_quantile,
                thresholds=event_protocol["thresholds"],
            )
        test_event_map = test_event_protocol["rare_by_year"]
        test_event_labels = test_event_protocol["labels_by_year"][test_year]
    test_dataset = _make_dataset(
        {test_year: test_raw},
        include_physical_targets=True,
        event_map=test_event_map,
        filter_rare_events=False,
    )
    return {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
        "train_years": fit_years,
        "validation_year": validation_year,
        "loc_ids": loc_ids,
        "lats": canonical_lats,
        "lons": canonical_lons,
        "n_features": len(selected),
        "features": selected,
        "pv_scale": norm["pv_scale"],
        "normalization": norm,
        "pv_target_clip_max": pv_target_clip_max,
        "forecast_horizons": horizons,
        "anomaly_source": anomaly_source,
        "event_protocol": event_protocol,
        "test_event_protocol": test_event_protocol,
        "test_event_labels": test_event_labels,
        "event_filter_stats": {
            "train": train_dataset.event_filter_stats,
            "validation": validation_dataset.event_filter_stats,
        },
    }


# --------------------------------------------------------------------------- #
# Model reuse + train / predict
# --------------------------------------------------------------------------- #
def make_model(
    n_nodes: int,
    seq_len: int,
    n_features: int = N_FEATURES,
    dropout: float = 0.2,
    n_sde_steps: int = 4,
    sigma_max: float = 0.5,
    use_irradiance_head: bool = True,
    kt_poa_max: float = 1.6,
    forecast_horizons: tuple[int, ...] = (1,),
) -> STGNN:
    """Instantiate STGNN with the selected PVGIS feature count."""
    return STGNN(
        n_nodes=n_nodes,
        n_features=n_features,
        seq_len=seq_len,
        patch_len=4 if seq_len > 1 else 1,
        stride=2 if seq_len > 1 else 1,
        d_model=128,
        gat_dim=96,
        dropout=dropout,
        use_patchtst=True,
        bilstm_pooling="attn",
        n_sde_steps=n_sde_steps,
        sigma_max=sigma_max,
        use_irradiance_head=use_irradiance_head,
        kt_poa_max=kt_poa_max,
        forecast_horizons=forecast_horizons,
    )


