"""Leakage-free spatial comparison of detector scores and forecast errors.

The functions in this module are post-processing only.  They retain one row
per PVGIS location in spatial maps and expose neighbourhood aggregation as a
separate time series, so spatial detail is never collapsed into one pixel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def _normalise_events(
    events: Mapping[str, Sequence[str | pd.Timestamp]],
) -> dict[str, pd.DatetimeIndex]:
    if not events:
        raise ValueError("At least one event window is required.")
    result: dict[str, pd.DatetimeIndex] = {}
    for name, dates in events.items():
        index = pd.DatetimeIndex(pd.to_datetime(tuple(dates))).normalize().unique()
        if index.empty:
            raise ValueError(f"Event {name!r} has no dates.")
        result[str(name)] = index
    return result


def _event_for_days(
    days: pd.Series, events: Mapping[str, pd.DatetimeIndex]
) -> pd.Series:
    labels = pd.Series(pd.NA, index=days.index, dtype="string")
    for name, event_days in events.items():
        selected = days.isin(event_days)
        if bool((selected & labels.notna()).any()):
            raise ValueError("Event windows must not overlap.")
        labels.loc[selected] = name
    return labels


def _parse_boolean(values: pd.Series, *, name: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    parsed = values.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if parsed.isna().any():
        raise ValueError(f"{name} contains non-boolean values.")
    return parsed.to_numpy(dtype=bool)


def select_reference_neighbourhood(
    locations: pd.DataFrame,
    *,
    n_neighbours: int = 8,
    reference_location: str | None = None,
) -> pd.DataFrame:
    """Select one objective reference location and its geographical KNN.

    Without an explicit reference, the PVGIS point closest to the mean region
    coordinate is selected.  This rule is independent of anomaly scores.
    """
    required = {"location", "latitude", "longitude"}
    missing = required - set(locations.columns)
    if missing:
        raise ValueError(f"Location table is missing {sorted(missing)}.")
    work = locations[["location", "latitude", "longitude"]].copy()
    work["location"] = work["location"].astype(str)
    if work.empty or work["location"].duplicated().any():
        raise ValueError("Locations must be non-empty and unique.")
    coordinates = work[["latitude", "longitude"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("Location coordinates must be finite.")
    if not 1 <= int(n_neighbours) < len(work):
        raise ValueError("n_neighbours must be between 1 and n_locations - 1.")

    if reference_location is None:
        centre = coordinates.mean(axis=0)
        centre_scale = np.cos(np.deg2rad(centre[0]))
        centre_distance = (
            (coordinates[:, 0] - centre[0]) ** 2
            + ((coordinates[:, 1] - centre[1]) * centre_scale) ** 2
        )
        reference_index = int(np.argmin(centre_distance))
    else:
        matches = np.flatnonzero(work["location"].eq(str(reference_location)))
        if len(matches) != 1:
            raise ValueError(f"Unknown reference location {reference_location!r}.")
        reference_index = int(matches[0])

    latitude = np.deg2rad(coordinates[:, 0])
    longitude = np.deg2rad(coordinates[:, 1])
    reference_latitude = latitude[reference_index]
    reference_longitude = longitude[reference_index]
    dlat = latitude - reference_latitude
    dlon = longitude - reference_longitude
    haversine = np.sin(dlat / 2.0) ** 2 + (
        np.cos(reference_latitude) * np.cos(latitude) * np.sin(dlon / 2.0) ** 2
    )
    distance_km = 2.0 * 6371.0088 * np.arcsin(
        np.sqrt(np.clip(haversine, 0.0, 1.0))
    )
    selected = np.argsort(distance_km, kind="stable")[: int(n_neighbours) + 1]
    result = work.iloc[selected].copy()
    result["distance_km"] = distance_km[selected]
    result["neighbour_rank"] = np.arange(len(result), dtype=int)
    result["is_reference"] = result["neighbour_rank"].eq(0)
    result["reference_location"] = work.iloc[reference_index]["location"]
    return result.reset_index(drop=True)


def load_detector_event_rows(
    score_path: str | Path,
    *,
    detector: str,
    events: Mapping[str, Sequence[str | pd.Timestamp]],
    threshold_table: pd.DataFrame | None = None,
    pvgis_locations: pd.DataFrame | None = None,
    pvgis_times: pd.DatetimeIndex | None = None,
    poa_by_location_time: np.ndarray | None = None,
    daytime_threshold_wm2: float | None = 10.0,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load only event rows and create detector-specific comparable fields.

    ``intensity`` is detector-specific and must not be compared numerically
    across models: MTGFlow uses threshold excess in training-IQR units, while
    STGAN uses the exported global score percentile divided by 100.
    """
    detector = str(detector).strip().lower()
    if detector not in {"mtgflow", "stgan"}:
        raise ValueError("detector must be 'mtgflow' or 'stgan'.")
    event_days = _normalise_events(events)
    source = Path(score_path)
    header = set(pd.read_csv(source, nrows=0).columns)
    percentile_column = None
    if detector == "stgan":
        percentile_column = next(
            (
                name
                for name in ("global_percentile", "score_percentile")
                if name in header
            ),
            None,
        )
    common = {"location", "timestamp", "anomaly_score"}
    required = common | ({"threshold"} if detector == "mtgflow" else {"is_anomaly"})
    if percentile_column is not None:
        required.add(percentile_column)
    missing = required - header
    if detector == "stgan" and percentile_column is None:
        missing.add("global_percentile (or legacy score_percentile)")
    if missing:
        raise ValueError(f"{source} is missing {sorted(missing)}.")

    threshold = None
    if detector == "mtgflow":
        if threshold_table is None:
            raise ValueError("MTGFlow requires a threshold table with training IQR.")
        threshold = threshold_table.copy()
        threshold["location"] = threshold["location"].astype(str)
        if threshold["location"].duplicated().any() or not {
            "saved_threshold", "iqr"
        } <= set(threshold.columns):
            raise ValueError("Invalid MTGFlow threshold table.")
        threshold = threshold.set_index("location")

    use_daytime = daytime_threshold_wm2 is not None
    if use_daytime:
        if pvgis_locations is None or pvgis_times is None or poa_by_location_time is None:
            raise ValueError("Daytime filtering requires PVGIS locations, times and POA.")
        locations = pvgis_locations.copy()
        locations["location"] = locations["location"].astype(str)
        location_index = pd.Index(locations["location"])
        time_index = pd.DatetimeIndex(pvgis_times)
        if poa_by_location_time.shape != (len(location_index), len(time_index)):
            raise ValueError("POA shape must be [locations, time].")

    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(source, usecols=sorted(required), chunksize=chunksize):
        chunk["location"] = chunk["location"].astype(str)
        chunk["timestamp"] = pd.to_datetime(
            chunk["timestamp"], errors="raise", utc=True
        ).dt.tz_convert(None)
        chunk["day"] = chunk["timestamp"].dt.normalize()
        chunk["event"] = _event_for_days(chunk["day"], event_days)
        chunk = chunk.loc[chunk["event"].notna()].copy()
        if chunk.empty:
            continue
        chunk["anomaly_score"] = pd.to_numeric(
            chunk["anomaly_score"], errors="coerce"
        )
        if use_daytime:
            loc_pos = location_index.get_indexer(chunk["location"])
            time_pos = time_index.get_indexer(chunk["timestamp"])
            if (loc_pos < 0).any() or (time_pos < 0).any():
                raise ValueError("Detector rows do not align with PVGIS coordinates/times.")
            keep = (
                poa_by_location_time[loc_pos, time_pos]
                > float(daytime_threshold_wm2)
            )
            chunk = chunk.loc[keep].copy()
            if chunk.empty:
                continue

        if detector == "mtgflow":
            saved = chunk["location"].map(threshold["saved_threshold"])
            iqr = chunk["location"].map(threshold["iqr"])
            csv_threshold = pd.to_numeric(chunk["threshold"], errors="coerce")
            numeric = np.column_stack(
                [chunk["anomaly_score"].to_numpy(float), csv_threshold, saved, iqr]
            )
            if not np.isfinite(numeric).all() or not np.allclose(
                csv_threshold, saved, rtol=1e-6, atol=1e-5
            ):
                raise ValueError("Invalid or inconsistent MTGFlow thresholds.")
            chunk["is_anomaly"] = chunk["anomaly_score"].ge(saved)
            chunk["intensity"] = np.maximum(
                (chunk["anomaly_score"] - saved) / iqr, 0.0
            )
        else:
            percentile = pd.to_numeric(chunk[percentile_column], errors="coerce")
            if not np.isfinite(
                np.column_stack([chunk["anomaly_score"], percentile])
            ).all() or bool(((percentile < 0) | (percentile > 100)).any()):
                raise ValueError("Invalid STGAN score or percentile.")
            chunk["is_anomaly"] = _parse_boolean(
                chunk["is_anomaly"], name="STGAN is_anomaly"
            )
            chunk["intensity"] = percentile / 100.0
        chunk["detector"] = detector
        parts.append(
            chunk[[
                "detector", "event", "day", "location", "timestamp",
                "anomaly_score", "intensity", "is_anomaly",
            ]]
        )
    if not parts:
        raise ValueError(f"No usable {detector} rows fall in the event windows.")
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["location", "timestamp"]).any():
        raise ValueError(f"{detector} contains duplicate location/timestamp rows.")
    return result


def load_quality_filtered_stgan_event_rows(
    prediction_path: str | Path,
    *,
    events: Mapping[str, Sequence[str | pd.Timestamp]],
    daytime_threshold_wm2: float = 10.0,
    coordinate_horizon: int = 1,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load clean STGAN decisions exported by pointwise post-processing.

    Direct multi-output forecasts repeat each detector coordinate for every
    horizon. One horizon is selected only to recover a unique copy of the
    quality-filtered ``(location, timestamp)`` decision; the detector itself is
    horizon-independent.
    """
    source = Path(prediction_path)
    event_days = _normalise_events(events)
    required = {
        "location", "timestamp", "horizon_hours",
        "detector_anomaly_score", "detector_is_anomaly",
        "clean_global_percentile", "solar_irradiance_poa_target",
    }
    header = set(pd.read_csv(source, nrows=0).columns)
    missing = required - header
    if missing:
        raise ValueError(f"{source} is missing {sorted(missing)}.")
    coordinate_horizon = int(coordinate_horizon)
    if coordinate_horizon < 1:
        raise ValueError("coordinate_horizon must be positive.")

    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        source,
        usecols=sorted(required),
        dtype={"location": "string"},
        chunksize=chunksize,
        low_memory=False,
    ):
        horizon = pd.to_numeric(chunk["horizon_hours"], errors="coerce")
        chunk = chunk.loc[horizon.eq(coordinate_horizon)].copy()
        if chunk.empty:
            continue
        chunk["timestamp"] = pd.to_datetime(
            chunk["timestamp"], errors="raise", utc=True
        ).dt.tz_convert(None)
        chunk["day"] = chunk["timestamp"].dt.normalize()
        chunk["event"] = _event_for_days(chunk["day"], event_days)
        chunk = chunk.loc[chunk["event"].notna()].copy()
        if chunk.empty:
            continue
        score = pd.to_numeric(chunk["detector_anomaly_score"], errors="coerce")
        percentile = pd.to_numeric(
            chunk["clean_global_percentile"], errors="coerce"
        )
        solar = pd.to_numeric(
            chunk["solar_irradiance_poa_target"], errors="coerce"
        )
        valid = (
            np.isfinite(np.column_stack([score, percentile, solar])).all(axis=1)
            & percentile.between(0.0, 100.0).to_numpy()
            & solar.gt(float(daytime_threshold_wm2)).to_numpy()
        )
        chunk = chunk.loc[valid].copy()
        if chunk.empty:
            continue
        chunk["anomaly_score"] = score.loc[valid].to_numpy(float)
        chunk["intensity"] = percentile.loc[valid].to_numpy(float) / 100.0
        chunk["is_anomaly"] = _parse_boolean(
            chunk["detector_is_anomaly"], name="clean STGAN detector_is_anomaly"
        )
        chunk["detector"] = "stgan"
        parts.append(chunk[[
            "detector", "event", "day", "location", "timestamp",
            "anomaly_score", "intensity", "is_anomaly",
        ]])
    if not parts:
        raise ValueError("No clean STGAN rows fall in the event windows.")
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["location", "timestamp"]).any():
        raise ValueError("Clean STGAN rows are not unique by location/timestamp.")
    return result


def aggregate_daily_quality_filtered_stgan(
    prediction_path: str | Path,
    *,
    daytime_threshold_wm2: float = 10.0,
    coordinate_horizon: int = 1,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Aggregate clean STGAN decisions into exact location/day cells."""
    source = Path(prediction_path)
    required = {
        "location", "timestamp", "horizon_hours", "detector_is_anomaly",
        "clean_global_percentile", "solar_irradiance_poa_target",
    }
    header = set(pd.read_csv(source, nrows=0).columns)
    missing = required - header
    if missing:
        raise ValueError(f"{source} is missing {sorted(missing)}.")
    coordinate_horizon = int(coordinate_horizon)
    if coordinate_horizon < 1:
        raise ValueError("coordinate_horizon must be positive.")

    partials: list[pd.DataFrame] = []
    source_rows = horizon_rows = retained_rows = 0
    for chunk in pd.read_csv(
        source,
        usecols=sorted(required),
        dtype={"location": "string"},
        chunksize=chunksize,
        low_memory=False,
    ):
        source_rows += len(chunk)
        horizon = pd.to_numeric(chunk["horizon_hours"], errors="coerce")
        chunk = chunk.loc[horizon.eq(coordinate_horizon)].copy()
        horizon_rows += len(chunk)
        if chunk.empty:
            continue
        chunk["timestamp"] = pd.to_datetime(
            chunk["timestamp"], errors="raise", utc=True
        ).dt.tz_convert(None)
        percentile = pd.to_numeric(
            chunk["clean_global_percentile"], errors="coerce"
        )
        solar = pd.to_numeric(
            chunk["solar_irradiance_poa_target"], errors="coerce"
        )
        valid = (
            np.isfinite(np.column_stack([percentile, solar])).all(axis=1)
            & percentile.between(0.0, 100.0).to_numpy()
            & solar.gt(float(daytime_threshold_wm2)).to_numpy()
        )
        chunk = chunk.loc[valid].copy()
        if chunk.empty:
            continue
        retained_rows += len(chunk)
        chunk["date"] = chunk["timestamp"].dt.normalize()
        chunk["is_anomaly"] = _parse_boolean(
            chunk["detector_is_anomaly"], name="clean STGAN detector_is_anomaly"
        ).astype(np.int8)
        chunk["intensity"] = percentile.loc[valid].to_numpy(float) / 100.0
        partials.append(
            chunk.groupby(["location", "date"], observed=True)
            .agg(
                n_observations=("is_anomaly", "size"),
                n_anomalous=("is_anomaly", "sum"),
                sum_intensity=("intensity", "sum"),
                max_intensity=("intensity", "max"),
            )
            .reset_index()
        )
    if horizon_rows == 0 or not partials:
        raise ValueError("No clean STGAN daytime coordinates are available.")
    daily = (
        pd.concat(partials, ignore_index=True)
        .groupby(["location", "date"], observed=True)
        .agg(
            n_observations=("n_observations", "sum"),
            n_anomalous=("n_anomalous", "sum"),
            sum_intensity=("sum_intensity", "sum"),
            max_intensity=("max_intensity", "max"),
        )
        .reset_index()
    )
    daily["anomaly_fraction"] = daily["n_anomalous"] / daily["n_observations"]
    daily["mean_intensity"] = daily["sum_intensity"] / daily["n_observations"]
    daily.attrs.update(
        source_rows=int(source_rows),
        coordinate_horizon_rows=int(horizon_rows),
        retained_rows=int(retained_rows),
        daytime_threshold_wm2=float(daytime_threshold_wm2),
    )
    return daily


def aggregate_daily_detector_clusters(
    daily: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate generic detector location/day cells into cluster/day cells."""
    required = {
        "location", "date", "n_observations", "n_anomalous",
        "sum_intensity", "max_intensity",
    }
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"Daily detector table is missing {sorted(missing)}.")
    mapping = clusters[["location", "geo_cluster"]].copy()
    mapping["location"] = mapping["location"].astype(str)
    work = daily.copy()
    work["location"] = work["location"].astype(str)
    work = work.merge(mapping, on="location", how="left", validate="many_to_one")
    if work["geo_cluster"].isna().any():
        raise ValueError("At least one detector location has no geographical cluster.")
    result = (
        work.groupby(["geo_cluster", "date"], observed=True)
        .agg(
            n_locations=("location", "nunique"),
            n_observations=("n_observations", "sum"),
            n_anomalous=("n_anomalous", "sum"),
            sum_intensity=("sum_intensity", "sum"),
            max_intensity=("max_intensity", "max"),
        )
        .reset_index()
    )
    result["anomaly_fraction"] = result["n_anomalous"] / result["n_observations"]
    result["mean_intensity"] = result["sum_intensity"] / result["n_observations"]
    return result


def aggregate_detector_locations(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event detector rows over time while retaining each location."""
    required = {
        "detector", "event", "location", "anomaly_score", "intensity", "is_anomaly"
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Detector rows are missing {sorted(missing)}.")
    result = (
        rows.groupby(["detector", "event", "location"], observed=True)
        .agg(
            n_observations=("is_anomaly", "size"),
            n_anomalies=("is_anomaly", "sum"),
            score_mean=("anomaly_score", "mean"),
            score_max=("anomaly_score", "max"),
            intensity_mean=("intensity", "mean"),
            intensity_max=("intensity", "max"),
        )
        .reset_index()
    )
    result["anomaly_fraction"] = result["n_anomalies"] / result["n_observations"]
    return result


def aggregate_detector_timeline(
    rows: pd.DataFrame,
    *,
    locations: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate a detector over locations at every timestamp."""
    work = rows.copy()
    if locations is not None:
        selected = {str(value) for value in locations}
        work = work.loc[work["location"].astype(str).isin(selected)]
    if work.empty:
        raise ValueError("No detector rows remain for the requested neighbourhood.")
    result = (
        work.groupby(["detector", "event", "timestamp"], observed=True)
        .agg(
            n_locations=("location", "nunique"),
            n_anomalies=("is_anomaly", "sum"),
            score_mean=("anomaly_score", "mean"),
            score_max=("anomaly_score", "max"),
            intensity_mean=("intensity", "mean"),
            intensity_max=("intensity", "max"),
        )
        .reset_index()
    )
    result["anomaly_fraction"] = result["n_anomalies"] / result["n_locations"]
    return result


def load_forecast_event_rows(
    prediction_path: str | Path,
    *,
    events: Mapping[str, Sequence[str | pd.Timestamp]],
    horizons: Sequence[int] = (1, 6),
    daytime_threshold_wm2: float | None = 10.0,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load forecast errors for event target timestamps and selected horizons."""
    source = Path(prediction_path)
    event_days = _normalise_events(events)
    selected_horizons = tuple(dict.fromkeys(int(value) for value in horizons))
    if not selected_horizons or any(value < 1 for value in selected_horizons):
        raise ValueError("Forecast horizons must be positive.")
    header = set(pd.read_csv(source, nrows=0).columns)
    prediction_column = "y_pred_mean" if "y_pred_mean" in header else "y_pred"
    required = {
        "location", "timestamp", "horizon_hours", "y_true", prediction_column
    }
    if daytime_threshold_wm2 is not None:
        required.add("solar_irradiance_poa_target")
    missing = required - header
    if missing:
        raise ValueError(f"{source} is missing {sorted(missing)}.")

    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(source, usecols=sorted(required), chunksize=chunksize):
        horizon = pd.to_numeric(chunk["horizon_hours"], errors="coerce")
        chunk = chunk.loc[horizon.isin(selected_horizons)].copy()
        if chunk.empty:
            continue
        chunk["horizon_hours"] = pd.to_numeric(
            chunk["horizon_hours"], errors="raise"
        ).astype(int)
        chunk["location"] = chunk["location"].astype(str)
        chunk["timestamp"] = pd.to_datetime(
            chunk["timestamp"], errors="raise", utc=True
        ).dt.tz_convert(None)
        chunk["day"] = chunk["timestamp"].dt.normalize()
        chunk["event"] = _event_for_days(chunk["day"], event_days)
        chunk = chunk.loc[chunk["event"].notna()].copy()
        if chunk.empty:
            continue
        numeric_columns = ["y_true", prediction_column]
        if daytime_threshold_wm2 is not None:
            numeric_columns.append("solar_irradiance_poa_target")
        for column in numeric_columns:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        finite = np.isfinite(chunk[numeric_columns].to_numpy(dtype=float)).all(axis=1)
        if daytime_threshold_wm2 is not None:
            finite &= chunk["solar_irradiance_poa_target"].gt(
                float(daytime_threshold_wm2)
            )
        chunk = chunk.loc[finite].copy()
        error = chunk[prediction_column] - chunk["y_true"]
        chunk["error"] = error
        chunk["abs_error"] = error.abs()
        chunk["squared_error"] = error**2
        parts.append(
            chunk[[
                "event", "day", "location", "timestamp", "horizon_hours",
                "error", "abs_error", "squared_error",
            ]]
        )
    if not parts:
        raise ValueError("No usable forecast rows fall in the event windows.")
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["location", "timestamp", "horizon_hours"]).any():
        raise ValueError("Predictions contain duplicate location/target/horizon rows.")
    available = set(result["horizon_hours"].unique())
    missing_horizons = sorted(set(selected_horizons) - available)
    if missing_horizons:
        raise ValueError(f"Missing forecast horizons {missing_horizons}.")
    return result


def aggregate_forecast_locations(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate forecast errors by event, horizon and location."""
    result = (
        rows.groupby(["event", "horizon_hours", "location"], observed=True)
        .agg(
            n_forecasts=("error", "size"),
            sum_error=("error", "sum"),
            sum_abs_error=("abs_error", "sum"),
            sum_squared_error=("squared_error", "sum"),
        )
        .reset_index()
    )
    result["bias"] = result["sum_error"] / result["n_forecasts"]
    result["mae"] = result["sum_abs_error"] / result["n_forecasts"]
    result["rmse"] = np.sqrt(result["sum_squared_error"] / result["n_forecasts"])
    return result


def aggregate_forecast_timeline(
    rows: pd.DataFrame,
    *,
    locations: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate forecast errors over locations at each target timestamp."""
    work = rows.copy()
    if locations is not None:
        selected = {str(value) for value in locations}
        work = work.loc[work["location"].astype(str).isin(selected)]
    if work.empty:
        raise ValueError("No forecast rows remain for the requested neighbourhood.")
    result = (
        work.groupby(["event", "horizon_hours", "timestamp"], observed=True)
        .agg(
            n_locations=("location", "nunique"),
            n_forecasts=("error", "size"),
            sum_error=("error", "sum"),
            sum_abs_error=("abs_error", "sum"),
            sum_squared_error=("squared_error", "sum"),
        )
        .reset_index()
    )
    result["bias"] = result["sum_error"] / result["n_forecasts"]
    result["mae"] = result["sum_abs_error"] / result["n_forecasts"]
    result["rmse"] = np.sqrt(result["sum_squared_error"] / result["n_forecasts"])
    return result
