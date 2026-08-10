"""Post-hoc spatial and temporal summaries of dense MTGFlow scores.

The detector itself remains location-wise.  This module never trains MTGFlow
and never uses 2019 anomaly scores to define geographical clusters: clusters
are learned once from latitude/longitude, while MTGFlow only determines each
space-time cell's colour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


REQUIRED_SCORE_COLUMNS = ("location", "timestamp", "anomaly_score", "threshold")


def load_pvgis_spatial_context(
    path: str | Path,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, np.ndarray]:
    """Load location coordinates, timestamps and reconstructed POA.

    The returned POA has shape ``[locations, time]`` and is reconstructed from
    tilted direct plus diffuse irradiance, consistently with the forecasting
    pipeline and the existing April-event notebook.
    """
    import xarray as xr

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with xr.open_dataset(source) as dataset:
        required = {
            "location",
            "time",
            "lat",
            "lon",
            "direct_irradiance_tilted",
            "diffuse_irradiance_tilted",
        }
        missing = required - set(dataset.variables)
        if missing:
            raise ValueError(f"PVGIS dataset is missing {sorted(missing)}.")
        location_ids = np.asarray(dataset["location"].values)
        locations = pd.DataFrame(
            {
                "location": [str(value) for value in location_ids],
                "latitude": np.asarray(dataset["lat"].values, dtype=float),
                "longitude": np.asarray(dataset["lon"].values, dtype=float),
            }
        )
        direct = np.asarray(
            dataset["direct_irradiance_tilted"]
            .transpose("location", "time")
            .values,
            dtype=np.float32,
        )
        diffuse = np.asarray(
            dataset["diffuse_irradiance_tilted"]
            .transpose("location", "time")
            .values,
            dtype=np.float32,
        )
        times = pd.DatetimeIndex(dataset["time"].values)

    if locations["location"].duplicated().any():
        raise ValueError("PVGIS location IDs must be unique.")
    if not np.isfinite(locations[["latitude", "longitude"]].to_numpy()).all():
        raise ValueError("PVGIS coordinates must be finite.")
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("PVGIS timestamps must be unique and increasing.")
    poa = direct + diffuse
    if poa.shape != (len(locations), len(times)) or not np.isfinite(poa).all():
        raise ValueError("Reconstructed PVGIS POA has an invalid shape or values.")
    return locations, times, poa


def _project_lonlat_km(locations: pd.DataFrame) -> np.ndarray:
    latitude = locations["latitude"].to_numpy(dtype=float)
    longitude = locations["longitude"].to_numpy(dtype=float)
    radius_km = 6371.0088
    reference_latitude = np.deg2rad(float(np.mean(latitude)))
    x = radius_km * np.deg2rad(longitude) * np.cos(reference_latitude)
    y = radius_km * np.deg2rad(latitude)
    return np.column_stack([x, y])


def build_geographic_clusters(
    locations: pd.DataFrame,
    *,
    n_clusters: int = 16,
    n_neighbors: int = 8,
) -> pd.DataFrame:
    """Create compact geographical clusters using a KNN-constrained Ward fit.

    Cluster IDs are subsequently ordered north-to-south and west-to-east, so
    their numbering is stable and interpretable.  ``within_cluster_order`` is
    suitable for the Y axis of the per-cluster heatmaps.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import kneighbors_graph

    required = {"location", "latitude", "longitude"}
    missing = required - set(locations.columns)
    if missing:
        raise ValueError(f"Location table is missing {sorted(missing)}.")
    work = locations[list(required)].copy()
    work["location"] = work["location"].astype(str)
    if work["location"].duplicated().any():
        raise ValueError("Location IDs must be unique.")
    coordinates = work[["latitude", "longitude"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("Location coordinates must be finite.")
    n_locations = len(work)
    if not 2 <= n_clusters <= n_locations:
        raise ValueError("n_clusters must be between 2 and the number of locations.")
    if not 1 <= n_neighbors < n_locations:
        raise ValueError("n_neighbors must be between 1 and n_locations - 1.")

    projected = _project_lonlat_km(work)
    connectivity = kneighbors_graph(
        projected, n_neighbors=n_neighbors, mode="connectivity", include_self=False
    )
    connectivity = connectivity.maximum(connectivity.T)
    raw_labels = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
        connectivity=connectivity,
    ).fit_predict(projected)
    work["_raw_cluster"] = raw_labels
    centroids = (
        work.groupby("_raw_cluster", observed=True)[["latitude", "longitude"]]
        .mean()
        .sort_values(["latitude", "longitude"], ascending=[False, True])
    )
    renumber = {int(raw): number for number, raw in enumerate(centroids.index)}
    work["geo_cluster"] = work["_raw_cluster"].map(renumber).astype(int)
    work = work.drop(columns="_raw_cluster")
    work = work.sort_values(
        ["geo_cluster", "latitude", "longitude", "location"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    work["within_cluster_order"] = (
        work.groupby("geo_cluster", observed=True).cumcount().astype(int)
    )
    return work


def _finite_quantiles(values: pd.Series) -> tuple[float, float, int]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        raise ValueError("Training anomaly scores contain no finite values.")
    q1, q3 = np.quantile(numeric, [0.25, 0.75])
    return float(q1), float(q3), int(numeric.size)


def read_saved_thresholds(
    score_path: str | Path,
    *,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Read and validate the single saved MTGFlow threshold per location."""
    source = Path(score_path)
    header = set(pd.read_csv(source, nrows=0).columns)
    missing = set(REQUIRED_SCORE_COLUMNS) - header
    if missing:
        raise ValueError(f"{source} is missing {sorted(missing)}.")
    parts = []
    for chunk in pd.read_csv(source, usecols=["location", "threshold"], chunksize=chunksize):
        chunk["location"] = chunk["location"].astype(str)
        chunk["threshold"] = pd.to_numeric(chunk["threshold"], errors="coerce")
        parts.append(
            chunk.groupby("location", observed=True)["threshold"]
            .agg(threshold_min="min", threshold_max="max")
            .reset_index()
        )
    if not parts:
        raise ValueError(f"No MTGFlow scores found in {source}.")
    combined = pd.concat(parts, ignore_index=True)
    result = (
        combined.groupby("location", observed=True)
        .agg(threshold_min=("threshold_min", "min"), threshold_max=("threshold_max", "max"))
        .reset_index()
    )
    numeric = result[["threshold_min", "threshold_max"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Saved thresholds must be finite.")
    if not np.allclose(result["threshold_min"], result["threshold_max"], rtol=0.0, atol=1e-8):
        raise ValueError("Every location must have one constant saved threshold.")
    return result[["location", "threshold_min"]].rename(
        columns={"threshold_min": "saved_threshold"}
    )


def build_threshold_table(
    saved_thresholds: pd.DataFrame,
    *,
    cached_statistics: Optional[str | Path] = None,
    per_site_training_paths: Iterable[str | Path] = (),
    aggregate_training_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Combine saved thresholds with a positive training IQR per location.

    Priority is: cached exact statistics, full per-site training files, then an
    aggregate training export.  The latter is explicitly marked as a fallback.
    The saved test threshold always defines the binary MTGFlow decision.
    """
    saved = saved_thresholds.copy()
    saved["location"] = saved["location"].astype(str)
    if saved["location"].duplicated().any():
        raise ValueError("Saved threshold table contains duplicate locations.")

    statistics: Optional[pd.DataFrame] = None
    source_mode: str
    cached = Path(cached_statistics) if cached_statistics is not None else None
    if cached is not None and cached.is_file():
        statistics = pd.read_csv(cached)
        source_mode = "cached_training_statistics"
    else:
        paths = [Path(path) for path in per_site_training_paths]
        if paths:
            rows = []
            for path in paths:
                frame = pd.read_csv(path, usecols=["location", "anomaly_score"])
                unique = frame["location"].astype(str).unique()
                if len(unique) != 1:
                    raise ValueError(f"{path} must contain exactly one location.")
                q1, q3, count = _finite_quantiles(frame["anomaly_score"])
                rows.append(
                    {"location": unique[0], "q1": q1, "q3": q3, "n_train_scores": count}
                )
            statistics = pd.DataFrame(rows)
            source_mode = "full_per_site_training"
        elif aggregate_training_path is not None and Path(aggregate_training_path).is_file():
            frame = pd.read_csv(
                aggregate_training_path, usecols=["location", "anomaly_score"]
            )
            frame["location"] = frame["location"].astype(str)
            frame["anomaly_score"] = pd.to_numeric(frame["anomaly_score"], errors="coerce")
            frame = frame[np.isfinite(frame["anomaly_score"].to_numpy(dtype=float))]
            quantiles = (
                frame.groupby("location", observed=True)["anomaly_score"]
                .quantile([0.25, 0.75])
                .unstack()
                .rename(columns={0.25: "q1", 0.75: "q3"})
                .reset_index()
            )
            counts = (
                frame.groupby("location", observed=True)
                .size()
                .rename("n_train_scores")
                .reset_index()
            )
            statistics = quantiles.merge(counts, on="location", validate="one_to_one")
            source_mode = "aggregate_training_fallback"
        else:
            raise FileNotFoundError(
                "Training IQR unavailable: provide cached statistics, per-site "
                "train_scores.csv files, or train_anomaly_scores.csv."
            )

    required = {"location", "q1", "q3"}
    missing = required - set(statistics.columns)
    if missing:
        raise ValueError(f"Training statistics are missing {sorted(missing)}.")
    statistics = statistics.copy()
    statistics["location"] = statistics["location"].astype(str)
    if statistics["location"].duplicated().any():
        raise ValueError("Training statistics contain duplicate locations.")
    statistics["iqr"] = (
        pd.to_numeric(statistics["q3"], errors="coerce")
        - pd.to_numeric(statistics["q1"], errors="coerce")
    )
    table = saved.merge(statistics, on="location", how="left", validate="one_to_one")
    if len(table) != len(saved) or table["iqr"].isna().any():
        missing_locations = table.loc[table["iqr"].isna(), "location"].tolist()[:10]
        raise ValueError(f"Training IQR missing for locations {missing_locations}.")
    if not np.isfinite(table["iqr"].to_numpy(dtype=float)).all() or (table["iqr"] <= 0).any():
        raise ValueError("Every training IQR must be finite and positive.")
    table["threshold_statistics_source"] = source_mode
    return table


def aggregate_daily_scores(
    score_path: str | Path,
    threshold_table: pd.DataFrame,
    pvgis_locations: pd.DataFrame,
    pvgis_times: pd.DatetimeIndex,
    poa_by_location_time: np.ndarray,
    *,
    daytime_threshold_wm2: Optional[float] = 10.0,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Aggregate dense MTGFlow scores into location/day visualization cells."""
    source = Path(score_path)
    header = set(pd.read_csv(source, nrows=0).columns)
    missing = set(REQUIRED_SCORE_COLUMNS) - header
    if missing:
        raise ValueError(f"{source} is missing {sorted(missing)}.")
    locations = pvgis_locations.copy()
    locations["location"] = locations["location"].astype(str)
    if locations["location"].duplicated().any():
        raise ValueError("PVGIS locations must be unique.")
    if poa_by_location_time.shape != (len(locations), len(pvgis_times)):
        raise ValueError("POA shape must be [locations, time].")
    threshold = threshold_table.copy()
    threshold["location"] = threshold["location"].astype(str)
    threshold = threshold.set_index("location")
    if not {"saved_threshold", "iqr"} <= set(threshold.columns):
        raise ValueError("Threshold table requires saved_threshold and iqr.")

    location_index = pd.Index(locations["location"])
    time_index = pd.DatetimeIndex(pvgis_times)
    partials = []
    matched_rows = 0
    daytime_rows = 0
    for chunk in pd.read_csv(source, usecols=list(REQUIRED_SCORE_COLUMNS), chunksize=chunksize):
        chunk["location"] = chunk["location"].astype(str)
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="raise")
        chunk["anomaly_score"] = pd.to_numeric(chunk["anomaly_score"], errors="coerce")
        chunk["threshold"] = pd.to_numeric(chunk["threshold"], errors="coerce")
        loc_pos = location_index.get_indexer(chunk["location"])
        time_pos = time_index.get_indexer(chunk["timestamp"])
        if (loc_pos < 0).any() or (time_pos < 0).any():
            raise ValueError("MTGFlow rows do not align with PVGIS location/timestamps.")
        expected_threshold = chunk["location"].map(threshold["saved_threshold"]).to_numpy(float)
        iqr = chunk["location"].map(threshold["iqr"]).to_numpy(float)
        values = chunk[["anomaly_score", "threshold"]].to_numpy(dtype=float)
        finite = np.isfinite(values).all(axis=1) & np.isfinite(expected_threshold) & np.isfinite(iqr)
        if not finite.all():
            raise ValueError("MTGFlow score, threshold or training IQR is non-finite.")
        if not np.allclose(chunk["threshold"], expected_threshold, rtol=1e-6, atol=1e-5):
            raise ValueError("CSV thresholds disagree with the saved per-location thresholds.")
        matched_rows += len(chunk)
        if daytime_threshold_wm2 is not None:
            keep = poa_by_location_time[loc_pos, time_pos] > float(daytime_threshold_wm2)
            chunk = chunk.loc[keep].copy()
            expected_threshold = expected_threshold[keep]
            iqr = iqr[keep]
        else:
            chunk = chunk.copy()
        daytime_rows += len(chunk)
        if chunk.empty:
            continue
        scores = chunk["anomaly_score"].to_numpy(dtype=float)
        is_anomaly = scores >= expected_threshold
        excess = (scores - expected_threshold) / iqr
        chunk["date"] = chunk["timestamp"].dt.floor("D")
        chunk["is_anomaly"] = is_anomaly.astype(np.int8)
        chunk["positive_excess_iqr"] = np.maximum(excess, 0.0)
        grouped = (
            chunk.groupby(["location", "date"], observed=True)
            .agg(
                n_observations=("is_anomaly", "size"),
                n_anomalous=("is_anomaly", "sum"),
                sum_positive_excess_iqr=("positive_excess_iqr", "sum"),
                max_positive_excess_iqr=("positive_excess_iqr", "max"),
            )
            .reset_index()
        )
        partials.append(grouped)
    if matched_rows == 0 or not partials:
        raise ValueError("No usable MTGFlow rows were found.")
    daily = (
        pd.concat(partials, ignore_index=True)
        .groupby(["location", "date"], observed=True)
        .agg(
            n_observations=("n_observations", "sum"),
            n_anomalous=("n_anomalous", "sum"),
            sum_positive_excess_iqr=("sum_positive_excess_iqr", "sum"),
            max_positive_excess_iqr=("max_positive_excess_iqr", "max"),
        )
        .reset_index()
    )
    daily["anomaly_fraction"] = daily["n_anomalous"] / daily["n_observations"]
    daily["mean_positive_excess_iqr"] = (
        daily["sum_positive_excess_iqr"] / daily["n_observations"]
    )
    daily.attrs.update(
        source_rows=int(matched_rows),
        retained_rows=int(daytime_rows),
        daytime_threshold_wm2=daytime_threshold_wm2,
    )
    return daily


def aggregate_clusters(
    daily: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate location/day cells into cluster/day overview cells."""
    required_daily = {
        "location",
        "date",
        "n_observations",
        "n_anomalous",
        "sum_positive_excess_iqr",
        "max_positive_excess_iqr",
    }
    missing = required_daily - set(daily.columns)
    if missing:
        raise ValueError(f"Daily table is missing {sorted(missing)}.")
    mapping = clusters[["location", "geo_cluster"]].copy()
    mapping["location"] = mapping["location"].astype(str)
    work = daily.copy()
    work["location"] = work["location"].astype(str)
    work = work.merge(mapping, on="location", how="left", validate="many_to_one")
    if work["geo_cluster"].isna().any():
        raise ValueError("At least one scored location has no geographical cluster.")
    work["location_has_anomaly"] = work["n_anomalous"] > 0
    result = (
        work.groupby(["geo_cluster", "date"], observed=True)
        .agg(
            n_locations=("location", "nunique"),
            n_locations_anomalous=("location_has_anomaly", "sum"),
            n_observations=("n_observations", "sum"),
            n_anomalous=("n_anomalous", "sum"),
            sum_positive_excess_iqr=("sum_positive_excess_iqr", "sum"),
            max_positive_excess_iqr=("max_positive_excess_iqr", "max"),
        )
        .reset_index()
    )
    result["anomaly_fraction"] = result["n_anomalous"] / result["n_observations"]
    result["anomalous_location_fraction"] = (
        result["n_locations_anomalous"] / result["n_locations"]
    )
    result["mean_positive_excess_iqr"] = (
        result["sum_positive_excess_iqr"] / result["n_observations"]
    )
    return result
