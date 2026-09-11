from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physiq_pv.reporting.mtgflow_spatiotemporal import (
    aggregate_clusters,
    aggregate_daily_forecast_errors,
    aggregate_daily_scores,
    aggregate_forecast_error_clusters,
    build_geographic_clusters,
    build_threshold_table,
    load_pvgis_spatial_context,
    read_saved_thresholds,
)


def _locations() -> pd.DataFrame:
    rows = []
    for latitude in (44.8, 45.0, 45.2, 45.4):
        for longitude in (7.0, 7.2, 7.4, 7.6):
            rows.append(
                {
                    "location": str(len(rows)),
                    "latitude": latitude,
                    "longitude": longitude,
                }
            )
    return pd.DataFrame(rows)


def test_geographic_clusters_use_all_locations_once() -> None:
    locations = _locations()
    clustered = build_geographic_clusters(locations, n_clusters=4, n_neighbors=3)
    assert set(clustered["location"]) == set(locations["location"])
    assert sorted(clustered["geo_cluster"].unique()) == [0, 1, 2, 3]
    assert clustered.groupby("geo_cluster")["within_cluster_order"].min().eq(0).all()
    repeated = build_geographic_clusters(locations, n_clusters=4, n_neighbors=3)
    assert clustered[["location", "geo_cluster"]].equals(
        repeated[["location", "geo_cluster"]]
    )


def test_threshold_and_daily_aggregation_use_saved_decision_and_iqr_excess(
    tmp_path: Path,
) -> None:
    times = pd.date_range("2019-06-28", periods=4, freq="h")
    scores = pd.DataFrame(
        {
            "location": ["0"] * 4 + ["1"] * 4,
            "timestamp": list(times) * 2,
            "anomaly_score": [0.0, 2.0, 3.0, 1.0, 8.0, 10.0, 12.0, 9.0],
            "threshold": [2.0] * 4 + [10.0] * 4,
        }
    )
    score_path = tmp_path / "anomaly_scores.csv"
    scores.to_csv(score_path, index=False)
    training = pd.DataFrame(
        {
            "location": ["0"] * 4 + ["1"] * 4,
            "anomaly_score": [0.0, 1.0, 2.0, 3.0, 6.0, 8.0, 10.0, 12.0],
        }
    )
    train_path = tmp_path / "train_anomaly_scores.csv"
    training.to_csv(train_path, index=False)

    saved = read_saved_thresholds(score_path, chunksize=3)
    threshold_table = build_threshold_table(
        saved, aggregate_training_path=train_path
    )
    locations = pd.DataFrame(
        {
            "location": ["0", "1"],
            "latitude": [45.0, 45.1],
            "longitude": [7.0, 7.1],
        }
    )
    poa = np.asarray([[20.0, 20.0, 0.0, 20.0], [20.0, 20.0, 20.0, 0.0]])
    daily = aggregate_daily_scores(
        score_path,
        threshold_table,
        locations,
        times,
        poa,
        daytime_threshold_wm2=10.0,
        chunksize=3,
    ).sort_values("location")
    assert daily["n_observations"].tolist() == [3, 3]
    # Equality to the saved threshold is anomalous, consistently with MTGFlow.
    assert daily["n_anomalous"].tolist() == [1, 2]
    assert np.allclose(daily["anomaly_fraction"], [1 / 3, 2 / 3])
    assert bool((daily["max_positive_excess_iqr"] >= 0).all())

    clusters = locations.assign(geo_cluster=[0, 0])
    summary = aggregate_clusters(daily, clusters)
    assert len(summary) == 1
    assert int(summary.iloc[0]["n_observations"]) == 6
    assert int(summary.iloc[0]["n_anomalous"]) == 3
    assert float(summary.iloc[0]["anomaly_fraction"]) == 0.5
    assert float(summary.iloc[0]["anomalous_location_fraction"]) == 1.0


def test_load_pvgis_spatial_context_reconstructs_poa(tmp_path: Path) -> None:
    times = pd.date_range("2019-01-01", periods=3, freq="h")
    dataset = xr.Dataset(
        {
            "direct_irradiance_tilted": (("location", "time"), np.ones((2, 3))),
            "diffuse_irradiance_tilted": (("location", "time"), np.full((2, 3), 2.0)),
            "lat": (("location",), [45.0, 45.1]),
            "lon": (("location",), [7.0, 7.1]),
        },
        coords={"location": [10, 11], "time": times},
    )
    path = tmp_path / "pvgis.nc"
    dataset.to_netcdf(path)
    locations, loaded_times, poa = load_pvgis_spatial_context(path)
    assert locations["location"].tolist() == ["10", "11"]
    assert loaded_times.equals(times)
    np.testing.assert_allclose(poa, 3.0)


def test_forecast_error_heatmap_data_filters_direct_t6(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    base = pd.DataFrame({
        "location": ["0", "1"],
        "timestamp": ["2019-06-28 12:00", "2019-06-28 12:00"],
        "y_true": [10.0, 20.0],
        "solar_irradiance_poa_target": [100.0, 100.0],
    })
    pd.concat([
        base.assign(horizon_hours=1, y_pred_mean=[100.0, 200.0]),
        base.assign(horizon_hours=6, y_pred_mean=[11.0, 18.0]),
    ], ignore_index=True).to_csv(path, index=False)

    daily = aggregate_daily_forecast_errors(
        path, horizon_hours=6, chunksize=1
    ).sort_values("location")
    assert daily["horizon_hours"].tolist() == [6, 6]
    assert daily["mae"].tolist() == [1.0, 2.0]
    assert daily["rmse"].tolist() == [1.0, 2.0]
    assert daily.attrs["horizon_rows"] == 2
    assert daily.attrs["skipped_other_horizons"] == 2

    clusters = pd.DataFrame({"location": ["0", "1"], "geo_cluster": [0, 0]})
    summary = aggregate_forecast_error_clusters(daily, clusters)
    assert summary["horizon_hours"].tolist() == [6]
    assert float(summary.iloc[0]["mae"]) == 1.5
    assert np.isclose(float(summary.iloc[0]["rmse"]), np.sqrt(2.5))


def test_spatiotemporal_notebook_is_valid_and_posthoc_only() -> None:
    path = _REPO_ROOT / "notebooks" / "mtgflow_spatiotemporal_anomaly_heatmaps.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    cell_ids = [cell["id"] for cell in cells]
    assert len(cell_ids) == len(set(cell_ids))
    assert all(
        cell.get("execution_count") is None and not cell.get("outputs")
        for cell in cells
        if cell["cell_type"] == "code"
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in cells)
    for required in (
        "N_GEO_CLUSTERS = 16",
        "GEO_K_NEIGHBORS = 8",
        "APRIL_EVENT_DATES",
        "JUNE_EVENT_DATES",
        "FORECAST_HORIZONS = (1, 6)",
        "aggregate_daily_scores",
        "aggregate_daily_forecast_errors",
        "horizon_hours=horizon_hours",
        "build_event_map",
        "build_forecast_event_map",
        "annual_cluster_heatmap_2019.png",
        "annual_cluster_forecast_error_heatmap_t_plus_{horizon_hours}.png",
        "anomaly_score >= saved_threshold",
    ):
        assert required in source
    for forbidden in ("subprocess.run", "train_model(", "RUN_TRAINING = True"):
        assert forbidden not in source
    for number, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path}:cell-{number}", "exec")


def test_spatiotemporal_notebook_executes_on_synthetic_data(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed_15"
    seed_dir.mkdir()
    output_dir = tmp_path / "output"
    pvgis_path = tmp_path / "pvgis_2019.nc"
    statistics_path = tmp_path / "training_iqr_by_location.csv"
    predictions_path = tmp_path / "predictions.csv"
    location_ids = np.arange(16)
    dates = pd.to_datetime(
        [
            "2019-04-23 12:00",
            "2019-04-24 12:00",
            "2019-04-25 12:00",
            "2019-04-26 12:00",
            "2019-06-28 12:00",
            "2019-06-29 12:00",
        ]
    )
    latitude = 44.8 + 0.1 * (location_ids // 4)
    longitude = 7.0 + 0.1 * (location_ids % 4)
    pvgis_dates = pd.date_range(dates.min(), dates.max(), freq="h")
    irradiance = np.full((len(location_ids), len(pvgis_dates)), 100.0, dtype=np.float32)
    xr.Dataset(
        {
            "direct_irradiance_tilted": (("location", "time"), irradiance),
            "diffuse_irradiance_tilted": (("location", "time"), irradiance * 0.2),
            "sun_height": (("location", "time"), irradiance * 0.1),
            "pv_power_output": (("location", "time"), irradiance),
            "lat": (("location",), latitude),
            "lon": (("location",), longitude),
        },
        coords={"location": location_ids, "time": pvgis_dates},
    ).to_netcdf(pvgis_path)
    rows = []
    for location in location_ids:
        for number, timestamp in enumerate(dates):
            rows.append(
                {
                    "location": str(location),
                    "timestamp": timestamp,
                    "anomaly_score": 2.0 if (location + number) % 5 == 0 else 0.0,
                    "threshold": 1.0,
                }
            )
    pd.DataFrame(rows).to_csv(seed_dir / "anomaly_scores.csv", index=False)
    prediction_rows = []
    for horizon in (1, 6):
        for location in location_ids:
            for number, timestamp in enumerate(dates):
                y_true = 100.0 + float(location)
                prediction_rows.append({
                    "location": str(location),
                    "timestamp": timestamp,
                    "horizon_hours": horizon,
                    "y_true": y_true,
                    "y_pred_mean": (
                        y_true + float((location + number) % 3 - 1)
                        if horizon == 6 else y_true + 1000.0
                    ),
                    "solar_irradiance_poa_target": 100.0,
                })
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    pd.DataFrame(
        {
            "location": [str(value) for value in location_ids],
            "q1": np.zeros(len(location_ids)),
            "q3": np.ones(len(location_ids)),
            "n_train_scores": np.full(len(location_ids), 100),
        }
    ).to_csv(statistics_path, index=False)

    environment = {
        "MTGFLOW_SEED_DIR": str(seed_dir),
        "STGAN_SEED_DIR": str(seed_dir),
        "MTGFLOW_TRAINING_STATS_CSV": str(statistics_path),
        "PVGIS_2019_PATH": str(pvgis_path),
        "SDE_PREDICTIONS_CSV": str(predictions_path),
        "MTGFLOW_SPATIOTEMPORAL_OUT_DIR": str(output_dir),
        "EXPECTED_LOCATIONS": "16",
        "MPLBACKEND": "Agg",
    }
    previous = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    notebook_path = (
        _REPO_ROOT / "notebooks" / "mtgflow_spatiotemporal_anomaly_heatmaps.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__notebook_smoke__"}
    original_cwd = Path.cwd()
    try:
        os.chdir(_REPO_ROOT)
        for number, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                exec(
                    compile(
                        "".join(cell["source"]),
                        f"{notebook_path}:cell-{number}",
                        "exec",
                    ),
                    namespace,
                )
    finally:
        os.chdir(original_cwd)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    assert (output_dir / "geographic_clusters.csv").is_file()
    for detector in ("stgan", "mtgflow"):
        prefix = output_dir / "daytime_maps" / f"{detector}_spatial_frequency_daytime"
        assert prefix.with_suffix(".png").is_file()
        frequency = pd.read_csv(prefix.with_suffix(".csv"))
        assert len(frequency) == len(location_ids)
        assert frequency["n_valid_hours"].eq(len(dates)).all()
        assert frequency["n_eligible_hours"].eq(len(pvgis_dates)).all()
    assert (output_dir / "daily_location_anomalies_2019.csv").is_file()
    for horizon in (1, 6):
        assert (
            output_dir / f"daily_location_forecast_errors_t_plus_{horizon}.csv"
        ).is_file()
    assert (output_dir / "figures" / "annual_cluster_heatmap_2019.png").is_file()
    for horizon in (1, 6):
        assert (
            output_dir
            / "figures"
            / f"annual_cluster_forecast_error_heatmap_t_plus_{horizon}.png"
        ).is_file()
    assert (
        output_dir / "figures" / "april_dust_23_26_geographic_anomaly_map.png"
    ).is_file()
    assert (
        output_dir / "figures" / "june_extreme_28_29_geographic_anomaly_map.png"
    ).is_file()
    for horizon in (1, 6):
        assert (
            output_dir
            / "figures"
            / f"april_dust_23_26_geographic_forecast_error_t_plus_{horizon}.png"
        ).is_file()
        assert (
            output_dir
            / "figures"
            / f"june_extreme_28_29_geographic_forecast_error_t_plus_{horizon}.png"
        ).is_file()
    assert len(list((output_dir / "figures").glob("annual_location_heatmap_cluster_*.png"))) == 16
    for horizon in (1, 6):
        assert len(list(
            (output_dir / "figures").glob(
                f"annual_location_forecast_error_t_plus_{horizon}_cluster_*.png"
            )
        )) == 16


if __name__ == "__main__":
    test_geographic_clusters_use_all_locations_once()
    with TemporaryDirectory() as directory:
        test_threshold_and_daily_aggregation_use_saved_decision_and_iqr_excess(
            Path(directory)
        )
    with TemporaryDirectory() as directory:
        test_load_pvgis_spatial_context_reconstructs_poa(Path(directory))
    with TemporaryDirectory() as directory:
        test_forecast_error_heatmap_data_filters_direct_t6(Path(directory))
    test_spatiotemporal_notebook_is_valid_and_posthoc_only()
    with TemporaryDirectory() as directory:
        test_spatiotemporal_notebook_executes_on_synthetic_data(Path(directory))
    print("PASS: MTGFlow spatio-temporal post-hoc tests")
