from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physiq_pv.reporting.detector_spatial_comparison import (
    aggregate_detector_locations,
    aggregate_detector_timeline,
    aggregate_forecast_locations,
    aggregate_forecast_timeline,
    load_detector_event_rows,
    load_forecast_event_rows,
    select_reference_neighbourhood,
)
from physiq_pv.reporting.detector_threshold_sensitivity import (
    join_detector_errors,
    load_mtgflow_coordinates,
    load_prediction_errors,
    load_stgan_coordinates,
    sensitivity_sweep,
)


def _locations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "location": ["0", "1", "2"],
            "latitude": [45.0, 45.0, 45.2],
            "longitude": [7.0, 7.1, 7.0],
        }
    )


def _write_inputs(root: Path) -> tuple[Path, Path, Path, pd.DatetimeIndex]:
    times = pd.to_datetime(
        ["2019-06-28 12:00", "2019-06-28 13:00", "2019-07-02 12:00"]
    )
    mtg_rows = []
    stgan_rows = []
    prediction_rows = []
    for location in ("0", "1", "2"):
        for number, timestamp in enumerate(times):
            mtg_rows.append(
                {
                    "location": location,
                    "timestamp": timestamp,
                    "anomaly_score": float(number + int(location) + 2),
                    "threshold": 4.0,
                }
            )
            percentile = 99.5 if (location == "0" and number == 0) else 80.0
            stgan_rows.append(
                {
                    "location": location,
                    "timestamp": timestamp,
                    "anomaly_score": percentile / 100.0,
                    "score_percentile": percentile,
                    "is_anomaly": percentile >= 99.0,
                }
            )
            for horizon in (1, 6):
                prediction_rows.append(
                    {
                        "location": location,
                        "timestamp": timestamp,
                        "horizon_hours": horizon,
                        "y_true": 10.0,
                        "y_pred_mean": 10.0 + horizon + int(location),
                        "solar_irradiance_poa_target": 100.0,
                    }
                )
    mtg_path = root / "mtg.csv"
    stgan_path = root / "stgan.csv"
    prediction_path = root / "predictions.csv"
    pd.DataFrame(mtg_rows).to_csv(mtg_path, index=False)
    pd.DataFrame(stgan_rows).to_csv(stgan_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)
    return mtg_path, stgan_path, prediction_path, times


def test_spatial_comparison_preserves_locations_and_has_separate_neighbourhood(
    tmp_path: Path,
) -> None:
    mtg_path, stgan_path, prediction_path, times = _write_inputs(tmp_path)
    locations = _locations()
    neighbourhood = select_reference_neighbourhood(
        locations, n_neighbours=2, reference_location="1"
    )
    assert neighbourhood["location"].tolist()[0] == "1"
    assert neighbourhood["is_reference"].sum() == 1
    assert len(neighbourhood) == 3

    events = {"june": ("2019-06-28",), "july": ("2019-07-02",)}
    threshold_table = pd.DataFrame(
        {
            "location": ["0", "1", "2"],
            "q3": [1.0] * 3,
            "iqr": [2.0] * 3,
            "saved_threshold": [4.0] * 3,
        }
    )
    poa = np.full((3, len(times)), 100.0)
    common = {
        "events": events,
        "pvgis_locations": locations,
        "pvgis_times": times,
        "poa_by_location_time": poa,
    }
    mtg = load_detector_event_rows(
        mtg_path, detector="mtgflow", threshold_table=threshold_table, **common
    )
    stgan = load_detector_event_rows(stgan_path, detector="stgan", **common)
    rows = pd.concat([mtg, stgan], ignore_index=True)
    by_location = aggregate_detector_locations(rows)
    assert len(by_location) == 2 * 2 * 3
    assert set(by_location["detector"]) == {"mtgflow", "stgan"}
    timeline = aggregate_detector_timeline(rows, locations=["0", "1"])
    assert timeline["n_locations"].eq(2).all()

    forecasts = load_forecast_event_rows(
        prediction_path, events=events, horizons=(1, 6)
    )
    forecast_locations = aggregate_forecast_locations(forecasts)
    assert set(forecast_locations["horizon_hours"]) == {1, 6}
    forecast_timeline = aggregate_forecast_timeline(
        forecasts, locations=["0", "1"]
    )
    assert forecast_timeline["n_locations"].eq(2).all()


def test_threshold_sensitivity_uses_exact_mtgflow_k_and_stgan_top_percent(
    tmp_path: Path,
) -> None:
    mtg_path, stgan_path, prediction_path, _ = _write_inputs(tmp_path)
    errors = load_prediction_errors(prediction_path, horizons=(1, 6))
    threshold_table = pd.DataFrame(
        {
            "location": ["0", "1", "2"],
            "q3": [1.0] * 3,
            "iqr": [2.0] * 3,
            "saved_threshold": [4.0] * 3,
        }
    )
    mtg_coordinate = load_mtgflow_coordinates(mtg_path, threshold_table)
    mtg_joined = join_detector_errors(errors, mtg_coordinate)
    mtg = sensitivity_sweep(
        mtg_joined,
        [1.5],
        detector="mtgflow",
        rare_when="coordinate_ge_threshold",
    )
    assert set(mtg["horizon_hours"]) == {1, 6}
    assert mtg["n_normal"].add(mtg["n_rare"]).eq(9).all()

    stgan_coordinate = load_stgan_coordinates(stgan_path)
    stgan_joined = join_detector_errors(errors, stgan_coordinate)
    stgan = sensitivity_sweep(
        stgan_joined,
        [1.0],
        detector="stgan",
        rare_when="coordinate_le_threshold",
    )
    # One flagged target per horizon at the reference top 1% cutoff.
    assert stgan["n_rare"].tolist() == [1, 1]
    assert stgan["rare_fraction"].tolist() == [1 / 9, 1 / 9]


def test_new_notebooks_are_valid_posthoc_wrappers() -> None:
    expected = {
        "spatial_anomaly_comparison_mtgflow_stgan.ipynb": (
            "select_reference_neighbourhood",
            "HORIZONS = (1, 6)",
            "spatial_pixels_aggregated",
        ),
        "anomaly_threshold_sensitivity_mtgflow_stgan.ipynb": (
            "MTGFLOW_REFERENCE_K = 1.5",
            "STGAN_REFERENCE_TOP_PERCENT = 1.0",
            "sensitivity_sweep",
        ),
        "anomaly_analysis_results_summary.ipynb": (
            "event_detector_summary.csv",
            "event_forecast_summary.csv",
            "reference_decision_metrics.csv",
        ),
    }
    for name, required in expected.items():
        path = ROOT / "notebooks" / name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert all(value in source for value in required)
        assert "train_model(" not in source
        assert "subprocess.run" not in source
        ids = [cell["id"] for cell in notebook["cells"]]
        assert len(ids) == len(set(ids))
        for number, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None
                assert not cell.get("outputs")
                compile("".join(cell["source"]), f"{path}:cell-{number}", "exec")


def test_new_notebooks_execute_in_order_on_synthetic_data(tmp_path: Path) -> None:
    mtg_dir = tmp_path / "mtg"
    stgan_dir = tmp_path / "stgan"
    mtg_dir.mkdir()
    stgan_dir.mkdir()
    spatial_out = tmp_path / "spatial"
    sensitivity_out = tmp_path / "sensitivity"
    summary_out = tmp_path / "summary"
    prediction_path = tmp_path / "predictions.csv"
    statistics_path = tmp_path / "statistics.csv"
    pvgis_path = tmp_path / "pvgis.nc"
    locations = np.arange(9)
    times = pd.to_datetime(
        [
            "2019-04-23 12:00", "2019-04-24 12:00",
            "2019-04-25 12:00", "2019-04-26 12:00",
            "2019-06-28 12:00", "2019-06-29 12:00",
            "2019-07-02 12:00",
        ]
    )
    latitude = 44.9 + 0.1 * (locations // 3)
    longitude = 7.0 + 0.1 * (locations % 3)
    irradiance = np.full((len(locations), len(times)), 100.0, dtype=np.float32)
    xr.Dataset(
        {
            "direct_irradiance_tilted": (("location", "time"), irradiance),
            "diffuse_irradiance_tilted": (("location", "time"), irradiance * 0.2),
            "lat": (("location",), latitude),
            "lon": (("location",), longitude),
        },
        coords={"location": locations, "time": times},
    ).to_netcdf(pvgis_path)
    mtg_rows = []
    stgan_rows = []
    prediction_rows = []
    total = len(locations) * len(times)
    rank = 0
    for location in locations:
        for number, timestamp in enumerate(times):
            rank += 1
            mtg_rows.append(
                {
                    "location": str(location), "timestamp": timestamp,
                    "anomaly_score": 5.0 if (location + number) % 5 == 0 else 2.0,
                    "threshold": 4.0,
                }
            )
            percentile = 100.0 * rank / total
            stgan_rows.append(
                {
                    "location": str(location), "timestamp": timestamp,
                    "anomaly_score": percentile / 100.0,
                    "score_percentile": percentile,
                    "is_anomaly": percentile >= 99.0,
                }
            )
            for horizon in (1, 6):
                prediction_rows.append(
                    {
                        "location": str(location), "timestamp": timestamp,
                        "horizon_hours": horizon, "y_true": 100.0 + location,
                        "y_pred_mean": 100.0 + location + horizon,
                        "solar_irradiance_poa_target": 100.0,
                    }
                )
    pd.DataFrame(mtg_rows).to_csv(mtg_dir / "anomaly_scores.csv", index=False)
    pd.DataFrame(stgan_rows).to_csv(stgan_dir / "anomaly_scores.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)
    pd.DataFrame(
        {
            "location": locations.astype(str), "q1": -1.0, "q3": 1.0,
            "iqr": 2.0, "saved_threshold": 4.0, "n_train_scores": 100,
        }
    ).to_csv(statistics_path, index=False)

    environment = {
        "MTGFLOW_SEED_DIR": str(mtg_dir),
        "STGAN_SEED_DIR": str(stgan_dir),
        "SDE_MULTIHORIZON_PREDICTIONS": str(prediction_path),
        "PVGIS_2019_PATH": str(pvgis_path),
        "MTGFLOW_TRAINING_STATS_CSV": str(statistics_path),
        "SPATIAL_COMPARISON_OUT_DIR": str(spatial_out),
        "ANOMALY_SENSITIVITY_OUT_DIR": str(sensitivity_out),
        "ANOMALY_SUMMARY_OUT_DIR": str(summary_out),
        "MPLBACKEND": "Agg",
    }
    previous = {key: os.environ.get(key) for key in environment}
    original_cwd = Path.cwd()
    try:
        os.environ.update(environment)
        os.chdir(ROOT)
        for name in (
            "spatial_anomaly_comparison_mtgflow_stgan.ipynb",
            "anomaly_threshold_sensitivity_mtgflow_stgan.ipynb",
            "anomaly_analysis_results_summary.ipynb",
        ):
            path = ROOT / "notebooks" / name
            notebook = json.loads(path.read_text(encoding="utf-8"))
            namespace = {"__name__": "__notebook_smoke__", "display": lambda *args: None}
            for number, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    exec(
                        compile("".join(cell["source"]), f"{path}:cell-{number}", "exec"),
                        namespace,
                    )
    finally:
        os.chdir(original_cwd)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    assert (spatial_out / "reference_neighbourhood.csv").is_file()
    assert len(list((spatial_out / "figures").glob("*_spatial_comparison_*.png"))) == 6
    assert (sensitivity_out / "detector_threshold_sensitivity_metrics.csv").is_file()
    assert (sensitivity_out / "figures/mtgflow_mae_rmse_sensitivity_t1_t6.png").is_file()
    assert (sensitivity_out / "figures/stgan_mae_rmse_sensitivity_t1_t6.png").is_file()
    assert (summary_out / "event_detector_summary.csv").is_file()
    assert (summary_out / "reference_detector_forecast_summary.csv").is_file()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        test_spatial_comparison_preserves_locations_and_has_separate_neighbourhood(
            Path(directory)
        )
    with tempfile.TemporaryDirectory() as directory:
        test_threshold_sensitivity_uses_exact_mtgflow_k_and_stgan_top_percent(
            Path(directory)
        )
    test_new_notebooks_are_valid_posthoc_wrappers()
    with tempfile.TemporaryDirectory() as directory:
        test_new_notebooks_execute_in_order_on_synthetic_data(Path(directory))
    print("PASS: detector spatial and threshold comparison tests")
