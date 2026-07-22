"""Synthetic CPU tests for paper-faithful M2AD and its PVGIS adapter."""

from __future__ import annotations

from pathlib import Path
import sys
import json
import tempfile

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import xarray as xr

from physiq_pv.anomaly_detection.m2ad import M2AD
from physiq_pv.anomaly_detection.m2ad_calibration import M2ADCalibrator
from physiq_pv.anomaly_detection.m2ad_errors import (
    M2ADDiscrepancy,
    compute_discrepancy,
)
from physiq_pv.anomaly_detection.m2ad_forecaster import (
    M2ADForecaster,
    M2ADWindowDataset,
)
from physiq_pv.anomaly_detection.m2ad_preprocessing import M2ADPreprocessor
from physiq_pv.data.pvgis_m2ad import (
    DEFAULT_M2AD_SENSORS,
    extract_location_segments,
    prepare_pvgis_years,
)
from physiq_pv.experiments.pvgis_m2ad_runner import (
    build_arg_parser,
    parse_years,
    run_from_args,
)
from physiq_pv.experiments.pvgis_m2ad_pipeline import PVGISM2ADConfig
from physiq_pv.reporting.m2ad_outputs import merge_detection_intervals


def _segment(length: int, phase: float = 0.0) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)
    daily = np.sin(2 * np.pi * t / 24.0 + phase)
    return np.column_stack(
        [
            daily + 0.08 * np.sin(2 * np.pi * t / 7.0),
            0.6 * daily + 0.2 * np.cos(2 * np.pi * t / 12.0 + phase),
            0.01 * t + np.cos(2 * np.pi * t / 48.0),
        ]
    )


def _detector(error: str = "point") -> M2AD:
    return M2AD(
        ["pv", "poa", "temperature"],
        window_size=12,
        hidden_size=8,
        n_layers=1,
        dropout=0.0,
        error=error,
        area_half_window=2,
        ewma_com=2.0,
        n_components=1,
        epochs=3,
        batch_size=32,
        validation_split=0.2,
        patience=3,
        device="cpu",
        seed=7,
        verbose=False,
    )


def test_window_target_is_immediate_next_sample_and_years_do_not_cross() -> None:
    first = np.arange(20, dtype=np.float32)[:, None]
    second = (100 + np.arange(20, dtype=np.float32))[:, None]
    dataset = M2ADWindowDataset([first, second], window_size=4, horizon=1)
    window, target, segment_id, target_index = dataset[0]
    np.testing.assert_array_equal(window[:, 0], [0, 1, 2, 3])
    assert target.item() == 4
    assert segment_id == 0 and target_index == 4
    assert len(dataset) == 2 * (20 - 4)
    last_first = 20 - 4 - 1
    assert dataset[last_first][2] == 0
    assert dataset[last_first + 1][2] == 1


def test_point_and_area_discrepancies_are_finite_and_directional() -> None:
    observed = np.zeros((30, 2))
    predicted = np.zeros_like(observed)
    observed[12:18, 0] = 2.0
    point = compute_discrepancy(observed, predicted, error="point", ewma_com=None)
    area = compute_discrepancy(
        observed, predicted, error="area", area_half_window=2, ewma_com=None
    )
    assert np.isfinite(point).all() and np.isfinite(area).all()
    assert (point >= 0).all()
    assert area[15, 0] > 0
    assert np.allclose(area[:, 1], 0.0)


def test_fit_score_is_label_unaware_leakage_safe_and_interpretable() -> None:
    train = [_segment(180, 0.0), _segment(180, 0.2)]
    detector = _detector().fit(train)
    train_min = detector.data_min.copy()
    train_max = detector.data_max.copy()

    test = _segment(100, 0.1)
    test[60:64, 0] += 20.0
    scores = detector.score_segments([test])

    # Inference does not refit preprocessing even when values exceed train range.
    np.testing.assert_array_equal(detector.data_min, train_min)
    np.testing.assert_array_equal(detector.data_max, train_max)
    assert np.isfinite(scores.global_scores).all()
    assert np.isfinite(scores.gamma_p_values).all()
    np.testing.assert_allclose(
        scores.weighted_contributions.sum(axis=1), scores.global_scores
    )

    timestamps = [pd.date_range("2019-01-01", periods=len(test), freq="h")]
    frame = detector.scores_frame(scores, timestamps, entity="loc_a")
    required = {
        "location",
        "timestamp",
        "global_score",
        "gamma_p_value",
        "is_anomaly",
        "top_sensor",
        "observed__pv",
        "predicted__pv",
        "contribution__pv",
    }
    assert required <= set(frame.columns)
    np.testing.assert_allclose(
        frame[[f"contribution__{name}" for name in detector.sensor_names]].sum(axis=1),
        1.0,
    )
    assert frame.loc[frame["timestamp"].between("2019-01-03 12:00", "2019-01-03 15:00"),
                     "global_score"].mean() > frame["global_score"].median()


def test_facade_is_composed_from_independent_public_stages() -> None:
    detector = _detector(error="area")
    assert isinstance(detector.preprocessor, M2ADPreprocessor)
    assert isinstance(detector.forecaster, M2ADForecaster)
    assert isinstance(detector.discrepancy, M2ADDiscrepancy)
    assert isinstance(detector.calibrator, M2ADCalibrator)

    preprocessor = M2ADPreprocessor(3)
    scaled = preprocessor.fit_transform([_segment(40)])
    restored = preprocessor.inverse_transform(scaled[0])
    np.testing.assert_allclose(restored, _segment(40), atol=1e-6)

    rng = np.random.default_rng(4)
    errors = np.abs(rng.normal(size=(200, 3)))
    calibrator = M2ADCalibrator(
        ["pv", "poa", "temperature"],
        error="point",
        n_components=1,
        significance=0.01,
        seed=4,
    ).fit(errors)
    result = calibrator.score(errors[:10])
    assert len(result) == 6
    assert result[0].shape == (10, 3)
    assert result[3].shape == result[4].shape == result[5].shape == (10,)


def _tiny_pvgis(year: int) -> xr.Dataset:
    time = pd.date_range(f"{year}-01-01", periods=48, freq="h")
    shape = (2, len(time))
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return xr.Dataset(
        {
            "pv_power_output": (("location", "time"), base),
            "direct_irradiance_tilted": (("location", "time"), base * 0.7),
            "diffuse_irradiance_tilted": (("location", "time"), base * 0.3),
            "temperature_2m": (("location", "time"), base * 0.1),
            "wind_speed_10m": (("location", "time"), base * 0.01),
        },
        coords={"location": ["a", "b"], "time": time},
    )


def test_pvgis_adapter_preserves_separate_years_and_effective_poa() -> None:
    datasets = prepare_pvgis_years(
        {2005: _tiny_pvgis(2005), 2006: _tiny_pvgis(2006)}, DEFAULT_M2AD_SENSORS
    )
    segments, timestamps = extract_location_segments(datasets, "a", DEFAULT_M2AD_SENSORS)
    assert len(segments) == len(timestamps) == 2
    assert segments[0].shape == (48, 4)
    np.testing.assert_allclose(segments[0][:, 1], segments[0][:, 0])
    assert timestamps[0].year.min() == timestamps[0].year.max() == 2005
    assert timestamps[1].year.min() == timestamps[1].year.max() == 2006


def test_year_parser_and_interval_merging() -> None:
    assert parse_years("2005-2007,2009") == [2005, 2006, 2007, 2009]
    detections = pd.DataFrame(
        {
            "location": ["a", "a", "a"],
            "timestamp": pd.to_datetime(
                ["2019-01-01 00:00", "2019-01-01 01:00", "2019-01-01 03:00"]
            ),
            "global_score": [10.0, 12.0, 11.0],
            "gamma_p_value": [1e-4, 1e-5, 1e-4],
        }
    )
    intervals = merge_detection_intervals(detections)
    assert intervals["n_points"].tolist() == [2, 1]
    assert intervals["max_global_score"].tolist() == [12.0, 11.0]


def test_typed_pipeline_config_builds_detector_without_argparse() -> None:
    config = PVGISM2ADConfig(
        pvgis_dir="unused-in-this-unit-test",
        train_years=(2005, 2006),
        test_year=2019,
        sensors=("pv", "poa", "temperature"),
        window_size=12,
        hidden_size=8,
        n_layers=1,
        epochs=1,
        device="cpu",
        verbose=False,
    )
    detector = config.detector()
    assert isinstance(detector, M2AD)
    assert detector.sensor_names == ["pv", "poa", "temperature"]


def test_pvgis_runner_end_to_end(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    _tiny_pvgis(2005).to_netcdf(data_dir / "piedmont_pvgis_2005.nc")
    _tiny_pvgis(2019).to_netcdf(data_dir / "piedmont_pvgis_2019.nc")
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--pvgis-dir", str(data_dir),
            "--train-years", "2005",
            "--test-year", "2019",
            "--out-dir", str(out_dir),
            "--max-locations", "1",
            "--window-size", "8",
            "--hidden-size", "4",
            "--n-layers", "1",
            "--dropout", "0",
            "--error", "point",
            "--ewma-com", "1",
            "--gmm-components", "1",
            "--epochs", "1",
            "--batch-size", "16",
            "--device", "cpu",
            "--quiet",
        ]
    )
    paths = run_from_args(args, parser)
    assert all(path.exists() for path in paths.values())
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    assert meta["label_unaware"] is True
    assert meta["train_years"] == [2005] and meta["test_year"] == 2019
    scores = pd.read_csv(paths["scores"])
    assert len(scores) == 48 - 8
    assert {"global_score", "gamma_p_value", "top_sensor"} <= set(scores.columns)


if __name__ == "__main__":
    test_window_target_is_immediate_next_sample_and_years_do_not_cross()
    test_point_and_area_discrepancies_are_finite_and_directional()
    test_fit_score_is_label_unaware_leakage_safe_and_interpretable()
    test_facade_is_composed_from_independent_public_stages()
    test_pvgis_adapter_preserves_separate_years_and_effective_poa()
    test_year_parser_and_interval_merging()
    test_typed_pipeline_config_builds_detector_without_argparse()
    with tempfile.TemporaryDirectory() as directory:
        test_pvgis_runner_end_to_end(Path(directory))
    print("PASS: M2AD tests")
