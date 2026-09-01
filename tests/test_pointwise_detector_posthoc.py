from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physiq_pv.reporting.pointwise_detector_posthoc import (
    build_pointwise_detector_evaluation,
    detect_isolated_regional_solar_dropouts,
)
from physiq_pv.reporting.anomaly_extremes import (
    DaytimeFilter,
    rank_flagged_days,
    regional_extreme_series,
    regional_flag_series,
)
from physiq_pv.reporting.posthoc_outputs import build_direct_multihorizon_posthoc


def _write_inputs(root: Path) -> tuple[Path, Path]:
    prediction_path = root / "sde_predictions.csv"
    score_path = root / "stgan_scores.csv"
    pd.DataFrame(
        {
            "location": ["a", "b", "a", "b"],
            "timestamp": [
                "2019-01-01 01:00:00",
                "2019-01-01 01:00:00",
                "2019-01-01 02:00:00",
                "2019-01-01 02:00:00",
            ],
            "y_true": [1.0, 2.0, 3.0, 4.0],
            "y_pred_mean": [1.0, 1.0, 2.0, 2.0],
            "y_pred_std": [0.1, 0.1, 0.2, 0.2],
            "solar_irradiance_poa_target": [100.0] * 4,
            "lower_pi": [0.0] * 4,
            "upper_pi": [5.0] * 4,
            # Stale labels from another detector must be replaced, not reused.
            "anomaly_group": ["rare_or_extreme"] * 4,
            "event_group": ["rare_or_extreme"] * 4,
        }
    ).to_csv(prediction_path, index=False)
    pd.DataFrame(
        {
            "location": ["a", "b", "a"],
            "timestamp": [
                "2019-01-01T01:00:00Z",
                "2019-01-01T01:00:00Z",
                "2019-01-01T02:00:00Z",
            ],
            "method": ["stgan"] * 3,
            "anomaly_score": [0.1, 2.0, 0.2],
            "threshold": [1.0] * 3,
            "is_anomaly": [False, True, False],
        }
    ).to_csv(score_path, index=False)
    return prediction_path, score_path


def test_pointwise_stgan_join_excludes_unscored_rows_and_has_no_event_group() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        predictions, scores = _write_inputs(root)
        result = build_pointwise_detector_evaluation(
            predictions,
            scores,
            root / "evaluation",
            min_match_fraction=0.70,
        )
        joined = pd.read_csv(result["predictions"])
        metadata = json.loads(Path(result["evaluation_source"]).read_text())

    assert len(joined) == 3
    assert "event_group" not in joined
    assert joined["anomaly_group"].tolist() == [
        "normal",
        "rare_or_extreme",
        "normal",
    ]
    assert joined["anomaly_label"].fillna("").tolist() == ["", "stgan", ""]
    assert metadata["join_keys"] == ["location", "timestamp"]
    assert metadata["excluded_unmatched_rows"] == 1
    assert metadata["regional_event_group_created"] is False


def test_pointwise_stgan_join_rejects_low_overlap() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        predictions, scores = _write_inputs(root)
        try:
            build_pointwise_detector_evaluation(
                predictions,
                scores,
                root / "evaluation",
                min_match_fraction=0.90,
            )
        except ValueError as exc:
            assert "Insufficient exact detector/SDE overlap" in str(exc)
        else:
            raise AssertionError("Low-overlap pointwise joins must be rejected.")


def test_pointwise_stgan_join_supports_one_direct_multihorizon_run() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        predictions, scores = _write_inputs(root)
        base = pd.read_csv(predictions)
        direct = pd.concat(
            [base.assign(horizon_hours=h) for h in (1, 2)], ignore_index=True
        )
        direct["issue_timestamp"] = pd.to_datetime(direct["timestamp"]) - pd.to_timedelta(
            direct["horizon_hours"], unit="h"
        )
        direct.to_csv(predictions, index=False)

        evaluation_dir = root / "evaluation"
        result = build_pointwise_detector_evaluation(
            predictions,
            scores,
            evaluation_dir,
            min_match_fraction=0.70,
        )
        joined = pd.read_csv(result["predictions"])
        metadata = json.loads(Path(result["evaluation_source"]).read_text())
        posthoc = build_direct_multihorizon_posthoc(
            evaluation_dir,
            location="a",
            start="2019-01-01",
            end="2019-01-02",
        )
        posthoc_metadata = json.loads(posthoc["metadata"].read_text())

    assert len(joined) == 6
    assert joined.groupby(["location", "timestamp"]).size().max() == 2
    assert metadata["join_keys"] == ["location", "timestamp"]
    assert metadata["prediction_row_key"] == [
        "location",
        "timestamp",
        "horizon_hours",
    ]
    assert metadata["forecast_mode"] == "direct_multi_output"
    assert metadata["horizons_hours"] == [1, 2]
    assert posthoc_metadata["detector"] == "stgan"


def test_quality_filter_excludes_dropout_and_recovery_then_reranks_top_k() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        times = pd.date_range("2019-06-12 07:10:00", periods=4, freq="h")
        locations = ["a", "b"]
        positive = np.array(
            [[100.0, 120.0], [0.0, 0.0], [300.0, 320.0], [400.0, 420.0]]
        )
        dataset = xr.Dataset(
            {
                "direct_irradiance_tilted": (("time", "location"), positive),
                "diffuse_irradiance_tilted": (("time", "location"), positive / 2),
                "sun_height": (("time", "location"), positive / 10),
                "pv_power_output": (("time", "location"), positive * 0.8),
            },
            coords={"time": times, "location": locations},
        )
        netcdf = root / "pvgis_2019.nc"
        dataset.to_netcdf(netcdf)
        prepared = root / "prepared"
        prepared.mkdir()
        manifest_rows = []
        for location_index, location in enumerate(locations):
            test_csv = prepared / f"{location}_test.csv"
            pd.DataFrame(
                {
                    "timestamp": times,
                    "solar_irradiance_poa": positive[:, location_index] * 1.5,
                }
            ).to_csv(test_csv, index=False)
            manifest_rows.append({"location": location, "test_csv": test_csv})
        manifest_path = prepared / "manifest.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

        rows = pd.MultiIndex.from_product(
            [locations, times], names=["location", "timestamp"]
        ).to_frame(index=False)
        predictions = rows.assign(
            issue_timestamp=lambda frame: frame["timestamp"] - pd.Timedelta(hours=1),
            horizon_hours=1,
            y_true=1.0,
            y_pred_mean=1.0,
        )
        prediction_path = root / "predictions.csv"
        predictions.to_csv(prediction_path, index=False)
        scores = rows.assign(
            method="stgan",
            anomaly_score=[1.0, 100.0, 90.0, 4.0, 2.0, 80.0, 70.0, 3.0],
            global_rank=[8, 1, 2, 4, 7, 3, 5, 6],
            threshold=np.nan,
            is_anomaly=[False, True, True, False, False, True, False, False],
        )
        score_path = root / "scores.csv"
        scores.to_csv(score_path, index=False)

        issues = detect_isolated_regional_solar_dropouts(netcdf)
        manifest_issues = detect_isolated_regional_solar_dropouts(manifest_path)
        result = build_pointwise_detector_evaluation(
            prediction_path,
            score_path,
            root / "evaluation",
            detector_name="stgan",
            pvgis_quality_source=netcdf,
            clean_top_k_percent=50.0,
        )
        clean = pd.read_csv(result["predictions"])
        quality = pd.read_csv(result["data_quality_predictions"])
        metadata = json.loads(Path(result["evaluation_source"]).read_text())

    assert issues["quality_issue"].tolist() == [
        "regional_solar_dropout",
        "recovery_after_regional_solar_dropout",
    ]
    assert pd.to_datetime(issues["timestamp"]).tolist() == [times[1], times[2]]
    pd.testing.assert_frame_equal(issues, manifest_issues)
    assert set(pd.to_datetime(clean["timestamp"])) == {times[0], times[3]}
    assert set(pd.to_datetime(quality["timestamp"])) == {times[1], times[2]}
    assert len(clean) == 4
    assert int(clean["detector_is_anomaly"].sum()) == 2
    assert metadata["solar_dropout_timestamps"] == 1
    assert metadata["data_quality_timestamps"] == 2
    assert metadata["excluded_data_quality_rows"] == 4
    assert metadata["clean_top_k_percent"] == 50.0
    assert metadata["eligible_detector_coordinates"] == 4
    assert metadata["data_quality_detector_coordinates"] == 4
    assert metadata["clean_detector_anomalies"] == 2


def test_stgan_regional_series_uses_saved_binary_decision() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        score_path = Path(temporary) / "scores.csv"
        pd.DataFrame({
            "location": ["a", "b", "a", "b"],
            "timestamp": [
                "2019-07-01 10:00:00", "2019-07-01 10:00:00",
                "2019-07-02 10:00:00", "2019-07-02 10:00:00",
            ],
            # Scores deliberately disagree with a naive score >= threshold
            # reconstruction: the exported detector decision is authoritative.
            "anomaly_score": [100.0, 100.0, 0.0, 0.0],
            "threshold": [1.0] * 4,
            "is_anomaly": [False, True, True, True],
        }).to_csv(score_path, index=False)
        series = regional_flag_series(score_path, chunksize=1)
        days = rank_flagged_days(series)

    assert series.attrs["decision_source"] == "saved_is_anomaly"
    assert series["n_extreme"].tolist() == [1, 2]
    assert series["extreme_share"].tolist() == [0.5, 1.0]
    assert days.iloc[0]["day"] == pd.Timestamp("2019-07-02")
    assert int(days.iloc[0]["n_anomalies"]) == 2


def test_regional_series_use_exact_pvgis_daytime_coordinates() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        score_path = Path(temporary) / "scores.csv"
        pd.DataFrame({
            "location": ["a", "b", "a", "b"],
            "timestamp": [
                "2019-07-01 10:00:00", "2019-07-01 10:00:00",
                "2019-07-01 11:00:00", "2019-07-01 11:00:00",
            ],
            "anomaly_score": [1.0, 100.0, 100.0, 4.0],
            "is_anomaly": [False, True, True, True],
        }).to_csv(score_path, index=False)
        daytime = DaytimeFilter.from_pvgis(
            pd.DataFrame({"location": ["a", "b"]}),
            pd.to_datetime(["2019-07-01 10:00:00", "2019-07-01 11:00:00"]),
            np.array([[20.0, 0.0], [0.0, 20.0]]),
        )
        flags = regional_flag_series(
            score_path, chunksize=1, daytime_filter=daytime
        )
        tail = regional_extreme_series(
            score_path, quantile=0.5, chunksize=1, daytime_filter=daytime
        )

    assert flags["n_scored"].tolist() == [1, 1]
    assert flags["n_extreme"].tolist() == [0, 1]
    assert flags.attrs["daytime_threshold_wm2"] == 10.0
    assert tail.attrs["cut"] == 2.5
    assert tail["n_extreme"].tolist() == [0, 1]


def test_stgan_notebook_uses_one_direct_multihorizon_prediction_file() -> None:
    path = ROOT / "notebooks" / "stgan_pointwise_posthoc_sdenet.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "FORECAST_HORIZONS = pipe.FORECAST_HORIZONS" in source
    assert "SDE_MULTIHORIZON_PREDICTIONS" in source
    assert "build_direct_multihorizon_posthoc" in source
    assert "DETAILED_HORIZONS = (1, 6)" in source
    assert "horizon_hours=horizon_hours" in source
    assert "posthoc_by_horizon" in source
    assert "build_posthoc_figures" in source
    assert "FULL_POSTHOC_FIGURES" in source
    assert "PVGIS_2019_FILE" in source
    assert "STGAN_PREPARED_MANIFEST" in source
    assert "pvgis_quality_source=PVGIS_QUALITY_SOURCE" in source
    assert "clean_top_k_percent=CLEAN_TOP_K_PERCENT" in source
    assert "stgan_clean_daytime_timestamp_ranking.csv" in source
    assert "data_quality_predictions.csv" in source
    for prefix in ("mae_", "rmse_", "nmpil_", "picp_", "clc_"):
        assert prefix in source
    assert "SDE_PREDICTIONS_T1" not in source
    assert "HORIZON_CONFIGS" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(path), "exec")


def test_stgan_extreme_event_notebook_is_pointwise_and_t6() -> None:
    path = ROOT / "notebooks" / "stgan_extreme_events_t6.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "FORECAST_HORIZON = 6" in source
    assert "regional_flag_series(STGAN_SCORES)" in source
    assert "horizon_hours=FORECAST_HORIZON" in source
    assert "reference_peak_path=REFERENCE_PEAK" in source
    assert "is_anomaly" in source
    assert "pvgis_mtgflow" not in source.lower()
    assert "MTGFLOW_SCORES" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(path), "exec")


def test_stgan_selected_event_notebook_compares_t1_and_t6() -> None:
    path = ROOT / "notebooks" / "stgan_may08_may17_t1_t6.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "HORIZONS = (1, 6)" in source
    assert "'2019-05-08'" in source
    assert "'2019-05-17'" in source
    assert "detector_is_anomaly" in source
    assert "quality_filtered" in source
    assert "clean_top_k_percent" in source
    assert "solar_irradiance_poa_target" in source
    assert "build_extreme_event_diagnostic" in source
    assert "build_extreme_event_comparison_figures" in source
    assert "generate_figures=False" in source
    assert "{metric_name}_by_bin_t_plus_{horizon_hours}.png" in source
    assert "('mae', 'MAE [W]')" in source
    assert "('rmse', 'RMSE [W]')" in source
    assert "('picp', 'PICP')" in source
    assert "('nmpil', 'NMPIL')" in source
    assert "('clc', 'CLC')" in source
    assert "saved_figure_count" in source
    assert "len(figure_manifest) != 15" in source
    assert "BEGIN_STGAN_EVENT_BIN_METRICS_CSV" in source
    assert "BEGIN_STGAN_SELECTED_DAYS_CSV" in source
    assert "BEGIN_STGAN_TEMPORAL_DIAGNOSTICS_CSV" in source
    assert "may_08_stgan_clean_rank_1" not in source
    assert "may_17_stgan_clean_rank_2" not in source
    assert "horizon_hours=horizon_hours" in source
    assert "['location', 'timestamp']" in source
    assert "timestamps_are_target_times" in source
    assert "train_model(" not in source
    assert "RUN_TRAINING" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert not cell.get("outputs")
            compile("".join(cell["source"]), str(path), "exec")


if __name__ == "__main__":
    test_pointwise_stgan_join_excludes_unscored_rows_and_has_no_event_group()
    test_pointwise_stgan_join_rejects_low_overlap()
    test_pointwise_stgan_join_supports_one_direct_multihorizon_run()
    test_quality_filter_excludes_dropout_and_recovery_then_reranks_top_k()
    test_stgan_regional_series_uses_saved_binary_decision()
    test_stgan_notebook_uses_one_direct_multihorizon_prediction_file()
    test_stgan_extreme_event_notebook_is_pointwise_and_t6()
    test_stgan_selected_event_notebook_compares_t1_and_t6()
    print("PASS: pointwise detector post-hoc tests")
