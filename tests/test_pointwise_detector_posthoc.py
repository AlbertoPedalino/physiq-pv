from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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


def test_posthoc_reuse_skips_csv_io_and_rebuilds_changed_inputs() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        predictions, scores = _write_inputs(root)
        output = root / "evaluation"
        options = dict(min_match_fraction=0.7, reuse_existing=True, allow_overwrite=True)
        first = build_pointwise_detector_evaluation(predictions, scores, output, **options)
        assert first["reused"] is False
        mtime = first["predictions"].stat().st_mtime_ns
        with patch.object(pd, "read_csv", side_effect=AssertionError("Cache must not read large CSVs")):
            again = build_pointwise_detector_evaluation(predictions, scores, output, **options)
        assert again["reused"] is True
        assert first["predictions"].stat().st_mtime_ns == mtime

        # A source changed at the same path must invalidate reuse.
        frame = pd.read_csv(predictions)
        frame["y_pred_mean"] = 0.125
        frame.to_csv(predictions, index=False)
        updated = build_pointwise_detector_evaluation(predictions, scores, output, **options)
        assert updated["reused"] is False
        assert (pd.read_csv(updated["predictions"])["y_pred_mean"] == 0.125).all()
        frame = pd.read_csv(scores)
        frame["is_anomaly"] = False
        frame.to_csv(scores, index=False)
        updated = build_pointwise_detector_evaluation(predictions, scores, output, **options)
        assert updated["reused"] is False
        assert updated["rare_rows"] == 0
        assert build_pointwise_detector_evaluation(
            predictions, scores, output, **options
        )["reused"] is True

        options["min_match_fraction"] = 0.6
        assert build_pointwise_detector_evaluation(
            predictions, scores, output, **options
        )["reused"] is False
        assert build_pointwise_detector_evaluation(
            predictions, scores, output, force_rebuild=True, **options
        )["reused"] is False
        assert build_pointwise_detector_evaluation(
            predictions, scores, output, **options
        )["reused"] is True
        assert [p.name for p in root.iterdir() if p.is_dir()] == ["evaluation"]


def test_posthoc_reuse_recovers_missing_partial_and_interrupted_outputs() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        predictions, scores = _write_inputs(root)
        output = root / "evaluation"
        options = dict(min_match_fraction=0.7, reuse_existing=True, allow_overwrite=True)
        result = build_pointwise_detector_evaluation(predictions, scores, output, **options)
        (output / "figure.png").write_bytes(b"keep image")
        for key in ("predictions", "data_quality_predictions", "metrics", "evaluation_source"):
            result[key].unlink()
            result = build_pointwise_detector_evaluation(predictions, scores, output, **options)
            assert result["reused"] is False
            assert result[key].is_file()
        result["predictions"].write_text("truncated", encoding="utf-8")
        assert build_pointwise_detector_evaluation(
            predictions, scores, output, **options
        )["reused"] is False
        with patch.object(pd.DataFrame, "to_csv", side_effect=RuntimeError("Simulated interruption")):
            try:
                build_pointwise_detector_evaluation(predictions, scores, output, force_rebuild=True, **options)
            except RuntimeError as exc:
                assert "Simulated interruption" in str(exc)
            else:
                raise AssertionError("Expected simulated write failure")
        assert not (output / "evaluation_source.json").exists()
        assert build_pointwise_detector_evaluation(
            predictions, scores, output, **options
        )["reused"] is False
        assert (output / "figure.png").read_bytes() == b"keep image"


def test_posthoc_reuse_tracks_prepared_quality_files_and_threshold() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        predictions, scores = _write_inputs(root)
        quality_csv = root / "quality.csv"
        pd.DataFrame({
            "timestamp": pd.date_range("2019-01-01", periods=4, freq="h"),
            "solar_irradiance_poa": [100.0] * 4,
        }).to_csv(quality_csv, index=False)
        manifest = root / "manifest.csv"
        pd.DataFrame({"test_csv": ["quality.csv"]}).to_csv(manifest, index=False)
        options = dict(min_match_fraction=0.7, reuse_existing=True, allow_overwrite=True,
                       pvgis_quality_source=manifest, clean_top_k_percent=50.0)
        output = root / "evaluation"
        build_pointwise_detector_evaluation(predictions, scores, output, **options)
        assert build_pointwise_detector_evaluation(
            predictions, scores, output, **options
        )["reused"] is True
        with quality_csv.open("a", encoding="utf-8") as stream:
            stream.write("2019-01-01 04:00:00,100.0\n")
        assert build_pointwise_detector_evaluation(
            predictions, scores, output, **options
        )["reused"] is False
        options["clean_top_k_percent"] = 25.0
        assert build_pointwise_detector_evaluation(
            predictions, scores, output, **options
        )["reused"] is False


def test_posthoc_cannot_overwrite_original_predictions() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        predictions, scores = _write_inputs(root)
        predictions = predictions.rename(root / "predictions.csv")
        original = predictions.read_bytes()
        try:
            build_pointwise_detector_evaluation(
                predictions, scores, root, allow_overwrite=True, reuse_existing=True
            )
        except ValueError as exc:
            assert "must not overwrite source" in str(exc)
        else:
            raise AssertionError("Source predictions must be protected")
        assert predictions.read_bytes() == original


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


def test_stgan_posthoc_configuration_isolates_cnn_runs() -> None:
    notebook = json.loads(
        (ROOT / "notebooks/stgan_pointwise_posthoc_sdenet.ipynb").read_text(encoding="utf-8")
    )
    config = next(
        "".join(cell["source"]) for cell in notebook["cells"]
        if "FORECAST_HORIZONS = pipe.FORECAST_HORIZONS" in "".join(cell["source"])
    )
    import os

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        namespace = {
            "ROOT": root, "Path": Path, "os": os,
            "pipe": SimpleNamespace(
                FORECAST_HORIZONS=(1, 2, 3, 4, 5, 6), DEFAULT_CONFIG={},
                make_out_dir=lambda config: "outputs/sde_original",
            ),
        }
        with patch.dict(os.environ, {}, clear=True):
            exec(config, namespace)
            assert namespace["STGAN_SCORES"] == (
                root / "outputs/pvgis_stgan_cnn/convgru_reference/seed_20/anomaly_scores.csv"
            )
            first = namespace["EVALUATION_DIR"]
            first.mkdir(parents=True)
            sentinel = first / "evaluation_source.json"
            sentinel.write_text('{"old": true}', encoding="utf-8")
            exec(config, namespace)
            assert namespace["EVALUATION_DIR"] == first
            assert namespace["EVALUATION_DIR"].parent == root / "outputs/sde_stgan_cnn_quality_filtered"
            assert namespace["EVALUATION_DIR"].exists()
            assert sentinel.read_text(encoding="utf-8") == '{"old": true}'

        custom_training = root / "custom_training"
        custom_seed = root / "another_training/seed_42"
        with patch.dict(os.environ, {
            "STGAN_CNN_OUT_DIR": str(custom_training),
            "STGAN_POSTHOC_ROOT": str(first),
        }, clear=True):
            exec(config, namespace)
            assert namespace["STGAN_SEED_DIR"] == custom_training / "seed_20"
            assert namespace["EVALUATION_DIR"].parent == first
            assert not namespace["EVALUATION_DIR"].exists()
            with patch.dict(os.environ, {"STGAN_SEED_DIR": str(custom_seed)}):
                exec(config, namespace)
                assert namespace["STGAN_SEED_DIR"] == custom_seed

        # Running the actual notebook join twice reuses the CSV without rewriting it.
        predictions, scores = _write_inputs(root)
        relabel = next(
            "".join(cell["source"]) for cell in notebook["cells"]
            if "RELABEL_RESULT = build_pointwise_detector_evaluation(" in "".join(cell["source"])
        )
        namespace.update(
            SDE_PREDICTIONS=predictions, STGAN_SCORES=scores, EVALUATION_DIR=first,
            build_pointwise_detector_evaluation=build_pointwise_detector_evaluation,
            PVGIS_QUALITY_SOURCE=None, CLEAN_TOP_K_PERCENT=None, MIN_MATCH_FRACTION=0.7,
            json=json, pd=pd, display=lambda value: None,
        )
        exec(relabel, namespace)
        assert namespace["RELABEL_RESULT"]["reused"] is False
        generated = first / "predictions.csv"
        original_mtime = generated.stat().st_mtime_ns
        exec(relabel, namespace)
        assert namespace["RELABEL_RESULT"]["reused"] is True
        assert generated.stat().st_mtime_ns == original_mtime

        # Run All explicitly selects CNN kernel 3 at top-5%, even in a kernel
        # previously used for another detector. Generic config supports both.
        selection = next(
            "".join(cell["source"]) for cell in notebook["cells"]
            if cell.get("id") == "selected-stgan-run"
        )
        with patch.dict(os.environ, {
            "STGAN_SEED_DIR": str(custom_seed),
            "STGAN_POSTHOC_ROOT": str(first),
        }, clear=True):
            exec(selection, namespace)
            exec(config, namespace)
            assert namespace["KERNEL_SIZE"] == 3
            assert namespace["STGAN_SCORES"] == (
                root / "outputs/pvgis_stgan_cnn/convgru_reference/seed_20/anomaly_scores.csv"
            )
            assert namespace["EVALUATION_DIR"].parent == (
                root / "outputs/sde_stgan_cnn_quality_filtered"
            )
            assert namespace["CLEAN_TOP_K_PERCENT"] == 5.0
            assert namespace["EVALUATION_DIR"].name == "convgru_reference_seed_20_top5pct"
            assert namespace["EVALUATION_DIR"] == first


if __name__ == "__main__":
    test_pointwise_stgan_join_excludes_unscored_rows_and_has_no_event_group()
    test_pointwise_stgan_join_rejects_low_overlap()
    test_pointwise_stgan_join_supports_one_direct_multihorizon_run()
    test_quality_filter_excludes_dropout_and_recovery_then_reranks_top_k()
    test_posthoc_reuse_skips_csv_io_and_rebuilds_changed_inputs()
    test_posthoc_reuse_recovers_missing_partial_and_interrupted_outputs()
    test_posthoc_reuse_tracks_prepared_quality_files_and_threshold()
    test_posthoc_cannot_overwrite_original_predictions()
    test_stgan_notebook_uses_one_direct_multihorizon_prediction_file()
    test_stgan_posthoc_configuration_isolates_cnn_runs()
    print("PASS: pointwise detector post-hoc tests")
