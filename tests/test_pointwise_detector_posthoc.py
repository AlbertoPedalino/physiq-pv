from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physiq_pv.reporting.pointwise_detector_posthoc import (
    build_pointwise_detector_evaluation,
)
from physiq_pv.reporting.anomaly_extremes import (
    rank_flagged_days,
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


if __name__ == "__main__":
    test_pointwise_stgan_join_excludes_unscored_rows_and_has_no_event_group()
    test_pointwise_stgan_join_rejects_low_overlap()
    test_pointwise_stgan_join_supports_one_direct_multihorizon_run()
    test_stgan_regional_series_uses_saved_binary_decision()
    test_stgan_notebook_uses_one_direct_multihorizon_prediction_file()
    test_stgan_extreme_event_notebook_is_pointwise_and_t6()
    print("PASS: pointwise detector post-hoc tests")
