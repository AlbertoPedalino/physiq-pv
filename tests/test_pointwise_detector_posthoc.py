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


def test_stgan_notebook_evaluates_only_t_plus_one() -> None:
    path = ROOT / "notebooks" / "stgan_pointwise_posthoc_sdenet.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "'horizon': 1" in source
    assert "SDE_PREDICTIONS_T1" in source
    assert "SDE_PREDICTIONS_T6" not in source
    assert "SDE_PREDICTIONS_T12" not in source
    assert "FORECAST_HORIZONS" not in source
    assert "HORIZON_CONFIGS" not in source
    assert "build_horizon_comparison_figures" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(path), "exec")


if __name__ == "__main__":
    test_pointwise_stgan_join_excludes_unscored_rows_and_has_no_event_group()
    test_pointwise_stgan_join_rejects_low_overlap()
    test_stgan_notebook_evaluates_only_t_plus_one()
    print("PASS: pointwise detector post-hoc tests")
