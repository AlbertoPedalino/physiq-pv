"""Build detector-specific, pointwise post-hoc inputs for SDE forecasts.

The detector decision is matched on the exact ``(location, timestamp)`` pair.
Regional timestamp labels are deliberately neither created nor consumed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from physiq_pv.reporting.run_metrics import build_wandb_metrics, compute_metrics


GROUP_NORMAL = "normal"
GROUP_RARE = "rare_or_extreme"


def _normalise_timestamp(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError("Timestamps contain invalid or missing values.")
    return parsed.dt.tz_convert(None)


def _boolean_flags(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    mapped = values.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        examples = values[mapped.isna()].astype(str).drop_duplicates().head(5).tolist()
        raise ValueError(f"Detector is_anomaly contains invalid values: {examples}.")
    return mapped.astype(bool)


def _detector_name(scores: pd.DataFrame, explicit: str | None) -> str:
    if explicit:
        return str(explicit)
    for column in ("detector", "method"):
        if column in scores:
            names = scores[column].dropna().astype(str).str.strip()
            names = names[names.ne("")].drop_duplicates()
            if len(names) == 1:
                return str(names.iloc[0])
    return "detector"


def build_pointwise_detector_evaluation(
    source_predictions: str | Path,
    detector_scores: str | Path,
    out_dir: str | Path,
    *,
    detector_name: str | None = None,
    min_match_fraction: float = 0.90,
    allow_overwrite: bool = False,
) -> dict[str, Path | int | float | str]:
    """Join SDE forecasts to detector labels and write evaluation-only outputs.

    Only rows present in both files at the exact location and target timestamp
    are evaluated. This is important for detectors such as STGAN whose context
    window leaves an unscored prefix at the beginning of the test period.
    """
    source_path = Path(source_predictions)
    score_path = Path(detector_scores)
    output_root = Path(out_dir)
    if not source_path.is_file():
        raise FileNotFoundError(f"SDE predictions not found: {source_path}")
    if not score_path.is_file():
        raise FileNotFoundError(f"Detector scores not found: {score_path}")
    if not np.isfinite(min_match_fraction) or not 0.0 < min_match_fraction <= 1.0:
        raise ValueError("min_match_fraction must be in (0, 1].")
    if output_root.exists() and any(output_root.iterdir()) and not allow_overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. "
            "Set allow_overwrite=True only to regenerate this evaluation."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(source_path)
    required_predictions = {"location", "timestamp", "y_true"}
    if not ({"y_pred", "y_pred_mean"} & set(predictions.columns)):
        required_predictions.add("y_pred_mean")
    missing_predictions = required_predictions - set(predictions.columns)
    if missing_predictions:
        raise ValueError(
            f"SDE predictions missing columns: {sorted(missing_predictions)}"
        )
    predictions = predictions.drop(
        columns=[
            "anomaly_group",
            "anomaly_label",
            "event_group",
            "event_score",
            "event_driver",
            "detector_anomaly_score",
            "detector_threshold",
            "detector_is_anomaly",
        ],
        errors="ignore",
    )
    predictions["location"] = predictions["location"].astype(str)
    predictions["timestamp"] = _normalise_timestamp(predictions["timestamp"])
    if predictions.duplicated(["location", "timestamp"]).any():
        raise ValueError("SDE predictions require one row per (location, timestamp).")

    score_header = set(pd.read_csv(score_path, nrows=0).columns)
    required_scores = {"location", "timestamp", "is_anomaly", "anomaly_score"}
    missing_scores = required_scores - score_header
    if missing_scores:
        raise ValueError(f"Detector scores missing columns: {sorted(missing_scores)}")
    score_columns = list(required_scores)
    score_columns.extend(
        column
        for column in ("threshold", "detector", "method")
        if column in score_header
    )
    scores = pd.read_csv(score_path, usecols=score_columns)
    scores["location"] = scores["location"].astype(str)
    scores["timestamp"] = _normalise_timestamp(scores["timestamp"])
    if scores.duplicated(["location", "timestamp"]).any():
        raise ValueError("Detector scores require one row per (location, timestamp).")
    scores["is_anomaly"] = _boolean_flags(scores["is_anomaly"])
    name = _detector_name(scores, detector_name)

    labels = scores[["location", "timestamp", "anomaly_score", "is_anomaly"]].copy()
    labels = labels.rename(
        columns={
            "anomaly_score": "detector_anomaly_score",
            "is_anomaly": "detector_is_anomaly",
        }
    )
    if "threshold" in scores:
        labels["detector_threshold"] = pd.to_numeric(
            scores["threshold"], errors="coerce"
        )
    labels["detector_anomaly_score"] = pd.to_numeric(
        labels["detector_anomaly_score"], errors="coerce"
    )
    if not np.isfinite(labels["detector_anomaly_score"].to_numpy(float)).all():
        raise ValueError("Detector anomaly_score contains non-finite values.")

    source_rows = len(predictions)
    joined = predictions.merge(
        labels,
        on=["location", "timestamp"],
        how="inner",
        validate="one_to_one",
    )
    matched_rows = len(joined)
    match_fraction = matched_rows / source_rows if source_rows else 0.0
    if matched_rows == 0 or match_fraction < min_match_fraction:
        raise ValueError(
            "Insufficient exact detector/SDE overlap: "
            f"{matched_rows}/{source_rows} rows ({match_fraction:.2%}); "
            f"required {min_match_fraction:.2%}. Check location identifiers, "
            "timestamps, seed output and score stride."
        )
    joined["anomaly_group"] = np.where(
        joined["detector_is_anomaly"], GROUP_RARE, GROUP_NORMAL
    )
    joined["anomaly_label"] = np.where(
        joined["detector_is_anomaly"], name, ""
    )
    if "event_group" in joined:
        raise AssertionError("Pointwise evaluation must not contain event_group.")

    prediction_column = "y_pred_mean" if "y_pred_mean" in joined else "y_pred"
    residual = joined[prediction_column].astype(float) - joined["y_true"].astype(float)
    joined["abs_error"] = residual.abs()
    joined["squared_error"] = residual**2
    if "y_pred_std" in joined:
        if "y_pred_lower" not in joined and "lower_pi" in joined:
            joined["y_pred_lower"] = joined["lower_pi"]
        if "y_pred_upper" not in joined and "upper_pi" in joined:
            joined["y_pred_upper"] = joined["upper_pi"]
        missing_interval = {"y_pred_lower", "y_pred_upper"} - set(joined.columns)
        if missing_interval:
            raise ValueError(
                "Uncertainty metrics require interval columns: "
                f"{sorted(missing_interval)}."
            )

    predictions_path = output_root / "predictions.csv"
    metrics_global_path = output_root / "metrics_global.csv"
    metrics_by_path = output_root / "metrics_by_anomaly_label.csv"
    metrics_path = output_root / "metrics.json"
    metadata_path = output_root / "evaluation_source.json"
    joined.to_csv(predictions_path, index=False)
    global_metrics, grouped_metrics = compute_metrics(joined)
    global_metrics.to_csv(metrics_global_path, index=False)
    grouped_metrics.to_csv(metrics_by_path, index=False)
    flat_metrics = build_wandb_metrics(
        global_metrics,
        grouped_metrics,
        sde_uncertainty=(
            "y_pred_std" in joined and joined["y_pred_std"].notna().any()
        ),
    )
    metrics_path.write_text(
        json.dumps(flat_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    metadata = {
        "mode": "pointwise_detector_evaluation_only",
        "detector": name,
        "join_keys": ["location", "timestamp"],
        "regional_event_group_created": False,
        "source_predictions": str(source_path.resolve()),
        "detector_scores": str(score_path.resolve()),
        "source_prediction_rows": source_rows,
        "matched_rows": matched_rows,
        "excluded_unmatched_rows": source_rows - matched_rows,
        "match_fraction": match_fraction,
        "normal_rows": int((joined["anomaly_group"] == GROUP_NORMAL).sum()),
        "rare_rows": int((joined["anomaly_group"] == GROUP_RARE).sum()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "predictions": predictions_path,
        "metrics_global": metrics_global_path,
        "metrics_by_anomaly_label": metrics_by_path,
        "metrics": metrics_path,
        "evaluation_source": metadata_path,
        **metadata,
    }
