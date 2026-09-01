"""Build detector-specific, pointwise post-hoc inputs for SDE forecasts.

The detector decision is matched on the exact ``(location, timestamp)`` pair.
Regional timestamp labels are deliberately neither created nor consumed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from physiq_pv.reporting.run_metrics import build_wandb_metrics, compute_metrics


GROUP_NORMAL = "normal"
GROUP_RARE = "rare_or_extreme"
QUALITY_DROPOUT = "regional_solar_dropout"
QUALITY_RECOVERY = "recovery_after_regional_solar_dropout"


def detect_isolated_regional_solar_dropouts(
    pvgis_year_path: str | Path,
) -> pd.DataFrame:
    """Find isolated, region-wide zero solar records in one PVGIS year.

    A timestamp is invalid only when all locations simultaneously report zero
    direct irradiance, diffuse irradiance, sun height and PV production while
    both adjacent timestamps have positive regional irradiance.  The exact
    zero/bracketing rule has no fitted threshold and does not confuse the
    regular night-time zero block with a one-record daytime dropout.

    The following timestamp is also returned because STGAN's recent branch
    consumes the immediately preceding observation.  Original detector scores
    remain untouched; this table is an evaluation-only quality mask.
    """
    source = Path(pvgis_year_path)
    if not source.is_file():
        raise FileNotFoundError(f"PVGIS NetCDF not found: {source}")
    required = {
        "direct_irradiance_tilted",
        "diffuse_irradiance_tilted",
        "sun_height",
        "pv_power_output",
    }
    with xr.open_dataset(source) as dataset:
        missing = required - set(dataset.data_vars)
        if missing:
            raise ValueError(
                f"PVGIS NetCDF is missing quality variables: {sorted(missing)}"
            )
        if "time" not in dataset.coords:
            raise ValueError("PVGIS NetCDF is missing the time coordinate.")
        timestamps = pd.DatetimeIndex(pd.to_datetime(dataset["time"].values))
        if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
            raise ValueError("PVGIS timestamps must be unique and increasing.")

        all_zero = np.ones(len(timestamps), dtype=bool)
        for variable in sorted(required):
            values = dataset[variable]
            reduce_dims = [dim for dim in values.dims if dim != "time"]
            if not reduce_dims:
                variable_zero = np.asarray(values.values == 0, dtype=bool)
            else:
                variable_zero = np.asarray(
                    (values == 0).all(dim=reduce_dims).values, dtype=bool
                )
            if variable_zero.shape != (len(timestamps),):
                raise ValueError(
                    f"PVGIS variable {variable!r} does not reduce to one value per time."
                )
            all_zero &= variable_zero

        effective = (
            dataset["direct_irradiance_tilted"]
            + dataset["diffuse_irradiance_tilted"]
        )
        reduce_dims = [dim for dim in effective.dims if dim != "time"]
        regional_irradiance = np.asarray(
            effective.mean(dim=reduce_dims).values if reduce_dims else effective.values,
            dtype=float,
        )

    bracketed_by_daylight = np.zeros(len(timestamps), dtype=bool)
    if len(timestamps) >= 3:
        bracketed_by_daylight[1:-1] = (
            (regional_irradiance[:-2] > 0.0)
            & (regional_irradiance[2:] > 0.0)
        )
    dropout_positions = np.flatnonzero(all_zero & bracketed_by_daylight)
    records: list[dict[str, object]] = []
    for position in dropout_positions:
        dropout_time = timestamps[position]
        records.append(
            {
                "timestamp": dropout_time,
                "quality_issue": QUALITY_DROPOUT,
                "source_dropout_timestamp": dropout_time,
            }
        )
        if position + 1 < len(timestamps):
            records.append(
                {
                    "timestamp": timestamps[position + 1],
                    "quality_issue": QUALITY_RECOVERY,
                    "source_dropout_timestamp": dropout_time,
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=["timestamp", "quality_issue", "source_dropout_timestamp"],
    )


def _rerank_clean_top_k(scores: pd.DataFrame, percentage: float) -> pd.DataFrame:
    """Reproduce the paper's exact global top-K on quality-eligible scores."""
    if not np.isfinite(percentage) or not 0.0 < percentage <= 100.0:
        raise ValueError("clean_top_k_percent must be in (0, 100].")
    result = scores.copy()
    values = pd.to_numeric(result["anomaly_score"], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Detector anomaly_score contains non-finite values.")
    if "global_rank" in result:
        original_rank = pd.to_numeric(result["global_rank"], errors="coerce")
        if not np.isfinite(original_rank.to_numpy(dtype=float)).all():
            raise ValueError("Detector global_rank contains non-finite values.")
        order = np.argsort(original_rank.to_numpy(dtype=float), kind="stable")
    else:
        order = np.argsort(-values.to_numpy(dtype=float), kind="stable")
    count = min(len(result), max(1, int(np.ceil(len(result) * percentage / 100.0))))
    ranks = np.empty(len(result), dtype=np.int64)
    ranks[order] = np.arange(1, len(result) + 1, dtype=np.int64)
    flags = np.zeros(len(result), dtype=bool)
    flags[order[:count]] = True
    result["clean_global_rank"] = ranks
    result["clean_global_percentile"] = (
        100.0 * (len(result) - ranks + 1.0) / len(result)
    )
    result["is_anomaly"] = flags
    return result


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
    pvgis_quality_source: str | Path | None = None,
    clean_top_k_percent: float | None = None,
) -> dict[str, Path | int | float | str]:
    """Join SDE forecasts to detector labels and write evaluation-only outputs.

    Only rows present in both files at the exact location and target timestamp
    are evaluated. This is important for detectors such as STGAN whose context
    window leaves an unscored prefix at the beginning of the test period. In a
    direct multi-output run, multiple horizons may share that detector key; the
    full prediction-row key also includes ``horizon_hours``.

    When ``pvgis_quality_source`` is provided, isolated regional solar
    dropouts and their immediate recovery are exported separately.  The paper
    top-K decision is then recomputed over the remaining detector coordinates;
    neither model is retrained and the original decision is retained for audit.
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
    prediction_key = ["location", "timestamp"]
    if "horizon_hours" in predictions:
        horizons = pd.to_numeric(predictions["horizon_hours"], errors="raise")
        if (
            not np.isfinite(horizons.to_numpy(dtype=float)).all()
            or (horizons <= 0).any()
            or (horizons % 1 != 0).any()
        ):
            raise ValueError("horizon_hours must contain positive integers.")
        predictions["horizon_hours"] = horizons.astype(int)
        prediction_key.append("horizon_hours")
    if predictions.duplicated(prediction_key).any():
        raise ValueError(
            "SDE predictions require one row per "
            f"({', '.join(prediction_key)})."
        )

    score_header = set(pd.read_csv(score_path, nrows=0).columns)
    required_scores = {"location", "timestamp", "is_anomaly", "anomaly_score"}
    missing_scores = required_scores - score_header
    if missing_scores:
        raise ValueError(f"Detector scores missing columns: {sorted(missing_scores)}")
    score_columns = list(required_scores)
    score_columns.extend(
        column
        for column in ("threshold", "detector", "method", "global_rank")
        if column in score_header
    )
    scores = pd.read_csv(score_path, usecols=score_columns)
    scores["location"] = scores["location"].astype(str)
    scores["timestamp"] = _normalise_timestamp(scores["timestamp"])
    if scores.duplicated(["location", "timestamp"]).any():
        raise ValueError("Detector scores require one row per (location, timestamp).")
    scores["is_anomaly"] = _boolean_flags(scores["is_anomaly"])
    name = _detector_name(scores, detector_name)

    quality_issues = pd.DataFrame(
        columns=["timestamp", "quality_issue", "source_dropout_timestamp"]
    )
    if pvgis_quality_source is not None:
        if clean_top_k_percent is None:
            raise ValueError(
                "clean_top_k_percent is required with pvgis_quality_source."
            )
        quality_issues = detect_isolated_regional_solar_dropouts(
            pvgis_quality_source
        )
        invalid_timestamps = pd.DatetimeIndex(quality_issues["timestamp"])
        scores["data_quality_issue"] = scores["timestamp"].isin(invalid_timestamps)
        scores["original_is_anomaly"] = scores["is_anomaly"]
        clean_scores = _rerank_clean_top_k(
            scores.loc[~scores["data_quality_issue"]].copy(),
            clean_top_k_percent,
        )
        quality_scores = scores.loc[scores["data_quality_issue"]].copy()
        scores = pd.concat([clean_scores, quality_scores], ignore_index=True)
    else:
        scores["data_quality_issue"] = False
        scores["original_is_anomaly"] = scores["is_anomaly"]

    label_columns = [
        "location", "timestamp", "anomaly_score", "is_anomaly",
        "original_is_anomaly", "data_quality_issue",
    ]
    label_columns.extend(
        column
        for column in ("clean_global_rank", "clean_global_percentile")
        if column in scores
    )
    labels = scores[label_columns].copy()
    labels = labels.rename(
        columns={
            "anomaly_score": "detector_anomaly_score",
            "is_anomaly": "detector_is_anomaly",
            "original_is_anomaly": "detector_is_anomaly_original",
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
    joined_all = predictions.merge(
        labels,
        on=["location", "timestamp"],
        how="inner",
        # Direct multi-output forecasts legitimately contain one prediction row
        # per horizon for the same target. Detector decisions remain unique at
        # (location, timestamp) and are therefore shared by those rows.
        validate="many_to_one",
    )
    detector_matched_rows = len(joined_all)
    match_fraction = detector_matched_rows / source_rows if source_rows else 0.0
    if detector_matched_rows == 0 or match_fraction < min_match_fraction:
        raise ValueError(
            "Insufficient exact detector/SDE overlap: "
            f"{detector_matched_rows}/{source_rows} rows ({match_fraction:.2%}); "
            f"required {min_match_fraction:.2%}. Check location identifiers, "
            "timestamps, seed output and score stride."
        )
    quality_predictions = joined_all.loc[joined_all["data_quality_issue"]].copy()
    if not quality_predictions.empty:
        quality_predictions = quality_predictions.merge(
            quality_issues,
            on="timestamp",
            how="left",
            validate="many_to_one",
        )
    joined = joined_all.loc[~joined_all["data_quality_issue"]].copy()
    matched_rows = len(joined)
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
    quality_predictions_path = output_root / "data_quality_predictions.csv"
    quality_issues_path = output_root / "pvgis_data_quality_issues.csv"
    metrics_global_path = output_root / "metrics_global.csv"
    metrics_by_path = output_root / "metrics_by_anomaly_label.csv"
    metrics_path = output_root / "metrics.json"
    metadata_path = output_root / "evaluation_source.json"
    joined.to_csv(predictions_path, index=False)
    quality_predictions.to_csv(quality_predictions_path, index=False)
    quality_issues.to_csv(quality_issues_path, index=False)
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
        "prediction_row_key": prediction_key,
        "forecast_mode": (
            "direct_multi_output"
            if "horizon_hours" in joined
            else "single_horizon"
        ),
        "horizons_hours": (
            sorted(joined["horizon_hours"].drop_duplicates().astype(int).tolist())
            if "horizon_hours" in joined
            else None
        ),
        "regional_event_group_created": False,
        "source_predictions": str(source_path.resolve()),
        "detector_scores": str(score_path.resolve()),
        "pvgis_quality_source": (
            str(Path(pvgis_quality_source).resolve())
            if pvgis_quality_source is not None
            else None
        ),
        "quality_filter_policy": (
            "isolated_regional_solar_dropout_plus_immediate_recovery"
            if pvgis_quality_source is not None
            else None
        ),
        "source_prediction_rows": source_rows,
        "detector_matched_rows_before_quality_filter": detector_matched_rows,
        "matched_rows": matched_rows,
        "excluded_unmatched_rows": source_rows - detector_matched_rows,
        "excluded_data_quality_rows": len(quality_predictions),
        "data_quality_timestamps": int(len(quality_issues)),
        "solar_dropout_timestamps": int(
            (quality_issues["quality_issue"] == QUALITY_DROPOUT).sum()
        ),
        "eligible_detector_coordinates": int((~scores["data_quality_issue"]).sum()),
        "data_quality_detector_coordinates": int(scores["data_quality_issue"].sum()),
        "original_detector_anomalies": int(scores["original_is_anomaly"].sum()),
        "clean_detector_anomalies": int(
            scores.loc[~scores["data_quality_issue"], "is_anomaly"].sum()
        ),
        "clean_top_k_percent": clean_top_k_percent,
        "match_fraction": match_fraction,
        "normal_rows": int((joined["anomaly_group"] == GROUP_NORMAL).sum()),
        "rare_rows": int((joined["anomaly_group"] == GROUP_RARE).sum()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "predictions": predictions_path,
        "data_quality_predictions": quality_predictions_path,
        "data_quality_issues": quality_issues_path,
        "metrics_global": metrics_global_path,
        "metrics_by_anomaly_label": metrics_by_path,
        "metrics": metrics_path,
        "evaluation_source": metadata_path,
        **metadata,
    }
