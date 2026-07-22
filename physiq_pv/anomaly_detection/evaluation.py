"""Post-hoc forecast evaluation on detector-defined climate anomalies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def attach_detector_scores(
    predictions: pd.DataFrame,
    detector_scores: pd.DataFrame,
    *,
    method: str | None = None,
) -> pd.DataFrame:
    """Left-join one detector output without changing the prediction sample."""
    pred = predictions.copy()
    scores = detector_scores.copy()
    required = {"location", "timestamp", "anomaly_score", "is_anomaly"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Detector scores missing columns: {sorted(missing)}")
    if method is not None:
        if "method" not in scores:
            raise ValueError("method was requested but detector CSV has no method column.")
        scores = scores[scores["method"] == method].copy()
    elif "method" in scores and scores["method"].nunique() > 1:
        raise ValueError("Detector CSV contains multiple methods; select one explicitly.")
    pred["location"] = pred["location"].astype(str)
    scores["location"] = scores["location"].astype(str)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    scores["timestamp"] = pd.to_datetime(scores["timestamp"])
    if scores.duplicated(["location", "timestamp"]).any():
        raise ValueError("Detector scores are not unique by location and timestamp.")
    keep = [
        c for c in ("location", "timestamp", "method", "anomaly_score", "threshold", "is_anomaly")
        if c in scores
    ]
    return pred.merge(scores[keep], on=["location", "timestamp"], how="left", validate="many_to_one")


def forecast_metrics_by_detection(joined: pd.DataFrame) -> pd.DataFrame:
    """Return comparable metrics for matched normal and anomalous timestamps."""
    required = {"y_true", "y_pred", "is_anomaly", "anomaly_score"}
    missing = required - set(joined.columns)
    if missing:
        raise ValueError(f"Joined predictions missing columns: {sorted(missing)}")
    work = joined.dropna(subset=["is_anomaly", "y_true", "y_pred"]).copy()
    work["is_anomaly"] = work["is_anomaly"].astype(bool)
    rows = []
    for flag, group in work.groupby("is_anomaly", sort=True):
        error = group["y_pred"].to_numpy(float) - group["y_true"].to_numpy(float)
        row = {
            "stratum": "anomaly" if flag else "normal",
            "n": len(group),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "bias": float(np.mean(error)),
            "mean_anomaly_score": float(group["anomaly_score"].mean()),
        }
        if "y_pred_std" in group:
            row["mean_predictive_std"] = float(group["y_pred_std"].mean())
        if {"lower_pi", "upper_pi"} <= set(group.columns):
            lower = group["lower_pi"].to_numpy(float)
            upper = group["upper_pi"].to_numpy(float)
            truth = group["y_true"].to_numpy(float)
            row["mean_pi_width"] = float(np.mean(upper - lower))
            row["picp"] = float(np.mean((truth >= lower) & (truth <= upper)))
        rows.append(row)
    return pd.DataFrame(rows)
