"""Anomaly-score loading and attachment for local and regional evaluation.

Reads the climatology anomaly-score CSV and attaches anomaly_group /
anomaly_label to a predictions frame. Regional event labels are derived before
dataset construction and can be attached separately. Neither label type is a
model input or a supervised prediction target.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from physiq_pv.data.pvgis_dataset import GROUP_NORMAL, GROUP_RARE


def load_anomaly_labels(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Anomaly scores file not found: {p}")
    columns = pd.read_csv(p, nrows=0).columns
    missing = {"location", "timestamp", "label"} - set(columns)
    if missing:
        raise ValueError(f"Anomaly scores file missing columns: {sorted(missing)}")
    optional = [
        column
        for column in ("variable", "anomaly_score")
        if column in columns
    ]
    usecols = ["location", "timestamp", "label", *optional]
    df = pd.read_csv(p, usecols=usecols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def attach_anomaly_labels(
    predictions: pd.DataFrame, anomaly_scores: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Add `anomaly_group` (normal / rare_or_extreme) and `anomaly_label` (specific)."""
    out = predictions.copy()
    if anomaly_scores is None or anomaly_scores.empty:
        out["anomaly_group"] = GROUP_NORMAL
        out["anomaly_label"] = ""
        return out
    agg = (
        anomaly_scores.groupby(["location", "timestamp"])["label"]
        .agg(lambda s: ",".join(sorted(set(s))))
        .reset_index()
        .rename(columns={"label": "anomaly_label"})
    )
    agg["location"] = agg["location"].astype(out["location"].dtype)
    out = out.merge(agg, on=["location", "timestamp"], how="left")
    out["anomaly_group"] = np.where(out["anomaly_label"].notna(), GROUP_RARE, GROUP_NORMAL)
    out["anomaly_label"] = out["anomaly_label"].fillna("")
    return out


def attach_event_labels(
    predictions: pd.DataFrame,
    event_labels: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Attach graph-wide normal/rare labels to every node at a target timestamp."""
    out = predictions.copy()
    if event_labels is None or event_labels.empty:
        out["event_group"] = GROUP_NORMAL
        out["event_score"] = 0.0
        out["event_driver"] = ""
        return out
    required = {"timestamp", "event_group", "event_score", "event_driver"}
    missing = required - set(event_labels.columns)
    if missing:
        raise ValueError(f"Regional event labels missing columns: {sorted(missing)}")
    labels = event_labels[list(required)].copy()
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    if labels["timestamp"].duplicated().any():
        raise ValueError("Regional event labels require one row per timestamp.")
    out = out.merge(labels, on="timestamp", how="left")
    if out["event_group"].isna().any():
        missing_count = int(out["event_group"].isna().sum())
        raise ValueError(
            f"Regional event labels did not match {missing_count} prediction rows."
        )
    return out


