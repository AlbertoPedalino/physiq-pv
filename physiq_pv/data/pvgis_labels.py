"""Anomaly-score loading and attachment for local and regional evaluation.

Both supported sources are evaluation/filtering metadata, never model inputs:

* ``climatology`` rows carry a per-variable score and semantic ``label``;
* ``detector`` rows carry the thresholded MTGFlow/CATCH/M2AD ``is_anomaly``
  decision and a common global anomaly score.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from physiq_pv.data.pvgis_dataset import GROUP_NORMAL, GROUP_RARE


def load_anomaly_labels(
    path: Optional[str],
    *,
    source: str = "climatology",
) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Anomaly scores file not found: {p}")
    columns = pd.read_csv(p, nrows=0).columns
    source = str(source).strip().lower()
    if source == "climatology":
        missing = {"location", "timestamp", "label"} - set(columns)
        if missing:
            raise ValueError(
                f"Climatology scores file missing columns: {sorted(missing)}"
            )
        optional = [
            column
            for column in ("variable", "anomaly_score")
            if column in columns
        ]
        usecols = ["location", "timestamp", "label", *optional]
    elif source == "detector":
        missing = {"location", "timestamp", "is_anomaly"} - set(columns)
        if missing:
            raise ValueError(
                f"Detector scores file missing columns: {sorted(missing)}"
            )
        if "anomaly_score" in columns:
            score_column = "anomaly_score"
        elif "global_score" in columns:
            score_column = "global_score"
        else:
            raise ValueError(
                "Detector scores require 'anomaly_score' or 'global_score'."
            )
        optional = [
            column
            for column in ("threshold", "detector", "method", "seed")
            if column in columns
        ]
        usecols = [
            "location",
            "timestamp",
            score_column,
            "is_anomaly",
            *optional,
        ]
    else:
        raise ValueError(
            f"Unknown anomaly source {source!r}; expected 'climatology' or 'detector'."
        )
    df = pd.read_csv(p, usecols=usecols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if source == "detector":
        if score_column != "anomaly_score":
            df = df.rename(columns={score_column: "anomaly_score"})
        if "detector" not in df:
            if "method" in df:
                df["detector"] = df["method"].astype(str)
            elif "gamma_p_value" in columns:
                df["detector"] = "m2ad"
            elif {"time_score", "frequency_score"}.issubset(columns):
                df["detector"] = "catch"
            else:
                df["detector"] = "detector"
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
    scores = anomaly_scores.copy()
    if "is_anomaly" in scores:
        flags = scores["is_anomaly"]
        if not pd.api.types.is_bool_dtype(flags):
            flags = flags.astype(str).str.strip().str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            )
        if flags.isna().any():
            raise ValueError("Detector is_anomaly contains invalid boolean values.")
        scores = scores[flags.astype(bool)].copy()
        if "label" not in scores:
            scores["label"] = scores.get(
                "detector", pd.Series("detector", index=scores.index)
            ).astype(str)
    if scores.empty:
        out["anomaly_group"] = GROUP_NORMAL
        out["anomaly_label"] = ""
        return out
    agg = (
        scores.groupby(["location", "timestamp"])["label"]
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


