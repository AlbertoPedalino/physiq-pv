"""Exploratory global MTGFlow top-percent labels on a training-IQR scale."""

import numpy as np
import pandas as pd


def rank_mtgflow_top_percent(scores, training_table, *, top_percent=1.0):
    """Rank quality-eligible scores globally, before any daytime/date selection.

    Preserve raw scores. Rank the signed (score - saved_threshold) / training_IQR
    coordinate; never clip it to zero. Equal coordinates retain CSV row order.
    """
    if not np.isfinite(top_percent) or not 0 < top_percent <= 100:
        raise ValueError("top_percent must be in (0, 100].")
    work = scores.copy()
    table = training_table.copy()
    table["location"] = table["location"].astype(str)
    if table["location"].duplicated().any():
        raise ValueError("Training table must have one row per location.")
    table = table.set_index("location")
    iqr = work["location"].astype(str).map(table["iqr"]).to_numpy(float)
    saved = work["location"].astype(str).map(table["saved_threshold"]).to_numpy(float)
    values = work["anomaly_score"].to_numpy(float)
    thresholds = work["threshold"].to_numpy(float)
    if not len(work) or not np.isfinite(np.column_stack([iqr, saved, values, thresholds])).all():
        raise ValueError("Finite scores, thresholds and training IQR are required for every location.")
    if (iqr <= 0).any():
        raise ValueError("Training IQR must be positive.")
    if not np.allclose(saved, thresholds, rtol=1e-6, atol=1e-5):
        raise ValueError("Training table and score-file saved thresholds disagree.")
    coordinate = (values - saved) / iqr
    if not np.isfinite(coordinate).all():
        raise ValueError("Normalized MTGFlow coordinates must be finite.")
    count = min(len(work), max(1, int(np.ceil(len(work) * top_percent / 100))))
    selected = np.argsort(-coordinate, kind="stable")[:count]
    flags = np.zeros(len(work), dtype=bool)
    flags[selected] = True
    cutoff = float(coordinate[selected].min())
    work["is_anomaly"] = flags
    work.attrs.update(
        top_percent=float(top_percent),
        ranking_coordinate="(anomaly_score - saved_threshold) / training_iqr",
        ranking_cutoff_iqr=cutoff,
        ranking_tie_policy="stable original CSV row order; exact ceil(N * percent / 100) budget",
        n_quality_eligible_before_daytime=len(work),
        n_top_percent_before_daytime=count,
        n_saved_threshold_anomalies_before_daytime=int((values >= thresholds).sum()),
        training_iqr_source=(sorted(training_table["threshold_statistics_source"].astype(str).unique())
                             if "threshold_statistics_source" in training_table else ["provided training table"]),
    )
    return work
