"""Pointwise STGAN input/target cases and the standard per-bin error plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from physiq_pv.reporting.daytime_bin_anomaly_report import _boxplot_stats
from physiq_pv.reporting.pointwise_detector_posthoc import (
    _normalise_timestamp,
    _rerank_clean_top_k,
)
from physiq_pv.reporting.posthoc_outputs import PERCENT_PRODUCTION_BINS


CASE_LABELS = {
    "normal_normal": "Input normale → target normale",
    "normal_anomalous": "Input normale → target anomalo",
    "anomalous_normal": "Input anomalo → target normale",
    "anomalous_anomalous": "Input anomalo → target anomalo",
}


def load_clean_stgan_labels(score_path, *, excluded_timestamps=(), top_percent=1.0):
    """Reproduce the existing quality-filtered global top-K before any slicing.

    Keep all scored hours (including night) for input-window classification.
    The score cutoff is for display; exact ties use the saved global rank.
    """
    header = set(pd.read_csv(score_path, nrows=0).columns)
    columns = ["location", "timestamp", "anomaly_score"]
    if "global_rank" in header:
        columns.append("global_rank")
    scores = pd.read_csv(score_path, usecols=columns, dtype={"location": str})
    scores["timestamp"] = _normalise_timestamp(scores["timestamp"])
    if scores.duplicated(["location", "timestamp"]).any():
        raise ValueError("STGAN contains duplicate location/timestamp rows.")
    excluded = pd.to_datetime(list(excluded_timestamps), utc=True).tz_convert(None)
    scores = scores.loc[~scores["timestamp"].isin(excluded)].copy()
    if scores.empty:
        raise ValueError("No quality-eligible STGAN scores.")
    clean = _rerank_clean_top_k(scores, top_percent)
    cutoff = float(clean.loc[clean["is_anomaly"], "anomaly_score"].min())
    clean = clean[["location", "timestamp", "anomaly_score", "is_anomaly"]]
    clean.attrs["score_cutoff"] = cutoff
    clean.attrs["top_percent"] = float(top_percent)
    clean.attrs["excluded_timestamps"] = len(excluded.unique())
    return clean


def plot_stgan_score_timeline(labels, out_dir, *, location=None, start=None, end=None):
    """Save a local score time series, decisions and the global top-K cutoff."""
    import matplotlib.pyplot as plt

    if location is None:
        location = labels.groupby("location", sort=True)["is_anomaly"].sum().idxmax()
    location = str(location)
    series = labels.loc[labels["location"].eq(location)].sort_values("timestamp").copy()
    for value, lower in ((start, True), (end, False)):
        if value is not None:
            bound = pd.to_datetime(value, utc=True).tz_convert(None)
            series = series.loc[series["timestamp"].ge(bound) if lower
                                else series["timestamp"].le(bound)]
    if series.empty:
        raise ValueError(f"No STGAN scores for location {location} in this interval.")
    cutoff = float(labels.attrs["score_cutoff"])
    # Insert NaNs at unscored/quality-excluded hours instead of joining gaps.
    grid = pd.date_range(series["timestamp"].min(), series["timestamp"].max(), freq="h")
    hourly = series.set_index("timestamp")["anomaly_score"].reindex(grid)
    fig, ax = plt.subplots(figsize=(15, 5), layout="constrained")
    ax.plot(hourly.index, hourly.values, color="0.65", linewidth=0.7)
    for flag, label, color in ((False, "Normali", "tab:blue"),
                               (True, "Anomali", "tab:red")):
        group = series.loc[series["is_anomaly"].eq(flag)]
        ax.scatter(group["timestamp"], group["anomaly_score"], s=10,
                   color=color, label=f"{label} (n={len(group):,})", zorder=3)
    ax.axhline(cutoff, color="black", linestyle="--",
               label=f"Soglia globale top {labels.attrs['top_percent']:g}%")
    ax.set(xlabel="Tempo (UTC)", ylabel="Anomaly score STGAN",
           title=f"STGAN — località {location}")
    ax.grid(alpha=0.25)
    ax.legend()
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "stgan_anomaly_score_timeline.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    series.to_csv(output / "stgan_anomaly_score_timeline.csv", index=False)
    return path, series


def classify_input_target_cases(predictions, labels, *, seq_len=24, min_anomalous_steps=1):
    """Classify same-site input [issue-(L-1)h, issue] and the target timestamp.

    Full hourly detector coverage is required. Unknown/quality-excluded input
    hours or targets produce an excluded row, never a normal label. No target
    label enters the input window. This describes the explicit same-site
    sequence, not all graph nodes or extra derived-feature lookback hours.
    """
    if int(seq_len) != seq_len or seq_len < 1:
        raise ValueError("seq_len must be a positive integer.")
    if int(min_anomalous_steps) != min_anomalous_steps or not 1 <= min_anomalous_steps <= seq_len:
        raise ValueError("min_anomalous_steps must be an integer in [1, seq_len].")
    work = predictions.copy()
    work["location"] = work["location"].astype(str)
    for column in ("timestamp", "issue_timestamp"):
        work[column] = _normalise_timestamp(work[column])
    horizons = pd.to_numeric(work["horizon_hours"], errors="raise")
    if not ((horizons >= 1) & (horizons % 1 == 0)).all():
        raise ValueError("Forecast horizons must be positive integers.")
    if not (work["timestamp"] - work["issue_timestamp"]).eq(
        pd.to_timedelta(horizons, unit="h")
    ).all():
        raise ValueError("Issue timestamps do not match target minus horizon.")
    if work.duplicated(["location", "timestamp", "horizon_hours"]).any():
        raise ValueError("Duplicate forecast coordinates.")
    detector = labels[["location", "timestamp", "is_anomaly"]].copy()
    detector["location"] = detector["location"].astype(str)
    detector["timestamp"] = _normalise_timestamp(detector["timestamp"])
    if detector.duplicated(["location", "timestamp"]).any():
        raise ValueError("Duplicate detector coordinates.")
    if detector["is_anomaly"].isna().any() or not detector["is_anomaly"].isin([True, False]).all():
        raise ValueError("Detector labels must be boolean, without missing values.")
    work = work.merge(
        detector.rename(columns={"is_anomaly": "target_is_anomaly"}),
        on=["location", "timestamp"], how="left", validate="many_to_one",
    )
    work["input_scored_steps"] = 0
    work["input_anomalous_steps"] = 0
    groups = detector.groupby("location", sort=False).indices
    hour_ns = pd.Timedelta(hours=1).value
    for location, indices in work.groupby("location", sort=False).groups.items():
        if location not in groups:
            continue
        site = detector.iloc[groups[location]].sort_values("timestamp")
        times = pd.DatetimeIndex(site["timestamp"]).as_unit("ns").asi8
        issues = pd.DatetimeIndex(work.loc[indices, "issue_timestamp"]).as_unit("ns").asi8
        # All records must lie on the same hourly grid, which may be :10 UTC.
        if np.any((times - times[0]) % hour_ns) or np.any((issues - times[0]) % hour_ns):
            raise ValueError("Detector and prediction timestamps are not on the same hourly grid.")
        left = np.searchsorted(times, issues - (seq_len - 1) * hour_ns, side="left")
        right = np.searchsorted(times, issues, side="right")
        prefix = np.r_[0, np.cumsum(site["is_anomaly"].to_numpy(dtype=np.int64))]
        work.loc[indices, "input_scored_steps"] = right - left
        work.loc[indices, "input_anomalous_steps"] = prefix[right] - prefix[left]
    complete = work["input_scored_steps"].eq(seq_len)
    known_target = work["target_is_anomaly"].notna()
    valid = complete & known_target
    work["input_is_anomaly"] = pd.Series(pd.NA, index=work.index, dtype="boolean")
    work.loc[complete, "input_is_anomaly"] = work.loc[
        complete, "input_anomalous_steps"
    ].ge(min_anomalous_steps)
    work["input_anomaly_fraction"] = work["input_anomalous_steps"].div(seq_len).where(complete)
    work["case"] = pd.Series(pd.NA, index=work.index, dtype="string")
    work.loc[valid, "case"] = (
        np.where(work.loc[valid, "input_is_anomaly"], "anomalous", "normal")
        + np.full(valid.sum(), "_", dtype=object)
        + np.where(work.loc[valid, "target_is_anomaly"].astype(bool), "anomalous", "normal")
    )
    work["exclusion_reason"] = np.select(
        [~complete & ~known_target, ~complete, ~known_target],
        ["incomplete_input_and_unscored_target", "incomplete_input", "unscored_target"],
        default="",
    )
    return work


def summarize_input_target_cases(classified, *, horizons=(1, 6)):
    """Exact pooled MAE/RMSE and Tukey boxes; include empty cases/bins as NaN."""
    valid = classified.loc[classified["case"].notna()]
    groups = valid.groupby(["horizon_hours", "production_bin", "case"], observed=True)
    rows = []
    for horizon in horizons:
        for bin_name, _, _ in PERCENT_PRODUCTION_BINS:
            for case in CASE_LABELS:
                key = (horizon, bin_name, case)
                group = groups.get_group(key) if key in groups.indices else valid.iloc[:0]
                absolute = group["abs_error"].to_numpy(float)
                square = group["squared_error"].to_numpy(float)
                if not np.isfinite(absolute).all() or not np.isfinite(square).all():
                    raise ValueError("Non-finite errors in case summary.")
                stats = _boxplot_stats(absolute, "abs_error")
                rows.append({
                    "horizon_hours": int(horizon), "production_bin": bin_name,
                    "case": case, "count": len(group),
                    "mae": stats["abs_error_mean"],
                    "rmse": float(np.sqrt(square.mean())) if len(group) else np.nan,
                    **stats,
                })
    return pd.DataFrame(rows)


def plot_input_target_cases(metrics, out_dir):
    """Same per-bin absolute-error boxes and pooled RMSE bars as the pipeline.

    Each bin/horizon figure compares all four cases with fixed positions and
    counts; absent groups are labelled n=0 and do not receive a zero metric.
    """
    import matplotlib.pyplot as plt

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for (horizon, bin_name), group in metrics.groupby(
        ["horizon_hours", "production_bin"], sort=False
    ):
        group = group.set_index("case").reindex(CASE_LABELS)
        ticklabels = [f"{CASE_LABELS[case]}\n(n={int(row['count']):,})"
                      for case, row in group.iterrows()]
        for metric in ("mae", "rmse"):
            fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
            if metric == "mae":
                boxes, positions = [], []
                for position, (_, row) in enumerate(group.iterrows(), start=1):
                    if not row["count"]:
                        continue
                    boxes.append({
                        "mean": row["mae"], "med": row["abs_error_median"],
                        "q1": row["abs_error_q1"], "q3": row["abs_error_q3"],
                        "whislo": row["abs_error_whisker_low"],
                        "whishi": row["abs_error_whisker_high"], "fliers": [],
                    })
                    positions.append(position)
                if boxes:
                    ax.bxp(boxes, positions=positions, showfliers=False, showmeans=True,
                           meanprops={"marker": "D", "markerfacecolor": "red",
                                      "markeredgecolor": "red", "markersize": 5})
                ax.set_ylabel("Errore assoluto [W] — rombo rosso: MAE")
            else:
                ax.bar(range(1, 5), group["rmse"].to_numpy(float), color="steelblue")
                ax.set_ylabel("RMSE [W]")
            ax.set_xticks(range(1, 5), ticklabels, rotation=15, ha="right")
            ax.set_xlim(0.5, 4.5)
            ax.set_title(f"{metric.upper()} — {bin_name} — t+{horizon}h")
            ax.grid(axis="y", alpha=0.25)
            key = f"{metric}_{bin_name}_t_plus_{horizon}"
            path = output / f"{key}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths[key] = path
    return paths
