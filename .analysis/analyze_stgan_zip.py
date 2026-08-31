from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent / "stgan_results_seed20"
SCORES = ROOT / "anomaly_scores.csv"
ENTITY = ROOT / "entity_anomaly_scores.csv"
SUMMARY = ROOT / "summary.csv"
CHUNK = 500_000


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


summary = pd.read_csv(SUMMARY)
summary["location"] = summary["location"].astype(str)

score_parts: list[np.ndarray] = []
flagged_parts: list[pd.DataFrame] = []
top_parts: list[pd.DataFrame] = []
rows = anomalies = threshold_non_null = 0
minimum_time = maximum_time = None
methods: set[str] = set()
rows_by_location: dict[str, int] = {}

usecols = [
    "location", "timestamp", "method", "anomaly_score", "global_rank",
    "global_percentile", "threshold", "is_anomaly",
]
for chunk in pd.read_csv(SCORES, usecols=usecols, chunksize=CHUNK):
    chunk["location"] = chunk["location"].astype(str)
    stamp = pd.to_datetime(chunk["timestamp"], errors="raise")
    score = pd.to_numeric(chunk["anomaly_score"], errors="coerce").to_numpy(float)
    if not np.isfinite(score).all():
        raise ValueError("Non-finite anomaly scores")
    flag = chunk["is_anomaly"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if flag.isna().any():
        raise ValueError("Invalid is_anomaly")
    chunk["timestamp"] = stamp
    chunk["anomaly_score"] = score
    chunk["is_anomaly"] = flag.astype(bool)
    rows += len(chunk)
    anomalies += int(flag.sum())
    threshold_non_null += int(chunk["threshold"].notna().sum())
    methods.update(chunk["method"].dropna().astype(str).unique())
    chunk_counts = chunk.groupby("location", observed=True).size()
    for location, count in chunk_counts.items():
        rows_by_location[location] = rows_by_location.get(location, 0) + int(count)
    chunk_min, chunk_max = stamp.min(), stamp.max()
    minimum_time = chunk_min if minimum_time is None else min(minimum_time, chunk_min)
    maximum_time = chunk_max if maximum_time is None else max(maximum_time, chunk_max)
    score_parts.append(score.astype(np.float32, copy=False))
    selected = chunk.loc[chunk["is_anomaly"], [
        "location", "timestamp", "anomaly_score", "global_rank", "global_percentile"
    ]]
    if not selected.empty:
        flagged_parts.append(selected.copy())
    top_parts.append(chunk.nlargest(20, "anomaly_score")[[
        "location", "timestamp", "anomaly_score", "global_rank", "global_percentile",
        "is_anomaly",
    ]])

scores = np.concatenate(score_parts).astype(np.float64, copy=False)
flagged = pd.concat(flagged_parts, ignore_index=True)
top_scores = pd.concat(top_parts, ignore_index=True).nlargest(20, "anomaly_score")
expected_anomalies = int(np.ceil(rows * 0.01))

flagged["day"] = flagged["timestamp"].dt.normalize()
flagged["month"] = flagged["timestamp"].dt.month
flagged["hour"] = flagged["timestamp"].dt.hour

daily = (
    flagged.groupby("day", observed=True)
    .agg(
        n_anomalies=("location", "size"),
        n_locations=("location", "nunique"),
        max_score=("anomaly_score", "max"),
        mean_score=("anomaly_score", "mean"),
    )
    .reset_index()
)
daily["anomaly_rate"] = daily["n_anomalies"] / (len(summary) * 24)
daily = daily.sort_values(["n_anomalies", "max_score"], ascending=False)

days_per_month = pd.Series(
    {month: pd.Period(f"2019-{month:02d}").days_in_month for month in range(1, 13)}
)
monthly = (
    flagged.groupby("month", observed=True)
    .agg(
        n_anomalies=("location", "size"),
        n_locations=("location", "nunique"),
        max_score=("anomaly_score", "max"),
        mean_score=("anomaly_score", "mean"),
    )
    .reindex(range(1, 13), fill_value=0)
    .reset_index()
)
monthly["anomaly_rate"] = monthly.apply(
    lambda row: row["n_anomalies"]
    / (len(summary) * 24 * int(days_per_month.loc[int(row["month"])])),
    axis=1,
)

hourly = (
    flagged.groupby("hour", observed=True)
    .agg(n_anomalies=("location", "size"), n_locations=("location", "nunique"))
    .reindex(range(24), fill_value=0)
    .reset_index()
)
hourly["anomaly_rate"] = hourly["n_anomalies"] / (len(summary) * 365)

timestamp_regional = (
    flagged.groupby("timestamp", observed=True)
    .agg(
        n_anomalies=("location", "size"),
        max_score=("anomaly_score", "max"),
        mean_score=("anomaly_score", "mean"),
    )
    .reset_index()
)
timestamp_regional["location_share"] = timestamp_regional["n_anomalies"] / len(summary)
top_timestamps = timestamp_regional.nlargest(20, ["n_anomalies", "max_score"])


def event_summary(name: str, start: str, end: str) -> dict:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    selected = flagged[
        (flagged["timestamp"] >= start_ts) & (flagged["timestamp"] < end_ts)
    ]
    total = int((end_ts - start_ts) / pd.Timedelta(hours=1)) * len(summary)
    return {
        "event": name,
        "start": start,
        "end_exclusive": end,
        "n_anomalies": int(len(selected)),
        "anomaly_rate": float(len(selected) / total),
        "n_locations": int(selected["location"].nunique()),
        "max_score": float(selected["anomaly_score"].max()) if len(selected) else None,
        "mean_flagged_score": float(selected["anomaly_score"].mean()) if len(selected) else None,
    }


events = pd.DataFrame([
    event_summary("april_23_26", "2019-04-23", "2019-04-27"),
    event_summary("june_28_29", "2019-06-28", "2019-06-30"),
    event_summary("july", "2019-07-01", "2019-08-01"),
])

summary_stats = summary["anomaly_rate"].describe(
    percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
)
top_locations = summary.nlargest(15, ["n_anomaly", "anomaly_rate"])[[
    "location", "latitude", "longitude", "n_anomaly", "anomaly_rate"
]]
bottom_locations = summary.nsmallest(15, ["n_anomaly", "anomaly_rate"])[[
    "location", "latitude", "longitude", "n_anomaly", "anomaly_rate"
]]
spatial_correlations = summary[["latitude", "longitude", "anomaly_rate"]].corr()[
    "anomaly_rate"
].drop("anomaly_rate")

entity = pd.read_csv(ENTITY)
entity["location"] = entity["location"].astype(str)
entity["timestamp"] = pd.to_datetime(entity["timestamp"], errors="raise")
entity["anomaly_score"] = pd.to_numeric(entity["anomaly_score"], errors="coerce")
entity["contribution_fraction"] = pd.to_numeric(
    entity["contribution_fraction"], errors="coerce"
)
if not np.isfinite(entity[["anomaly_score", "contribution_fraction"]].to_numpy()).all():
    raise ValueError("Non-finite entity scores")
point_keys = ["location", "timestamp"]
contribution_sums = entity.groupby(point_keys, observed=True)["contribution_fraction"].sum()
dominant_indices = entity.groupby(point_keys, observed=True)["contribution_fraction"].idxmax()
dominant = entity.loc[dominant_indices, point_keys + ["entity", "contribution_fraction"]].copy()
dominant["month"] = dominant["timestamp"].dt.month
feature_summary = (
    entity.groupby("entity", observed=True)
    .agg(
        rows=("contribution_fraction", "size"),
        mean_contribution=("contribution_fraction", "mean"),
        median_contribution=("contribution_fraction", "median"),
        p90_contribution=("contribution_fraction", lambda x: x.quantile(0.9)),
        mean_feature_score=("anomaly_score", "mean"),
    )
    .reset_index()
)
dominant_summary = (
    dominant.groupby("entity", observed=True)
    .agg(n_dominant=("entity", "size"), mean_dominant_share=("contribution_fraction", "mean"))
    .reset_index()
)
dominant_summary["dominant_fraction"] = dominant_summary["n_dominant"] / len(dominant)
feature_summary = feature_summary.merge(dominant_summary, on="entity", how="left")

dominant_month = (
    dominant.groupby(["month", "entity"], observed=True).size().rename("n").reset_index()
)
dominant_month["fraction_in_month"] = dominant_month["n"] / dominant_month.groupby(
    "month", observed=True
)["n"].transform("sum")


def feature_event(name: str, start: str, end: str) -> pd.DataFrame:
    selected = dominant[
        (dominant["timestamp"] >= pd.Timestamp(start))
        & (dominant["timestamp"] < pd.Timestamp(end))
    ]
    result = selected.groupby("entity", observed=True).size().rename("n").reset_index()
    result["fraction"] = result["n"] / max(len(selected), 1)
    result.insert(0, "event", name)
    return result


feature_events = pd.concat([
    feature_event("april_23_26", "2019-04-23", "2019-04-27"),
    feature_event("june_28_29", "2019-06-28", "2019-06-30"),
    feature_event("july", "2019-07-01", "2019-08-01"),
], ignore_index=True)

report = {
    "integrity": {
        "rows": rows,
        "locations": len(rows_by_location),
        "rows_per_location_min": min(rows_by_location.values()),
        "rows_per_location_max": max(rows_by_location.values()),
        "timestamp_min": str(minimum_time),
        "timestamp_max": str(maximum_time),
        "methods": sorted(methods),
        "threshold_non_null": threshold_non_null,
        "anomalies": anomalies,
        "expected_top_1_percent": expected_anomalies,
        "exact_top_k": anomalies == expected_anomalies,
        "entity_rows": len(entity),
        "entity_points": len(contribution_sums),
        "entity_rows_per_point": float(len(entity) / len(contribution_sums)),
        "contribution_sum_min": float(contribution_sums.min()),
        "contribution_sum_max": float(contribution_sums.max()),
    },
    "score_distribution": {
        f"q{q:g}": float(np.quantile(scores, q))
        for q in (0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1)
    },
    "decision_boundary": {
        "minimum_flagged_score": float(flagged["anomaly_score"].min()),
        "maximum_normal_score": float(
            np.partition(scores, len(scores) - anomalies - 1)[
                len(scores) - anomalies - 1
            ]
        ) if anomalies < len(scores) else None,
        "minimum_flagged_percentile": float(flagged["global_percentile"].min()),
        "maximum_flagged_rank": float(flagged["global_rank"].max()),
    },
    "location_rate_distribution": {
        key: float(value) for key, value in summary_stats.items()
    },
    "spatial_rate_correlation": {
        key: float(value) for key, value in spatial_correlations.items()
    },
    "top_locations": records(top_locations),
    "bottom_locations": records(bottom_locations),
    "monthly": records(monthly),
    "hour_of_day": records(hourly),
    "top_days": records(daily.head(20)),
    "top_timestamps": records(top_timestamps),
    "events": records(events),
    "top_scores": records(top_scores),
    "feature_summary": records(feature_summary),
    "feature_dominance_by_month": records(dominant_month),
    "feature_dominance_events": records(feature_events),
}

print(json.dumps(report, indent=2, ensure_ascii=False))
