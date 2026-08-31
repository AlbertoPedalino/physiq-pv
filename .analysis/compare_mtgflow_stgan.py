from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MTG = (
    ROOT
    / ".analysis/mtgflow_results_seed15/outputs/pvgis_mtgflow/downstream_dense/seed_15/anomaly_scores.csv"
)
STGAN = ROOT / ".analysis/stgan_results_seed20/anomaly_scores.csv"
OUT = ROOT / ".analysis/mtgflow_stgan_comparison"
OUT.mkdir(parents=True, exist_ok=True)


def binary_metrics(mtg: np.ndarray, stgan: np.ndarray) -> dict[str, float | int]:
    total = len(mtg)
    n_mtg = int(mtg.sum())
    n_stgan = int(stgan.sum())
    both = int(np.logical_and(mtg, stgan).sum())
    union = int(np.logical_or(mtg, stgan).sum())
    expected = n_mtg * n_stgan / total if total else np.nan
    observed_agreement = float((mtg == stgan).mean()) if total else np.nan
    p_mtg = n_mtg / total if total else np.nan
    p_stgan = n_stgan / total if total else np.nan
    chance_agreement = p_mtg * p_stgan + (1 - p_mtg) * (1 - p_stgan)
    kappa = (
        (observed_agreement - chance_agreement) / (1 - chance_agreement)
        if chance_agreement < 1
        else np.nan
    )
    return {
        "n": total,
        "mtgflow_anomalies": n_mtg,
        "stgan_anomalies": n_stgan,
        "both_anomalies": both,
        "union_anomalies": union,
        "mtgflow_only": n_mtg - both,
        "stgan_only": n_stgan - both,
        "mtgflow_rate": n_mtg / total if total else np.nan,
        "stgan_rate": n_stgan / total if total else np.nan,
        "jaccard": both / union if union else np.nan,
        "stgan_covered_by_mtgflow": both / n_stgan if n_stgan else np.nan,
        "mtgflow_covered_by_stgan": both / n_mtg if n_mtg else np.nan,
        "overlap_coefficient": both / min(n_mtg, n_stgan)
        if min(n_mtg, n_stgan)
        else np.nan,
        "expected_both_if_independent": expected,
        "overlap_lift": both / expected if expected else np.nan,
        "binary_agreement": observed_agreement,
        "cohen_kappa": kappa,
    }


print("Loading MTGFlow flags...", flush=True)
mtg = pd.read_csv(
    MTG,
    usecols=["location", "timestamp", "is_anomaly"],
    dtype={"location": "int32", "is_anomaly": "bool"},
    parse_dates=["timestamp"],
)
print(f"MTGFlow rows: {len(mtg):,}", flush=True)
print("Loading STGAN flags...", flush=True)
stgan = pd.read_csv(
    STGAN,
    usecols=["location", "timestamp", "is_anomaly"],
    dtype={"location": "int32", "is_anomaly": "bool"},
    parse_dates=["timestamp"],
)
print(f"STGAN rows: {len(stgan):,}", flush=True)

start, end = mtg["timestamp"].min(), mtg["timestamp"].max()
stgan = stgan.loc[stgan["timestamp"].between(start, end)].reset_index(drop=True)
mtg = mtg.reset_index(drop=True)
if len(mtg) != len(stgan):
    raise ValueError(f"Different overlap lengths: MTGFlow={len(mtg)}, STGAN={len(stgan)}")
if not np.array_equal(mtg["location"].to_numpy(), stgan["location"].to_numpy()):
    raise ValueError("Location order differs in the common interval.")
if not np.array_equal(mtg["timestamp"].to_numpy(), stgan["timestamp"].to_numpy()):
    raise ValueError("Timestamp order differs in the common interval.")

comparison = pd.DataFrame(
    {
        "location": mtg["location"].to_numpy(dtype=np.int32),
        "timestamp": mtg["timestamp"].to_numpy(),
        "mtgflow": mtg["is_anomaly"].to_numpy(dtype=bool),
        "stgan": stgan["is_anomaly"].to_numpy(dtype=bool),
    }
)
del mtg, stgan
comparison["both"] = comparison["mtgflow"] & comparison["stgan"]
comparison["either"] = comparison["mtgflow"] | comparison["stgan"]

overall = binary_metrics(
    comparison["mtgflow"].to_numpy(), comparison["stgan"].to_numpy()
)
overall.update(
    {
        "overlap_start": start,
        "overlap_end": end,
        "n_locations": int(comparison["location"].nunique()),
        "timestamps_per_location": int(
            comparison.groupby("location", observed=True).size().iloc[0]
        ),
    }
)
pd.DataFrame([overall]).to_csv(OUT / "overall.csv", index=False)

clock_daylight_mask = comparison["timestamp"].dt.hour.between(6, 18)
clock_daylight = binary_metrics(
    comparison.loc[clock_daylight_mask, "mtgflow"].to_numpy(),
    comparison.loc[clock_daylight_mask, "stgan"].to_numpy(),
)
clock_daylight["scope"] = "clock_hours_06_18_inclusive"
clock_night = binary_metrics(
    comparison.loc[~clock_daylight_mask, "mtgflow"].to_numpy(),
    comparison.loc[~clock_daylight_mask, "stgan"].to_numpy(),
)
clock_night["scope"] = "clock_hours_19_05"
pd.DataFrame([clock_daylight, clock_night]).to_csv(
    OUT / "clock_daylight_vs_night.csv", index=False
)

comparison["day"] = comparison["timestamp"].dt.floor("D")
daily = (
    comparison.groupby("day", observed=True)
    .agg(
        n=("location", "size"),
        mtgflow_anomalies=("mtgflow", "sum"),
        stgan_anomalies=("stgan", "sum"),
        both_anomalies=("both", "sum"),
        union_anomalies=("either", "sum"),
        mtgflow_locations=("location", lambda x: x[comparison.loc[x.index, "mtgflow"]].nunique()),
        stgan_locations=("location", lambda x: x[comparison.loc[x.index, "stgan"]].nunique()),
        both_locations=("location", lambda x: x[comparison.loc[x.index, "both"]].nunique()),
    )
    .reset_index()
)
daily["mtgflow_rate"] = daily["mtgflow_anomalies"] / daily["n"]
daily["stgan_rate"] = daily["stgan_anomalies"] / daily["n"]
daily["jaccard"] = daily["both_anomalies"] / daily["union_anomalies"].replace(0, np.nan)
daily["stgan_covered_by_mtgflow"] = daily["both_anomalies"] / daily[
    "stgan_anomalies"
].replace(0, np.nan)
daily.to_csv(OUT / "daily.csv", index=False)

timestamp = (
    comparison.groupby("timestamp", observed=True)
    .agg(
        mtgflow_anomalies=("mtgflow", "sum"),
        stgan_anomalies=("stgan", "sum"),
        both_anomalies=("both", "sum"),
    )
    .reset_index()
)
timestamp["mtgflow_share"] = timestamp["mtgflow_anomalies"] / overall["n_locations"]
timestamp["stgan_share"] = timestamp["stgan_anomalies"] / overall["n_locations"]
timestamp["both_share"] = timestamp["both_anomalies"] / overall["n_locations"]
timestamp.to_csv(OUT / "timestamp.csv", index=False)

location = (
    comparison.groupby("location", observed=True)
    .agg(
        n=("timestamp", "size"),
        mtgflow_anomalies=("mtgflow", "sum"),
        stgan_anomalies=("stgan", "sum"),
        both_anomalies=("both", "sum"),
        union_anomalies=("either", "sum"),
    )
    .reset_index()
)
location["mtgflow_rate"] = location["mtgflow_anomalies"] / location["n"]
location["stgan_rate"] = location["stgan_anomalies"] / location["n"]
location["jaccard"] = location["both_anomalies"] / location["union_anomalies"].replace(
    0, np.nan
)
location["stgan_covered_by_mtgflow"] = location["both_anomalies"] / location[
    "stgan_anomalies"
].replace(0, np.nan)
location.to_csv(OUT / "location.csv", index=False)

events = {
    "april_dust_23_26": ("2019-04-23", "2019-04-26"),
    "june_extreme_28_29": ("2019-06-28", "2019-06-29"),
    "july_02": ("2019-07-02", "2019-07-02"),
    "july_full": ("2019-07-01", "2019-07-31"),
}
event_rows = []
event_clock_daylight_rows = []
for name, (event_start, event_end) in events.items():
    mask = comparison["day"].between(pd.Timestamp(event_start), pd.Timestamp(event_end))
    frame = comparison.loc[mask]
    metrics = binary_metrics(frame["mtgflow"].to_numpy(), frame["stgan"].to_numpy())
    metrics.update(
        {
            "event": name,
            "start": event_start,
            "end": event_end,
            "mtgflow_locations": int(frame.loc[frame["mtgflow"], "location"].nunique()),
            "stgan_locations": int(frame.loc[frame["stgan"], "location"].nunique()),
            "both_locations": int(frame.loc[frame["both"], "location"].nunique()),
        }
    )
    event_rows.append(metrics)
    daylight_frame = frame.loc[frame["timestamp"].dt.hour.between(6, 18)]
    daylight_metrics = binary_metrics(
        daylight_frame["mtgflow"].to_numpy(), daylight_frame["stgan"].to_numpy()
    )
    daylight_metrics.update(
        {
            "event": name,
            "start": event_start,
            "end": event_end,
            "clock_hours": "06-18_inclusive",
            "mtgflow_locations": int(
                daylight_frame.loc[daylight_frame["mtgflow"], "location"].nunique()
            ),
            "stgan_locations": int(
                daylight_frame.loc[daylight_frame["stgan"], "location"].nunique()
            ),
            "both_locations": int(
                daylight_frame.loc[daylight_frame["both"], "location"].nunique()
            ),
        }
    )
    event_clock_daylight_rows.append(daylight_metrics)
event_table = pd.DataFrame(event_rows)
event_table.to_csv(OUT / "events.csv", index=False)
event_clock_daylight_table = pd.DataFrame(event_clock_daylight_rows)
event_clock_daylight_table.to_csv(OUT / "events_clock_daylight.csv", index=False)

hour = (
    comparison.assign(hour=comparison["timestamp"].dt.hour)
    .groupby("hour", observed=True)
    .agg(
        n=("location", "size"),
        mtgflow_anomalies=("mtgflow", "sum"),
        stgan_anomalies=("stgan", "sum"),
        both_anomalies=("both", "sum"),
    )
    .reset_index()
)
for detector in ("mtgflow", "stgan", "both"):
    hour[f"{detector}_rate"] = hour[f"{detector}_anomalies"] / hour["n"]
hour.to_csv(OUT / "hour.csv", index=False)

top_mtg_days = daily.nlargest(15, "mtgflow_anomalies")
top_stgan_days = daily.nlargest(15, "stgan_anomalies")
top_common_days = daily.nlargest(15, "both_anomalies")
top_mtg_times = timestamp.nlargest(15, "mtgflow_anomalies")
top_stgan_times = timestamp.nlargest(15, "stgan_anomalies")
top_common_times = timestamp.nlargest(15, "both_anomalies")

print("\nOVERALL")
for key, value in overall.items():
    print(f"{key}: {value}")
print("\nCORRELATIONS")
print(
    "daily anomaly counts Pearson:",
    daily[["mtgflow_anomalies", "stgan_anomalies"]].corr().iloc[0, 1],
)
print(
    "timestamp regional shares Pearson:",
    timestamp[["mtgflow_share", "stgan_share"]].corr().iloc[0, 1],
)
print(
    "location anomaly rates Pearson:",
    location[["mtgflow_rate", "stgan_rate"]].corr().iloc[0, 1],
)
print("\nEVENTS")
print(event_table.to_string(index=False))
print("\nCLOCK DAYLIGHT 06-18")
print(pd.DataFrame([clock_daylight]).to_string(index=False))
print("\nEVENTS CLOCK DAYLIGHT 06-18")
print(event_clock_daylight_table.to_string(index=False))
print("\nTOP MTGFLOW DAYS")
print(top_mtg_days[["day", "mtgflow_anomalies", "stgan_anomalies", "both_anomalies", "jaccard"]].to_string(index=False))
print("\nTOP STGAN DAYS")
print(top_stgan_days[["day", "mtgflow_anomalies", "stgan_anomalies", "both_anomalies", "jaccard"]].to_string(index=False))
print("\nTOP COMMON DAYS")
print(top_common_days[["day", "mtgflow_anomalies", "stgan_anomalies", "both_anomalies", "jaccard"]].to_string(index=False))
print("\nTOP MTGFLOW TIMESTAMPS")
print(top_mtg_times.to_string(index=False))
print("\nTOP STGAN TIMESTAMPS")
print(top_stgan_times.to_string(index=False))
print("\nTOP COMMON TIMESTAMPS")
print(top_common_times.to_string(index=False))
