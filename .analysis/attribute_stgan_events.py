from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STGAN = ROOT / ".analysis/stgan_results_seed20"
MTG = ROOT / ".analysis/mtgflow_results_seed15/outputs/pvgis_mtgflow/downstream_dense/seed_15"

entity_path = STGAN / "entity_anomaly_scores.csv"
print("ENTITY HEADER/SAMPLE")
print(pd.read_csv(entity_path, nrows=5).to_string(index=False))

events = {
    "may_27": ("2019-05-27", "2019-05-27"),
    "jun_09_10": ("2019-06-09", "2019-06-10"),
    "jun_22": ("2019-06-22", "2019-06-22"),
    "aug_07": ("2019-08-07", "2019-08-07"),
    "aug_22": ("2019-08-22", "2019-08-22"),
    "aug_28": ("2019-08-28", "2019-08-28"),
    "sep_10": ("2019-09-10", "2019-09-10"),
    "sep_19": ("2019-09-19", "2019-09-19"),
    "sep_26_29": ("2019-09-26", "2019-09-29"),
    "oct_13_18": ("2019-10-13", "2019-10-18"),
    "nov_27": ("2019-11-27", "2019-11-27"),
    "dec_01": ("2019-12-01", "2019-12-01"),
}

entity = pd.read_csv(entity_path, parse_dates=["timestamp"])
print("\nENTITY COLUMNS", list(entity.columns))
numeric = [c for c in entity.columns if c not in {"location", "timestamp", "method", "entity", "feature"}]
print("NUMERIC", numeric)

locations = pd.read_csv(STGAN / "locations.csv")

stgan_flags = pd.read_csv(
    STGAN / "anomaly_scores.csv",
    usecols=["location", "timestamp", "is_anomaly"],
    parse_dates=["timestamp"],
)
stgan_flags = stgan_flags.loc[stgan_flags["is_anomaly"]]
mtg_flags = pd.read_csv(
    MTG / "anomaly_scores.csv",
    usecols=["location", "timestamp", "is_anomaly"],
    parse_dates=["timestamp"],
)
mtg_flags = mtg_flags.loc[mtg_flags["is_anomaly"]]

rows = []
for name, (start, end) in events.items():
    mask = entity["timestamp"].dt.floor("D").between(start, end)
    frame = entity.loc[mask].copy()
    row = {"event": name, "rows": len(frame), "locations": frame["location"].nunique()}
    for col in numeric:
        row[f"{col}_mean"] = pd.to_numeric(frame[col], errors="coerce").mean()
        row[f"{col}_median"] = pd.to_numeric(frame[col], errors="coerce").median()
    feature_col = "entity" if "entity" in frame else "feature"
    if feature_col in frame:
        score_means = frame.groupby(feature_col)["anomaly_score"].mean().sort_values(ascending=False)
        contribution_means = frame.groupby(feature_col)["contribution_fraction"].mean().sort_values(ascending=False)
        row["feature_score_order"] = "; ".join(f"{k}={v:.6g}" for k, v in score_means.items())
        row["feature_contribution_order"] = "; ".join(f"{k}={v:.3%}" for k, v in contribution_means.items())
        idx = frame.groupby(["location", "timestamp"])["contribution_fraction"].idxmax()
        dominant = frame.loc[idx, feature_col].value_counts(normalize=True)
        row["dominant_feature_points"] = "; ".join(f"{k}={v:.3%}" for k, v in dominant.items())
    joined = frame[["location"]].drop_duplicates().merge(locations, on="location", how="left")
    row["lat_mean"] = joined["latitude"].mean()
    row["lon_mean"] = joined["longitude"].mean()
    row["lat_min"] = joined["latitude"].min()
    row["lat_max"] = joined["latitude"].max()
    row["lon_min"] = joined["longitude"].min()
    row["lon_max"] = joined["longitude"].max()
    st = stgan_flags.loc[stgan_flags["timestamp"].dt.floor("D").between(start, end), ["location", "timestamp"]]
    mt = mtg_flags.loc[mtg_flags["timestamp"].dt.floor("D").between(start, end), ["location", "timestamp"]]
    exact = st.merge(mt, on=["location", "timestamp"], how="left", indicator=True)
    exact = exact.loc[exact["_merge"] == "left_only"].drop(columns="_merge")
    hourly = exact.assign(hour=exact["timestamp"].dt.hour).groupby("hour").size().sort_values(ascending=False)
    row["stgan_only_peak_hours"] = "; ".join(f"{int(k):02d}={int(v)}" for k, v in hourly.head(5).items())
    peak_times = exact.groupby("timestamp").size().sort_values(ascending=False)
    row["stgan_only_peak_timestamps"] = "; ".join(f"{k}={int(v)}" for k, v in peak_times.head(3).items())
    if len(peak_times):
        peak = peak_times.index[0]
        peak_locations = exact.loc[exact["timestamp"] == peak, ["location"]].merge(locations, on="location", how="left")
        row["peak_lat_mean"] = peak_locations["latitude"].mean()
        row["peak_lon_mean"] = peak_locations["longitude"].mean()
        row["peak_north_pct"] = (peak_locations["latitude"] >= 45.3).mean()
        row["peak_south_pct"] = (peak_locations["latitude"] < 44.7).mean()
        row["peak_west_pct"] = (peak_locations["longitude"] < 7.5).mean()
        row["peak_east_pct"] = (peak_locations["longitude"] >= 8.2).mean()
    rows.append(row)

print("\nEVENT ATTRIBUTION")
print(pd.DataFrame(rows).to_string(index=False))
