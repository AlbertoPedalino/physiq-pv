"""Compare exported detector decisions on matching location/timestamp pairs."""
from pathlib import Path

import numpy as np
import pandas as pd


def compare_stgan_exports(cnn_seed_dir, baseline_seed_dir, *, out_dir):
    """Stream one location at a time; report raw global-top-K agreement, not accuracy.

    Uses the saved decisions without quality filtering or reranking. Existing
    posthoc notebooks remain responsible for clean-top-K and forecast metrics.
    """
    cnn, baseline, out = Path(cnn_seed_dir), Path(baseline_seed_dir), Path(out_dir)
    sites = pd.read_csv(cnn.parent / "locations.csv", dtype={"location": str, "site_key": str})
    baseline_sites = pd.read_csv(baseline.parent / "locations.csv", dtype={"location": str, "site_key": str})
    if sites.location.duplicated().any() or baseline_sites.location.duplicated().any():
        raise ValueError("Location IDs must be unique in each export.")
    if sites.location.isna().any() or baseline_sites.location.isna().any():
        raise ValueError("Location IDs must not be missing.")
    # An identical ID must refer to the same physical site in both runs.
    paired_sites = sites.merge(baseline_sites, on="location", suffixes=("_cnn", "_baseline"),
                               validate="one_to_one")
    for coordinate in ("latitude", "longitude"):
        a = paired_sites[f"{coordinate}_cnn"].to_numpy(dtype=float)
        b = paired_sites[f"{coordinate}_baseline"].to_numpy(dtype=float)
        if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a, b, rtol=0, atol=1e-6):
            raise ValueError(f"Mismatched {coordinate} for shared location IDs.")
    original_keys = baseline_sites.set_index("location").site_key.to_dict()
    layout = pd.read_csv(cnn.parent / "grid_locations.csv", dtype={"location": str})
    if (layout.location.duplicated().any() or layout.location.isna().any()
        or set(layout.location) != set(sites.location)):
        raise ValueError("Grid layout must contain exactly the CNN location IDs.")
    if not pd.api.types.is_bool_dtype(layout.complete_patch):
        raise ValueError("Grid complete_patch flags must be non-missing booleans.")
    sites = sites.merge(layout[["location", "complete_patch"]], on="location", validate="one_to_one")
    results = []

    def read(path):
        df = pd.read_csv(path, usecols=["timestamp", "is_anomaly"])
        df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
        if df.timestamp.isna().any() or df.timestamp.duplicated().any():
            raise ValueError(f"Duplicate timestamps: {path}")
        if not pd.api.types.is_bool_dtype(df.is_anomaly):
            raise ValueError(f"Expected boolean saved decisions: {path}")
        return df

    for row in sites.itertuples(index=False):
        a = read(cnn / "locations" / row.site_key / "test_scores.csv")
        key = original_keys.get(row.location)
        b = (read(baseline / "locations" / key / "test_scores.csv") if key is not None
             else pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                                "is_anomaly": pd.Series(dtype=bool)}))
        common = a.merge(b, on="timestamp", suffixes=("_cnn", "_baseline"), validate="one_to_one")
        x, y = common.is_anomaly_cnn, common.is_anomaly_baseline
        results.append({"location": row.location,
            "group": "interior" if row.complete_patch else "boundary",
            "n_cnn": len(a), "n_baseline": len(b), "n_common": len(common),
            "n_cnn_only_timestamps": len(a)-len(common), "n_baseline_only_timestamps": len(b)-len(common),
            "both_anomalous": int((x & y).sum()), "cnn_only_anomalous": int((x & ~y).sum()),
            "baseline_only_anomalous": int((~x & y).sum()), "both_normal": int((~x & ~y).sum())})
    # Keep sites present only in the baseline in the coverage audit. They
    # cannot be assigned to CNN interior/boundary or counted as normal.
    for row in baseline_sites.loc[~baseline_sites.location.isin(sites.location)].itertuples(index=False):
        b = read(baseline / "locations" / row.site_key / "test_scores.csv")
        results.append({"location": row.location, "group": "baseline_only_location",
            "n_cnn": 0, "n_baseline": len(b), "n_common": 0,
            "n_cnn_only_timestamps": 0, "n_baseline_only_timestamps": len(b),
            "both_anomalous": 0, "cnn_only_anomalous": 0,
            "baseline_only_anomalous": 0, "both_normal": 0})
    by_location = pd.DataFrame(results)
    grouped = by_location.groupby("group", sort=False).sum(numeric_only=True)
    grouped.loc["all"] = grouped.sum()
    union = grouped.both_anomalous + grouped.cnn_only_anomalous + grouped.baseline_only_anomalous
    grouped["anomaly_jaccard"] = grouped.both_anomalous / union.replace(0, np.nan)
    denominator = grouped.n_common.replace(0, np.nan)
    grouped["cnn_anomaly_share_pct"] = 100*(grouped.both_anomalous+grouped.cnn_only_anomalous)/denominator
    grouped["baseline_anomaly_share_pct"] = 100*(grouped.both_anomalous+grouped.baseline_only_anomalous)/denominator
    grouped["decision_protocol"] = "saved_global_top_k_no_quality_filter_no_reranking"
    out.mkdir(parents=True, exist_ok=True)
    by_location.to_csv(out/"detector_comparison_by_location.csv", index=False)
    grouped.reset_index().to_csv(out/"detector_comparison_by_boundary.csv", index=False)
    return grouped.reset_index(), by_location
