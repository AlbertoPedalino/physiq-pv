"""Geographical anomaly frequency over all scored hours, including night."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from physiq_pv.reporting.pointwise_detector_posthoc import (
    _normalise_timestamp,
    _rerank_clean_top_k,
    detect_isolated_regional_solar_dropouts,
)


def build_detector_spatial_frequency(score_path, pvgis_path, out_dir, *, detector):
    """Save one point per PVGIS location, exact counts, coverage and metadata.

    STGAN uses the same quality-filtered global top-1% as its post-hoc notebook.
    MTGFlow keeps its saved per-location thresholds. Both exclude the regional
    solar dropout/recovery timestamps. No POA/daytime or forecast-horizon filter
    is applied. Missing coordinates remain missing, never normal.
    """
    detector = detector.lower()
    if detector not in {"stgan", "mtgflow"}:
        raise ValueError("detector must be stgan or mtgflow.")
    with xr.open_dataset(pvgis_path) as dataset:
        locations = pd.DataFrame({
            "location": dataset["location"].values.astype(str),
            "latitude": dataset["lat"].values,
            "longitude": dataset["lon"].values,
        })
        times = pd.DatetimeIndex(_normalise_timestamp(pd.Series(dataset["time"].values)))
    if locations.empty or locations["location"].duplicated().any():
        raise ValueError("PVGIS locations must be nonempty and unique.")
    if not np.isfinite(locations[["latitude", "longitude"]].to_numpy(float)).all():
        raise ValueError("PVGIS coordinates must be finite.")
    if times.empty or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("PVGIS times must be nonempty, unique and increasing.")
    if len(times) > 1 and not (times[1:] - times[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError("PVGIS must use an hourly grid.")

    quality = detect_isolated_regional_solar_dropouts(pvgis_path)
    excluded = pd.DatetimeIndex(quality["timestamp"])
    valid_times = times[~times.isin(excluded)]
    columns = ["location", "timestamp", "anomaly_score"]
    if detector == "mtgflow":
        columns.append("threshold")
    elif "global_rank" in pd.read_csv(score_path, nrows=0).columns:
        columns.append("global_rank")
    scores = pd.read_csv(score_path, usecols=columns, dtype={"location": str})
    scores["timestamp"] = _normalise_timestamp(scores["timestamp"])
    if scores["location"].isna().any() or not scores["location"].isin(locations["location"]).all():
        raise ValueError("Score locations do not align with PVGIS.")
    if not scores["timestamp"].isin(times).all():
        raise ValueError("Score timestamps do not align with the PVGIS period/grid.")
    if scores.duplicated(["location", "timestamp"]).any():
        raise ValueError("Duplicate score location/timestamp coordinates.")
    excluded_rows = int(scores["timestamp"].isin(excluded).sum())
    scores = scores.loc[~scores["timestamp"].isin(excluded)].copy()
    if scores.empty:
        raise ValueError("No scores remain after the quality filter.")
    for column in ("anomaly_score", "threshold") if detector == "mtgflow" else ("anomaly_score",):
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
        if not np.isfinite(scores[column].to_numpy(float)).all():
            raise ValueError(f"Non-finite {column} in quality-eligible scores.")
    if detector == "stgan":
        scores = _rerank_clean_top_k(scores, 1.0)
        decision_source = "global top-1% reranked after quality exclusion"
    else:
        if scores.groupby("location")["threshold"].nunique().ne(1).any():
            raise ValueError("MTGFlow requires one constant saved threshold per location.")
        scores["is_anomaly"] = scores["anomaly_score"] >= scores["threshold"]
        decision_source = "anomaly_score >= saved per-location threshold"

    counts = scores.groupby("location", observed=True).agg(
        n_valid_hours=("is_anomaly", "size"),
        n_anomalous_hours=("is_anomaly", "sum"),
    )
    summary = locations.merge(counts, on="location", how="left", validate="one_to_one")
    for column in ("n_valid_hours", "n_anomalous_hours"):
        summary[column] = summary[column].fillna(0).astype(int)
    summary["anomaly_share_pct"] = (
        100 * summary["n_anomalous_hours"]
        / summary["n_valid_hours"].where(summary["n_valid_hours"].gt(0))
    )
    summary["n_eligible_hours"] = len(valid_times)
    summary["coverage_pct"] = 100 * summary["n_valid_hours"] / len(valid_times)
    summary = summary.sort_values(
        ["anomaly_share_pct", "n_anomalous_hours", "location"],
        ascending=[False, False, True], na_position="last",
    ).reset_index(drop=True)

    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from matplotlib.ticker import PercentFormatter

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"{detector}_spatial_frequency_all_hours"
    paths = {"figure": output / f"{prefix}.png", "locations": output / f"{prefix}.csv",
             "metadata": output / f"{prefix}_metadata.json"}
    fig, axis = plt.subplots(figsize=(10, 10), layout="constrained")
    missing = summary["n_valid_hours"].eq(0)
    axis.scatter(summary.loc[missing, "longitude"], summary.loc[missing, "latitude"],
                 s=24, marker="s", color="#b8b8b8", label="Nessuna ora valida")
    points = axis.scatter(
        summary.loc[~missing, "longitude"], summary.loc[~missing, "latitude"],
        c=summary.loc[~missing, "anomaly_share_pct"], s=24, marker="s",
        cmap="YlOrRd", norm=PowerNorm(gamma=0.5, vmin=0, vmax=100), linewidths=0,
    )
    axis.set_aspect(1 / np.cos(np.deg2rad(summary["latitude"].mean())))
    axis.set(
        xlabel="Longitudine [°E]", ylabel="Latitudine [°N]",
        title=f"{detector.upper()} — frequenza delle anomalie in Piemonte\n"
              f"{times.min():%Y-%m-%d} – {times.max():%Y-%m-%d} · giorno e notte · "
              f"{len(summary):,} località",
    )
    axis.grid(alpha=0.2)
    if missing.any():
        axis.legend(loc="best")
    colorbar = fig.colorbar(points, ax=axis, shrink=0.8, ticks=[0, 1, 5, 10, 25, 50, 100])
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    colorbar.set_label("Ore anomale / ore valide [%] — scala colore a radice quadrata")
    fig.savefig(paths["figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)
    summary.to_csv(paths["locations"], index=False)
    metadata = {
        "detector": detector, "score_source": str(score_path), "pvgis_source": str(pvgis_path),
        "decision_source": decision_source, "time_scope": "all scored hours, including night",
        "start": str(times.min()), "end": str(times.max()),
        "n_locations": len(summary), "n_valid_coordinates": int(summary["n_valid_hours"].sum()),
        "excluded_quality_timestamps": len(excluded), "excluded_score_rows": excluded_rows,
        "quality_filter": "regional solar dropout and next-hour recovery, same for both detectors",
        "denominator": "valid scored hours per location",
        "coverage_denominator": "PVGIS hourly timestamps excluding quality issues",
        "missing_policy": "missing scores excluded; locations without valid scores shown grey",
        "color_range_pct": [0, 100], "color_norm": "PowerNorm(gamma=0.5)",
    }
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths, summary
