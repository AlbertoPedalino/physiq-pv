"""Regional STGAN views from the already quality-filtered pointwise decisions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from physiq_pv.reporting.pointwise_detector_posthoc import _normalise_timestamp


def aggregate_stgan_region(labels: pd.DataFrame, *, start=None, end=None):
    """Hourly regional prevalence and daily per-location prevalence.

    Consume the cleaned global top-K labels unchanged. Hourly denominators
    count scored locations; daily denominators count scored hours at that
    location. Missing observations are never converted to normal decisions.
    Keep every input location, even if it has no scores in the selected range.
    """
    work = labels[["location", "timestamp", "is_anomaly"]].copy()
    if work.empty or work["location"].isna().any():
        raise ValueError("Regional STGAN analysis requires non-missing locations.")
    work["location"] = work["location"].astype(str)
    work["timestamp"] = _normalise_timestamp(work["timestamp"])
    if work["is_anomaly"].isna().any() or not work["is_anomaly"].isin([True, False]).all():
        raise ValueError("STGAN decisions must be boolean and non-missing.")
    work["is_anomaly"] = work["is_anomaly"].astype(bool)
    if work.duplicated(["location", "timestamp"]).any():
        raise ValueError("Duplicate STGAN location/timestamp coordinates.")
    locations = sorted(work["location"].unique(), key=lambda value: (
        (0, int(value), value) if value.isdecimal() else (1, value, value)
    ))
    times = pd.DatetimeIndex(work["timestamp"].unique()).sort_values().as_unit("ns")
    hour_ns = pd.Timedelta(hours=1).value
    phase = int(times.asi8[0] % hour_ns)
    if np.any(times.asi8 % hour_ns != phase):
        raise ValueError("STGAN observations must lie on one common hourly grid.")
    lower = pd.to_datetime(start, utc=True).tz_convert(None) if start is not None else times[0]
    upper = pd.to_datetime(end, utc=True).tz_convert(None) if end is not None else times[-1]
    if pd.isna(lower) or pd.isna(upper) or lower > upper:
        raise ValueError("Regional date interval is invalid.")
    work = work.loc[work["timestamp"].between(lower, upper)].copy()
    if work.empty:
        raise ValueError("No valid STGAN observations in the selected date interval.")
    # Respect PVGIS's hourly offset (e.g. :10). Explicit date bounds preserve
    # even wholly missing leading/trailing hours within the requested interval.
    first = lower.floor("h") + pd.Timedelta(phase, unit="ns")
    if first < lower:
        first += pd.Timedelta(hours=1)
    last = upper.floor("h") + pd.Timedelta(phase, unit="ns")
    if last > upper:
        last -= pd.Timedelta(hours=1)
    grid = pd.date_range(first, last, freq="h", name="timestamp")
    hourly = work.groupby("timestamp", observed=True).agg(
        n_valid_locations=("location", "size"), n_anomalous_locations=("is_anomaly", "sum"),
    ).reindex(grid, fill_value=0)
    hourly["anomaly_share_pct"] = (
        100 * hourly["n_anomalous_locations"]
        / hourly["n_valid_locations"].where(hourly["n_valid_locations"].gt(0))
    )
    hourly["location_coverage_pct"] = 100 * hourly["n_valid_locations"] / len(locations)
    days = pd.date_range(first.normalize(), last.normalize(), freq="D", name="day")
    daily_index = pd.MultiIndex.from_product([locations, days], names=["location", "day"])
    daily = work.assign(day=work["timestamp"].dt.normalize()).groupby(
        ["location", "day"], observed=True,
    ).agg(n_valid_hours=("is_anomaly", "size"), n_anomalous_hours=("is_anomaly", "sum"))
    daily = daily.reindex(daily_index, fill_value=0)
    daily["anomalous_hours_pct"] = (
        100 * daily["n_anomalous_hours"]
        / daily["n_valid_hours"].where(daily["n_valid_hours"].gt(0))
    )
    return hourly.reset_index(), daily.reset_index()


def build_stgan_regional_overview(labels, out_dir, *, start=None, end=None):
    """Save a two-panel figure, exact hourly/daily CSVs and plotting metadata."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    hourly, daily = aggregate_stgan_region(labels, start=start, end=end)
    locations = daily["location"].drop_duplicates().tolist()
    days = pd.DatetimeIndex(daily["day"].drop_duplicates()).sort_values()
    matrix = daily.pivot(index="location", columns="day", values="anomalous_hours_pct")
    matrix = matrix.reindex(index=locations, columns=days)
    fig, (curve_ax, heat_ax) = plt.subplots(
        2, 1, figsize=(17, 12), sharex=True,
        gridspec_kw={"height_ratios": [1, 2.4]}, layout="constrained",
    )
    curve_ax.plot(hourly["timestamp"], hourly["anomaly_share_pct"],
                  color="tab:red", linewidth=1.0, label="Località anomale / località valide")
    curve_ax.fill_between(hourly["timestamp"], hourly["anomaly_share_pct"],
                          color="tab:red", alpha=0.12)
    curve_ax.set_ylabel("Località anomale [%]")
    curve_ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    max_share = float(hourly["anomaly_share_pct"].max())
    curve_ax.set_ylim(0, min(100, max(1.0, max_share * 1.08)))
    curve_ax.grid(alpha=0.25)
    valid_counts = hourly["n_valid_locations"]
    curve_ax.set_title(
        f"STGAN — tutte le {len(locations):,} località, ora per ora\n"
        f"Località con score valido per ora: {int(valid_counts.min()):,}–{int(valid_counts.max()):,}"
    )
    curve_ax.legend(loc="upper right")
    cmap = plt.colormaps["YlOrRd"].copy()
    cmap.set_bad("#b8b8b8")
    left = mdates.date2num(days[0])
    right = mdates.date2num(days[-1] + pd.Timedelta(days=1))
    heat = heat_ax.imshow(
        np.ma.masked_invalid(matrix.to_numpy(float)),
        aspect="auto", interpolation="nearest", origin="upper",
        extent=(left, right, len(locations) - 0.5, -0.5),
        cmap=cmap, vmin=0, vmax=100,
    )
    tick_positions = np.unique(np.linspace(0, len(locations) - 1, min(16, len(locations))).astype(int))
    heat_ax.set_yticks(tick_positions, [locations[index] for index in tick_positions])
    heat_ax.set(ylabel="Località (una riga per località, ordine ID)", xlabel="Data (UTC)",
                title="Per località e giorno: ore anomale / ore con score valido")
    heat_ax.set_xlim(left, right)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=12)
    heat_ax.xaxis.set_major_locator(locator)
    heat_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    heat_ax.legend(handles=[Patch(facecolor="#b8b8b8", label="Nessuna ora valida")],
                   loc="upper right")
    colorbar = fig.colorbar(heat, ax=[curve_ax, heat_ax], fraction=0.025, pad=0.015)
    colorbar.set_label("Ore anomale nel giorno [% delle ore valide]")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "figure": output / "stgan_regional_overview.png",
        "hourly": output / "stgan_regional_hourly.csv",
        "daily": output / "stgan_regional_daily_by_location.csv",
        "metadata": output / "stgan_regional_metadata.json",
    }
    fig.savefig(paths["figure"], dpi=180, bbox_inches="tight")
    plt.close(fig)
    hourly.to_csv(paths["hourly"], index=False)
    daily.to_csv(paths["daily"], index=False)
    metadata = {
        "n_locations": len(locations), "n_valid_coordinates": int(daily["n_valid_hours"].sum()),
        "start": str(hourly["timestamp"].min()), "end": str(hourly["timestamp"].max()),
        "clean_top_k_percent": labels.attrs.get("top_percent"),
        "score_cutoff": labels.attrs.get("score_cutoff"),
        "excluded_quality_timestamps": labels.attrs.get("excluded_timestamps"),
        "decision_source": "cleaned pointwise STGAN labels; no regional threshold fitted",
        "time_scope": "all scored hours, including night",
        "hourly_denominator": "locations with valid score at that timestamp",
        "daily_denominator": "valid scored hours at that location and day",
        "missing_policy": "hourly gaps and grey heatmap cells; never normal by imputation",
        "heatmap_location_order": "numeric ID when possible, otherwise lexical; all locations retained",
        "heatmap_color_range_pct": [0, 100],
    }
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths
