"""Driver-resolved anomaly analysis.

MTGFlow scores every entity separately (paper Eq. 14) and only the summed
window score reaches the forecasting pipeline.  This module recovers the
per-entity decomposition and answers a different question than the event
notebooks: not *when* the model degrades, but *which physical variable* made
the window improbable when it did.

The entity CSV is large (one row per window and entity), so it is never loaded
whole: a streaming pass reduces it to one compact label per anomalous
(location, timestamp).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .posthoc_outputs import (
    DAYTIME_IRRADIANCE_THRESHOLD_WM2,
    GROUP_NORMAL,
    PERCENT_PRODUCTION_BINS,
    REFERENCE_PRODUCTION_PEAKS_FILE,
)

# Short names for the three PVGIS climate entities scored by the detector.
DRIVER_NAMES = {
    "solar_irradiance_poa": "solar",
    "temperature_2m": "temperature",
    "wind_speed_10m": "wind",
}
MULTI_DRIVER = "multi"
NORMAL_CATEGORY = "normal"
DRIVER_LABELS_FILE = "entity_driver_labels.csv"
DRIVER_METRICS_FILE = "anomaly_driver_comparison_metrics.csv"

# How an anomalous window becomes a comparison category.
#
# ``driver``      the dominant entity always wins, so a window driven by
#                 irradiance stays in ``solar`` even when a second entity also
#                 crosses its threshold.  Multi-entity windows are visible
#                 through ``n_entities_flagged`` instead of being hidden in a
#                 separate bucket.
# ``multi_split`` windows with two or more flagged entities form their own
#                 category, leaving the driver categories single-entity only.
# ``severity``    driver crossed with single/multi, for the finer grid.
CATEGORY_MODES = ("driver", "multi_split", "severity")


def assign_categories(labels: pd.DataFrame, *, mode: str = "driver") -> pd.DataFrame:
    """Return ``labels`` with ``category`` recomputed under ``mode``."""
    if mode not in CATEGORY_MODES:
        raise ValueError(f"Unknown category mode {mode!r}; expected {CATEGORY_MODES}.")
    required = {"driver", "n_entities_flagged"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Driver labels are missing columns {sorted(missing)}.")
    out = labels.copy()
    driver = out["driver"].astype(str)
    is_multi = out["n_entities_flagged"].to_numpy(int) >= 2
    if mode == "driver":
        out["category"] = driver
    elif mode == "multi_split":
        out["category"] = np.where(is_multi, MULTI_DRIVER, driver)
    else:
        out["category"] = driver + np.where(is_multi, "_multi", "_single")
    return out


def _category_order(present: Iterable[str]) -> List[str]:
    """Order categories as driver, then severity variant, then anything else."""
    available = set(present)
    preferred: List[str] = []
    for driver in (*DRIVER_NAMES.values(), MULTI_DRIVER):
        for candidate in (driver, f"{driver}_single", f"{driver}_multi"):
            if candidate in available:
                preferred.append(candidate)
    return preferred + sorted(available - set(preferred))


def _as_boolean(values: pd.Series, *, context: str) -> pd.Series:
    """Parse an is_anomaly column written either as bool or as text."""
    if pd.api.types.is_bool_dtype(values):
        return values
    text = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    parsed = text.map(mapping)
    if parsed.isna().any():
        examples = sorted(text[parsed.isna()].unique().tolist())[:5]
        raise ValueError(f"{context}: invalid is_anomaly values {examples}.")
    return parsed.astype(bool)


def _anomalous_index(
    global_scores_path: str | Path, *, chunksize: int
) -> pd.MultiIndex:
    """Return the (location, timestamp) pairs the pipeline treats as rare."""
    path = Path(global_scores_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is required.")
    available = set(pd.read_csv(path, nrows=0).columns)
    required = {"location", "timestamp", "is_anomaly"}
    missing = required - available
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}.")
    parts: List[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        usecols=["location", "timestamp", "is_anomaly"],
        dtype={"location": "string"},
        chunksize=chunksize,
    )
    for chunk in reader:
        flags = _as_boolean(chunk["is_anomaly"], context=str(path))
        selected = chunk.loc[flags.to_numpy(), ["location", "timestamp"]]
        if not selected.empty:
            selected = selected.copy()
            selected["timestamp"] = pd.to_datetime(selected["timestamp"])
            parts.append(selected)
    if not parts:
        raise ValueError(f"{path} flags no anomalous window.")
    rare = pd.concat(parts, ignore_index=True).drop_duplicates()
    return pd.MultiIndex.from_arrays(
        [rare["location"].astype(str), rare["timestamp"]],
        names=["location", "timestamp"],
    )


def build_driver_labels(
    entity_scores_path: str | Path,
    global_scores_path: str | Path,
    *,
    out_path: Optional[str | Path] = None,
    chunksize: int = 2_000_000,
) -> Dict[str, object]:
    """Reduce per-entity scores to one driver label per anomalous window.

    ``category`` is ``multi`` when two or more entities exceed their own
    Eq. 15 threshold, otherwise the entity with the largest contribution to the
    window score.  The argmax is always defined, so every window the pipeline
    calls rare receives a driver even when no single entity crosses its
    threshold.
    """
    entity_path = Path(entity_scores_path)
    if not entity_path.exists():
        raise FileNotFoundError(f"{entity_path} is required.")
    available = set(pd.read_csv(entity_path, nrows=0).columns)
    required = {"location", "timestamp", "entity", "anomaly_score", "is_anomaly"}
    missing = required - available
    if missing:
        raise ValueError(f"{entity_path} is missing columns {sorted(missing)}.")

    rare_index = _anomalous_index(global_scores_path, chunksize=chunksize)
    score_parts: List[pd.DataFrame] = []
    flag_parts: List[pd.DataFrame] = []
    reader = pd.read_csv(
        entity_path,
        usecols=sorted(required),
        dtype={"location": "string", "entity": "string"},
        chunksize=chunksize,
    )
    for chunk in reader:
        chunk = chunk.copy()
        chunk["location"] = chunk["location"].astype(str)
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
        keys = pd.MultiIndex.from_arrays(
            [chunk["location"], chunk["timestamp"]],
            names=["location", "timestamp"],
        )
        selected = chunk.loc[keys.isin(rare_index)]
        if selected.empty:
            continue
        # A window's entity rows may straddle a chunk boundary; partial pivots
        # are merged after the scan, which is exact because every
        # (location, timestamp, entity) triple occurs once in the file.
        score_parts.append(
            selected.pivot_table(
                index=["location", "timestamp"],
                columns="entity",
                values="anomaly_score",
                aggfunc="first",
            )
        )
        flags = selected.assign(
            flag=_as_boolean(
                selected["is_anomaly"], context=str(entity_path)
            ).astype(float)
        )
        flag_parts.append(
            flags.pivot_table(
                index=["location", "timestamp"],
                columns="entity",
                values="flag",
                aggfunc="first",
            )
        )
    if not score_parts:
        raise ValueError(
            "No entity rows matched the anomalous windows; check that both "
            "CSVs come from the same detector run and seed."
        )

    scores = pd.concat(score_parts).groupby(level=[0, 1]).max()
    flags = pd.concat(flag_parts).groupby(level=[0, 1]).max().fillna(0.0)
    flags = flags.reindex(columns=scores.columns, fill_value=0.0).astype(bool)
    if scores.isna().any().any():
        incomplete = int(scores.isna().any(axis=1).sum())
        raise ValueError(
            f"{incomplete} anomalous windows lack a score for every entity."
        )

    entity_columns = list(scores.columns)
    dominant = scores.idxmax(axis=1).map(
        lambda name: DRIVER_NAMES.get(name, str(name))
    )
    n_flagged = flags.sum(axis=1).astype(int)
    labels = pd.DataFrame(
        {"driver": dominant, "n_entities_flagged": n_flagged},
        index=scores.index,
    )
    labels["category"] = labels["driver"]
    for column in entity_columns:
        short = DRIVER_NAMES.get(column, str(column))
        labels[f"score_{short}"] = scores[column]
        labels[f"flag_{short}"] = flags[column]
    labels = labels.reset_index().sort_values(["location", "timestamp"])

    resolved_path = None
    if out_path is not None:
        resolved_path = Path(out_path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        labels.to_csv(resolved_path, index=False)

    composition = (
        labels["category"].value_counts().rename_axis("category")
        .reset_index(name="n_windows")
    )
    composition["share"] = composition["n_windows"] / len(labels)
    return {
        "labels": labels,
        "labels_path": resolved_path,
        "composition": composition,
        "entities": entity_columns,
        "n_anomalous_windows": len(labels),
    }


def load_driver_labels(path: str | Path) -> pd.DataFrame:
    """Read a label file written by :func:`build_driver_labels`."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{resolved} is required; build the labels first.")
    labels = pd.read_csv(resolved, dtype={"location": "string"})
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    labels["location"] = labels["location"].astype(str)
    return labels


def driver_composition_by_day(
    labels: pd.DataFrame, days: Iterable[str]
) -> pd.DataFrame:
    """Return the driver mix of specific days, as a physical sanity check."""
    wanted = [pd.Timestamp(day).normalize() for day in days]
    if not wanted:
        raise ValueError("days must contain at least one date.")
    work = labels.copy()
    work["day"] = work["timestamp"].dt.normalize()
    subset = work.loc[work["day"].isin(wanted)]
    if subset.empty:
        return pd.DataFrame(columns=["day", "category", "n_windows", "share"])
    counts = (
        subset.groupby(["day", "category"]).size().rename("n_windows").reset_index()
    )
    totals = counts.groupby("day")["n_windows"].transform("sum")
    counts["share"] = counts["n_windows"] / totals
    counts["day"] = counts["day"].dt.strftime("%Y-%m-%d")
    return counts.sort_values(["day", "n_windows"], ascending=[True, False])


def build_anomaly_driver_comparison_figures(
    out_dir: str | Path,
    labels: pd.DataFrame,
    *,
    figure_subdir: str = "anomaly_driver",
    metrics_name: Optional[str] = None,
    chunksize: int = 500_000,
    coverage_target: float = 0.95,
    clc_eta: float = 9.0,
) -> Dict[str, object]:
    """Compare forecast quality across anomaly drivers.

    Mirrors the extreme-event comparison: every valid daytime prediction row is
    used, metrics are exact, and the histogram places the joint upper 0.5% in
    the final bin.  The only change is the category, which comes from the
    detector's per-entity decomposition instead of a calendar date.
    """
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    predictions_path = out / "predictions.csv"
    peaks_path = out / REFERENCE_PRODUCTION_PEAKS_FILE
    if not predictions_path.exists():
        raise FileNotFoundError(f"{predictions_path} is required.")
    if not peaks_path.exists():
        raise FileNotFoundError(
            f"{peaks_path} is required; run the daytime analysis first."
        )
    reference_peak = float(pd.read_csv(peaks_path)["reference_peak_w"].iloc[0])
    if not np.isfinite(reference_peak) or reference_peak <= 0.0:
        raise ValueError(f"Invalid reference peak: {reference_peak!r}.")

    target_range = reference_peak
    sharpness_path = out / "sharpness_overview.csv"
    if sharpness_path.exists():
        sharpness = pd.read_csv(sharpness_path)
        if "target_range" in sharpness:
            candidates = pd.to_numeric(
                sharpness["target_range"], errors="coerce"
            ).dropna()
            if len(candidates) and float(candidates.iloc[0]) > 0.0:
                target_range = float(candidates.iloc[0])

    available = set(pd.read_csv(predictions_path, nrows=0).columns)

    def choose(*candidates: str) -> str:
        column = next((name for name in candidates if name in available), None)
        if column is None:
            raise ValueError(
                f"predictions.csv requires one of {list(candidates)}."
            )
        return column

    timestamp_col = choose("timestamp", "target_timestamp", "time")
    location_col = choose("location", "location_id", "node_id")
    y_true_col = choose("y_true")
    y_pred_col = choose("y_pred_mean", "y_pred")
    lower_col = choose("lower_pi")
    upper_col = choose("upper_pi")
    solar_col = choose(
        "solar_irradiance_poa_target", "solar_irradiance_poa", "ghi_target"
    )
    group_col = choose("anomaly_group", "event_group")
    usecols = [
        timestamp_col, location_col, y_true_col, y_pred_col,
        lower_col, upper_col, solar_col, group_col,
    ]

    lookup = labels[["location", "timestamp", "category"]].copy()
    lookup["location"] = lookup["location"].astype(str)
    lookup["timestamp"] = pd.to_datetime(lookup["timestamp"])
    if lookup.duplicated(["location", "timestamp"]).any():
        raise ValueError("Driver labels must be unique by location and timestamp.")
    categories = (NORMAL_CATEGORY, *_category_order(lookup["category"].unique()))

    storage = {
        (band[0], category): {"abs_error": [], "row_nmpil": [], "covered": []}
        for band in PERCENT_PRODUCTION_BINS
        for category in categories
    }
    unmatched_rare = 0

    reader = pd.read_csv(
        predictions_path,
        usecols=usecols,
        dtype={timestamp_col: "string", group_col: "string", location_col: "string"},
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        chunk = chunk.copy()
        chunk["_timestamp"] = pd.to_datetime(chunk[timestamp_col], errors="coerce")
        chunk["_location"] = chunk[location_col].astype(str)
        merged = chunk.merge(
            lookup.rename(
                columns={"location": "_location", "timestamp": "_timestamp"}
            ),
            on=["_location", "_timestamp"],
            how="left",
            validate="many_to_one",
        )
        y_true = pd.to_numeric(merged[y_true_col], errors="coerce").to_numpy(float)
        y_pred = pd.to_numeric(merged[y_pred_col], errors="coerce").to_numpy(float)
        lower = pd.to_numeric(merged[lower_col], errors="coerce").to_numpy(float)
        upper = pd.to_numeric(merged[upper_col], errors="coerce").to_numpy(float)
        solar = pd.to_numeric(merged[solar_col], errors="coerce").to_numpy(float)
        group = merged[group_col].astype(str).to_numpy()
        finite = np.isfinite(
            np.column_stack((y_true, y_pred, lower, upper, solar))
        ).all(axis=1)
        valid = finite & (solar > DAYTIME_IRRADIANCE_THRESHOLD_WM2)
        if not valid.any():
            continue

        category = merged["category"].fillna("").to_numpy(dtype=object)
        is_normal = group == GROUP_NORMAL
        category[is_normal & (category == "")] = NORMAL_CATEGORY
        # Rare rows without a driver label mean the two CSVs disagree; they are
        # excluded from every category and reported instead of being merged
        # into the normal stratum.
        unmatched_rare += int((valid & ~is_normal & (category == "")).sum())

        production_pct = np.clip(100.0 * y_true / reference_peak, 0.0, 100.0)
        abs_error = np.abs(y_pred - y_true)
        row_nmpil = (upper - lower) / target_range
        covered = (y_true >= lower) & (y_true <= upper)

        for band_name, lower_pct, upper_pct in PERCENT_PRODUCTION_BINS:
            band_mask = production_pct >= lower_pct
            if upper_pct is not None:
                band_mask &= production_pct < upper_pct
            for name in categories:
                mask = valid & band_mask & (category == name)
                if mask.any():
                    storage[(band_name, name)]["abs_error"].append(abs_error[mask])
                    storage[(band_name, name)]["row_nmpil"].append(row_nmpil[mask])
                    storage[(band_name, name)]["covered"].append(covered[mask])

    def combined(band_name: str, name: str, metric: str) -> np.ndarray:
        parts = storage[(band_name, name)][metric]
        return np.concatenate(parts) if parts else np.empty(0, dtype=float)

    def tukey(values: np.ndarray) -> dict:
        if values.size == 0:
            return {}
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        low = values[values >= q1 - 1.5 * iqr]
        high = values[values <= q3 + 1.5 * iqr]
        return {
            "mean": float(np.mean(values)),
            "med": float(median),
            "q1": float(q1),
            "q3": float(q3),
            "whislo": float(np.min(low)),
            "whishi": float(np.max(high)),
            "fliers": [],
        }

    rows = []
    for band_name, _, _ in PERCENT_PRODUCTION_BINS:
        for name in categories:
            errors = combined(band_name, name, "abs_error")
            nmpil = combined(band_name, name, "row_nmpil")
            covered = combined(band_name, name, "covered")
            if errors.size == 0:
                continue
            picp = float(np.mean(covered))
            rows.append({
                "bin": band_name,
                "category": name,
                "count": int(errors.size),
                "mae": float(np.mean(errors)),
                "rmse": float(np.sqrt(np.mean(errors ** 2))),
                "mpiw": float(np.mean(nmpil) * target_range),
                "nmpil": float(np.mean(nmpil)),
                "picp": picp,
                "clc": float(
                    np.mean(nmpil)
                    * (1.0 + np.exp(-clc_eta * (picp - coverage_target)))
                ),
            })
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise ValueError("No rows available for the driver comparison.")

    figure_dir = out / "figures" / str(figure_subdir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: Dict[str, Path] = {}
    color_map = plt.get_cmap("tab10")
    colors = tuple(color_map(index % 10) for index in range(len(categories)))
    display_labels = tuple(
        "normal 2019" if name == NORMAL_CATEGORY else name.replace("_", " ")
        for name in categories
    )

    def save_boxplot(band_name: str, metric: str, ylabel: str) -> None:
        boxes, ticks, box_colors = [], [], []
        for name, display_label, color in zip(categories, display_labels, colors):
            values = combined(band_name, name, metric)
            stats = tukey(values)
            if not stats:
                continue
            stats["label"] = ""
            boxes.append(stats)
            ticks.append(f"{display_label}\n(n={len(values):,})")
            box_colors.append(color)
        if not boxes:
            return
        fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(boxes)), 4.8))
        ax.bxp(
            boxes,
            showfliers=False,
            showmeans=True,
            meanprops={
                "marker": "D", "markerfacecolor": "red",
                "markeredgecolor": "red", "markersize": 5,
            },
        )
        ax.set_xticks(range(1, len(ticks) + 1))
        ax.set_xticklabels(ticks)
        ax.set(
            title=f"{metric.replace('_', ' ').upper()} — {band_name} (all rows)",
            ylabel=ylabel,
        )
        ax.grid(axis="y", alpha=0.25)
        key = f"driver_compare_{metric}_{band_name}_boxplot"
        path = figure_dir / f"{key}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        figure_paths[key] = path

    def save_histogram(band_name: str) -> None:
        groups = [combined(band_name, name, "abs_error") for name in categories]
        nonempty = [values for values in groups if values.size]
        if not nonempty:
            return
        all_values = np.concatenate(nonempty)
        cap = max(float(np.quantile(all_values, 0.995)), 1e-6)
        edges = np.linspace(0.0, cap, 41)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for values, display_label, color in zip(groups, display_labels, colors):
            if values.size == 0:
                continue
            clipped = np.minimum(values, np.nextafter(cap, 0.0))
            weights = np.full(values.size, 1.0 / values.size)
            ax.hist(
                clipped,
                bins=edges,
                weights=weights,
                histtype="step",
                linewidth=2,
                label=f"{display_label} (n={len(values):,})",
                color=color,
            )
        ax.set(
            title=(
                f"Absolute-error distribution — {band_name} "
                "(all rows; upper 0.5% in final bin)"
            ),
            xlabel="Absolute error [W]",
            ylabel="Fraction of category",
        )
        ax.legend()
        ax.grid(alpha=0.25)
        key = f"driver_compare_abs_error_{band_name}_histogram"
        path = figure_dir / f"{key}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        figure_paths[key] = path

    def save_metric_bars(band_name: str) -> None:
        subset = metrics.loc[metrics["bin"] == band_name].set_index("category")
        present = [name for name in categories if name in subset.index]
        if not present:
            return
        fig, axes = plt.subplots(1, 3, figsize=(max(14, 3.2 * len(present)), 4.5))
        x = np.arange(len(present))
        ticks = [
            f"{display_labels[categories.index(name)]}\n"
            f"(n={int(subset.loc[name, 'count']):,})"
            for name in present
        ]
        for axis, metric, ylabel in zip(
            axes, ("rmse", "picp", "clc"), ("RMSE [W]", "PICP", "CLC")
        ):
            axis.bar(
                x,
                subset.loc[present, metric].to_numpy(float),
                color=[colors[categories.index(name)] for name in present],
            )
            axis.set_xticks(x)
            axis.set_xticklabels(ticks, rotation=20, ha="right")
            axis.set_title(metric.upper())
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.25)
            if metric == "picp":
                axis.axhline(coverage_target, color="black", ls="--", lw=1)
        fig.suptitle(f"Reliability and error by driver — {band_name} (all rows)")
        fig.tight_layout()
        key = f"driver_compare_metrics_{band_name}_bars"
        path = figure_dir / f"{key}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        figure_paths[key] = path

    for band_name, _, _ in PERCENT_PRODUCTION_BINS:
        if not any(
            combined(band_name, name, "abs_error").size for name in categories
        ):
            continue
        save_boxplot(band_name, "abs_error", "Absolute error [W]")
        save_boxplot(band_name, "row_nmpil", "Row-wise NMPIL")
        save_histogram(band_name)
        save_metric_bars(band_name)

    metrics_path = out / (metrics_name or DRIVER_METRICS_FILE)
    metrics.to_csv(metrics_path, index=False)
    return {
        "metrics": metrics,
        "metrics_path": metrics_path,
        "figure_paths": figure_paths,
        "reference_peak_w": reference_peak,
        "target_range": target_range,
        "categories": categories,
        "unmatched_rare_rows": unmatched_rare,
    }
