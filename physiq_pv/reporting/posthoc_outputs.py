"""Post-hoc CSV summaries, figures and W&B logging."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

PRODUCTION_BINS = [
    ("daytime_0_20", 0.0, 20.0),
    ("daytime_20_40", 20.0, 40.0),
    ("daytime_40_60", 40.0, 60.0),
    ("daytime_60_80", 60.0, 80.0),
    ("daytime_80_100", 80.0, 100.0),
    ("daytime_gt_100", 100.0, None),
]

# Raw-watt bins are legacy; figures use report percentage bins.
PERCENT_PRODUCTION_BINS = [
    ("daytime_0_20_pct", 0.0, 20.0),
    ("daytime_20_40_pct", 20.0, 40.0),
    ("daytime_40_60_pct", 40.0, 60.0),
    ("daytime_60_80_pct", 60.0, 80.0),
    ("daytime_80_100_pct", 80.0, None),
]
REFERENCE_PRODUCTION_PEAKS_FILE = "reference_production_peaks.csv"

POSTHOC_KEYS = (
    "posthoc/daytime_picp",
    "posthoc/daytime_mpiw",
    "posthoc/daytime_nmpil",
    "posthoc/normal_picp",
    "posthoc/rare_extreme_picp",
    "posthoc/unusually_low_picp",
    "posthoc/high_production_picp",
    "posthoc/gt100_picp",
)
WANDB_RUN_METADATA_FILE = "wandb_run.json"
FIGURE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

DAYTIME_IRRADIANCE_THRESHOLD_WM2 = 10.0
GROUP_NORMAL = "normal"
GROUP_RARE = "rare_or_extreme"
SPECIFIC_ANOMALY_LABELS = (
    "unusually_low_solar_potential",
    "unusually_high_solar_potential",
    "extreme_temperature_condition",
    "extreme_wind_condition",
)


def _figure_category_masks(work):
    """Return pointwise MTGFlow groups shown in post-hoc figures."""
    if "anomaly_group" not in work.columns:
        raise ValueError(
            "Post-hoc normal/rare figures require anomaly_group matched on "
            "(location, timestamp); event_group is only a regional label."
        )
    return [
        ("normal", work["anomaly_group"] == GROUP_NORMAL),
        ("rare_extreme", work["anomaly_group"] == GROUP_RARE),
    ]


def _scope_value(df, scope_col: str, scope: str, value_col: str) -> float:
    import math

    hit = df[df[scope_col] == scope]
    if len(hit) == 0 or value_col not in hit.columns:
        return math.nan
    try:
        return float(hit.iloc[0][value_col])
    except (TypeError, ValueError):
        return math.nan


def read_posthoc_summary(out_dir: str) -> Dict[str, float]:
    """Read analysis CSVs and return W&B `posthoc/*` scalars."""
    import pandas as pd

    out = Path(out_dir)
    summary: Dict[str, float] = {k: float("nan") for k in POSTHOC_KEYS}

    sharp_path = out / "sharpness_overview.csv"
    if sharp_path.exists():
        s = pd.read_csv(sharp_path)
        summary["posthoc/daytime_picp"] = _scope_value(
            s, "scope", "overall_daytime", "picp"
        )
        summary["posthoc/daytime_mpiw"] = _scope_value(
            s, "scope", "overall_daytime", "mpiw"
        )
        summary["posthoc/daytime_nmpil"] = _scope_value(
            s, "scope", "overall_daytime", "nmpil"
        )
        summary["posthoc/normal_picp"] = _scope_value(s, "scope", "normal", "picp")
        summary["posthoc/rare_extreme_picp"] = _scope_value(
            s, "scope", "rare_extreme", "picp"
        )
        summary["posthoc/unusually_low_picp"] = _scope_value(
            s, "scope", "unusually_low_solar_potential", "picp"
        )

    bins_path = out / "daytime_bin_summary.csv"
    if bins_path.exists():
        b = pd.read_csv(bins_path)
        high_picp = _scope_value(
            b, "bin", "daytime_80_100_pct", "picp"
        )
        if high_picp != high_picp:
            high_picp = _scope_value(
                b, "bin", "daytime_gt_100", "picp"
            )
        summary["posthoc/high_production_picp"] = high_picp
        summary["posthoc/gt100_picp"] = high_picp

    return summary


def load_wandb_run_metadata(out_dir: str) -> Dict[str, Any]:
    path = Path(out_dir) / WANDB_RUN_METADATA_FILE
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def init_wandb_run_for_out_dir(
    wandb,
    out_dir: str,
    *,
    run_name: Optional[str] = None,
    project: str,
    entity: Optional[str],
    reinit: bool = True,
):
    """Resume the training W&B run for `out_dir`, or create a named fallback."""
    meta = load_wandb_run_metadata(out_dir)
    kwargs: Dict[str, Any] = {
        "project": meta.get("project") or project,
        "reinit": reinit,
    }
    resolved_entity = meta.get("entity") or entity
    if resolved_entity:
        kwargs["entity"] = resolved_entity
    if meta.get("id"):
        kwargs["id"] = str(meta["id"])
        kwargs["resume"] = "allow"
    else:
        kwargs["name"] = run_name
    return wandb.init(**kwargs)


def _dedupe(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for path in paths:
        if not path.exists():
            continue
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def collect_figure_paths(
    out_dir: str,
    figure_paths: Optional[Mapping[str, Any] | Iterable[Any]] = None,
) -> Dict[str, Path]:
    labelled: Dict[str, Path] = {}
    if figure_paths is not None:
        items = (
            figure_paths.items()
            if isinstance(figure_paths, Mapping)
            else ((Path(p).stem, p) for p in figure_paths)
        )
        for label, raw_path in items:
            path = Path(raw_path)
            if path.exists():
                labelled[str(label)] = path

    fig_dir = Path(out_dir) / "figures"
    if fig_dir.exists():
        for path in sorted(fig_dir.iterdir()):
            if path.suffix.lower() in FIGURE_SUFFIXES:
                labelled.setdefault(path.stem, path)
    return labelled


def collect_run_artifact_files(
    out_dir: str,
    *,
    figure_paths: Optional[Mapping[str, Any] | Iterable[Any]] = None,
    include_predictions: bool = False,
) -> List[Path]:
    out = Path(out_dir)
    files = [
        p
        for pattern in ("*.md", "*.json", "*.csv")
        for p in sorted(out.glob(pattern))
    ]
    if not include_predictions:
        files = [p for p in files if p.name != "predictions.csv"]
    files.extend(collect_figure_paths(out_dir, figure_paths).values())
    return _dedupe(files)


def _artifact_member_name(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return path.name


def _wandb_label(label: str) -> str:
    return str(label).replace("\\", "/").strip("/").replace("/", "_")


def log_posthoc_to_wandb(
    wandb,
    run,
    out_dir: str,
    *,
    figure_paths: Optional[Mapping[str, Any] | Iterable[Any]] = None,
    upload_artifact: bool = True,
    log_figures: bool = True,
    include_predictions: bool = False,
    artifact_name: Optional[str] = None,
    artifact_type: str = "posthoc-report",
) -> Dict[str, Any]:
    summary = read_posthoc_summary(out_dir)
    run.log(summary)
    run.summary.update(summary)

    figures = collect_figure_paths(out_dir, figure_paths) if log_figures else {}
    for label, path in figures.items():
        run.log({f"figures/{_wandb_label(label)}": wandb.Image(str(path))})

    artifact_files: List[Path] = []
    if upload_artifact:
        artifact_files = collect_run_artifact_files(
            out_dir,
            figure_paths=figure_paths,
            include_predictions=include_predictions,
        )
        if artifact_files:
            run_id = getattr(run, "id", None) or "manual"
            artifact = wandb.Artifact(
                artifact_name or f"pvgis-sde-posthoc-{run_id}",
                type=artifact_type,
            )
            root = Path(out_dir).resolve()
            for path in artifact_files:
                artifact.add_file(str(path), name=_artifact_member_name(path, root))
            run.log_artifact(artifact)

    return {
        "summary": summary,
        "artifact_uploaded": bool(artifact_files),
        "artifact_files": [str(path) for path in artifact_files],
        "figures_logged": list(figures),
    }


def load_prediction_sample(
    predictions_path,
    max_rows: int = 500_000,
    random_state: int = 1,
):
    import pandas as pd

    path = Path(predictions_path)
    with open(path, "r", encoding="utf-8") as fh:
        total = sum(1 for _ in fh) - 1
    if total <= 0:
        return pd.read_csv(path)
    if total <= max_rows:
        return pd.read_csv(path)

    frac = max_rows / total
    parts = [
        chunk.sample(frac=frac, random_state=random_state)
        for chunk in pd.read_csv(path, chunksize=200_000)
    ]
    out = pd.concat(parts, ignore_index=True)
    return out.sample(n=min(max_rows, len(out)), random_state=random_state).reset_index(
        drop=True
    )


def attach_reference_peak_production_pct(day, reference_peaks):
    """Attach the report-derived global PVGIS reference peak to sampled rows."""
    import numpy as np

    required_peaks = {"reference_peak_w"}
    missing_peaks = required_peaks - set(reference_peaks.columns)
    if missing_peaks:
        raise ValueError(
            "Reference peak table is missing columns "
            f"{sorted(missing_peaks)}."
        )

    work = day.copy()
    peak = float(reference_peaks["reference_peak_w"].iloc[0])
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError(
            f"Reference peak is invalid: {peak!r}."
        )
    work["reference_peak_w"] = peak
    work["production_pct"] = np.clip(
        100.0 * work["y_true"].to_numpy(float) / peak,
        0.0,
        100.0,
    )
    return work


def build_posthoc_figures(
    out_dir: str,
    *,
    max_plot_rows: Optional[int] = None,
    random_state: int = 1,
    coverage_target: float = 0.95,
    clc_eta: float = 9.0,
    horizon_hours: Optional[int] = None,
) -> Dict[str, Path]:
    """Build exact full-data figures from the chunked post-hoc summaries.

    ``max_plot_rows`` and ``random_state`` remain accepted for notebook/API
    compatibility, but sampling is deliberately disabled. The preceding
    daytime analysis scans every prediction row and stores exact pooled metrics
    and exact Tukey boxplot statistics in CSV form.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    out = Path(out_dir)
    metrics_path = out / "daytime_bin_anomaly_metrics.csv"
    if not metrics_path.exists():
        print(
            "[figures] daytime_bin_anomaly_metrics.csv is missing; run the "
            "daytime report before generating full-data figures."
        )
        return {}
    metrics = pd.read_csv(metrics_path)
    required = {
        "bin", "category", "count", "picp", "rmse", "nmpil",
        "abs_error_mean", "abs_error_q1", "abs_error_median", "abs_error_q3",
        "abs_error_whisker_low", "abs_error_whisker_high",
        "row_nmpil_mean", "row_nmpil_q1", "row_nmpil_median",
        "row_nmpil_q3", "row_nmpil_whisker_low",
        "row_nmpil_whisker_high",
    }
    missing = required - set(metrics.columns)
    if missing:
        print(
            "[figures] full-data boxplot statistics are missing "
            f"({sorted(missing)}); rerun the daytime analysis with the current code."
        )
        return {}

    # Pointwise detector groups, matched on (location, timestamp), are the
    # primary scientific comparison. Regional event labels remain optional.
    metrics = metrics[
        metrics["category"].isin((GROUP_NORMAL, "rare_extreme"))
        & (pd.to_numeric(metrics["count"], errors="coerce") > 0)
    ].copy()
    if metrics.empty:
        return {}

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: Dict[str, Path] = {}
    if horizon_hours is not None and int(horizon_hours) < 1:
        raise ValueError("horizon_hours must be a positive integer.")
    horizon_title = (
        "" if horizon_hours is None else f" — forecast t+{int(horizon_hours)}h"
    )

    def exact_boxplot(label, rows, prefix, title, ylabel) -> None:
        if rows.empty:
            return
        boxes = []
        ticklabels = []
        for _, row in rows.iterrows():
            boxes.append({
                "label": "",
                "mean": float(row[f"{prefix}_mean"]),
                "med": float(row[f"{prefix}_median"]),
                "q1": float(row[f"{prefix}_q1"]),
                "q3": float(row[f"{prefix}_q3"]),
                "whislo": float(row[f"{prefix}_whisker_low"]),
                "whishi": float(row[f"{prefix}_whisker_high"]),
                "fliers": [],
            })
            ticklabels.append(
                f"{str(row['category']).replace('_', ' ')}\n"
                f"(n={int(row['count']):,})"
            )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bxp(
            boxes,
            showfliers=False,
            showmeans=True,
            meanprops={
                "marker": "D", "markerfacecolor": "red",
                "markeredgecolor": "red", "markersize": 5,
            },
        )
        ax.set_xticks(range(1, len(ticklabels) + 1))
        ax.set_xticklabels(ticklabels, rotation=30, ha="right")
        ax.set(title=f"{title}{horizon_title}", ylabel=ylabel)
        path = fig_dir / f"{label}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        figure_paths[label] = path

    def barchart(label, rows, value_col, title, ylabel, hline=None) -> None:
        if rows.empty:
            return
        values = pd.to_numeric(rows[value_col], errors="coerce").to_numpy(float)
        ticklabels = [
            f"{str(row['category']).replace('_', ' ')}\n"
            f"(n={int(row['count']):,})"
            for _, row in rows.iterrows()
        ]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = range(len(values))
        ax.bar(x, values, color="steelblue")
        ax.set_xticks(list(x))
        ax.set_xticklabels(ticklabels, rotation=30, ha="right")
        if hline is not None:
            ax.axhline(hline, color="r", ls="--", lw=1, label=f"target {hline:g}")
            ax.legend()
        ax.set(title=f"{title}{horizon_title}", ylabel=ylabel)
        path = fig_dir / f"{label}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        figure_paths[label] = path

    metrics["clc"] = metrics["nmpil"] * (
        1.0 + np.exp(-clc_eta * (metrics["picp"] - coverage_target))
    )
    bins = list(dict.fromkeys(metrics["bin"].astype(str)))

    for key, ylabel in (("mae", "Absolute error [W]"), ("nmpil", "NMPIL")):
        for b in bins:
            rows = metrics.loc[metrics["bin"].astype(str) == b]
            prefix = "abs_error" if key == "mae" else "row_nmpil"
            exact_boxplot(
                f"{key}_{b}_boxplot",
                rows,
                prefix,
                f"{key.upper()} — {b} (all daytime rows)",
                ylabel,
            )

    for key, ylabel, hline in (
        ("picp", "PICP", coverage_target),
        ("clc", "CLC", None),
        ("rmse", "RMSE [W]", None),
    ):
        for b in bins:
            barchart(
                f"{key}_{b}_bar",
                metrics.loc[metrics["bin"].astype(str) == b],
                key,
                f"{key.upper()} — {b} (all daytime rows)",
                ylabel,
                hline=hline,
            )

    components_path = out / "uncertainty_components.csv"
    if components_path.exists():
        components = pd.read_csv(components_path)
        components = components[
            components["scope"].isin((GROUP_NORMAL, "rare_extreme"))
            & (pd.to_numeric(components["count"], errors="coerce") > 0)
        ].copy()
        if not components.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            x = np.arange(len(components))
            width = 0.36
            ax.bar(
                x - width / 2,
                components["mean_epistemic_std"],
                width,
                label="epistemic",
            )
            ax.bar(
                x + width / 2,
                components["mean_aleatoric_std"],
                width,
                label="aleatoric",
            )
            ax.set_xticks(x)
            ax.set_xticklabels([
                f"{scope.replace('_', ' ')}\n(n={int(count):,})"
                for scope, count in zip(components["scope"], components["count"])
            ])
            ax.set(
                title=f"Uncertainty components (all daytime rows){horizon_title}",
                ylabel="Mean predictive std [W]",
            )
            ax.legend()
            path = fig_dir / "uncertainty_components_bar.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            figure_paths["uncertainty_components_bar"] = path

    return figure_paths


def build_horizon_comparison_figures(
    run_dirs: Mapping[int, str | Path],
    out_dir: str | Path,
    *,
    location: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    comparison_days: int = 7,
    chunksize: int = 500_000,
    coverage_target: float = 0.95,
) -> Dict[str, Path]:
    """Compare paper-identical SDE-Net runs at multiple forecast horizons.

    Every input directory must contain the normal per-run post-hoc summary and
    ``predictions.csv``.  Metrics use the exact full-data summaries; the time
    series uses one common location and target-time interval so t+1h, t+6h and
    t+12h remain directly comparable.  Prediction timestamps are target times,
    not forecast-origin times.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if not run_dirs:
        raise ValueError("At least one horizon run is required.")
    horizons = sorted(int(value) for value in run_dirs)
    if any(value < 1 for value in horizons) or len(horizons) != len(set(horizons)):
        raise ValueError("Forecast horizons must be unique positive integers.")
    if comparison_days < 1:
        raise ValueError("comparison_days must be >= 1.")

    resolved = {int(h): Path(path) for h, path in run_dirs.items()}
    output = Path(out_dir)
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    prediction_paths: Dict[int, Path] = {}
    for horizon in horizons:
        run_dir = resolved[horizon]
        sharpness_path = run_dir / "sharpness_overview.csv"
        predictions_path = run_dir / "predictions.csv"
        if not sharpness_path.is_file():
            raise FileNotFoundError(
                f"Missing post-hoc summary for t+{horizon}h: {sharpness_path}"
            )
        if not predictions_path.is_file():
            raise FileNotFoundError(
                f"Missing predictions for t+{horizon}h: {predictions_path}"
            )
        sharpness = pd.read_csv(sharpness_path)
        if "scope" not in sharpness:
            raise ValueError(f"{sharpness_path} is missing the scope column.")
        overall = sharpness.loc[sharpness["scope"] == "overall_daytime"]
        if len(overall) != 1:
            raise ValueError(
                f"{sharpness_path} must contain one overall_daytime row."
            )
        row = overall.iloc[0]
        metric_rows.append(
            {
                "horizon_hours": horizon,
                **{
                    name: float(row[name]) if name in row else float("nan")
                    for name in ("count", "mae", "rmse", "picp", "mean_std", "mpiw", "nmpil")
                },
            }
        )
        prediction_paths[horizon] = predictions_path

    metrics = pd.DataFrame(metric_rows).sort_values("horizon_hours")
    metrics_path = output / "horizon_comparison_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    figure_paths: Dict[str, Path] = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(metrics["horizon_hours"], metrics["mae"], "o-", label="MAE")
    axes[0].plot(metrics["horizon_hours"], metrics["rmse"], "o-", label="RMSE")
    axes[0].set(
        xlabel="Forecast horizon [h]",
        ylabel="Error [W]",
        title="Daytime point-forecast error",
        xticks=horizons,
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(metrics["horizon_hours"], metrics["picp"], "o-", label="PICP")
    axes[1].axhline(
        coverage_target, color="tab:red", linestyle="--", label=f"target {coverage_target:g}"
    )
    axes[1].set(
        xlabel="Forecast horizon [h]",
        ylabel="Coverage",
        title="Daytime predictive-interval coverage",
        xticks=horizons,
        ylim=(0.0, 1.05),
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    metric_figure = figure_dir / "horizon_metrics_comparison.png"
    fig.savefig(metric_figure, dpi=140, bbox_inches="tight")
    plt.close(fig)
    figure_paths["horizon_metrics_comparison"] = metric_figure

    reference_path = prediction_paths[horizons[0]]
    reference_header = set(pd.read_csv(reference_path, nrows=0).columns)
    required = {"location", "timestamp", "y_true"}
    missing = required - reference_header
    if missing:
        raise ValueError(f"{reference_path} is missing columns {sorted(missing)}.")
    if location is None:
        first = pd.read_csv(reference_path, usecols=["location"], nrows=1)
        if first.empty:
            raise ValueError(f"No predictions in {reference_path}.")
        selected_location = str(first.iloc[0]["location"])
    else:
        selected_location = str(location)

    def selected_rows(path: Path, *, begin=None, finish=None, columns=None):
        parts = []
        usecols = list(columns or ["location", "timestamp", "y_true"])
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
            keep = chunk["location"].astype(str) == selected_location
            if not bool(keep.any()):
                continue
            selected = chunk.loc[keep].copy()
            selected["timestamp"] = pd.to_datetime(selected["timestamp"], errors="raise")
            if begin is not None:
                selected = selected.loc[selected["timestamp"] >= begin]
            if finish is not None:
                selected = selected.loc[selected["timestamp"] < finish]
            if not selected.empty:
                parts.append(selected)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)

    if start is None and end is not None:
        raise ValueError("start is required when end is supplied.")
    if start is None:
        reference_location = selected_rows(reference_path)
        if reference_location.empty:
            raise ValueError(
                f"Location {selected_location!r} is absent from {reference_path}."
            )
        target = pd.to_numeric(reference_location["y_true"], errors="coerce")
        if not np.isfinite(target.to_numpy(float)).any():
            raise ValueError("Reference location has no finite y_true values.")
        peak_time = reference_location.loc[target.idxmax(), "timestamp"]
        begin = pd.Timestamp(peak_time).floor("D") - pd.Timedelta(
            days=comparison_days // 2
        )
        finish = begin + pd.Timedelta(days=comparison_days)
    else:
        begin = pd.Timestamp(start)
        finish = (
            pd.Timestamp(end)
            if end is not None
            else begin + pd.Timedelta(days=comparison_days)
        )
    if finish <= begin:
        raise ValueError("Comparison end must be after start.")

    series_by_horizon = {}
    for horizon in horizons:
        path = prediction_paths[horizon]
        header = set(pd.read_csv(path, nrows=0).columns)
        pred_col = "y_pred_mean" if "y_pred_mean" in header else "y_pred"
        if pred_col not in header:
            raise ValueError(f"{path} is missing y_pred_mean/y_pred.")
        lower_col = "lower_pi" if "lower_pi" in header else None
        upper_col = "upper_pi" if "upper_pi" in header else None
        columns = ["location", "timestamp", "y_true", pred_col]
        if lower_col and upper_col:
            columns.extend([lower_col, upper_col])
        series = selected_rows(
            path, begin=begin, finish=finish, columns=columns
        ).rename(columns={pred_col: "y_pred"})
        if series.empty:
            raise ValueError(
                f"No t+{horizon}h rows for location {selected_location!r} "
                f"between {begin} and {finish}."
            )
        if series["timestamp"].duplicated().any():
            raise ValueError(
                f"Duplicate target timestamps for t+{horizon}h/location {selected_location}."
            )
        series_by_horizon[horizon] = series.sort_values("timestamp")

    fig, axes = plt.subplots(
        len(horizons), 1, figsize=(14, 3.4 * len(horizons)), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    for axis, horizon in zip(axes, horizons):
        series = series_by_horizon[horizon]
        axis.plot(series["timestamp"], series["y_true"], color="black", lw=1.8, label="Actual")
        axis.plot(series["timestamp"], series["y_pred"], color="tab:blue", lw=1.4, label="Prediction")
        if {"lower_pi", "upper_pi"} <= set(series.columns):
            axis.fill_between(
                series["timestamp"], series["lower_pi"], series["upper_pi"],
                color="tab:blue", alpha=0.16, label="Predictive interval",
            )
        axis.set(title=f"t+{horizon}h", ylabel="PV power [W]")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Target timestamp")
    fig.suptitle(
        f"SDE-Net forecasts — location {selected_location} — "
        f"{begin:%Y-%m-%d} to {finish:%Y-%m-%d}",
        y=1.01,
    )
    fig.tight_layout()
    prediction_figure = figure_dir / "horizon_prediction_timeseries.png"
    fig.savefig(prediction_figure, dpi=140, bbox_inches="tight")
    plt.close(fig)
    figure_paths["horizon_prediction_timeseries"] = prediction_figure

    metadata_path = output / "horizon_comparison_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "horizons_hours": horizons,
                "run_dirs": {str(key): str(resolved[key].resolve()) for key in horizons},
                "location": selected_location,
                "start": begin.isoformat(),
                "end_exclusive": finish.isoformat(),
                "timestamps_are_target_times": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return figure_paths


def build_direct_multihorizon_posthoc(
    out_dir: str | Path,
    *,
    location: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    comparison_days: int = 7,
    detail_horizons: Optional[Iterable[int]] = None,
) -> Dict[str, Path]:
    """Post-hoc analysis for one direct multi-output t+1...t+H run.

    Normal/rare curves use the pointwise ``anomaly_group`` joined on each
    output's ``(location, target timestamp)``.  ``event_group`` is deliberately
    ignored.  In addition to the all-horizon summary and forecast channels,
    detailed absolute-error boxplots and histograms are generated for the
    requested horizons.  By default these are the first and last direct
    outputs (t+1 and t+6 for the production notebook).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    output = Path(out_dir)
    predictions_path = output / "predictions.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"{predictions_path} is required.")
    predictions = pd.read_csv(predictions_path, parse_dates=["timestamp"])
    required = {
        "location", "timestamp", "horizon_hours", "y_true", "abs_error",
        "squared_error", "anomaly_group",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(
            f"Direct multi-horizon post-hoc requires columns {sorted(required)}; "
            f"missing {sorted(missing)}."
        )
    if predictions["anomaly_group"].isna().any():
        raise ValueError("anomaly_group contains missing pointwise labels.")
    detector_names = (
        predictions.loc[
            predictions["anomaly_group"] == GROUP_RARE, "anomaly_label"
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda values: values.ne("")]
        .drop_duplicates()
        .tolist()
        if "anomaly_label" in predictions
        else []
    )
    detector_name = detector_names[0] if len(detector_names) == 1 else "detector"
    horizons = sorted(
        pd.to_numeric(predictions["horizon_hours"], errors="raise")
        .astype(int).unique().tolist()
    )
    if len(horizons) < 2:
        raise ValueError("A direct multi-horizon run must contain at least two horizons.")
    if detail_horizons is None:
        selected_detail_horizons = [horizons[0], horizons[-1]]
    else:
        selected_detail_horizons = list(
            dict.fromkeys(int(value) for value in detail_horizons)
        )
        if not selected_detail_horizons:
            raise ValueError("detail_horizons must contain at least one horizon.")
        unknown = sorted(set(selected_detail_horizons) - set(horizons))
        if unknown:
            raise ValueError(
                f"Detailed horizons {unknown} are absent; available horizons: {horizons}."
            )

    work = predictions.copy()
    if "solar_irradiance_poa_target" in work:
        work = work.loc[
            pd.to_numeric(
                work["solar_irradiance_poa_target"], errors="coerce"
            ) >= DAYTIME_IRRADIANCE_THRESHOLD_WM2
        ].copy()
    rows = []
    for horizon in horizons:
        horizon_rows = work.loc[work["horizon_hours"] == horizon]
        for group in ("all", GROUP_NORMAL, GROUP_RARE):
            selected = (
                horizon_rows if group == "all" else
                horizon_rows.loc[horizon_rows["anomaly_group"] == group]
            )
            rows.append({
                "horizon_hours": horizon,
                "anomaly_group": group,
                "count": int(len(selected)),
                "mae": float(selected["abs_error"].mean()) if len(selected) else np.nan,
                "rmse": (
                    float(np.sqrt(selected["squared_error"].mean()))
                    if len(selected) else np.nan
                ),
            })
    metrics = pd.DataFrame(rows)
    metrics_path = output / "multihorizon_anomaly_group_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    figure_dir = output / "figures" / "direct_multihorizon"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    labels = ((GROUP_NORMAL, "normal", "tab:blue"),
              (GROUP_RARE, "rare/anomalous", "tab:red"))
    for group, label, color in labels:
        group_metrics = metrics.loc[metrics["anomaly_group"] == group]
        axes[0].plot(
            group_metrics["horizon_hours"], group_metrics["mae"],
            "o-", color=color, label=label,
        )
        axes[1].plot(
            group_metrics["horizon_hours"], group_metrics["rmse"],
            "o-", color=color, label=label,
        )
    for axis, metric in zip(axes, ("MAE", "RMSE")):
        axis.set(
            xlabel="Direct forecast horizon [h]",
            ylabel=f"{metric} [W]",
            title=f"{metric} by pointwise anomaly_group",
            xticks=horizons,
        )
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    error_figure = figure_dir / "mae_rmse_by_horizon_anomaly_group.png"
    fig.savefig(error_figure, dpi=140, bbox_inches="tight")
    plt.close(fig)

    def _finite_absolute_errors(frame):
        values = pd.to_numeric(frame["abs_error"], errors="coerce").to_numpy(float)
        return values[np.isfinite(values)]

    def _tukey_stats(values, label):
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        inside = values[
            (values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)
        ]
        if inside.size == 0:
            inside = values
        return {
            "label": label,
            "mean": float(np.mean(values)),
            "med": float(median),
            "q1": float(q1),
            "q3": float(q3),
            "whislo": float(np.min(inside)),
            "whishi": float(np.max(inside)),
            "fliers": [],
        }

    detail_labels = (
        (GROUP_NORMAL, "normal", "tab:blue"),
        (GROUP_RARE, "rare/anomalous", "tab:red"),
    )
    fig, axes = plt.subplots(
        1, len(selected_detail_horizons),
        figsize=(6.0 * len(selected_detail_horizons), 4.8),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for axis, horizon in zip(axes, selected_detail_horizons):
        boxes = []
        colors = []
        horizon_rows = work.loc[work["horizon_hours"] == horizon]
        for group, label, color in detail_labels:
            values = _finite_absolute_errors(
                horizon_rows.loc[horizon_rows["anomaly_group"] == group]
            )
            if values.size:
                boxes.append(_tukey_stats(values, f"{label}\n(n={values.size:,})"))
                colors.append(color)
        if boxes:
            artists = axis.bxp(
                boxes,
                showfliers=False,
                showmeans=True,
                patch_artist=True,
                meanprops={
                    "marker": "D", "markerfacecolor": "black",
                    "markeredgecolor": "black", "markersize": 4,
                },
            )
            for patch, color in zip(artists["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.35)
        else:
            axis.text(0.5, 0.5, "No daytime rows", ha="center", va="center")
        axis.set(title=f"Absolute error — t+{horizon}h", ylabel="Absolute error [W]")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    boxplot_figure = figure_dir / "absolute_error_boxplots_t1_t6.png"
    fig.savefig(boxplot_figure, dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        1, len(selected_detail_horizons),
        figsize=(6.0 * len(selected_detail_horizons), 4.8),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for axis, horizon in zip(axes, selected_detail_horizons):
        horizon_rows = work.loc[work["horizon_hours"] == horizon]
        distributions = []
        for group, label, color in detail_labels:
            values = _finite_absolute_errors(
                horizon_rows.loc[horizon_rows["anomaly_group"] == group]
            )
            if values.size:
                distributions.append((values, label, color))
        if distributions:
            pooled = np.concatenate([values for values, _, _ in distributions])
            upper = float(np.quantile(pooled, 0.995))
            if not np.isfinite(upper) or upper <= 0.0:
                upper = max(float(np.max(pooled)), 1.0)
            bins = np.linspace(0.0, upper, 51)
            for values, label, color in distributions:
                # The final bin contains the upper 0.5% tail; no rows are dropped.
                plotted = np.minimum(values, upper)
                axis.hist(
                    plotted,
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=1.8,
                    color=color,
                    label=f"{label} (n={values.size:,})",
                )
            axis.set_xlabel("Absolute error [W]; final bin contains >= P99.5")
            axis.legend()
        else:
            axis.text(0.5, 0.5, "No daytime rows", ha="center", va="center")
        axis.set(title=f"Absolute-error distribution — t+{horizon}h", ylabel="Density")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    histogram_figure = figure_dir / "absolute_error_histograms_t1_t6.png"
    fig.savefig(histogram_figure, dpi=140, bbox_inches="tight")
    plt.close(fig)

    selected_location = (
        str(location) if location is not None else str(predictions.iloc[0]["location"])
    )
    series = predictions.loc[
        predictions["location"].astype(str) == selected_location
    ].copy()
    if series.empty:
        raise ValueError(f"Location {selected_location!r} is absent from predictions.")
    if start is None and end is not None:
        raise ValueError("start is required when end is supplied.")
    if start is None:
        peak_index = pd.to_numeric(series["y_true"], errors="coerce").idxmax()
        begin = series.loc[peak_index, "timestamp"].floor("D") - pd.Timedelta(
            days=comparison_days // 2
        )
        finish = begin + pd.Timedelta(days=comparison_days)
    else:
        begin = pd.Timestamp(start)
        finish = (
            pd.Timestamp(end) if end is not None
            else begin + pd.Timedelta(days=comparison_days)
        )
    if finish <= begin:
        raise ValueError("Comparison end must be after start.")
    series = series.loc[
        (series["timestamp"] >= begin) & (series["timestamp"] < finish)
    ]
    pred_col = "y_pred_mean" if "y_pred_mean" in series else "y_pred"
    fig, axes = plt.subplots(
        len(horizons), 1, figsize=(14, 3.0 * len(horizons)),
        sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes)
    for axis, horizon in zip(axes, horizons):
        channel = series.loc[series["horizon_hours"] == horizon].sort_values(
            "timestamp"
        )
        if channel.empty:
            raise ValueError(
                f"No t+{horizon} rows for {selected_location!r} in the selected interval."
            )
        if channel["timestamp"].duplicated().any():
            raise ValueError(
                f"Duplicate target timestamps for t+{horizon}/{selected_location}."
            )
        axis.plot(channel["timestamp"], channel["y_true"], color="black", label="Actual")
        axis.plot(channel["timestamp"], channel[pred_col], color="tab:blue", label="Prediction")
        if {"lower_pi", "upper_pi"} <= set(channel.columns):
            axis.fill_between(
                channel["timestamp"], channel["lower_pi"], channel["upper_pi"],
                color="tab:blue", alpha=0.16, label="Predictive interval",
            )
        rare = channel["anomaly_group"] == GROUP_RARE
        if rare.any():
            axis.scatter(
                channel.loc[rare, "timestamp"], channel.loc[rare, "y_true"],
                color="tab:red", s=18, zorder=3,
                label=f"{detector_name} rare/anomalous",
            )
        axis.set(title=f"Direct output t+{horizon}h", ylabel="PV power [W]")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", ncols=2)
    axes[-1].set_xlabel("Target timestamp")
    fig.suptitle(
        f"Direct t+1...t+{max(horizons)} forecasts — location {selected_location}",
        y=1.002,
    )
    fig.tight_layout()
    prediction_figure = figure_dir / "prediction_channels_t1_t6.png"
    fig.savefig(prediction_figure, dpi=140, bbox_inches="tight")
    plt.close(fig)

    metadata_path = output / "multihorizon_posthoc_metadata.json"
    metadata_path.write_text(
        json.dumps({
            "forecast_mode": "direct_multi_output",
            "horizons_hours": horizons,
            "detail_horizons_hours": selected_detail_horizons,
            "label_column": "anomaly_group",
            "label_join_key": ["location", "timestamp"],
            "detector": detector_name,
            "event_group_used": False,
            "timestamps_are_target_times": True,
            "location": selected_location,
            "start": begin.isoformat(),
            "end_exclusive": finish.isoformat(),
        }, indent=2),
        encoding="utf-8",
    )
    return {
        "metrics": metrics_path,
        "error_figure": error_figure,
        "boxplot_figure": boxplot_figure,
        "histogram_figure": histogram_figure,
        "prediction_figure": prediction_figure,
        "metadata": metadata_path,
    }


def _production_bin(y, bins=PRODUCTION_BINS):
    for name, lo, hi in bins:
        if y >= lo and (hi is None or y < hi):
            return name
    return "unknown"
