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
    """Return the event groups shown in post-hoc comparison figures."""
    if "event_group" in work.columns:
        return [
            ("normal", work["event_group"] == GROUP_NORMAL),
            ("rare_extreme", work["event_group"] == GROUP_RARE),
        ]

    # Backward-compatible fallback for legacy prediction files that predate
    # regional event labels.
    categories = [("normal", work["anomaly_group"] == GROUP_NORMAL)]
    categories += [
        (
            label.replace("_solar_potential", "").replace("_condition", ""),
            work["anomaly_label"].str.contains(label, na=False),
        )
        for label in SPECIFIC_ANOMALY_LABELS
    ]
    return categories


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

    metrics = metrics[
        metrics["category"].isin((GROUP_NORMAL, "rare_extreme"))
        & (pd.to_numeric(metrics["count"], errors="coerce") > 0)
    ].copy()
    if metrics.empty:
        return {}

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: Dict[str, Path] = {}

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
        ax.set(title=title, ylabel=ylabel)
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
        ax.set(title=title, ylabel=ylabel)
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
                title="Uncertainty components (all daytime rows)",
                ylabel="Mean predictive std [W]",
            )
            ax.legend()
            path = fig_dir / "uncertainty_components_bar.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            figure_paths["uncertainty_components_bar"] = path

    return figure_paths


def build_extreme_event_diagnostic(
    out_dir: str,
    *,
    start: str = "2019-06-28",
    end: str = "2019-06-30",
    regional_threshold: Optional[float] = None,
    chunksize: int = 500_000,
) -> Dict[str, Any]:
    """Analyse every node prediction in one extreme-event interval.

    ``end`` is exclusive. The diagnostic never samples locations or rows.
    Regional labels are preferred over node-level labels.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    out = Path(out_dir)
    predictions_path = out / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"{predictions_path} is required for the event diagnostic."
        )

    available = set(pd.read_csv(predictions_path, nrows=0).columns)

    def choose(*candidates: str, required: bool = False) -> Optional[str]:
        column = next((name for name in candidates if name in available), None)
        if required and column is None:
            raise ValueError(
                f"predictions.csv requires one of {list(candidates)}."
            )
        return column

    columns = {
        "timestamp": choose(
            "timestamp", "target_timestamp", "time", required=True
        ),
        "location": choose("location", "location_id", "node_id"),
        "y_true": choose("y_true", required=True),
        "y_pred": choose("y_pred_mean", "y_pred", required=True),
        "lower_pi": choose("lower_pi", required=True),
        "upper_pi": choose("upper_pi", required=True),
        "y_std": choose("y_pred_std_raw", "y_pred_std"),
        "event_group": choose("event_group", "anomaly_group", required=True),
        "event_score": choose("event_score"),
        "solar": choose(
            "solar_irradiance_poa_target", "solar_irradiance_poa", "ghi_target"
        ),
        "epistemic_std": choose("epistemic_std", "y_pred_epistemic_std"),
        "aleatoric_std": choose("aleatoric_std", "y_pred_aleatoric_std"),
    }
    usecols = list(dict.fromkeys(
        column for column in columns.values() if column is not None
    ))
    text_columns = {
        columns[key]: "string"
        for key in ("timestamp", "location", "event_group")
        if columns[key] is not None
    }
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts <= start_ts:
        raise ValueError("Event diagnostic requires end > start.")
    if regional_threshold is None:
        checkpoint_path = out / "best_model.pt"
        if checkpoint_path.exists():
            try:
                import torch

                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                protocol = checkpoint.get("event_protocol") or {}
                seasonal = protocol.get("seasonal_thresholds") or {}
                season_by_month = {
                    12: "DJF", 1: "DJF", 2: "DJF",
                    3: "MAM", 4: "MAM", 5: "MAM",
                    6: "JJA", 7: "JJA", 8: "JJA",
                    9: "SON", 10: "SON", 11: "SON",
                }
                season = season_by_month[start_ts.month]
                end_season = season_by_month[
                    (end_ts - pd.Timedelta(nanoseconds=1)).month
                ]
                if season == end_season and season in seasonal:
                    regional_threshold = float(seasonal[season])
            except (ImportError, KeyError, OSError, TypeError, ValueError, RuntimeError):
                regional_threshold = None

    parts = []
    reader = pd.read_csv(
        predictions_path,
        usecols=usecols,
        dtype=text_columns,
        chunksize=chunksize,
        low_memory=False,
    )
    numeric_roles = (
        "y_true", "y_pred", "lower_pi", "upper_pi", "y_std", "event_score",
        "solar", "epistemic_std", "aleatoric_std",
    )
    for chunk in reader:
        timestamp = pd.to_datetime(
            chunk[columns["timestamp"]], errors="coerce"
        )
        mask = (timestamp >= start_ts) & (timestamp < end_ts)
        if not mask.any():
            continue
        selected = chunk.loc[mask]
        event = pd.DataFrame({
            "timestamp": timestamp.loc[mask].to_numpy(),
            "event_group": selected[columns["event_group"]].astype(str).to_numpy(),
        })
        if columns["location"] is not None:
            event["location"] = selected[columns["location"]].to_numpy()
        for role in numeric_roles:
            column = columns[role]
            if column is not None:
                event[role] = pd.to_numeric(
                    selected[column], errors="coerce"
                ).to_numpy(float)
        parts.append(event)

    if not parts:
        raise ValueError(
            f"No predictions found in [{start_ts}, {end_ts})."
        )
    event = pd.concat(parts, ignore_index=True)
    core = ["y_true", "y_pred", "lower_pi", "upper_pi"]
    event = event.loc[
        np.isfinite(event[core].to_numpy(float)).all(axis=1)
    ].copy()
    event["abs_error"] = np.abs(event["y_pred"] - event["y_true"])
    event["squared_error"] = (event["y_pred"] - event["y_true"]) ** 2
    event["interval_width"] = event["upper_pi"] - event["lower_pi"]
    event["covered"] = (
        (event["y_true"] >= event["lower_pi"])
        & (event["y_true"] <= event["upper_pi"])
    )
    event["is_rare"] = event["event_group"] == GROUP_RARE

    aggregations = {
        "n_rows": ("y_true", "size"),
        "mean_y_true": ("y_true", "mean"),
        "mean_y_pred": ("y_pred", "mean"),
        "mean_lower_pi": ("lower_pi", "mean"),
        "mean_upper_pi": ("upper_pi", "mean"),
        "mae": ("abs_error", "mean"),
        "mean_squared_error": ("squared_error", "mean"),
        "picp": ("covered", "mean"),
        "mpiw": ("interval_width", "mean"),
        "is_rare": ("is_rare", "max"),
    }
    for column in (
        "y_std", "event_score", "solar", "epistemic_std", "aleatoric_std"
    ):
        if column in event:
            aggregations[f"mean_{column}"] = (column, "mean")
    hourly = (
        event.groupby("timestamp", sort=True)
        .agg(**aggregations)
        .reset_index()
    )
    hourly["rmse"] = np.sqrt(hourly.pop("mean_squared_error"))

    daytime = (
        event["solar"] > DAYTIME_IRRADIANCE_THRESHOLD_WM2
        if "solar" in event else np.ones(len(event), dtype=bool)
    )

    def metric_row(scope: str, mask) -> dict:
        sub = event.loc[mask]
        if sub.empty:
            return {"scope": scope, "count": 0}
        row = {
            "scope": scope,
            "count": int(len(sub)),
            "locations": (
                int(sub["location"].nunique())
                if "location" in sub else float("nan")
            ),
            "timestamps": int(sub["timestamp"].nunique()),
            "mae": float(sub["abs_error"].mean()),
            "rmse": float(np.sqrt(sub["squared_error"].mean())),
            "picp": float(sub["covered"].mean()),
            "mpiw": float(sub["interval_width"].mean()),
            "rare_timestamp_count": int(
                sub.groupby("timestamp")["is_rare"].max().sum()
            ),
        }
        for column in ("y_std", "epistemic_std", "aleatoric_std"):
            if column in sub:
                row[f"mean_{column}"] = float(sub[column].mean())
        return row

    summary = pd.DataFrame([
        metric_row("all_event_hours", np.ones(len(event), dtype=bool)),
        metric_row("daytime", daytime),
        metric_row("regional_rare_hours", event["is_rare"]),
    ])

    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    x = hourly["timestamp"]
    axes[0].plot(x, hourly["mean_y_true"], label="PV target mean", lw=2)
    axes[0].plot(x, hourly["mean_y_pred"], label="PV prediction mean", lw=2)
    axes[0].fill_between(
        x,
        hourly["mean_lower_pi"],
        hourly["mean_upper_pi"],
        alpha=0.2,
        label="mean node-level 95% PI limits",
    )
    axes[0].set_ylabel("PV power [W]")
    axes[0].legend(loc="upper left")

    axes[1].plot(x, hourly["mae"], label="MAE", color="tab:red")
    axes[1].plot(x, hourly["rmse"], label="RMSE", color="tab:purple")
    axes[1].set_ylabel("Error [W]")
    coverage_axis = axes[1].twinx()
    coverage_axis.plot(
        x, hourly["picp"], label="PICP", color="tab:green", alpha=0.8
    )
    coverage_axis.axhline(0.95, color="tab:green", ls="--", lw=1)
    coverage_axis.set_ylabel("PICP")
    axes[1].legend(loc="upper left")
    coverage_axis.legend(loc="upper right")

    if "mean_event_score" in hourly:
        axes[2].plot(
            x, hourly["mean_event_score"], color="tab:orange",
            label="regional anomalous-node fraction",
        )
        if regional_threshold is not None:
            axes[2].axhline(
                regional_threshold,
                color="black",
                ls="--",
                lw=1,
                label=f"seasonal P97.5 threshold = {regional_threshold:.6f}",
            )
        axes[2].set_ylabel("Regional score")
    else:
        axes[2].step(
            x, hourly["is_rare"].astype(float), where="mid",
            label="regional rare label", color="tab:orange",
        )
        axes[2].set_ylabel("Rare label")
    axes[2].legend(loc="upper left")

    if "mean_y_std" in hourly:
        axes[3].plot(x, hourly["mean_y_std"], label="total std", lw=2)
    if "mean_epistemic_std" in hourly:
        axes[3].plot(x, hourly["mean_epistemic_std"], label="epistemic std")
    if "mean_aleatoric_std" in hourly:
        axes[3].plot(x, hourly["mean_aleatoric_std"], label="aleatoric std")
    axes[3].set_ylabel("Predictive std [W]")
    axes[3].legend(loc="upper left")
    axes[3].set_xlabel("Window-end timestamp")

    rare_times = hourly.loc[hourly["is_rare"], "timestamp"]
    half_hour = pd.Timedelta(minutes=30)
    for axis in axes:
        for timestamp in rare_times:
            axis.axvspan(
                timestamp - half_hour,
                timestamp + half_hour,
                color="red",
                alpha=0.045,
                lw=0,
            )
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"Extreme-event response: {start_ts:%Y-%m-%d} to "
        f"{end_ts - pd.Timedelta(days=1):%Y-%m-%d} "
        "(all locations; red shading = regional rare)"
    )
    fig.autofmt_xdate()
    fig.tight_layout()

    stem = (
        f"extreme_event_{start_ts:%Y%m%d}_"
        f"{end_ts - pd.Timedelta(days=1):%Y%m%d}"
    )
    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / f"{stem}.png"
    hourly_path = out / f"{stem}_hourly.csv"
    summary_path = out / f"{stem}_summary.csv"
    fig.savefig(figure_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    hourly.to_csv(hourly_path, index=False)
    summary.to_csv(summary_path, index=False)

    return {
        "summary": summary,
        "hourly": hourly,
        "figure_path": figure_path,
        "hourly_path": hourly_path,
        "summary_path": summary_path,
        "rows_used": int(len(event)),
        "regional_threshold": regional_threshold,
    }


def _production_bin(y, bins=PRODUCTION_BINS):
    for name, lo, hi in bins:
        if y >= lo and (hi is None or y < hi):
            return name
    return "unknown"
