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

POSTHOC_KEYS = (
    "posthoc/daytime_picp",
    "posthoc/daytime_mpiw",
    "posthoc/daytime_nmpil",
    "posthoc/normal_picp",
    "posthoc/rare_extreme_picp",
    "posthoc/unusually_low_picp",
    "posthoc/gt100_picp",
)
WANDB_RUN_METADATA_FILE = "wandb_run.json"
FIGURE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


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
        summary["posthoc/gt100_picp"] = _scope_value(
            b, "bin", "daytime_gt_100", "picp"
        )

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


def build_posthoc_figures(
    out_dir: str,
    *,
    max_plot_rows: int = 500_000,
    random_state: int = 1,
) -> Dict[str, Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    out = Path(out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: Dict[str, Path] = {}

    def save(fig, label: str) -> None:
        path = fig_dir / f"{label}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        figure_paths[label] = path

    def _box(ax, data, ticklabels) -> None:
        """Boxplot + tick labels, compatible across matplotlib versions
        (the boxplot `labels`/`tick_labels` kwarg was renamed)."""
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(ticklabels) + 1))
        ax.set_xticklabels(ticklabels)

    pred_path = out / "predictions.csv"
    if pred_path.exists():
        pred = load_prediction_sample(
            pred_path, max_rows=max_plot_rows, random_state=random_state
        )
        ycol = "y_pred_mean" if "y_pred_mean" in pred.columns else "y_pred"
        pred["residual"] = pred["y_true"] - pred[ycol]
        pred["abs_error"] = pred["residual"].abs()
        pred["interval_width"] = pred["upper_pi"] - pred["lower_pi"]
        pred["prod_bin"] = pred["y_true"].apply(_production_bin)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(pred["residual"].dropna(), bins=100)
        ax.set(
            title="Residual (y_true - y_pred_mean)",
            xlabel="residual [W]",
            ylabel="count",
        )
        save(fig, "residual_histogram")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(pred["interval_width"].dropna(), bins=100)
        ax.set(
            title="Interval width (upper_pi - lower_pi)",
            xlabel="interval width [W]",
            ylabel="count",
        )
        save(fig, "interval_width_histogram")

        bin_order = [b[0] for b in PRODUCTION_BINS]
        groups = [
            pred.loc[pred["prod_bin"] == b, "abs_error"].dropna().values
            for b in bin_order
        ]
        fig, ax = plt.subplots(figsize=(8, 4))
        _box(ax, groups, bin_order)
        ax.set(
            title="Absolute error by production bin",
            ylabel="|y_true - y_pred_mean| [W]",
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        save(fig, "absolute_error_by_bin_boxplot")

        groups = [
            pred.loc[pred["prod_bin"] == b, "interval_width"].dropna().values
            for b in bin_order
        ]
        fig, ax = plt.subplots(figsize=(8, 4))
        _box(ax, groups, bin_order)
        ax.set(title="Interval width by production bin", ylabel="interval width [W]")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        save(fig, "interval_width_by_bin_boxplot")

    bins_path = out / "daytime_bin_summary.csv"
    if bins_path.exists():
        bins = pd.read_csv(bins_path)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, col in zip(axes, ["picp", "mpiw", "nmpil"]):
            ax.bar(bins["bin"], bins[col])
            ax.set_title(col + " by bin")
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
        fig.tight_layout()
        save(fig, "picp_mpiw_nmpil_by_bin")

    unc_path = out / "uncertainty_response.csv"
    if unc_path.exists():
        unc = pd.read_csv(unc_path)
        cols = [
            c
            for c in (
                "mae_ratio_vs_normal",
                "std_ratio_vs_normal",
                "mpiw_ratio_vs_normal",
                "picp_delta_vs_normal",
            )
            if c in unc.columns
        ]
        if cols:
            fig, ax = plt.subplots(figsize=(11, 5))
            x = np.arange(len(unc))
            width = 0.8 / len(cols)
            for i, col in enumerate(cols):
                ax.bar(x + i * width, unc[col], width=width, label=col)
            ax.set_xticks(x + width * (len(cols) - 1) / 2)
            ax.set_xticklabels(unc["category"], rotation=30, ha="right")
            ax.axhline(1.0, color="k", ls=":", lw=0.8)
            ax.legend()
            ax.set_title("Uncertainty response vs normal")
            save(fig, "uncertainty_response_ratios")

    return figure_paths


def _production_bin(y):
    for name, lo, hi in PRODUCTION_BINS:
        if y >= lo and (hi is None or y < hi):
            return name
    return "unknown"
