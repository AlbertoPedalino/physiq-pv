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

DAYTIME_IRRADIANCE_THRESHOLD_WM2 = 10.0
GROUP_NORMAL = "normal"
SPECIFIC_ANOMALY_LABELS = (
    "unusually_low_solar_potential",
    "unusually_high_solar_potential",
    "extreme_temperature_condition",
    "extreme_wind_condition",
)


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
    coverage_target: float = 0.95,
    clc_eta: float = 9.0,
) -> Dict[str, Path]:
    """Boxplots of the report tables (distributions taken across locations):
    one figure per daytime production bin, with MAE, RMSE, PICP, NMPIL and CLC
    each broken down by anomaly category (normal + specific rare-event labels)."""
    import matplotlib.pyplot as plt
    import numpy as np

    out = Path(out_dir)
    pred_path = out / "predictions.csv"
    if not pred_path.exists():
        return {}
    pred = load_prediction_sample(
        pred_path, max_rows=max_plot_rows, random_state=random_state
    )
    # CLC/NMPIL normaliser: full-test target range, matching the runner.
    target_range = float(pred["y_true"].max() - pred["y_true"].min()) or 1e-6
    day = pred[pred["solar_irradiance_poa_target"] > DAYTIME_IRRADIANCE_THRESHOLD_WM2]
    if day.empty:
        return {}

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: Dict[str, Path] = {}

    def boxplot(label, groups, ticklabels, title, ylabel, hline=None) -> None:
        if not groups:
            return
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.boxplot(groups, showfliers=False)
        ax.set_xticks(range(1, len(ticklabels) + 1))
        ax.set_xticklabels(ticklabels, rotation=30, ha="right")
        if hline is not None:
            ax.axhline(hline, color="r", ls="--", lw=1, label=f"target {hline:g}")
            ax.legend()
        ax.set(title=title, ylabel=ylabel)
        path = fig_dir / f"{label}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        figure_paths[label] = path

    ycol = "y_pred_mean" if "y_pred_mean" in day.columns else "y_pred"
    err = day["y_true"] - day[ycol]
    work = day.assign(
        prod_bin=day["y_true"].map(_production_bin),
        abs_error=err.abs(),
        sq_error=err ** 2,
        width=day["upper_pi"] - day["lower_pi"],
        covered=(day["y_true"] >= day["lower_pi"]) & (day["y_true"] <= day["upper_pi"]),
    )

    # Per-location metric within a subset: MAE, RMSE, PICP, NMPIL or CLC.
    def per_location(sub, key):
        g = sub.groupby("location")
        if key == "mae":
            return g["abs_error"].mean().values
        if key == "rmse":
            return np.sqrt(g["sq_error"].mean()).values
        if key == "picp":
            return g["covered"].mean().values
        nmpil = g["width"].mean() / target_range
        if key == "nmpil":
            return nmpil.values
        picp = g["covered"].mean()
        return (nmpil * (1.0 + np.exp(-clc_eta * (picp - coverage_target)))).values  # clc

    # Anomaly categories: normal + each specific rare-event label (overlapping).
    cats = [("normal", work["anomaly_group"] == GROUP_NORMAL)]
    cats += [
        (lab.replace("_solar_potential", "").replace("_condition", ""),
         work["anomaly_label"].str.contains(lab, na=False))
        for lab in SPECIFIC_ANOMALY_LABELS
    ]

    # One figure per production bin; categories on the x-axis.
    bins = [b[0] for b in PRODUCTION_BINS if (work["prod_bin"] == b[0]).any()]
    for key, ylabel, hline in (
        ("mae", "MAE [W]", None),
        ("rmse", "RMSE [W]", None),
        ("picp", "PICP", coverage_target),
        ("nmpil", "NMPIL", None),
        ("clc", "CLC", None),
    ):
        for b in bins:
            bin_mask = work["prod_bin"] == b
            groups, labels = [], []
            for name, cat_mask in cats:
                sub = work[bin_mask & cat_mask]
                if sub.empty:
                    continue
                groups.append(per_location(sub, key))
                labels.append(f"{name}\n(n={len(sub)})")
            boxplot(
                f"{key}_{b}_boxplot", groups, labels,
                f"{key.upper()} — {b} (across locations)", ylabel, hline=hline,
            )

    return figure_paths


def _production_bin(y):
    for name, lo, hi in PRODUCTION_BINS:
        if y >= lo and (hi is None or y < hi):
            return name
    return "unknown"
