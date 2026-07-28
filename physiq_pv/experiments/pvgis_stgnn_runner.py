"""
PVGIS-only ST-GNN experiment runner.

Entrypoint:
  * `python -m physiq_pv.experiments.pvgis_stgnn_runner ...`

Built for sweep / ablation / uncertainty work:
  * `--feature-set`   selects a subset of the 11 PVGIS-only features; the model
                      is instantiated with STGNN(n_features=len(selected)).
  * SDE encoder       BiLSTM+GAT drift and diffusion encoders are aligned stage
                      by stage as in Monaco; `--n-sde-steps`, `--sigma-max`,
                      `--ood-noise-std`, `--lr-g` control it.
  * `--sde-uncertainty`  stochastic inference: model.eval() + `--mc-samples`
                      Brownian-path forward passes -> y_pred_mean/std and a ~95%
                      band. Adds uncertainty metrics per anomaly stratum.
  * `--wandb`         optional, lazily imported; logs namespaced params + metrics
                      (mae/*, rmse/*, ratio/*, uncertainty/*, coverage_95/*).

Hard constraints (PVGIS-only): NO ENERGIA, NO real plant production,
NO Sentinel/SCADA, NO kWp/UPN/load_kwp, NO compute_qs / real QS, NO anomaly
labels as input or target. Anomaly labels are used ONLY for stratified eval.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from physiq_pv.data.pvgis_dataset import (
    ANOMALY_SOURCES,
    DAYTIME_IRRADIANCE_THRESHOLD_WM2,
    DEFAULT_DETECTOR_REGIONAL_QUANTILE,
    DEFAULT_DETECTOR_MIN_TEMPORAL_COVERAGE,
    DEFAULT_EVENT_SPATIAL_QUANTILE,
    DEFAULT_EVENT_TAIL_QUANTILE,
    DEFAULT_TARGET_VARIABLE,
    FEATURE_SETS,
    SPECIFIC_ANOMALY_LABELS,
    build_datasets,
    load_pvgis_year,
    load_pvgis_years,
    make_model,
    resolve_feature_set,
)
from physiq_pv.data.pvgis_labels import (
    attach_anomaly_labels,
    attach_event_labels,
    load_anomaly_labels,
)
from physiq_pv.reporting.posthoc_outputs import PRODUCTION_BINS
from physiq_pv.reporting.run_metrics import (
    build_wandb_metrics,
    compute_metrics,
)
from physiq_pv.reporting.run_report import build_meta, write_outputs, write_report
from physiq_pv.training.train_loop import train_model
from physiq_pv.training.uncertainty import (
    predict,
    predict_sde,
)
from physiq_pv.model.graph_builder import build_graph

# Model-type registry. The STGNN carries aligned drift/diffusion SDE encoders;
# uncertainty comes from the SDE Brownian term, so there is no dropout ablation.
SUPPORTED_MODEL_TYPES = ("stgnn",)
IMPLEMENTED_MODEL_TYPES = ("stgnn",)

# Default --out-dir. Under --wandb (and when left at this default), each run is
# redirected to outputs/wandb_pvgis_stgnn/<run_id>/ so sweep runs never collide.
DEFAULT_OUT_DIR = "outputs/pvgis_stgnn_forecasting"
WANDB_OUT_ROOT = "outputs/wandb_pvgis_stgnn"


def _parse_years(text: str) -> List[int]:
    return [int(y) for y in text.split(",") if y.strip()]


def _slug(text: str) -> str:
    """Filesystem-safe slug for a W&B run name placeholder."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "run"


def _resolve_out_dir(out_dir: str, run_id: str, run_name: Optional[str]) -> str:
    """
    Resolve a per-run output directory under W&B.

    * `{wandb_run_id}` / `{wandb_run_name}` placeholders are substituted (the
      name is slugified for the filesystem);
    * otherwise, if `out_dir` is still the bare default, it is redirected to
      `outputs/wandb_pvgis_stgnn/<run_id>/` so concurrent sweep runs never
      overwrite each other.
    """
    name_slug = _slug(run_name) if run_name else run_id
    if "{wandb_run_id}" in out_dir or "{wandb_run_name}" in out_dir:
        return (
            out_dir.replace("{wandb_run_id}", run_id)
            .replace("{wandb_run_name}", name_slug)
        )
    if out_dir == DEFAULT_OUT_DIR:
        return f"{WANDB_OUT_ROOT}/{run_id}"
    return out_dir


def _write_wandb_run_metadata(
    out_dir: str,
    wandb_run,
    *,
    project: Optional[str],
    entity: Optional[str],
) -> Path:
    """Write enough W&B metadata for post-hoc code to resume this exact run."""
    path = Path(out_dir) / "wandb_run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": getattr(wandb_run, "id", None),
        "name": getattr(wandb_run, "name", None),
        "project": project,
        "entity": entity,
        "url": getattr(wandb_run, "url", None),
        "path": list(getattr(wandb_run, "path", []) or []),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _log_wandb_artifact(
    wandb, wandb_run, paths: dict, log_predictions: bool
) -> bool:
    """
    Attach the run's local outputs to W&B as a versioned artifact.

    Always includes report.md and available metrics CSVs; predictions.csv is
    added only when `log_predictions` is set (it can be very large).
    """
    try:
        artifact = wandb.Artifact(
            f"pvgis-stgnn-report-{wandb_run.id}",
            type="report",
        )
    except Exception as exc:  # noqa: BLE001 - artifact failure must not lose local outputs
        print(f"[wandb-artifact] failed to create main artifact: {exc}")
        return False

    candidates = []
    for key, value in paths.items():
        if value is None:
            continue
        path = Path(value)
        if key == "report" or path.suffix.lower() == ".csv":
            if key != "predictions" or log_predictions:
                candidates.append((key, path))
    report_path = paths.get("report")
    if report_path is not None:
        for path in sorted(Path(report_path).parent.glob("*.csv")):
            if path.name != "predictions.csv" or log_predictions:
                candidates.append((path.name, path))

    added = 0
    seen = set()
    for key, path in candidates:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path.exists():
            print(f"[wandb-artifact] skipped missing file: {path}")
            continue
        try:
            artifact.add_file(str(path))
        except Exception as exc:  # noqa: BLE001 - skip one bad file, keep the artifact
            print(f"[wandb-artifact] skipped file {path}: {exc}")
            continue
        added += 1

    if added == 0:
        print("[wandb-artifact] no files found; main artifact not logged")
        return False
    try:
        wandb_run.log_artifact(artifact)
    except Exception as exc:  # noqa: BLE001 - local report remains authoritative
        print(f"[wandb-artifact] failed to upload main artifact: {exc}")
        return False
    print(
        f"[wandb] logged artifact {artifact.name} "
        f"({'with' if log_predictions else 'without'} predictions.csv, {added} file(s))"
    )
    return True


def compute_interval_metrics(
    y_true,
    lower,
    upper,
    target_range: float,
    gamma: float,
    eta: float,
) -> Dict[str, float]:
    """
    Reliability/sharpness metrics for one set of predictive intervals.

    Pure function (no I/O, no globals). Inspired by uncertainty-aware rainfall
    prediction. For a stratum:
        covered = (y_true >= lower) & (y_true <= upper)
        PICP    = mean(covered)                     empirical coverage
        MPIW    = mean(upper - lower)               mean interval width
        NMPIL   = MPIW / target_range               width normalized by target range
        sigma   = 1 + exp(-eta * (PICP - gamma))    coverage penalty
        CLC     = NMPIL * sigma                      sharpness/reliability trade-off

    `gamma` is the coverage target (e.g. 0.95); `eta` (--clc-eta) sets how hard
    under-coverage is penalised. Returns {picp, mpiw, nmpil, clc}.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if y_true.size == 0:
        return {"picp": float("nan"), "mpiw": float("nan"),
                "nmpil": float("nan"), "clc": float("nan")}
    covered = (y_true >= lower) & (y_true <= upper)
    picp = float(np.mean(covered))
    mpiw = float(np.mean(upper - lower))
    nmpil = mpiw / target_range if target_range else float("nan")
    sigma = 1.0 + float(np.exp(-eta * (picp - gamma)))
    clc = nmpil * sigma
    return {"picp": picp, "mpiw": mpiw, "nmpil": nmpil, "clc": float(clc)}


# Interval kinds -> (lower_col, upper_col).
#   pi         = PRIMARY interval: empirical SDE-sample quantiles.
#   gaussian   = diagnostic Gaussian band (mean ± 1.96*std_raw).
_INTERVAL_KINDS = {
    "pi": ("lower_pi", "upper_pi"),
    "gaussian": ("lower_gaussian", "upper_gaussian"),
}
# Eval strata. "normal"/"rare_extreme" map to the anomaly_group values; anomaly
# labels are used ONLY here for stratified eval, never as model input or target.
_INTERVAL_GROUPS = {
    "global": (None, None),
    "event_normal": ("event_group", "normal"),
    "event_rare_extreme": ("event_group", "rare_or_extreme"),
    "normal": ("anomaly_group", "normal"),
    "rare_extreme": ("anomaly_group", "rare_or_extreme"),
}


def build_interval_metrics(
    predictions, target_range: float, gamma: float, eta: float
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Interval metrics for every available kind (pi/gaussian) x stratum."""
    cols = set(predictions.columns)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for kind, (lo, hi) in _INTERVAL_KINDS.items():
        if not {lo, hi} <= cols:
            continue
        out[kind] = {}
        for gname, (group_col, group_val) in _INTERVAL_GROUPS.items():
            if group_col is not None and group_col not in predictions:
                continue
            sub = predictions if group_col is None else predictions[
                predictions[group_col] == group_val
            ]
            if len(sub) == 0:
                continue
            out[kind][gname] = compute_interval_metrics(
                sub["y_true"], sub[lo], sub[hi], target_range, gamma, eta
            )
    return out


def flatten_interval_metrics(interval_metrics: dict) -> Dict[str, float]:
    """Flatten to namespaced W&B scalars: {metric}_{kind}/{group}."""
    out: Dict[str, float] = {}
    for kind, groups in interval_metrics.items():
        for gname, m in groups.items():
            for metric in ("picp", "mpiw", "nmpil", "clc"):
                v = m.get(metric)
                if v is not None and np.isfinite(v):
                    out[f"{metric}_{kind}/{gname}"] = float(v)
    return out


_DAYTIME_STRATA = (
    "global",
    "daytime",
    "nighttime",
    "normal",
    "rare_extreme",
    "normal_daytime",
    "rare_extreme_daytime",
    "normal_nighttime",
    "rare_extreme_nighttime",
    "event_normal",
    "event_rare_extreme",
    "event_normal_daytime",
    "event_rare_extreme_daytime",
    "event_normal_nighttime",
    "event_rare_extreme_nighttime",
    "high_daytime",
    "peak_daytime",
    "extreme_peak_daytime",
)
_LEGACY_STRATA = {"global", "normal", "rare_extreme"}
_RESIDUAL_MAIN_STRATA = (
    "global",
    "daytime",
    "nighttime",
    "normal",
    "rare_extreme",
    "normal_daytime",
    "rare_extreme_daytime",
    "normal_nighttime",
    "rare_extreme_nighttime",
)
def build_daytime_metrics(
    predictions,
    target_range: float,
    gamma: float,
    eta: float,
    threshold_wm2: float = DAYTIME_IRRADIANCE_THRESHOLD_WM2,
) -> Dict[str, Dict[str, float]]:
    """Eval-only point, uncertainty and interval diagnostics by solar regime."""
    required = {
        "y_true",
        "anomaly_group",
        "event_group",
        "solar_irradiance_poa_target",
        "lower_pi",
        "upper_pi",
        "lower_gaussian",
        "upper_gaussian",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(
            "Daytime diagnostics missing prediction columns: "
            f"{sorted(missing)}"
        )

    pred_col = "y_pred_mean" if "y_pred_mean" in predictions else "y_pred"
    std_col = (
        "y_pred_std_raw"
        if "y_pred_std_raw" in predictions
        else "y_pred_std"
    )
    if pred_col not in predictions or std_col not in predictions:
        raise ValueError(
            "Daytime diagnostics require point predictions and predictive "
            "deviations."
        )

    solar = predictions["solar_irradiance_poa_target"].to_numpy(dtype=float)
    if not np.all(np.isfinite(solar)):
        raise ValueError(
            "solar_irradiance_poa_target contains non-finite values; "
            "daytime/nighttime would not form a complete partition."
        )
    daytime = solar > threshold_wm2
    nighttime = solar <= threshold_wm2
    normal = predictions["anomaly_group"].to_numpy() == "normal"
    rare = predictions["anomaly_group"].to_numpy() == "rare_or_extreme"
    event_normal = predictions["event_group"].to_numpy() == "normal"
    event_rare = predictions["event_group"].to_numpy() == "rare_or_extreme"
    y_true_all = predictions["y_true"].to_numpy(dtype=float)
    daytime_y_true = y_true_all[daytime]
    if daytime_y_true.size:
        p75_daytime, p90_daytime, p95_daytime = np.percentile(
            daytime_y_true, [75, 90, 95]
        )
    else:
        p75_daytime = p90_daytime = p95_daytime = float("nan")
    masks = {
        "global": np.ones(len(predictions), dtype=bool),
        "daytime": daytime,
        "nighttime": nighttime,
        "normal": normal,
        "rare_extreme": rare,
        "normal_daytime": normal & daytime,
        "rare_extreme_daytime": rare & daytime,
        "normal_nighttime": normal & nighttime,
        "rare_extreme_nighttime": rare & nighttime,
        "event_normal": event_normal,
        "event_rare_extreme": event_rare,
        "event_normal_daytime": event_normal & daytime,
        "event_rare_extreme_daytime": event_rare & daytime,
        "event_normal_nighttime": event_normal & nighttime,
        "event_rare_extreme_nighttime": event_rare & nighttime,
        "high_daytime": daytime & (y_true_all > p75_daytime),
        "peak_daytime": daytime & (y_true_all > p90_daytime),
        "extreme_peak_daytime": daytime & (y_true_all > p95_daytime),
    }

    out: Dict[str, Dict[str, float]] = {}
    for stratum in _DAYTIME_STRATA:
        sub = predictions.loc[masks[stratum]]
        y_true = sub["y_true"].to_numpy(dtype=float)
        y_pred = sub[pred_col].to_numpy(dtype=float)
        y_std = sub[std_col].to_numpy(dtype=float)
        count = len(sub)
        row: Dict[str, float] = {"count": int(count)}
        if count == 0:
            for metric in (
                "mae",
                "rmse",
                "mean_std",
                "median_std",
                "p90_std",
                "picp_pi",
                "mpiw_pi",
                "nmpil_pi",
                "clc_pi",
                "picp_gaussian",
                "mpiw_gaussian",
                "nmpil_gaussian",
                "clc_gaussian",
                "fraction_y_true_zero",
                "fraction_lower_pi_leq_zero",
                "fraction_lower_gaussian_leq_zero",
                "coverage_pi_y_true_zero",
                "coverage_gaussian_y_true_zero",
                "coverage_pi_y_true_positive",
                "coverage_gaussian_y_true_positive",
                "mean_residual",
                "median_residual",
                "fraction_underprediction",
                "fraction_above_interval",
            ):
                row[metric] = float("nan")
            out[stratum] = row
            continue

        error = y_pred - y_true
        row.update(
            {
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "mean_std": float(np.mean(y_std)),
                "median_std": float(np.median(y_std)),
                "p90_std": float(np.percentile(y_std, 90)),
                "mean_residual": float(np.mean(error)),
                "median_residual": float(np.median(error)),
                "fraction_underprediction": float(np.mean(error < 0.0)),
            }
        )

        interval_coverage = {}
        for kind, (lower_col, upper_col) in {
            "pi": ("lower_pi", "upper_pi"),
            "gaussian": ("lower_gaussian", "upper_gaussian"),
        }.items():
            lower = sub[lower_col].to_numpy(dtype=float)
            upper = sub[upper_col].to_numpy(dtype=float)
            interval = compute_interval_metrics(
                y_true, lower, upper, target_range, gamma, eta
            )
            for metric, value in interval.items():
                row[f"{metric}_{kind}"] = value
            interval_coverage[kind] = (y_true >= lower) & (y_true <= upper)
            if kind == "pi":
                row["fraction_above_interval"] = float(np.mean(y_true > upper))

        y_zero = np.isclose(y_true, 0.0, rtol=0.0, atol=1e-8)
        y_positive = y_true > 0.0
        lower_pi = sub["lower_pi"].to_numpy(dtype=float)
        lower_gaussian = sub["lower_gaussian"].to_numpy(dtype=float)
        row.update(
            {
                "fraction_y_true_zero": float(np.mean(y_zero)),
                "fraction_lower_pi_leq_zero": float(np.mean(lower_pi <= 0.0)),
                "fraction_lower_gaussian_leq_zero": float(
                    np.mean(lower_gaussian <= 0.0)
                ),
                "coverage_pi_y_true_zero": (
                    float(np.mean(interval_coverage["pi"][y_zero]))
                    if np.any(y_zero)
                    else float("nan")
                ),
                "coverage_gaussian_y_true_zero": (
                    float(np.mean(interval_coverage["gaussian"][y_zero]))
                    if np.any(y_zero)
                    else float("nan")
                ),
                "coverage_pi_y_true_positive": (
                    float(np.mean(interval_coverage["pi"][y_positive]))
                    if np.any(y_positive)
                    else float("nan")
                ),
                "coverage_gaussian_y_true_positive": (
                    float(np.mean(interval_coverage["gaussian"][y_positive]))
                    if np.any(y_positive)
                    else float("nan")
                ),
            }
        )
        out[stratum] = row
    return out


def flatten_daytime_metrics(daytime_metrics: dict) -> Dict[str, float]:
    """Flatten new daytime diagnostics without replacing legacy W&B metrics."""
    out: Dict[str, float] = {}
    for stratum, metrics in daytime_metrics.items():
        out[f"count/{stratum}"] = int(metrics["count"])
        median_std = metrics.get("median_std")
        if median_std is not None and np.isfinite(median_std):
            out[f"uncertainty/median_std_{stratum}"] = float(median_std)

        if stratum not in _LEGACY_STRATA:
            for metric in ("mae", "rmse"):
                value = metrics.get(metric)
                if value is not None and np.isfinite(value):
                    out[f"{metric}/{stratum}"] = float(value)
            for metric in ("mean_std", "p90_std"):
                value = metrics.get(metric)
                if value is not None and np.isfinite(value):
                    out[f"uncertainty/{metric}_{stratum}"] = float(value)
            for kind in ("pi", "gaussian"):
                for metric in ("picp", "mpiw", "nmpil", "clc"):
                    value = metrics.get(f"{metric}_{kind}")
                    if value is not None and np.isfinite(value):
                        out[f"{metric}_{kind}/{stratum}"] = float(value)

        for metric in (
            "fraction_y_true_zero",
            "fraction_lower_pi_leq_zero",
            "fraction_lower_gaussian_leq_zero",
            "coverage_pi_y_true_zero",
            "coverage_gaussian_y_true_zero",
            "coverage_pi_y_true_positive",
            "coverage_gaussian_y_true_positive",
        ):
            value = metrics.get(metric)
            if value is not None and np.isfinite(value):
                out[f"{metric}/{stratum}"] = float(value)
    return out


def _residual_metric_row(
    stratum: str,
    kind: str,
    sub,
    target_range: float,
    gamma: float,
    eta: float,
) -> Dict[str, float]:
    """Compute signed residual and interval-miss diagnostics for one subset."""
    metric_names = (
        "mae",
        "rmse",
        "mean_residual",
        "median_residual",
        "std_residual",
        "p05_residual",
        "p25_residual",
        "p75_residual",
        "p95_residual",
        "fraction_overprediction",
        "fraction_underprediction",
        "mean_overprediction_error",
        "mean_underprediction_error",
        "picp_pi",
        "mpiw_pi",
        "nmpil_pi",
        "clc_pi",
        "picp_gaussian",
        "mpiw_gaussian",
        "fraction_below_interval",
        "fraction_above_interval",
    )
    row: Dict[str, float] = {
        "stratum": stratum,
        "kind": kind,
        "count": int(len(sub)),
    }
    if len(sub) == 0:
        row.update({name: float("nan") for name in metric_names})
        return row

    pred_col = "y_pred_mean" if "y_pred_mean" in sub else "y_pred"
    y_true = sub["y_true"].to_numpy(dtype=float)
    y_pred = sub[pred_col].to_numpy(dtype=float)
    residual = y_pred - y_true
    over = residual > 0.0
    under = residual < 0.0

    row.update(
        {
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mean_residual": float(np.mean(residual)),
            "median_residual": float(np.median(residual)),
            "std_residual": float(np.std(residual)),
            "p05_residual": float(np.percentile(residual, 5)),
            "p25_residual": float(np.percentile(residual, 25)),
            "p75_residual": float(np.percentile(residual, 75)),
            "p95_residual": float(np.percentile(residual, 95)),
            "fraction_overprediction": float(np.mean(over)),
            "fraction_underprediction": float(np.mean(under)),
            "mean_overprediction_error": (
                float(np.mean(residual[over])) if np.any(over) else float("nan")
            ),
            "mean_underprediction_error": (
                float(np.mean(residual[under])) if np.any(under) else float("nan")
            ),
        }
    )

    lower_pi = sub["lower_pi"].to_numpy(dtype=float)
    upper_pi = sub["upper_pi"].to_numpy(dtype=float)
    pi_metrics = compute_interval_metrics(
        y_true, lower_pi, upper_pi, target_range, gamma, eta
    )
    row.update({f"{name}_pi": value for name, value in pi_metrics.items()})
    row["fraction_below_interval"] = float(np.mean(y_true < lower_pi))
    row["fraction_above_interval"] = float(np.mean(y_true > upper_pi))

    gaussian = compute_interval_metrics(
        y_true,
        sub["lower_gaussian"].to_numpy(dtype=float),
        sub["upper_gaussian"].to_numpy(dtype=float),
        target_range,
        gamma,
        eta,
    )
    row["picp_gaussian"] = gaussian["picp"]
    row["mpiw_gaussian"] = gaussian["mpiw"]
    return row


def build_residual_bias_metrics(
    predictions,
    target_range: float,
    gamma: float,
    eta: float,
    threshold_wm2: float = DAYTIME_IRRADIANCE_THRESHOLD_WM2,
) -> List[Dict[str, float]]:
    """Eval-only residual diagnostics by regime, anomaly label and fixed PV bin."""
    required = {
        "y_true",
        "anomaly_group",
        "anomaly_label",
        "solar_irradiance_poa_target",
        "lower_pi",
        "upper_pi",
        "lower_gaussian",
        "upper_gaussian",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(
            "Residual diagnostics missing prediction columns: "
            f"{sorted(missing)}"
        )
    if not ({"y_pred_mean", "y_pred"} & set(predictions.columns)):
        raise ValueError("Residual diagnostics require y_pred_mean or y_pred.")

    solar = predictions["solar_irradiance_poa_target"].to_numpy(dtype=float)
    if not np.all(np.isfinite(solar)):
        raise ValueError(
            "solar_irradiance_poa_target contains non-finite values."
        )
    y_true = predictions["y_true"].to_numpy(dtype=float)
    daytime = solar > threshold_wm2
    nighttime = ~daytime
    groups = predictions["anomaly_group"].to_numpy()
    normal = groups == "normal"
    rare = groups == "rare_or_extreme"
    masks = {
        "global": np.ones(len(predictions), dtype=bool),
        "daytime": daytime,
        "nighttime": nighttime,
        "normal": normal,
        "rare_extreme": rare,
        "normal_daytime": normal & daytime,
        "rare_extreme_daytime": rare & daytime,
        "normal_nighttime": normal & nighttime,
        "rare_extreme_nighttime": rare & nighttime,
    }

    rows: List[Dict[str, float]] = []
    for stratum in _RESIDUAL_MAIN_STRATA:
        rows.append(
            _residual_metric_row(
                stratum,
                "main_stratum",
                predictions.loc[masks[stratum]],
                target_range,
                gamma,
                eta,
            )
        )

    label_sets = predictions["anomaly_label"].fillna("").astype(str).apply(
        lambda value: {
            part.strip() for part in value.split(",") if part.strip()
        }
    )
    for label in SPECIFIC_ANOMALY_LABELS:
        label_mask = label_sets.apply(lambda values: label in values).to_numpy()
        rows.append(
            _residual_metric_row(
                f"label:{label}",
                "anomaly_label",
                predictions.loc[label_mask],
                target_range,
                gamma,
                eta,
            )
        )

    for name, lower, upper in PRODUCTION_BINS:
        mask = daytime & (y_true >= lower)
        if upper is not None:
            mask &= y_true < upper
        rows.append(
            _residual_metric_row(
                name,
                "daytime_production_bin",
                predictions.loc[mask],
                target_range,
                gamma,
                eta,
            )
        )
    return rows


def flatten_residual_bias_metrics(
    residual_metrics: List[Dict[str, float]],
) -> Dict[str, float]:
    """Flatten diagnostic-only residual metrics without replacing legacy keys."""
    out: Dict[str, float] = {}
    residual_names = (
        "mean_residual",
        "median_residual",
        "std_residual",
        "p05_residual",
        "p25_residual",
        "p75_residual",
        "p95_residual",
        "fraction_overprediction",
        "fraction_underprediction",
        "mean_overprediction_error",
        "mean_underprediction_error",
        "fraction_below_interval",
        "fraction_above_interval",
    )
    supplemental_names = (
        "mae",
        "rmse",
        "picp_pi",
        "mpiw_pi",
        "nmpil_pi",
        "clc_pi",
        "picp_gaussian",
        "mpiw_gaussian",
    )
    for row in residual_metrics:
        stratum = str(row["stratum"])
        wandb_stratum = (
            stratum.removeprefix("label:")
            if stratum.startswith("label:")
            else stratum
        )
        out[f"residual_count/{wandb_stratum}"] = int(row["count"])
        for metric in residual_names:
            value = row.get(metric)
            if value is not None and np.isfinite(value):
                out[f"{metric}/{wandb_stratum}"] = float(value)
        if row["kind"] in {"anomaly_label", "daytime_production_bin"}:
            for metric in supplemental_names:
                value = row.get(metric)
                if value is not None and np.isfinite(value):
                    out[f"{metric}/{wandb_stratum}"] = float(value)
    return out


def _optional_clip_max(value: str) -> Optional[float]:
    """Parse a positive upper clip or the strings none/null."""
    if value.strip().lower() in {"none", "null"}:
        return None
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            "pv_target_clip_max must be positive, or none/null"
        )
    return parsed


def add_pvgis_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Register PVGIS-only experiment arguments on `parser`.

    Nothing is marked required at the argparse level (so `python main.py` in
    default mode never trips over them); required args are validated in
    `run_from_args` only when this mode actually runs.
    """
    # Each hyphenated flag also gets an underscore alias (e.g. --train_years) so
    # W&B sweep `${args}` (which emits --param_name=value) works without a wrapper.
    g = parser.add_argument_group("pvgis_stgnn mode")
    g.add_argument("--pvgis-dir", "--pvgis_dir", default=None,
                   help="Dir with piedmont_pvgis_{year}.nc files.")
    g.add_argument("--train-years", "--train_years", default=None,
                   help="Comma-separated train years.")
    g.add_argument("--validation-year", "--validation_year", type=int, default=None,
                   help="Held-out validation year among --train-years. Default: latest.")
    g.add_argument("--test-year", "--test_year", type=int, default=None)
    g.add_argument("--anomaly-scores", "--anomaly_scores", default=None,
                   help="Test-year climatology or detector scores. Used only for "
                        "stratified evaluation/event metadata, never as model input.")
    g.add_argument(
        "--anomaly-source",
        "--anomaly_source",
        choices=ANOMALY_SOURCES,
        default="climatology",
        help="Interpret anomaly CSVs as per-variable climatology scores or as "
             "pre-thresholded detector decisions.",
    )
    g.add_argument("--out-dir", "--out_dir", default=DEFAULT_OUT_DIR,
                   help="Output dir. Supports {wandb_run_id}/{wandb_run_name} "
                        "placeholders; under --wandb the default is redirected to "
                        f"{WANDB_OUT_ROOT}/<run_id>/ so sweep runs stay unique.")
    g.add_argument("--seq-len", "--seq_len", type=int, default=24)
    g.add_argument("--horizon", type=int, default=1)
    g.add_argument("--target-variable", "--target_variable", default=DEFAULT_TARGET_VARIABLE)
    g.add_argument(
        "--pv-target-clip-max",
        "--pv_target_clip_max",
        type=_optional_clip_max,
        default=None,
        help="Upper clip for normalized PV target and pv_lag_pvgis. "
        "Default none keeps only the lower non-negativity clip.",
    )
    g.add_argument("--epochs", type=int, default=60)
    g.add_argument("--batch-size", "--batch_size", type=int, default=16)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--max-dist-km", "--max_dist_km", type=float, default=20.0)
    g.add_argument("--distance-scale-km", "--distance_scale_km", type=float,
                   default=None, help="Gaussian graph-prior length scale. "
                   "Default: half --max-dist-km.")
    g.add_argument("--edge-prior-strength", "--edge_prior_strength", type=float,
                   default=1.0, help="Multiplier of the Gaussian log-prior "
                   "added to learned GAT logits.")
    g.add_argument("--kt-poa-max", "--kt_poa_max", type=float, default=1.6,
                   help="Shared upper bound for target and predicted kt_poa.")
    g.add_argument("--validation-metric", "--validation_metric",
                   choices=["rmse_daytime", "mae_daytime", "mse"],
                   default="rmse_daytime")
    g.add_argument("--early-stopping-patience", "--early_stopping_patience",
                   type=int, default=10)
    g.add_argument("--early-stopping-min-delta", "--early_stopping_min_delta",
                   type=float, default=0.0)
    g.add_argument("--max-train-samples", "--max_train_samples", type=int, default=None)
    g.add_argument("--max-test-samples", "--max_test_samples", type=int, default=None)
    g.add_argument("--skip-predictions-csv", "--skip_predictions_csv",
                   action="store_true",
                   help="Do not write the (large) predictions.csv; metrics + "
                        "report.md are still produced.")
    g.add_argument("--file-template", "--file_template", default="piedmont_pvgis_{year}.nc")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--seed", type=int, default=42)
    # Model / ablation
    g.add_argument("--model-type", "--model_type", default="stgnn", choices=SUPPORTED_MODEL_TYPES,
                   help="stgnn (the only model type): ST-GNN with aligned Monaco SDE encoders.")
    g.add_argument("--feature-set", "--feature_set", default="full", choices=sorted(FEATURE_SETS),
                   help="Feature ablation; n_features = len(selected features).")
    g.add_argument("--dropout", type=float, default=0.0,
                   help="STGNN dropout (regulariser inside the GAT/encoder).")
    # Training point-loss ablation. Isolated knob: only the loss module changes.
    # Monaco-style aligned drift/diffusion encoder. The diffusion path is trained
    # low in-distribution / high on a Gaussian-noise pseudo-OOD batch.
    g.add_argument("--n-sde-steps", "--n_sde_steps", type=int, default=2,
                   help="Aligned stochastic encoder stages. This branch fixes the "
                        "original backbone to two stages: one BiLSTM plus one GAT.")
    g.add_argument("--sigma-max", "--sigma_max", type=float, default=0.5,
                   help="Global multiplier of every bounded sigmoid diffusion gate; "
                        "Default 0.5 matches Monaco's SDE U-Net repo.")
    g.add_argument("--ood-noise-std", "--ood_noise_std", type=float, default=1.0,
                   help="Std of the Gaussian noise added to training inputs to build "
                        "the pseudo-OOD batch on which g is pushed high (> 0). "
                        "Default 1.0 matches Monaco's randn_like(x) + x.")
    g.add_argument("--lr-g", "--lr_g", type=float, default=None,
                   help="Learning rate for the diffusion-net optimiser (Algorithm 1). "
                        "Defaults to --lr when omitted.")
    g.add_argument("--train-normal-only", "--train_normal_only",
                   action="store_true",
                   help="Physically keep only graph-wide events whose complete "
                        "input history and target are normal, for both training "
                        "and validation. "
                        "Requires --train-anomaly-scores.")
    g.add_argument("--train-anomaly-scores", "--train_anomaly_scores", default=None,
                   help="TRAIN+validation scores. Required by detector sources "
                        "to fit seasonal regional thresholds and by "
                        "--train-normal-only; never a model input/target.")
    g.add_argument(
        "--event-spatial-quantile",
        "--event_spatial_quantile",
        type=float,
        default=DEFAULT_EVENT_SPATIAL_QUANTILE,
        help="Spatial quantile of |anomaly_score| across graph nodes used as "
             "regional severity per timestamp/variable.",
    )
    g.add_argument(
        "--event-tail-quantile",
        "--event_tail_quantile",
        type=float,
        default=DEFAULT_EVENT_TAIL_QUANTILE,
        help="Training-only temporal quantile of regional severity above which "
             "a graph-wide event is rare.",
    )
    g.add_argument(
        "--detector-regional-quantile",
        "--detector_regional_quantile",
        type=float,
        default=DEFAULT_DETECTOR_REGIONAL_QUANTILE,
        help="Training-only seasonal quantile of the anomalous-node fraction "
             "used to define a regional detector event (default: 0.975).",
    )
    g.add_argument(
        "--detector-min-temporal-coverage",
        "--detector_min_temporal_coverage",
        type=float,
        default=DEFAULT_DETECTOR_MIN_TEMPORAL_COVERAGE,
        help="Minimum fraction of hourly timestamps that a detector CSV must "
             "cover. Rejects sparse detector score strides.",
    )
    # Inclined clear-sky-index auxiliary head.
    g.add_argument("--use-irradiance-head", "--use_irradiance_head",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Create the clear-sky-index head. Disable for a "
                        "production-only model.")
    g.add_argument("--use-irradiance-loss", "--use_irradiance_loss",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Add an auxiliary point-loss term on the irradiance head "
                        "(pred_kt_poa vs target-time inclined clear-sky index) to the "
                        "training loss. Requires --use-irradiance-head.")
    g.add_argument("--irradiance-loss-weight", "--irradiance_loss_weight",
                   type=float, default=1.0,
                   help="Weight of the auxiliary irradiance loss term (only "
                        "meaningful with --use-irradiance-loss).")
    g.add_argument("--kt-aux-loss-weight", "--kt_aux_loss_weight",
                   type=float, default=None,
                   help="Sweep-friendly single-flag interface for the kt auxiliary "
                        "loss: 0.0 disables it; w > 0 enables it with weight w. "
                        "Equivalent to --use-irradiance-loss --irradiance-loss-weight w; "
                        "cannot be combined with --use-irradiance-loss.")
    # SDE uncertainty — eval() + N stochastic Brownian paths -> mean/std + PIs.
    g.add_argument("--sde-uncertainty", "--sde_uncertainty",
                   dest="sde_uncertainty", action="store_true",
                   help="Produce predictive intervals from the SDE: N stochastic "
                        "Brownian-path forward passes per batch (no dropout needed).")
    g.add_argument("--mc-samples", "--mc_samples", type=int, default=20,
                   help="Number of stochastic SDE forward passes per batch (>= 2).")
    g.add_argument("--clc-eta", "--clc_eta", type=float, default=9.0,
                   help="Sharpness sensitivity eta for the CLC interval metric "
                        "CLC = NMPIL * (1 + exp(-eta * (PICP - gamma))); "
                        "gamma is the coverage target. Eval-only, never affects training.")
    g.add_argument("--coverage-target", "--coverage_target", type=float, default=0.95,
                   help="Target coverage for the empirical SDE predictive interval.")
    # Optional W&B
    g.add_argument("--wandb", action="store_true", help="Enable optional W&B logging.")
    g.add_argument("--wandb-project", "--wandb_project", default="physiq_pv")
    g.add_argument("--wandb-entity", "--wandb_entity", default=None,
                   help="W&B entity (team/user); e.g. albertopedalino-politecnico-di-torino.")
    g.add_argument("--wandb-run-name", "--wandb_run_name", default=None)
    g.add_argument("--wandb-log-predictions", "--wandb_log_predictions",
                   action="store_true",
                   help="Also log predictions.csv to the W&B artifact (off by "
                        "default — it can be very large).")
    g.add_argument("--wandb-upload-artifacts", "--wandb_upload_artifacts",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Upload report.md + metrics CSVs (and post-hoc files) as W&B "
                        "artifacts. Only effective with --wandb. Default True; pass "
                        "--no-wandb-upload-artifacts to disable.")
    g.add_argument("--wandb-upload-predictions", "--wandb_upload_predictions",
                   action="store_true",
                   help="Also upload the (large) predictions.csv to the W&B artifact "
                        "(default False). Alias-equivalent to --wandb-log-predictions.")
    return parser


def build_arg_parser() -> argparse.ArgumentParser:
    """Standalone parser for the script wrapper."""
    p = argparse.ArgumentParser(
        description="PVGIS-only ST-GNN forecasting with stratified eval, feature "
                    "ablation, optional W&B, and neural-SDE uncertainty."
    )
    return add_pvgis_arguments(p)


def _fail(parser: Optional[argparse.ArgumentParser], msg: str) -> None:
    if parser is not None:
        parser.error(msg)
    raise SystemExit(f"error: {msg}")


def _validate(args: argparse.Namespace, parser: Optional[argparse.ArgumentParser]) -> None:
    missing = [
        name for name, val in (
            ("--pvgis-dir", args.pvgis_dir),
            ("--train-years", args.train_years),
            ("--test-year", args.test_year),
        ) if val in (None, "")
    ]
    if missing:
        _fail(parser, f"--mode pvgis_stgnn requires: {', '.join(missing)}.")
    train_years = _parse_years(args.train_years)
    if len(set(train_years)) < 2:
        _fail(
            parser,
            "--train-years must contain at least two distinct years so one can "
            "remain held out for validation.",
        )
    if args.validation_year is not None and args.validation_year not in train_years:
        _fail(parser, "--validation-year must be included in --train-years.")
    if args.validation_year is None:
        args.validation_year = max(train_years)
    if args.test_year in train_years:
        _fail(parser, "--test-year must not also appear in --train-years.")
    if not np.isfinite(args.max_dist_km) or args.max_dist_km <= 0:
        _fail(parser, "--max-dist-km must be finite and positive.")
    if args.distance_scale_km is not None and (
        not np.isfinite(args.distance_scale_km) or args.distance_scale_km <= 0
    ):
        _fail(parser, "--distance-scale-km must be positive.")
    if not np.isfinite(args.edge_prior_strength) or args.edge_prior_strength < 0:
        _fail(parser, "--edge-prior-strength must be finite and non-negative.")
    if not np.isfinite(args.kt_poa_max) or args.kt_poa_max <= 0:
        _fail(parser, "--kt-poa-max must be finite and positive.")
    if args.early_stopping_patience < 1:
        _fail(parser, "--early-stopping-patience must be >= 1.")
    if (
        not np.isfinite(args.early_stopping_min_delta)
        or args.early_stopping_min_delta < 0
    ):
        _fail(parser, "--early-stopping-min-delta must be finite and >= 0.")
    # Fail fast on missing anomaly-scores files BEFORE the (expensive) training.
    for flag, path in (
        ("--anomaly-scores", args.anomaly_scores),
        ("--train-anomaly-scores", args.train_anomaly_scores),
    ):
        if path and not Path(path).exists():
            _fail(parser, f"{flag} file not found: {path}")
    if args.train_normal_only and not args.train_anomaly_scores:
        _fail(parser, "--train-normal-only requires --train-anomaly-scores (TRAIN-year scores).")
    if args.anomaly_source == "detector" and not args.train_anomaly_scores:
        _fail(
            parser,
            "--anomaly-source detector requires --train-anomaly-scores to fit "
            "training-only seasonal regional thresholds.",
        )
    if args.train_normal_only and not args.anomaly_scores:
        _fail(
            parser,
            "--train-normal-only requires --anomaly-scores for regional event "
            "stratification on the complete test set.",
        )
    for flag, value in (
        ("--event-spatial-quantile", args.event_spatial_quantile),
        ("--event-tail-quantile", args.event_tail_quantile),
        ("--detector-regional-quantile", args.detector_regional_quantile),
    ):
        if not np.isfinite(value) or not 0.5 < value < 1.0:
            _fail(parser, f"{flag} must be finite and in (0.5, 1.0), got {value}.")
    for flag, value in (
        (
            "--detector-min-temporal-coverage",
            args.detector_min_temporal_coverage,
        ),
    ):
        if not np.isfinite(value) or not 0.0 < value <= 1.0:
            _fail(parser, f"{flag} must be finite and in (0, 1], got {value}.")
    if args.model_type not in IMPLEMENTED_MODEL_TYPES:
        _fail(
            parser,
            f"model_type='{args.model_type}' is not implemented. "
            f"Available: {list(IMPLEMENTED_MODEL_TYPES)}.",
        )
    # --kt-aux-loss-weight is pure sugar over the use_irradiance_loss interface:
    # resolve it FIRST so every check below sees the effective configuration.
    if args.kt_aux_loss_weight is not None:
        if not np.isfinite(args.kt_aux_loss_weight) or args.kt_aux_loss_weight < 0.0:
            _fail(
                parser,
                "--kt-aux-loss-weight must be finite and >= 0, got "
                f"{args.kt_aux_loss_weight}.",
            )
        args.use_irradiance_loss = args.kt_aux_loss_weight > 0.0
        if args.use_irradiance_loss:
            args.irradiance_loss_weight = float(args.kt_aux_loss_weight)
    if args.use_irradiance_loss and not args.use_irradiance_head:
        _fail(
            parser,
            "--use-irradiance-loss requires the irradiance head: it cannot be "
            "combined with --no-use-irradiance-head (there is no head to "
            "supervise). Drop --use-irradiance-loss or re-enable the head.",
        )
    if not np.isfinite(args.irradiance_loss_weight) or args.irradiance_loss_weight < 0.0:
        _fail(
            parser,
            "--irradiance-loss-weight must be finite and >= 0, got "
            f"{args.irradiance_loss_weight}.",
        )
    # Neural-SDE block hyper-parameters.
    if args.n_sde_steps != 2:
        _fail(
            parser,
            "--n-sde-steps must be 2 to preserve the original backbone "
            f"(one BiLSTM stage + one GAT stage), got {args.n_sde_steps}.",
        )
    if not np.isfinite(args.sigma_max) or args.sigma_max <= 0.0:
        _fail(parser, f"--sigma-max must be finite and > 0, got {args.sigma_max}.")
    if not np.isfinite(args.ood_noise_std) or args.ood_noise_std <= 0.0:
        _fail(parser, f"--ood-noise-std must be finite and > 0, got {args.ood_noise_std}.")
    if args.lr_g is not None and (not np.isfinite(args.lr_g) or args.lr_g <= 0.0):
        _fail(parser, f"--lr-g must be finite and > 0 when set, got {args.lr_g}.")
    if args.sde_uncertainty and args.mc_samples < 2:
        _fail(parser, f"--mc-samples must be >= 2 for SDE sampling, got {args.mc_samples}.")
    if not 0.0 < args.coverage_target < 1.0:
        _fail(parser, f"--coverage-target must be in (0, 1), got {args.coverage_target}.")


def run_from_args(
    args: argparse.Namespace, parser: Optional[argparse.ArgumentParser] = None
) -> dict:
    """Run the PVGIS-only ST-GNN experiment. Returns the written-output paths."""
    _validate(args, parser)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # W&B upload plan (deterministic from flags; surfaced in report.md).
    upload_artifacts = bool(args.wandb) and bool(args.wandb_upload_artifacts)
    predictions_upload = bool(args.wandb_log_predictions) or bool(args.wandb_upload_predictions)

    features = resolve_feature_set(args.feature_set)

    if args.sde_uncertainty:
        print(
            "INFO: paper-style protocol — predictive intervals are built directly "
            "from the SDE Brownian-path sample distribution (lower_pi/upper_pi, "
            "empirical quantiles). The Gaussian band "
            "(mean +/- 1.96*std_raw) is logged only as a secondary diagnostic. "
        )

    # Optional W&B (lazy import; never required).
    wandb_run = None
    if args.wandb:
        import wandb  # noqa: PLC0415 — optional dependency, imported only when enabled
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "mode": "pvgis_stgnn",
                "model_type": args.model_type,
                "feature_set": args.feature_set,
                "selected_features": features,
                "n_features": len(features),
                "target_variable": args.target_variable,
                "pv_target_clip_max": args.pv_target_clip_max,
                "train_years": args.train_years,
                "validation_year": args.validation_year,
                "test_year": args.test_year,
                "seq_len": args.seq_len,
                "horizon": args.horizon,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "dropout": args.dropout,
                "n_sde_steps": int(args.n_sde_steps),
                "sigma_max": float(args.sigma_max),
                "ood_noise_std": float(args.ood_noise_std),
                "lr_g": args.lr_g,
                "train_normal_only": bool(args.train_normal_only),
                "anomaly_source": args.anomaly_source,
                "event_spatial_quantile": float(args.event_spatial_quantile),
                "event_tail_quantile": float(args.event_tail_quantile),
                "detector_regional_quantile": float(
                    args.detector_regional_quantile
                ),
                "detector_min_temporal_coverage": float(
                    args.detector_min_temporal_coverage
                ),
                "kt_poa_max": float(args.kt_poa_max),
                "distance_scale_km": args.distance_scale_km,
                "edge_prior_strength": float(args.edge_prior_strength),
                "validation_metric": args.validation_metric,
                "early_stopping_patience": int(args.early_stopping_patience),
                "early_stopping_min_delta": float(args.early_stopping_min_delta),
                "use_irradiance_head": bool(args.use_irradiance_head),
                "use_irradiance_loss": bool(args.use_irradiance_loss),
                "irradiance_loss_weight": float(args.irradiance_loss_weight),
                "device": args.device,
                "max_train_samples": args.max_train_samples,
                "max_test_samples": args.max_test_samples,
                "skip_predictions_csv": args.skip_predictions_csv,
                "sde_uncertainty": args.sde_uncertainty,
                "mc_samples": args.mc_samples,
                "clc_eta": args.clc_eta,
                "coverage_target": args.coverage_target,
                "anomaly_scores": args.anomaly_scores,
                "wandb_log_predictions": args.wandb_log_predictions,
                "wandb_upload_artifacts": bool(args.wandb_upload_artifacts),
                "wandb_upload_predictions": bool(args.wandb_upload_predictions),
            },
        )

    # Per-run output dir: unique under --wandb so sweep runs never collide.
    out_dir = args.out_dir
    if wandb_run is not None:
        out_dir = _resolve_out_dir(out_dir, wandb_run.id, wandb_run.name)
        print(f"[wandb] run id={wandb_run.id} name={wandb_run.name} -> out_dir={out_dir}")
        meta_path = _write_wandb_run_metadata(
            out_dir,
            wandb_run,
            project=args.wandb_project,
            entity=args.wandb_entity,
        )
        print(f"[wandb] run metadata -> {meta_path}")

    # Effective config banner — confirms which values the sweep actually injected.
    print(
        "[config] " + "  ".join(
            f"{k}={v}" for k, v in (
                ("seed", args.seed),
                ("batch_size", args.batch_size),
                ("epochs", args.epochs),
                ("mc_samples", args.mc_samples),
                ("skip_predictions_csv", args.skip_predictions_csv),
                ("train_years", args.train_years),
                ("validation_year", args.validation_year),
                ("test_year", args.test_year),
                ("pv_target_clip_max", args.pv_target_clip_max),
                ("n_sde_steps", args.n_sde_steps),
                ("sigma_max", args.sigma_max),
                ("ood_noise_std", args.ood_noise_std),
                ("lr_g", args.lr_g),
                ("train_normal_only", args.train_normal_only),
                ("anomaly_source", args.anomaly_source),
                ("event_spatial_quantile", args.event_spatial_quantile),
                ("event_tail_quantile", args.event_tail_quantile),
                (
                    "detector_regional_quantile",
                    args.detector_regional_quantile,
                ),
                ("use_irradiance_head", args.use_irradiance_head),
                ("use_irradiance_loss", args.use_irradiance_loss),
                ("irradiance_loss_weight", args.irradiance_loss_weight),
                ("kt_poa_max", args.kt_poa_max),
                ("distance_scale_km", args.distance_scale_km),
                ("edge_prior_strength", args.edge_prior_strength),
                ("validation_metric", args.validation_metric),
                ("early_stopping_patience", args.early_stopping_patience),
                ("out_dir", out_dir),
            )
        )
    )
    t_run_start = time.perf_counter()

    train_years = _parse_years(args.train_years)
    print(
        f"[1/6] Loading PVGIS years "
        f"(train+validation={train_years}, "
        f"validation={args.validation_year or max(train_years)}, "
        f"test={args.test_year}) "
        f"| model={args.model_type} feature_set={args.feature_set} "
        f"n_features={len(features)}"
    )
    train_map = load_pvgis_years(args.pvgis_dir, train_years, file_template=args.file_template)
    test_path = f"{args.pvgis_dir}/{args.file_template.format(year=args.test_year)}"
    test_ds = load_pvgis_year(test_path)
    test_event_scores = (
        load_anomaly_labels(args.anomaly_scores, source=args.anomaly_source)
        if args.anomaly_scores
        and (args.train_normal_only or args.anomaly_source == "detector")
        else None
    )
    train_scores = (
        load_anomaly_labels(
            args.train_anomaly_scores, source=args.anomaly_source
        )
        if args.train_anomaly_scores
        else None
    )

    try:
        print(f"[2/6] Building datasets (features={features})")
        t0 = time.perf_counter()
        built = build_datasets(
            train_map, test_ds, args.seq_len, args.horizon, args.target_variable,
            feature_names=features,
            pv_target_clip_max=args.pv_target_clip_max,
            validation_year=args.validation_year,
            kt_poa_max=float(args.kt_poa_max),
            train_normal_only=bool(args.train_normal_only),
            train_anomaly_scores=train_scores,
            test_anomaly_scores=test_event_scores,
            anomaly_source=args.anomaly_source,
            event_spatial_quantile=float(args.event_spatial_quantile),
            event_tail_quantile=float(args.event_tail_quantile),
            detector_regional_quantile=float(
                args.detector_regional_quantile
            ),
            detector_min_temporal_coverage=float(
                args.detector_min_temporal_coverage
            ),
        )
        # The aggregated TRAIN anomaly CSV can contain millions of rows. All
        # required regional labels/thresholds now live in `built`.
        train_scores = None
        test_event_scores = None
        built["test"].subsample(args.max_test_samples, seed=args.seed)
        print(
            f"      nodes={len(built['loc_ids'])}  n_features={built['n_features']}  "
            f"train_windows={len(built['train'])}  "
            f"validation_windows={len(built['validation'])} "
            f"(year={built['validation_year']})  "
            f"test_windows={len(built['test'])}"
        )
        if args.train_normal_only:
            train_filter = built["event_filter_stats"]["train"]
            val_filter = built["event_filter_stats"]["validation"]
            thresholds = (
                built["event_protocol"]["seasonal_thresholds"]
                if args.anomaly_source == "detector"
                else built["event_protocol"]["thresholds"]
            )
            print(
                "      paper-style regional event filter: "
                f"train={train_filter['after']}/{train_filter['before']} windows; "
                f"validation={val_filter['after']}/{val_filter['before']} windows; "
                f"normalization_timestamps="
                f"{built['normalization']['fit_timestamp_count']}"
            )
            threshold_label = (
                "detector seasonal regional thresholds"
                if args.anomaly_source == "detector"
                else "regional thresholds"
            )
            print(
                f"      {threshold_label}: "
                + ", ".join(
                    f"{name}={value:.6g}"
                    for name, value in thresholds.items()
                )
            )
            coverage = built["event_protocol"].get("coverage_by_year")
            if coverage:
                print(
                    "      detector score coverage: "
                    + ", ".join(
                        f"{year}={values['temporal']:.3%}"
                        for year, values in coverage.items()
                    )
                )
            inactive = built["event_protocol"]["inactive_variables"]
            if inactive:
                print(
                    "      inactive regional variables (q threshold=0): "
                    + ", ".join(inactive)
                )
        print(f"      [time] building datasets: {time.perf_counter() - t0:.1f}s")

        print(f"[3/6] Building graph (max_dist_km={args.max_dist_km})")
        edge_index, edge_weight = build_graph(
            built["lats"],
            built["lons"],
            max_dist_km=args.max_dist_km,
            distance_scale_km=args.distance_scale_km,
        )
        print(f"      edges={edge_index.shape[1]}")

        print(
            f"[4/6] Training STGNN+SDE "
            f"(n_features={built['n_features']}, "
            f"dropout={args.dropout}, epochs={args.epochs}, device={args.device})"
        )
        print(
            f"[model] use_irradiance_head={bool(args.use_irradiance_head)}  "
            f"use_irradiance_loss={bool(args.use_irradiance_loss)}  "
            f"irradiance_loss_weight={float(args.irradiance_loss_weight)}"
        )
        print(
            f"[model] n_sde_steps={int(args.n_sde_steps)}  "
            f"sigma_max={float(args.sigma_max)}  "
            f"ood_noise_std={float(args.ood_noise_std)}  "
            f"lr_g={args.lr_g if args.lr_g is not None else args.lr}  "
            f"train_normal_only={bool(args.train_normal_only)}"
        )
        model = make_model(
            len(built["loc_ids"]), args.seq_len, built["n_features"],
            dropout=args.dropout,
            n_sde_steps=int(args.n_sde_steps),
            sigma_max=float(args.sigma_max),
            use_irradiance_head=bool(args.use_irradiance_head),
            kt_poa_max=float(args.kt_poa_max),
            edge_prior_strength=float(args.edge_prior_strength),
        )
        if args.max_train_samples is not None:
            before_subsample = len(built["train"])
            built["train"].subsample(args.max_train_samples, seed=args.seed)
            print(
                f"  [stgnn] max-train-samples: "
                f"kept {len(built['train'])}/{before_subsample} windows"
            )
        t_train = time.perf_counter()
        model = train_model(
            model, built["train"], built["validation"], edge_index, edge_weight,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device,
            use_irradiance_loss=bool(args.use_irradiance_loss),
            irradiance_loss_weight=float(args.irradiance_loss_weight),
            ood_noise_std=float(args.ood_noise_std),
            lr_g=args.lr_g,
            feature_names=features,
            train_normal_only=bool(args.train_normal_only),
            validation_metric=args.validation_metric,
            early_stopping_patience=int(args.early_stopping_patience),
            early_stopping_min_delta=float(args.early_stopping_min_delta),
        )
        print(f"      [time] training total: {time.perf_counter() - t_train:.1f}s")
        checkpoint_path = Path(out_dir) / "best_model.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_payload = {
            "format_version": 1,
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "best_epoch": int(model.best_epoch),
            "best_validation_metric": model.best_validation_metric,
            "best_validation_score": float(model.best_validation_score),
            "train_years": list(built["train_years"]),
            "validation_year": int(built["validation_year"]),
            "test_year": int(args.test_year),
            "location_ids": [str(value) for value in built["loc_ids"]],
            "latitudes": np.asarray(built["lats"], dtype=float),
            "longitudes": np.asarray(built["lons"], dtype=float),
            "feature_names": list(built["features"]),
            "normalization": built["normalization"],
            "event_protocol": (
                {
                    "source": built["event_protocol"].get("source", "climatology"),
                    "detector": built["event_protocol"].get("detector"),
                    "spatial_quantile": built["event_protocol"]["spatial_quantile"],
                    "event_quantile": built["event_protocol"]["event_quantile"],
                    "regional_quantile": built["event_protocol"].get(
                        "regional_quantile"
                    ),
                    "seasonal_thresholds": built["event_protocol"].get(
                        "seasonal_thresholds"
                    ),
                    "min_temporal_coverage": built["event_protocol"].get(
                        "min_temporal_coverage"
                    ),
                    "coverage_by_year": built["event_protocol"].get(
                        "coverage_by_year"
                    ),
                    "variables": built["event_protocol"]["variables"],
                    "inactive_variables": built["event_protocol"][
                        "inactive_variables"
                    ],
                    "thresholds": built["event_protocol"]["thresholds"],
                    "filter_stats": built["event_filter_stats"],
                }
                if built["event_protocol"] is not None
                else None
            ),
            "pv_target_clip_max": built["pv_target_clip_max"],
            "graph": {
                "edge_index": edge_index.cpu(),
                "edge_weight": edge_weight.cpu(),
                "max_dist_km": float(args.max_dist_km),
                "distance_scale_km": (
                    float(args.distance_scale_km)
                    if args.distance_scale_km is not None
                    else float(args.max_dist_km) / 2.0
                ),
                "edge_prior_strength": float(args.edge_prior_strength),
            },
            "model_config": {
                "n_nodes": len(built["loc_ids"]),
                "n_features": built["n_features"],
                "seq_len": int(args.seq_len),
                "horizon": int(args.horizon),
                "dropout": float(args.dropout),
                "d_model": 128,
                "gat_dim": 96,
                "gat_heads": 4,
                "gat_layers": 1,
                "bilstm_pooling": "attn",
                "n_sde_steps": int(args.n_sde_steps),
                "sigma_max": float(args.sigma_max),
                "kt_poa_max": float(args.kt_poa_max),
                "edge_prior_strength": float(args.edge_prior_strength),
                "use_irradiance_head": bool(args.use_irradiance_head),
            },
            "training_config": dict(vars(args)),
        }
        torch.save(checkpoint_payload, checkpoint_path)
        print(
            f"      best checkpoint: epoch={model.best_epoch + 1}, "
            f"{model.best_validation_metric}={model.best_validation_score:.6f} "
            f"-> {checkpoint_path}"
        )
        # Per-epoch loss components (loss/pv, loss/irradiance, loss/total) -> W&B.
        train_history = getattr(model, "train_loss_history", None)
        if wandb_run is not None and train_history:
            for ep_i, rec in enumerate(train_history, start=1):
                wandb_run.log({"epoch": ep_i, **rec})

        print("[5/6] Predicting on test year + attaching anomaly labels")
        t_test = time.perf_counter()
        if args.sde_uncertainty:
            print(
                f"      SDE sampling: eval() + {args.mc_samples} stochastic "
                f"Brownian-path forward passes/batch"
            )
            predictions = predict_sde(
                model, built["test"], edge_index, edge_weight,
                args.device, args.batch_size, mc_samples=args.mc_samples,
                coverage_target=args.coverage_target,
            )
        else:
            predictions = predict(
                model, built["test"], edge_index, edge_weight, args.device, args.batch_size
            )
        print(f"      [time] test inference: {time.perf_counter() - t_test:.1f}s")
        anomaly_scores = load_anomaly_labels(
            args.anomaly_scores, source=args.anomaly_source
        )
        predictions = attach_anomaly_labels(predictions, anomaly_scores)
        predictions = attach_event_labels(predictions, built["test_event_labels"])
        global_df, by_df = compute_metrics(predictions)

        # Interval reliability/sharpness (PICP/MPIW/NMPIL/CLC). Eval-only; needs
        # SDE sample intervals. A single global target_range normalises NMPIL.
        interval_metrics = None
        daytime_metrics = None
        residual_bias_metrics = None
        target_range = None
        clc_gamma = float(args.coverage_target)
        if args.sde_uncertainty and {"lower_pi", "upper_pi"} <= set(predictions.columns):
            eps = 1e-6
            y_true_test = predictions["y_true"].to_numpy(dtype=float)
            target_range = float(np.nanmax(y_true_test) - np.nanmin(y_true_test))
            if not np.isfinite(target_range) or target_range < eps:
                target_range = eps
            interval_metrics = build_interval_metrics(
                predictions, target_range, clc_gamma, args.clc_eta
            )
            daytime_metrics = build_daytime_metrics(
                predictions,
                target_range,
                clc_gamma,
                args.clc_eta,
                threshold_wm2=DAYTIME_IRRADIANCE_THRESHOLD_WM2,
            )
            residual_bias_metrics = build_residual_bias_metrics(
                predictions,
                target_range,
                clc_gamma,
                args.clc_eta,
                threshold_wm2=DAYTIME_IRRADIANCE_THRESHOLD_WM2,
            )
            print(
                f"[interval] reliability/sharpness  target_range={target_range:.4f}  "
                f"gamma={clc_gamma}  eta={args.clc_eta}"
            )
            for kind in ("pi", "gaussian"):
                gm = interval_metrics.get(kind, {}).get("global")
                if gm:
                    tag = "PRIMARY" if kind == "pi" else "diag"
                    print(
                        f"      {kind:11s}[{tag}] global  PICP={gm['picp']:.3f}  "
                        f"MPIW={gm['mpiw']:.4f}  NMPIL={gm['nmpil']:.4f}  CLC={gm['clc']:.4f}"
                    )
            day = daytime_metrics["daytime"]
            night = daytime_metrics["nighttime"]
            print(
                "[interval] daytime diagnostic  "
                f"threshold={DAYTIME_IRRADIANCE_THRESHOLD_WM2:.1f} W/m2  "
                f"counts(day/night)={day['count']}/{night['count']}  "
                f"PICP_PI(day/night)={day['picp_pi']:.3f}/{night['picp_pi']:.3f}"
            )

        print(f"[6/6] Writing outputs to {out_dir}")
        meta = build_meta(
            {
                "mode": "pvgis_stgnn",
                "model_type": args.model_type,
                "feature_set": args.feature_set,
                "target_variable": args.target_variable,
                "pv_target_clip_max": args.pv_target_clip_max,
                "seq_len": args.seq_len,
                "horizon": args.horizon,
                "train_years": args.train_years,
                "effective_train_years": list(built["train_years"]),
                "validation_year": int(built["validation_year"]),
                "test_year": args.test_year,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "n_sde_steps": int(args.n_sde_steps),
                "sigma_max": float(args.sigma_max),
                "ood_noise_std": float(args.ood_noise_std),
                "lr_g": args.lr_g,
                "train_normal_only": bool(args.train_normal_only),
                "anomaly_source": args.anomaly_source,
                "event_spatial_quantile": float(args.event_spatial_quantile),
                "event_tail_quantile": float(args.event_tail_quantile),
                "detector_regional_quantile": float(
                    args.detector_regional_quantile
                ),
                "detector_min_temporal_coverage": float(
                    args.detector_min_temporal_coverage
                ),
                "event_protocol": (
                    {
                        "source": built["event_protocol"].get(
                            "source", "climatology"
                        ),
                        "detector": built["event_protocol"].get("detector"),
                        "regional_quantile": built["event_protocol"].get(
                            "regional_quantile"
                        ),
                        "seasonal_thresholds": built["event_protocol"].get(
                            "seasonal_thresholds"
                        ),
                        "coverage_by_year": built["event_protocol"].get(
                            "coverage_by_year"
                        ),
                        "variables": built["event_protocol"]["variables"],
                        "inactive_variables": built["event_protocol"][
                            "inactive_variables"
                        ],
                        "thresholds": built["event_protocol"]["thresholds"],
                        "filter_stats": built["event_filter_stats"],
                        "normalization_timestamp_count": built["normalization"][
                            "fit_timestamp_count"
                        ],
                    }
                    if built["event_protocol"] is not None
                    else None
                ),
                "kt_poa_max": float(args.kt_poa_max),
                "distance_scale_km": args.distance_scale_km,
                "edge_prior_strength": float(args.edge_prior_strength),
                "validation_metric": args.validation_metric,
                "early_stopping_patience": int(args.early_stopping_patience),
                "early_stopping_min_delta": float(args.early_stopping_min_delta),
                "best_epoch": int(model.best_epoch),
                "best_validation_score": float(model.best_validation_score),
                "use_irradiance_head": bool(args.use_irradiance_head),
                "use_irradiance_loss": bool(args.use_irradiance_loss),
                "irradiance_loss_weight": float(args.irradiance_loss_weight),
                "anomaly_scores": args.anomaly_scores,
                "device": args.device,
                "wandb_enabled": bool(args.wandb),
                "sde_uncertainty": bool(args.sde_uncertainty),
                "mc_samples": args.mc_samples,
                "coverage_target": args.coverage_target,
                "skip_predictions_csv": args.skip_predictions_csv,
                "wandb_artifacts_uploaded": False,
                "posthoc_executed": False,
                "posthoc_uploaded": False,
            },
            n_predictions=len(predictions),
            n_nodes=len(built["loc_ids"]),
            features=features,
        )
        if interval_metrics is not None:
            meta["interval_metrics"] = interval_metrics
            meta["clc_eta"] = float(args.clc_eta)
            meta["clc_gamma"] = clc_gamma
            meta["target_range"] = target_range
        if daytime_metrics is not None:
            meta["daytime_metrics"] = daytime_metrics
            meta["daytime_threshold_wm2"] = DAYTIME_IRRADIANCE_THRESHOLD_WM2
        if residual_bias_metrics is not None:
            meta["residual_bias_metrics"] = residual_bias_metrics
        t_write = time.perf_counter()
        paths = write_outputs(
            predictions, global_df, by_df, out_dir, meta,
            skip_predictions=args.skip_predictions_csv,
        )
        paths["checkpoint"] = checkpoint_path
        # metrics.json: machine-readable global + per-stratum + interval metrics.
        metrics_payload = {
            "global": global_df.iloc[0].to_dict(),
            "by_stratum": by_df.to_dict(orient="records"),
            "interval_metrics": interval_metrics,
            "daytime_metrics": daytime_metrics,
            "residual_bias_metrics": residual_bias_metrics,
            "daytime_threshold_wm2": DAYTIME_IRRADIANCE_THRESHOLD_WM2,
            "clc_eta": float(args.clc_eta),
            "clc_gamma": clc_gamma,
            "target_range": target_range,
        }
        metrics_json_path = Path(out_dir) / "metrics.json"
        metrics_json_path.write_text(
            json.dumps(metrics_payload, indent=2, default=str), encoding="utf-8"
        )
        paths["metrics_json"] = metrics_json_path
        print(f"      metrics.json -> {metrics_json_path}")
        print(f"      [time] writing outputs: {time.perf_counter() - t_write:.1f}s")

        summary = build_wandb_metrics(
            global_df, by_df, sde_uncertainty=bool(args.sde_uncertainty)
        )
        if interval_metrics is not None:
            summary.update(flatten_interval_metrics(interval_metrics))
        if daytime_metrics is not None:
            summary.update(flatten_daytime_metrics(daytime_metrics))
            summary["evaluation/daytime_threshold_wm2"] = (
                DAYTIME_IRRADIANCE_THRESHOLD_WM2
            )
        if residual_bias_metrics is not None:
            residual_summary = flatten_residual_bias_metrics(
                residual_bias_metrics
            )
            collisions = set(summary) & set(residual_summary)
            if collisions:
                raise RuntimeError(
                    "Residual diagnostics would overwrite existing W&B metrics: "
                    f"{sorted(collisions)}"
                )
            summary.update(residual_summary)
        summary.update(
            {
                "best_epoch": int(model.best_epoch + 1),
                "best_validation_score": float(model.best_validation_score),
            }
        )
        if wandb_run is not None:
            wandb_run.log(summary)
            wandb_run.summary.update(summary)
            wandb_run.summary["best_validation_metric"] = (
                model.best_validation_metric
            )

        meta["wandb_artifacts_uploaded"] = upload_artifacts
        write_report(paths["report"], global_df, by_df, meta)

        main_artifact_uploaded = False
        if wandb_run is not None and upload_artifacts:
            main_artifact_uploaded = _log_wandb_artifact(
                wandb, wandb_run, paths, predictions_upload
            )
        elif wandb_run is not None:
            print("[wandb] artifact upload disabled (--no-wandb-upload-artifacts)")

        if main_artifact_uploaded != meta["wandb_artifacts_uploaded"]:
            meta["wandb_artifacts_uploaded"] = main_artifact_uploaded
            write_report(paths["report"], global_df, by_df, meta)

        print(f"\nDone. [time] total run: {time.perf_counter() - t_run_start:.1f}s")
        print(global_df.to_string(index=False))
        if not by_df.empty:
            print(by_df.to_string(index=False))
        print("\nKey metrics:")
        for k in sorted(summary):
            print(f"  {k:32s} {summary[k]:.4f}")
        for key in (
            "predictions",
            "metrics_global",
            "metrics_by_anomaly_label",
            "metrics_daytime",
            "residual_bias_metrics",
            "report",
        ):
            print(f"  {key:24s} -> {paths[key] if paths[key] is not None else '(skipped)'}")
        return paths
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        test_ds.close()
        for ds in train_map.values():
            ds.close()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_from_args(args, parser=parser)


if __name__ == "__main__":
    main()
