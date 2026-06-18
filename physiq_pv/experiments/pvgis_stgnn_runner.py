"""
PVGIS-only ST-GNN experiment runner.

Single source of truth for both entrypoints:
  * `python main.py --mode pvgis_stgnn ...`
  * `python scripts/run_pvgis_stgnn_forecasting.py ...` (thin wrapper)

Built for sweep / ablation / future uncertainty work:
  * `--feature-set`   selects a subset of the 11 PVGIS-only features; the model
                      is instantiated with STGNN(n_features=len(selected)).
  * `--model-type`    stgnn | lstm (implemented). lstm is a simple per-node
                      temporal baseline (no graph / adjacency / message passing)
                      on the SAME dataset, windowing, normalisation and metrics.
                      persistence / mlp are scaffolded but raise a clean
                      "not implemented yet" — never run silently.
  * `--mc-dropout`    Monte Carlo Dropout: model.eval() + reactivate only the
                      nn.Dropout layers + `--mc-samples` forward passes ->
                      y_pred_mean/std and a ~95% band. Adds uncertainty metrics
                      (mean/median/p90 std, coverage_95) per anomaly stratum.
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
    DAYTIME_IRRADIANCE_THRESHOLD_WM2,
    DEFAULT_TARGET_VARIABLE,
    FEATURE_SETS,
    SPECIFIC_ANOMALY_LABELS,
    attach_anomaly_labels,
    build_datasets,
    build_meta,
    build_wandb_metrics,
    compute_metrics,
    load_anomaly_labels,
    load_pvgis_year,
    load_pvgis_years,
    make_model,
    resolve_feature_set,
    write_outputs,
    write_report,
)
from physiq_pv.training.train_loop import train_model
from physiq_pv.training.uncertainty import (
    CALIBRATION_STRATEGIES,
    apply_mc_uncertainty_calibration_stratified,
    estimate_mc_calibration_factors,
    predict,
    predict_mc,
)
from physiq_pv.model.graph_builder import build_graph
from physiq_pv.model.lstm_baseline import LSTMBaseline

# Model-type registry. "stgnn", "stgnn_enhanced_dropout" and "lstm" are
# implemented; the rest are scaffolded so the dispatch is ready, but they fail
# cleanly instead of running silently. "lstm" is the no-graph temporal baseline:
# same dataset/windowing/normalisation/metrics as stgnn, no adjacency, no
# message passing. "stgnn_enhanced_dropout" is the Enhanced MC Dropout ablation:
# the SAME STGNN plus explicit nn.Dropout modules (after the BiLSTM temporal
# embedding, after the projection, inside the pv head) so enable_dropout_only()
# reactivates more than the single GAT attention dropout at MC inference.
# Dataset, splits, target, MSE loss, metrics and the anomaly-labels-eval-only
# protocol are unchanged.
SUPPORTED_MODEL_TYPES = ("stgnn", "stgnn_enhanced_dropout", "lstm", "persistence", "mlp")
IMPLEMENTED_MODEL_TYPES = ("stgnn", "stgnn_enhanced_dropout", "lstm")

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


# Post-hoc daytime-bins x anomaly files (produced by the standalone analysis script).
_POSTHOC_FILES = (
    "daytime_bin_anomaly_report.md",
    "daytime_anomaly_overview.csv",
    "daytime_bin_anomaly_counts.csv",
    "daytime_bin_anomaly_metrics.csv",
    "uncertainty_error_growth_by_anomaly.csv",
)


def _run_posthoc_daytime_bins(predictions_path, posthoc_dir):
    """Run scripts/analyze_pvgis_daytime_bins_by_anomaly.py as a subprocess.

    Eval-only post-processing: never touches the model / loss / training. Returns
    the posthoc dir on success, None on any failure (never raises — the main
    report must survive a post-hoc error).
    """
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    if predictions_path is None or not Path(predictions_path).exists():
        print(f"[posthoc] skipped: predictions.csv not found ({predictions_path})")
        return None
    script = Path(__file__).resolve().parents[2] / "scripts" / \
        "analyze_pvgis_daytime_bins_by_anomaly.py"
    if not script.exists():
        print(f"[posthoc] skipped: script not found ({script})")
        return None
    cmd = [sys.executable, str(script), "--predictions", str(predictions_path),
           "--out-dir", str(posthoc_dir)]
    print(f"[posthoc] running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:  # noqa: BLE001 — post-hoc must never kill the run
        print(f"[posthoc] failed, main report still available: {e}")
        return None
    print(f"[posthoc] done -> {posthoc_dir}")
    return posthoc_dir


def _log_wandb_posthoc_artifact(wandb, wandb_run, posthoc_dir) -> bool:
    """Attach the post-hoc analysis files as a SEPARATE W&B artifact.

    Skips any missing file with a warning; logs nothing if none are present.
    """
    try:
        art = wandb.Artifact(
            f"pvgis-stgnn-posthoc-{wandb_run.id}",
            type="posthoc-analysis",
        )
    except Exception as exc:  # noqa: BLE001 - post-hoc upload is non-fatal
        print(f"[wandb-artifact] failed to create post-hoc artifact: {exc}")
        return False

    added = 0
    posthoc_path = Path(posthoc_dir)
    expected = [posthoc_path / name for name in _POSTHOC_FILES]
    discovered = sorted(posthoc_path.glob("*.csv")) + sorted(posthoc_path.glob("*.md"))
    seen = set()
    for p in expected + discovered:
        resolved = str(p.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        if p.exists():
            try:
                art.add_file(str(p))
            except Exception as exc:  # noqa: BLE001 - skip one bad file
                print(f"[wandb-artifact] skipped file {p}: {exc}")
                continue
            added += 1
        else:
            print(f"[wandb-artifact] skipped missing file: {p}")
    if added == 0:
        print("[wandb-artifact] no post-hoc files found; post-hoc artifact not logged")
        return False
    try:
        wandb_run.log_artifact(art)
    except Exception as exc:  # noqa: BLE001 - main artifact must still be attempted
        print(f"[wandb-artifact] failed to upload post-hoc artifact: {exc}")
        return False
    print(f"[wandb] logged artifact {art.name} ({added} file(s))")
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
#   pi         = PRIMARY paper-style interval: empirical MC-sample quantiles.
#   gaussian   = diagnostic Gaussian band (mean ± 1.96*std_raw).
#   calibrated = post-hoc calibrated band, present only when the opt-in
#                --enable-posthoc-calibration flag is set (diagnostic).
_INTERVAL_KINDS = {
    "pi": ("lower_pi", "upper_pi"),
    "gaussian": ("lower_gaussian", "upper_gaussian"),
    "calibrated": ("lower_calibrated", "upper_calibrated"),
}
# Eval strata. "normal"/"rare_extreme" map to the anomaly_group values; anomaly
# labels are used ONLY here for stratified eval, never as model input or target.
_INTERVAL_GROUPS = {
    "global": None,
    "normal": "normal",
    "rare_extreme": "rare_or_extreme",
}


def build_interval_metrics(
    predictions, target_range: float, gamma: float, eta: float
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Interval metrics for every available kind (raw/calibrated) x stratum."""
    cols = set(predictions.columns)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for kind, (lo, hi) in _INTERVAL_KINDS.items():
        if not {lo, hi} <= cols:
            continue
        out[kind] = {}
        for gname, group_val in _INTERVAL_GROUPS.items():
            sub = (
                predictions if group_val is None
                else predictions[predictions["anomaly_group"] == group_val]
            )
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
_DAYTIME_PRODUCTION_BINS = (
    ("daytime_0_20", 0.0, 20.0),
    ("daytime_20_40", 20.0, 40.0),
    ("daytime_40_60", 40.0, 60.0),
    ("daytime_60_80", 60.0, 80.0),
    ("daytime_80_100", 80.0, 100.0),
    ("daytime_gt_100", 100.0, None),
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
            "Daytime diagnostics require point predictions and MC standard "
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

    for name, lower, upper in _DAYTIME_PRODUCTION_BINS:
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
    g.add_argument("--test-year", "--test_year", type=int, default=None)
    g.add_argument("--anomaly-scores", "--anomaly_scores", default=None,
                   help="pvgis_climatology_scores.csv (stratified eval only; never model input).")
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
        default=1.5,
        help="Upper clip for normalized PV target and pv_lag_pvgis. "
        "Default 1.5 preserves existing behavior; none/null keeps only "
        "the lower non-negativity clip.",
    )
    g.add_argument("--epochs", type=int, default=10)
    g.add_argument("--batch-size", "--batch_size", type=int, default=8)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--max-dist-km", "--max_dist_km", type=float, default=20.0)
    g.add_argument("--max-train-samples", "--max_train_samples", type=int, default=None)
    g.add_argument("--max-test-samples", "--max_test_samples", type=int, default=None)
    g.add_argument("--max-calibration-samples", "--max_calibration_samples",
                   type=int, default=None,
                   help="Randomly subsample calibration windows to at most N "
                        "(speeds up MC calibration; eval/test untouched).")
    g.add_argument("--skip-predictions-csv", "--skip_predictions_csv",
                   action="store_true",
                   help="Do not write the (large) predictions.csv; metrics + "
                        "report.md are still produced.")
    g.add_argument("--file-template", "--file_template", default="piedmont_pvgis_{year}.nc")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--seed", type=int, default=42)
    # Model / ablation
    g.add_argument("--model-type", "--model_type", default="stgnn", choices=SUPPORTED_MODEL_TYPES,
                   help="stgnn | stgnn_enhanced_dropout | lstm (implemented; "
                        "stgnn_enhanced_dropout = same STGNN + explicit nn.Dropout on "
                        "temporal embedding / projected representation / pv head for "
                        "real MC-Dropout stochasticity; lstm = no-graph temporal "
                        "baseline); persistence/mlp scaffolded (not implemented yet).")
    g.add_argument("--hidden-size", "--hidden_size", type=int, default=64,
                   help="LSTM baseline hidden size (model_type=lstm only).")
    g.add_argument("--lstm-layers", "--lstm_layers", type=int, default=2,
                   help="LSTM baseline number of layers (model_type=lstm only).")
    g.add_argument("--feature-set", "--feature_set", default="full", choices=sorted(FEATURE_SETS),
                   help="Feature ablation; n_features = len(selected features).")
    g.add_argument("--dropout", type=float, default=0.2,
                   help="STGNN dropout (also the basis for MC Dropout sampling).")
    # Training point-loss ablation. Isolated knob: only the loss module changes.
    g.add_argument("--loss-type", "--loss_type", default="mse",
                   choices=("mse", "huber"),
                   help="Training point loss: mse (default, historical baseline) or "
                        "huber (torch.nn.HuberLoss, robust to large residuals). "
                        "Also governs the auxiliary kt loss when enabled.")
    g.add_argument("--huber-delta", "--huber_delta", type=float, default=1.0,
                   help="HuberLoss delta (transition |err| where quadratic -> linear). "
                        "Only used with --loss-type huber. NOTE: the PV target is "
                        "normalised (~p99 scale), so a delta near the residual scale "
                        "(~0.1) is where Huber departs from MSE; default 1.0 behaves "
                        "close to MSE on this target.")
    # Train-time MC under-dispersion penalty (opt-in). Penalises high error +
    # low MC-Dropout std. Anomaly labels are NOT involved. Default off = baseline.
    g.add_argument("--train-mc-uncertainty-penalty", "--train_mc_uncertainty_penalty",
                   action="store_true",
                   help="Enable the train-time under-dispersion penalty: per batch, "
                        "run --train-mc-samples stochastic passes, compute MC mean+std "
                        "and add lambda_under*relu(|err|-k*std)^2 + lambda_std*std^2 to "
                        "the PV loss. Needs --train-mc-samples >= 2 and --dropout > 0. "
                        "Default off -> training identical to baseline.")
    g.add_argument("--train-mc-samples", "--train_mc_samples", type=int, default=1,
                   help="Stochastic forward passes per batch when the penalty is on "
                        "(>= 2 required then; 5 is a cost/quality compromise). "
                        "Ignored when the penalty is off.")
    g.add_argument("--uncertainty-penalty-weight", "--uncertainty_penalty_weight",
                   type=float, default=0.0,
                   help="lambda_under: weight of relu(|err|-k*std)^2 (default 0.0).")
    g.add_argument("--uncertainty-penalty-k", "--uncertainty_penalty_k",
                   type=float, default=1.0,
                   help="k in relu(|err|-k*std)^2: how many std the error may exceed "
                        "before being penalised (default 1.0).")
    g.add_argument("--uncertainty-std-reg-weight", "--uncertainty_std_reg_weight",
                   type=float, default=0.0,
                   help="lambda_std: weight of std^2 regularisation that stops the "
                        "model inflating uncertainty everywhere (default 0.0).")
    g.add_argument("--uncertainty-penalty-mode", "--uncertainty_penalty_mode",
                   default="underdispersion", choices=("underdispersion", "sde_proxy"),
                   help="underdispersion (default): relu(|err|-k*std)^2. sde_proxy "
                        "(SDE-Net style): minimise std on normal/in-distribution "
                        "cells, keep std above --sde-proxy-std-min-ood on "
                        "rare_or_extreme/OOD cells. sde_proxy needs "
                        "--train-mc-uncertainty-penalty, --train-mc-samples>=2 and "
                        "--train-noise-mode anomaly (uses the anomaly mask).")
    g.add_argument("--sde-proxy-in-weight", "--sde_proxy_in_weight",
                   type=float, default=0.001,
                   help="sde_proxy: weight of mean(std_normal^2) (in-distribution).")
    g.add_argument("--sde-proxy-out-weight", "--sde_proxy_out_weight",
                   type=float, default=0.1,
                   help="sde_proxy: weight of mean(relu(std_min_ood - std_ood)^2).")
    g.add_argument("--sde-proxy-std-min-ood", "--sde_proxy_std_min_ood",
                   type=float, default=0.05,
                   help="sde_proxy: desired minimum MC std on OOD/anomalous cells, "
                        "in NORMALISED target units (initial value, not definitive).")
    # Train-time input noise injection (opt-in). Train only; targets untouched.
    g.add_argument("--train-noise-std", "--train_noise_std", type=float, default=0.0,
                   help="Std of Gaussian noise added to continuous feature channels "
                        "during training (sin_elev/cos_elev excluded). 0.0 -> off. "
                        "Features are normalised (~unit scale), so ~0.01 is a "
                        "conservative first value.")
    g.add_argument("--train-noise-prob", "--train_noise_prob", type=float, default=0.0,
                   help="Per-sample probability of applying input noise during "
                        "training. 0.0 -> off. Random-mode noise is active only "
                        "when both --train-noise-std > 0 and --train-noise-prob > 0.")
    # Anomaly-aware noise injection (opt-in). Couples TRAINING to anomaly labels.
    g.add_argument("--train-noise-mode", "--train_noise_mode",
                   default="random", choices=("random", "anomaly"),
                   help="random (default): noise on a random fraction of train "
                        "samples (anomaly labels NOT used). anomaly: stronger / "
                        "higher-probability noise on rare_or_extreme samples "
                        "(uses --anomaly-noise-std/--anomaly-noise-prob; normal "
                        "samples keep --train-noise-std/--train-noise-prob). "
                        "anomaly mode REQUIRES anomaly scores covering the train "
                        "years (--train-anomaly-scores, else --anomaly-scores) and "
                        "is NO LONGER an eval-only-labels configuration.")
    g.add_argument("--anomaly-noise-std", "--anomaly_noise_std", type=float, default=0.0,
                   help="Noise std for anomalous (rare_or_extreme) train samples "
                        "(train-noise-mode=anomaly only). Must be > 0 in that mode.")
    g.add_argument("--anomaly-noise-prob", "--anomaly_noise_prob", type=float, default=0.0,
                   help="Per-sample noise probability for anomalous train samples "
                        "(train-noise-mode=anomaly only). Must be in (0, 1] there.")
    g.add_argument("--train-anomaly-scores", "--train_anomaly_scores", default=None,
                   help="Anomaly scores CSV covering the TRAINING years (same "
                        "format as --anomaly-scores: location,timestamp,label). "
                        "Used ONLY by --train-noise-mode anomaly to flag which "
                        "training (location, target_time) cells are rare/extreme. "
                        "Falls back to --anomaly-scores if omitted.")
    # Irradiance ablation. NOTE on the historical behaviour: the STGNN irradiance
    # head (head_ghi) has always been CREATED in this pipeline, but the training
    # loss never supervised it (plain MSE on pred_pv only), so it received no
    # gradient. Hence the defaults: head=True, loss=False == current behaviour.
    g.add_argument("--use-irradiance-head", "--use_irradiance_head",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Create the STGNN irradiance (clear-sky index) head. "
                        "Default True = historical architecture (head present but "
                        "untrained unless --use-irradiance-loss). "
                        "--no-use-irradiance-head -> production-only model.")
    g.add_argument("--use-irradiance-loss", "--use_irradiance_loss",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Add an auxiliary MSE term on the irradiance head "
                        "(pred_kt vs target-time clear-sky index kt) to the "
                        "training loss. Default False = historical behaviour "
                        "(production-only MSE). Requires --use-irradiance-head "
                        "and an STGNN model type.")
    g.add_argument("--irradiance-loss-weight", "--irradiance_loss_weight",
                   type=float, default=1.0,
                   help="Weight of the auxiliary irradiance loss term (only "
                        "meaningful with --use-irradiance-loss).")
    g.add_argument("--kt-aux-loss-weight", "--kt_aux_loss_weight",
                   type=float, default=None,
                   help="Sweep-friendly single-flag interface for the kt auxiliary "
                        "loss: 0.0 -> baseline (identical to no flags), w > 0 -> "
                        "loss = MSE(pred_pv, y) + w * MSE(pred_kt, kt_target). "
                        "Equivalent to --use-irradiance-loss --irradiance-loss-weight w; "
                        "cannot be combined with --use-irradiance-loss.")
    # MC Dropout — eval() + reactivate only nn.Dropout + N passes -> mean/std.
    g.add_argument("--mc-dropout", "--mc_dropout", action="store_true",
                   help="Enable Monte Carlo Dropout uncertainty (needs --dropout > 0).")
    g.add_argument("--mc-samples", "--mc_samples", type=int, default=30,
                   help="Number of MC Dropout forward passes per batch (>= 2).")
    g.add_argument("--clc-eta", "--clc_eta", type=float, default=10.0,
                   help="Sharpness sensitivity eta for the CLC interval metric "
                        "CLC = NMPIL * (1 + exp(-eta * (PICP - gamma))); "
                        "gamma is the coverage target. Eval-only, never affects training.")
    g.add_argument("--enable-posthoc-calibration", "--enable_posthoc_calibration",
                   action="store_true",
                   help="OPT-IN: enable the legacy post-hoc MC-std calibration "
                        "(mean ± k*std). Default OFF — the main paper-style protocol "
                        "builds intervals directly from MC samples (lower_pi/upper_pi) "
                        "with no calibration. All --calibration-* flags require this.")
    g.add_argument("--calibration-years", "--calibration_years", default=None,
                   help="(post-hoc only) Comma-separated calibration years for MC std "
                        "scaling. Requires --enable-posthoc-calibration.")
    g.add_argument("--coverage-target", "--coverage_target", type=float, default=0.95,
                   help="Target coverage quantile for MC std calibration.")
    g.add_argument("--calibration-eps", "--calibration_eps", type=float, default=1e-6,
                   help="Minimum std denominator for MC std calibration ratios.")
    g.add_argument("--calibration-strategy", "--calibration_strategy",
                   default="global", choices=list(CALIBRATION_STRATEGIES),
                   help="MC std calibration: global (one factor), group "
                        "(normal vs rare_or_extreme), or label (per anomaly label).")
    g.add_argument("--calibration-anomaly-scores", "--calibration_anomaly_scores",
                   default=None,
                   help="pvgis_climatology_scores.csv for the calibration year "
                        "(needed by group/label strategies; calibration/eval only).")
    g.add_argument("--min-calibration-samples-per-stratum",
                   "--min_calibration_samples_per_stratum", type=int, default=1000,
                   help="Strata with fewer calibration samples fall back to a "
                        "coarser factor (label -> rare/extreme -> global).")
    # Optional W&B
    g.add_argument("--wandb", action="store_true", help="Enable optional W&B logging.")
    g.add_argument("--wandb-project", "--wandb_project", default="PhysiQ-PV")
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
    g.add_argument("--run-posthoc-analysis", "--run_posthoc_analysis",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="After the run, execute scripts/analyze_pvgis_daytime_bins_by_anomaly.py "
                        "on predictions.csv -> <out_dir>/daytime_bin_anomaly (eval-only, "
                        "needs --mc-dropout). Default True.")
    g.add_argument("--skip-posthoc-analysis", "--skip_posthoc_analysis",
                   action="store_true",
                   help="Force-disable the post-hoc daytime-bins x anomaly analysis.")
    return parser


def build_arg_parser() -> argparse.ArgumentParser:
    """Standalone parser for the script wrapper."""
    p = argparse.ArgumentParser(
        description="PVGIS-only ST-GNN forecasting with stratified eval, feature "
                    "ablation, optional W&B, and MC-Dropout predisposition."
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
    if args.model_type not in IMPLEMENTED_MODEL_TYPES:
        _fail(
            parser,
            f"model_type='{args.model_type}' is not implemented yet. "
            f"Only {list(IMPLEMENTED_MODEL_TYPES)} is available; "
            "persistence/mlp are scaffolded for future work.",
        )
    if args.model_type == "lstm":
        if args.hidden_size < 1:
            _fail(parser, f"--hidden-size must be >= 1, got {args.hidden_size}.")
        if args.lstm_layers < 1:
            _fail(parser, f"--lstm-layers must be >= 1, got {args.lstm_layers}.")
    if args.model_type == "stgnn_enhanced_dropout" and args.dropout <= 0.0:
        _fail(
            parser,
            "model_type=stgnn_enhanced_dropout needs --dropout > 0 (the ablation "
            f"exists to add stochastic capacity; got {args.dropout}).",
        )
    # --kt-aux-loss-weight is pure sugar over the use_irradiance_loss interface:
    # resolve it FIRST so every check below sees the effective configuration.
    if args.kt_aux_loss_weight is not None:
        if args.use_irradiance_loss:
            _fail(
                parser,
                "--kt-aux-loss-weight and --use-irradiance-loss are two interfaces "
                "for the SAME auxiliary loss; pass only one of them.",
            )
        if not np.isfinite(args.kt_aux_loss_weight) or args.kt_aux_loss_weight < 0.0:
            _fail(
                parser,
                "--kt-aux-loss-weight must be finite and >= 0, got "
                f"{args.kt_aux_loss_weight}.",
            )
        if args.kt_aux_loss_weight > 0.0:
            args.use_irradiance_loss = True
            args.irradiance_loss_weight = float(args.kt_aux_loss_weight)
        # 0.0 -> baseline: use_irradiance_loss stays False, weight untouched.
    if args.use_irradiance_loss and not args.use_irradiance_head:
        _fail(
            parser,
            "--use-irradiance-loss requires the irradiance head: it cannot be "
            "combined with --no-use-irradiance-head (there is no head to "
            "supervise). Drop --use-irradiance-loss or re-enable the head.",
        )
    if args.use_irradiance_loss and args.model_type == "lstm":
        _fail(
            parser,
            "--use-irradiance-loss is only supported for STGNN model types "
            "(the LSTM baseline has no irradiance head).",
        )
    if not np.isfinite(args.irradiance_loss_weight) or args.irradiance_loss_weight < 0.0:
        _fail(
            parser,
            "--irradiance-loss-weight must be finite and >= 0, got "
            f"{args.irradiance_loss_weight}.",
        )
    if args.loss_type == "huber" and (
        not np.isfinite(args.huber_delta) or args.huber_delta <= 0.0
    ):
        _fail(
            parser,
            f"--huber-delta must be finite and > 0, got {args.huber_delta}.",
        )
    # Train-time MC under-dispersion penalty.
    if args.train_mc_uncertainty_penalty:
        if args.train_mc_samples < 2:
            _fail(
                parser,
                "--train-mc-uncertainty-penalty needs --train-mc-samples >= 2 "
                f"(a per-target std needs >= 2 MC passes); got {args.train_mc_samples}.",
            )
        if args.dropout <= 0.0:
            _fail(
                parser,
                "--train-mc-uncertainty-penalty needs --dropout > 0 for stochastic "
                f"MC passes (otherwise std is identically 0); got {args.dropout}.",
            )
        for name, val in (
            ("--uncertainty-penalty-weight", args.uncertainty_penalty_weight),
            ("--uncertainty-std-reg-weight", args.uncertainty_std_reg_weight),
            ("--uncertainty-penalty-k", args.uncertainty_penalty_k),
        ):
            if not np.isfinite(val) or val < 0.0:
                _fail(parser, f"{name} must be finite and >= 0, got {val}.")
    elif args.train_mc_samples < 1:
        _fail(parser, f"--train-mc-samples must be >= 1, got {args.train_mc_samples}.")
    # Train-time input noise injection.
    if not np.isfinite(args.train_noise_std) or args.train_noise_std < 0.0:
        _fail(parser, f"--train-noise-std must be finite and >= 0, got {args.train_noise_std}.")
    if not (0.0 <= args.train_noise_prob <= 1.0):
        _fail(parser, f"--train-noise-prob must be in [0, 1], got {args.train_noise_prob}.")
    if not np.isfinite(args.anomaly_noise_std) or args.anomaly_noise_std < 0.0:
        _fail(parser, f"--anomaly-noise-std must be finite and >= 0, got {args.anomaly_noise_std}.")
    if not (0.0 <= args.anomaly_noise_prob <= 1.0):
        _fail(parser, f"--anomaly-noise-prob must be in [0, 1], got {args.anomaly_noise_prob}.")
    sde_proxy = args.uncertainty_penalty_mode == "sde_proxy"
    if args.train_noise_mode == "anomaly":
        # The mask must be consumed by anomaly noise and/or the sde_proxy penalty.
        if not (args.anomaly_noise_std > 0.0 and args.anomaly_noise_prob > 0.0) \
                and not sde_proxy:
            _fail(
                parser,
                "--train-noise-mode anomaly needs --anomaly-noise-std > 0 and "
                f"--anomaly-noise-prob > 0 (got std={args.anomaly_noise_std}, "
                f"prob={args.anomaly_noise_prob}), unless "
                "--uncertainty-penalty-mode sde_proxy consumes the mask.",
            )
        if not (args.train_anomaly_scores or args.anomaly_scores):
            _fail(
                parser,
                "--train-noise-mode anomaly requires anomaly scores covering the "
                "training years: pass --train-anomaly-scores (or --anomaly-scores). "
                "Anomaly mode cannot run without training-year anomaly labels.",
            )
    # SDE-proxy uncertainty penalty.
    if sde_proxy:
        if not args.train_mc_uncertainty_penalty:
            _fail(
                parser,
                "--uncertainty-penalty-mode sde_proxy requires "
                "--train-mc-uncertainty-penalty (it needs the MC std).",
            )
        if args.train_mc_samples < 2:
            _fail(
                parser,
                "--uncertainty-penalty-mode sde_proxy needs --train-mc-samples >= 2; "
                f"got {args.train_mc_samples}.",
            )
        if args.train_noise_mode != "anomaly":
            _fail(
                parser,
                "--uncertainty-penalty-mode sde_proxy is only supported with "
                "--train-noise-mode anomaly for now (it needs the OOD/anomaly "
                "mask). The OOD split comes from the training anomaly labels.",
            )
        for name, val in (
            ("--sde-proxy-in-weight", args.sde_proxy_in_weight),
            ("--sde-proxy-out-weight", args.sde_proxy_out_weight),
            ("--sde-proxy-std-min-ood", args.sde_proxy_std_min_ood),
        ):
            if not np.isfinite(val) or val < 0.0:
                _fail(parser, f"{name} must be finite and >= 0, got {val}.")
    if args.mc_dropout:
        if args.mc_samples < 2:
            _fail(parser, f"--mc-samples must be >= 2 for MC Dropout, got {args.mc_samples}.")
        if args.dropout <= 0.0:
            _fail(
                parser,
                f"--mc-dropout needs --dropout > 0 for stochasticity (got {args.dropout}).",
            )
    # Post-hoc calibration is opt-in. The main paper-style protocol builds
    # predictive intervals directly from the MC samples (lower_pi/upper_pi).
    posthoc_args_set = any((
        bool(args.calibration_years),
        bool(args.calibration_anomaly_scores),
        args.calibration_strategy != "global",
        args.max_calibration_samples is not None,
    ))
    if posthoc_args_set and not args.enable_posthoc_calibration:
        _fail(
            parser,
            "post-hoc calibration flags (--calibration-years / "
            "--calibration-anomaly-scores / --calibration-strategy / "
            "--max-calibration-samples) require --enable-posthoc-calibration. "
            "The default protocol uses MC-sample predictive intervals (no calibration).",
        )
    if args.calibration_years and not args.mc_dropout:
        _fail(parser, "--calibration-years requires --mc-dropout.")
    if not 0.0 < args.coverage_target < 1.0:
        _fail(parser, f"--coverage-target must be in (0, 1), got {args.coverage_target}.")
    if args.calibration_eps <= 0.0:
        _fail(parser, f"--calibration-eps must be > 0, got {args.calibration_eps}.")
    if args.min_calibration_samples_per_stratum < 1:
        _fail(
            parser,
            "--min-calibration-samples-per-stratum must be >= 1, got "
            f"{args.min_calibration_samples_per_stratum}.",
        )
    if args.calibration_strategy != "global":
        if not args.calibration_years:
            _fail(parser, f"--calibration-strategy {args.calibration_strategy} requires --calibration-years.")
        if not args.calibration_anomaly_scores:
            _fail(
                parser,
                f"--calibration-strategy {args.calibration_strategy} requires "
                "--calibration-anomaly-scores (the calibration year's anomaly labels).",
            )
    if args.calibration_anomaly_scores and not args.calibration_years:
        _fail(parser, "--calibration-anomaly-scores requires --calibration-years.")


def run_from_args(
    args: argparse.Namespace, parser: Optional[argparse.ArgumentParser] = None
) -> dict:
    """Run the PVGIS-only ST-GNN experiment. Returns the written-output paths."""
    _validate(args, parser)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # W&B upload + post-hoc plan (deterministic from flags; surfaced in report.md).
    upload_artifacts = bool(args.wandb) and bool(args.wandb_upload_artifacts)
    predictions_upload = bool(args.wandb_log_predictions) or bool(args.wandb_upload_predictions)
    posthoc_enabled = bool(args.run_posthoc_analysis) and not bool(args.skip_posthoc_analysis)
    posthoc_will_run = posthoc_enabled and not bool(args.skip_predictions_csv)

    # Post-hoc calibration is opt-in; only honour calibration years when enabled.
    posthoc = bool(args.enable_posthoc_calibration)
    train_years = _parse_years(args.train_years)
    calibration_years = (
        _parse_years(args.calibration_years)
        if (posthoc and args.calibration_years) else []
    )
    if args.test_year in calibration_years:
        _fail(parser, "--calibration-years must not include --test-year.")
    overlap = sorted(set(train_years) & set(calibration_years))
    if overlap:
        _fail(parser, f"--calibration-years must be separate from train years; overlap={overlap}.")
    features = resolve_feature_set(args.feature_set)

    if args.mc_dropout and not posthoc:
        print(
            "INFO: paper-style protocol — predictive intervals are built directly "
            "from the MC Dropout sample distribution (lower_pi/upper_pi, empirical "
            "quantiles). No post-hoc calibration. The Gaussian band "
            "(mean +/- 1.96*std_raw) is logged only as a secondary diagnostic. "
            "Use --enable-posthoc-calibration to opt into the legacy calibrated band."
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
                "test_year": args.test_year,
                "seq_len": args.seq_len,
                "horizon": args.horizon,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "dropout": args.dropout,
                "loss_type": args.loss_type,
                "huber_delta": float(args.huber_delta),
                "train_mc_uncertainty_penalty": bool(args.train_mc_uncertainty_penalty),
                "train_mc_samples": int(args.train_mc_samples),
                "uncertainty_penalty_mode": args.uncertainty_penalty_mode,
                "uncertainty_penalty_weight": float(args.uncertainty_penalty_weight),
                "uncertainty_penalty_k": float(args.uncertainty_penalty_k),
                "uncertainty_std_reg_weight": float(args.uncertainty_std_reg_weight),
                "sde_proxy_in_weight": float(args.sde_proxy_in_weight),
                "sde_proxy_out_weight": float(args.sde_proxy_out_weight),
                "sde_proxy_std_min_ood": float(args.sde_proxy_std_min_ood),
                "train_noise_std": float(args.train_noise_std),
                "train_noise_prob": float(args.train_noise_prob),
                "train_noise_mode": args.train_noise_mode,
                "anomaly_noise_std": float(args.anomaly_noise_std),
                "anomaly_noise_prob": float(args.anomaly_noise_prob),
                "train_anomaly_scores": args.train_anomaly_scores,
                "use_irradiance_head": bool(args.use_irradiance_head),
                "use_irradiance_loss": bool(args.use_irradiance_loss),
                "irradiance_loss_weight": float(args.irradiance_loss_weight),
                "hidden_size": args.hidden_size,
                "lstm_layers": args.lstm_layers,
                "device": args.device,
                "max_train_samples": args.max_train_samples,
                "max_test_samples": args.max_test_samples,
                "max_calibration_samples": args.max_calibration_samples,
                "skip_predictions_csv": args.skip_predictions_csv,
                "mc_dropout": args.mc_dropout,
                "mc_samples": args.mc_samples,
                "clc_eta": args.clc_eta,
                "enable_posthoc_calibration": bool(args.enable_posthoc_calibration),
                "calibration_years": args.calibration_years if posthoc else None,
                "coverage_target": args.coverage_target,
                "calibration_eps": args.calibration_eps,
                "calibration_strategy": args.calibration_strategy,
                "calibration_anomaly_scores": args.calibration_anomaly_scores,
                "min_calibration_samples_per_stratum": args.min_calibration_samples_per_stratum,
                "anomaly_scores": args.anomaly_scores,
                "wandb_log_predictions": args.wandb_log_predictions,
                "wandb_upload_artifacts": bool(args.wandb_upload_artifacts),
                "wandb_upload_predictions": bool(args.wandb_upload_predictions),
                "run_posthoc_analysis": bool(args.run_posthoc_analysis),
                "skip_posthoc_analysis": bool(args.skip_posthoc_analysis),
            },
        )

    # Per-run output dir: unique under --wandb so sweep runs never collide.
    out_dir = args.out_dir
    if wandb_run is not None:
        out_dir = _resolve_out_dir(out_dir, wandb_run.id, wandb_run.name)
        print(f"[wandb] run id={wandb_run.id} name={wandb_run.name} -> out_dir={out_dir}")

    # Effective config banner — confirms which values the sweep actually injected.
    print(
        "[config] " + "  ".join(
            f"{k}={v}" for k, v in (
                ("seed", args.seed),
                ("batch_size", args.batch_size),
                ("epochs", args.epochs),
                ("mc_samples", args.mc_samples),
                ("calibration_strategy", args.calibration_strategy),
                ("skip_predictions_csv", args.skip_predictions_csv),
                ("max_calibration_samples", args.max_calibration_samples),
                ("train_years", args.train_years),
                ("calibration_years", args.calibration_years),
                ("test_year", args.test_year),
                ("pv_target_clip_max", args.pv_target_clip_max),
                ("loss_type", args.loss_type),
                ("huber_delta", args.huber_delta),
                ("train_mc_uncertainty_penalty", args.train_mc_uncertainty_penalty),
                ("train_mc_samples", args.train_mc_samples),
                ("uncertainty_penalty_weight", args.uncertainty_penalty_weight),
                ("uncertainty_penalty_k", args.uncertainty_penalty_k),
                ("uncertainty_std_reg_weight", args.uncertainty_std_reg_weight),
                ("uncertainty_penalty_mode", args.uncertainty_penalty_mode),
                ("sde_proxy_in_weight", args.sde_proxy_in_weight),
                ("sde_proxy_out_weight", args.sde_proxy_out_weight),
                ("sde_proxy_std_min_ood", args.sde_proxy_std_min_ood),
                ("train_noise_std", args.train_noise_std),
                ("train_noise_prob", args.train_noise_prob),
                ("train_noise_mode", args.train_noise_mode),
                ("anomaly_noise_std", args.anomaly_noise_std),
                ("anomaly_noise_prob", args.anomaly_noise_prob),
                ("use_irradiance_head", args.use_irradiance_head),
                ("use_irradiance_loss", args.use_irradiance_loss),
                ("irradiance_loss_weight", args.irradiance_loss_weight),
                ("out_dir", out_dir),
            )
        )
    )
    t_run_start = time.perf_counter()

    print(
        f"[1/6] Loading PVGIS years "
        f"(train={train_years}, calibration={calibration_years or 'none'}, "
        f"test={args.test_year}) "
        f"| model={args.model_type} feature_set={args.feature_set} "
        f"n_features={len(features)}"
    )
    train_map = load_pvgis_years(args.pvgis_dir, train_years, file_template=args.file_template)
    calibration_map = (
        load_pvgis_years(args.pvgis_dir, calibration_years, file_template=args.file_template)
        if calibration_years else {}
    )
    test_path = f"{args.pvgis_dir}/{args.file_template.format(year=args.test_year)}"
    test_ds = load_pvgis_year(test_path)

    try:
        print(f"[2/6] Building datasets (features={features})")
        t0 = time.perf_counter()
        built = build_datasets(
            train_map, test_ds, args.seq_len, args.horizon, args.target_variable,
            feature_names=features,
            calibration_ds_map=calibration_map,
            pv_target_clip_max=args.pv_target_clip_max,
        )
        built["train"].subsample(args.max_train_samples, seed=args.seed)
        built["test"].subsample(args.max_test_samples, seed=args.seed)
        if built["calibration"] is not None:
            built["calibration"].subsample(args.max_calibration_samples, seed=args.seed)
        calibration_windows = len(built["calibration"]) if built["calibration"] is not None else 0
        print(
            f"      nodes={len(built['loc_ids'])}  n_features={built['n_features']}  "
            f"train_windows={len(built['train'])}  "
            f"calibration_windows={calibration_windows}  test_windows={len(built['test'])}"
        )
        print(f"      [time] building datasets: {time.perf_counter() - t0:.1f}s")

        # Anomaly-aware training noise: attach the per-(sample, node) anomaly mask
        # to the TRAINING dataset (labels target the noise only; never input/target).
        # Fails loud if the scores do not cover any training (location, target_time).
        if args.train_noise_mode == "anomaly":
            train_scores_path = args.train_anomaly_scores or args.anomaly_scores
            print(
                f"[2b/6] Anomaly-aware training noise: loading train-year anomaly "
                f"scores ({train_scores_path})"
            )
            train_anomaly_scores = load_anomaly_labels(train_scores_path)
            n_anom_cells = built["train"].attach_anomaly_mask(train_anomaly_scores)
            total_cells = len(built["train"]) * len(built["loc_ids"])
            if n_anom_cells == 0:
                _fail(
                    parser,
                    "--train-noise-mode anomaly: the anomaly scores "
                    f"({train_scores_path}) do not cover any training "
                    "(location, target_time) cell — there are 0 rare_or_extreme "
                    "training samples to perturb. Provide anomaly scores covering "
                    f"the training years {train_years} via --train-anomaly-scores.",
                )
            frac = n_anom_cells / total_cells if total_cells else float("nan")
            print(
                f"      anomalous training cells: {n_anom_cells:,} / {total_cells:,} "
                f"({frac:.4f})"
            )
            print(
                "[protocol] ANOMALY-AWARE TRAINING: anomaly labels are used during "
                "training to target input noise -> NOT an eval-only-labels run."
            )

        if args.model_type == "lstm":
            # No-graph baseline: the LSTM ignores adjacency entirely. Empty edge
            # tensors keep the shared train/predict/MC helpers' signatures intact.
            print("[3/6] Skipping graph (model_type=lstm)")
            print("[model] model_type=lstm")
            print("[model] graph disabled / no adjacency used")
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty(0, dtype=torch.float32)
        else:
            print(f"[3/6] Building graph (max_dist_km={args.max_dist_km})")
            edge_index, edge_weight = build_graph(
                built["lats"], built["lons"], max_dist_km=args.max_dist_km
            )
            print(f"      edges={edge_index.shape[1]}")

        if args.model_type == "lstm":
            print(
                f"[4/6] Training LSTM baseline "
                f"(input shape per batch: [B, n_nodes={len(built['loc_ids'])}, "
                f"seq_len={args.seq_len}, n_features={built['n_features']}], "
                f"hidden_size={args.hidden_size}, layers={args.lstm_layers}, "
                f"dropout={args.dropout}, epochs={args.epochs}, device={args.device})"
            )
            print(
                f"[model] n_features={built['n_features']}  "
                f"hidden_size={args.hidden_size}  lstm_layers={args.lstm_layers}  "
                f"dropout={args.dropout}"
            )
            print(
                f"[protocol] train_years={args.train_years}  test_year={args.test_year}  "
                f"posthoc_calibration={'enabled' if posthoc else 'disabled'}"
            )
            if args.train_noise_mode == "anomaly":
                print("[protocol] anomaly labels: ALSO used in training (anomaly-aware "
                      "noise) -> NOT eval-only")
            else:
                print("[protocol] anomaly labels: eval/stratification only (never input/target)")
            model = LSTMBaseline(
                n_features=built["n_features"],
                hidden_size=args.hidden_size,
                num_layers=args.lstm_layers,
                dropout=args.dropout,
            )
        else:
            enhanced = args.model_type == "stgnn_enhanced_dropout"
            print(
                f"[4/6] Training {'STGNN (Enhanced MC Dropout ablation)' if enhanced else 'STGNN'} "
                f"(n_features={built['n_features']}, "
                f"dropout={args.dropout}, epochs={args.epochs}, device={args.device})"
            )
            if enhanced:
                print("[model] model_type=stgnn_enhanced_dropout")
                print("[model] enhanced_mc_dropout=true")
                print("[model] explicit dropout modules added:")
                print("  - temporal_dropout (after BiLSTM temporal embedding)")
                print("  - representation_dropout (after projection, before GAT)")
                print("  - head_pv.2 dropout (before the final Linear of the pv head)")
                print("  - gat dropout existing (gat.0.dropout, unchanged)")
            print(
                f"[model] loss_type={args.loss_type}  "
                f"huber_delta={float(args.huber_delta)}  "
                f"use_irradiance_head={bool(args.use_irradiance_head)}  "
                f"use_irradiance_loss={bool(args.use_irradiance_loss)}  "
                f"irradiance_loss_weight={float(args.irradiance_loss_weight)}"
            )
            print(
                f"[model] train_mc_uncertainty_penalty="
                f"{bool(args.train_mc_uncertainty_penalty)}  "
                f"train_mc_samples={int(args.train_mc_samples)}  "
                f"uncertainty_penalty_mode={args.uncertainty_penalty_mode}  "
                f"uncertainty_penalty_weight={float(args.uncertainty_penalty_weight)}  "
                f"uncertainty_penalty_k={float(args.uncertainty_penalty_k)}  "
                f"uncertainty_std_reg_weight={float(args.uncertainty_std_reg_weight)}  "
                f"sde_proxy_in_weight={float(args.sde_proxy_in_weight)}  "
                f"sde_proxy_out_weight={float(args.sde_proxy_out_weight)}  "
                f"sde_proxy_std_min_ood={float(args.sde_proxy_std_min_ood)}  "
                f"train_noise_mode={args.train_noise_mode}  "
                f"train_noise_std={float(args.train_noise_std)}  "
                f"train_noise_prob={float(args.train_noise_prob)}  "
                f"anomaly_noise_std={float(args.anomaly_noise_std)}  "
                f"anomaly_noise_prob={float(args.anomaly_noise_prob)}"
            )
            model = make_model(
                len(built["loc_ids"]), args.seq_len, built["n_features"],
                dropout=args.dropout, enhanced_dropout=enhanced,
                use_irradiance_head=bool(args.use_irradiance_head),
            )
        t_train = time.perf_counter()
        model = train_model(
            model, built["train"], edge_index, edge_weight,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device,
            use_irradiance_loss=bool(args.use_irradiance_loss),
            irradiance_loss_weight=float(args.irradiance_loss_weight),
            loss_type=args.loss_type,
            huber_delta=float(args.huber_delta),
            train_mc_uncertainty_penalty=bool(args.train_mc_uncertainty_penalty),
            train_mc_samples=int(args.train_mc_samples),
            uncertainty_penalty_mode=args.uncertainty_penalty_mode,
            uncertainty_penalty_weight=float(args.uncertainty_penalty_weight),
            uncertainty_penalty_k=float(args.uncertainty_penalty_k),
            uncertainty_std_reg_weight=float(args.uncertainty_std_reg_weight),
            sde_proxy_in_weight=float(args.sde_proxy_in_weight),
            sde_proxy_out_weight=float(args.sde_proxy_out_weight),
            sde_proxy_std_min_ood=float(args.sde_proxy_std_min_ood),
            train_noise_std=float(args.train_noise_std),
            train_noise_prob=float(args.train_noise_prob),
            train_noise_mode=args.train_noise_mode,
            anomaly_noise_std=float(args.anomaly_noise_std),
            anomaly_noise_prob=float(args.anomaly_noise_prob),
            feature_names=features,
        )
        print(f"      [time] training total: {time.perf_counter() - t_train:.1f}s")
        # Per-epoch loss components (loss/pv, loss/irradiance, loss/total) -> W&B.
        train_history = getattr(model, "train_loss_history", None)
        if wandb_run is not None and train_history:
            for ep_i, rec in enumerate(train_history, start=1):
                wandb_run.log({"epoch": ep_i, **rec})

        calibration = None
        calibration_factor = None
        n_calibration_predictions = 0
        if posthoc and args.mc_dropout and built["calibration"] is not None:
            print(
                "[5/6] (opt-in) Post-hoc calibrating MC Dropout uncertainty on "
                f"calibration years {calibration_years} "
                f"(strategy={args.calibration_strategy}, target={args.coverage_target})"
            )
            t_cal = time.perf_counter()
            calibration_predictions = predict_mc(
                model, built["calibration"], edge_index, edge_weight,
                args.device, args.batch_size, mc_samples=args.mc_samples,
                coverage_target=args.coverage_target,
            )
            print(f"      [time] MC calibration inference: {time.perf_counter() - t_cal:.1f}s")
            n_calibration_predictions = len(calibration_predictions)
            calibration_anomaly = load_anomaly_labels(args.calibration_anomaly_scores)
            calibration_predictions = attach_anomaly_labels(
                calibration_predictions, calibration_anomaly
            )
            calibration = estimate_mc_calibration_factors(
                calibration_predictions,
                strategy=args.calibration_strategy,
                coverage_target=args.coverage_target,
                eps=args.calibration_eps,
                min_samples=args.min_calibration_samples_per_stratum,
            )
            calibration_factor = calibration["global"]
            print(
                f"      k_global={calibration['global']:.6f} "
                f"from {n_calibration_predictions} calibration predictions"
            )
            for key, k in sorted(calibration["factors"].items()):
                print(f"      {key:28s} k={k:.6f}  (n={calibration['counts'].get(key)})")
            for key, fb in sorted(calibration["fallbacks"].items()):
                print(
                    f"      {key:28s} fallback -> {fb}  "
                    f"(n={calibration['counts'].get(key)} < "
                    f"{args.min_calibration_samples_per_stratum})"
                )
            # Raw-std diagnostics: confirm the (large) factors are not an artefact
            # of near-zero MC std rather than a genuine under-dispersed posterior.
            sd = calibration.get("std_diagnostics", {})
            gd = sd.get("global", {})
            nd = sd.get("normal", {})
            rd = sd.get("rare_or_extreme", {})
            print("      [std-diag] raw MC std on calibration set (sanity for large k):")
            if gd:
                print(
                    f"        global : min={gd['min']:.6g} max={gd['max']:.6g} "
                    f"pct<eps={gd['pct_below_eps'] * 100:.2f}%  (n={gd['n']})"
                )
            if nd:
                print(
                    f"        normal : mean={nd['mean']:.6g} median={nd['median']:.6g}  "
                    f"(n={nd['n']})"
                )
            if rd:
                print(
                    f"        rare   : mean={rd['mean']:.6g} median={rd['median']:.6g}  "
                    f"(n={rd['n']})"
                )

        print("[5/6] Predicting on test year + attaching anomaly labels")
        t_test = time.perf_counter()
        if args.mc_dropout:
            print(
                f"      MC Dropout: eval() + reactivate only nn.Dropout, "
                f"{args.mc_samples} forward passes/batch (no model.train())"
            )
            predictions = predict_mc(
                model, built["test"], edge_index, edge_weight,
                args.device, args.batch_size, mc_samples=args.mc_samples,
                coverage_target=args.coverage_target,
            )
        else:
            predictions = predict(
                model, built["test"], edge_index, edge_weight, args.device, args.batch_size
            )
        print(f"      [time] MC test inference: {time.perf_counter() - t_test:.1f}s")
        anomaly_scores = load_anomaly_labels(args.anomaly_scores)
        predictions = attach_anomaly_labels(predictions, anomaly_scores)
        if calibration is not None:
            predictions = apply_mc_uncertainty_calibration_stratified(predictions, calibration)
        global_df, by_df = compute_metrics(predictions, calibration=calibration)

        # Interval reliability/sharpness (PICP/MPIW/NMPIL/CLC). Eval-only; needs
        # MC-Dropout intervals. A single global target_range normalises NMPIL.
        interval_metrics = None
        daytime_metrics = None
        residual_bias_metrics = None
        target_range = None
        clc_gamma = float(args.coverage_target)
        if args.mc_dropout and {"lower_pi", "upper_pi"} <= set(predictions.columns):
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
            for kind in ("pi", "gaussian", "calibrated"):
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
                "test_year": args.test_year,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "loss_type": args.loss_type,
                "huber_delta": float(args.huber_delta),
                "train_mc_uncertainty_penalty": bool(args.train_mc_uncertainty_penalty),
                "train_mc_samples": int(args.train_mc_samples),
                "uncertainty_penalty_mode": args.uncertainty_penalty_mode,
                "uncertainty_penalty_weight": float(args.uncertainty_penalty_weight),
                "uncertainty_penalty_k": float(args.uncertainty_penalty_k),
                "uncertainty_std_reg_weight": float(args.uncertainty_std_reg_weight),
                "sde_proxy_in_weight": float(args.sde_proxy_in_weight),
                "sde_proxy_out_weight": float(args.sde_proxy_out_weight),
                "sde_proxy_std_min_ood": float(args.sde_proxy_std_min_ood),
                "train_noise_std": float(args.train_noise_std),
                "train_noise_prob": float(args.train_noise_prob),
                "train_noise_mode": args.train_noise_mode,
                "anomaly_noise_std": float(args.anomaly_noise_std),
                "anomaly_noise_prob": float(args.anomaly_noise_prob),
                "use_irradiance_head": bool(args.use_irradiance_head),
                "use_irradiance_loss": bool(args.use_irradiance_loss),
                "irradiance_loss_weight": float(args.irradiance_loss_weight),
                "anomaly_scores": args.anomaly_scores,
                "device": args.device,
                "wandb_enabled": bool(args.wandb),
                "mc_dropout": bool(args.mc_dropout),
                "mc_samples": args.mc_samples,
                "calibration_years": args.calibration_years,
                "coverage_target": args.coverage_target,
                "calibration_eps": args.calibration_eps,
                "calibration_factor": calibration_factor,
                "calibration_strategy": args.calibration_strategy,
                "calibration_anomaly_scores": args.calibration_anomaly_scores,
                "min_calibration_samples_per_stratum": args.min_calibration_samples_per_stratum,
                "calibration": calibration,
                "n_calibration_predictions": n_calibration_predictions,
                "max_calibration_samples": args.max_calibration_samples,
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
            global_df, by_df, mc_dropout=bool(args.mc_dropout), calibration=calibration
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
        # Post-hoc daytime-bins x anomaly analysis (eval-only). Runs after
        # predictions.csv is written; never touches the model/loss/training. A
        # failure here must NOT lose the main report.
        posthoc_dir = None
        if posthoc_will_run:
            posthoc_dir = _run_posthoc_daytime_bins(
                paths.get("predictions"), Path(out_dir) / "daytime_bin_anomaly"
            )
        elif posthoc_enabled and args.skip_predictions_csv:
            print("[posthoc] skipped: --skip-predictions-csv (no predictions.csv to analyse).")

        posthoc_uploaded = False
        if wandb_run is not None:
            wandb_run.log(summary)
            wandb_run.summary.update(summary)
            if calibration is not None:
                # strategy is a string -> summary only (kept out of the numeric dict).
                wandb_run.summary["calibration/strategy"] = calibration["strategy"]
            if posthoc_dir is not None and upload_artifacts:
                posthoc_uploaded = _log_wandb_posthoc_artifact(
                    wandb, wandb_run, posthoc_dir
                )

        meta["posthoc_executed"] = posthoc_dir is not None
        meta["posthoc_uploaded"] = posthoc_uploaded
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
        for ds in calibration_map.values():
            ds.close()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_from_args(args, parser=parser)


if __name__ == "__main__":
    main()
