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

from physiq_pv.data.pvgis_stgnn_dataset import (
    CALIBRATION_STRATEGIES,
    DEFAULT_TARGET_VARIABLE,
    FEATURE_SETS,
    apply_mc_uncertainty_calibration_stratified,
    attach_anomaly_labels,
    build_datasets,
    build_meta,
    build_wandb_metrics,
    compute_metrics,
    estimate_mc_calibration_factors,
    load_anomaly_labels,
    load_pvgis_year,
    load_pvgis_years,
    make_model,
    predict,
    predict_mc,
    resolve_feature_set,
    train_model,
    write_outputs,
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


def _log_wandb_artifact(wandb, wandb_run, paths: dict, log_predictions: bool) -> None:
    """
    Attach the run's local outputs to W&B as a versioned artifact.

    Always includes report.md + the two metrics CSVs; predictions.csv is added
    only when `log_predictions` is set (it can be very large).
    """
    artifact = wandb.Artifact(f"pvgis_stgnn_{wandb_run.id}", type="pvgis_stgnn_outputs")
    keys = ["report", "metrics_global", "metrics_by_anomaly_label"]
    if log_predictions:
        keys.append("predictions")
    for key in keys:
        p = paths.get(key)
        if p is not None and Path(p).exists():
            artifact.add_file(str(p))
    wandb_run.log_artifact(artifact)
    print(
        f"[wandb] logged artifact {artifact.name} "
        f"({'with' if log_predictions else 'without'} predictions.csv)"
    )


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


ENSEMBLE_PREDICTIONS_ROOT = "outputs/pvgis_deep_ensemble/predictions"


def resolve_ensemble_dir(
    ensemble_dir: str, ensemble_id: Optional[str], wandb_run=None
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Resolve the per-seed dump directory for the Deep Ensemble.

    Goal: ALL seed members of ONE sweep write into ONE shared folder, while
    different sweeps stay isolated (so the aggregator never mixes ensembles).

    * `ensemble_dir != "auto"` -> used verbatim (back-compat).
    * `ensemble_dir == "auto"` -> `<ROOT>/<key>` where key is the W&B sweep id
      (wandb_run.sweep_id or $WANDB_SWEEP_ID) else `--ensemble-id`. If neither is
      available, raise ValueError (caller turns it into a clean CLI error).

    Returns (resolved_dir, sweep_id, ensemble_id).
    """
    sweep_id = None
    if wandb_run is not None:
        sweep_id = getattr(wandb_run, "sweep_id", None) or None
    if not sweep_id:
        sweep_id = os.environ.get("WANDB_SWEEP_ID") or None

    if ensemble_dir != "auto":
        return ensemble_dir, sweep_id, ensemble_id

    key = sweep_id or ensemble_id
    if not key:
        raise ValueError(
            "ensemble_predictions_dir=auto requires W&B sweep id or --ensemble-id"
        )
    return f"{ENSEMBLE_PREDICTIONS_ROOT}/{key}", sweep_id, ensemble_id


def build_sample_id(predictions) -> np.ndarray:
    """
    Deterministic per-row sample id, STABLE across seeds.

    Built from (target timestamp, location): the test set is built deterministically
    and predicted with shuffle=False, so the same physical (timestamp, node) maps to
    the same id in every seed run. The ensemble aggregator aligns seeds on this id.
    """
    import pandas as pd  # noqa: PLC0415 — local import keeps module import light
    ts = pd.to_datetime(predictions["timestamp"]).astype("int64").astype(str)
    loc = predictions["location"].astype(str)
    # dtype=str -> fixed-width unicode array (NOT object), so np.load works with
    # allow_pickle=False in the aggregator.
    return np.asarray(ts + "_" + loc, dtype=str)


def save_ensemble_predictions(predictions, out_path: Path, seed: int) -> None:
    """
    Save a lightweight per-seed .npz for Deep Ensemble aggregation.

    Stores the per-seed MEAN prediction (not the raw mc_predictions, which can be
    large): sample_id / y_true / y_pred_mean / anomaly_group / seed, plus optional
    timestamp / location_id / y_pred_std_mc. Anomaly labels are eval-only metadata.
    """
    import pandas as pd  # noqa: PLC0415
    y_mean = predictions["y_pred_mean"] if "y_pred_mean" in predictions else predictions["y_pred"]
    data = {
        "sample_id": build_sample_id(predictions),
        "y_true": predictions["y_true"].to_numpy(dtype=float),
        "y_pred_mean": y_mean.to_numpy(dtype=float),
        # dtype=str -> fixed-width unicode (not object) so allow_pickle=False loads.
        "anomaly_group": np.asarray(predictions["anomaly_group"], dtype=str),
        "seed": np.asarray(int(seed)),
    }
    if "timestamp" in predictions:
        data["timestamp"] = pd.to_datetime(predictions["timestamp"]).astype("int64").to_numpy()
    if "location" in predictions:
        data["location_id"] = np.asarray(predictions["location"], dtype=str)
    std_col = next((c for c in ("y_pred_std_raw", "y_pred_std") if c in predictions), None)
    if std_col is not None:
        data["y_pred_std_mc"] = predictions[std_col].to_numpy(dtype=float)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)


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
    # Deep Ensemble: save a lightweight per-seed .npz (NOT the big predictions.csv)
    # so scripts/analyze_pvgis_deep_ensemble.py can combine seeds per-sample.
    g.add_argument("--save-ensemble-predictions", "--save_ensemble_predictions",
                   action="store_true",
                   help="Save a lightweight per-seed .npz (sample_id/y_true/"
                        "y_pred_mean/anomaly_group/seed) for Deep Ensemble aggregation.")
    g.add_argument("--ensemble-predictions-dir", "--ensemble_predictions_dir",
                   default="outputs/pvgis_deep_ensemble/predictions",
                   help="Directory for the per-seed Deep Ensemble .npz files. Use "
                        "'auto' to resolve a per-sweep subdir "
                        f"{ENSEMBLE_PREDICTIONS_ROOT}/<sweep_id or ensemble_id> so all "
                        "seeds of ONE sweep share ONE folder (and different sweeps stay "
                        "separate).")
    g.add_argument("--ensemble-id", "--ensemble_id", default=None,
                   help="Explicit ensemble id for --ensemble-predictions-dir=auto when "
                        "there is no W&B sweep id (manual runs).")
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
                "train_years": args.train_years,
                "test_year": args.test_year,
                "seq_len": args.seq_len,
                "horizon": args.horizon,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "dropout": args.dropout,
                "hidden_size": args.hidden_size,
                "lstm_layers": args.lstm_layers,
                "device": args.device,
                "max_train_samples": args.max_train_samples,
                "max_test_samples": args.max_test_samples,
                "max_calibration_samples": args.max_calibration_samples,
                "skip_predictions_csv": args.skip_predictions_csv,
                "save_ensemble_predictions": bool(args.save_ensemble_predictions),
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
            feature_names=features, calibration_ds_map=calibration_map,
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
            model = make_model(
                len(built["loc_ids"]), args.seq_len, built["n_features"],
                dropout=args.dropout, enhanced_dropout=enhanced,
            )
        t_train = time.perf_counter()
        model = train_model(
            model, built["train"], edge_index, edge_weight,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device,
        )
        print(f"      [time] training total: {time.perf_counter() - t_train:.1f}s")

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

        # Deep Ensemble: lightweight per-seed dump (mean prediction per sample) for
        # later per-sample aggregation across seeds. Not the big predictions.csv.
        # All seeds of ONE sweep share ONE folder (auto -> <ROOT>/<sweep_id>).
        if args.save_ensemble_predictions:
            try:
                resolved_dir, sweep_id, ensemble_id = resolve_ensemble_dir(
                    args.ensemble_predictions_dir, args.ensemble_id, wandb_run
                )
            except ValueError as e:
                _fail(parser, str(e))
            rid = wandb_run.id if wandb_run is not None else None
            fname = f"{rid}_seed{args.seed}.npz" if rid else f"run_seed{args.seed}.npz"
            ens_path = Path(resolved_dir) / fname
            print(f"[ensemble] predictions_dir_resolved={resolved_dir}")
            print(f"[ensemble] sweep_id={sweep_id}")
            print(f"[ensemble] ensemble_id={ensemble_id}")
            print(f"[ensemble] seed={args.seed}")
            print(f"[ensemble] saving predictions to {ens_path}")
            save_ensemble_predictions(predictions, ens_path, args.seed)
            print(f"[ensemble] saved per-seed predictions ({len(predictions)} rows) -> {ens_path}")

        print(f"[6/6] Writing outputs to {out_dir}")
        meta = build_meta(
            {
                "mode": "pvgis_stgnn",
                "model_type": args.model_type,
                "feature_set": args.feature_set,
                "target_variable": args.target_variable,
                "seq_len": args.seq_len,
                "horizon": args.horizon,
                "train_years": args.train_years,
                "test_year": args.test_year,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
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
        if wandb_run is not None:
            wandb_run.log(summary)
            wandb_run.summary.update(summary)
            if calibration is not None:
                # strategy is a string -> summary only (kept out of the numeric dict).
                wandb_run.summary["calibration/strategy"] = calibration["strategy"]
            _log_wandb_artifact(wandb, wandb_run, paths, args.wandb_log_predictions)

        print(f"\nDone. [time] total run: {time.perf_counter() - t_run_start:.1f}s")
        print(global_df.to_string(index=False))
        if not by_df.empty:
            print(by_df.to_string(index=False))
        print("\nKey metrics:")
        for k in sorted(summary):
            print(f"  {k:32s} {summary[k]:.4f}")
        for key in ("predictions", "metrics_global", "metrics_by_anomaly_label", "report"):
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
