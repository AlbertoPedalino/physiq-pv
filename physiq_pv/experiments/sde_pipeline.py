"""Reproducible orchestration helpers for the PVGIS-only SDE pipeline.

Pure, importable helpers shared by:
  * notebooks/pvgis_sde_pipeline.ipynb
  * scripts/run_pvgis_sde_sweep_member.py
  * tests/test_sde_pipeline.py

They ONLY build commands / paths and read the CSVs that the runner and the
analysis script already write. No training, model, loss or report logic is
duplicated here — the runner and the analysis script remain the single source
of truth.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from physiq_pv.reporting.posthoc_outputs import (
    POSTHOC_KEYS,
    PRODUCTION_BINS,
    build_extreme_event_comparison_figures,
    build_extreme_event_diagnostic,
    build_direct_multihorizon_posthoc,
    build_horizon_comparison_figures,
    build_posthoc_figures,
    collect_run_artifact_files,
    init_wandb_run_for_out_dir as _init_wandb_run_for_out_dir,
    load_prediction_sample,
    log_posthoc_to_wandb,
    read_posthoc_summary,
)

# --------------------------------------------------------------------------- #
# Defaults (server newzealand)
# --------------------------------------------------------------------------- #
PVGIS_DIR = "/data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance"
TEST_ANOMALY_SCORES = (
    "outputs/pvgis_anomaly_2019_2005_2018_w15_q0975/pvgis_climatology_scores.csv"
)
TRAIN_ANOMALY_SCORES = (
    "outputs/pvgis_anomaly_train_2016_2018_2005_past_w15_q0975/"
    "pvgis_climatology_scores.csv"
)
RUNNER_MODULE = "physiq_pv.experiments.pvgis_stgnn_runner"
ANALYSIS_SCRIPT = "scripts/analyze_pvgis_daytime_report.py"
SWEEP_MEMBER_SCRIPT = "scripts/run_pvgis_sde_sweep_member.py"
WANDB_PROJECT = "physiq_pv"
WANDB_ENTITY = "albertopedalino-politecnico-di-torino"
FORECAST_HORIZONS = (1, 2, 3, 4, 5, 6)


def init_wandb_run_for_out_dir(
    wandb,
    out_dir: str,
    *,
    run_name: Optional[str] = None,
    project: str = WANDB_PROJECT,
    entity: Optional[str] = WANDB_ENTITY,
    reinit: bool = True,
):
    return _init_wandb_run_for_out_dir(
        wandb,
        out_dir,
        run_name=run_name,
        project=project,
        entity=entity,
        reinit=reinit,
    )


# The single-run base config (mirrors the documented final run).
DEFAULT_CONFIG: Dict = {
    "train_years": "2016,2017,2018",
    "test_year": 2019,
    "seq_len": 24,
    "horizon": 1,
    "target_variable": "pv_power_output",
    "model_type": "stgnn",
    "feature_set": "full",
    "epochs": 60,
    "batch_size": 16,
    "lr": 0.0001,  # paper regression drift/backbone learning rate
    "lr_g": 0.01,  # paper diffusion lr
    "dropout": 0.0,
    "mc_samples": 10,
    "seed": 1,
    "n_sde_steps": 4,
    "sigma_max": 0.5,
    "sde_sigma_initial": 0.01,  # v1 PDF; public repo uses 0.1
    "sde_sigma_warmup_epochs": 30,

    "ood_noise_std": 2.0,
    "ood_smoke_test": True,
    "ood_smoke_max_samples": 2048,
    "gradient_clip_norm": 100.0,
    "lr_decay_epoch": 20,
    "lr_decay_factor": 0.1,
    "irradiance_loss_weight": 0.1,
    "pv_target_clip_max": "none",
    "kt_poa_max": 1.6,
    "distance_scale_km": 10.0,
    "edge_prior_strength": 1.0,
    "validation_metric": "rmse_daytime",
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.0,
    "anomaly_source": "climatology",
    "event_spatial_quantile": 0.99,
    "event_tail_quantile": 0.975,
    "detector_regional_quantile": 0.975,
    "detector_min_temporal_coverage": 0.95,
}


def _tag(x) -> str:
    """Compact, filesystem-safe tag for a config value.

    Floats: strip trailing zeros and replace '.'->'p', '-'->'m'
    (0.1 -> '0p1', 0.00005 -> '5em05', 0.5 -> '0p5'). Other values: str().
    """
    if isinstance(x, float):
        s = repr(x)
        if "e" in s or "E" in s:
            s = format(x, ".10f").rstrip("0").rstrip(".")
    elif isinstance(x, int):
        s = str(x)
    else:
        return str(x)
    return s.replace(".", "p").replace("-", "m")


def make_run_name(config: Dict) -> str:
    """Deterministic, collision-free run name from the salient params.

    An explicit `config["name"]` is used verbatim (still suffixed with the seed)
    so callers can pin a human-readable tag; otherwise it is derived from the
    loss / SDE-block / seed values.
    """
    seed = config.get("seed", 0)
    horizon = int(config.get("horizon", DEFAULT_CONFIG["horizon"]))
    if horizon < 1:
        raise ValueError(f"Forecast horizon must be >= 1 hour, got {horizon}.")
    # Direct multi-output and legacy scalar runs receive distinct paths.
    direct = str(config.get("forecast_horizons", "")).strip()
    horizon_tag = (
        f"_h{direct.replace(',', '-')}_direct" if direct else
        ("" if horizon == 1 else f"_h{horizon}")
    )
    if config.get("name"):
        return f"pvgis_stgnn_{config['name']}{horizon_tag}_seed{seed}"
    return (
        f"pvgis_stgnn_sde{_tag(config.get('n_sde_steps', 4))}"
        f"_sm{_tag(config.get('sigma_max', 0.5))}"
        f"_si{_tag(config.get('sde_sigma_initial', 0.01))}"
        f"_sw{_tag(config.get('sde_sigma_warmup_epochs', 30))}"
        f"_ood{_tag(config.get('ood_noise_std', 2.0))}"
        f"{horizon_tag}"
        f"_seed{seed}"
    )


def make_out_dir(config: Dict, root: str = "outputs") -> str:
    """`<root>/<run_name>` — deterministic, so re-running the same config reuses
    the same dir; distinct configs (incl. seed) never collide."""
    return f"{root}/{make_run_name(config)}"


def ensure_output_dir_available(
    out_dir: str | Path, *, allow_overwrite: bool = False
) -> Path:
    """Reject an existing non-empty run path unless explicitly allowed."""
    path = Path(out_dir)
    occupied = path.exists() and (not path.is_dir() or any(path.iterdir()))
    if occupied and not allow_overwrite:
        raise FileExistsError(
            f"Output path is not empty: {path}. Change detector/configuration/seed "
            "or set allow_overwrite=True explicitly."
        )
    return path


def relabel_detector_predictions(
    predictions: pd.DataFrame,
    detector_scores: pd.DataFrame,
    training_detector_scores: pd.DataFrame,
    *,
    regional_quantile: float = 0.975,
    min_temporal_coverage: float = 0.95,
) -> tuple[pd.DataFrame, dict]:
    """Re-label predictions using train-fitted seasonal regional thresholds."""
    from physiq_pv.data.pvgis_dataset import build_detector_event_protocol
    from physiq_pv.data.pvgis_labels import (
        attach_anomaly_labels,
        attach_event_labels,
    )

    required = {"location", "timestamp"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing label join columns: {sorted(missing)}")
    clean = predictions.drop(
        columns=[
            "anomaly_group",
            "anomaly_label",
            "event_group",
            "event_score",
            "event_driver",
        ],
        errors="ignore",
    ).copy()
    clean["timestamp"] = pd.to_datetime(
        clean["timestamp"], utc=True
    ).dt.tz_convert(None)
    locations = clean["location"].astype(str).drop_duplicates().to_numpy()
    years = clean["timestamp"].dt.year
    times_by_year = {
        int(year): pd.DatetimeIndex(
            clean.loc[years == year, "timestamp"].drop_duplicates().sort_values()
        )
        for year in sorted(years.unique())
    }
    calibration_scores = training_detector_scores.copy()
    calibration_scores["timestamp"] = pd.to_datetime(
        calibration_scores["timestamp"], utc=True
    ).dt.tz_convert(None)
    calibration_years = calibration_scores["timestamp"].dt.year
    calibration_times_by_year = {
        int(year): pd.DatetimeIndex(
            calibration_scores.loc[
                calibration_years == year, "timestamp"
            ].drop_duplicates().sort_values()
        )
        for year in sorted(calibration_years.unique())
    }
    calibration_protocol = build_detector_event_protocol(
        calibration_scores,
        calibration_times_by_year,
        locations,
        regional_quantile=regional_quantile,
        min_temporal_coverage=min_temporal_coverage,
    )
    protocol = build_detector_event_protocol(
        detector_scores,
        times_by_year,
        locations,
        regional_quantile=regional_quantile,
        seasonal_thresholds=calibration_protocol["seasonal_thresholds"],
        min_temporal_coverage=min_temporal_coverage,
    )
    protocol["calibration_years"] = sorted(calibration_times_by_year)
    local = attach_anomaly_labels(clean, detector_scores)
    event_labels = pd.concat(
        [protocol["labels_by_year"][year] for year in sorted(times_by_year)],
        ignore_index=True,
    )
    return attach_event_labels(local, event_labels), protocol


def relabel_detector_predictions_file(
    source_predictions: str | Path,
    detector_scores_path: str | Path,
    training_detector_scores_path: str | Path,
    out_dir: str | Path,
    *,
    regional_quantile: float = 0.975,
    min_temporal_coverage: float = 0.95,
    allow_overwrite: bool = False,
) -> dict[str, Path]:
    """Create an evaluation-only detector-specific predictions directory."""
    from physiq_pv.data.pvgis_labels import load_anomaly_labels
    from physiq_pv.reporting.run_metrics import (
        build_wandb_metrics,
        compute_metrics,
    )

    source = Path(source_predictions)
    scores_path = Path(detector_scores_path)
    training_scores_path = Path(training_detector_scores_path)
    if not source.is_file():
        raise FileNotFoundError(f"Source predictions not found: {source}")
    scores = load_anomaly_labels(str(scores_path), source="detector")
    training_scores = load_anomaly_labels(
        str(training_scores_path), source="detector"
    )
    predictions = pd.read_csv(source, parse_dates=["timestamp"])
    relabelled, protocol = relabel_detector_predictions(
        predictions,
        scores,
        training_scores,
        regional_quantile=regional_quantile,
        min_temporal_coverage=min_temporal_coverage,
    )
    output_root = ensure_output_dir_available(
        out_dir, allow_overwrite=allow_overwrite
    )
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = output_root / "predictions.csv"
    metadata_path = output_root / "evaluation_source.json"
    metrics_global_path = output_root / "metrics_global.csv"
    metrics_by_path = output_root / "metrics_by_anomaly_label.csv"
    metrics_path = output_root / "metrics.json"
    report_path = output_root / "report.md"
    relabelled.to_csv(predictions_path, index=False)
    global_df, by_df = compute_metrics(relabelled)
    global_df.to_csv(metrics_global_path, index=False)
    by_df.to_csv(metrics_by_path, index=False)
    flat_metrics = build_wandb_metrics(
        global_df,
        by_df,
        sde_uncertainty=(
            "y_pred_std" in relabelled
            and relabelled["y_pred_std"].notna().any()
        ),
    )
    metrics_path.write_text(
        json.dumps(flat_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    source_checkpoint = source.parent / "best_model.pt"
    report_path.write_text(
        "\n".join(
            [
                "# Detector evaluation-only report",
                "",
                f"- Detector: `{protocol['detector']}`",
                f"- Source predictions: `{source.resolve()}`",
                (
                    f"- Source checkpoint: `{source_checkpoint.resolve()}`"
                    if source_checkpoint.is_file()
                    else "- Source checkpoint: not found beside source predictions"
                ),
                f"- Detector scores: `{scores_path.resolve()}`",
                f"- Training detector scores: `{training_scores_path.resolve()}`",
                f"- Seasonal regional quantile: `{regional_quantile:g}`",
                f"- Frozen seasonal thresholds: `{protocol['seasonal_thresholds']}`",
                f"- Prediction rows: `{len(relabelled)}`",
                "",
                "## Metrics",
                "",
                *[
                    f"- `{name}`: `{value:.6g}`"
                    for name, value in sorted(flat_metrics.items())
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "mode": "detector_evaluation_only",
                "source_predictions": str(source.resolve()),
                "detector_scores": str(scores_path.resolve()),
                "training_detector_scores": str(training_scores_path.resolve()),
                "detector": protocol["detector"],
                "regional_quantile": regional_quantile,
                "seasonal_thresholds": protocol["seasonal_thresholds"],
                "min_temporal_coverage": min_temporal_coverage,
                "prediction_rows": len(relabelled),
                "source_checkpoint": (
                    str(source_checkpoint.resolve())
                    if source_checkpoint.is_file()
                    else None
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "predictions": predictions_path,
        "evaluation_source": metadata_path,
        "metrics_global": metrics_global_path,
        "metrics_by_anomaly_label": metrics_by_path,
        "metrics": metrics_path,
        "report": report_path,
    }


def _value_flags(config: Dict) -> List[tuple]:
    """(cli_flag, config_key) pairs for the value-taking runner arguments."""
    return [
        ("--train-years", "train_years"),
        ("--test-year", "test_year"),
        ("--seq-len", "seq_len"),
        ("--horizon", "horizon"),
        ("--forecast-horizons", "forecast_horizons"),
        ("--target-variable", "target_variable"),
        ("--model-type", "model_type"),
        ("--feature-set", "feature_set"),
        ("--irradiance-loss-weight", "irradiance_loss_weight"),
        ("--epochs", "epochs"),
        ("--batch-size", "batch_size"),
        ("--lr", "lr"),
        ("--lr-g", "lr_g"),
        ("--dropout", "dropout"),
        ("--mc-samples", "mc_samples"),
        ("--pv-target-clip-max", "pv_target_clip_max"),
        ("--seed", "seed"),
        ("--n-sde-steps", "n_sde_steps"),
        ("--sigma-max", "sigma_max"),
        ("--sde-sigma-initial", "sde_sigma_initial"),
        ("--sde-sigma-warmup-epochs", "sde_sigma_warmup_epochs"),
        ("--ood-noise-std", "ood_noise_std"),
        ("--ood-smoke-max-samples", "ood_smoke_max_samples"),
        ("--gradient-clip-norm", "gradient_clip_norm"),
        ("--lr-decay-epoch", "lr_decay_epoch"),
        ("--lr-decay-factor", "lr_decay_factor"),
        ("--kt-poa-max", "kt_poa_max"),
        ("--distance-scale-km", "distance_scale_km"),
        ("--edge-prior-strength", "edge_prior_strength"),
        ("--validation-metric", "validation_metric"),
        ("--early-stopping-patience", "early_stopping_patience"),
        ("--early-stopping-min-delta", "early_stopping_min_delta"),
        ("--anomaly-source", "anomaly_source"),
        ("--event-spatial-quantile", "event_spatial_quantile"),
        ("--event-tail-quantile", "event_tail_quantile"),
        (
            "--detector-regional-quantile",
            "detector_regional_quantile",
        ),
        (
            "--detector-min-temporal-coverage",
            "detector_min_temporal_coverage",
        ),
    ]


def build_train_command(
    config: Dict,
    *,
    out_dir: str,
    run_name: str,
    pvgis_dir: str = PVGIS_DIR,
    test_anomaly_scores: str = TEST_ANOMALY_SCORES,
    train_anomaly_scores: str = TRAIN_ANOMALY_SCORES,
    device: str = "cuda",
    use_wandb: bool = True,
    wandb_project: str = WANDB_PROJECT,
    wandb_entity: str = WANDB_ENTITY,
    python_exe: Optional[str] = None,
) -> List[str]:
    """Argument list for `subprocess.run` that launches the runner.

    Equivalent to the documented final command. Returns a list (never a fragile
    shell string). `use_wandb=False` drops the W&B flags (e.g. the sweep wrapper
    owns the W&B run itself).
    """
    cfg = {**DEFAULT_CONFIG, **config}
    py = python_exe or sys.executable
    cmd: List[str] = [py, "-m", RUNNER_MODULE,
                      "--pvgis-dir", str(pvgis_dir),
                      "--anomaly-scores", str(test_anomaly_scores)]
    for flag, key in _value_flags(cfg):
        if key in cfg and cfg[key] is not None:
            cmd += [flag, str(cfg[key])]
    # boolean store_true flags
    cmd += ["--use-irradiance-head", "--use-irradiance-loss", "--sde-uncertainty"]
    if cfg.get("ood_smoke_test"):
        cmd.append("--ood-smoke-test")
    if cfg.get("train_normal_only"):
        # Label-defined normal-only ablation: target and input history are normal.
        cmd.append("--train-normal-only")
    if cfg.get("train_normal_only") or cfg.get("anomaly_source") == "detector":
        cmd += ["--train-anomaly-scores", str(train_anomaly_scores)]
    cmd += ["--device", str(device), "--out-dir", str(out_dir)]
    if use_wandb:
        cmd += ["--wandb",
                "--wandb-project", str(wandb_project),
                "--wandb-entity", str(wandb_entity),
                "--wandb-run-name", str(run_name),
                "--no-wandb-upload-artifacts"]
    return cmd


def build_analysis_command(
    out_dir: str,
    config: Dict,
    *,
    predictions: Optional[str] = None,
    horizon_hours: Optional[int] = None,
    analysis_script: str = ANALYSIS_SCRIPT,
    python_exe: Optional[str] = None,
) -> List[str]:
    """Argument list for the post-hoc analysis script."""
    cfg = {**DEFAULT_CONFIG, **config}
    py = python_exe or sys.executable
    preds = predictions or str(Path(out_dir) / "predictions.csv")
    cmd = [py, analysis_script,
           "--predictions", preds,
           "--out-dir", str(out_dir),
           "--epochs", str(cfg["epochs"]),
           "--dropout", str(cfg["dropout"]),
           "--mc-samples", str(cfg["mc_samples"])]
    if cfg.get("train_normal_only"):
        cmd.append("--train-normal-only")
    if horizon_hours is not None:
        horizon = int(horizon_hours)
        if horizon < 1:
            raise ValueError("horizon_hours must be a positive integer.")
        cmd += ["--horizon-hours", str(horizon)]
    return cmd


# --------------------------------------------------------------------------- #
# W&B sweep
# --------------------------------------------------------------------------- #
def make_sweep_config(
    sweep_parameters: Dict,
    *,
    program: str = SWEEP_MEMBER_SCRIPT,
    method: str = "grid",
    metric_name: str = "posthoc/daytime_picp",
    metric_goal: str = "maximize",
) -> Dict:
    """Build a real W&B sweep config dict driving the sweep-member wrapper."""
    return {
        "program": program,
        "method": method,
        "metric": {"name": metric_name, "goal": metric_goal},
        "parameters": dict(sweep_parameters),
    }
