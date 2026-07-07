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

import sys
from pathlib import Path
from typing import Dict, List, Optional

from physiq_pv.reporting.posthoc_outputs import (
    POSTHOC_KEYS,
    PRODUCTION_BINS,
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
WANDB_PROJECT = "PhysiQ-PV"
WANDB_ENTITY = "albertopedalino-politecnico-di-torino"


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
    "lr": 0.001,
    "dropout": 0.3,
    "mc_samples": 20,
    "seed": 1,
    "n_sde_steps": 4,
    "sigma_max": 0.5,  # Monaco SDE U-Net repo: self.sigma = 0.5
    "ood_noise_std": 1.0,
    "irradiance_loss_weight": 0.1,
    "pv_target_clip_max": "none",
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
    if config.get("name"):
        return f"pvgis_stgnn_{config['name']}_seed{seed}"
    return (
        f"pvgis_stgnn_sde{_tag(config.get('n_sde_steps', 4))}"
        f"_sm{_tag(config.get('sigma_max', 0.5))}"
        f"_ood{_tag(config.get('ood_noise_std', 1.0))}"
        f"_seed{seed}"
    )


def make_out_dir(config: Dict, root: str = "outputs") -> str:
    """`<root>/<run_name>` — deterministic, so re-running the same config reuses
    the same dir; distinct configs (incl. seed) never collide."""
    return f"{root}/{make_run_name(config)}"


def _value_flags(config: Dict) -> List[tuple]:
    """(cli_flag, config_key) pairs for the value-taking runner arguments."""
    return [
        ("--train-years", "train_years"),
        ("--test-year", "test_year"),
        ("--seq-len", "seq_len"),
        ("--horizon", "horizon"),
        ("--target-variable", "target_variable"),
        ("--model-type", "model_type"),
        ("--feature-set", "feature_set"),
        ("--irradiance-loss-weight", "irradiance_loss_weight"),
        ("--epochs", "epochs"),
        ("--batch-size", "batch_size"),
        ("--lr", "lr"),
        ("--dropout", "dropout"),
        ("--mc-samples", "mc_samples"),
        ("--pv-target-clip-max", "pv_target_clip_max"),
        ("--seed", "seed"),
        ("--n-sde-steps", "n_sde_steps"),
        ("--sigma-max", "sigma_max"),
        ("--ood-noise-std", "ood_noise_std"),
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
        cmd += [flag, str(cfg[key])]
    # boolean store_true flags
    cmd += ["--use-irradiance-head", "--use-irradiance-loss", "--sde-uncertainty"]
    if cfg.get("train_normal_only"):
        # Paper-style normal-only protocol: train only on fully normal windows.
        cmd += ["--train-normal-only",
                "--train-anomaly-scores", str(train_anomaly_scores)]
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
