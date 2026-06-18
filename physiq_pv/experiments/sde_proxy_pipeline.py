"""Reproducible orchestration helpers for the PVGIS-only SDE-proxy pipeline.

Pure, importable helpers shared by:
  * notebooks/pvgis_sde_proxy_pipeline.ipynb
  * scripts/run_pvgis_sde_proxy_sweep_member.py
  * tests/test_sde_proxy_pipeline.py

They ONLY build commands / paths and read the CSVs that the runner and the
analysis script already write. No training, model, loss or report logic is
duplicated here — the runner and the analysis script remain the single source
of truth.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Defaults (server newzealand)
# --------------------------------------------------------------------------- #
PVGIS_DIR = "/data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance"
TEST_ANOMALY_SCORES = (
    "outputs/pvgis_anomaly_2019_2005_2023_w15_q0975/pvgis_climatology_scores.csv"
)
TRAIN_ANOMALY_SCORES = (
    "outputs/pvgis_anomaly_train_2016_2018_2005_2023_w15_q0975/"
    "pvgis_climatology_scores.csv"
)
RUNNER_MODULE = "physiq_pv.experiments.pvgis_stgnn_runner"
ANALYSIS_SCRIPT = "scripts/analyze_pvgis_huber_daytime_report.py"
SWEEP_MEMBER_SCRIPT = "scripts/run_pvgis_sde_proxy_sweep_member.py"
WANDB_PROJECT = "PhysiQ-PV"
WANDB_ENTITY = "albertopedalino-politecnico-di-torino"

# The single-run base config (mirrors the documented final run).
DEFAULT_CONFIG: Dict = {
    "train_years": "2016,2017,2018",
    "test_year": 2019,
    "seq_len": 24,
    "horizon": 1,
    "target_variable": "pv_power_output",
    "model_type": "stgnn_enhanced_dropout",
    "feature_set": "full",
    "epochs": 5,
    "batch_size": 16,
    "lr": 0.001,
    "dropout": 0.3,
    "mc_samples": 20,
    "seed": 1,
    "loss_type": "huber",
    "huber_delta": 0.1,
    "train_mc_samples": 5,
    "uncertainty_penalty_mode": "sde_proxy",
    "sde_proxy_in_weight": 0.00005,
    "sde_proxy_out_weight": 0.5,
    "sde_proxy_std_min_ood": 0.10,
    "train_noise_mode": "anomaly",
    "train_noise_std": 0.0,
    "train_noise_prob": 0.0,
    "anomaly_noise_std": 0.05,
    "anomaly_noise_prob": 0.7,
    "irradiance_loss_weight": 0.1,
    "pv_target_clip_max": "none",
}

# Daytime production bins (physical watt y_true), same convention as the runner
# and the analysis script. Used by the notebook boxplots.
PRODUCTION_BINS = [
    ("daytime_0_20", 0.0, 20.0),
    ("daytime_20_40", 20.0, 40.0),
    ("daytime_40_60", 40.0, 60.0),
    ("daytime_60_80", 60.0, 80.0),
    ("daytime_80_100", 80.0, 100.0),
    ("daytime_gt_100", 100.0, None),
]

# Per-scope / per-bin keys logged to W&B from the post-hoc CSVs.
POSTHOC_KEYS = (
    "posthoc/daytime_picp",
    "posthoc/daytime_mpiw",
    "posthoc/daytime_nmpil",
    "posthoc/normal_picp",
    "posthoc/rare_extreme_picp",
    "posthoc/unusually_low_picp",
    "posthoc/gt100_picp",
)


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
    loss / penalty / sde-proxy / anomaly-noise / seed values.
    """
    seed = config.get("seed", 0)
    if config.get("name"):
        return f"pvgis_stgnn_{config['name']}_seed{seed}"
    return (
        f"pvgis_stgnn_{config.get('loss_type', 'huber')}"
        f"_d{_tag(config.get('huber_delta', 0.1))}"
        f"_{config.get('uncertainty_penalty_mode', 'sde_proxy')}"
        f"_in{_tag(config.get('sde_proxy_in_weight', 0.0))}"
        f"_out{_tag(config.get('sde_proxy_out_weight', 0.0))}"
        f"_ood{_tag(config.get('sde_proxy_std_min_ood', 0.0))}"
        f"_an{_tag(config.get('anomaly_noise_std', 0.0))}"
        f"_ap{_tag(config.get('anomaly_noise_prob', 0.0))}"
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
        ("--loss-type", "loss_type"),
        ("--huber-delta", "huber_delta"),
        ("--train-mc-samples", "train_mc_samples"),
        ("--uncertainty-penalty-mode", "uncertainty_penalty_mode"),
        ("--sde-proxy-in-weight", "sde_proxy_in_weight"),
        ("--sde-proxy-out-weight", "sde_proxy_out_weight"),
        ("--sde-proxy-std-min-ood", "sde_proxy_std_min_ood"),
        ("--train-noise-mode", "train_noise_mode"),
        ("--train-noise-std", "train_noise_std"),
        ("--train-noise-prob", "train_noise_prob"),
        ("--anomaly-noise-std", "anomaly_noise_std"),
        ("--anomaly-noise-prob", "anomaly_noise_prob"),
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
    skip_posthoc: bool = True,
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
                      "--anomaly-scores", str(test_anomaly_scores),
                      "--train-anomaly-scores", str(train_anomaly_scores)]
    for flag, key in _value_flags(cfg):
        cmd += [flag, str(cfg[key])]
    # boolean store_true flags
    cmd += ["--use-irradiance-head", "--use-irradiance-loss",
            "--mc-dropout", "--train-mc-uncertainty-penalty"]
    if skip_posthoc:
        cmd += ["--skip-posthoc-analysis"]
    cmd += ["--device", str(device), "--out-dir", str(out_dir)]
    if use_wandb:
        cmd += ["--wandb",
                "--wandb-project", str(wandb_project),
                "--wandb-entity", str(wandb_entity),
                "--wandb-run-name", str(run_name)]
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
    return [py, analysis_script,
            "--predictions", preds,
            "--out-dir", str(out_dir),
            "--loss-type", str(cfg["loss_type"]),
            "--huber-delta", str(cfg["huber_delta"]),
            "--epochs", str(cfg["epochs"]),
            "--dropout", str(cfg["dropout"]),
            "--mc-samples", str(cfg["mc_samples"])]


# --------------------------------------------------------------------------- #
# Reading the post-hoc CSVs (eval-only; never recomputes anything)
# --------------------------------------------------------------------------- #
def _scope_value(df, scope_col: str, scope: str, value_col: str) -> float:
    import math
    hit = df[df[scope_col] == scope]
    if len(hit) == 0 or value_col not in hit.columns:
        return float("nan")
    try:
        return float(hit.iloc[0][value_col])
    except (TypeError, ValueError):
        return math.nan


def read_posthoc_summary(out_dir: str) -> Dict[str, float]:
    """Read the analysis-script CSVs and return the W&B `posthoc/*` scalars.

    Per-scope PICP/MPIW/NMPIL come from sharpness_overview.csv; the gt100 PICP
    from daytime_bin_summary.csv. Missing files/rows yield NaN (never raises).
    """
    import pandas as pd

    out = Path(out_dir)
    summary: Dict[str, float] = {k: float("nan") for k in POSTHOC_KEYS}

    sharp_path = out / "sharpness_overview.csv"
    if sharp_path.exists():
        s = pd.read_csv(sharp_path)
        summary["posthoc/daytime_picp"] = _scope_value(s, "scope", "overall_daytime", "picp")
        summary["posthoc/daytime_mpiw"] = _scope_value(s, "scope", "overall_daytime", "mpiw")
        summary["posthoc/daytime_nmpil"] = _scope_value(s, "scope", "overall_daytime", "nmpil")
        summary["posthoc/normal_picp"] = _scope_value(s, "scope", "normal", "picp")
        summary["posthoc/rare_extreme_picp"] = _scope_value(s, "scope", "rare_extreme", "picp")
        summary["posthoc/unusually_low_picp"] = _scope_value(
            s, "scope", "unusually_low_solar_potential", "picp"
        )

    bins_path = out / "daytime_bin_summary.csv"
    if bins_path.exists():
        b = pd.read_csv(bins_path)
        summary["posthoc/gt100_picp"] = _scope_value(b, "bin", "daytime_gt_100", "picp")

    return summary


def load_prediction_sample(predictions_path, max_rows: int = 500_000, random_state: int = 1):
    """Load a (capped) sample of predictions.csv without forcing all of it into
    RAM. Counts rows first; returns the whole file when small, else a uniform
    random sample of ~`max_rows` rows drawn chunk-by-chunk."""
    import pandas as pd

    path = Path(predictions_path)
    # row count (minus header), streamed
    with open(path, "r", encoding="utf-8") as fh:
        total = sum(1 for _ in fh) - 1
    if total <= 0:
        return pd.read_csv(path)
    if total <= max_rows:
        return pd.read_csv(path)
    frac = max_rows / total
    parts = []
    for chunk in pd.read_csv(path, chunksize=200_000):
        parts.append(chunk.sample(frac=frac, random_state=random_state))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(n=min(max_rows, len(out)), random_state=random_state).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------------- #
def iter_manual_sweep(base_config: Dict, sweep_configs: List[Dict]):
    """Yield (run_name, merged_config) for a manual notebook sweep. Each entry in
    `sweep_configs` overrides `base_config`; an optional `name` is preserved."""
    for entry in sweep_configs:
        merged = {**base_config, **entry}
        yield make_run_name(merged), merged


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
