"""Light tests for the SDE-proxy pipeline orchestration helpers.

No real training: only command construction, path naming, post-hoc CSV reading
and sweep-config generation are exercised.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physiq_pv.experiments.sde_proxy_pipeline import (  # noqa: E402
    DEFAULT_CONFIG,
    POSTHOC_KEYS,
    build_analysis_command,
    build_train_command,
    iter_manual_sweep,
    make_out_dir,
    make_run_name,
    make_sweep_config,
    read_posthoc_summary,
)


def test_build_train_command_has_required_flags() -> None:
    cmd = build_train_command(DEFAULT_CONFIG, out_dir="outputs/x", run_name="run1")
    assert isinstance(cmd, list) and all(isinstance(c, str) for c in cmd)
    assert cmd[1:3] == ["-m", "physiq_pv.experiments.pvgis_stgnn_runner"]
    # value flags resolved from config
    for flag, val in [
        ("--loss-type", "huber"), ("--huber-delta", "0.1"),
        ("--uncertainty-penalty-mode", "sde_proxy"),
        ("--sde-proxy-in-weight", "5e-05"), ("--train-noise-mode", "anomaly"),
        ("--anomaly-noise-prob", "0.7"), ("--out-dir", "outputs/x"),
    ]:
        assert flag in cmd, flag
        assert cmd[cmd.index(flag) + 1] == val, (flag, cmd[cmd.index(flag) + 1])
    # store_true flags present
    for f in ("--use-irradiance-head", "--use-irradiance-loss", "--mc-dropout",
              "--train-mc-uncertainty-penalty", "--skip-posthoc-analysis"):
        assert f in cmd, f
    # wandb on by default
    assert "--wandb" in cmd and "--wandb-run-name" in cmd


def test_build_train_command_wandb_off() -> None:
    cmd = build_train_command(DEFAULT_CONFIG, out_dir="o", run_name="r", use_wandb=False)
    assert "--wandb" not in cmd
    assert "--wandb-run-name" not in cmd


def test_make_out_dir_deterministic_and_seed_unique() -> None:
    a = make_out_dir(DEFAULT_CONFIG)
    b = make_out_dir(DEFAULT_CONFIG)
    assert a == b
    assert a.startswith("outputs/")
    assert "seed1" in a
    c = make_out_dir({**DEFAULT_CONFIG, "seed": 2})
    assert c != a and "seed2" in c


def test_make_run_name_explicit_name() -> None:
    name = make_run_name({**DEFAULT_CONFIG, "name": "abl", "seed": 3})
    assert name == "pvgis_stgnn_abl_seed3"


def test_build_analysis_command() -> None:
    cmd = build_analysis_command("outputs/x", DEFAULT_CONFIG)
    assert cmd[1].endswith("analyze_pvgis_huber_daytime_report.py")
    assert "--predictions" in cmd
    assert cmd[cmd.index("--predictions") + 1].replace("\\", "/").endswith(
        "outputs/x/predictions.csv"
    )
    assert cmd[cmd.index("--out-dir") + 1] == "outputs/x"
    assert cmd[cmd.index("--mc-samples") + 1] == "20"


def test_read_posthoc_summary(tmp_path: Path) -> None:
    pd.DataFrame({
        "scope": ["overall_daytime", "normal", "rare_extreme",
                  "unusually_low_solar_potential"],
        "count": [100, 80, 20, 5],
        "picp": [0.94, 0.95, 0.80, 0.70],
        "mae": [1.0, 0.9, 2.0, 3.0], "rmse": [1.5, 1.4, 2.5, 3.5],
        "mean_std": [0.2, 0.2, 0.3, 0.3],
        "mpiw": [10.0, 9.5, 14.0, 16.0], "nmpil": [0.05, 0.048, 0.07, 0.08],
        "target_range": [200.0] * 4,
    }).to_csv(tmp_path / "sharpness_overview.csv", index=False)
    pd.DataFrame({
        "bin": ["daytime_0_20", "daytime_gt_100"],
        "count": [50, 30], "mae": [0.5, 2.0], "rmse": [0.7, 2.5],
        "picp": [0.96, 0.91], "mean_std": [0.1, 0.4],
        "mpiw": [5.0, 18.0], "nmpil": [0.025, 0.09],
    }).to_csv(tmp_path / "daytime_bin_summary.csv", index=False)

    s = read_posthoc_summary(str(tmp_path))
    assert set(s.keys()) == set(POSTHOC_KEYS)
    assert s["posthoc/daytime_picp"] == 0.94
    assert s["posthoc/normal_picp"] == 0.95
    assert s["posthoc/rare_extreme_picp"] == 0.80
    assert s["posthoc/unusually_low_picp"] == 0.70
    assert s["posthoc/daytime_mpiw"] == 10.0
    assert s["posthoc/daytime_nmpil"] == 0.05
    assert s["posthoc/gt100_picp"] == 0.91


def test_read_posthoc_summary_missing_files(tmp_path: Path) -> None:
    s = read_posthoc_summary(str(tmp_path))
    assert set(s.keys()) == set(POSTHOC_KEYS)
    assert all(v != v for v in s.values())  # all NaN, never raises


def test_iter_manual_sweep_overrides() -> None:
    sweeps = [
        {"name": "a", "sde_proxy_out_weight": 0.3},
        {"name": "b", "sde_proxy_out_weight": 0.5},
    ]
    out = list(iter_manual_sweep(DEFAULT_CONFIG, sweeps))
    assert [n for n, _ in out] == ["pvgis_stgnn_a_seed1", "pvgis_stgnn_b_seed1"]
    assert out[0][1]["sde_proxy_out_weight"] == 0.3
    assert out[1][1]["sde_proxy_out_weight"] == 0.5
    # base untouched
    assert DEFAULT_CONFIG["sde_proxy_out_weight"] == 0.5


def test_make_sweep_config_structure() -> None:
    params = {
        "sde_proxy_in_weight": {"values": [0.00005, 0.0001]},
        "sde_proxy_out_weight": {"values": [0.3, 0.5]},
    }
    cfg = make_sweep_config(params)
    assert cfg["method"] == "grid"
    assert cfg["program"].endswith("run_pvgis_sde_proxy_sweep_member.py")
    assert cfg["metric"]["name"] == "posthoc/daytime_picp"
    assert cfg["parameters"] == params
