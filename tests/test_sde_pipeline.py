"""Light tests for the SDE pipeline orchestration helpers.

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

from physiq_pv.experiments.sde_pipeline import (  # noqa: E402
    DEFAULT_CONFIG,
    POSTHOC_KEYS,
    build_analysis_command,
    build_train_command,
    collect_run_artifact_files,
    log_posthoc_to_wandb,
    make_out_dir,
    make_run_name,
    make_sweep_config,
    read_posthoc_summary,
)
from scripts.run_pvgis_climatology_anomaly_years import (  # noqa: E402
    default_aggregate_dir,
    effective_climatology_end_year,
    year_out_dir,
)


def test_build_train_command_has_required_flags() -> None:
    cmd = build_train_command(DEFAULT_CONFIG, out_dir="outputs/x", run_name="run1")
    assert isinstance(cmd, list) and all(isinstance(c, str) for c in cmd)
    assert cmd[1:3] == ["-m", "physiq_pv.experiments.pvgis_stgnn_runner"]
    # value flags resolved from config
    for flag, val in [
        ("--epochs", "60"), ("--n-sde-steps", "4"), ("--sigma-max", "0.5"),
        ("--sde-sigma-initial", "0.01"), ("--sde-sigma-warmup-epochs", "30"),
        ("--ood-noise-std", "2.0"), ("--lr-g", "0.01"), ("--out-dir", "outputs/x"),
        ("--beta-nll", "0.5"), ("--nll-dist", "student_t"),
        ("--student-t-nu", "5.0"), ("--student-t-samples-per-path", "64"),
        ("--gradient-clip-norm", "100.0"), ("--lr-decay-epoch", "20"),
        ("--lr-decay-factor", "0.1"),
    ]:
        assert flag in cmd, flag
        assert cmd[cmd.index(flag) + 1] == val, (flag, cmd[cmd.index(flag) + 1])
    # store_true flags present
    for f in ("--use-irradiance-head", "--use-irradiance-loss", "--sde-uncertainty"):
        assert f in cmd, f
    # wandb on by default
    assert "--wandb" in cmd and "--wandb-run-name" in cmd
    assert "--no-wandb-upload-artifacts" in cmd


def test_default_anomaly_paths_are_past_only() -> None:
    cmd = build_train_command(
        {**DEFAULT_CONFIG, "train_normal_only": True},
        out_dir="outputs/x",
        run_name="run1",
        use_wandb=False,
    )
    assert "outputs/pvgis_anomaly_2019_2005_2018_w15_q0975/pvgis_climatology_scores.csv" in cmd
    assert (
        "outputs/pvgis_anomaly_train_2016_2018_2005_past_w15_q0975/"
        "pvgis_climatology_scores.csv"
    ) in cmd


def test_rolling_past_output_names() -> None:
    assert effective_climatology_end_year(2016, 2005, 2018, True) == 2015
    assert effective_climatology_end_year(2018, 2005, 2018, True) == 2017
    assert effective_climatology_end_year(2019, 2005, 2018, True) == 2018
    assert effective_climatology_end_year(2016, 2005, 2018, False) == 2018
    p2016 = year_out_dir("outputs", 2016, 2005, 2015, 15, 0.975)
    p2018 = year_out_dir("outputs", 2018, 2005, 2017, 15, 0.975)
    agg = default_aggregate_dir(
        "outputs", [2016, 2017, 2018], 2005, 2018, 15, 0.975,
        rolling_past_climatology=True,
    )
    assert str(p2016).replace("\\", "/").endswith("pvgis_anomaly_2016_2005_2015_w15_q0975")
    assert str(p2018).replace("\\", "/").endswith("pvgis_anomaly_2018_2005_2017_w15_q0975")
    assert str(agg).replace("\\", "/").endswith("pvgis_anomaly_train_2016_2018_2005_past_w15_q0975")


def test_build_train_command_wandb_off() -> None:
    cmd = build_train_command(DEFAULT_CONFIG, out_dir="o", run_name="r", use_wandb=False)
    assert "--wandb" not in cmd
    assert "--wandb-run-name" not in cmd
    assert "--no-wandb-upload-artifacts" not in cmd


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
    assert cmd[1].endswith("analyze_pvgis_daytime_report.py")
    assert "--predictions" in cmd
    assert cmd[cmd.index("--predictions") + 1].replace("\\", "/").endswith(
        "outputs/x/predictions.csv"
    )
    assert cmd[cmd.index("--out-dir") + 1] == "outputs/x"
    assert cmd[cmd.index("--mc-samples") + 1] == "10"
    assert "--train-normal-only" not in cmd


def test_build_analysis_command_marks_train_normal_only() -> None:
    cmd = build_analysis_command(
        "outputs/x",
        {**DEFAULT_CONFIG, "train_normal_only": True},
    )
    assert "--train-normal-only" in cmd


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
    assert s["posthoc/high_production_picp"] == 0.91
    assert s["posthoc/gt100_picp"] == 0.91


def test_read_posthoc_summary_missing_files(tmp_path: Path) -> None:
    s = read_posthoc_summary(str(tmp_path))
    assert set(s.keys()) == set(POSTHOC_KEYS)
    assert all(v != v for v in s.values())  # all NaN, never raises


def test_collect_run_artifact_files_excludes_predictions_by_default(tmp_path: Path) -> None:
    for name in [
        "report.md",
        "daytime_bin_anomaly_report.md",
        "metrics.json",
        "sharpness_overview.csv",
        "predictions.csv",
    ]:
        (tmp_path / name).write_text("x", encoding="utf-8")
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()
    (fig_dir / "coverage.png").write_bytes(b"fake")

    files = collect_run_artifact_files(str(tmp_path))
    names = {p.name for p in files}
    assert "predictions.csv" not in names
    assert {
        "report.md",
        "daytime_bin_anomaly_report.md",
        "metrics.json",
        "sharpness_overview.csv",
        "coverage.png",
    } <= names


class _FakeArtifact:
    def __init__(self, name: str, type: str) -> None:  # noqa: A002 - mirrors W&B
        self.name = name
        self.type = type
        self.files: list[tuple[str, str | None]] = []

    def add_file(self, path: str, name: str | None = None) -> None:
        self.files.append((path, name))


class _FakeImage:
    def __init__(self, path: str) -> None:
        self.path = path


class _FakeWandb:
    Artifact = _FakeArtifact
    Image = _FakeImage


class _FakeRun:
    id = "run123"

    def __init__(self) -> None:
        self.logs: list[dict] = []
        self.summary: dict = {}
        self.artifacts: list[_FakeArtifact] = []

    def log(self, payload: dict) -> None:
        self.logs.append(payload)

    def log_artifact(self, artifact: _FakeArtifact) -> None:
        self.artifacts.append(artifact)


def test_log_posthoc_to_wandb_logs_scalars_figures_and_artifact(tmp_path: Path) -> None:
    pd.DataFrame({
        "scope": ["overall_daytime", "normal", "rare_extreme",
                  "unusually_low_solar_potential"],
        "picp": [0.94, 0.95, 0.80, 0.70],
        "mpiw": [10.0, 9.5, 14.0, 16.0],
        "nmpil": [0.05, 0.048, 0.07, 0.08],
    }).to_csv(tmp_path / "sharpness_overview.csv", index=False)
    pd.DataFrame({
        "bin": ["daytime_gt_100"],
        "picp": [0.91],
    }).to_csv(tmp_path / "daytime_bin_summary.csv", index=False)
    (tmp_path / "daytime_bin_anomaly_report.md").write_text("report", encoding="utf-8")
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()
    fig = fig_dir / "coverage.png"
    fig.write_bytes(b"fake")

    run = _FakeRun()
    result = log_posthoc_to_wandb(_FakeWandb, run, str(tmp_path))

    assert run.logs[0]["posthoc/daytime_picp"] == 0.94
    assert run.summary["posthoc/gt100_picp"] == 0.91
    assert any("figures/coverage" in payload for payload in run.logs)
    assert result["artifact_uploaded"] is True
    assert result["figures_logged"] == ["coverage"]
    assert len(run.artifacts) == 1
    artifact_names = {name for _, name in run.artifacts[0].files}
    assert "daytime_bin_anomaly_report.md" in artifact_names
    assert "figures/coverage.png" in artifact_names


def test_make_sweep_config_structure() -> None:
    params = {
        "n_sde_steps": {"values": [4, 6]},
        "sigma_max": {"values": [0.3, 0.5]},
    }
    cfg = make_sweep_config(params)
    assert cfg["method"] == "grid"
    assert cfg["program"].endswith("run_pvgis_sde_sweep_member.py")
    assert cfg["metric"]["name"] == "posthoc/daytime_picp"
    assert cfg["parameters"] == params


if __name__ == "__main__":
    import tempfile

    test_build_train_command_has_required_flags()
    test_default_anomaly_paths_are_past_only()
    test_rolling_past_output_names()
    test_build_train_command_wandb_off()
    test_make_out_dir_deterministic_and_seed_unique()
    test_make_run_name_explicit_name()
    test_build_analysis_command()
    test_build_analysis_command_marks_train_normal_only()
    with tempfile.TemporaryDirectory() as d:
        test_read_posthoc_summary(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_read_posthoc_summary_missing_files(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_collect_run_artifact_files_excludes_predictions_by_default(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_log_posthoc_to_wandb_logs_scalars_figures_and_artifact(Path(d))
    test_make_sweep_config_structure()
    print("PASS: SDE pipeline tests")
