"""Light tests for the SDE pipeline orchestration helpers.

No real training: only command construction, path naming, post-hoc CSV reading
and sweep-config generation are exercised.
"""
from __future__ import annotations

import json
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
    ensure_output_dir_available,
    log_posthoc_to_wandb,
    make_out_dir,
    make_run_name,
    make_sweep_config,
    read_posthoc_summary,
    relabel_detector_predictions,
    relabel_detector_predictions_file,
)
from scripts.run_pvgis_climatology_anomaly_years import (  # noqa: E402
    default_aggregate_dir,
    effective_climatology_end_year,
    year_out_dir,
)
from physiq_pv.reporting.daytime_bin_anomaly_report import (  # noqa: E402
    resolve_columns,
)
from physiq_pv.reporting.run_metrics import (  # noqa: E402
    build_wandb_metrics,
    compute_metrics,
)
from physiq_pv.data.pvgis_dataset import resolve_feature_set  # noqa: E402


def test_build_train_command_has_required_flags() -> None:
    cmd = build_train_command(DEFAULT_CONFIG, out_dir="outputs/x", run_name="run1")
    assert isinstance(cmd, list) and all(isinstance(c, str) for c in cmd)
    assert cmd[1:3] == ["-m", "physiq_pv.experiments.pvgis_stgnn_runner"]
    assert DEFAULT_CONFIG["feature_set"] == "no_pv_lag"
    features = resolve_feature_set(DEFAULT_CONFIG["feature_set"])
    assert len(features) == 10
    assert "pv_lag_pvgis" not in features
    # value flags resolved from config
    for flag, val in [
        ("--epochs", "60"), ("--n-sde-steps", "4"), ("--sigma-max", "0.5"),
        ("--sde-sigma-initial", "0.01"), ("--sde-sigma-warmup-epochs", "30"),
        ("--ood-noise-std", "2.0"), ("--ood-smoke-max-samples", "2048"),
        ("--lr-g", "0.01"), ("--lr", "0.0001"), ("--batch-size", "16"),
        ("--dropout", "0.0"), ("--out-dir", "outputs/x"),
        ("--feature-set", "no_pv_lag"),
        ("--gradient-clip-norm", "100.0"), ("--lr-decay-epoch", "20"),
        ("--lr-decay-factor", "0.1"),
        ("--anomaly-source", "climatology"),
        ("--event-spatial-quantile", "0.99"),
        ("--event-tail-quantile", "0.975"),
        ("--detector-regional-quantile", "0.975"),
        ("--detector-min-temporal-coverage", "0.95"),
    ]:
        assert flag in cmd, flag
        assert cmd[cmd.index(flag) + 1] == val, (flag, cmd[cmd.index(flag) + 1])
    # store_true flags present
    for f in (
        "--use-irradiance-head", "--use-irradiance-loss", "--sde-uncertainty",
        "--ood-smoke-test",
    ):
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


def test_detector_source_is_forwarded_with_and_without_normal_only() -> None:
    detector = {
        **DEFAULT_CONFIG,
        "anomaly_source": "detector",
        "detector_regional_quantile": 0.975,
    }
    all_data = build_train_command(
        detector,
        out_dir="outputs/all",
        run_name="all",
        test_anomaly_scores="outputs/mtgflow_test.csv",
        train_anomaly_scores="outputs/mtgflow_train.csv",
        use_wandb=False,
    )
    assert "--train-normal-only" not in all_data
    assert (
        all_data[all_data.index("--train-anomaly-scores") + 1]
        == "outputs/mtgflow_train.csv"
    )
    assert all_data[all_data.index("--anomaly-source") + 1] == "detector"
    assert (
        all_data[all_data.index("--detector-regional-quantile") + 1]
        == "0.975"
    )

    normal_only = build_train_command(
        {**detector, "train_normal_only": True},
        out_dir="outputs/normal",
        run_name="normal",
        test_anomaly_scores="outputs/mtgflow_test.csv",
        train_anomaly_scores="outputs/mtgflow_train.csv",
        use_wandb=False,
    )
    assert "--train-normal-only" in normal_only
    assert (
        normal_only[normal_only.index("--train-anomaly-scores") + 1]
        == "outputs/mtgflow_train.csv"
    )


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


def test_output_guard_rejects_nonempty_directory(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    assert ensure_output_dir_available(out_dir) == out_dir
    out_dir.mkdir()
    assert ensure_output_dir_available(out_dir) == out_dir
    (out_dir / "predictions.csv").write_text("x", encoding="utf-8")
    try:
        ensure_output_dir_available(out_dir)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected non-empty output directory to be rejected.")
    assert ensure_output_dir_available(
        out_dir, allow_overwrite=True
    ) == out_dir


def _detector_relabel_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    times = pd.date_range("2019-01-01", periods=4, freq="h")
    prediction_rows = [
        {
            "location": location,
            "timestamp": timestamp,
            "anomaly_group": "stale",
            "anomaly_label": "stale",
            "y_true": 1.0,
            "y_pred": 0.5,
            "abs_error": 0.5,
            "squared_error": 0.25,
            "y_pred_std": 0.2,
            "y_pred_lower": 0.1,
            "y_pred_upper": 0.9,
        }
        for timestamp in times
        for location in ("a", "b")
    ]
    score_rows = [
        {
            "location": location,
            "timestamp": timestamp,
            "method": "catch",
            "anomaly_score": float(timestamp == times[2] and location == "a"),
            "threshold": 0.5,
            "is_anomaly": timestamp == times[2] and location == "a",
        }
        for timestamp in times
        for location in ("a", "b")
    ]
    return pd.DataFrame(prediction_rows), pd.DataFrame(score_rows)


def test_detector_relabel_uses_regional_event_groups() -> None:
    predictions, scores = _detector_relabel_frames()
    training_scores = scores.copy()
    training_scores["timestamp"] = training_scores["timestamp"].map(
        lambda value: value.replace(year=2016)
    )
    relabelled, protocol = relabel_detector_predictions(
        predictions,
        scores,
        training_scores,
        regional_quantile=0.975,
        min_temporal_coverage=1.0,
    )
    assert protocol["detector"] == "catch"
    assert (relabelled["anomaly_group"] == "rare_or_extreme").sum() == 1
    assert (relabelled["event_group"] == "rare_or_extreme").sum() == 2
    assert set(relabelled.loc[
        relabelled["event_group"] == "rare_or_extreme", "timestamp"
    ]) == {pd.Timestamp("2019-01-01 02:00")}


def test_detector_relabel_file_writes_audit_metadata(tmp_path: Path) -> None:
    predictions, scores = _detector_relabel_frames()
    source = tmp_path / "source_predictions.csv"
    score_path = tmp_path / "anomaly_scores.csv"
    training_score_path = tmp_path / "train_anomaly_scores.csv"
    training_scores = scores.copy()
    training_scores["timestamp"] = training_scores["timestamp"].map(
        lambda value: value.replace(year=2016)
    )
    predictions.to_csv(source, index=False)
    scores.to_csv(score_path, index=False)
    training_scores.to_csv(training_score_path, index=False)
    paths = relabel_detector_predictions_file(
        source,
        score_path,
        training_score_path,
        tmp_path / "catch_eval",
        regional_quantile=0.975,
        min_temporal_coverage=1.0,
    )
    assert paths["predictions"].is_file()
    assert paths["evaluation_source"].is_file()
    assert paths["metrics_global"].is_file()
    assert paths["metrics_by_anomaly_label"].is_file()
    assert paths["metrics"].is_file()
    assert paths["report"].is_file()
    written = pd.read_csv(paths["predictions"])
    assert (written["event_group"] == "rare_or_extreme").sum() == 2
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert "mae/event_normal" in metrics
    assert "mae/event_rare_extreme" in metrics


def test_daytime_report_prefers_regional_event_group(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "y_true": [1.0],
            "y_pred": [1.0],
            "y_pred_std": [0.1],
            "lower_pi": [0.8],
            "upper_pi": [1.2],
            "solar_irradiance_poa_target": [100.0],
            "event_group": ["rare_or_extreme"],
            "anomaly_group": ["normal"],
            "anomaly_label": [""],
        }
    ).to_csv(path, index=False)
    assert resolve_columns(str(path))["group"] == "event_group"


def test_wandb_metrics_include_regional_event_strata() -> None:
    predictions = pd.DataFrame(
        {
            "y_true": [1.0, 1.0, 3.0, 3.0],
            "abs_error": [1.0, 1.0, 2.0, 2.0],
            "squared_error": [1.0, 1.0, 4.0, 4.0],
            "anomaly_group": ["normal"] * 4,
            "anomaly_label": [""] * 4,
            "event_group": ["normal", "normal", "rare_or_extreme", "rare_or_extreme"],
            "y_pred_std": [0.5, 0.5, 1.0, 1.0],
            "y_pred_lower": [0.0] * 4,
            "y_pred_upper": [4.0] * 4,
        }
    )
    global_df, by_df = compute_metrics(predictions)
    metrics = build_wandb_metrics(global_df, by_df, sde_uncertainty=True)
    assert metrics["mae/event_normal"] == 1.0
    assert metrics["mae/event_rare_extreme"] == 2.0
    assert metrics["ratio/mae_event_rare_normal"] == 2.0
    assert metrics["uncertainty/ratio_event_rare_normal"] == 2.0


def test_runner_diagnostics_prefer_regional_event_groups() -> None:
    from physiq_pv.experiments.pvgis_stgnn_runner import (
        build_daytime_metrics,
        build_interval_metrics,
    )

    predictions = pd.DataFrame({
        "y_true": [1.0, 1.0, 3.0, 3.0],
        "y_pred": [1.0, 1.0, 1.0, 1.0],
        "y_pred_std": [0.5, 0.5, 1.0, 1.0],
        "solar_irradiance_poa_target": [100.0] * 4,
        "lower_pi": [0.0] * 4,
        "upper_pi": [4.0] * 4,
        "lower_gaussian": [0.0] * 4,
        "upper_gaussian": [4.0] * 4,
        # Local labels deliberately contradict the regional event labels.
        "anomaly_group": ["rare_or_extreme"] * 2 + ["normal"] * 2,
        "event_group": ["normal"] * 2 + ["rare_or_extreme"] * 2,
    })
    interval = build_interval_metrics(
        predictions, target_range=3.0, gamma=0.95, eta=9.0
    )
    daytime = build_daytime_metrics(
        predictions, target_range=3.0, gamma=0.95, eta=9.0
    )
    assert interval["pi"]["normal"]["picp"] == 1.0
    assert daytime["normal_daytime"]["mae"] == 0.0
    assert daytime["rare_extreme_daytime"]["mae"] == 2.0


def test_wandb_metadata_preserves_string_path(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from physiq_pv.experiments.pvgis_stgnn_runner import (
        _write_wandb_run_metadata,
    )

    run = SimpleNamespace(
        id="abc", name="run", url="https://example.invalid/run",
        path="entity/project/abc",
    )
    path = _write_wandb_run_metadata(
        str(tmp_path), run, project="project", entity="entity"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["path"] == "entity/project/abc"


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


def test_april_dust_notebook_is_valid_and_posthoc_only() -> None:
    notebook_path = (
        _REPO_ROOT
        / "notebooks"
        / "pvgis_sde_april_dust_event_analysis.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    cell_ids = [cell["id"] for cell in cells]
    assert len(cell_ids) == len(set(cell_ids))
    assert all(
        cell.get("execution_count") is None
        for cell in cells
        if cell["cell_type"] == "code"
    )
    source = "\n".join(
        "".join(cell["source"]) for cell in cells
    )
    assert "build_extreme_event_diagnostic" in source
    assert "build_extreme_event_comparison_figures" in source
    assert "2019-04-23" in source and "2019-04-26" in source
    assert "figure_subdir='april_dust_event'" in source
    assert (
        "pvgis_stgnn_paper_faithful_gaussian_no_pv_lag_"
        "detector_mtgflow_ep60_seed1"
    ) in source
    assert "build_train_command" not in source
    for cell in cells:
        if cell["cell_type"] == "code":
            compile(
                "".join(cell["source"]),
                f"{notebook_path}:{cell['id']}",
                "exec",
            )


def test_june_extreme_event_notebook_is_valid_and_posthoc_only() -> None:
    notebook_path = (
        _REPO_ROOT
        / "notebooks"
        / "pvgis_sde_june_extreme_event_analysis.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    cell_ids = [cell["id"] for cell in cells]
    assert len(cell_ids) == len(set(cell_ids))
    assert all(
        cell.get("execution_count") is None
        for cell in cells
        if cell["cell_type"] == "code"
    )
    source = "\n".join(
        "".join(cell["source"]) for cell in cells
    )
    assert "build_extreme_event_diagnostic" in source
    assert "build_extreme_event_comparison_figures" in source
    assert "2019-06-28" in source and "2019-06-30" in source
    assert "figure_subdir='june_extreme_event'" in source
    assert "comparison_name='june_extreme_28_29'" in source
    assert (
        "pvgis_stgnn_paper_faithful_gaussian_no_pv_lag_"
        "detector_mtgflow_ep60_seed1"
    ) in source
    assert "build_train_command" not in source
    for cell in cells:
        if cell["cell_type"] == "code":
            compile(
                "".join(cell["source"]),
                f"{notebook_path}:{cell['id']}",
                "exec",
            )

    pipeline_path = _REPO_ROOT / "notebooks" / "pvgis_sde_pipeline.ipynb"
    pipeline_source = pipeline_path.read_text(encoding="utf-8")
    assert "2019-06-28" not in pipeline_source
    assert "2019-06-29" not in pipeline_source


if __name__ == "__main__":
    import tempfile

    test_build_train_command_has_required_flags()
    test_default_anomaly_paths_are_past_only()
    test_detector_source_is_forwarded_with_and_without_normal_only()
    test_rolling_past_output_names()
    test_build_train_command_wandb_off()
    test_make_out_dir_deterministic_and_seed_unique()
    with tempfile.TemporaryDirectory() as d:
        test_output_guard_rejects_nonempty_directory(Path(d))
    test_detector_relabel_uses_regional_event_groups()
    with tempfile.TemporaryDirectory() as d:
        test_detector_relabel_file_writes_audit_metadata(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_daytime_report_prefers_regional_event_group(Path(d))
    test_wandb_metrics_include_regional_event_strata()
    test_runner_diagnostics_prefer_regional_event_groups()
    with tempfile.TemporaryDirectory() as d:
        test_wandb_metadata_preserves_string_path(Path(d))
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
    test_april_dust_notebook_is_valid_and_posthoc_only()
    test_june_extreme_event_notebook_is_valid_and_posthoc_only()
    print("PASS: SDE pipeline tests")
