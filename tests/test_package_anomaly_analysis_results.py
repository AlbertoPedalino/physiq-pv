from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.package_anomaly_analysis_results import (
    QUALITY_POLICY,
    SDE_RUN_NAME,
    build_bundle,
)


def _write_figure(directory: Path, name: str = "figure.png") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(b"synthetic-png")


def test_bundle_contains_figures_and_only_compact_clean_reports(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    mtgflow = outputs / SDE_RUN_NAME
    _write_figure(mtgflow / "figures", "pipeline.png")
    _write_figure(mtgflow / "figures/events/t_plus_1/extreme_events", "event.png")
    _write_figure(mtgflow / "figures/events/t_plus_6/extreme_events", "event.png")
    (mtgflow / "metrics.csv").write_text("mae\n1\n", encoding="utf-8")

    stgan = outputs / "sde_stgan_direct_multihorizon_seed20_quality_filtered"
    _write_figure(stgan / "posthoc_by_horizon/t_plus_1/figures")
    _write_figure(stgan / "posthoc_by_horizon/t_plus_6/figures")
    _write_figure(stgan / "stgan_may08_may17_t1_t6_pipeline_style/figures")
    (stgan / "evaluation_source.json").write_text(
        json.dumps(
            {
                "detector": "stgan",
                "quality_filter_policy": QUALITY_POLICY,
                "clean_top_k_percent": 1.0,
                "data_quality_timestamps": 18,
                "excluded_data_quality_rows": 100,
            }
        ),
        encoding="utf-8",
    )
    (stgan / "pvgis_data_quality_issues.csv").write_text(
        "timestamp\n2019-01-01\n", encoding="utf-8"
    )
    (stgan / "predictions.csv").write_text("must,not,ship\n", encoding="utf-8")

    spatial = outputs / "anomaly_spatial_comparison_quality_filtered"
    _write_figure(spatial / "figures")
    (spatial / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "stgan_quality_filter_policy": QUALITY_POLICY,
                "stgan_clean_top_k_percent": 1.0,
            }
        ),
        encoding="utf-8",
    )

    threshold = outputs / "anomaly_threshold_sensitivity_t1_t6"
    _write_figure(threshold / "figures")
    (threshold / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "quality_filter_policy": QUALITY_POLICY,
                "excluded_quality_timestamps": 18,
                "stgan_ranking": "global ranking recomputed after quality exclusion",
            }
        ),
        encoding="utf-8",
    )

    destination = build_bundle(tmp_path)
    with ZipFile(destination) as archive:
        names = set(archive.namelist())
        metadata = json.loads(archive.read("bundle_metadata.json"))
    assert "bundle_manifest.csv" in names
    assert metadata["figure_count"] == 8
    assert metadata["data_quality_timestamps"] == 18
    assert any(name.endswith("metrics.csv") for name in names)
    expected_notebook_folders = {
        "01_pvgis_sde_pipeline_mtgflow",
        "02_pvgis_sde_extreme_events",
        "03_stgan_pointwise_posthoc_sdenet",
        "04_stgan_may08_may17_t1_t6",
        "05_spatial_anomaly_comparison_mtgflow_stgan",
        "06_anomaly_threshold_sensitivity_mtgflow_stgan",
    }
    present_notebook_folders = {
        name.split("/")[1]
        for name in names
        if name.startswith("notebooks/") and len(name.split("/")) > 2
    }
    assert present_notebook_folders == expected_notebook_folders
    assert not any(name.endswith("predictions.csv") for name in names)
    assert not any("pvgis_stgan/paper_reference" in name for name in names)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        test_bundle_contains_figures_and_only_compact_clean_reports(Path(directory))
    print("PASS: quality-filtered anomaly analysis bundle")
