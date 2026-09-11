"""Verify both notebook maps count night, quality exclusions and missing sites."""

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import xarray as xr
from IPython.display import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physiq_pv.reporting.detector_spatial_frequency import build_detector_spatial_frequency


def spatial_inputs(tmp_path):
    times = pd.date_range("2019-01-01 00:10", periods=6, freq="h")
    # Hour 0 is ordinary night. Hour 2 is an isolated dropout; hour 3 recovery.
    solar = np.tile([0., 10., 0., 10., 10., 10.], (3, 1))
    dataset = xr.Dataset(
        {name: (("location", "time"), solar) for name in (
            "direct_irradiance_tilted", "diffuse_irradiance_tilted",
            "sun_height", "pv_power_output",
        )},
        coords={"location": [0, 1, 2], "time": times,
                "lat": ("location", [45., 45.1, 45.2]),
                "lon": ("location", [7., 7.1, 7.2])},
    )
    pvgis = tmp_path / "pvgis.nc"
    dataset.to_netcdf(pvgis)
    # Location 1 lacks the last hour; location 2 has no scores at all.
    scores = pd.DataFrame({
        "location": ["0"] * 6 + ["1"] * 5,
        "timestamp": list(times) + list(times[:5]),
        "anomaly_score": [50., 10., 200., 100., 0., 0., 1., 0., 200., 100., 0.],
        "threshold": [10.] * 6 + [1.] * 5,
    })
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    score_path = seed_dir / "anomaly_scores.csv"
    scores.to_csv(score_path, index=False)
    return pvgis, score_path


def test_both_notebook_cells_and_frequency_semantics(tmp_path):
    pvgis, score_path = spatial_inputs(tmp_path)
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/mtgflow_spatiotemporal_anomaly_heatmaps.ipynb").read_text(encoding="utf-8"))
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    for detector, expected_counts in (("stgan", [1, 0, 0]), ("mtgflow", [2, 1, 0])):
        # Each cell executes independently after setup, with no SDE/training data.
        namespace = {"Path": Path, "os": os, "ROOT": root, "PVGIS_2019_PATH": pvgis,
                     "MTGFLOW_TEST_CSV": score_path, "OUT_DIR": tmp_path / "out",
                     "Image": Image, "display": lambda *args: None}
        code = "".join(next(c for c in notebook["cells"] if c["id"] == f"{detector}-all-hours-map")["source"])
        with patch.dict(os.environ, {"STGAN_SEED_DIR": str(score_path.parent)}):
            exec(compile(code, detector, "exec"), namespace)
        summary = namespace[f"{detector}_spatial_frequency"].sort_values("location")
        assert summary["n_anomalous_hours"].tolist() == expected_counts
        assert summary["n_valid_hours"].tolist() == [4, 3, 0]
        assert summary["coverage_pct"].tolist() == [100., 75., 0.]
        assert summary["n_eligible_hours"].eq(4).all()
        assert pd.isna(summary.iloc[2]["anomaly_share_pct"])
        assert summary.iloc[0]["anomaly_share_pct"] == (25. if detector == "stgan" else 50.)
        paths = namespace[f"{detector.upper()}_SPATIAL_PATHS"]
        assert all(p.is_file() for p in paths.values())
        assert paths["figure"].read_bytes().startswith(b"\x89PNG")
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        assert metadata["excluded_quality_timestamps"] == 2
        assert metadata["excluded_score_rows"] == 4
        assert metadata["n_valid_coordinates"] == 7
        assert metadata["color_range_pct"] == [0, 100]


def test_rejects_ambiguous_or_invalid_scores(tmp_path):
    pvgis, score_path = spatial_inputs(tmp_path)
    original = pd.read_csv(score_path)
    for invalid in ("duplicate", "off_grid", "nonfinite", "threshold"):
        scores = original.copy()
        if invalid == "duplicate":
            scores = pd.concat([scores, scores.iloc[:1]])
        elif invalid == "off_grid":
            scores.loc[0, "timestamp"] = "2019-01-01 00:00"
        elif invalid == "nonfinite":
            scores.loc[0, "anomaly_score"] = np.nan
        else:
            scores.loc[0, "threshold"] = 11.
        scores.to_csv(score_path, index=False)
        try:
            build_detector_spatial_frequency(score_path, pvgis, tmp_path / "out", detector="mtgflow")
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid scores: {invalid}")


def test_mtgflow_regional_posthoc_cell(tmp_path):
    from physiq_pv.reporting.stgan_regional import (
        aggregate_detector_region, load_mtgflow_regional_labels,
    )

    pvgis, score_path = spatial_inputs(tmp_path)
    labels = load_mtgflow_regional_labels(score_path, pvgis_quality_source=pvgis)
    hourly, daily = aggregate_detector_region(labels)
    assert hourly["n_valid_locations"].tolist() == [2, 2, 0, 0, 2, 1]
    assert hourly["n_anomalous_locations"].tolist() == [2, 1, 0, 0, 0, 0]
    assert hourly.iloc[0]["anomaly_share_pct"] == 100.  # Includes night.
    assert hourly.iloc[1]["anomaly_share_pct"] == 50.  # Includes score == threshold.
    assert hourly.iloc[2:4]["anomaly_share_pct"].isna().all()  # Quality gaps.
    assert hourly.iloc[-1]["location_coverage_pct"] == 50.
    assert daily["n_valid_hours"].sum() == 7
    cropped, _ = aggregate_detector_region(labels, start="2019-01-01 01:10", end="2019-01-01 04:10")
    assert len(cropped) == 4
    assert cropped.iloc[0]["anomaly_share_pct"] == 50.

    notebook = json.loads((ROOT / "notebooks/pvgis_sde_pipeline.ipynb").read_text(encoding="utf-8"))
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    code = "".join(next(c for c in notebook["cells"] if c["id"] == "regional-timeline")["source"])
    namespace = {"Path": Path, "os": os, "pd": pd, "PVGIS_DIR": tmp_path,
                 "TEST_ANOMALY_SCORES": score_path, "OUT_DIR": tmp_path / "out"}
    environment = {"PVGIS_2019_FILE": str(pvgis), "MTGFLOW_REGIONAL_START": "",
                   "MTGFLOW_REGIONAL_END": ""}
    with patch.dict(os.environ, environment), patch("IPython.display.display"):
        exec(compile(code, "regional-timeline", "exec"), namespace)
    paths = namespace["REGIONAL_PATHS"]
    assert all(path.is_file() for path in paths.values())
    assert paths["figure"].name == "mtgflow_regional_overview.png"
    assert paths["figure"].read_bytes().startswith(b"\x89PNG")
    exported = pd.read_csv(paths["hourly"])
    np.testing.assert_allclose(exported["anomaly_share_pct"], hourly["anomaly_share_pct"], equal_nan=True)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["detector"] == "mtgflow"
    assert metadata["n_valid_coordinates"] == 7
    assert metadata["excluded_quality_timestamps"] == 2
    assert "clean_top_k_percent" not in metadata
    assert "regional_labels" not in namespace


def test_hourly_overlay_preserves_each_detector_and_date_invariant_labels(tmp_path):
    import matplotlib.pyplot as plt

    pvgis, stgan_path = spatial_inputs(tmp_path)
    scores = pd.read_csv(stgan_path, dtype={"location": str})
    times = pd.to_datetime(scores["timestamp"])
    # MTGFlow has different coverage at night and a wholly missing daytime hour.
    remove = (scores["location"].eq("1") & times.dt.hour.eq(0)) | times.dt.hour.eq(4)
    mtgflow_path = tmp_path / "mtgflow.csv"
    scores.loc[~remove].to_csv(mtgflow_path, index=False)
    notebook = json.loads((ROOT / "notebooks/mtgflow_spatiotemporal_anomaly_heatmaps.ipynb").read_text(encoding="utf-8"))
    code = "".join(next(c for c in notebook["cells"] if c["id"] == "regional-comparison")["source"])
    namespace = {"Path": Path, "os": os, "pd": pd, "np": np, "plt": plt, "json": json,
                 "ROOT": ROOT, "PVGIS_2019_PATH": pvgis, "MTGFLOW_TEST_CSV": mtgflow_path,
                 "OUT_DIR": tmp_path / "out", "Image": Image, "display": lambda *args: None}
    environment = {"STGAN_SEED_DIR": str(stgan_path.parent),
                   "ANOMALY_COMPARISON_START": "", "ANOMALY_COMPARISON_END": ""}
    with patch.dict(os.environ, environment):
        exec(compile(code, "regional-comparison", "exec"), namespace)
    comparison = namespace["comparison"]
    assert len(comparison) == 6
    assert comparison.index.minute.tolist() == [10] * 6
    assert comparison.iloc[0]["stgan_anomaly_share_pct"] == 50.
    assert comparison.iloc[0]["mtgflow_anomaly_share_pct"] == 100.
    assert comparison.iloc[0]["stgan_n_valid_locations"] == 2
    assert comparison.iloc[0]["mtgflow_n_valid_locations"] == 1
    for detector in ("stgan", "mtgflow"):
        assert comparison.iloc[2:4][f"{detector}_anomaly_share_pct"].isna().all()
    assert comparison.iloc[4]["stgan_anomaly_share_pct"] == 0.
    assert pd.isna(comparison.iloc[4]["mtgflow_anomaly_share_pct"])
    paths = namespace["COMPARISON_PATHS"]
    assert all(path.is_file() for path in paths.values())
    assert paths["figure"].read_bytes().startswith(b"\x89PNG")
    exported = pd.read_csv(paths["hourly"])
    np.testing.assert_allclose(exported["stgan_anomaly_share_pct"],
                               comparison["stgan_anomaly_share_pct"], equal_nan=True)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["excluded_quality_timestamps"] == 2
    assert "each detector" in metadata["denominator"]
    # Zoom must not promote another STGAN point after the night-time maximum is removed.
    environment["ANOMALY_COMPARISON_START"] = "2019-01-01 01:10"
    environment["ANOMALY_COMPARISON_END"] = "2019-01-01 05:10"
    with patch.dict(os.environ, environment):
        exec(compile(code, "regional-comparison-zoom", "exec"), namespace)
    cropped = namespace["comparison"]
    assert len(cropped) == 5
    assert cropped["stgan_n_anomalous_locations"].sum() == 0
    assert cropped.iloc[0]["mtgflow_anomaly_share_pct"] == 50.


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        test_both_notebook_cells_and_frequency_semantics(Path(directory))
    with TemporaryDirectory() as directory:
        test_rejects_ambiguous_or_invalid_scores(Path(directory))
    with TemporaryDirectory() as directory:
        test_mtgflow_regional_posthoc_cell(Path(directory))
    with TemporaryDirectory() as directory:
        test_hourly_overlay_preserves_each_detector_and_date_invariant_labels(Path(directory))
    print("PASS: spatial maps, regional timeline and overlay, night, quality, coverage and invalid inputs")
