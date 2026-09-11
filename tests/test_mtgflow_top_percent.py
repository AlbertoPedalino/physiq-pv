"""Global normalized ranking and both exploratory notebook cells."""

import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image

from test_detector_spatial_frequency import ROOT, spatial_inputs
from physiq_pv.reporting import mtgflow_spatiotemporal as spatial
from physiq_pv.reporting.mtgflow_top_percent import rank_mtgflow_top_percent


def test_global_budget_uses_training_scale_not_raw_score_or_per_site_quota():
    scores = pd.DataFrame({
        "location": ["0"] * 100 + ["1"] * 100,
        "anomaly_score": [2000.] + [100.] * 99 + [20., 19.] + [0.] * 98,
        "threshold": [100.] * 100 + [0.] * 100,
    })
    table = pd.DataFrame({"location": ["0", "1"], "saved_threshold": [100., 0.], "iqr": [1000., 1.]})
    result = rank_mtgflow_top_percent(scores, table)
    assert result["is_anomaly"].sum() == 2
    assert result.loc[result["is_anomaly"], "location"].tolist() == ["1", "1"]
    assert result["anomaly_score"].equals(scores["anomaly_score"])
    assert result.attrs["ranking_cutoff_iqr"] == 19.
    assert result.attrs["n_quality_eligible_before_daytime"] == 200
    # Ties below the saved threshold still receive an exact global budget.
    tied = pd.DataFrame({"location": ["1"] * 4, "anomaly_score": [-1.] * 3 + [-2.], "threshold": [0.] * 4})
    selected = rank_mtgflow_top_percent(tied, table, top_percent=50.)
    assert selected["is_anomaly"].tolist() == [True, True, False, False]
    assert selected.attrs["ranking_cutoff_iqr"] == -1.
    bad_table = table.assign(iqr=0.)
    try:
        rank_mtgflow_top_percent(scores, bad_table)
    except ValueError:
        pass
    else:
        raise AssertionError("Non-positive training IQR accepted")


def test_top1_notebook_maps_and_overlay(tmp_path):
    pvgis, stgan_scores = spatial_inputs(tmp_path)
    mtg_dir = tmp_path / "mtgflow"
    mtg_dir.mkdir()
    mtg_path = mtg_dir / "anomaly_scores.csv"
    mtg_scores = pd.read_csv(stgan_scores, dtype={"location": str})
    mtg_scores.loc[7, "anomaly_score"] = 4.  # Daytime at location 1 wins after normalization.
    mtg_scores.to_csv(mtg_path, index=False)
    statistics = tmp_path / "iqr.csv"
    pd.DataFrame({"location": ["0", "1"], "q1": [0., 0.], "q3": [100., 1.]}).to_csv(statistics, index=False)
    notebook = json.loads((ROOT / "notebooks/mtgflow_spatiotemporal_anomaly_heatmaps.ipynb").read_text(encoding="utf-8"))
    code = {c["id"]: "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"}
    namespace = {"importlib": importlib, "os": os, "Path": Path, "ROOT": ROOT,
                 "np": np, "pd": pd, "plt": plt, "json": json, "spatial": spatial,
                 "PVGIS_2019_PATH": pvgis, "MTGFLOW_TEST_CSV": mtg_path,
                 "MTGFLOW_SEED_DIR": mtg_dir, "MTGFLOW_TRAIN_CSV": mtg_dir / "unused.csv",
                 "TRAINING_STATS_CSV": statistics, "CSV_CHUNKSIZE": 4,
                 "DAYTIME_THRESHOLD_WM2": 10., "OUT_DIR": tmp_path / "out",
                 "Image": Image, "display": lambda *args: None}
    environment = {"STGAN_SEED_DIR": str(stgan_scores.parent),
                   "TOP1_COMPARISON_START": "", "TOP1_COMPARISON_END": ""}
    with patch.dict(os.environ, environment):
        for cell_id in ("top1-maps", "top1-overlay"):
            exec(compile(code[cell_id], cell_id, "exec"), namespace)
    audit = namespace["TOP1_AUDIT"].set_index("detector")
    assert audit["n_valid_before_daytime"].tolist() == [7, 7]
    assert audit["n_anomalies_before_daytime"].tolist() == [1, 1]
    assert audit["n_anomalies_daytime"].tolist() == [0, 1]
    assert audit["n_valid_daytime"].tolist() == [5, 5]
    assert namespace["top1_cutoff"] == 3.
    assert namespace["TOP1_TRAINING_TABLE"]["top1_score_cutoff"].tolist() == [310., 4.]
    overlay = namespace["top1_overlay"]
    assert overlay.iloc[1]["mtgflow_anomaly_share_pct"] == 50.
    assert overlay["stgan_n_anomalous_locations"].sum() == 0
    assert overlay.iloc[[0, 2, 3]]["mtgflow_anomaly_share_pct"].isna().all()
    for paths, summary in namespace["TOP1_MAP_PAIR"].values():
        assert all(path.is_file() for path in paths.values())
        assert summary["n_valid_hours"].sum() == 5
    assert namespace["TOP1_OVERLAY_PATH"].is_file()
    environment.update(TOP1_COMPARISON_START="2019-01-01 04:10", TOP1_COMPARISON_END="2019-01-01 05:10")
    with patch.dict(os.environ, environment):
        exec(compile(code["top1-overlay"], "top1-overlay-zoom", "exec"), namespace)
    assert namespace["top1_overlay"]["mtgflow_n_anomalous_locations"].sum() == 0


if __name__ == "__main__":
    test_global_budget_uses_training_scale_not_raw_score_or_per_site_quota()
    with TemporaryDirectory() as directory:
        test_top1_notebook_maps_and_overlay(Path(directory))
    print("PASS: global MTGFlow top-1% IQR ranking, exact ties, daytime maps and overlay")
