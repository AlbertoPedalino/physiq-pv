"""Regional denominators, missing observations and the executable notebook cell."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physiq_pv.reporting.stgan_regional import aggregate_stgan_region


def labels_fixture():
    origin = pd.Timestamp("2019-01-01 00:10")
    rows = [("2", 0, True), ("2", 1, False), ("2", 3, True), ("2", 48, False),
            ("10", 0, False), ("10", 1, False), ("10", 3, True),
            ("a", 0, False), ("a", 48, True)]
    labels = pd.DataFrame([
        {"location": location, "timestamp": origin + pd.Timedelta(hours=hour), "is_anomaly": flag}
        for location, hour, flag in rows
    ])
    labels.attrs.update(top_percent=1., score_cutoff=.65, excluded_timestamps=24)
    return labels


def test_regional_denominators_and_missing_hours():
    labels = labels_fixture()
    hourly, daily = aggregate_stgan_region(labels)
    assert len(hourly) == 49
    assert np.isclose(hourly.iloc[0]["anomaly_share_pct"], 100/3)
    assert hourly.iloc[1]["anomaly_share_pct"] == 0
    assert hourly.iloc[1]["n_valid_locations"] == 2
    assert hourly.iloc[3]["anomaly_share_pct"] == 100
    assert hourly.iloc[48]["anomaly_share_pct"] == 50
    assert hourly.iloc[2]["n_valid_locations"] == 0
    assert pd.isna(hourly.iloc[2]["anomaly_share_pct"])
    assert hourly["n_anomalous_locations"].sum() == labels["is_anomaly"].sum()
    assert daily["n_valid_hours"].sum() == len(labels)
    assert daily["location"].drop_duplicates().tolist() == ["2", "10", "a"]
    assert daily.loc[daily["day"].eq("2019-01-02"), "anomalous_hours_pct"].isna().all()
    first_site_day = daily.loc[daily["location"].eq("2")].iloc[0]
    assert first_site_day["n_valid_hours"] == 3
    assert np.isclose(first_site_day["anomalous_hours_pct"], 200/3)
    short_hourly, short_daily = aggregate_stgan_region(
        labels, start="2019-01-03 00:00", end="2019-01-03 02:00",
    )
    assert short_hourly["timestamp"].dt.minute.eq(10).all()
    assert len(short_hourly) == 2
    assert pd.isna(short_hourly.iloc[1]["anomaly_share_pct"])
    assert short_daily["location"].tolist() == ["2", "10", "a"]
    assert short_daily.loc[short_daily["location"].eq("10"), "n_valid_hours"].iloc[0] == 0
    assert np.isnan(short_daily.loc[short_daily["location"].eq("10"), "anomalous_hours_pct"].iloc[0])
    invalid_frames = [pd.concat([labels, labels.iloc[:1]]), labels.assign(is_anomaly="False")]
    irregular = labels.copy()
    irregular.loc[0, "timestamp"] += pd.Timedelta(minutes=5)
    invalid_frames.append(irregular)
    for invalid in invalid_frames:
        try:
            aggregate_stgan_region(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid regional input accepted")


def test_regional_notebook_cell(tmp_path):
    notebook = json.loads((ROOT / "notebooks/stgan_pointwise_posthoc_sdenet.ipynb").read_text(encoding="utf-8"))
    ids = [cell.get("id") for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    code = "".join(next(cell for cell in notebook["cells"] if cell.get("id") == "regional-timeline")["source"])
    namespace = {"os": os, "pd": pd, "Image": Image, "Markdown": Markdown,
                 "display": lambda *args: None, "score_labels": labels_fixture(), "EVALUATION_DIR": tmp_path}
    with patch.dict(os.environ, {"STGAN_REGIONAL_START": "", "STGAN_REGIONAL_END": ""}):
        exec(compile(code, "regional-timeline", "exec"), namespace)
    paths = namespace["REGIONAL_PATHS"]
    assert all(path.is_file() for path in paths.values())
    assert paths["figure"].read_bytes().startswith(b"\x89PNG")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["n_locations"] == 3
    assert metadata["n_valid_coordinates"] == 9
    assert metadata["heatmap_color_range_pct"] == [0, 100]
    assert metadata["clean_top_k_percent"] == 1
    assert "score_labels" not in namespace


if __name__ == "__main__":
    test_regional_denominators_and_missing_hours()
    with tempfile.TemporaryDirectory() as directory:
        test_regional_notebook_cell(Path(directory))
    print("PASS: STGAN regional aggregation, missing data and notebook cell")
