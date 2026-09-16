import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from physiq_pv.reporting.stgan_meteo_errors import (
    QUALITY_POLICY, build_stgan_meteo_errors,
)


def _inputs(root):
    times = pd.date_range("2019-05-01 10:00", periods=7, freq="h")
    records = []
    for h in (1, 6):
        for location, errors in (("01", [3, 6, 2, 1e6, 1e6, 1e6, 1e6]),
                                 ("02", [4, 8, 2, 1e6, 1e6, 1e6, 1e6])):
            for i, (time, error) in enumerate(zip(times, errors)):
                records.append(dict(location=location, timestamp=time, horizon_hours=h,
                                    y_true=100, y_pred_mean=100 + error,
                                    solar_irradiance_poa_target=100 if i < 6 else 0,
                                    detector_is_anomaly=i > 0, data_quality_issue=i == 5))
    pd.DataFrame(records).to_csv(root / "predictions.csv", index=False)
    (root / "evaluation_source.json").write_text(json.dumps({
        "quality_filter_policy": QUALITY_POLICY, "pvgis_quality_source": "source.nc",
        "clean_top_k_percent": 1, "excluded_data_quality_rows": 12,
    }))
    pd.DataFrame({"timestamp": times[3:5]}).to_csv(root / "pvgis_data_quality_issues.csv", index=False)
    flags = []
    for location in ("01", "02"):
        flags += [(location, times[1], "temperature_2m", "extreme_temperature_condition"),
                  (location, times[2], "pv_power_output", "unusually_high_solar_potential"),
                  (location, times[3], "solar_irradiance_poa", "unusually_high_solar_potential")]
    flags += [("01", times[1], "wind_speed_10m", "extreme_wind_condition"),
              ("01", times[1], "solar_irradiance_poa", "unusually_high_solar_potential"),
              ("02", times[1], "solar_irradiance_poa", "unusually_low_solar_potential")]
    # Duplicated labels must never duplicate forecast rows.
    flags += [flags[-1]]
    scores = root / "climatology.csv"
    pd.DataFrame(flags, columns=["location", "timestamp", "variable", "label"]).to_csv(scores, index=False)
    return scores


def test_meteo_errors_quality_exact_join_overlap_and_rmse(tmp_path):
    scores = _inputs(tmp_path)
    paths = build_stgan_meteo_errors(tmp_path, scores)
    metrics = pd.read_csv(paths["metrics"])
    rows = metrics.loc[metrics["horizon_hours"].eq(1) & metrics["bin"].eq("all_daytime")].set_index("category")
    assert rows["count"].to_dict() == dict(normal=2, temperature=2, wind=1, irradiance_high=1, irradiance_low=1)
    assert rows.loc["normal", "mae"] == 3.5
    assert rows.loc["temperature", "mae"] == 7
    assert np.isclose(rows.loc["temperature", "rmse"], np.sqrt(50))
    assert rows.loc["temperature", "rmse_site_median"] == 7
    assert rows.loc["temperature", "rmse_site_q1"] == 6.5
    assert rows.loc["temperature", "rmse_site_q3"] == 7.5
    counts = pd.read_csv(paths["counts"])
    assert counts["untyped_anomaly"].tolist() == [2, 2]
    assert counts["multiple_conditions"].tolist() == [2, 2]
    assert counts["daytime_rows"].tolist() == [6, 6]
    audit = json.loads(paths["audit"].read_text())
    assert audit["excluded_quality_rows_on_reload"] == 12
    assert audit["excluded_nighttime_rows"] == 4
    assert audit["reference_peak_w"] == 100
    empty = metrics.loc[metrics["bin"].eq("0_20_pct")]
    assert empty["count"].eq(0).all()
    assert empty["rmse"].isna().all()
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


def test_meteo_errors_rejects_unfiltered_inputs(tmp_path):
    scores = _inputs(tmp_path)
    (tmp_path / "evaluation_source.json").write_text('{}')
    with unittest.TestCase().assertRaisesRegex(ValueError, "quality-filtered"):
        build_stgan_meteo_errors(tmp_path, scores)


def test_meteo_errors_requires_quality_audit(tmp_path):
    scores = _inputs(tmp_path)
    (tmp_path / "pvgis_data_quality_issues.csv").unlink()
    with unittest.TestCase().assertRaises(FileNotFoundError):
        build_stgan_meteo_errors(tmp_path, scores)


def test_stgan_meteo_notebook_cells_compile():
    root = Path(__file__).resolve().parents[1]
    for name in ("stgan_pointwise_posthoc_sdenet", "stgan_cnn_pvgis_workflow"):
        notebook = json.loads((root / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), name, "exec")


if __name__ == "__main__":
    for test in (test_meteo_errors_quality_exact_join_overlap_and_rmse,
                 test_meteo_errors_rejects_unfiltered_inputs,
                 test_meteo_errors_requires_quality_audit):
        with tempfile.TemporaryDirectory() as temporary:
            test(Path(temporary))
    test_stgan_meteo_notebook_cells_compile()
    print("PASS: STGAN meteorological error plots")
