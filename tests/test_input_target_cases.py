from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import chdir
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physiq_pv.reporting.detector_threshold_sensitivity import (
    plot_mae_dispersion,
    sensitivity_sweep,
)
from physiq_pv.reporting.input_target_cases import (
    CASE_LABELS,
    classify_input_target_cases,
    load_clean_stgan_labels,
    plot_stgan_score_timeline,
    summarize_input_target_cases,
)


def _check_mae_bands(rule):
    coordinate = np.array([0, 0, 1, 1, 1, 1, 2, 2], dtype=float)
    errors = np.array([1, 3, 0, 2, 4, 100, 5, 8], dtype=float)
    frame = pd.DataFrame({"horizon_hours": 1, "decision_coordinate": coordinate,
                          "abs_error": errors, "squared_error": errors**2})
    result = sensitivity_sweep(frame, [-1, 1, 3], detector="test", rare_when=rule)
    for _, row in result.iterrows():
        rare = coordinate >= row.decision_threshold if rule == "coordinate_ge_threshold" else coordinate <= row.decision_threshold
        for group, mask in (("rare", rare), ("normal", ~rare)):
            values = errors[mask]
            prefix = f"abs_error_{group}"
            if not len(values):
                assert np.isnan(row[f"{prefix}_q1"])
                assert np.isnan(row[f"mae_{group}"])
                continue
            q1, med, q3 = np.quantile(values, [.25, .5, .75])
            assert row[f"{prefix}_q1"] == q1
            assert row[f"{prefix}_median"] == med
            assert row[f"{prefix}_q3"] == q3
            assert row[f"{prefix}_whisker_low"] == values[values >= q1 - 1.5*(q3-q1)].min()
            assert row[f"{prefix}_whisker_high"] == values[values <= q3 + 1.5*(q3-q1)].max()
            assert np.isclose(row[f"mae_{group}"], values.mean())
            assert np.isclose(row[f"rmse_{group}"], np.sqrt(np.mean(values**2)))
    fig, ax = plt.subplots()
    plot_mae_dispersion(ax, result)
    assert len(ax.collections) == 4
    assert len(ax.lines) == 2
    plt.close(fig)


def test_mae_bands_are_exact_absolute_error_boxes_including_empty_groups():
    for rule in ("coordinate_ge_threshold", "coordinate_le_threshold"):
        _check_mae_bands(rule)


def test_input_windows_include_issue_exclude_target_and_require_every_hour():
    # Offset hours match PVGIS :10 timestamps, not necessarily clock-hour starts.
    times = pd.date_range("2019-01-01 00:10", periods=12, freq="h")
    labels = pd.DataFrame({"location": "a", "timestamp": times,
                           "is_anomaly": [False, False, False, True, True,
                                          False, False, False, False, False, False, False]})
    predictions = pd.DataFrame({
        "location": "a", "timestamp": times[[3, 4, 5, 8]],
        "issue_timestamp": times[[2, 3, 4, 7]], "horizon_hours": 1,
        "abs_error": [1., 2., 3., 4.], "squared_error": [1., 4., 9., 16.],
        "production_bin": "daytime_0_20_pct",
    })
    classified = classify_input_target_cases(predictions, labels, seq_len=3)
    assert classified["case"].tolist() == [
        "normal_anomalous", "anomalous_anomalous", "anomalous_normal", "normal_normal",
    ]
    assert classified["input_anomalous_steps"].tolist() == [0, 1, 2, 0]
    majority = classify_input_target_cases(predictions, labels, seq_len=3, min_anomalous_steps=2)
    assert majority["case"].iloc[1] == "normal_anomalous"
    # An absent (e.g. quality-excluded) hour must not silently count as normal.
    missing = classify_input_target_cases(predictions, labels.drop(index=1), seq_len=3)
    assert pd.isna(missing["case"].iloc[0])
    assert missing["exclusion_reason"].iloc[0] == "incomplete_input"
    no_target = classify_input_target_cases(predictions, labels.drop(index=8), seq_len=3)
    assert no_target["exclusion_reason"].iloc[-1] == "unscored_target"
    other_site = labels.assign(location="b")
    isolated = classify_input_target_cases(predictions, other_site, seq_len=3)
    assert isolated["case"].isna().all()
    bad_issue = predictions.copy()
    bad_issue.loc[0, "issue_timestamp"] += pd.Timedelta(hours=1)
    try:
        classify_input_target_cases(bad_issue, labels, seq_len=3)
    except ValueError as exc:
        assert "Issue timestamps" in str(exc)
    else:
        raise AssertionError("Invalid issue timestamp was accepted")
    metrics = summarize_input_target_cases(classified, horizons=(1,))
    assert len(metrics) == 20  # five bins x four cases, including empty bins
    assert metrics["count"].sum() == 4
    assert metrics.loc[metrics["count"].eq(0), "rmse"].isna().all()
    assert set(metrics.loc[metrics["count"].gt(0), "case"]) == set(CASE_LABELS)


def test_clean_ties_quality_and_score_timeline(tmp_path):
    times = pd.date_range("2019-01-01", periods=6, freq="h")
    raw = pd.DataFrame({"location": "a", "timestamp": times,
                        "anomaly_score": [9., 8., 8., 3., 2., 1.],
                        "global_rank": [1, 3, 2, 4, 5, 6]})
    path = tmp_path / "scores.csv"
    raw.to_csv(path, index=False)
    clean = load_clean_stgan_labels(path, excluded_timestamps=times[:1], top_percent=20)
    assert clean.loc[clean["is_anomaly"], "timestamp"].tolist() == [times[2]]
    assert clean.attrs["score_cutoff"] == 8
    figure, series = plot_stgan_score_timeline(clean, tmp_path, start=times[1], end=times[3])
    assert figure.is_file()
    assert len(series) == 3
    assert series["is_anomaly"].sum() == 1


def test_four_case_notebook_executes_on_hourly_multisite_predictions(tmp_path):
    times = pd.date_range("2019-01-01 00:10", periods=300, freq="h")
    stgan_dir = tmp_path / "stgan"
    stgan_dir.mkdir()
    rows, forecasts = [], []
    flagged = {(0, 40), (0, 41), (0, 46), (1, 140), (1, 141), (1, 146)}
    for location in range(2):
        for i, time in enumerate(times):
            rows.append({"location": str(location), "timestamp": time,
                         "anomaly_score": 10.0 if (location, i) in flagged else i/1000})
            for horizon in (1, 6):
                if i < 24 + horizon - 1:
                    continue
                real = float(10 + 20*(i % 5))
                forecasts.append({"location": str(location), "timestamp": time,
                                  "issue_timestamp": times[i-horizon], "horizon_hours": horizon,
                                  "y_true": real, "y_pred_mean": real + horizon + i % 3,
                                  "solar_irradiance_poa_target": 100.0})
    pd.DataFrame(rows).to_csv(stgan_dir / "anomaly_scores.csv", index=False)
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(forecasts).to_csv(predictions, index=False)
    reference = tmp_path / "reference.csv"
    pd.DataFrame({"reference_peak_w": [100.]}).to_csv(reference, index=False)
    pvgis = tmp_path / "pvgis.nc"
    xr.Dataset({variable: (("location", "time"), np.ones((2, len(times))))
                for variable in ("direct_irradiance_tilted", "diffuse_irradiance_tilted",
                                 "sun_height", "pv_power_output")},
               coords={"location": [0, 1], "time": times}).to_netcdf(pvgis)
    out = tmp_path / "cases"
    environment = {
        "STGAN_SEED_DIR": stgan_dir, "SDE_MULTIHORIZON_PREDICTIONS": predictions,
        "SDE_REFERENCE_PEAK_CSV": reference, "PVGIS_QUALITY_SOURCE": pvgis,
        "STGAN_INPUT_TARGET_OUT_DIR": out, "SDE_INPUT_SEQ_LEN": "24",
        "STGAN_INPUT_MIN_ANOMALOUS_STEPS": "1",
    }
    path = ROOT / "notebooks/stgan_input_target_cases_sdenet.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__notebook_smoke__"}
    with patch.dict(os.environ, {key: str(value) for key, value in environment.items()}), chdir(ROOT):
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), f"{path}:{i}", "exec"), namespace)
    metrics = pd.read_csv(out / "input_target_bin_metrics.csv")
    assert metrics["count"].sum() == len(forecasts)
    for _, group in metrics.groupby("horizon_hours"):
        assert set(group.loc[group["count"].gt(0), "case"]) == set(CASE_LABELS)
    assert len(list((out / "figures").glob("*.png"))) == 20
    metadata = json.loads((out / "analysis_metadata.json").read_text())
    assert metadata["n_excluded_rows"] == 0
    assert metadata["input_min_anomalous_steps"] == 1
    classifications = pd.read_csv(out / "input_target_classification.csv")
    # Six-hour forecast at hour 46 must see the flagged hour 40 at issue time.
    sample = classifications.loc[
        classifications["location"].eq(0)
        & classifications["timestamp"].eq(str(times[46]))
        & classifications["horizon_hours"].eq(6)
    ]
    assert sample["case"].tolist() == ["anomalous_anomalous"]


if __name__ == "__main__":
    test_mae_bands_are_exact_absolute_error_boxes_including_empty_groups()
    test_input_windows_include_issue_exclude_target_and_require_every_hour()
    with tempfile.TemporaryDirectory() as directory:
        test_clean_ties_quality_and_score_timeline(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_four_case_notebook_executes_on_hourly_multisite_predictions(Path(directory))
    print("PASS: MAE dispersion, STGAN timeline and input/target case tests")
