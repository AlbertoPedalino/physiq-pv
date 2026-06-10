"""Smoke tests for PVGIS residual-bias and fixed daytime-bin diagnostics."""

from pathlib import Path
import sys
import tempfile

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from physiq_pv.data.pvgis_stgnn_dataset import (
    build_wandb_metrics,
    compute_metrics,
    write_outputs,
)
from physiq_pv.experiments.pvgis_stgnn_runner import (
    build_daytime_metrics,
    build_interval_metrics,
    build_residual_bias_metrics,
    flatten_daytime_metrics,
    flatten_interval_metrics,
    flatten_residual_bias_metrics,
)


def _predictions() -> pd.DataFrame:
    daytime_y = np.array(
        [0.0, 19.999, 20.0, 39.999, 40.0, 59.999, 60.0, 79.999, 80.0, 99.999, 100.0]
    )
    nighttime_y = np.array([0.0, 0.0])
    y_true = np.concatenate([daytime_y, nighttime_y])
    residual = np.array(
        [2.0, -1.0, 3.0, -2.0, 4.0, -3.0, -5.0, -6.0, -7.0, -8.0, -10.0, 1.0, 2.0]
    )
    y_pred = y_true + residual
    lower_pi = y_pred - 4.0
    upper_pi = y_pred + 4.0
    lower_pi[0] = 0.5
    lower_pi[-2:] = np.array([0.5, 1.0])
    lower_gaussian = y_pred - 12.0
    upper_gaussian = y_pred + 12.0
    anomaly_label = [
        "unusually_low_solar_potential",
        "",
        "extreme_temperature_condition",
        "",
        "",
        "",
        "",
        "extreme_wind_condition",
        "",
        "",
        "unusually_high_solar_potential,extreme_temperature_condition",
        "",
        "unusually_low_solar_potential",
    ]
    anomaly_group = [
        "rare_or_extreme" if value else "normal" for value in anomaly_label
    ]
    return pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_pred_mean": y_pred,
            "y_pred_std": np.full(len(y_true), 2.0),
            "y_pred_std_raw": np.full(len(y_true), 2.0),
            "solar_irradiance_poa_target": np.concatenate(
                [np.full(len(daytime_y), 100.0), np.array([0.0, 10.0])]
            ),
            "anomaly_group": anomaly_group,
            "anomaly_label": anomaly_label,
            "lower_pi": lower_pi,
            "upper_pi": upper_pi,
            "lower_gaussian": lower_gaussian,
            "upper_gaussian": upper_gaussian,
            "y_pred_lower": lower_gaussian,
            "y_pred_upper": upper_gaussian,
            "error": residual,
            "abs_error": np.abs(residual),
            "squared_error": residual ** 2,
        }
    )


def _rows_by_stratum(rows):
    return {row["stratum"]: row for row in rows}


def test_fixed_daytime_bins_and_boundaries() -> None:
    predictions = _predictions()
    target_range = float(predictions["y_true"].max() - predictions["y_true"].min())
    rows = build_residual_bias_metrics(
        predictions, target_range, gamma=0.95, eta=10.0
    )
    by = _rows_by_stratum(rows)

    expected_counts = {
        "daytime_0_20": 2,
        "daytime_20_40": 2,
        "daytime_40_60": 2,
        "daytime_60_80": 2,
        "daytime_80_100": 2,
        "daytime_gt_100": 1,
    }
    assert {name: by[name]["count"] for name in expected_counts} == expected_counts
    assert sum(expected_counts.values()) == by["daytime"]["count"]

    assert by["daytime_20_40"]["mean_residual"] == 0.5
    assert by["daytime_gt_100"]["mean_residual"] == -10.0
    assert by["daytime_gt_100"]["fraction_underprediction"] == 1.0
    assert by["daytime_gt_100"]["fraction_above_interval"] == 1.0
    assert by["daytime_0_20"]["fraction_below_interval"] == 0.5


def test_signed_residual_and_anomaly_label_metrics() -> None:
    predictions = _predictions()
    predictions.loc[2, "upper_pi"] = predictions.loc[2, "y_true"]
    predictions.loc[3, "lower_pi"] = predictions.loc[3, "y_true"]
    target_range = 100.0
    rows = build_residual_bias_metrics(
        predictions, target_range, gamma=0.95, eta=10.0
    )
    by = _rows_by_stratum(rows)
    expected_residual = (
        predictions["y_pred_mean"].to_numpy() - predictions["y_true"].to_numpy()
    )
    assert np.isclose(by["global"]["mean_residual"], expected_residual.mean())
    expected_above = np.mean(
        predictions["y_true"].to_numpy() > predictions["upper_pi"].to_numpy()
    )
    expected_below = np.mean(
        predictions["y_true"].to_numpy() < predictions["lower_pi"].to_numpy()
    )
    assert by["global"]["fraction_above_interval"] == expected_above
    assert by["global"]["fraction_below_interval"] == expected_below

    low = by["label:unusually_low_solar_potential"]
    high = by["label:unusually_high_solar_potential"]
    temperature = by["label:extreme_temperature_condition"]
    assert low["count"] == 2
    assert low["mean_residual"] == 2.0
    assert low["fraction_overprediction"] == 1.0
    assert high["count"] == 1
    assert high["mean_residual"] == -10.0
    assert high["fraction_underprediction"] == 1.0
    assert temperature["count"] == 2


def test_diagnostics_do_not_change_legacy_metrics_or_wandb_names() -> None:
    predictions = _predictions()
    global_before, by_before = compute_metrics(predictions.copy())
    rows = build_residual_bias_metrics(
        predictions, target_range=100.0, gamma=0.95, eta=10.0
    )
    global_after, by_after = compute_metrics(predictions.copy())
    pd.testing.assert_frame_equal(global_before, global_after)
    pd.testing.assert_frame_equal(by_before, by_after)

    flattened = flatten_residual_bias_metrics(rows)
    expected = {
        "mean_residual/daytime",
        "mean_residual/nighttime",
        "mean_residual/rare_extreme_daytime",
        "mean_residual/unusually_low_solar_potential",
        "mean_residual/unusually_high_solar_potential",
        "fraction_underprediction/daytime_gt_100",
        "fraction_above_interval/daytime_gt_100",
        "mae/daytime_gt_100",
        "picp_pi/daytime_gt_100",
        "fraction_below_interval/nighttime",
        "fraction_above_interval/daytime",
    }
    assert expected <= set(flattened)
    assert "mae/global" not in flattened
    assert "picp_pi/global" not in flattened

    target_range = 100.0
    daytime = build_daytime_metrics(
        predictions, target_range, gamma=0.95, eta=10.0
    )
    intervals = build_interval_metrics(
        predictions, target_range, gamma=0.95, eta=10.0
    )
    legacy = build_wandb_metrics(
        global_before, by_before, mc_dropout=True
    )
    legacy.update(flatten_interval_metrics(intervals))
    legacy.update(flatten_daytime_metrics(daytime))
    assert not (set(legacy) & set(flattened))


def test_report_and_residual_csv_are_written() -> None:
    predictions = _predictions()
    target_range = 100.0
    global_df, by_df = compute_metrics(predictions)
    interval_metrics = build_interval_metrics(
        predictions, target_range, gamma=0.95, eta=10.0
    )
    daytime_metrics = build_daytime_metrics(
        predictions, target_range, gamma=0.95, eta=10.0
    )
    residual_metrics = build_residual_bias_metrics(
        predictions, target_range, gamma=0.95, eta=10.0
    )
    meta = {
        "mode": "pvgis_stgnn",
        "model_type": "stgnn_enhanced_dropout",
        "feature_set": "full",
        "target_variable": "pv_power_output",
        "pv_target_clip_max": 1.5,
        "features": ["solar_irradiance_poa"],
        "n_features": 1,
        "seq_len": 24,
        "horizon": 1,
        "train_years": "2016,2017,2018",
        "test_year": 2019,
        "n_nodes": 1,
        "n_predictions": len(predictions),
        "epochs": 0,
        "batch_size": 1,
        "lr": 0.001,
        "anomaly_scores": "dummy.csv",
        "device": "cpu",
        "generated_utc": "2026-06-10T00:00:00+00:00",
        "wandb_enabled": False,
        "mc_dropout": True,
        "mc_samples": 20,
        "calibration": None,
        "interval_metrics": interval_metrics,
        "daytime_metrics": daytime_metrics,
        "residual_bias_metrics": residual_metrics,
        "daytime_threshold_wm2": 10.0,
        "clc_gamma": 0.95,
        "clc_eta": 10.0,
        "target_range": target_range,
    }
    with tempfile.TemporaryDirectory() as tmp:
        paths = write_outputs(
            predictions,
            global_df,
            by_df,
            tmp,
            meta,
            skip_predictions=True,
        )
        report = paths["report"].read_text(encoding="utf-8")
        csv = pd.read_csv(paths["residual_bias_metrics"])

    assert "## Residual bias diagnostics by stratum" in report
    assert "## Daytime production-bin diagnostics" in report
    assert "## Automatic interpretation of residual asymmetry" in report
    assert set(csv["kind"]) == {
        "main_stratum",
        "anomaly_label",
        "daytime_production_bin",
    }
    assert "daytime_gt_100" in set(csv["stratum"])


if __name__ == "__main__":
    test_fixed_daytime_bins_and_boundaries()
    test_signed_residual_and_anomaly_label_metrics()
    test_diagnostics_do_not_change_legacy_metrics_or_wandb_names()
    test_report_and_residual_csv_are_written()
    print("PASS: PVGIS residual bias diagnostics")
