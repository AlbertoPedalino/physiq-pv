"""Smoke tests for PVGIS daytime/nighttime evaluation diagnostics."""

from pathlib import Path
import sys
import tempfile

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from physiq_pv.data.pvgis_stgnn_dataset import (
    DAYTIME_IRRADIANCE_THRESHOLD_WM2,
    PVGISWindowDataset,
    build_wandb_metrics,
    compute_metrics,
    make_model,
    predict,
    predict_mc,
    write_outputs,
)
from physiq_pv.experiments.pvgis_stgnn_runner import (
    build_daytime_metrics,
    build_interval_metrics,
    flatten_daytime_metrics,
)
from physiq_pv.model.lstm_baseline import LSTMBaseline


class _DummyDropoutModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, _edge_index, _edge_weight, _ghi_cs):
        batch_size, n_nodes = x.shape[:2]
        base = torch.full((batch_size, n_nodes), 0.2, device=x.device)
        pred = self.dropout(base)
        return pred, pred


def _window_dataset() -> PVGISWindowDataset:
    times = pd.date_range("2019-01-01", periods=5, freq="h")
    feats = np.zeros((5, 2, 1), dtype=np.float32)
    pv_raw = np.array(
        [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]],
        dtype=np.float32,
    )
    solar_raw = np.array(
        [[100.0, 101.0], [200.0, 201.0], [0.0, 20.0], [10.0, 11.0], [5.0, 100.0]],
        dtype=np.float32,
    )
    return PVGISWindowDataset(
        feats_by_year={2019: feats},
        pvnorm_by_year={2019: pv_raw / 10.0},
        pvraw_by_year={2019: pv_raw},
        solarraw_by_year={2019: solar_raw},
        times_by_year={2019: times},
        seq_len=2,
        horizon=1,
        pv_scale=np.array([10.0, 10.0]),
        loc_ids=np.array(["a", "b"]),
    )


def _diagnostic_predictions() -> pd.DataFrame:
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 0.0, 0.0, 0.0, 5.0])
    y_pred = np.array([12.0, 18.0, 35.0, 35.0, 1.0, 2.0, 3.0, 6.0])
    lower_gaussian = np.array(
        [8.0, 14.0, 26.0, 31.0, -2.0, -2.0, -2.0, 2.0]
    )
    upper_gaussian = np.array(
        [16.0, 22.0, 34.0, 39.0, 4.0, 4.0, 4.0, 10.0]
    )
    error = y_pred - y_true
    return pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_pred_mean": y_pred,
            "y_pred_std": np.full(8, 2.0),
            "y_pred_std_raw": np.full(8, 2.0),
            "solar_irradiance_poa_target": [20.0, 30.0, 40.0, 50.0, 0.0, 5.0, 10.0, 10.0],
            "anomaly_group": [
                "normal",
                "normal",
                "rare_or_extreme",
                "rare_or_extreme",
                "normal",
                "normal",
                "rare_or_extreme",
                "rare_or_extreme",
            ],
            "anomaly_label": [""] * 8,
            "lower_pi": [8.0, 15.0, 25.0, 30.0, 0.5, 1.0, 2.0, 4.0],
            "upper_pi": [16.0, 22.0, 36.0, 42.0, 3.0, 4.0, 5.0, 8.0],
            "lower_gaussian": lower_gaussian,
            "upper_gaussian": upper_gaussian,
            "y_pred_lower": lower_gaussian,
            "y_pred_upper": upper_gaussian,
            "error": error,
            "abs_error": np.abs(error),
            "squared_error": error ** 2,
        }
    )


def test_target_irradiance_alignment() -> None:
    dataset = _window_dataset()
    expected_y = np.array(
        [[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]], dtype=np.float32
    )
    expected_solar = np.array(
        [[0.0, 20.0], [10.0, 11.0], [5.0, 100.0]], dtype=np.float32
    )
    np.testing.assert_array_equal(dataset.y_true_all, expected_y)
    np.testing.assert_array_equal(
        dataset.solar_irradiance_poa_target_all, expected_solar
    )

    model = _DummyDropoutModel()
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_weight = torch.empty(0, dtype=torch.float32)
    deterministic = predict(
        model, dataset, edge_index, edge_weight, device="cpu", batch_size=2
    )
    mc = predict_mc(
        model,
        dataset,
        edge_index,
        edge_weight,
        device="cpu",
        batch_size=2,
        mc_samples=4,
    )
    np.testing.assert_array_equal(
        deterministic["solar_irradiance_poa_target"].to_numpy(),
        expected_solar.reshape(-1),
    )
    np.testing.assert_array_equal(
        mc["solar_irradiance_poa_target"].to_numpy(),
        expected_solar.reshape(-1),
    )
    np.testing.assert_array_equal(
        deterministic["y_true"].to_numpy(), expected_y.reshape(-1)
    )


def test_supported_model_prediction_contracts() -> None:
    dataset = _window_dataset()
    edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    edge_weight = torch.ones(2, dtype=torch.float32)
    models = {
        "stgnn": make_model(
            n_nodes=2,
            seq_len=2,
            n_features=1,
            dropout=0.2,
            enhanced_dropout=False,
        ),
        "stgnn_enhanced_dropout": make_model(
            n_nodes=2,
            seq_len=2,
            n_features=1,
            dropout=0.2,
            enhanced_dropout=True,
        ),
        "lstm": LSTMBaseline(
            n_features=1,
            hidden_size=4,
            num_layers=1,
            dropout=0.2,
        ),
    }
    for model_type, model in models.items():
        predictions = predict_mc(
            model,
            dataset,
            edge_index,
            edge_weight,
            device="cpu",
            batch_size=3,
            mc_samples=2,
        )
        assert "solar_irradiance_poa_target" in predictions, model_type
        assert len(predictions) == len(dataset) * dataset.n_nodes, model_type


def test_daytime_metrics_and_legacy_outputs() -> None:
    predictions = _diagnostic_predictions()
    target_range = float(predictions["y_true"].max() - predictions["y_true"].min())

    global_before, by_before = compute_metrics(predictions.copy())
    wandb_before = build_wandb_metrics(
        global_before, by_before, mc_dropout=True
    )
    interval_before = build_interval_metrics(
        predictions.copy(), target_range, gamma=0.95, eta=10.0
    )

    metrics = build_daytime_metrics(
        predictions,
        target_range,
        gamma=0.95,
        eta=10.0,
        threshold_wm2=DAYTIME_IRRADIANCE_THRESHOLD_WM2,
    )
    flattened = flatten_daytime_metrics(metrics)

    assert metrics["daytime"]["count"] == 4
    assert metrics["nighttime"]["count"] == 4
    assert (
        metrics["daytime"]["count"] + metrics["nighttime"]["count"]
        == metrics["global"]["count"]
    )
    assert metrics["normal_daytime"]["count"] == 2
    assert metrics["rare_extreme_daytime"]["count"] == 2
    assert metrics["normal_nighttime"]["count"] == 2
    assert metrics["rare_extreme_nighttime"]["count"] == 2

    expected_wandb_keys = {
        "picp_pi/daytime",
        "picp_pi/nighttime",
        "picp_pi/normal_daytime",
        "picp_pi/rare_extreme_daytime",
        "mae/daytime",
        "mae/nighttime",
        "fraction_lower_pi_leq_zero/daytime",
        "fraction_lower_gaussian_leq_zero/nighttime",
    }
    assert expected_wandb_keys <= set(flattened)
    assert "mae/global" not in flattened
    assert "mae/normal" not in flattened
    assert "picp_pi/global" not in flattened
    assert "picp_pi/rare_extreme" not in flattened

    global_after, by_after = compute_metrics(predictions.copy())
    interval_after = build_interval_metrics(
        predictions.copy(), target_range, gamma=0.95, eta=10.0
    )
    pd.testing.assert_frame_equal(global_before, global_after)
    pd.testing.assert_frame_equal(by_before, by_after)
    assert interval_before == interval_after
    wandb_after = build_wandb_metrics(
        global_after, by_after, mc_dropout=True
    )
    assert wandb_before == wandb_after


def test_report_and_daytime_csv() -> None:
    predictions = _diagnostic_predictions()
    target_range = float(predictions["y_true"].max() - predictions["y_true"].min())
    global_df, by_df = compute_metrics(predictions)
    interval_metrics = build_interval_metrics(
        predictions, target_range, gamma=0.95, eta=10.0
    )
    daytime_metrics = build_daytime_metrics(
        predictions, target_range, gamma=0.95, eta=10.0
    )
    meta = {
        "mode": "pvgis_stgnn",
        "model_type": "stgnn_enhanced_dropout",
        "feature_set": "full",
        "target_variable": "pv_power_output",
        "features": ["solar_irradiance_poa"],
        "n_features": 1,
        "seq_len": 2,
        "horizon": 1,
        "train_years": "2016,2017,2018",
        "test_year": 2019,
        "n_nodes": 2,
        "n_predictions": len(predictions),
        "epochs": 0,
        "batch_size": 2,
        "lr": 0.001,
        "anomaly_scores": None,
        "device": "cpu",
        "generated_utc": "2026-06-10T00:00:00+00:00",
        "wandb_enabled": False,
        "mc_dropout": True,
        "mc_samples": 4,
        "calibration": None,
        "interval_metrics": interval_metrics,
        "daytime_metrics": daytime_metrics,
        "daytime_threshold_wm2": DAYTIME_IRRADIANCE_THRESHOLD_WM2,
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
        daytime_csv = pd.read_csv(paths["metrics_daytime"])
    assert "## Daytime-only interval reliability" in report
    assert set(daytime_csv["stratum"]) == {
        "all",
        "daytime",
        "nighttime",
        "normal",
        "rare_extreme",
        "normal_daytime",
        "rare_extreme_daytime",
        "normal_nighttime",
        "rare_extreme_nighttime",
    }


if __name__ == "__main__":
    test_target_irradiance_alignment()
    test_supported_model_prediction_contracts()
    test_daytime_metrics_and_legacy_outputs()
    test_report_and_daytime_csv()
    print("PASS: PVGIS daytime/nighttime diagnostics")
