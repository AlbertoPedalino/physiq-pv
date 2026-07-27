from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from physiq_pv.anomaly_detection.common import (
    detector_features,
    runtime_environment,
    seed_everything,
)
from physiq_pv.anomaly_detection.evaluation import (
    attach_detector_scores,
    forecast_metrics_by_detection,
)
from physiq_pv.anomaly_detection.pvgis_climate import (
    SeasonalRobustScaler,
    prepare_pvgis_climate_data,
    raw_to_climate_frame,
)
from physiq_pv.anomaly_detection.mtgflow import (
    DynamicGraphAttention,
    MTGFlow,
    REFERENCE_CONFIG,
    REFERENCE_SEEDS,
    SpatioTemporalConditioner,
    fit_and_score_mtgflow,
    load_mtgflow_checkpoint,
    reference_protocol_deviations,
)
from physiq_pv.anomaly_detection.mtgflow.pipeline import (
    _regular_window_starts,
)
from physiq_pv.anomaly_detection.thresholds import (
    apply_entity_thresholds,
    apply_threshold,
    fit_entity_iqr_thresholds,
    fit_threshold,
)
from scripts.run_pvgis_mtgflow import _global_output, parse_args


def test_thresholds_are_training_score_only() -> None:
    scores = np.array([0.0, 1.0, 2.0, 3.0, np.nan])
    iqr = fit_threshold(scores, method="iqr", iqr_k=1.5)
    quantile = fit_threshold(scores, method="quantile", quantile=0.75)
    assert iqr.n_train_scores == 4
    assert iqr.value == 4.5
    assert quantile.value == 2.25
    assert apply_threshold([2.0, 5.0, np.nan], iqr).tolist() == [False, True, False]

    entity_scores = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0]])
    entity_thresholds = fit_entity_iqr_thresholds(entity_scores, scale=0.8)
    assert np.allclose(entity_thresholds, [3.6, 11.6])
    assert apply_entity_thresholds([[4.0, 11.0], [3.0, 12.0]], entity_thresholds).tolist() == [
        [True, False],
        [False, True],
    ]


def test_global_output_matches_downstream_detector_contract() -> None:
    times = pd.date_range("2016-01-01 02:00", periods=3, freq="h")
    starts = times - pd.Timedelta(hours=2)
    threshold = fit_threshold([0.0, 1.0, 2.0, 3.0], method="quantile", quantile=0.5)
    frame = _global_output(
        location="7",
        seed=15,
        window_starts=starts,
        timestamps=times,
        scores=np.asarray([0.5, 2.0, 3.0]),
        threshold=threshold,
    )
    assert {
        "location",
        "timestamp",
        "anomaly_score",
        "threshold",
        "is_anomaly",
        "method",
    } <= set(frame.columns)
    assert frame["method"].eq("mtgflow").all()
    assert frame["is_anomaly"].tolist() == [False, True, True]


def test_dynamic_graph_attention_is_row_normalised() -> None:
    torch.manual_seed(1)
    attention = DynamicGraphAttention(window_size=5, dropout=0.2).eval()
    adjacency = attention(torch.randn(3, 4, 5, 1))
    assert adjacency.shape == (3, 4, 4)
    assert torch.allclose(adjacency.sum(dim=-1), torch.ones(3, 4), atol=1e-6)


def test_spatiotemporal_conditioner_is_AH_plus_history() -> None:
    layer = SpatioTemporalConditioner(hidden_size=2)
    with torch.no_grad():
        for linear in (layer.graph_projection, layer.output_projection):
            linear.weight.copy_(torch.eye(2))
        layer.history_projection.weight.copy_(torch.eye(2))
    hidden = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]
    )
    adjacency = torch.tensor([[[0.25, 0.75], [1.0, 0.0]]])
    actual = layer(hidden, adjacency)
    neighbours = torch.einsum("bij,bjth->bith", adjacency, hidden)
    history = torch.zeros_like(neighbours)
    history[:, :, 1:] = hidden[:, :, :-1]
    assert torch.allclose(actual, torch.relu(neighbours + history))


def test_mtgflow_likelihood_aggregates_time_then_entities() -> None:
    torch.manual_seed(2)
    model = MTGFlow(
        n_blocks=2,
        input_size=1,
        hidden_size=8,
        n_hidden=1,
        window_size=4,
        n_entities=3,
    ).eval()
    x = torch.randn(5, 3, 4, 1)
    entity_log_prob, latent, adjacency = model.likelihood_components(x)
    assert entity_log_prob.shape == (5, 3)
    assert latent.shape == x.shape
    assert adjacency.shape == (5, 3, 3)
    assert torch.allclose(model.test(x), entity_log_prob.mean(dim=1))
    assert torch.allclose(model.locate(x), entity_log_prob)
    assert torch.allclose(model(x), entity_log_prob.mean())
    assert model.flow.entity_means.shape == (3, 1)
    assert model.rnn.num_layers == 1
    assert model.attention.query.bias is None
    assert model.attention.key.bias is None
    assert model.attention.dropout.p == REFERENCE_CONFIG.attention_dropout
    assert model.graph_condition.graph_projection.bias is None
    assert model.graph_condition.history_projection.bias is None
    assert model.graph_condition.output_projection.bias is None


def test_mtgflow_joint_optimization_reaches_every_module() -> None:
    torch.manual_seed(3)
    model = MTGFlow(
        n_blocks=1,
        input_size=1,
        hidden_size=8,
        n_hidden=1,
        window_size=4,
        n_entities=3,
    )
    loss = -model(torch.randn(2, 3, 4, 1))
    loss.backward()
    gradients = (
        model.attention.query.weight.grad,
        model.rnn.weight_ih_l0.grad,
        model.graph_condition.graph_projection.weight.grad,
        model.flow.blocks[0].parameter_net[0].weight.grad,
    )
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(gradient.abs().sum() > 0 for gradient in gradients)


def test_mtgflow_trainable_parameters_are_shared_across_entities() -> None:
    def trainable_parameter_count(n_entities: int) -> int:
        model = MTGFlow(
            n_blocks=2,
            input_size=1,
            hidden_size=8,
            n_hidden=1,
            window_size=4,
            n_entities=n_entities,
        )
        return sum(parameter.numel() for parameter in model.parameters())

    assert trainable_parameter_count(3) == trainable_parameter_count(9)


def test_mtgflow_windows_do_not_cross_gaps() -> None:
    timestamps = pd.DatetimeIndex(
        list(pd.date_range("2019-01-01", periods=5, freq="h"))
        + list(pd.date_range("2019-01-02", periods=5, freq="h"))
    )
    starts = _regular_window_starts(timestamps, window_size=4, stride=1)
    assert starts.tolist() == [0, 1, 5, 6]


def _seasonal_frame(year: int, offset: float = 0.0) -> pd.DataFrame:
    times = pd.date_range(f"{year}-01-01", periods=96, freq="h")
    frame = pd.DataFrame({"timestamp": times})
    base = np.tile(np.arange(24, dtype=float), 4) + offset
    for name in (
        "temperature_2m", "solar_irradiance_poa", "wind_speed_10m", "kt",
        "kt_std_3h", "dghi_dt", "dni_norm", "dhi_norm",
    ):
        frame[name] = base
    return frame


def test_seasonal_scaler_uses_fitted_training_buckets() -> None:
    train = _seasonal_frame(2017)
    shifted = _seasonal_frame(2019, offset=10.0)
    scaler = SeasonalRobustScaler.fit(train)
    transformed_train = scaler.transform(train)
    transformed_shifted = scaler.transform(shifted)
    assert np.allclose(transformed_train["temperature_2m"], 0.0)
    assert (transformed_shifted["temperature_2m"] > 0).all()


def test_raw_to_climate_frame_excludes_pv_target() -> None:
    times = pd.date_range("2019-01-01", periods=4, freq="h")
    column = np.arange(4, dtype=np.float32)[:, None]
    raw = {
        "times": times,
        "temp": column,
        "solar_wm2": column,
        "wind": column,
        "kt": column,
        "kt_std": column,
        "dghi": column,
        "dni": column,
        "dhi": column,
        "sin": column,
        "cos": column,
        "day": np.ones_like(column, dtype=bool),
        "pv": column + 100.0,
    }
    frame = raw_to_climate_frame(raw)
    assert "pv_power_output" not in frame
    assert "pv" not in frame
    assert {"hour_sin", "hour_cos", "doy_sin", "doy_cos"}.issubset(frame.columns)


def test_detector_rejects_forecast_target_leakage() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2019-01-01", periods=2, freq="h"),
            "temperature_2m": [0.0, 1.0],
            "pv_power_output": [2.0, 3.0],
        }
    )
    try:
        detector_features(frame)
    except ValueError as exc:
        assert "Supervision or forecast targets" in str(exc)
    else:
        raise AssertionError("Detector accepted pv_power_output as an input feature.")

    labelled = frame.drop(columns="pv_power_output").assign(is_anomaly=[False, True])
    try:
        detector_features(labelled)
    except ValueError as exc:
        assert "is_anomaly" in str(exc)
    else:
        raise AssertionError("Detector accepted an anomaly label as an input feature.")


def test_reference_configuration_defaults_do_not_drift() -> None:
    mtgflow = inspect.signature(fit_and_score_mtgflow).parameters
    assert mtgflow["epochs"].default == REFERENCE_CONFIG.epochs
    assert mtgflow["window_size"].default == REFERENCE_CONFIG.window_size
    assert mtgflow["batch_size"].default == REFERENCE_CONFIG.batch_size
    assert mtgflow["lr"].default == REFERENCE_CONFIG.learning_rate
    assert mtgflow["weight_decay"].default == REFERENCE_CONFIG.weight_decay
    assert mtgflow["train_stride"].default == REFERENCE_CONFIG.train_stride
    assert mtgflow["score_stride"].default == REFERENCE_CONFIG.score_stride
    assert mtgflow["n_blocks"].default == REFERENCE_CONFIG.n_blocks
    assert mtgflow["seed"].default == REFERENCE_SEEDS[0]
    preparation = inspect.signature(prepare_pvgis_climate_data).parameters
    assert preparation["train_end"].default == 2018
    assert preparation["validation_year"].default is None
    assert preparation["seasonal_normalization"].default is False
    parsed = parse_args(["--manifest", "manifest.csv", "--out-dir", "out"])
    assert parsed.seeds == REFERENCE_SEEDS
    assert parsed.score_stride == REFERENCE_CONFIG.score_stride
    assert reference_protocol_deviations(vars(parsed), REFERENCE_SEEDS, True) == []
    dense = parse_args(
        [
            "--manifest",
            "manifest.csv",
            "--out-dir",
            "out",
            "--score-stride",
            "1",
            "--seed",
            "15",
        ]
    )
    deviations = reference_protocol_deviations(vars(dense), (15,), False)
    assert any(item.startswith("score_stride=") for item in deviations)
    assert any(item.startswith("seeds=") for item in deviations)
    assert "preparation metadata not verified" in deviations


def _mtgflow_frame(start: str, n: int, offset: float = 0.0) -> pd.DataFrame:
    index = np.arange(n, dtype=np.float32)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=n, freq="h"),
            "f1": np.sin(index / 3.0) + offset,
            "f2": np.cos(index / 5.0) - offset,
            "is_daytime": True,
        }
    )


def test_mtgflow_validation_is_not_used_for_fitting() -> None:
    kwargs = dict(
        epochs=1,
        window_size=4,
        train_stride=2,
        score_stride=2,
        batch_size=8,
        n_blocks=1,
        hidden_size=8,
        device="cpu",
        seed=15,
    )
    train = _mtgflow_frame("2017-01-01", 16)
    test = _mtgflow_frame("2019-01-01", 12)
    first = fit_and_score_mtgflow(
        train,
        test,
        validation=_mtgflow_frame("2018-01-01", 8),
        **kwargs,
    )
    shifted = fit_and_score_mtgflow(
        train,
        test,
        validation=_mtgflow_frame("2018-01-01", 8, offset=10_000.0),
        **kwargs,
    )
    without_validation = fit_and_score_mtgflow(train, test, **kwargs)
    assert np.allclose(first.train_scores, shifted.train_scores)
    assert np.allclose(first.test_scores, shifted.test_scores)
    assert np.allclose(first.train_entity_scores, shifted.train_entity_scores)
    assert np.allclose(first.train_scores, without_validation.train_scores)
    assert np.allclose(first.test_scores, without_validation.test_scores)
    assert np.allclose(first.train_scores, first.train_entity_scores.sum(axis=1))
    assert np.allclose(first.test_scores, first.test_entity_scores.sum(axis=1))
    assert first.train_window_starts is not None
    assert first.test_window_starts is not None
    assert first.train_window_starts[0] == train["timestamp"].iloc[0]
    assert first.train_timestamps[0] == train["timestamp"].iloc[3]
    assert first.metadata["runtime_dependency_on_official_repo"] is False
    assert first.metadata["validation_used_for_training"] is False
    assert first.metadata["validation_supplied"] is True
    assert without_validation.metadata["validation_supplied"] is False
    assert first.metadata["entity_score_definition"] == "negative_log_likelihood_divided_by_K"


def test_mtgflow_checkpoint_contains_model_scaler_and_environment() -> None:
    train = _mtgflow_frame("2017-01-01", 16)
    validation = _mtgflow_frame("2018-01-01", 8)
    test = _mtgflow_frame("2019-01-01", 12)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        result = fit_and_score_mtgflow(
            train,
            test,
            validation=validation,
            epochs=1,
            window_size=4,
            train_stride=2,
            score_stride=1,
            batch_size=8,
            n_blocks=1,
            hidden_size=8,
            device="cpu",
            seed=15,
            checkpoint_path=checkpoint,
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["format_version"] == 2
        assert payload["model_class"] == "MTGFlow"
        assert payload["features"] == ["f1", "f2"]
        assert payload["normalization"]["kind"] == "training_only_zscore"
        assert payload["training"]["score_stride"] == 1
        assert payload["environment"]["deterministic_algorithms"] is True
        restored, restored_payload = load_mtgflow_checkpoint(checkpoint)
        assert restored.training is False
        assert restored_payload["features"] == payload["features"]
        for name, value in restored.state_dict().items():
            assert torch.equal(value, payload["model_state_dict"][name])
        assert len(result.train_scores) == 13
        assert len(result.test_scores) == 9
        assert result.metadata["scoring_profile"] == "custom_window_sampling"


def test_seed_and_environment_capture_are_deterministic() -> None:
    seed_everything(15)
    first = torch.rand(3)
    seed_everything(15)
    second = torch.rand(3)
    assert torch.equal(first, second)
    environment = runtime_environment()
    assert environment["python"]
    assert environment["numpy"] == np.__version__
    assert environment["deterministic_algorithms"] is True


def test_forecast_evaluation_is_posthoc_and_stratified() -> None:
    times = pd.date_range("2019-01-01", periods=4, freq="h")
    predictions = pd.DataFrame(
        {
            "location": ["a"] * 4,
            "timestamp": times,
            "y_true": [0.0, 1.0, 2.0, 3.0],
            "y_pred": [0.0, 1.5, 1.0, 5.0],
            "y_pred_std": [0.1, 0.1, 0.4, 0.5],
            "lower_pi": [-0.2, 1.0, 0.0, 4.0],
            "upper_pi": [0.2, 2.0, 2.0, 6.0],
        }
    )
    detector = pd.DataFrame(
        {
            "location": ["a"] * 4,
            "timestamp": times,
            "method": ["mtgflow"] * 4,
            "anomaly_score": [0.0, 0.1, 3.0, 4.0],
            "threshold": [2.0] * 4,
            "is_anomaly": [False, False, True, True],
        }
    )
    joined = attach_detector_scores(predictions, detector, method="mtgflow")
    metrics = forecast_metrics_by_detection(joined).set_index("stratum")
    assert metrics.loc["normal", "n"] == 2
    assert metrics.loc["anomaly", "n"] == 2
    assert metrics.loc["anomaly", "mean_predictive_std"] > metrics.loc["normal", "mean_predictive_std"]


if __name__ == "__main__":
    test_thresholds_are_training_score_only()
    test_global_output_matches_downstream_detector_contract()
    test_dynamic_graph_attention_is_row_normalised()
    test_spatiotemporal_conditioner_is_AH_plus_history()
    test_mtgflow_likelihood_aggregates_time_then_entities()
    test_mtgflow_joint_optimization_reaches_every_module()
    test_mtgflow_trainable_parameters_are_shared_across_entities()
    test_mtgflow_windows_do_not_cross_gaps()
    test_seasonal_scaler_uses_fitted_training_buckets()
    test_raw_to_climate_frame_excludes_pv_target()
    test_detector_rejects_forecast_target_leakage()
    test_reference_configuration_defaults_do_not_drift()
    test_mtgflow_validation_is_not_used_for_fitting()
    test_mtgflow_checkpoint_contains_model_scaler_and_environment()
    test_seed_and_environment_capture_are_deterministic()
    test_forecast_evaluation_is_posthoc_and_stratified()
    print("PASS: PVGIS climate anomaly tests")
