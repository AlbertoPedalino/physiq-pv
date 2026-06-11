"""Tests for the weighted_mse training-loss ablation (loss_type=mse|weighted_mse).

Covers: backward-compatible CLI defaults, correct weight assignment (day/night
+ watt bands, boundaries included), sum(w)-normalisation of the loss, the
"weighted flags require --loss-type weighted_mse" guard, that no anomaly label
can enter the loss, and an end-to-end CPU smoke run on a tiny synthetic year.
"""

import inspect
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import xarray as xr

from physiq_pv.data.pvgis_stgnn_dataset import (
    WEIGHTED_MSE_DEFAULTS,
    build_datasets,
    compute_weighted_mse_weights,
    make_model,
    train_model,
    weighted_mse_loss,
)
from physiq_pv.experiments.pvgis_stgnn_runner import _validate, build_arg_parser
from physiq_pv.model.graph_builder import build_graph


def test_cli_defaults_backward_compatible() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.loss_type == "mse"
    assert args.weighted_mse_daytime_threshold == WEIGHTED_MSE_DEFAULTS["daytime_threshold"]
    assert args.weighted_mse_night_weight == WEIGHTED_MSE_DEFAULTS["night_weight"]
    assert args.weighted_mse_day_weight == WEIGHTED_MSE_DEFAULTS["day_weight"]
    assert args.weighted_mse_high_threshold_w == WEIGHTED_MSE_DEFAULTS["high_threshold_w"]
    assert args.weighted_mse_high_weight == WEIGHTED_MSE_DEFAULTS["high_weight"]
    assert args.weighted_mse_peak_threshold_w == WEIGHTED_MSE_DEFAULTS["peak_threshold_w"]
    assert args.weighted_mse_peak_weight == WEIGHTED_MSE_DEFAULTS["peak_weight"]
    # Underscore aliases (W&B sweep ${args}) parse too.
    assert parser.parse_args(["--loss_type", "weighted_mse"]).loss_type == "weighted_mse"


def test_weight_assignment_bands_and_boundaries() -> None:
    y_watts = np.array([50.0, 79.9, 80.0, 99.9, 100.0, 150.0, 150.0, 150.0])
    solar = np.array([500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 5.0, 10.0])
    weights = compute_weighted_mse_weights(y_watts, solar)
    # daytime <80 -> 1.0 | [80,100) -> 1.5 | >=100 -> 2.0
    # solar=5 and solar=10 (threshold is strict >) -> nighttime 0.2 even at 150 W
    expected = np.array([1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 0.2, 0.2], dtype=np.float32)
    np.testing.assert_array_equal(weights, expected)
    assert weights.dtype == np.float32


def test_weight_param_validation() -> None:
    y = np.zeros(2)
    s = np.zeros(2)
    for kwargs in (
        {"night_weight": 0.0},
        {"day_weight": -1.0},
        {"high_threshold_w": 100.0, "peak_threshold_w": 100.0},
    ):
        try:
            compute_weighted_mse_weights(y, s, **kwargs)
            raise AssertionError(f"expected ValueError for {kwargs}")
        except ValueError:
            pass
    try:
        compute_weighted_mse_weights(np.zeros(3), np.zeros(2))
        raise AssertionError("expected ValueError for shape mismatch")
    except ValueError:
        pass


def test_loss_normalises_by_sum_of_weights() -> None:
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([0.0, 0.0, 0.0])
    weights = torch.tensor([0.2, 1.0, 2.0])
    expected = (0.2 * 1.0 + 1.0 * 4.0 + 2.0 * 9.0) / (0.2 + 1.0 + 2.0 + 1e-8)
    assert abs(weighted_mse_loss(pred, target, weights).item() - expected) < 1e-6


def test_uniform_weights_match_plain_mse() -> None:
    g = torch.Generator().manual_seed(0)
    pred = torch.rand(64, generator=g)
    target = torch.rand(64, generator=g)
    weights = torch.ones(64)
    wmse = weighted_mse_loss(pred, target, weights).item()
    mse = torch.nn.functional.mse_loss(pred, target).item()
    assert abs(wmse - mse) < 1e-6


def test_training_invariants() -> None:
    src = inspect.getsource(train_model)
    # mse default path stays the historical torch.nn.MSELoss.
    assert "torch.nn.MSELoss()" in src
    assert inspect.signature(train_model).parameters["loss_type"].default == "mse"
    # No anomaly label can reach the loss: the loss path never touches the
    # anomaly columns/labels (docstrings may mention the word to state this).
    for source in (
        src,
        inspect.getsource(compute_weighted_mse_weights),
        inspect.getsource(weighted_mse_loss),
    ):
        assert "anomaly_label" not in source
        assert "anomaly_group" not in source
        assert "SPECIFIC_ANOMALY_LABELS" not in source


def _parse(extra):
    base = ["--pvgis-dir", "x", "--train-years", "2016", "--test-year", "2019"]
    return build_arg_parser().parse_args(base + extra)


def test_weighted_flags_require_weighted_loss_type() -> None:
    try:
        _validate(_parse(["--weighted-mse-peak-weight", "3.0"]), None)
        raise AssertionError("expected SystemExit: weighted flag under loss_type=mse")
    except SystemExit:
        pass
    # Same flag under weighted_mse is fine.
    _validate(_parse(["--loss-type", "weighted_mse", "--weighted-mse-peak-weight", "3.0"]), None)
    # Invalid band ordering fails under weighted_mse.
    try:
        _validate(
            _parse([
                "--loss-type", "weighted_mse",
                "--weighted-mse-high-threshold-w", "120.0",
            ]),
            None,
        )
        raise AssertionError("expected SystemExit: high_threshold >= peak_threshold")
    except SystemExit:
        pass


def _tiny_year(year: int, t_hours: int = 72) -> xr.Dataset:
    """Synthetic PVGIS year: 2 nodes, diurnal solar, pv up to ~144 W."""
    times = pd.date_range(f"{year}-06-01", periods=t_hours, freq="h")
    hours = times.hour.to_numpy()
    solar_1d = np.where(
        (hours >= 6) & (hours <= 18),
        800.0 * np.sin(np.pi * (hours - 6) / 12.0),
        0.0,
    )
    solar = np.stack([solar_1d, solar_1d * 0.9]).astype(np.float32)  # (N=2, T)
    pv = (solar * 0.18).astype(np.float32)
    temp = np.full_like(solar, 20.0)
    wind = np.full_like(solar, 3.0)
    return xr.Dataset(
        {
            "temperature_2m": (("location", "time"), temp),
            "solar_irradiance_poa": (("location", "time"), solar),
            "wind_speed_10m": (("location", "time"), wind),
            "pv_power_output": (("location", "time"), pv),
        },
        coords={
            "time": times,
            "location": ["loc_a", "loc_b"],
            "lat": ("location", [45.0, 45.05]),
            "lon": ("location", [7.6, 7.65]),
        },
    )


def test_end_to_end_train_smoke_cpu() -> None:
    seq_len, horizon = 24, 1
    built = build_datasets(
        {2016: _tiny_year(2016)}, _tiny_year(2019),
        seq_len=seq_len, horizon=horizon,
    )
    train = built["train"]
    # The TRAIN dataset now carries target-time solar irradiance, aligned on
    # the same target index as y (i + seq_len + horizon - 1).
    assert train.solar_irradiance_poa_target_all is not None
    assert train.solar_irradiance_poa_target_all.shape == train.y_true_all.shape
    raw_solar = _tiny_year(2016)["solar_irradiance_poa"].values.T  # (T, N)
    tgt = seq_len + horizon - 1
    np.testing.assert_allclose(
        train.solar_irradiance_poa_target_all[0], raw_solar[tgt], rtol=1e-6
    )

    edge_index, edge_weight = build_graph(built["lats"], built["lons"], max_dist_km=20.0)
    torch.manual_seed(0)
    for loss_type in ("mse", "weighted_mse"):
        model = make_model(
            n_nodes=2, seq_len=seq_len, n_features=built["n_features"], dropout=0.2
        )
        model = train_model(
            model, train, edge_index, edge_weight,
            epochs=1, batch_size=8, lr=1e-3, device="cpu",
            loss_type=loss_type,
        )
        for p in model.parameters():
            assert torch.isfinite(p).all(), f"non-finite params after {loss_type}"


if __name__ == "__main__":
    test_cli_defaults_backward_compatible()
    test_weight_assignment_bands_and_boundaries()
    test_weight_param_validation()
    test_loss_normalises_by_sum_of_weights()
    test_uniform_weights_match_plain_mse()
    test_training_invariants()
    test_weighted_flags_require_weighted_loss_type()
    test_end_to_end_train_smoke_cpu()
    print("PASS: PVGIS weighted MSE loss ablation")
