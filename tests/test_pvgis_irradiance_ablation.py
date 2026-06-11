"""Irradiance head/loss ablation tests for the pvgis_stgnn pipeline.

Historical behaviour (the pinned default): the STGNN irradiance head (head_ghi)
is CREATED but the training loss is plain MSE on pred_pv only, so head_ghi
receives no gradient. The ablation flags are:
    --use-irradiance-head / --no-use-irradiance-head   (default: True)
    --use-irradiance-loss / --no-use-irradiance-loss   (default: False)
    --irradiance-loss-weight <float>                   (default: 1.0)
head=False + loss=True is invalid (no head to supervise).
"""

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
    attach_anomaly_labels,
    build_datasets,
    compute_metrics,
    make_model,
    predict,
    predict_mc,
    train_model,
)
from physiq_pv.experiments.pvgis_stgnn_runner import _validate, build_arg_parser
from physiq_pv.model.graph_builder import build_graph

BASE_ARGS = [
    "--pvgis-dir", "x", "--model-type", "stgnn_enhanced_dropout",
    "--train-years", "2016,2017,2018", "--test-year", "2019",
    "--dropout", "0.3",
]


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


def _built():
    built = build_datasets(
        {2016: _tiny_year(2016)}, _tiny_year(2019), seq_len=24, horizon=1
    )
    edge_index, edge_weight = build_graph(
        built["lats"], built["lons"], max_dist_km=20.0
    )
    return built, edge_index, edge_weight


def _params_snapshot(module: torch.nn.Module) -> list:
    return [p.detach().clone() for p in module.parameters()]


def _params_equal(module: torch.nn.Module, snapshot: list) -> bool:
    return all(
        torch.equal(p.detach(), s) for p, s in zip(module.parameters(), snapshot)
    )


def _expect_system_exit(fn, label: str) -> None:
    try:
        fn()
        raise AssertionError(f"expected SystemExit: {label}")
    except SystemExit:
        pass


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #
def test_cli_defaults_match_current_behaviour() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(BASE_ARGS)
    assert args.use_irradiance_head is True
    assert args.use_irradiance_loss is False
    assert args.irradiance_loss_weight == 1.0
    _validate(args, None)  # default combo must validate


def test_cli_flags_parse() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        BASE_ARGS + ["--no-use-irradiance-head", "--no-use-irradiance-loss"]
    )
    assert args.use_irradiance_head is False
    assert args.use_irradiance_loss is False

    args = parser.parse_args(
        BASE_ARGS + [
            "--use-irradiance-head", "--use-irradiance-loss",
            "--irradiance-loss-weight", "0.0",
        ]
    )
    assert args.use_irradiance_head is True
    assert args.use_irradiance_loss is True
    assert args.irradiance_loss_weight == 0.0
    _validate(args, None)

    # Underscore aliases (W&B sweep ${args_no_boolean_flags} compatibility).
    args = parser.parse_args(
        BASE_ARGS + [
            "--use_irradiance_head", "--use_irradiance_loss",
            "--irradiance_loss_weight", "0.5",
        ]
    )
    assert args.use_irradiance_head is True
    assert args.use_irradiance_loss is True
    assert args.irradiance_loss_weight == 0.5
    args = parser.parse_args(BASE_ARGS + ["--no-use_irradiance_head"])
    assert args.use_irradiance_head is False


def test_cli_invalid_combinations_rejected() -> None:
    parser = build_arg_parser()
    # loss without head -> clear error
    args = parser.parse_args(
        BASE_ARGS + ["--no-use-irradiance-head", "--use-irradiance-loss"]
    )
    _expect_system_exit(lambda: _validate(args, None), "loss without head")
    # lstm has no irradiance head -> loss unsupported
    args = parser.parse_args([
        "--pvgis-dir", "x", "--model-type", "lstm",
        "--train-years", "2016", "--test-year", "2019",
        "--use-irradiance-loss",
    ])
    _expect_system_exit(lambda: _validate(args, None), "lstm with irradiance loss")
    # negative weight
    args = parser.parse_args(
        BASE_ARGS + ["--use-irradiance-loss", "--irradiance-loss-weight", "-1.0"]
    )
    _expect_system_exit(lambda: _validate(args, None), "negative weight")


# --------------------------------------------------------------------------- #
# Model surface
# --------------------------------------------------------------------------- #
def test_model_head_optional() -> None:
    with_head = make_model(n_nodes=2, seq_len=24, dropout=0.3, use_irradiance_head=True)
    without_head = make_model(
        n_nodes=2, seq_len=24, dropout=0.3, use_irradiance_head=False
    )
    assert with_head.head_ghi is not None
    assert without_head.head_ghi is None

    x = torch.randn(3, 2, 24, 11)  # make_model default n_features=11
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.ones(2)
    pred_ghi, pred_pv = with_head(x, edge_index, edge_weight, None)
    assert pred_ghi is not None and pred_ghi.shape == (3, 2)
    assert pred_pv.shape == (3, 2)
    pred_ghi, pred_pv = without_head(x, edge_index, edge_weight, None)
    assert pred_ghi is None
    assert pred_pv.shape == (3, 2)
    assert torch.isfinite(pred_pv).all()


# --------------------------------------------------------------------------- #
# Training-loss composition
# --------------------------------------------------------------------------- #
def test_default_training_gives_no_gradient_to_irradiance_head() -> None:
    built, edge_index, edge_weight = _built()
    torch.manual_seed(0)
    model = make_model(n_nodes=2, seq_len=24, n_features=built["n_features"], dropout=0.3)
    ghi_before = _params_snapshot(model.head_ghi)
    pv_before = _params_snapshot(model.head_pv)
    model = train_model(
        model, built["train"], edge_index, edge_weight,
        epochs=1, batch_size=8, lr=1e-3, device="cpu",
    )
    # Pinned default: irradiance head untouched, pv head trained.
    assert _params_equal(model.head_ghi, ghi_before)
    assert not _params_equal(model.head_pv, pv_before)
    hist = model.train_loss_history
    assert len(hist) == 1
    assert "loss/irradiance" not in hist[0]
    assert np.isclose(hist[0]["loss/total"], hist[0]["loss/pv"])


def test_irradiance_loss_trains_head_and_composes_total() -> None:
    built, edge_index, edge_weight = _built()
    assert built["train"].kt_target_all is not None
    torch.manual_seed(0)
    model = make_model(n_nodes=2, seq_len=24, n_features=built["n_features"], dropout=0.3)
    ghi_before = _params_snapshot(model.head_ghi)
    model = train_model(
        model, built["train"], edge_index, edge_weight,
        epochs=1, batch_size=8, lr=1e-3, device="cpu",
        use_irradiance_loss=True, irradiance_loss_weight=0.7,
    )
    assert not _params_equal(model.head_ghi, ghi_before)
    rec = model.train_loss_history[0]
    assert np.isclose(
        rec["loss/total"], rec["loss/pv"] + 0.7 * rec["loss/irradiance"], rtol=1e-5
    )
    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_irradiance_loss_weight_zero_runs() -> None:
    built, edge_index, edge_weight = _built()
    torch.manual_seed(0)
    model = make_model(n_nodes=2, seq_len=24, n_features=built["n_features"], dropout=0.3)
    model = train_model(
        model, built["train"], edge_index, edge_weight,
        epochs=1, batch_size=8, lr=1e-3, device="cpu",
        use_irradiance_loss=True, irradiance_loss_weight=0.0,
    )
    rec = model.train_loss_history[0]
    assert np.isclose(rec["loss/total"], rec["loss/pv"], rtol=1e-6)


def test_irradiance_loss_without_head_raises() -> None:
    built, edge_index, edge_weight = _built()
    model = make_model(
        n_nodes=2, seq_len=24, n_features=built["n_features"], dropout=0.3,
        use_irradiance_head=False,
    )
    try:
        train_model(
            model, built["train"], edge_index, edge_weight,
            epochs=1, batch_size=8, lr=1e-3, device="cpu",
            use_irradiance_loss=True,
        )
        raise AssertionError("expected ValueError: irradiance loss without head")
    except ValueError as e:
        assert "head" in str(e)


# --------------------------------------------------------------------------- #
# Production-only end-to-end (head=False, loss=False)
# --------------------------------------------------------------------------- #
def test_production_only_end_to_end() -> None:
    built, edge_index, edge_weight = _built()
    torch.manual_seed(0)
    model = make_model(
        n_nodes=2, seq_len=24, n_features=built["n_features"], dropout=0.3,
        enhanced_dropout=True, use_irradiance_head=False,
    )
    model = train_model(
        model, built["train"], edge_index, edge_weight,
        epochs=1, batch_size=8, lr=1e-3, device="cpu",
    )
    deterministic = predict(
        model, built["test"], edge_index, edge_weight, device="cpu", batch_size=8
    )
    assert np.isfinite(deterministic["y_pred"].to_numpy()).all()
    mc = predict_mc(
        model, built["test"], edge_index, edge_weight,
        device="cpu", batch_size=8, mc_samples=4,
    )
    assert {"y_pred_mean", "lower_pi", "upper_pi"} <= set(mc.columns)
    labelled = attach_anomaly_labels(mc, None)
    global_df, by_df = compute_metrics(labelled)
    assert np.isfinite(global_df.iloc[0]["MAE"])


if __name__ == "__main__":
    test_cli_defaults_match_current_behaviour()
    test_cli_flags_parse()
    test_cli_invalid_combinations_rejected()
    test_model_head_optional()
    test_default_training_gives_no_gradient_to_irradiance_head()
    test_irradiance_loss_trains_head_and_composes_total()
    test_irradiance_loss_weight_zero_runs()
    test_irradiance_loss_without_head_raises()
    test_production_only_end_to_end()
    print("PASS: PVGIS irradiance head/loss ablation")
