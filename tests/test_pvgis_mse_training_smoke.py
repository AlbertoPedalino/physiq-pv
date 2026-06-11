"""End-to-end CPU smoke test for the standard-MSE pvgis_stgnn training path.

The only test that actually trains the STGNN: tiny synthetic PVGIS year ->
build_datasets -> train_model (1 epoch, torch.nn.MSELoss) -> finite parameters.
Also pins the CLI surface: the baseline run parses, the removed weighted-MSE
ablation flags (--loss-type / --weighted-mse-*) are rejected.
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
    build_datasets,
    make_model,
    train_model,
)
from physiq_pv.experiments.pvgis_stgnn_runner import build_arg_parser
from physiq_pv.model.graph_builder import build_graph


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


def test_train_model_uses_plain_mse_only() -> None:
    src = inspect.getsource(train_model)
    assert "torch.nn.MSELoss()" in src
    assert "weighted" not in src
    assert "loss_type" not in src
    # No anomaly label can reach the training path.
    assert "anomaly" not in src.lower()
    assert "loss_type" not in inspect.signature(train_model).parameters


def test_cli_baseline_parses_and_weighted_flags_rejected() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([
        "--pvgis-dir", "x", "--model-type", "stgnn_enhanced_dropout",
        "--target-variable", "pv_power_output", "--feature-set", "full",
        "--train-years", "2016,2017,2018", "--test-year", "2019",
        "--seq-len", "24", "--horizon", "1", "--epochs", "5",
        "--batch-size", "16", "--lr", "0.001", "--dropout", "0.2",
        "--mc-dropout", "--mc-samples", "20",
        "--pv-target-clip-max", "none", "--seed", "1",
    ])
    assert args.model_type == "stgnn_enhanced_dropout"
    assert args.pv_target_clip_max is None
    assert not hasattr(args, "loss_type")
    for bad in (
        ["--loss-type", "mse"],
        ["--weighted-mse-peak-weight", "2.0"],
        ["--weighted-mse-daytime-threshold", "10.0"],
    ):
        try:
            parser.parse_args(["--pvgis-dir", "x"] + bad)
            raise AssertionError(f"expected rejection of removed flag {bad[0]}")
        except SystemExit:
            pass


def test_end_to_end_mse_train_smoke_cpu() -> None:
    seq_len, horizon = 24, 1
    built = build_datasets(
        {2016: _tiny_year(2016)}, _tiny_year(2019),
        seq_len=seq_len, horizon=horizon,
    )
    edge_index, edge_weight = build_graph(built["lats"], built["lons"], max_dist_km=20.0)
    torch.manual_seed(0)
    model = make_model(
        n_nodes=2, seq_len=seq_len, n_features=built["n_features"], dropout=0.2
    )
    model = train_model(
        model, built["train"], edge_index, edge_weight,
        epochs=1, batch_size=8, lr=1e-3, device="cpu",
    )
    for p in model.parameters():
        assert torch.isfinite(p).all(), "non-finite parameters after MSE training"


if __name__ == "__main__":
    test_train_model_uses_plain_mse_only()
    test_cli_baseline_parses_and_weighted_flags_rejected()
    test_end_to_end_mse_train_smoke_cpu()
    print("PASS: PVGIS standard-MSE training smoke")
