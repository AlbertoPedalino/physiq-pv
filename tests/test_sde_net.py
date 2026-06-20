"""Synthetic/CPU tests for the neural-SDE ST-GNN (Monaco SDE U-Net; SDE-Net core, Kong et al. 2020).

No PVGIS data needed: a tiny synthetic year drives build_datasets + train_model.
Covers:
  1. SDEBlock output shape and raw per-feature diffusion gate g in (0, 1);
  2. forward is deterministic with stochastic=False, varies with stochastic=True;
  3. the diffusion net learns to separate in-distribution from Gaussian OOD;
  4. train_model runs, stays finite, and logs the g_in / g_ood / g_ratio metrics;
  5. predict is deterministic; predict_sde returns empirical-quantile intervals.
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

from physiq_pv.data.pvgis_dataset import build_datasets, make_model
from physiq_pv.model.st_gnn import SDEBlock
from physiq_pv.model.graph_builder import build_graph
from physiq_pv.training.train_loop import train_model
from physiq_pv.training.uncertainty import predict, predict_sde


def _tiny_year(year: int, t_hours: int = 72) -> xr.Dataset:
    times = pd.date_range(f"{year}-06-01", periods=t_hours, freq="h")
    hours = times.hour.to_numpy()
    solar_1d = np.where(
        (hours >= 6) & (hours <= 18),
        800.0 * np.sin(np.pi * (hours - 6) / 12.0),
        0.0,
    )
    solar = np.stack([solar_1d, solar_1d * 0.9]).astype(np.float32)
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
    built = build_datasets({2016: _tiny_year(2016)}, _tiny_year(2019),
                           seq_len=24, horizon=1)
    edge_index, edge_weight = build_graph(built["lats"], built["lons"], max_dist_km=20.0)
    return built, edge_index, edge_weight


def _model(built):
    torch.manual_seed(0)
    return make_model(n_nodes=2, seq_len=24, n_features=built["n_features"],
                      dropout=0.2, n_sde_steps=4, sigma_max=0.5)


# --- 1. SDEBlock shape + bounded diffusion ---------------------------------- #
def test_sdeblock_shape_and_diffusion_bounds() -> None:
    torch.manual_seed(0)
    sde = SDEBlock(dim=16, n_steps=4, sigma_max=0.5)
    x0 = torch.randn(3, 5, 16)
    with torch.no_grad():
        xT, g = sde(x0, stochastic=True)
    assert xT.shape == x0.shape          # (B, N, dim)
    assert g.shape == x0.shape           # raw gate: one value per feature
    assert float(g.min()) >= 0.0 and float(g.max()) <= 1.0  # bare sigmoid in (0, 1)


# --- 2. deterministic vs stochastic forward --------------------------------- #
def test_forward_deterministic_vs_stochastic() -> None:
    built, ei, ew = _built()
    model = _model(built).eval()
    x, y, _ = next(iter(torch.utils.data.DataLoader(built["train"], batch_size=4)))
    d1 = model(x, ei, ew, None, stochastic=False)[1]
    d2 = model(x, ei, ew, None, stochastic=False)[1]
    assert torch.allclose(d1, d2)                 # drift-only is deterministic
    s1 = model(x, ei, ew, None, stochastic=True)[1]
    s2 = model(x, ei, ew, None, stochastic=True)[1]
    assert not torch.allclose(s1, s2)             # Brownian paths differ
    assert d1.shape == y.shape


# --- 3. the diffusion net learns OOD separation ----------------------------- #
def test_diffusion_learns_ood_separation() -> None:
    torch.manual_seed(0)
    sde = SDEBlock(dim=16, n_steps=4, sigma_max=0.5)
    opt_g = torch.optim.AdamW(sde.diffusion_net.parameters(), lr=1e-2)
    x_in = torch.randn(32, 16)
    bce = torch.nn.functional.binary_cross_entropy
    for _ in range(150):
        x_ood = x_in + 1.5 * torch.randn_like(x_in)
        g_in = sde.diffusion(x_in)                # (32, 16) in (0, 1)
        g_ood = sde.diffusion(x_ood)
        # Monaco/Kong BCE: g -> 0 in-distribution, g -> 1 on OOD.
        loss_g = bce(g_in, torch.zeros_like(g_in)) + bce(g_ood, torch.ones_like(g_ood))
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
    assert g_ood.mean().item() > g_in.mean().item()   # high diffusion on OOD


# --- 4. train_model runs and logs the SDE diagnostics ----------------------- #
def test_train_model_runs_and_logs_g() -> None:
    built, ei, ew = _built()
    model = train_model(_model(built), built["train"], ei, ew,
                        epochs=2, batch_size=8, lr=1e-3, device="cpu",
                        ood_noise_std=0.1, feature_names=built["features"])
    rec = model.train_loss_history[-1]
    assert {"loss/pv", "train/g_in", "train/g_ood", "train/g_ratio"} <= set(rec)
    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_train_model_rejects_zero_ood_noise() -> None:
    built, ei, ew = _built()
    try:
        train_model(_model(built), built["train"], ei, ew,
                    epochs=1, batch_size=8, lr=1e-3, device="cpu",
                    ood_noise_std=0.0, feature_names=built["features"])
        raise AssertionError("expected ValueError for ood_noise_std=0")
    except ValueError:
        pass


# --- 5. inference: deterministic predict + SDE-sampled intervals ------------ #
def test_predict_is_deterministic() -> None:
    built, ei, ew = _built()
    model = _model(built)
    df1 = predict(model, built["test"], ei, ew, "cpu", batch_size=8)
    df2 = predict(model, built["test"], ei, ew, "cpu", batch_size=8)
    assert np.allclose(df1["y_pred"].to_numpy(), df2["y_pred"].to_numpy())


def test_predict_sde_returns_intervals() -> None:
    built, ei, ew = _built()
    model = _model(built)
    df = predict_sde(model, built["test"], ei, ew, "cpu", batch_size=8, mc_samples=8)
    assert {"y_pred_mean", "y_pred_std", "lower_pi", "upper_pi"} <= set(df.columns)
    assert (df["upper_pi"] >= df["lower_pi"]).all()
    assert df["y_pred_std"].to_numpy().std() > 0.0  # non-degenerate uncertainty


# --- 6. train-normal-only --------------------------------------------------- #
def test_anomaly_mask_marks_target_and_input_history() -> None:
    built, _, _ = _built()
    train = built["train"]
    year, start = train.samples[0]
    input_timestamp = train.times_by_year[year][start]
    target_timestamp = train.target_time_all[0]
    scores = pd.DataFrame({
        "location": ["loc_a", "loc_a"],
        "timestamp": [input_timestamp, target_timestamp],
        "label": ["extreme_wind_condition", "unusually_low_solar_potential"],
    })
    assert train.attach_anomaly_mask(scores) == 1
    assert train.anomaly_mask_all[0, 0]
    assert train.anomaly_history_mask_all[0, 0]
    assert not train.anomaly_mask_all[0, 1]
    assert not train.anomaly_history_mask_all[0, 1]


def test_train_normal_only_runs_with_mask() -> None:
    built, ei, ew = _built()
    train = built["train"]
    scores = pd.DataFrame({
        "location": "loc_a",                       # mark loc_a rare at every target time
        "timestamp": train.target_time_all,
        "label": "unusually_low_solar_potential",
    })
    assert train.attach_anomaly_mask(scores) > 0
    model = train_model(_model(built), train, ei, ew, epochs=2, batch_size=8,
                        lr=1e-3, device="cpu", ood_noise_std=0.1,
                        feature_names=built["features"], train_normal_only=True)
    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_train_normal_only_requires_mask() -> None:
    built, ei, ew = _built()
    try:
        train_model(_model(built), built["train"], ei, ew, epochs=1, batch_size=8,
                    lr=1e-3, device="cpu", ood_noise_std=0.1,
                    feature_names=built["features"], train_normal_only=True)
        raise AssertionError("expected ValueError without an anomaly mask")
    except ValueError:
        pass


# --- 7. CLC sharpness/reliability primitive --------------------------------- #
def test_clc_primitive() -> None:
    from physiq_pv.reporting.daytime_bin_anomaly_report import _clc
    assert abs(_clc(0.2, 0.95, 0.95, 9.0) - 0.4) < 1e-12


if __name__ == "__main__":
    test_sdeblock_shape_and_diffusion_bounds()
    test_forward_deterministic_vs_stochastic()
    test_diffusion_learns_ood_separation()
    test_train_model_runs_and_logs_g()
    test_train_model_rejects_zero_ood_noise()
    test_predict_is_deterministic()
    test_predict_sde_returns_intervals()
    test_anomaly_mask_marks_target_and_input_history()
    test_train_normal_only_runs_with_mask()
    test_train_normal_only_requires_mask()
    test_clc_primitive()
    print("PASS: neural-SDE ST-GNN tests")
