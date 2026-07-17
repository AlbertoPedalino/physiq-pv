"""Synthetic/CPU tests for the neural-SDE ST-GNN (Monaco SDE U-Net; SDE-Net core, Kong et al. 2020).

No PVGIS data needed: a tiny synthetic year drives build_datasets + train_model.
Covers:
  1. Monaco-aligned per-stage diffusion gates and their bounds;
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

from physiq_pv.data.pvgis_dataset import build_datasets, build_year_raw, make_model
from physiq_pv.model.st_gnn import MonacoDiffusionEncoder
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


# --- 1. Monaco stage alignment + bounded diffusion -------------------------- #


def test_build_year_raw_uses_tilted_poa_fallback() -> None:
    ds = _tiny_year(2019)
    physical_poa = ds["solar_irradiance_poa"].copy()
    ds["direct_irradiance_tilted"] = physical_poa * 0.75
    ds["diffuse_irradiance_tilted"] = physical_poa * 0.25
    ds["solar_irradiance_poa"] = physical_poa * 0.0

    raw = build_year_raw(ds, "pv_power_output")

    np.testing.assert_allclose(raw["solar_wm2"], physical_poa.transpose("time", "location").values)
    assert raw["day"].any()


def test_monaco_diffusion_stage_shapes_and_bounds() -> None:
    built, ei, ew = _built()
    model = _model(built).eval()
    x, _, _ = next(iter(torch.utils.data.DataLoader(built["train"], batch_size=3)))
    with torch.no_grad():
        terms = model.diffusion(x, ei, ew)
        _, pred, returned_terms = model(
            x, ei, ew, None, stochastic=True, return_diffusion=True
        )
    assert len(terms) == model.n_sde_stages == 4  # BiLSTM + three GAT stages
    assert len(returned_terms) == len(terms)
    assert pred.shape == x.shape[:2]
    for gate in terms:
        assert gate.shape == (x.shape[0], x.shape[1], 96)
        assert float(gate.min()) >= 0.0 and float(gate.max()) <= 1.0


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
    diffusion = MonacoDiffusionEncoder(
        n_features=3,
        seq_len=4,
        d_model=4,
        gat_dim=8,
        gat_heads=2,
        gat_layers=1,
        bilstm_pooling="attn",
        use_temporal_encoder=True,
    )
    opt_g = torch.optim.Adam(diffusion.parameters(), lr=1e-2)
    x_in = torch.randn(24, 2, 4, 3)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.ones(2)
    bce = torch.nn.functional.binary_cross_entropy
    for _ in range(100):
        x_ood = x_in + 1.5 * torch.randn_like(x_in)
        g_in = diffusion(x_in, edge_index, edge_weight)
        g_ood = diffusion(x_ood, edge_index, edge_weight)
        loss_g = sum(bce(g, torch.zeros_like(g)) for g in g_in)
        loss_g = loss_g + sum(bce(g, torch.ones_like(g)) for g in g_ood)
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
    mean_in = torch.stack([g.mean() for g in g_in]).mean()
    mean_ood = torch.stack([g.mean() for g in g_ood]).mean()
    assert mean_ood.item() > mean_in.item()   # high diffusion on OOD


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


# --- 6. train-normal-only window filtering --------------------------------- #
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


def test_train_normal_only_filters_windows_and_runs() -> None:
    built, ei, ew = _built()
    train = built["train"]
    first_target = train.target_time_all[0]
    later_target = train.target_time_all[10]
    scores = pd.DataFrame({
        "location": ["loc_a", "loc_b"],
        "timestamp": [first_target, later_target],
        "label": ["unusually_low_solar_potential", "extreme_wind_condition"],
    })
    before = len(train)
    assert train.attach_anomaly_mask(scores) > 0
    kept, total = train.filter_normal_only_windows()
    assert total == before
    assert 0 < kept < before
    assert not (train.anomaly_mask_all | train.anomaly_history_mask_all).any()
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


def test_frequency_weighted_bin_summary() -> None:
    from physiq_pv.reporting.daytime_bin_anomaly_report import (
        build_frequency_weighted_bin_summary,
    )

    bins = pd.DataFrame({
        "bin": ["daytime_0_20", "daytime_gt_100"],
        "category": ["unusually_low_solar_potential"] * 2,
        "count": [80, 20],
        "picp": [0.95, 0.80],
        "mae": [1.0, 3.0],
        "rmse": [2.0, 4.0],
    })
    result = build_frequency_weighted_bin_summary(bins, 1_000, 0.95)
    row = result.set_index("category").loc["unusually_low_solar_potential"]

    assert row["count"] == 100
    assert abs(row["frequency_of_daytime"] - 0.1) < 1e-12
    assert abs(row["frequency_weighted_picp"] - 0.92) < 1e-12
    assert abs(row["frequency_weighted_abs_picp_gap"] - 0.03) < 1e-12
    assert abs(row["frequency_weighted_undercoverage_gap"] - 0.03) < 1e-12
    assert row["max_gap_bin"] == "daytime_gt_100"
    assert abs(row["max_gap_bin_frequency"] - 0.2) < 1e-12


def test_reference_peak_bins_use_global_scale() -> None:
    from physiq_pv.reporting.daytime_bin_anomaly_report import (
        add_reference_peak_production_pct,
    )

    day = pd.DataFrame({
        "location": ["a", "a", "b", "b"],
        "y_true": [20.0, 100.0, 30.0, 60.0],
    })
    stats = add_reference_peak_production_pct(day, 1.0)

    np.testing.assert_allclose(day["production_pct"], [20.0, 100.0, 30.0, 60.0])
    np.testing.assert_allclose(day["reference_peak_w"], [100.0, 100.0, 100.0, 100.0])
    assert stats["reference_peak_scope"] == "global_daytime"
    assert stats["reference_peak_w"] == 100.0


def test_figure_sample_uses_reference_peak() -> None:
    from physiq_pv.reporting.posthoc_outputs import (
        attach_reference_peak_production_pct,
    )

    sample = pd.DataFrame({
        "location": ["a", "a", "b"],
        "y_true": [20.0, 80.0, 30.0],
    })
    peaks = pd.DataFrame({
        "reference_peak_w": [100.0],
    })
    result = attach_reference_peak_production_pct(sample, peaks)

    np.testing.assert_allclose(result["production_pct"], [20.0, 80.0, 30.0])


def test_production_peak_nmpil_is_rowwise() -> None:
    from physiq_pv.reporting.daytime_bin_anomaly_report import subset_metrics

    sample = pd.DataFrame({
        "y_true": [50.0, 100.0],
        "y_pred": [50.0, 100.0],
        "y_std": [1.0, 1.0],
        "lower_pi": [40.0, 80.0],
        "upper_pi": [60.0, 120.0],
        "production_reference_w": [100.0, 200.0],
    })
    metrics = subset_metrics(sample)

    assert abs(metrics["mpiw"] - 30.0) < 1e-12
    assert abs(metrics["production_peak_nmpil"] - 0.2) < 1e-12


if __name__ == "__main__":
    test_monaco_diffusion_stage_shapes_and_bounds()
    test_build_year_raw_uses_tilted_poa_fallback()
    test_forward_deterministic_vs_stochastic()
    test_diffusion_learns_ood_separation()
    test_train_model_runs_and_logs_g()
    test_train_model_rejects_zero_ood_noise()
    test_predict_is_deterministic()
    test_predict_sde_returns_intervals()
    test_anomaly_mask_marks_target_and_input_history()
    test_train_normal_only_filters_windows_and_runs()
    test_train_normal_only_requires_mask()
    test_clc_primitive()
    test_frequency_weighted_bin_summary()
    test_reference_peak_bins_use_global_scale()
    test_figure_sample_uses_reference_peak()
    test_production_peak_nmpil_is_rowwise()
    print("PASS: neural-SDE ST-GNN tests")
