"""Synthetic/CPU tests for the paper-faithful SDE-Net PV adaptation.

No PVGIS data needed: a tiny synthetic year drives build_datasets + train_model.
Covers:
  1. SDEBlock output shape and scalar per-example diffusion scale;
  2. forward is deterministic with stochastic=False, varies with stochastic=True;
  3. the diffusion net learns to separate in-distribution from Gaussian OOD;
  4. the direct YearMSD reference architecture has the paper interface;
  5. train_model runs, stays finite, and logs diffusion BCE diagnostics;
  6. predict is deterministic; predict_sde returns two-source (epistemic +
     aleatoric) Gaussian intervals.
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
from physiq_pv.model.sde_net import YearMSDSDENet, diffusion_bce_loss, yearmsd_nll_loss
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


# --- 1. SDEBlock shape + paper scalar diffusion ----------------------------- #
def test_sdeblock_shape_and_diffusion_bounds() -> None:
    torch.manual_seed(0)
    sde = SDEBlock(dim=16, n_steps=4, sigma_max=0.5)
    x0 = torch.randn(3, 5, 16)
    with torch.no_grad():
        xT, g = sde(x0, stochastic=True)
    assert xT.shape == x0.shape          # (B, N, dim)
    assert g.shape == (3, 1)             # one diffusion scale per graph example
    assert float(g.min()) >= 0.0 and float(g.max()) <= 0.5 + 1e-6


# --- 2. deterministic vs stochastic forward --------------------------------- #
def test_forward_deterministic_vs_stochastic() -> None:
    built, ei, ew = _built()
    model = _model(built).eval()
    x, y, _ = next(iter(torch.utils.data.DataLoader(built["train"], batch_size=4)))
    out = model(x, ei, ew, None, stochastic=False)
    pred_ghi, mean, sigma = out                   # Gaussian PV head (mean, sigma)
    assert (sigma > 0).all()                      # aleatoric std strictly positive
    d1 = out[1]
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
    x_in = torch.zeros(32, 16)
    x_ood = torch.full((32, 16), 2.0)
    for _ in range(200):
        g_in = sde.diffusion(x_in)
        g_ood = sde.diffusion(x_ood)
        loss_g, _, _ = diffusion_bce_loss(g_in, g_ood)
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
    assert g_ood.mean().item() > g_in.mean().item()  # high diffusion on OOD


# --- 4. direct YearMSD reference architecture -------------------------------- #
def test_yearmsd_reference_model_matches_paper_interface() -> None:
    torch.manual_seed(0)
    model = YearMSDSDENet()
    x = torch.randn(7, 90)
    mean, sigma = model(x)
    g = model(x, training_diffusion=True)
    assert mean.shape == (7,)
    assert sigma.shape == (7,)
    assert g.shape == (7, 1)
    assert torch.all(sigma >= 1e-3)
    assert torch.all((g >= 0.0) & (g <= 1.0))
    assert torch.isfinite(yearmsd_nll_loss(torch.randn(7), mean, sigma))
    # Table 2 of the v1 paper reports the 90 -> 50 YearMSD SDE-Net as 12.4K.
    assert sum(p.numel() for p in model.parameters()) == 12_403


# --- 5. train_model runs and logs the SDE diagnostics ----------------------- #
def test_train_model_runs_and_logs_g() -> None:
    built, ei, ew = _built()
    model = train_model(_model(built), built["train"], ei, ew,
                        epochs=2, batch_size=8, lr=1e-3, device="cpu",
                        ood_noise_std=0.1, feature_names=built["features"],
                        sde_sigma_initial=0.01, sde_sigma_warmup_epochs=1)
    rec = model.train_loss_history[-1]
    assert {
        "loss/pv", "train/g_in", "train/g_ood", "train/g_ratio",
        "loss/diffusion", "loss/diffusion_in", "loss/diffusion_ood", "train/sigma",
    } <= set(rec)
    assert rec["train/sigma"] == 0.5
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


# --- 6. inference: deterministic predict + SDE-sampled intervals ------------ #
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
    assert {
        "y_pred_mean", "y_pred_std", "epistemic_std", "aleatoric_std",
        "lower_pi", "upper_pi",
    } <= set(df.columns)
    assert (df["upper_pi"] >= df["lower_pi"]).all()
    assert df["y_pred_std"].to_numpy().std() > 0.0  # non-degenerate uncertainty
    assert (df["aleatoric_std"].to_numpy() > 0.0).all()  # heteroscedastic head
    # Total predictive variance is epistemic + aleatoric (law of total variance).
    total = df["y_pred_std"].to_numpy() ** 2
    parts = df["epistemic_std"].to_numpy() ** 2 + df["aleatoric_std"].to_numpy() ** 2
    assert np.allclose(total, parts, rtol=1e-6, atol=1e-9)


# --- 7. train-normal-only --------------------------------------------------- #
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


# --- 8. CLC sharpness/reliability primitive --------------------------------- #
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


def test_daily_peak_production_bins_are_percentages() -> None:
    from physiq_pv.reporting.daytime_bin_anomaly_report import (
        add_daily_peak_production_pct,
    )

    day = pd.DataFrame({
        "timestamp": [
            "2019-06-01T08:00:00Z", "2019-06-01T12:00:00Z",
            "2019-06-01T08:00:00Z", "2019-06-01T12:00:00Z",
        ],
        "location": ["a", "a", "b", "b"],
        "y_true": [20.0, 100.0, 30.0, 60.0],
    })
    stats = add_daily_peak_production_pct(day, "Europe/Rome")

    np.testing.assert_allclose(day["production_pct"], [20.0, 100.0, 50.0, 100.0])
    assert stats["production_curve_count"] == 2


def test_figure_sample_uses_report_daily_peaks() -> None:
    from physiq_pv.reporting.posthoc_outputs import (
        attach_daily_peak_production_pct,
    )

    sample = pd.DataFrame({
        "timestamp": ["2019-06-01T08:00:00Z", "2019-06-01T12:00:00Z"],
        "location": ["a", "a"],
        "y_true": [20.0, 80.0],
    })
    peaks = pd.DataFrame({
        "location": ["a"],
        "production_curve_date": ["2019-06-01"],
        "daily_peak_w": [100.0],
    })
    result = attach_daily_peak_production_pct(sample, peaks, "Europe/Rome")

    np.testing.assert_allclose(result["production_pct"], [20.0, 80.0])


def test_daily_peak_nmpil_normalizes_each_interval_before_averaging() -> None:
    from physiq_pv.reporting.daytime_bin_anomaly_report import subset_metrics

    sample = pd.DataFrame({
        "y_true": [50.0, 100.0],
        "y_pred": [50.0, 100.0],
        "y_std": [1.0, 1.0],
        "lower_pi": [40.0, 80.0],
        "upper_pi": [60.0, 120.0],
        "daily_peak_w": [100.0, 200.0],
    })
    metrics = subset_metrics(sample)

    assert abs(metrics["mpiw"] - 30.0) < 1e-12
    assert abs(metrics["daily_peak_nmpil"] - 0.2) < 1e-12


if __name__ == "__main__":
    test_sdeblock_shape_and_diffusion_bounds()
    test_forward_deterministic_vs_stochastic()
    test_diffusion_learns_ood_separation()
    test_yearmsd_reference_model_matches_paper_interface()
    test_train_model_runs_and_logs_g()
    test_train_model_rejects_zero_ood_noise()
    test_predict_is_deterministic()
    test_predict_sde_returns_intervals()
    test_anomaly_mask_marks_target_and_input_history()
    test_train_normal_only_runs_with_mask()
    test_train_normal_only_requires_mask()
    test_clc_primitive()
    test_frequency_weighted_bin_summary()
    test_daily_peak_production_bins_are_percentages()
    test_figure_sample_uses_report_daily_peaks()
    test_daily_peak_nmpil_normalizes_each_interval_before_averaging()
    print("PASS: neural-SDE ST-GNN tests")
