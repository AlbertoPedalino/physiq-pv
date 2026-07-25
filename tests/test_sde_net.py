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
from tempfile import TemporaryDirectory

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import xarray as xr

from physiq_pv.data.pvgis_dataset import (
    build_datasets,
    build_regional_event_protocol,
    build_year_raw,
    make_model,
    normal_event_timestamp_mask,
    normal_event_window_mask,
)
from physiq_pv.model.st_gnn import MonacoDiffusionEncoder
from physiq_pv.model.graph_builder import build_graph
from physiq_pv.experiments.pvgis_stgnn_runner import build_arg_parser, run_from_args
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
            "solar_irradiance_poa": (("location", "time"), np.zeros_like(solar)),
            "direct_irradiance_tilted": (("location", "time"), solar * 0.75),
            "diffuse_irradiance_tilted": (("location", "time"), solar * 0.25),
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
        {2016: _tiny_year(2016), 2017: _tiny_year(2017)},
        _tiny_year(2019),
                           seq_len=24, horizon=1)
    edge_index, edge_weight = build_graph(built["lats"], built["lons"], max_dist_km=20.0)
    return built, edge_index, edge_weight


def _model(built):
    torch.manual_seed(0)
    return make_model(n_nodes=2, seq_len=24, n_features=built["n_features"],
                      dropout=0.2, n_sde_steps=2, sigma_max=0.5)


# --- 1. Monaco stage alignment + bounded diffusion -------------------------- #


def test_zero_dropout_is_respected() -> None:
    built, _, _ = _built()
    model = make_model(
        n_nodes=2,
        seq_len=24,
        n_features=built["n_features"],
        dropout=0.0,
        n_sde_steps=2,
        sigma_max=0.5,
    )

    assert model.encoder.lstm.dropout == 0.0
    assert all(layer.dropout.p == 0.0 for layer in model.gat)


def test_build_year_raw_uses_tilted_poa_fallback() -> None:
    ds = _tiny_year(2019)
    physical_poa = (
        ds["direct_irradiance_tilted"] + ds["diffuse_irradiance_tilted"]
    )
    ds["solar_irradiance_poa"] = physical_poa * 2.0

    raw = build_year_raw(ds, "pv_power_output")

    np.testing.assert_allclose(raw["solar_wm2"], physical_poa.transpose("time", "location").values)
    assert raw["day"].any()


def test_time_grid_rejects_missing_hour() -> None:
    ds = _tiny_year(2019).isel(time=[i for i in range(72) if i != 20])
    try:
        build_year_raw(ds, "pv_power_output")
        raise AssertionError("expected a non-hourly-grid ValueError")
    except ValueError as exc:
        assert "non-hourly step" in str(exc)


def test_location_order_is_reindexed_and_default_target_is_not_upper_clipped() -> None:
    validation = _tiny_year(2017).isel(location=[1, 0])
    test = _tiny_year(2019).isel(location=[1, 0]).copy()
    test["pv_power_output"][0, 30] = 2_000.0
    built = build_datasets(
        {2016: _tiny_year(2016), 2017: validation},
        test,
        seq_len=24,
        horizon=1,
    )
    assert list(built["loc_ids"]) == ["loc_a", "loc_b"]
    assert built["pv_target_clip_max"] is None
    assert float(built["test"].y_norm_all.max()) > 1.5


def test_graph_has_bounded_prior_self_loops_and_no_isolated_nodes() -> None:
    edge_index, edge_weight = build_graph(
        np.asarray([45.0, 46.0, 47.0]),
        np.asarray([7.0, 8.0, 9.0]),
        max_dist_km=1.0,
    )
    assert int((edge_index[0] == edge_index[1]).sum()) == 3
    assert float(edge_weight.min()) > 0.0
    assert float(edge_weight.max()) <= 1.0
    assert int(torch.bincount(edge_index[1], minlength=3).min()) >= 2


def test_runner_saves_reproducible_best_checkpoint() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for year in (2016, 2017, 2019):
            _tiny_year(year).to_netcdf(root / f"piedmont_pvgis_{year}.nc")
        out_dir = root / "out"
        args = build_arg_parser().parse_args(
            [
                "--pvgis-dir", str(root),
                "--train-years", "2016,2017",
                "--test-year", "2019",
                "--out-dir", str(out_dir),
                "--epochs", "1",
                "--batch-size", "8",
                "--max-train-samples", "16",
                "--max-test-samples", "16",
                "--skip-predictions-csv",
                "--device", "cpu",
            ]
        )
        paths = run_from_args(args)
        checkpoint_path = Path(paths["checkpoint"])
        assert checkpoint_path.exists()
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        assert checkpoint["best_epoch"] == 0
        assert checkpoint["validation_year"] == 2017
        assert checkpoint["train_years"] == [2016]
        assert checkpoint["feature_names"]
        assert checkpoint["location_ids"] == ["loc_a", "loc_b"]
        assert checkpoint["pv_target_clip_max"] is None
        assert checkpoint["normalization"]["pv_scale"].shape == (2,)
        assert checkpoint["graph"]["edge_index"].shape[0] == 2
        assert checkpoint["model_state_dict"]


def test_runner_normal_only_filters_events_and_keeps_test_complete() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for year in (2016, 2017, 2019):
            _tiny_year(year).to_netcdf(root / f"piedmont_pvgis_{year}.nc")
        train_scores_path = root / "train_scores.csv"
        test_scores_path = root / "test_scores.csv"
        _event_scores(2016, 2017).to_csv(train_scores_path, index=False)
        _event_scores(2019).to_csv(test_scores_path, index=False)
        out_dir = root / "normal_only_out"
        args = build_arg_parser().parse_args(
            [
                "--pvgis-dir", str(root),
                "--train-years", "2016,2017",
                "--test-year", "2019",
                "--out-dir", str(out_dir),
                "--epochs", "1",
                "--batch-size", "8",
                "--device", "cpu",
                "--train-normal-only",
                "--train-anomaly-scores", str(train_scores_path),
                "--anomaly-scores", str(test_scores_path),
                "--event-spatial-quantile", "0.75",
                "--event-tail-quantile", "0.95",
            ]
        )
        paths = run_from_args(args)
        predictions = pd.read_csv(paths["predictions"])
        checkpoint = torch.load(
            paths["checkpoint"], map_location="cpu", weights_only=False
        )

        assert len(predictions) == 48 * 2
        assert {"event_group", "event_score", "event_driver"} <= set(
            predictions.columns
        )
        assert (
            predictions["event_group"] == "rare_or_extreme"
        ).sum() == 2 * 2  # only rare timestamps reachable as +1 h targets
        protocol = checkpoint["event_protocol"]
        assert protocol is not None
        assert 0 < protocol["filter_stats"]["train"]["after"] < 48
        assert 0 < protocol["filter_stats"]["validation"]["after"] < 48
        assert protocol["thresholds"]["wind_speed_10m"] < 4.0


def test_monaco_diffusion_stage_shapes_and_bounds() -> None:
    built, ei, ew = _built()
    model = _model(built).eval()
    x, _, _ = next(iter(torch.utils.data.DataLoader(built["train"], batch_size=3)))
    with torch.no_grad():
        terms = model.diffusion(x, ei, ew)
        _, pred, returned_terms = model(
            x, ei, ew, None, stochastic=True, return_diffusion=True
        )
    assert len(model.gat) == len(model.diffusion_encoder.gat) == 1
    assert len(terms) == model.n_sde_stages == 2  # BiLSTM + one GAT stage
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
        edge_prior_strength=1.0,
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
    model = train_model(_model(built), built["train"], built["validation"], ei, ew,
                        epochs=2, batch_size=8, lr=1e-3, device="cpu",
                        ood_noise_std=0.1, feature_names=built["features"])
    rec = model.train_loss_history[-1]
    assert {"loss/pv", "train/g_in", "train/g_ood", "train/g_ratio"} <= set(rec)
    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_train_model_rejects_zero_ood_noise() -> None:
    built, ei, ew = _built()
    try:
        train_model(_model(built), built["train"], built["validation"], ei, ew,
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
    np.testing.assert_allclose(df["y_pred_lower"], df["lower_pi"])
    np.testing.assert_allclose(df["y_pred_upper"], df["upper_pi"])
    assert df["y_pred_std"].to_numpy().std() > 0.0  # non-degenerate uncertainty


# --- 6. paper-style graph-wide normal-event filtering ---------------------- #
def _event_scores(*years: int) -> pd.DataFrame:
    rows = []
    for year in years:
        times = pd.date_range(f"{year}-06-01", periods=72, freq="h")
        for i, pos in enumerate((5, 15, 30, 60)):
            rows.append(
                {
                    "location": "loc_a",
                    "timestamp": times[pos],
                    "variable": "wind_speed_10m",
                    "anomaly_score": float(i + 1 + (10 if year != 2016 else 0)),
                    "label": "extreme_wind_condition",
                }
            )
    return pd.DataFrame(rows)


def test_regional_event_protocol_fits_training_only_threshold() -> None:
    scores = _event_scores(2016, 2017)
    times = {
        2016: pd.date_range("2016-06-01", periods=72, freq="h"),
        2017: pd.date_range("2017-06-01", periods=72, freq="h"),
    }
    protocol = build_regional_event_protocol(
        scores,
        times,
        np.asarray(["loc_a", "loc_b"]),
        fit_years=[2016],
        spatial_quantile=0.75,
        event_quantile=0.95,
    )
    threshold = protocol["thresholds"]["wind_speed_10m"]
    assert 0.0 < threshold < 4.0
    assert protocol["rare_by_year"][2016].sum() > 0
    assert protocol["rare_by_year"][2017].sum() == 4


def test_normal_event_masks_remove_complete_windows() -> None:
    rare = np.zeros(72, dtype=bool)
    rare[30] = True
    keep = normal_event_window_mask(rare, seq_len=24, horizon=1)
    used = normal_event_timestamp_mask(rare, seq_len=24, horizon=1)
    assert keep.shape == (48,)
    assert not keep[6:31].any()
    assert not used[30]
    assert keep.any()


def test_train_normal_only_physically_filters_train_and_validation() -> None:
    scores = _event_scores(2016, 2017)
    train_year = _tiny_year(2016)
    train_year["pv_power_output"].loc[
        {"location": "loc_a", "time": train_year["time"].values[60]}
    ] = 5_000.0
    built = build_datasets(
        {2016: train_year, 2017: _tiny_year(2017)},
        _tiny_year(2019),
        seq_len=24,
        horizon=1,
        train_normal_only=True,
        train_anomaly_scores=scores,
        test_anomaly_scores=_event_scores(2019),
        event_spatial_quantile=0.75,
        event_tail_quantile=0.95,
    )
    train_stats = built["event_filter_stats"]["train"]
    val_stats = built["event_filter_stats"]["validation"]
    assert 0 < train_stats["after"] < train_stats["before"]
    assert 0 < val_stats["after"] < val_stats["before"]
    assert built["train"].event_filter_applied
    assert built["validation"].event_filter_applied
    assert not built["train"].event_rare_target_all.any()
    assert not built["train"].event_rare_history_all.any()
    assert not built["validation"].event_rare_target_all.any()
    assert not built["validation"].event_rare_history_all.any()
    assert len(built["test"]) == 48
    assert built["test_event_labels"] is not None
    assert (
        built["test_event_labels"]["event_group"] == "rare_or_extreme"
    ).sum() == 4
    assert float(built["normalization"]["pv_scale"][0]) < 1_000.0

    edge_index, edge_weight = build_graph(
        built["lats"], built["lons"], max_dist_km=20.0
    )
    model = train_model(
        _model(built),
        built["train"],
        built["validation"],
        edge_index,
        edge_weight,
        epochs=1,
        batch_size=8,
        lr=1e-3,
        device="cpu",
        ood_noise_std=0.1,
        feature_names=built["features"],
        train_normal_only=True,
    )
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()


def test_train_normal_only_requires_event_filtered_dataset() -> None:
    built, ei, ew = _built()
    try:
        train_model(_model(built), built["train"], built["validation"], ei, ew,
                    epochs=1, batch_size=8,
                    lr=1e-3, device="cpu", ood_noise_std=0.1,
                    feature_names=built["features"], train_normal_only=True)
        raise AssertionError("expected ValueError without event filtering")
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
    test_zero_dropout_is_respected()
    test_monaco_diffusion_stage_shapes_and_bounds()
    test_build_year_raw_uses_tilted_poa_fallback()
    test_time_grid_rejects_missing_hour()
    test_location_order_is_reindexed_and_default_target_is_not_upper_clipped()
    test_graph_has_bounded_prior_self_loops_and_no_isolated_nodes()
    test_runner_saves_reproducible_best_checkpoint()
    test_runner_normal_only_filters_events_and_keeps_test_complete()
    test_forward_deterministic_vs_stochastic()
    test_diffusion_learns_ood_separation()
    test_train_model_runs_and_logs_g()
    test_train_model_rejects_zero_ood_noise()
    test_predict_is_deterministic()
    test_predict_sde_returns_intervals()
    test_regional_event_protocol_fits_training_only_threshold()
    test_normal_event_masks_remove_complete_windows()
    test_train_normal_only_physically_filters_train_and_validation()
    test_train_normal_only_requires_event_filtered_dataset()
    test_clc_primitive()
    test_frequency_weighted_bin_summary()
    test_reference_peak_bins_use_global_scale()
    test_figure_sample_uses_reference_peak()
    test_production_peak_nmpil_is_rowwise()
    print("PASS: neural-SDE ST-GNN tests")
