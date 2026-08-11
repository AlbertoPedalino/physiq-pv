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
from tempfile import TemporaryDirectory
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import xarray as xr

from physiq_pv.data.pvgis_dataset import (
    build_detector_event_protocol,
    build_datasets,
    build_regional_event_protocol,
    build_year_raw,
    make_model,
    normal_event_timestamp_mask,
    normal_event_window_mask,
)
from physiq_pv.experiments.pvgis_stgnn_runner import (
    build_arg_parser,
    run_from_args,
)
from physiq_pv.model.st_gnn import SDEBlock
from physiq_pv.model.sde_net import YearMSDSDENet, diffusion_bce_loss, yearmsd_nll_loss
from physiq_pv.model.graph_builder import build_graph
from physiq_pv.training.train_loop import train_model
from physiq_pv.training.uncertainty import (
    _gaussian_mixture_quantile,
    binary_ood_metrics,
    evaluate_pseudo_ood,
    predict,
    predict_sde,
)


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
        seq_len=24,
        horizon=1,
    )
    edge_index, edge_weight = build_graph(built["lats"], built["lons"], max_dist_km=20.0)
    return built, edge_index, edge_weight


def _model(built):
    torch.manual_seed(0)
    return make_model(n_nodes=2, seq_len=24, n_features=built["n_features"],
                      dropout=0.2, n_sde_steps=4, sigma_max=0.5)


def test_build_year_raw_uses_tilted_poa_fallback() -> None:
    ds = _tiny_year(2019)
    physical_poa = ds["solar_irradiance_poa"].copy()
    ds["direct_irradiance_tilted"] = physical_poa * 0.75
    ds["diffuse_irradiance_tilted"] = physical_poa * 0.25
    ds["diffuse_irradiance_tilted"][0, 0] = -0.01
    ds["solar_irradiance_poa"] = physical_poa * 4.0

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


def test_forecast_horizons_align_target_timestamp_and_value() -> None:
    test_year = _tiny_year(2019)
    raw_target = test_year["pv_power_output"].transpose("time", "location").values
    raw_times = pd.DatetimeIndex(test_year["time"].values)
    for horizon in (1, 6, 12):
        built = build_datasets(
            {2016: _tiny_year(2016), 2017: _tiny_year(2017)},
            test_year,
            seq_len=24,
            horizon=horizon,
        )
        test = built["test"]
        target_offset = 24 + horizon - 1
        assert len(test) == len(raw_times) - target_offset
        assert test.target_time_all[0] == raw_times[target_offset]
        assert test.target_time_all[-1] == raw_times[-1]
        np.testing.assert_allclose(test.y_true_all[0], raw_target[target_offset])


def test_direct_multihorizon_targets_model_and_prediction_rows() -> None:
    horizons = (1, 2, 3, 4, 5, 6)
    test_year = _tiny_year(2019)
    raw_target = test_year["pv_power_output"].transpose("time", "location").values
    raw_times = pd.DatetimeIndex(test_year["time"].values)
    built = build_datasets(
        {2016: _tiny_year(2016), 2017: _tiny_year(2017)},
        test_year,
        seq_len=24,
        horizon=1,
        forecast_horizons=horizons,
    )
    test = built["test"]
    assert len(test) == len(raw_times) - 24 - max(horizons) + 1
    assert test.y_norm_all.shape == (len(test), 2, 6)
    assert test.y_true_all.shape == (len(test), 2, 6)
    assert np.asarray(test.target_time_all).shape == (len(test), 6)
    np.testing.assert_allclose(
        test.y_true_all[0], raw_target[24:30].T
    )
    np.testing.assert_array_equal(
        np.asarray(test.target_time_all)[0], raw_times[24:30].to_numpy()
    )

    edge_index, edge_weight = build_graph(
        built["lats"], built["lons"], max_dist_km=20.0
    )
    model = make_model(
        n_nodes=2,
        seq_len=24,
        n_features=built["n_features"],
        dropout=0.0,
        forecast_horizons=horizons,
    ).eval()
    x, y, _ = next(iter(torch.utils.data.DataLoader(test, batch_size=3)))
    with torch.no_grad():
        pred_poa, mean, sigma = model(
            x, edge_index, edge_weight, None, stochastic=False
        )
    assert pred_poa.shape == y.shape == mean.shape == sigma.shape == (3, 2, 6)
    predictions = predict(
        model, test, edge_index, edge_weight, "cpu", batch_size=8
    )
    assert len(predictions) == len(test) * 2 * 6
    assert tuple(sorted(predictions["horizon_hours"].unique())) == horizons
    assert not predictions.duplicated(
        ["issue_timestamp", "location", "horizon_hours"]
    ).any()
    first = predictions.iloc[:12]
    assert first["location"].tolist() == ["loc_a"] * 6 + ["loc_b"] * 6
    assert first["horizon_hours"].tolist() == list(horizons) * 2


def test_graph_has_bounded_prior_self_loops_and_no_isolated_nodes() -> None:
    edge_index, edge_weight = build_graph(
        np.asarray([45.0, 46.0, 47.0]),
        np.asarray([7.0, 8.0, 9.0]),
        max_dist_km=1.0,
    )
    assert bool(((edge_index[0] == edge_index[1])).sum() == 3)
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
                "--max-validation-samples", "16",
                "--max-test-samples", "16",
                "--sde-sigma-warmup-epochs", "0",
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
    pred_poa, mean, sigma = out                   # Gaussian PV head (mean, sigma)
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
    real_clip = torch.nn.utils.clip_grad_norm_
    with patch(
        "physiq_pv.training.train_loop.torch.nn.utils.clip_grad_norm_",
        wraps=real_clip,
    ) as clip_mock:
        model = train_model(
            _model(built), built["train"], built["validation"], ei, ew,
            epochs=2, batch_size=8, lr=1e-3, device="cpu",
            ood_noise_std=0.1, feature_names=built["features"],
            sde_sigma_initial=0.01, sde_sigma_warmup_epochs=1,
            gradient_clip_norm=100.0, lr_decay_epoch=0, lr_decay_factor=0.1,
        )
    rec = model.train_loss_history[-1]
    assert {
        "loss/pv", "train/g_in", "train/g_ood", "train/g_ratio",
        "loss/diffusion", "loss/diffusion_in", "loss/diffusion_ood", "train/sigma",
        "train/lr_f", "train/lr_g",
    } <= set(rec)
    assert rec["train/sigma"] == 0.5
    assert clip_mock.call_count > 0
    assert all(call.args[1] == 100.0 for call in clip_mock.call_args_list)
    assert model.train_loss_history[0]["train/lr_f"] == 1e-3
    assert abs(model.train_loss_history[1]["train/lr_f"] - 1e-4) < 1e-12
    assert all(item["train/lr_g"] == 0.01 for item in model.train_loss_history)
    assert "loss/irradiance" in rec
    assert model.best_epoch == 0
    assert model.best_validation_metric == "rmse_daytime"
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


# --- 6. inference: deterministic predict + SDE-sampled intervals ------------ #
def test_gaussian_mixture_quantile_inverts_full_mixture_cdf() -> None:
    mu = np.array([[0.0], [4.0], [9.0]], dtype=np.float64)
    sigma = np.array([[0.5], [2.0], [1.0]], dtype=np.float64)
    probability = 0.025

    quantile = _gaussian_mixture_quantile(mu, sigma, probability)
    standardized = (quantile[None, :] - mu) / sigma
    cdf = (
        0.5
        * (1.0 + torch.erf(torch.from_numpy(standardized) / np.sqrt(2.0)))
    ).mean(dim=0).numpy()

    assert np.allclose(cdf, probability, atol=1e-12)
    moment_mean = mu.mean(axis=0)
    moment_std = np.sqrt(mu.var(axis=0) + (sigma ** 2).mean(axis=0))
    moment_lower = moment_mean - 1.959963984540054 * moment_std
    assert not np.allclose(quantile, moment_lower, atol=1e-3)

    identical_mu = np.full((4, 2), 3.0)
    identical_sigma = np.full((4, 2), 2.0)
    identical_q = _gaussian_mixture_quantile(
        identical_mu, identical_sigma, 0.975
    )
    expected = 3.0 + 1.959963984540054 * 2.0
    assert np.allclose(identical_q, expected, atol=1e-12)


def test_predict_is_deterministic() -> None:
    built, ei, ew = _built()
    model = _model(built)
    df1 = predict(model, built["test"], ei, ew, "cpu", batch_size=8)
    df2 = predict(model, built["test"], ei, ew, "cpu", batch_size=8)
    assert np.allclose(df1["y_pred"].to_numpy(), df2["y_pred"].to_numpy())


def test_binary_ood_metrics_perfect_separation() -> None:
    metrics = binary_ood_metrics(
        np.array([0.0, 0.1, 0.2]),
        np.array([0.8, 0.9, 1.0]),
    )
    assert metrics == {
        "auroc": 1.0,
        "aupr_out": 1.0,
        "aupr_in": 1.0,
        "tnr_at_tpr95": 1.0,
        "detection_accuracy": 1.0,
    }


def test_pseudo_ood_smoke_returns_paper_and_diagnostic_scores() -> None:
    built, ei, ew = _built()
    result = evaluate_pseudo_ood(
        _model(built), built["test"], ei, ew, "cpu",
        batch_size=8, mc_samples=2, ood_noise_std=2.0,
        max_samples=12, seed=1,
    )
    assert set(result["score"]) == {"epistemic_variance", "diffusion"}
    assert (result["n_id"] == 12).all()
    for column in (
        "id_mean", "ood_mean", "ood_to_id_ratio", "auroc", "aupr_out",
        "aupr_in", "tnr_at_tpr95", "detection_accuracy",
    ):
        assert np.isfinite(result[column]).all()


def test_predict_sde_returns_intervals() -> None:
    built, ei, ew = _built()
    model = _model(built)
    df = predict_sde(model, built["test"], ei, ew, "cpu", batch_size=8, mc_samples=8)
    assert {
        "y_pred_mean", "y_pred_std", "epistemic_std", "aleatoric_std",
        "lower_pi", "upper_pi", "lower_gaussian", "upper_gaussian",
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


def _event_scores(*years: int) -> pd.DataFrame:
    rows = []
    for year in years:
        times = pd.date_range(f"{year}-06-01", periods=72, freq="h")
        for i, position in enumerate((5, 15, 30, 60)):
            rows.append(
                {
                    "location": "loc_a",
                    "timestamp": times[position],
                    "variable": "wind_speed_10m",
                    "anomaly_score": float(i + 1 + (10 if year != 2016 else 0)),
                    "label": "extreme_wind_condition",
                }
            )
    return pd.DataFrame(rows)


def _detector_scores(*years: int) -> pd.DataFrame:
    rows = []
    for year in years:
        times = pd.date_range(f"{year}-06-01", periods=72, freq="h")
        for location in ("loc_a", "loc_b"):
            for position, timestamp in enumerate(times):
                is_anomaly = location == "loc_a" and position in (30, 60)
                rows.append(
                    {
                        "location": location,
                        "timestamp": timestamp,
                        "anomaly_score": 2.0 if is_anomaly else 0.1,
                        "threshold": 1.0,
                        "is_anomaly": is_anomaly,
                        "detector": "mtgflow",
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


def test_detector_event_protocol_fits_and_reuses_seasonal_thresholds() -> None:
    times = {2016: pd.date_range("2016-06-01", periods=72, freq="h")}
    protocol = build_detector_event_protocol(
        _detector_scores(2016),
        times,
        np.asarray(["loc_a", "loc_b"]),
        regional_quantile=0.975,
    )
    assert protocol["source"] == "detector"
    assert protocol["detector"] == "mtgflow"
    assert protocol["rare_by_year"][2016].sum() == 2
    assert 0.0 < protocol["seasonal_thresholds"]["JJA"] < 0.5
    assert protocol["thresholds_fitted_on_input"] is True
    assert protocol["coverage_by_year"][2016]["temporal"] == 1.0
    frozen = build_detector_event_protocol(
        _detector_scores(2019),
        {2019: pd.date_range("2019-06-01", periods=72, freq="h")},
        np.asarray(["loc_a", "loc_b"]),
        regional_quantile=0.975,
        seasonal_thresholds=protocol["seasonal_thresholds"],
    )
    assert frozen["seasonal_thresholds"] == protocol["seasonal_thresholds"]
    assert frozen["thresholds_fitted_on_input"] is False
    assert frozen["rare_by_year"][2019].sum() == 2
    all_normal = _detector_scores(2016)
    all_normal["is_anomaly"] = False
    zero_threshold = build_detector_event_protocol(
        all_normal,
        times,
        np.asarray(["loc_a", "loc_b"]),
        regional_quantile=0.975,
    )
    assert zero_threshold["seasonal_thresholds"]["JJA"] == 0.0
    assert zero_threshold["rare_by_year"][2016].sum() == 0


def test_detector_labels_work_without_normal_only_training() -> None:
    built = build_datasets(
        {2016: _tiny_year(2016), 2017: _tiny_year(2017)},
        _tiny_year(2019),
        seq_len=24,
        horizon=1,
        train_normal_only=False,
        train_anomaly_scores=_detector_scores(2016, 2017),
        test_anomaly_scores=_detector_scores(2019),
        anomaly_source="detector",
        detector_regional_quantile=0.975,
    )
    assert len(built["train"]) == 48
    assert built["event_protocol"]["thresholds_fitted_on_input"] is True
    assert built["test_event_protocol"]["source"] == "detector"
    assert (
        built["test_event_labels"]["event_group"] == "rare_or_extreme"
    ).sum() == 2


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
        train_model(
            _model(built), built["train"], built["validation"], ei, ew,
            epochs=1, batch_size=8,
            lr=1e-3, device="cpu", ood_noise_std=0.1,
            feature_names=built["features"], train_normal_only=True,
        )
        raise AssertionError("expected ValueError without event filtering")
    except ValueError:
        pass


# --- 8. CLC sharpness/reliability primitive --------------------------------- #
def test_clc_primitive() -> None:
    from physiq_pv.reporting.daytime_bin_anomaly_report import _clc
    assert abs(_clc(0.2, 0.95, 0.95, 9.0) - 0.4) < 1e-12


def test_prediction_csv_reader_uses_stable_dtypes() -> None:
    from unittest.mock import patch

    from physiq_pv.reporting.daytime_bin_anomaly_report import (
        _prediction_csv_reader,
    )

    columns = {
        "y_true": "y_true",
        "y_pred": "y_pred_mean",
        "y_std": "y_pred_std",
        "lower_pi": "lower_pi",
        "upper_pi": "upper_pi",
        "solar": "solar_irradiance_poa",
        "group": "event_group",
        "label": "anomaly_label",
        "timestamp": "timestamp",
        "location": "location",
    }
    sentinel = object()
    with patch(
        "physiq_pv.reporting.daytime_bin_anomaly_report.pd.read_csv",
        return_value=sentinel,
    ) as read_csv:
        result = _prediction_csv_reader("predictions.csv", columns, 1_000_000)

    assert result is sentinel
    options = read_csv.call_args.kwargs
    assert options["chunksize"] == 1_000_000
    assert options["low_memory"] is False
    assert options["dtype"] == {
        "event_group": "string",
        "anomaly_label": "string",
        "timestamp": "string",
        "location": "string",
    }


def test_load_daytime_marks_specific_labels_rare() -> None:
    from physiq_pv.reporting.daytime_bin_anomaly_report import (
        build_overview,
        load_daytime,
        resolve_columns,
    )

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "predictions.csv"
        pd.DataFrame({
            "timestamp": ["2019-06-01 12:00", "2019-06-01 13:00"],
            "location": ["loc_a", "loc_a"],
            "y_true": [50.0, 60.0],
            "y_pred_mean": [51.0, 59.0],
            "y_pred_std": [2.0, 2.0],
            "lower_pi": [45.0, 55.0],
            "upper_pi": [55.0, 65.0],
            "solar_irradiance_poa": [400.0, 500.0],
            "anomaly_group": ["normal", "rare_extreme"],
            "anomaly_label": ["normal", "extreme_wind_condition"],
        }).to_csv(path, index=False)

        day, stats = load_daytime(str(path), resolve_columns(str(path)), 10.0, 100)
        overview = build_overview(day, stats).iloc[0]

    assert int(day["is_rare"].sum()) == 1
    assert int(overview["rare_extreme_daytime_samples"]) == 1
    assert int(overview["extreme_wind_condition_count"]) == 1


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


def test_figure_categories_use_pointwise_anomaly_group() -> None:
    from physiq_pv.reporting.posthoc_outputs import _figure_category_masks

    sample = pd.DataFrame({
        "event_group": ["normal", "rare_or_extreme"],
        # Deliberately contradictory regional labels: figures must use the
        # detector decision for the exact location and timestamp.
        "anomaly_group": ["rare_or_extreme", "normal"],
        "anomaly_label": ["mtgflow", "normal"],
    })
    categories = _figure_category_masks(sample)

    assert [name for name, _ in categories] == ["normal", "rare_extreme"]
    assert categories[0][1].tolist() == [False, True]
    assert categories[1][1].tolist() == [True, False]


def test_figure_categories_reject_regional_group_only() -> None:
    from physiq_pv.reporting.posthoc_outputs import _figure_category_masks

    sample = pd.DataFrame({"event_group": ["normal", "rare_or_extreme"]})
    try:
        _figure_category_masks(sample)
    except ValueError as exc:
        assert "require anomaly_group" in str(exc)
    else:
        raise AssertionError("Regional event_group must not drive post-hoc figures.")


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


def test_full_data_boxplot_statistics_and_uncertainty_components() -> None:
    from physiq_pv.reporting.daytime_bin_anomaly_report import (
        _boxplot_stats,
        build_uncertainty_components,
    )

    stats = _boxplot_stats(np.array([1.0, 2.0, 3.0, 100.0]), "x")
    assert stats["x_mean"] == 26.5
    assert stats["x_median"] == 2.5
    assert stats["x_whisker_low"] == 1.0
    assert stats["x_whisker_high"] == 3.0

    day = pd.DataFrame({
        "is_normal": [True, True, False, False],
        "is_rare": [False, False, True, True],
        "epistemic_std": [1.0, 3.0, 2.0, 4.0],
        "aleatoric_std": [4.0, 6.0, 8.0, 10.0],
        "y_std": [5.0, 7.0, 9.0, 11.0],
    })
    components = build_uncertainty_components(day).set_index("scope")
    assert components.loc["normal", "count"] == 2
    assert components.loc["normal", "mean_epistemic_std"] == 2.0
    assert components.loc["rare_extreme", "mean_epistemic_std"] == 3.0
    assert components.loc["rare_extreme", "mean_aleatoric_std"] == 9.0


def test_posthoc_figures_require_full_data_summaries_not_predictions(
    tmp_path: Path,
) -> None:
    from physiq_pv.reporting.posthoc_outputs import build_posthoc_figures

    rows = []
    for category, count, scale in (
        ("normal", 100, 1.0),
        ("rare_extreme", 20, 2.0),
    ):
        rows.append({
            "bin": "daytime_0_20_pct",
            "category": category,
            "count": count,
            "picp": 0.95,
            "rmse": 2.0 * scale,
            "nmpil": 0.1 * scale,
            "abs_error_mean": 1.5 * scale,
            "abs_error_q1": 1.0 * scale,
            "abs_error_median": 1.4 * scale,
            "abs_error_q3": 2.0 * scale,
            "abs_error_whisker_low": 0.2 * scale,
            "abs_error_whisker_high": 3.0 * scale,
            "row_nmpil_mean": 0.1 * scale,
            "row_nmpil_q1": 0.05 * scale,
            "row_nmpil_median": 0.09 * scale,
            "row_nmpil_q3": 0.14 * scale,
            "row_nmpil_whisker_low": 0.01 * scale,
            "row_nmpil_whisker_high": 0.20 * scale,
        })
    pd.DataFrame(rows).to_csv(
        tmp_path / "daytime_bin_anomaly_metrics.csv", index=False
    )

    # No predictions.csv is present: figures must come from the exact summary.
    paths = build_posthoc_figures(str(tmp_path), max_plot_rows=1)
    assert "mae_daytime_0_20_pct_boxplot" in paths
    assert "picp_daytime_0_20_pct_bar" in paths
    assert all(path.is_file() for path in paths.values())


def test_extreme_event_diagnostic_uses_every_node(tmp_path: Path) -> None:
    from physiq_pv.reporting.posthoc_outputs import (
        build_extreme_event_diagnostic,
    )

    pd.DataFrame({
        "timestamp": [
            "2019-06-28 00:10:00", "2019-06-28 00:10:00",
            "2019-06-28 01:10:00", "2019-06-28 01:10:00",
        ],
        "location": [0, 1, 0, 1],
        "y_true": [0.0, 0.0, 10.0, 20.0],
        "y_pred_mean": [0.0, 1.0, 8.0, 18.0],
        "lower_pi": [0.0, 0.0, 5.0, 15.0],
        "upper_pi": [2.0, 2.0, 11.0, 21.0],
        "y_pred_std_raw": [1.0, 1.0, 2.0, 2.0],
        "epistemic_std": [0.2, 0.2, 0.5, 0.5],
        "aleatoric_std": [0.9, 0.9, 1.8, 1.8],
        "solar_irradiance_poa_target": [0.0, 0.0, 100.0, 100.0],
        "event_group": [
            "normal", "normal", "rare_or_extreme", "rare_or_extreme",
        ],
        "event_score": [0.2, 0.2, 0.8, 0.8],
    }).to_csv(tmp_path / "predictions.csv", index=False)

    result = build_extreme_event_diagnostic(
        str(tmp_path),
        start="2019-06-28",
        end="2019-06-29",
        regional_threshold=0.35,
        figure_subdir="june_extreme_event",
        chunksize=1,
    )
    assert result["rows_used"] == 4
    assert len(result["hourly"]) == 2
    overall = result["summary"].set_index("scope").loc["all_event_hours"]
    assert overall["locations"] == 2
    assert overall["rare_timestamp_count"] == 1
    np.testing.assert_allclose(
        result["hourly"]["regional_anomaly_fraction"],
        [0.07, 0.28],
    )
    assert result["figure_path"].is_file()
    assert result["figure_path"].parent.name == "june_extreme_event"
    assert result["hourly_path"].is_file()


def test_extreme_event_comparison_uses_all_rows(tmp_path: Path) -> None:
    from physiq_pv.reporting.posthoc_outputs import (
        build_extreme_event_comparison_figures,
    )

    pd.DataFrame({
        "timestamp": [
            "2019-04-20 12:10:00", "2019-04-20 12:10:00",
            "2019-04-23 12:10:00", "2019-04-23 12:10:00",
            "2019-04-24 12:10:00", "2019-04-24 12:10:00",
            "2019-04-25 12:10:00", "2019-04-25 12:10:00",
            "2019-04-26 12:10:00", "2019-04-26 12:10:00",
        ],
        "y_true": [10.0] * 10,
        "y_pred_mean": [
            9.0, 11.0, 8.0, 12.0, 7.0,
            13.0, 6.0, 14.0, 5.0, 15.0,
        ],
        "lower_pi": [0.0] * 10,
        "upper_pi": [20.0] * 10,
        "solar_irradiance_poa_target": [100.0] * 10,
        "anomaly_group": [
            "normal", "normal",
            "rare_or_extreme", "rare_or_extreme",
            "rare_or_extreme", "rare_or_extreme",
            "rare_or_extreme", "rare_or_extreme",
            "rare_or_extreme", "rare_or_extreme",
        ],
    }).to_csv(tmp_path / "predictions.csv", index=False)
    pd.DataFrame({"reference_peak_w": [100.0]}).to_csv(
        tmp_path / "reference_production_peaks.csv", index=False
    )
    pd.DataFrame({"target_range": [100.0]}).to_csv(
        tmp_path / "sharpness_overview.csv", index=False
    )

    result = build_extreme_event_comparison_figures(
        str(tmp_path),
        event_dates=(
            "2019-04-23",
            "2019-04-24",
            "2019-04-25",
            "2019-04-26",
        ),
        comparison_name="april_dust",
        figure_subdir="april_dust_event",
        chunksize=1,
    )
    metrics = result["metrics"].set_index("category")
    assert set(metrics.index) == {
        "normal_2019",
        "2019-04-23",
        "2019-04-24",
        "2019-04-25",
        "2019-04-26",
    }
    assert (metrics["count"] == 2).all()
    assert metrics.loc["normal_2019", "mae"] == 1.0
    assert metrics.loc["2019-04-23", "mae"] == 2.0
    assert metrics.loc["2019-04-24", "mae"] == 3.0
    assert metrics.loc["2019-04-25", "mae"] == 4.0
    assert metrics.loc["2019-04-26", "mae"] == 5.0
    assert result["comparison_name"] == "april_dust"
    assert result["metrics_path"].name == (
        "extreme_event_comparison_april_dust_metrics.csv"
    )
    assert result["metrics_path"].is_file()
    assert len(result["figure_paths"]) == 4
    assert all(path.is_file() for path in result["figure_paths"].values())
    assert all(
        path.parent.name == "april_dust_event"
        for path in result["figure_paths"].values()
    )


if __name__ == "__main__":
    test_sdeblock_shape_and_diffusion_bounds()
    test_build_year_raw_uses_tilted_poa_fallback()
    test_time_grid_rejects_missing_hour()
    test_location_order_is_reindexed_and_default_target_is_not_upper_clipped()
    test_forecast_horizons_align_target_timestamp_and_value()
    test_graph_has_bounded_prior_self_loops_and_no_isolated_nodes()
    test_runner_saves_reproducible_best_checkpoint()
    test_forward_deterministic_vs_stochastic()
    test_diffusion_learns_ood_separation()
    test_yearmsd_reference_model_matches_paper_interface()
    test_train_model_runs_and_logs_g()
    test_train_model_rejects_zero_ood_noise()
    test_gaussian_mixture_quantile_inverts_full_mixture_cdf()
    test_predict_is_deterministic()
    test_predict_sde_returns_intervals()
    test_anomaly_mask_marks_target_and_input_history()
    test_regional_event_protocol_fits_training_only_threshold()
    test_detector_event_protocol_fits_and_reuses_seasonal_thresholds()
    test_detector_labels_work_without_normal_only_training()
    test_normal_event_masks_remove_complete_windows()
    test_train_normal_only_physically_filters_train_and_validation()
    test_train_normal_only_requires_event_filtered_dataset()
    test_clc_primitive()
    test_prediction_csv_reader_uses_stable_dtypes()
    test_load_daytime_marks_specific_labels_rare()
    test_frequency_weighted_bin_summary()
    test_reference_peak_bins_use_global_scale()
    test_figure_sample_uses_reference_peak()
    test_figure_categories_use_pointwise_anomaly_group()
    test_figure_categories_reject_regional_group_only()
    test_production_peak_nmpil_is_rowwise()
    test_full_data_boxplot_statistics_and_uncertainty_components()
    with TemporaryDirectory() as d:
        test_posthoc_figures_require_full_data_summaries_not_predictions(Path(d))
    with TemporaryDirectory() as d:
        test_extreme_event_diagnostic_uses_every_node(Path(d))
    with TemporaryDirectory() as d:
        test_extreme_event_comparison_uses_all_rows(Path(d))
    print("PASS: neural-SDE ST-GNN tests")
