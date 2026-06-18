"""Synthetic/CPU tests for the Huber + noise + under-dispersion-penalty step.

No PVGIS data needed: a tiny synthetic year drives build_datasets + train_model.
Covers:
  1. new flags off -> training stays backward compatible (no penalty keys);
  2. penalty terms appear ONLY when --train-mc-uncertainty-penalty is on;
  3. train_mc_samples > 1 yields MC mean/std with shape (B, N);
  4. penalty is HIGH for high error + low std;
  5. penalty is LOW for high error + high std;
  6. input noise is NOT applied in eval/inference (predict is deterministic);
  7. input noise leaves the target untouched and never perturbs sin_elev/cos_elev.
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

from physiq_pv.data.pvgis_dataset import (
    PVGIS_STGNN_FEATURES,
    build_datasets,
    make_model,
)
from physiq_pv.training.losses import (
    sde_proxy_penalty,
    under_dispersion_penalty,
)
from physiq_pv.training.noise import (
    build_noise_feature_indices,
    inject_input_noise,
    inject_input_noise_anomaly,
)
from physiq_pv.training.train_loop import train_model
from physiq_pv.training.uncertainty import predict
from physiq_pv.experiments.pvgis_stgnn_runner import build_arg_parser, _validate
from physiq_pv.model.graph_builder import build_graph


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
    return make_model(n_nodes=2, seq_len=24, n_features=built["n_features"], dropout=0.2)


# --- 1. defaults off keep the baseline behaviour ---------------------------- #
def test_defaults_off_are_baseline() -> None:
    sig = inspect.signature(train_model)
    assert sig.parameters["train_mc_uncertainty_penalty"].default is False
    assert sig.parameters["train_mc_samples"].default == 1
    assert sig.parameters["uncertainty_penalty_weight"].default == 0.0
    assert sig.parameters["uncertainty_penalty_k"].default == 1.0
    assert sig.parameters["uncertainty_std_reg_weight"].default == 0.0
    assert sig.parameters["train_noise_std"].default == 0.0
    assert sig.parameters["train_noise_prob"].default == 0.0

    built, ei, ew = _built()
    model = train_model(_model(built), built["train"], ei, ew,
                        epochs=1, batch_size=8, lr=1e-3, device="cpu")
    rec = model.train_loss_history[0]
    assert "loss/under_penalty" not in rec  # penalty path not taken
    assert "loss/std_reg" not in rec
    for p in model.parameters():
        assert torch.isfinite(p).all()


# --- 2 + 3. penalty keys only when enabled; MC mean/std shape --------------- #
def test_penalty_enabled_adds_keys_and_trains() -> None:
    built, ei, ew = _built()
    model = train_model(
        _model(built), built["train"], ei, ew,
        epochs=1, batch_size=8, lr=1e-3, device="cpu",
        loss_type="huber", huber_delta=0.1,
        train_mc_uncertainty_penalty=True, train_mc_samples=5,
        uncertainty_penalty_weight=0.1, uncertainty_penalty_k=1.0,
        uncertainty_std_reg_weight=0.001,
    )
    rec = model.train_loss_history[0]
    assert "loss/under_penalty" in rec and np.isfinite(rec["loss/under_penalty"])
    assert "loss/std_reg" in rec and np.isfinite(rec["loss/std_reg"])
    assert "train/mean_pred_std" in rec
    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_mc_mean_std_shape() -> None:
    built, ei, ew = _built()
    model = _model(built).to("cpu")
    model.train()  # dropout active -> stochastic passes
    x, y, _ = next(iter(torch.utils.data.DataLoader(built["train"], batch_size=4)))
    samples = [model(x, ei, ew, None)[1] for _ in range(5)]
    stack = torch.stack(samples, dim=0)          # (S, B, N)
    mean = stack.mean(0)
    std = stack.std(0, unbiased=False)
    assert mean.shape == y.shape == std.shape     # (B, N)
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()


def test_penalty_requires_two_mc_samples() -> None:
    built, ei, ew = _built()
    try:
        train_model(_model(built), built["train"], ei, ew,
                    epochs=1, batch_size=8, lr=1e-3, device="cpu",
                    train_mc_uncertainty_penalty=True, train_mc_samples=1)
        raise AssertionError("expected ValueError for train_mc_samples=1 with penalty")
    except ValueError:
        pass


# --- 4 + 5. penalty high (low std) vs low (high std) ------------------------ #
def test_penalty_high_when_std_low_low_when_std_high() -> None:
    y = torch.tensor([[10.0, 10.0]])
    mean = torch.tensor([[2.0, 2.0]])           # error = 8 (large)
    low_std = torch.tensor([[0.5, 0.5]])        # under-dispersed
    high_std = torch.tensor([[20.0, 20.0]])     # wide enough to cover the error
    p_low = under_dispersion_penalty(y, mean, low_std, k=1.0).mean().item()
    p_high = under_dispersion_penalty(y, mean, high_std, k=1.0).mean().item()
    assert p_low > 0.0
    assert p_high == 0.0
    assert p_low > p_high
    # k scales the tolerance: bigger k -> smaller penalty for the same std.
    p_k3 = under_dispersion_penalty(y, mean, low_std, k=3.0).mean().item()
    assert p_k3 < p_low


# --- 6. eval/inference has no noise (predict is deterministic) -------------- #
def test_predict_is_deterministic_no_noise() -> None:
    built, ei, ew = _built()
    model = _model(built)
    out1 = predict(model, built["test"], ei, ew, "cpu", batch_size=8)
    out2 = predict(model, built["test"], ei, ew, "cpu", batch_size=8)
    np.testing.assert_allclose(
        out1["y_pred"].to_numpy(float), out2["y_pred"].to_numpy(float)
    )
    # The training-only noise helper must not be wired into the inference path.
    assert "inject_input_noise" not in inspect.getsource(predict)


# --- 7. noise leaves target + sin_elev/cos_elev untouched ------------------- #
def test_build_noise_indices_excludes_sin_cos() -> None:
    idx = build_noise_feature_indices(PVGIS_STGNN_FEATURES)
    assert PVGIS_STGNN_FEATURES.index("sin_elev") not in idx
    assert PVGIS_STGNN_FEATURES.index("cos_elev") not in idx
    assert PVGIS_STGNN_FEATURES.index("temperature_2m") in idx
    assert len(idx) == len(PVGIS_STGNN_FEATURES) - 2


def test_noise_perturbs_only_continuous_and_keeps_target() -> None:
    feats = list(PVGIS_STGNN_FEATURES)
    idx = build_noise_feature_indices(feats)
    idx_t = torch.tensor(idx, dtype=torch.long)
    torch.manual_seed(0)
    x = torch.randn(4, 2, 24, len(feats))   # (B, N, seq, C)
    y = torch.randn(4, 2)                    # target, must stay untouched
    y_ref = y.clone()
    x_noisy = inject_input_noise(x.clone(), idx_t, noise_std=0.5, noise_prob=1.0)

    sin_i = feats.index("sin_elev")
    cos_i = feats.index("cos_elev")
    assert torch.equal(x_noisy[..., sin_i], x[..., sin_i])  # excluded -> unchanged
    assert torch.equal(x_noisy[..., cos_i], x[..., cos_i])
    temp_i = feats.index("temperature_2m")
    assert not torch.equal(x_noisy[..., temp_i], x[..., temp_i])  # perturbed
    assert torch.equal(y, y_ref)  # target never touched by noise


def test_noise_noop_when_disabled() -> None:
    feats = list(PVGIS_STGNN_FEATURES)
    idx_t = torch.tensor(build_noise_feature_indices(feats), dtype=torch.long)
    x = torch.randn(3, 2, 24, len(feats))
    # std=0 -> no-op; prob=0 -> no-op
    assert torch.equal(inject_input_noise(x.clone(), idx_t, 0.0, 0.5), x)
    assert torch.equal(inject_input_noise(x.clone(), idx_t, 0.5, 0.0), x)


# --- anomaly-aware noise ---------------------------------------------------- #
def _scores_for(ds, locations, n_times=5):
    """Anomaly scores covering the first n_times target timestamps of `ds`."""
    tt = ds.target_time_all
    rows = []
    for loc in locations:
        for j in range(min(n_times, len(tt))):
            rows.append({"location": loc, "timestamp": tt[j],
                         "label": "unusually_low_solar_potential"})
    return pd.DataFrame(rows)


def test_attach_anomaly_mask_builds_and_subsamples() -> None:
    built, _, _ = _built()
    ds = built["train"]
    assert ds.attach_anomaly_mask(None) == 0          # no scores -> all False
    assert ds.anomaly_mask_all.shape == (len(ds), ds.n_nodes)
    assert not ds.anomaly_mask_all.any()

    n = ds.attach_anomaly_mask(_scores_for(ds, ["loc_a"], n_times=4))
    assert n > 0
    loc_idx = list(ds.loc_ids).index("loc_a")
    assert ds.anomaly_mask_all[:, loc_idx].sum() == 4
    other = list(ds.loc_ids).index("loc_b")
    assert ds.anomaly_mask_all[:, other].sum() == 0   # loc_b not in scores
    # subsample slices the mask consistently with the samples
    before = ds.anomaly_mask_all.shape[1]
    ds.subsample(5, seed=0)
    assert ds.anomaly_mask_all.shape == (len(ds), before)


def test_inject_anomaly_noise_only_on_anomalous_cells() -> None:
    feats = list(PVGIS_STGNN_FEATURES)
    idx_t = torch.tensor(build_noise_feature_indices(feats), dtype=torch.long)
    torch.manual_seed(0)
    x = torch.randn(3, 2, 24, len(feats))
    mask = torch.zeros(3, 2, dtype=torch.bool)
    mask[0, 0] = True  # only sample0/node0 is anomalous
    x_n = inject_input_noise_anomaly(
        x.clone(), idx_t, mask,
        normal_std=0.0, normal_prob=0.0,     # normal cells: no noise
        anomaly_std=0.5, anomaly_prob=1.0,   # anomalous cell: always noised
    )
    temp_i = feats.index("temperature_2m")
    sin_i = feats.index("sin_elev")
    assert not torch.equal(x_n[0, 0, :, temp_i], x[0, 0, :, temp_i])  # anomalous perturbed
    assert torch.equal(x_n[1, 1, :, temp_i], x[1, 1, :, temp_i])      # normal untouched
    assert torch.equal(x_n[0, 0, :, sin_i], x[0, 0, :, sin_i])        # sin_elev excluded


def test_train_model_anomaly_mode_requires_labels() -> None:
    built, ei, ew = _built()
    # No mask attached -> explicit failure.
    try:
        train_model(_model(built), built["train"], ei, ew,
                    epochs=1, batch_size=8, lr=1e-3, device="cpu",
                    train_noise_mode="anomaly", anomaly_noise_std=0.1,
                    anomaly_noise_prob=1.0, feature_names=PVGIS_STGNN_FEATURES)
        raise AssertionError("expected ValueError: anomaly mode without labels")
    except ValueError:
        pass
    # Mask attached but zero coverage -> explicit failure.
    built["train"].attach_anomaly_mask(None)
    try:
        train_model(_model(built), built["train"], ei, ew,
                    epochs=1, batch_size=8, lr=1e-3, device="cpu",
                    train_noise_mode="anomaly", anomaly_noise_std=0.1,
                    anomaly_noise_prob=1.0, feature_names=PVGIS_STGNN_FEATURES)
        raise AssertionError("expected ValueError: anomaly mode zero coverage")
    except ValueError:
        pass


def test_train_model_anomaly_mode_trains_with_coverage() -> None:
    built, ei, ew = _built()
    built["train"].attach_anomaly_mask(_scores_for(built["train"], ["loc_a", "loc_b"], 10))
    model = train_model(
        _model(built), built["train"], ei, ew,
        epochs=1, batch_size=8, lr=1e-3, device="cpu",
        loss_type="huber", huber_delta=0.1,
        train_noise_mode="anomaly", train_noise_std=0.0, train_noise_prob=0.0,
        anomaly_noise_std=0.05, anomaly_noise_prob=0.5,
        feature_names=PVGIS_STGNN_FEATURES,
    )
    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_default_noise_mode_is_random() -> None:
    assert build_arg_parser().parse_args(["--pvgis-dir", "x"]).train_noise_mode == "random"


def test_cli_validate_anomaly_requirements() -> None:
    parser = build_arg_parser()
    base = ["--pvgis-dir", "d", "--train-years", "2016", "--test-year", "2019"]
    # anomaly mode without any anomaly scores -> rejected
    try:
        a = parser.parse_args(base + ["--train-noise-mode", "anomaly",
                                      "--anomaly-noise-std", "0.05",
                                      "--anomaly-noise-prob", "0.9"])
        _validate(a, parser)
        raise AssertionError("expected failure: anomaly mode without scores")
    except SystemExit:
        pass
    # anomaly mode without anomaly noise params -> rejected
    try:
        a = parser.parse_args(base + ["--train-noise-mode", "anomaly",
                                      "--anomaly-scores", "s.csv"])
        _validate(a, parser)
        raise AssertionError("expected failure: anomaly mode without noise params")
    except SystemExit:
        pass
    # valid anomaly config -> passes
    a = parser.parse_args(base + ["--train-noise-mode", "anomaly",
                                  "--anomaly-noise-std", "0.05",
                                  "--anomaly-noise-prob", "0.9",
                                  "--train-anomaly-scores", "train_s.csv"])
    _validate(a, parser)
    assert a.train_noise_mode == "anomaly"


# --- SDE-proxy uncertainty penalty ------------------------------------------ #
def test_default_penalty_mode_is_underdispersion() -> None:
    import inspect as _i
    assert _i.signature(train_model).parameters["uncertainty_penalty_mode"].default \
        == "underdispersion"
    assert build_arg_parser().parse_args(["--pvgis-dir", "x"]).uncertainty_penalty_mode \
        == "underdispersion"


def test_sde_proxy_penalty_in_and_out_losses() -> None:
    mask = torch.tensor([[False, True]])  # idx0 in-dist, idx1 OOD
    # high normal std -> high in_loss; OOD std below floor -> out_loss > 0
    in_hi, out_below = sde_proxy_penalty(torch.tensor([[2.0, 0.01]]), mask, 0.05)
    in_lo, _ = sde_proxy_penalty(torch.tensor([[0.01, 0.01]]), mask, 0.05)
    assert in_hi.item() > in_lo.item()          # (4) in_loss rises with normal std
    assert out_below.item() > 0.0                # (5) OOD std under floor -> penalised
    # OOD std above floor -> out_loss == 0
    _, out_ok = sde_proxy_penalty(torch.tensor([[2.0, 0.2]]), mask, 0.05)
    assert out_ok.item() == 0.0                  # (6)


def test_sde_proxy_requires_anomaly_mode() -> None:
    built, ei, ew = _built()
    try:
        train_model(_model(built), built["train"], ei, ew,
                    epochs=1, batch_size=8, lr=1e-3, device="cpu",
                    train_mc_uncertainty_penalty=True, train_mc_samples=3,
                    uncertainty_penalty_mode="sde_proxy",
                    train_noise_mode="random", feature_names=PVGIS_STGNN_FEATURES)
        raise AssertionError("expected ValueError: sde_proxy needs anomaly mode")
    except ValueError:
        pass


def test_sde_proxy_fails_without_anomaly_cells() -> None:
    built, ei, ew = _built()
    built["train"].attach_anomaly_mask(None)  # all-False mask
    try:
        train_model(_model(built), built["train"], ei, ew,
                    epochs=1, batch_size=8, lr=1e-3, device="cpu",
                    train_mc_uncertainty_penalty=True, train_mc_samples=3,
                    uncertainty_penalty_mode="sde_proxy",
                    train_noise_mode="anomaly",
                    anomaly_noise_std=0.03, anomaly_noise_prob=0.7,
                    feature_names=PVGIS_STGNN_FEATURES)
        raise AssertionError("expected ValueError: sde_proxy with no anomaly cells")
    except ValueError:
        pass


def test_sde_proxy_trains_finite_and_logs() -> None:
    built, ei, ew = _built()
    built["train"].attach_anomaly_mask(_scores_for(built["train"], ["loc_a", "loc_b"], 10))
    model = train_model(
        _model(built), built["train"], ei, ew,
        epochs=1, batch_size=8, lr=1e-3, device="cpu",
        loss_type="huber", huber_delta=0.1,
        train_mc_uncertainty_penalty=True, train_mc_samples=3,
        uncertainty_penalty_mode="sde_proxy",
        sde_proxy_in_weight=0.001, sde_proxy_out_weight=0.1, sde_proxy_std_min_ood=0.05,
        train_noise_mode="anomaly", train_noise_std=0.0, train_noise_prob=0.0,
        anomaly_noise_std=0.03, anomaly_noise_prob=0.7,
        feature_names=PVGIS_STGNN_FEATURES,
    )
    rec = model.train_loss_history[0]
    for key in ("train/uncertainty_in_loss", "train/uncertainty_out_loss",
                "train/mean_std_normal", "train/mean_std_anomaly",
                "train/std_ratio_anomaly_vs_normal"):
        assert key in rec
    assert "loss/under_penalty" not in rec  # underdispersion keys absent in sde mode
    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_sde_proxy_not_applied_to_kt_aux() -> None:
    src = inspect.getsource(train_model)
    assert "sde_proxy_penalty(" in src
    # kt aux uses the plain loss module on the MC-mean kt head, with NO sde terms.
    assert "loss_fn(pred_ghi_mean, _kt_target(k))" in src
    for line in src.splitlines():
        if "_kt_target(k)" in line:
            assert not any(t in line for t in ("sde_proxy", "in_loss", "out_loss"))


def test_cli_sde_proxy_validation() -> None:
    parser = build_arg_parser()
    base = ["--pvgis-dir", "d", "--train-years", "2016", "--test-year", "2019"]
    # sde_proxy without the MC penalty -> rejected
    try:
        a = parser.parse_args(base + [
            "--uncertainty-penalty-mode", "sde_proxy",
            "--train-noise-mode", "anomaly", "--anomaly-noise-std", "0.03",
            "--anomaly-noise-prob", "0.7", "--anomaly-scores", "s.csv"])
        _validate(a, parser)
        raise AssertionError("expected failure: sde_proxy without MC penalty")
    except SystemExit:
        pass
    # sde_proxy with random noise mode -> rejected
    try:
        a = parser.parse_args(base + [
            "--train-mc-uncertainty-penalty", "--train-mc-samples", "5",
            "--uncertainty-penalty-mode", "sde_proxy", "--train-noise-mode", "random"])
        _validate(a, parser)
        raise AssertionError("expected failure: sde_proxy with random noise mode")
    except SystemExit:
        pass
    # valid sde_proxy config
    a = parser.parse_args(base + [
        "--dropout", "0.3", "--train-mc-uncertainty-penalty", "--train-mc-samples", "5",
        "--uncertainty-penalty-mode", "sde_proxy",
        "--sde-proxy-in-weight", "0.001", "--sde-proxy-out-weight", "0.1",
        "--sde-proxy-std-min-ood", "0.05",
        "--train-noise-mode", "anomaly", "--anomaly-noise-std", "0.03",
        "--anomaly-noise-prob", "0.7", "--train-anomaly-scores", "train_s.csv"])
    _validate(a, parser)
    assert a.uncertainty_penalty_mode == "sde_proxy"
    assert a.sde_proxy_in_weight == 0.001
    assert a.sde_proxy_out_weight == 0.1
    assert a.sde_proxy_std_min_ood == 0.05


# --- CLI surface ------------------------------------------------------------ #
def test_cli_new_flags_parse() -> None:
    parser = build_arg_parser()
    base = parser.parse_args(["--pvgis-dir", "x"])
    assert base.train_mc_uncertainty_penalty is False
    assert base.train_mc_samples == 1
    assert base.uncertainty_penalty_weight == 0.0
    assert base.uncertainty_penalty_k == 1.0
    assert base.uncertainty_std_reg_weight == 0.0
    assert base.train_noise_std == 0.0
    assert base.train_noise_prob == 0.0
    a = parser.parse_args([
        "--pvgis-dir", "x", "--train-mc-uncertainty-penalty",
        "--train-mc-samples", "5", "--uncertainty-penalty-weight", "0.1",
        "--uncertainty-penalty-k", "1.0", "--uncertainty-std-reg-weight", "0.001",
        "--train-noise-std", "0.01", "--train-noise-prob", "0.5",
    ])
    assert a.train_mc_uncertainty_penalty is True
    assert a.train_mc_samples == 5
    assert a.uncertainty_penalty_weight == 0.1
    assert a.train_noise_std == 0.01
    assert a.train_noise_prob == 0.5


if __name__ == "__main__":
    test_defaults_off_are_baseline()
    test_penalty_enabled_adds_keys_and_trains()
    test_mc_mean_std_shape()
    test_penalty_requires_two_mc_samples()
    test_penalty_high_when_std_low_low_when_std_high()
    test_predict_is_deterministic_no_noise()
    test_build_noise_indices_excludes_sin_cos()
    test_noise_perturbs_only_continuous_and_keeps_target()
    test_noise_noop_when_disabled()
    test_attach_anomaly_mask_builds_and_subsamples()
    test_inject_anomaly_noise_only_on_anomalous_cells()
    test_train_model_anomaly_mode_requires_labels()
    test_train_model_anomaly_mode_trains_with_coverage()
    test_default_noise_mode_is_random()
    test_cli_validate_anomaly_requirements()
    test_default_penalty_mode_is_underdispersion()
    test_sde_proxy_penalty_in_and_out_losses()
    test_sde_proxy_requires_anomaly_mode()
    test_sde_proxy_fails_without_anomaly_cells()
    test_sde_proxy_trains_finite_and_logs()
    test_sde_proxy_not_applied_to_kt_aux()
    test_cli_sde_proxy_validation()
    test_cli_new_flags_parse()
    print("PASS: Huber + noise + under-dispersion + sde_proxy penalty")
