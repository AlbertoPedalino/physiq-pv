"""
Sanity checks for the replay-based continual adaptation pipeline.

Run: python tests/test_replay_continual.py

Checks:
  1. Replay buffer adds elements correctly.
  2. Replay buffer samples batch with correct shapes.
  3. Replay buffer can oversample peak/over_100 samples.
  4. Replay buffer never exceeds capacity.
  5. Pipeline runs in debug mode (synthetic) and produces expected outputs.
  6. Replay samples are actually used during continual updates.
  7. Batch feature shape matches N_FEATURES=16 and m1..m5 at channels 5-9.
  8. Real-data debug run (skipped if Sentinel dir not found).
"""
import subprocess
import sys
import tempfile
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import xarray as xr

from physiq_pv.continual.simple_replay_buffer import SimpleReplayBuffer
from physiq_pv.data.dataset import N_FEATURES


def _ok(msg: str) -> None:
    print(f"  PASS: {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL: {msg}")


def _skip(msg: str) -> None:
    print(f"  SKIP: {msg}")


# ------------------------------------------------------------------ #
# Buffer unit tests
# ------------------------------------------------------------------ #

def test_buffer_add_and_size() -> bool:
    buf = SimpleReplayBuffer(capacity=10, seed=0)
    assert len(buf) == 0

    for _ in range(15):
        buf.add(torch.randn(5, 24, 16), torch.randn(5), torch.randn(5), torch.randn(5), torch.randn(5))

    assert len(buf) == 10, f"expected 10, got {len(buf)}"
    assert buf.total_added == 15
    _ok("buffer add + capacity")
    return True


def test_buffer_sample_shapes() -> bool:
    buf = SimpleReplayBuffer(capacity=100, seed=0)
    N, seq, C = 5, 24, 16
    for _ in range(20):
        buf.add(torch.randn(N, seq, C), torch.randn(N), torch.randn(N), torch.randn(N), torch.randn(N))

    x, y_ghi, y_pv, eta, ghi_cs = buf.sample(8)
    assert x.shape == (8, N, seq, C), f"x shape: {x.shape}"
    assert y_pv.shape == (8, N), f"y_pv shape: {y_pv.shape}"
    assert ghi_cs.shape == (8, N), f"ghi_cs shape: {ghi_cs.shape}"
    _ok("buffer sample shapes")
    return True


def test_buffer_peak_aware_sample() -> bool:
    buf = SimpleReplayBuffer(capacity=100, seed=0)
    N, seq, C = 5, 24, 16
    x = torch.randn(N, seq, C)
    y_ghi = torch.zeros(N)
    eta = torch.zeros(N)
    ghi_cs = torch.zeros(N)

    for _ in range(10):
        buf.add(x, y_ghi, torch.full((N,), 0.2), eta, ghi_cs)
    for _ in range(10):
        buf.add(x, y_ghi, torch.full((N,), 0.7), eta, ghi_cs)
    for _ in range(10):
        buf.add(x, y_ghi, torch.full((N,), 1.2), eta, ghi_cs)

    (sx, _sy_ghi, sy_pv, _seta, _sghi_cs), stats = buf.sample_peak_aware(
        8,
        peak_fraction=0.25,
        over_100_fraction=0.25,
        peak_threshold=0.6,
        over_100_threshold=1.0,
    )
    assert sx.shape[0] == 8
    assert stats["n_replay_peak_samples"] >= 2, stats
    assert stats["n_replay_over_100_samples"] >= 2, stats
    assert (sy_pv.max(dim=1).values >= 1.0).sum().item() >= 2

    _ok("buffer peak-aware sampling")
    return True


def test_buffer_capacity_enforced() -> bool:
    cap = 50
    buf = SimpleReplayBuffer(capacity=cap, seed=0)
    for _ in range(200):
        buf.add(torch.randn(3, 10, 5), torch.randn(3), torch.randn(3), torch.randn(3), torch.randn(3))
        assert len(buf) <= cap, f"buffer exceeded capacity: {len(buf)} > {cap}"
    assert len(buf) == cap
    _ok("buffer capacity enforced")
    return True


# ------------------------------------------------------------------ #
# Feature shape + m1..m5 check
# ------------------------------------------------------------------ #

def test_feature_shape_and_m_channels() -> bool:
    """Build PVDataset from synthetic data, verify x has N_FEATURES channels
    and that m1..m5 (channels 5-9) are in [0, 1]."""
    from physiq_pv.data.synthetic_generator import generate_synthetic_dataset
    from physiq_pv.data.quality_score import compute_qs
    from physiq_pv.data.dataset import PVDataset

    ds = generate_synthetic_dataset(seed=0)
    if "eta_base" not in ds.data_vars and "eta_base" not in ds.coords:
        ds = ds.assign_coords(
            eta_base=("plant", np.full(ds.sizes["plant"], 0.18, dtype=np.float64)),
        )
    ds = ds.isel(plant=slice(0, 3), time=slice(0, 200))
    _qs, m_comp = compute_qs(ds, debug=True)
    dataset = PVDataset(ds, m_comp, seq_len=24)

    x, y_ghi, y_pv, eta, ghi_cs = dataset[0]

    assert x.shape[-1] == N_FEATURES, f"expected {N_FEATURES} features, got {x.shape[-1]}"

    m_slice = x[:, :, 5:10]
    assert m_slice.shape[-1] == 5, "m1..m5 should be 5 channels at indices 5-9"

    m_vals = m_slice.numpy()
    assert np.all(m_vals >= 0.0) and np.all(m_vals <= 1.0), (
        f"m1..m5 values outside [0,1]: min={m_vals.min():.4f} max={m_vals.max():.4f}"
    )

    _ok(f"feature shape={x.shape}, m1..m5 at [5:10] in [0,1]")
    return True


# ------------------------------------------------------------------ #
# Bin metrics unit tests
# ------------------------------------------------------------------ #

def test_bin_metrics_basic() -> bool:
    """Binning, NaN handling, empty bin, over_100, diagnostic fields."""
    from physiq_pv.continual.train_replay_continual import compute_bin_metrics

    y_true = np.array([0.1, 0.15, 0.3, 0.5, 0.9, np.nan, 0.05, 1.1])
    y_pred = np.array([0.12, 0.14, 0.28, 0.55, 0.85, 0.5, 0.06, 1.05])

    rows = compute_bin_metrics(y_true, y_pred)

    labels = [r["bin_label"] for r in rows]
    assert labels == ["0_20", "20_40", "40_60", "60_80", "80_100", "over_100"], f"labels: {labels}"

    # 0_20 bin: 0.1, 0.15, 0.05 -> 3 samples
    r0 = rows[0]
    assert r0["count"] == 3, f"0_20 count: {r0['count']}"
    assert r0["mae"] > 0, "0_20 mae should be > 0"
    assert "mean_error" in r0, "missing mean_error field"
    assert "min_y_true" in r0, "missing min_y_true field"

    # 60_80 bin: empty
    r3 = rows[3]
    assert r3["count"] == 0, f"60_80 should be empty, got {r3['count']}"
    assert np.isnan(r3["mae"]), "empty bin mae should be NaN"

    # 80_100 bin: 0.9 -> 1 sample
    r4 = rows[4]
    assert r4["count"] == 1

    # over_100 bin: 1.1 -> 1 sample
    r5 = rows[5]
    assert r5["count"] == 1, f"over_100 count: {r5['count']}"
    assert r5["bin_label"] == "over_100"

    # Diagnostic fields present
    assert r0["n_nan_removed"] == 1, f"expected 1 NaN removed, got {r0['n_nan_removed']}"

    _ok("bin metrics (basic + NaN + empty + over_100 + diagnostics)")
    return True


# ------------------------------------------------------------------ #
# TemporalStream gap-skipping
# ------------------------------------------------------------------ #

def test_temporal_stream_skips_gaps() -> bool:
    """TemporalStream must skip empty months (e.g. June gap) and continue."""
    from physiq_pv.continual.temporal_stream import TemporalStream

    # Create dataset with a 1-month gap: Jan, Feb, Apr (March missing)
    times = pd.concat([
        pd.Series(pd.date_range("2023-01-01", periods=31 * 24, freq="h")),
        pd.Series(pd.date_range("2023-02-01", periods=28 * 24, freq="h")),
        pd.Series(pd.date_range("2023-04-01", periods=30 * 24, freq="h")),
    ]).reset_index(drop=True)
    times = pd.DatetimeIndex(times.unique()).sort_values()

    ds = xr.Dataset(
        {"val": (["time"], np.ones(len(times)))},
        coords={"time": times},
    )

    stream = TemporalStream(
        ds,
        initial_train_start="2023-01-01",
        initial_train_end="2023-01-31",
        window_months=1,
    )

    windows = list(stream.stream_windows())
    window_months = [w[1].month for w in windows]

    # Feb present, March skipped (no data), April present
    assert 2 in window_months, f"February missing from {window_months}"
    assert 4 in window_months, f"April missing from {window_months} (gap-skip bug)"
    assert 3 not in window_months, f"March should be skipped: {window_months}"

    _ok(f"temporal stream skips gaps correctly: months={window_months}")
    return True


# ------------------------------------------------------------------ #
# Pipeline integration (synthetic)
# ------------------------------------------------------------------ #

def _run_pipeline(extra_args: list[str], label: str) -> bool:
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, "-m", "physiq_pv.continual.train_replay_continual",
            "--debug",
            "--replay-buffer-size", "100",
            "--replay-batch-size", "8",
            "--initial-epochs", "1",
            "--update-epochs", "1",
            "--output-dir", tmpdir,
            "--run-name", "sanity",
            "--seed", "42",
        ] + extra_args

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, cwd=str(_REPO_ROOT),
        )

        out_dir = Path(tmpdir) / "sanity"
        if not out_dir.exists():
            _fail(f"{label}: no output (exit {result.returncode})")
            print(f"  STDERR:\n{result.stderr[-800:]}")
            return False

        expected = [
            "config.json", "metrics_per_window.csv", "metrics_by_bin.csv",
            "metrics_by_bin_summary.csv",
            "final_summary.json", "checkpoint_initial.pt", "checkpoint_final.pt",
            "bin_metric_audit.md",
        ]
        for fname in expected:
            if not (out_dir / fname).exists():
                _fail(f"{label}: missing {fname}")
                return False

        df = pd.read_csv(out_dir / "metrics_per_window.csv")
        if len(df) < 2:
            _fail(f"{label}: metrics CSV has only {len(df)} rows")
            return False

        required_cols = [
            "window_id", "window_start", "window_end", "phase",
            "mae", "rmse", "loss",
            "num_recent_samples", "num_replay_samples",
            "num_replay_peak_samples", "num_replay_over_100_samples",
            "num_replay_low_samples", "replay_buffer_size",
        ]
        for col in required_cols:
            if col not in df.columns:
                _fail(f"{label}: missing column {col}")
                return False

        update_rows = df[df["phase"] == "continual_update"]
        if len(update_rows) > 0 and update_rows["num_replay_samples"].sum() == 0:
            _fail(f"{label}: no replay samples used")
            return False

        # Point 8: no NaN in numeric columns
        numeric_cols = ["mae", "rmse", "loss", "num_recent_samples",
                        "num_replay_samples", "replay_buffer_size"]
        for col in numeric_cols:
            if col in df.columns and df[col].isna().any():
                _fail(f"{label}: NaN found in {col}")
                return False

        # Bin metrics CSV
        bin_csv = out_dir / "metrics_by_bin.csv"
        if bin_csv.exists():
            df_bin = pd.read_csv(bin_csv)
            expected_labels = {"0_20", "20_40", "40_60", "60_80", "80_100", "over_100"}
            actual_labels = set(df_bin["bin_label"].unique())
            if not expected_labels.issubset(actual_labels):
                _fail(f"{label}: bin labels {actual_labels} missing some of {expected_labels}")
                return False

        bin_summary_csv = out_dir / "metrics_by_bin_summary.csv"
        df_bin_summary = pd.read_csv(bin_summary_csv)
        required_summary_cols = {
            "bin_label", "n_windows", "n_nonempty_windows", "total_count",
            "mean_mae", "weighted_mean_mae", "worst_mae", "final_mae",
            "mean_rmse", "weighted_mean_rmse", "worst_rmse", "final_rmse",
            "final_count",
        }
        missing_summary_cols = required_summary_cols - set(df_bin_summary.columns)
        if missing_summary_cols:
            _fail(f"{label}: missing bin summary columns {missing_summary_cols}")
            return False

        with open(out_dir / "final_summary.json") as f:
            summary = json.load(f)
        if "bin_metrics_across_windows" not in summary:
            _fail(f"{label}: missing bin_metrics_across_windows in final_summary.json")
            return False
        if "60_80" not in summary["bin_metrics_across_windows"]:
            _fail(f"{label}: missing 60_80 aggregate bin metrics")
            return False

        _ok(f"{label} (files + CSV + bins + replay + no NaN)")
        return True


def test_pipeline_synthetic() -> bool:
    return _run_pipeline(
        ["--data-mode", "synthetic",
         "--initial-train-start", "2023-01-01",
         "--initial-train-end", "2023-03-31",
         "--window-months", "1"],
        "synthetic pipeline",
    )


def test_pipeline_real_debug() -> bool:
    """Real data debug — skipped if Sentinel dir not found."""
    sentinel_dir = "/data/SentinelPV/energy_data/piemonte_energy_data/single_ups"
    if not Path(sentinel_dir).exists():
        _skip("real pipeline (Sentinel dir not found on this machine)")
        return True

    return _run_pipeline(
        ["--data-mode", "real",
         "--initial-train-start", "2019-03-01",
         "--initial-train-end", "2019-05-31",
         "--window-months", "1",
         "--max-windows", "1",
         "--max-plants", "5"],
        "real pipeline",
    )


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def test_feature_names_registry() -> bool:
    """Feature name lists have correct length and differ at solar slot."""
    from physiq_pv.data.dataset import (
        get_feature_names, FEATURE_NAMES_PVGIS_LEGACY,
        FEATURE_NAMES_OPENMETEO_OPERATIONAL, N_FEATURES,
    )

    pvgis = get_feature_names("pvgis_legacy")
    om = get_feature_names("openmeteo_operational")

    assert len(pvgis) == N_FEATURES, f"pvgis list len={len(pvgis)} != {N_FEATURES}"
    assert len(om) == N_FEATURES, f"om list len={len(om)} != {N_FEATURES}"
    assert pvgis[1] == "solar_irradiance_poa", f"pvgis[1]={pvgis[1]}"
    assert om[1] == "shortwave_radiation_ghi_proxy", f"om[1]={om[1]}"
    assert pvgis[0] == om[0] == "temperature_2m"
    assert pvgis[14] == om[14] == "dni_norm"

    try:
        get_feature_names("invalid")
        _fail("feature_names: should raise on invalid feature_set")
        return False
    except ValueError:
        pass

    _ok("feature name registry (pvgis_legacy + openmeteo_operational)")
    return True


def test_openmeteo_dataset_synthetic() -> bool:
    """PVDataset with openmeteo feature_set on synthetic data."""
    from physiq_pv.data.synthetic_generator import generate_synthetic_dataset
    from physiq_pv.data.quality_score import compute_qs
    from physiq_pv.data.dataset import PVDataset, N_FEATURES

    ds = generate_synthetic_dataset(seed=0)
    if "eta_base" not in ds.data_vars and "eta_base" not in ds.coords:
        ds = ds.assign_coords(
            eta_base=("plant", np.full(ds.sizes["plant"], 0.18, dtype=np.float64)),
        )
    ds = ds.isel(plant=slice(0, 3), time=slice(0, 200))
    _qs, m_comp = compute_qs(ds, debug=True)

    dataset = PVDataset(
        ds, m_comp, seq_len=24,
        weather_source="openmeteo_historical_forecast",
        feature_set="openmeteo_operational",
    )
    x, y_ghi, y_pv, eta, ghi_cs = dataset[0]
    assert x.shape[-1] == N_FEATURES, f"expected {N_FEATURES}, got {x.shape[-1]}"
    assert dataset.feature_set == "openmeteo_operational"
    assert dataset._dni_dhi_source == "erbs_decomposition"

    _ok(f"openmeteo_operational synthetic: shape={x.shape}, dni_dhi={dataset._dni_dhi_source}")
    return True


def test_openmeteo_direct_dni_dhi() -> bool:
    """PVDataset uses direct DNI/DHI when available in dataset."""
    from physiq_pv.data.synthetic_generator import generate_synthetic_dataset
    from physiq_pv.data.quality_score import compute_qs
    from physiq_pv.data.dataset import PVDataset

    ds = generate_synthetic_dataset(seed=0)
    if "eta_base" not in ds.data_vars and "eta_base" not in ds.coords:
        ds = ds.assign_coords(
            eta_base=("plant", np.full(ds.sizes["plant"], 0.18, dtype=np.float64)),
        )
    ds = ds.isel(plant=slice(0, 3), time=slice(0, 200))

    N, T = ds.sizes["plant"], ds.sizes["time"]
    ds["direct_normal_irradiance"] = xr.DataArray(
        np.random.rand(N, T).astype(np.float32) * 500,
        dims=["plant", "time"],
    )
    ds["diffuse_radiation"] = xr.DataArray(
        np.random.rand(N, T).astype(np.float32) * 200,
        dims=["plant", "time"],
    )

    _qs, m_comp = compute_qs(ds, debug=True)
    dataset = PVDataset(
        ds, m_comp, seq_len=24,
        weather_source="openmeteo_historical_forecast",
        feature_set="openmeteo_operational",
    )

    assert dataset._dni_dhi_source == "direct_openmeteo"
    _ok(f"direct DNI/DHI: source={dataset._dni_dhi_source}")
    return True


def test_invalid_weather_source() -> bool:
    """Invalid weather_source raises ValueError."""
    from physiq_pv.data.synthetic_generator import generate_synthetic_dataset
    from physiq_pv.data.quality_score import compute_qs
    from physiq_pv.data.dataset import PVDataset

    ds = generate_synthetic_dataset(seed=0)
    if "eta_base" not in ds.data_vars and "eta_base" not in ds.coords:
        ds = ds.assign_coords(
            eta_base=("plant", np.full(ds.sizes["plant"], 0.18, dtype=np.float64)),
        )
    ds = ds.isel(plant=slice(0, 3), time=slice(0, 200))
    _qs, m_comp = compute_qs(ds, debug=True)

    try:
        PVDataset(ds, m_comp, seq_len=24, weather_source="invalid_source")
        _fail("invalid weather_source: should have raised ValueError")
        return False
    except ValueError:
        pass

    try:
        PVDataset(ds, m_comp, seq_len=24, feature_set="invalid_set")
        _fail("invalid feature_set: should have raised ValueError")
        return False
    except ValueError:
        pass

    _ok("invalid weather_source/feature_set raise ValueError")
    return True


def test_download_script_dryrun() -> bool:
    """Download script --dry-run parses without error."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/download_openmeteo_historical_forecast.py",
         "--plants-path", "data/energy_with_coordinates.csv",
         "--start-date", "2019-03-01", "--end-date", "2019-03-02",
         "--dry-run"],
        capture_output=True, text=True, timeout=30,
    )
    if not Path("data/energy_with_coordinates.csv").exists():
        _skip("download dry-run (energy_with_coordinates.csv not found)")
        return True
    if result.returncode != 0:
        _fail(f"download --dry-run failed: {result.stderr[-300:]}")
        return False
    if "dry-run" not in result.stdout.lower():
        _fail("download --dry-run output missing 'dry-run' marker")
        return False
    _ok("download script --dry-run")
    return True


def test_inspect_script_synthetic() -> bool:
    """Inspect script works on a synthetic NetCDF."""
    import subprocess
    import tempfile

    try:
        ds = xr.Dataset({
            "temperature_2m": (["location", "time"], np.random.rand(3, 48).astype(np.float32) * 30),
            "shortwave_radiation": (["location", "time"], np.random.rand(3, 48).astype(np.float32) * 800),
            "wind_speed_10m": (["location", "time"], np.random.rand(3, 48).astype(np.float32) * 10),
        }, coords={
            "lat": ("location", [45.0, 45.1, 45.2]),
            "lon": ("location", [7.0, 7.1, 7.2]),
            "time": pd.date_range("2019-03-01", periods=48, freq="h"),
        })
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            tmp = f.name
            ds.to_netcdf(tmp)

        result = subprocess.run(
            [sys.executable, "scripts/inspect_weather_netcdf.py", "--path", tmp],
            capture_output=True, text=True, timeout=30,
        )
        Path(tmp).unlink(missing_ok=True)

        if result.returncode != 0:
            _fail(f"inspect script failed: {result.stderr[-300:]}")
            return False
        if "shortwave_radiation" not in result.stdout:
            _fail("inspect output missing variable listing")
            return False
        _ok("inspect script on synthetic NetCDF")
        return True
    except Exception as e:
        _fail(f"inspect script: {e}")
        return False


def main() -> None:
    print("Replay continual adaptation sanity checks\n")
    results = [
        test_buffer_add_and_size(),
        test_buffer_sample_shapes(),
        test_buffer_peak_aware_sample(),
        test_buffer_capacity_enforced(),
        test_feature_shape_and_m_channels(),
        test_bin_metrics_basic(),
        test_temporal_stream_skips_gaps(),
        test_feature_names_registry(),
        test_openmeteo_dataset_synthetic(),
        test_openmeteo_direct_dni_dhi(),
        test_invalid_weather_source(),
        test_download_script_dryrun(),
        test_inspect_script_synthetic(),
        test_pipeline_synthetic(),
        test_pipeline_real_debug(),
    ]
    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} checks passed.")
    if n_pass < n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
