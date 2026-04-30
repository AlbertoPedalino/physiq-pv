import numpy as np
import pandas as pd
import pvlib
import torch
import xarray as xr
from torch.utils.data import Dataset

SEQ_LEN = 24
# temperature_2m, solar_irradiance_poa, wind_speed_10m, sin_solar_elev, cos_solar_elev, QS
# pvgis_ref removed as input feature — replaced by deterministic solar geometry (source-independent).
# pvgis_ref still used internally for eta_adjusted and day_mask thresholds.
N_FEATURES = 6


def _solar_geometry(times: pd.DatetimeIndex, lats: np.ndarray, lons: np.ndarray) -> tuple:
    """
    Compute sin/cos of apparent solar elevation per plant and timestep.
    Returns sin_elev (T, N), cos_elev (T, N), both in [0, 1].
    Plants with NaN coordinates fall back to fleet-mean lat/lon.
    """
    T, N = len(times), len(lats)
    times_utc = times.tz_localize("UTC") if times.tzinfo is None else times
    fleet_lat = float(np.nanmean(lats))
    fleet_lon = float(np.nanmean(lons))

    sin_elev = np.zeros((T, N), dtype=np.float32)
    cos_elev = np.zeros((T, N), dtype=np.float32)

    for p in range(N):
        lat_p = float(lats[p]) if np.isfinite(lats[p]) else fleet_lat
        lon_p = float(lons[p]) if np.isfinite(lons[p]) else fleet_lon
        loc = pvlib.location.Location(lat_p, lon_p, tz="UTC")
        sp = loc.get_solarposition(times_utc)
        elev = np.clip(sp["apparent_elevation"].values, 0.0, 90.0).astype(np.float32)
        sin_elev[:, p] = np.sin(np.radians(elev))
        cos_elev[:, p] = np.cos(np.radians(elev))

    return sin_elev, cos_elev


class PVDataset(Dataset):
    """
    Sliding-window PyTorch Dataset over a PV xarray.Dataset + QS DataArray.

    Yields (x, y_ghi, y_pv, qs, eta) for each valid timestep:
      x      (N, seq_len, 6)  — normalised input features
      y_ghi  (N,)             — GHI target [kW/m²]
      y_pv   (N,)             — ENERGIA target [kWh]
      qs     (N,)             — quality score at prediction step
      eta    (N,)             — eta_base per plant
    """

    def __init__(
        self,
        ds: xr.Dataset,
        qs: xr.DataArray,
        seq_len: int = SEQ_LEN,
        kwp: "np.ndarray | None" = None,
    ):
        self.seq_len = seq_len
        T = ds.sizes["time"]

        def _norm(arr: np.ndarray) -> np.ndarray:
            mu = np.nanmean(arr)
            s = np.nanstd(arr) + 1e-6
            return (np.nan_to_num(arr, nan=mu) - mu) / s

        temp  = ds["temperature_2m"].values.T           # (T, N)
        solar = ds["solar_irradiance_poa"].values.T
        wind  = ds["wind_speed_10m"].values.T
        qs_v  = np.nan_to_num(qs.values.T, nan=0.0)    # (T, N) — NaN=night/marginal → 0

        lats = ds["lat"].values.astype(float)
        lons = ds["lon"].values.astype(float)
        times_pd = pd.DatetimeIndex(ds.coords["time"].values)
        sin_elev, cos_elev = _solar_geometry(times_pd, lats, lons)  # (T, N)

        self.feats = np.stack(
            [_norm(temp), _norm(solar), _norm(wind), sin_elev, cos_elev, qs_v], axis=-1
        ).astype(np.float32)                            # (T, N, 6)

        energia_raw = np.nan_to_num(ds["ENERGIA"].values.T, nan=0.0)  # (T, N)
        pvgis_raw   = ds["pvgis_ref"].values.T                         # (T, N) kW/kWp
        # Low threshold for p99 → more samples → stable normalization for small plants.
        # Stricter threshold for eta ratio computation → avoids noisy dawn/dusk.
        p99_mask    = pvgis_raw > 0.1
        day_mask    = pvgis_raw > 0.25
        N_plants    = energia_raw.shape[1]

        # pv_scale: p99(daytime ENERGIA) per plant. Targets stay in [0, ~1] for physics loss.
        pv_scale  = np.ones(N_plants, dtype=np.float64)
        pvgis_p99 = np.ones(N_plants, dtype=np.float64)
        for p in range(N_plants):
            e_vals = energia_raw[p99_mask[:, p], p]; e_vals = e_vals[e_vals > 0]
            g_vals = pvgis_raw[p99_mask[:, p], p];  g_vals = g_vals[g_vals > 0]
            if len(e_vals) > 10:
                pv_scale[p]  = float(np.percentile(e_vals, 99)) + 1e-6
            if len(g_vals) > 10:
                pvgis_p99[p] = float(np.percentile(g_vals, 99)) + 1e-6

        # kWp_real stored for external analysis only — not used as normalization denominator.
        # p99(ENERGIA) is the correct scale for the physics loss: targets stay in [0, ~1].
        self.kwp_real  = kwp
        self.pv_scale  = pv_scale
        self.pvgis_p99 = pvgis_p99
        target_pv_norm = np.clip(energia_raw / pv_scale[None, :], 0.0, 1.5)  # (T, N) ∈ [0, 1.5]
        self.target_pv = target_pv_norm.astype(np.float32)

        # PR = median(ENERGIA / (pv_scale * pvgis_ref)) — both terms normalized to plant scale.
        # Equivalent: target_pv_norm / (pvgis_ref / pvgis_p99) → dimensionless PR ∈ [0, 1].
        eta_adjusted = np.ones(N_plants, dtype=np.float64)
        n_valid = np.zeros(N_plants, dtype=int)
        for p in range(N_plants):
            mask_p = day_mask[:, p] & (pvgis_raw[:, p] > 0) & (target_pv_norm[:, p] > 0)
            n_valid[p] = int(mask_p.sum())
            if n_valid[p] > 10:
                pvgis_norm = pvgis_raw[mask_p, p] / pvgis_p99[p]
                ratio = target_pv_norm[mask_p, p] / pvgis_norm
                eta_adjusted[p] = float(np.median(ratio))
        # Fleet median fallback only for plants without real kWp AND few samples.
        needs_fallback = (n_valid < 50)
        if kwp is not None:
            needs_fallback &= ~(np.isfinite(kwp) & (kwp > 0))
        if needs_fallback.any():
            fleet_eta_med = float(np.median(eta_adjusted[~needs_fallback])) if (~needs_fallback).any() else 0.8
            eta_adjusted[needs_fallback] = fleet_eta_med
        eta_adjusted = np.clip(eta_adjusted, 0.1, 1.0)
        self.eta_adjusted = eta_adjusted.astype(np.float32)            # (N,) per-plant PR

        solar_raw       = ds["solar_irradiance_poa"].values.T
        self.target_ghi = (solar_raw / 1000.0).astype(np.float32)   # W/m² → kW/m²
        self.eta_base   = ds["eta_base"].values.astype(np.float32)  # (N,) kept for reference
        self.qs_v       = qs_v.astype(np.float32)
        self.valid_starts = np.arange(seq_len, T - 1)

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int):
        t = self.valid_starts[idx]
        x     = torch.from_numpy(self.feats[t - self.seq_len : t].transpose(1, 0, 2))  # (N, seq_len, 5)
        y_pv  = torch.from_numpy(self.target_pv[t])   # (N,)
        y_ghi = torch.from_numpy(self.target_ghi[t])  # (N,)
        qs    = torch.from_numpy(self.qs_v[t])          # (N,)
        eta   = torch.from_numpy(self.eta_adjusted)    # (N,) physics-consistent normalised eta
        return x, y_ghi, y_pv, qs, eta
