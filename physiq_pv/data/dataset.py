import numpy as np
import pandas as pd
import pvlib
import torch
import xarray as xr
from torch.utils.data import Dataset

SEQ_LEN = 24
# temperature_2m, solar_irradiance_poa, wind_speed_10m, sin_solar_elev, cos_solar_elev, QS, m1_past
# PVGIS reference is not used in model inputs or preprocessing.
N_FEATURES = 7


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


def _causal_m1_past(
    target_pv_norm: np.ndarray,
    solar_norm: np.ndarray,
    day_mask: np.ndarray,
    window: int = 720,
) -> np.ndarray:
    """
    Causal rolling correlation between normalized PV and normalized irradiance.

    m1_past[t] uses data up to t-1 via shift(1), so it can be used as an
    input feature when predicting target t without leaking the target value.
    """
    T, N = target_pv_norm.shape
    min_p = max(window // 4, 10)
    m1 = np.zeros((T, N), dtype=np.float32)

    for p in range(N):
        r = pd.Series(np.where(day_mask[:, p], target_pv_norm[:, p], np.nan))
        v = pd.Series(np.where(day_mask[:, p], solar_norm[:, p], np.nan))
        corr = r.rolling(window, min_periods=min_p).corr(v).shift(1).values
        m1[:, p] = np.nan_to_num(np.clip(corr, 0.0, 1.0), nan=0.0).astype(np.float32)

    return m1


class PVDataset(Dataset):
    """
    Sliding-window PyTorch Dataset over a PV xarray.Dataset + QS DataArray.

    Yields (x, y_ghi, y_pv, qs, eta) for each valid timestep:
      x      (N, seq_len, 7)  - normalised input features
      y_ghi  (N,)             - GHI target [kW/m^2]
      y_pv   (N,)             - ENERGIA target [kWh]
      qs     (N,)             - quality score at prediction step
      eta    (N,)             - per-plant eta proxy
    """

    def __init__(
        self,
        ds: xr.Dataset,
        qs: xr.DataArray,
        seq_len: int = SEQ_LEN,
        kwp: "np.ndarray | None" = None,
        eta_max: float = 0.98,
        include_m1_past: bool = True,
    ):
        if eta_max <= 0.1:
            raise ValueError("eta_max must be greater than 0.1")
        self.seq_len = seq_len
        self.eta_max = float(eta_max)
        self.include_m1_past = bool(include_m1_past)
        T = ds.sizes["time"]

        def _norm(arr: np.ndarray) -> np.ndarray:
            mu = np.nanmean(arr)
            s = np.nanstd(arr) + 1e-6
            return (np.nan_to_num(arr, nan=mu) - mu) / s

        temp = ds["temperature_2m"].values.T
        solar = ds["solar_irradiance_poa"].values.T
        wind = ds["wind_speed_10m"].values.T
        qs_v = np.nan_to_num(qs.values.T, nan=0.0)

        lats = ds["lat"].values.astype(float)
        lons = ds["lon"].values.astype(float)
        times_pd = pd.DatetimeIndex(ds.coords["time"].values)
        sin_elev, cos_elev = _solar_geometry(times_pd, lats, lons)

        energia_raw = np.nan_to_num(ds["ENERGIA"].values.T, nan=0.0)  # (T, N)
        solar_raw_kwm2 = np.clip(ds["solar_irradiance_poa"].values.T / 1000.0, 0.0, None)  # (T, N)
        N_plants = energia_raw.shape[1]

        # Day mask from deterministic geometry plus irradiance floor.
        geom_day = sin_elev > 0.05
        irr_day = solar_raw_kwm2 > 0.03
        day_mask = geom_day & irr_day

        # pv_scale: p99(daytime ENERGIA) per plant. Targets stay in [0, ~1].
        pv_scale = np.ones(N_plants, dtype=np.float64)
        solar_p99 = np.ones(N_plants, dtype=np.float64)
        for p in range(N_plants):
            e_vals = energia_raw[day_mask[:, p], p]
            e_vals = e_vals[e_vals > 0]
            s_vals = solar_raw_kwm2[day_mask[:, p], p]
            s_vals = s_vals[s_vals > 0]
            if len(e_vals) > 10:
                pv_scale[p] = float(np.percentile(e_vals, 99)) + 1e-6
            if len(s_vals) > 10:
                solar_p99[p] = float(np.percentile(s_vals, 99)) + 1e-6

        self.kwp_real = kwp
        self.pv_scale = pv_scale
        # Backward-compatible name used in notebooks.
        self.pvgis_p99 = solar_p99
        self.solar_p99 = solar_p99

        target_pv_norm = np.clip(energia_raw / pv_scale[None, :], 0.0, 1.5)
        self.target_pv = target_pv_norm.astype(np.float32)
        solar_norm_full = solar_raw_kwm2 / (solar_p99[None, :] + 1e-6)
        m1_past = _causal_m1_past(target_pv_norm, solar_norm_full, day_mask)

        feature_arrays = [_norm(temp), _norm(solar), _norm(wind), sin_elev, cos_elev, qs_v]
        if self.include_m1_past:
            feature_arrays.append(m1_past)
        self.feats = np.stack(feature_arrays, axis=-1).astype(np.float32)

        # Eta proxy from normalized PV vs normalized irradiance.
        eta_adjusted = np.ones(N_plants, dtype=np.float64)
        n_valid = np.zeros(N_plants, dtype=int)
        for p in range(N_plants):
            mask_p = day_mask[:, p] & (solar_raw_kwm2[:, p] > 0) & (target_pv_norm[:, p] > 0)
            n_valid[p] = int(mask_p.sum())
            if n_valid[p] > 10:
                solar_norm = solar_raw_kwm2[mask_p, p] / solar_p99[p]
                ratio = target_pv_norm[mask_p, p] / (solar_norm + 1e-6)
                eta_adjusted[p] = float(np.median(ratio))

        # Fleet median fallback only for plants without real kWp and few samples.
        needs_fallback = n_valid < 50
        if kwp is not None:
            needs_fallback &= ~(np.isfinite(kwp) & (kwp > 0))
        if needs_fallback.any():
            fleet_eta_med = float(np.median(eta_adjusted[~needs_fallback])) if (~needs_fallback).any() else 0.8
            eta_adjusted[needs_fallback] = fleet_eta_med

        eta_adjusted = np.clip(eta_adjusted, 0.1, self.eta_max)
        self.eta_adjusted = eta_adjusted.astype(np.float32)

        self.target_ghi = solar_raw_kwm2.astype(np.float32)
        self.eta_base = ds["eta_base"].values.astype(np.float32)
        self.qs_v = qs_v.astype(np.float32)
        self.valid_starts = np.arange(seq_len, T - 1)

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int):
        t = self.valid_starts[idx]
        x = torch.from_numpy(self.feats[t - self.seq_len : t].transpose(1, 0, 2))  # (N, seq_len, C)
        y_pv = torch.from_numpy(self.target_pv[t])
        y_ghi = torch.from_numpy(self.target_ghi[t])
        qs = torch.from_numpy(self.qs_v[t])
        eta = torch.from_numpy(self.eta_adjusted)
        return x, y_ghi, y_pv, qs, eta
