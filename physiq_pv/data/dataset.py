import numpy as np
import pandas as pd
import pvlib
import torch
import xarray as xr
from torch.utils.data import Dataset

SEQ_LEN = 24
# Features: temperature_2m, solar_irradiance_poa, wind_speed_10m,
# sin_solar_elev, cos_solar_elev, m1, m2, m3, m4, m5, pv_lag,
# kt, kt_std_3h, dghi_dt
# pv_lag is the normalised past PV output (target_pv_norm); slicing feats[t-seq_len:t]
# at training time yields PV history strictly up to t-1 -> no target leakage.
# kt: clearness index = solar_poa / ghi_cs (cloud transparency proxy)
# kt_std_3h: 3-hour rolling std of kt (cloud-induced variability)
# dghi_dt: solar_poa first difference (ramp rate, transient regime)
N_FEATURES = 14


def _solar_geometry_and_clearsky(
    times: pd.DatetimeIndex, lats: np.ndarray, lons: np.ndarray,
) -> tuple:
    """
    Compute sin/cos of apparent solar elevation and clear-sky GHI per plant and timestep.

    Returns:
      sin_elev (T, N) in [0, 1]
      cos_elev (T, N) in [0, 1]
      ghi_cs   (T, N) clear-sky GHI in kW/m^2 (Ineichen model)

    Plants with NaN coordinates fall back to fleet-mean lat/lon.
    Ineichen requires Linke turbidity; pvlib provides a global lookup table.
    Falls back to simplified_solis if the lookup is unavailable.
    """
    T, N = len(times), len(lats)
    times_utc = times.tz_localize("UTC") if times.tzinfo is None else times
    fleet_lat = float(np.nanmean(lats))
    fleet_lon = float(np.nanmean(lons))

    sin_elev = np.zeros((T, N), dtype=np.float32)
    cos_elev = np.zeros((T, N), dtype=np.float32)
    ghi_cs = np.zeros((T, N), dtype=np.float32)

    for p in range(N):
        lat_p = float(lats[p]) if np.isfinite(lats[p]) else fleet_lat
        lon_p = float(lons[p]) if np.isfinite(lons[p]) else fleet_lon
        loc = pvlib.location.Location(lat_p, lon_p, tz="UTC")
        sp = loc.get_solarposition(times_utc)
        elev = np.clip(sp["apparent_elevation"].values, 0.0, 90.0).astype(np.float32)
        sin_elev[:, p] = np.sin(np.radians(elev))
        cos_elev[:, p] = np.cos(np.radians(elev))

        try:
            cs = loc.get_clearsky(times_utc, model="ineichen")
        except Exception:
            cs = loc.get_clearsky(times_utc, model="simplified_solis")
        ghi_p = np.nan_to_num(cs["ghi"].values, nan=0.0).astype(np.float32) / 1000.0
        ghi_cs[:, p] = np.clip(ghi_p, 0.0, None)

    return sin_elev, cos_elev, ghi_cs


class PVDataset(Dataset):
    """
    Sliding-window PyTorch Dataset over PV xarray.Dataset + QS components.

    Yields (x, y_ghi, y_pv, eta, ghi_cs) for each valid timestep:
      x       (N, seq_len, 10) - normalised input features
      y_ghi   (N,)             - GHI target [kW/m^2]
      y_pv    (N,)             - ENERGIA target [kWh]
      eta     (N,)             - per-plant eta proxy
      ghi_cs  (N,)             - clear-sky GHI [kW/m^2] at target timestep

    m_components must contain m1..m5 numpy arrays of shape (N_plants, T),
    each in [0, 1]. NaN entries are filled with 0.0.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        m_components: dict,
        seq_len: int = SEQ_LEN,
        kwp: "np.ndarray | None" = None,
        eta_max: float = 0.98,
    ):
        if eta_max <= 0.1:
            raise ValueError("eta_max must be greater than 0.1")
        for key in ("m1", "m2", "m3", "m4", "m5"):
            if key not in m_components:
                raise ValueError(f"m_components missing required key '{key}'")
        self.seq_len = seq_len
        self.eta_max = float(eta_max)
        T = ds.sizes["time"]

        def _norm(arr: np.ndarray) -> np.ndarray:
            mu = np.nanmean(arr)
            s = np.nanstd(arr) + 1e-6
            return (np.nan_to_num(arr, nan=mu) - mu) / s

        temp = ds["temperature_2m"].values.T
        solar = ds["solar_irradiance_poa"].values.T
        wind = ds["wind_speed_10m"].values.T

        # m_components arrive as (N_plants, T); transpose to (T, N_plants).
        m1 = np.nan_to_num(np.asarray(m_components["m1"]).T, nan=0.0).astype(np.float32)
        m2 = np.nan_to_num(np.asarray(m_components["m2"]).T, nan=0.0).astype(np.float32)
        m3 = np.nan_to_num(np.asarray(m_components["m3"]).T, nan=0.0).astype(np.float32)
        m4 = np.nan_to_num(np.asarray(m_components["m4"]).T, nan=0.0).astype(np.float32)
        m5 = np.nan_to_num(np.asarray(m_components["m5"]).T, nan=0.0).astype(np.float32)

        lats = ds["lat"].values.astype(float)
        lons = ds["lon"].values.astype(float)
        times_pd = pd.DatetimeIndex(ds.coords["time"].values)
        sin_elev, cos_elev, ghi_cs = _solar_geometry_and_clearsky(times_pd, lats, lons)
        self.ghi_cs = ghi_cs  # (T, N) clear-sky GHI in kW/m^2

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
        self.pvgis_p99 = solar_p99  # legacy alias
        self.solar_p99 = solar_p99

        target_pv_norm = np.clip(energia_raw / pv_scale[None, :], 0.0, 1.5)
        self.target_pv = target_pv_norm.astype(np.float32)

        # Lagged PV channel: at sample index t we slice feats[t-seq_len:t], so the
        # PV values exposed to the encoder are strictly target_pv_norm[t-seq_len..t-1].
        pv_lag = target_pv_norm.astype(np.float32)  # (T, N) in [0, ~1.5]

        # Cloud-dynamics features (Phase A feature engineering):
        # kt = clearness index in [0, ~1.2]; values > 1 occur due to cloud edge
        # enhancement. Clamp slightly above 1 to keep distribution stable.
        kt = np.where(ghi_cs > 0.01, solar_raw_kwm2 / (ghi_cs + 1e-6), 0.0)
        kt = np.clip(kt, 0.0, 1.5).astype(np.float32)

        # 3-hour rolling std of kt per plant: cloud-induced variability proxy.
        kt_std_3h = np.zeros_like(kt)
        for p in range(N_plants):
            series = pd.Series(kt[:, p])
            kt_std_3h[:, p] = series.rolling(window=3, min_periods=1).std().fillna(0.0).to_numpy().astype(np.float32)

        # First-difference of normalized solar (ramp rate). Pads first row with 0.
        dghi = np.zeros_like(solar_raw_kwm2, dtype=np.float32)
        dghi[1:, :] = (solar_raw_kwm2[1:, :] - solar_raw_kwm2[:-1, :]).astype(np.float32)

        feature_arrays = [
            _norm(temp),
            _norm(solar),
            _norm(wind),
            sin_elev,
            cos_elev,
            m1, m2, m3, m4, m5,
            pv_lag,
            kt,
            kt_std_3h,
            _norm(dghi),
        ]
        self.feats = np.stack(feature_arrays, axis=-1).astype(np.float32)

        # Eta proxy via weighted linear regression through origin (peso = irradianza).
        eta_adjusted = np.ones(N_plants, dtype=np.float64)
        n_valid = np.zeros(N_plants, dtype=int)
        for p in range(N_plants):
            mask_p = day_mask[:, p] & (solar_raw_kwm2[:, p] > 0) & (target_pv_norm[:, p] > 0)
            n_valid[p] = int(mask_p.sum())
            if n_valid[p] > 10:
                y = target_pv_norm[mask_p, p]
                x = solar_raw_kwm2[mask_p, p] / solar_p99[p]
                w = x
                num = float(np.sum(w * x * y))
                den = float(np.sum(w * x * x)) + 1e-12
                eta_adjusted[p] = num / den

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
        self.valid_starts = np.arange(seq_len, T - 1)

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int):
        t = self.valid_starts[idx]
        x = torch.from_numpy(self.feats[t - self.seq_len : t].transpose(1, 0, 2))  # (N, seq_len, C)
        y_pv = torch.from_numpy(self.target_pv[t])
        y_ghi = torch.from_numpy(self.target_ghi[t])
        eta = torch.from_numpy(self.eta_adjusted)
        ghi_cs = torch.from_numpy(self.ghi_cs[t])
        return x, y_ghi, y_pv, eta, ghi_cs
