import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

# Updated for hourly data: 24 hours of context (instead of 120 for 2-hourly = 10 days)
# Hourly: 24 timesteps = 1 day context (sufficient for intra-daily patterns)
SEQ_LEN = 24  
N_FEATURES = 5  # temperature_2m, solar_irradiance_poa, wind_speed_10m, pvgis_ref, QS


class PVDataset(Dataset):
    """
    Sliding-window PyTorch Dataset over a PV xarray.Dataset + QS DataArray.

    Yields (x, y_ghi, y_pv, qs, eta) for each valid timestep:
      x      (N, seq_len, 5)  — normalised input features
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
        ref   = ds["pvgis_ref"].values.T
        qs_v  = np.nan_to_num(qs.values.T, nan=0.0)    # (T, N) — NaN=night/marginal → 0 = no quality info

        self.feats = np.stack(
            [_norm(temp), _norm(solar), _norm(wind), _norm(ref), qs_v], axis=-1
        ).astype(np.float32)                            # (T, N, 5)

        energia_raw = np.nan_to_num(ds["ENERGIA"].values.T, nan=0.0)  # (T, N)
        pvgis_raw   = ds["pvgis_ref"].values.T                         # (T, N) kW/kWp
        day_mask    = pvgis_raw > 0.25                                 # skip noisy dawn/dusk (~100 W/kWp)
        N_plants    = energia_raw.shape[1]

        # pv_scale: use real registered kWp when available, fall back to p99 inference.
        # With real kWp: target_pv_norm = ENERGIA/kWp → [kW/kWp], same units as pvgis_ref,
        # so eta_adjusted = median(target_pv_norm/pvgis_ref) = true Performance Ratio ∈ [0,1].
        pv_scale  = np.ones(N_plants, dtype=np.float64)
        pvgis_p99 = np.ones(N_plants, dtype=np.float64)
        for p in range(N_plants):
            mask_p = day_mask[:, p]
            e_vals = energia_raw[mask_p, p]; e_vals = e_vals[e_vals > 0]
            g_vals = pvgis_raw[mask_p, p];  g_vals = g_vals[g_vals > 0]
            if len(e_vals) > 10:
                pv_scale[p]  = float(np.percentile(e_vals, 99)) + 1e-6
            if len(g_vals) > 10:
                pvgis_p99[p] = float(np.percentile(g_vals, 99)) + 1e-6

        # kWp_real stored for external analysis only — not used as normalization denominator.
        # p99(ENERGIA) is the correct scale for the physics loss: targets stay in [0, ~1].
        self.kwp_real  = kwp
        self.pv_scale  = pv_scale
        self.pvgis_p99 = pvgis_p99
        target_pv_norm = (energia_raw / pv_scale[None, :])             # (T, N) ∈ [0, ~1]
        self.target_pv = target_pv_norm.astype(np.float32)

        # eta_adjusted[p] = median(ENERGIA[p] / (pv_scale[p] * pvgis_ref[p]))
        # With real kWp: this equals actual PR; with p99 fallback: approximate PR.
        eta_adjusted = np.ones(N_plants, dtype=np.float64)
        n_valid = np.zeros(N_plants, dtype=int)
        for p in range(N_plants):
            mask_p = day_mask[:, p] & (pvgis_raw[:, p] > 0) & (target_pv_norm[:, p] > 0)
            n_valid[p] = int(mask_p.sum())
            if n_valid[p] > 10:
                ratio = target_pv_norm[mask_p, p] / pvgis_raw[mask_p, p]
                eta_adjusted[p] = float(np.median(ratio))
        # Fleet median fallback only for plants without real kWp AND few samples.
        needs_fallback = (n_valid < 50)
        if kwp is not None:
            needs_fallback &= ~(np.isfinite(kwp) & (kwp > 0))
        if needs_fallback.any():
            fleet_eta_med = float(np.median(eta_adjusted[~needs_fallback])) if (~needs_fallback).any() else 0.8
            eta_adjusted[needs_fallback] = fleet_eta_med
        eta_adjusted = np.clip(eta_adjusted, 0.1, 1.05)
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
