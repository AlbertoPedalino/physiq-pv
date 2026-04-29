import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

SEQ_LEN = 120
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

    def __init__(self, ds: xr.Dataset, qs: xr.DataArray, seq_len: int = SEQ_LEN):
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

        # Normalise ENERGIA per-plant by each plant's own p99 (daytime only).
        # Puts every plant in [0, ~1] regardless of installed capacity,
        # so the model learns shape/timing rather than absolute scale.
        energia_raw = np.nan_to_num(ds["ENERGIA"].values.T, nan=0.0)  # (T, N)
        pvgis_raw   = ds["pvgis_ref"].values.T                         # (T, N) kW/kWp
        day_mask    = pvgis_raw > 0.1                                  # daytime rows
        N_plants    = energia_raw.shape[1]
        pv_scale    = np.ones(N_plants, dtype=np.float64)
        pvgis_p99   = np.ones(N_plants, dtype=np.float64)
        for p in range(N_plants):
            mask_p   = day_mask[:, p]
            e_vals   = energia_raw[mask_p, p]
            e_vals   = e_vals[e_vals > 0]
            g_vals   = pvgis_raw[mask_p, p]
            g_vals   = g_vals[g_vals > 0]
            if len(e_vals) > 10:
                pv_scale[p]  = float(np.percentile(e_vals, 99)) + 1e-6
            if len(g_vals) > 10:
                pvgis_p99[p] = float(np.percentile(g_vals, 99)) + 1e-6
        self.pv_scale  = pv_scale                                      # (N,) per-plant kWh peak
        self.target_pv = (energia_raw / pv_scale[None, :]).astype(np.float32)

        # eta in normalised space: pred_pv_norm / pred_ghi ≈ 1 / pvgis_p99[p]
        # derivation: target_pv_norm/target_ghi = (E/p99_E)/pvgis = eta_base*kWp/(p99_E) = 1/pvgis_p99
        self.eta_adjusted = (1.0 / pvgis_p99).astype(np.float32)      # (N,)

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
