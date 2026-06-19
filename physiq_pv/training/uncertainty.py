"""SDE-Net inference: deterministic drift mean + stochastic predictive intervals.

Uncertainty comes from the SDE Brownian term: running the model M times with
`stochastic=True` samples M Brownian paths, and the spread of the outputs is the
(epistemic) predictive distribution. No dropout is involved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from physiq_pv.data.pvgis_dataset import PVGISWindowDataset
from physiq_pv.model.st_gnn import STGNN


@torch.no_grad()
def predict(
    model: STGNN,
    dataset: PVGISWindowDataset,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    """Deterministic prediction (SDE drift only) per (location, timestamp), physical units."""
    if dataset.solar_irradiance_poa_target_all is None:
        raise ValueError(
            "Prediction dataset is missing target-time solar irradiance diagnostics."
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ei, ew = edge_index.to(device), edge_weight.to(device)
    model = model.to(device).eval()
    pv_scale = dataset.pv_scale[None, :]  # (1, N)
    loc_ids = dataset.loc_ids

    locs, times, ytrue, solar_targets, ypred = [], [], [], [], []
    for x, _y, k in loader:
        pred_norm = model(x.to(device), ei, ew, None, stochastic=False)[1].cpu().numpy()
        k = k.numpy()
        y_true = dataset.y_true_all[k]  # (B, N) physical
        solar_target = dataset.solar_irradiance_poa_target_all[k]  # (B, N) W/m2
        pred_phys = pred_norm * pv_scale  # (B, N) physical
        ts = dataset.target_time_all[k].values  # (B,)
        B, N = pred_phys.shape
        locs.append(np.tile(loc_ids, B))
        times.append(np.repeat(ts, N))
        ytrue.append(y_true.reshape(-1))
        solar_targets.append(solar_target.reshape(-1))
        ypred.append(pred_phys.reshape(-1))

    y_true = np.concatenate(ytrue).astype(np.float64)
    solar_target = np.concatenate(solar_targets).astype(np.float64)
    y_pred = np.concatenate(ypred).astype(np.float64)
    error = y_pred - y_true
    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(np.concatenate(times)),
            "location": np.concatenate(locs),
            "y_true": y_true,
            "solar_irradiance_poa_target": solar_target,
            "y_pred": y_pred,
            "error": error,
            "abs_error": np.abs(error),
            "squared_error": error ** 2,
        }
    )


@torch.no_grad()
def predict_sde(
    model: STGNN,
    dataset: PVGISWindowDataset,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: str,
    batch_size: int,
    mc_samples: int,
    coverage_target: float = 0.95,
    z: float = 1.96,
) -> pd.DataFrame:
    """SDE inference: `mc_samples` stochastic Brownian paths -> empirical-quantile PIs.

    Monaco et al. (2025): the forecast uncertainty is the spread of the
    stochastic SDE samples. Each Brownian path yields one prediction mu; the
    predictive mean is E[mu] over paths, the predictive std is Std(mu), and the
    primary PI is the empirical quantile band of the mu samples. No aleatoric /
    epistemic split (Monaco does not separate the two sources).
    """
    if mc_samples < 2:
        raise ValueError(f"mc_samples must be >= 2 for SDE sampling, got {mc_samples}.")
    if not 0.0 < coverage_target < 1.0:
        raise ValueError(f"coverage_target must be in (0, 1), got {coverage_target}.")
    if dataset.solar_irradiance_poa_target_all is None:
        raise ValueError(
            "Prediction dataset is missing target-time solar irradiance diagnostics."
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ei, ew = edge_index.to(device), edge_weight.to(device)
    model = model.to(device).eval()
    print(f"  [sde] stochastic inference: {mc_samples} Brownian paths (SDE-sample spread)")

    pv_scale = dataset.pv_scale[None, :]  # (1, N)
    loc_ids = dataset.loc_ids

    alpha = 1.0 - coverage_target
    q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0
    print(f"  [sde] empirical PI quantiles q{q_lo:.3f}/q{q_hi:.3f}")

    locs, times, ytrue, solar_targets = [], [], [], []
    means, stds, pis_lo, pis_hi = [], [], [], []
    for x, _y, k in loader:
        x = x.to(device)
        k = k.numpy()
        B = len(k)
        # One prediction mu per Brownian path (physical units).
        mu_samples = np.empty((mc_samples, B, len(loc_ids)), dtype=np.float64)
        for s in range(mc_samples):
            mu = model(x, ei, ew, None, stochastic=True)[1].cpu().numpy()
            mu_samples[s] = mu * pv_scale                   # physical
        mean = mu_samples.mean(axis=0)                      # (B, N) predictive mean
        std = mu_samples.std(axis=0)                        # (B, N) SDE-spread std
        # PRIMARY interval: empirical quantiles of the SDE-sample spread.
        lo_pi = np.quantile(mu_samples, q_lo, axis=0)       # (B, N)
        hi_pi = np.quantile(mu_samples, q_hi, axis=0)       # (B, N)
        y_true = dataset.y_true_all[k]  # (B, N) physical
        solar_target = dataset.solar_irradiance_poa_target_all[k]  # (B, N) W/m2
        ts = dataset.target_time_all[k].values  # (B,)
        N = mean.shape[1]
        locs.append(np.tile(loc_ids, B))
        times.append(np.repeat(ts, N))
        ytrue.append(y_true.reshape(-1))
        solar_targets.append(solar_target.reshape(-1))
        means.append(mean.reshape(-1))
        stds.append(std.reshape(-1))
        pis_lo.append(lo_pi.reshape(-1))
        pis_hi.append(hi_pi.reshape(-1))

    y_true = np.concatenate(ytrue).astype(np.float64)
    solar_target = np.concatenate(solar_targets).astype(np.float64)
    y_mean = np.concatenate(means).astype(np.float64)
    y_std = np.concatenate(stds).astype(np.float64)
    y_lower_pi = np.concatenate(pis_lo).astype(np.float64)
    y_upper_pi = np.concatenate(pis_hi).astype(np.float64)
    # Gaussian band: DIAGNOSTIC only (secondary comparison), not the main PI.
    y_lower_g = y_mean - z * y_std
    y_upper_g = y_mean + z * y_std
    error = y_mean - y_true  # y_pred == y_pred_mean
    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(np.concatenate(times)),
            "location": np.concatenate(locs),
            "y_true": y_true,
            "solar_irradiance_poa_target": solar_target,
            "y_pred": y_mean,
            "y_pred_mean": y_mean,
            "y_pred_std": y_std,
            "y_pred_std_raw": y_std,
            "lower_pi": y_lower_pi,
            "upper_pi": y_upper_pi,
            "lower_gaussian": y_lower_g,
            "upper_gaussian": y_upper_g,
            "lower_raw": y_lower_g,
            "upper_raw": y_upper_g,
            "y_pred_lower": y_lower_g,
            "y_pred_upper": y_upper_g,
            "error": error,
            "abs_error": np.abs(error),
            "squared_error": error ** 2,
        }
    )
