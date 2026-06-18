"""MC-Dropout inference and predictive intervals (empirical-quantile bands)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from physiq_pv.data.pvgis_dataset import PVGISWindowDataset
from physiq_pv.model.st_gnn import STGNN

DROPOUT_TYPES = (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)


@torch.no_grad()
def predict(
    model: STGNN,
    dataset: PVGISWindowDataset,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    """Predict on `dataset`; return per-(location, timestamp) predictions in physical units."""
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
        pred_norm = model(x.to(device), ei, ew, None)[1].cpu().numpy()  # (B, N)
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


def enable_dropout_only(model: torch.nn.Module) -> int:
    """Reactivate only dropout layers while the rest of the model stays eval()."""
    n_active = 0
    for module in model.modules():
        if isinstance(module, DROPOUT_TYPES):
            module.train()
            n_active += 1
    return n_active


def active_dropout_names(model: torch.nn.Module) -> list[str]:
    """Qualified names of dropout modules currently in train mode."""
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, DROPOUT_TYPES) and module.training
    ]


@torch.no_grad()
def predict_mc(
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
    """MC-Dropout inference with empirical-quantile predictive intervals."""
    if mc_samples < 2:
        raise ValueError(f"mc_samples must be >= 2 for MC Dropout, got {mc_samples}.")
    if not 0.0 < coverage_target < 1.0:
        raise ValueError(f"coverage_target must be in (0, 1), got {coverage_target}.")
    if dataset.solar_irradiance_poa_target_all is None:
        raise ValueError(
            "Prediction dataset is missing target-time solar irradiance diagnostics."
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ei, ew = edge_index.to(device), edge_weight.to(device)
    model = model.to(device).eval()
    n_active = enable_dropout_only(model)
    if n_active == 0:
        raise RuntimeError(
            "MC Dropout requested but no nn.Dropout layers are present/active "
            "(dropout=0.0?). Re-run with --dropout > 0 so there is stochasticity."
        )
    names = ", ".join(active_dropout_names(model))
    print(f"  [mc] dropout-only inference: {n_active} active layer(s); {names}")

    pv_scale = dataset.pv_scale[None, :]  # (1, N)
    loc_ids = dataset.loc_ids

    alpha = 1.0 - coverage_target
    q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0
    print(f"  [mc] empirical PI quantiles q{q_lo:.3f}/q{q_hi:.3f}")

    locs, times, ytrue, solar_targets = [], [], [], []
    means, stds, pis_lo, pis_hi = [], [], [], []
    for x, _y, k in loader:
        x = x.to(device)
        k = k.numpy()
        B = len(k)
        samples = np.empty((mc_samples, B, len(loc_ids)), dtype=np.float64)
        for s in range(mc_samples):
            pred_norm = model(x, ei, ew, None)[1].cpu().numpy()  # (B, N) normalised
            samples[s] = pred_norm * pv_scale                    # (B, N) physical
        mean = samples.mean(axis=0)  # (B, N)
        std = samples.std(axis=0)    # (B, N) population std over passes
        # PRIMARY interval: empirical quantiles of the MC sample distribution.
        lo_pi = np.quantile(samples, q_lo, axis=0)  # (B, N)
        hi_pi = np.quantile(samples, q_hi, axis=0)  # (B, N)
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
