"""SDE-Net inference: deterministic drift mean + stochastic predictive intervals.

Uncertainty has the two sources of Kong et al. (2020), separated as in their
regression experiment: running the model M times with ``stochastic=True``
samples M Brownian paths, each yielding a Gaussian PV head (mu_s, sigma_s).
The *epistemic* uncertainty is the variance of the predictive mean across paths,
Var_s(mu_s); the *aleatoric* uncertainty is the expected head variance,
E_s[sigma_s^2]. The predictive (total) variance is their sum (law of total
variance for the Gaussian mixture). No dropout is involved.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from physiq_pv.data.pvgis_dataset import PVGISWindowDataset
from physiq_pv.model.st_gnn import STGNN


def _gaussian_mixture_quantile(
    mu_samples: np.ndarray,
    sigma_samples: np.ndarray,
    probability: float,
    *,
    bisection_steps: int = 60,
) -> np.ndarray:
    """Quantile of an equally weighted Gaussian mixture, vectorised over outputs.

    The first axis indexes SDE Brownian paths; every remaining axis is an
    independent predictive distribution. Bisection inverts

        F(y) = mean_s Phi((y - mu_s) / sigma_s)

    without drawing additional target samples. Component quantiles provide a
    guaranteed bracket for the mixture quantile.
    """
    mu = np.asarray(mu_samples, dtype=np.float64)
    sigma = np.asarray(sigma_samples, dtype=np.float64)
    if mu.shape != sigma.shape or mu.ndim < 2 or mu.shape[0] < 1:
        raise ValueError(
            "mu_samples and sigma_samples must have the same shape (M, ...), "
            f"got {mu.shape} and {sigma.shape}."
        )
    if not 0.0 < probability < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {probability}.")
    if bisection_steps < 1:
        raise ValueError(f"bisection_steps must be >= 1, got {bisection_steps}.")
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise ValueError("Gaussian-mixture parameters must all be finite.")
    if not (sigma > 0.0).all():
        raise ValueError("Every Gaussian-mixture sigma must be > 0.")

    component_z = NormalDist().inv_cdf(probability)
    component_quantiles = mu + component_z * sigma
    lower = torch.from_numpy(component_quantiles.min(axis=0))
    upper = torch.from_numpy(component_quantiles.max(axis=0))
    mu_t = torch.from_numpy(mu)
    sigma_t = torch.from_numpy(sigma)

    for _ in range(bisection_steps):
        midpoint = (lower + upper) * 0.5
        standardized = (midpoint.unsqueeze(0) - mu_t) / sigma_t
        mixture_cdf = (
            0.5 * (1.0 + torch.erf(standardized / np.sqrt(2.0)))
        ).mean(dim=0)
        move_lower = mixture_cdf < probability
        lower = torch.where(move_lower, midpoint, lower)
        upper = torch.where(move_lower, upper, midpoint)

    return ((lower + upper) * 0.5).numpy()


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
    z: float | None = None,
) -> pd.DataFrame:
    """SDE inference: `mc_samples` Brownian paths -> two-source predictive intervals.

    Each path yields a Gaussian PV head (mu_s, sigma_s). The predictive mean is
    E_s[mu_s]; the epistemic std is Std_s(mu_s); the aleatoric std is
    sqrt(E_s[sigma_s^2]); the predictive (total) std combines the two. The
    primary PI consists of equal-tail quantiles of the resulting equally
    weighted Gaussian mixture. The moment-matched Gaussian band is retained in
    ``lower_gaussian``/``upper_gaussian`` as a diagnostic. The epistemic and
    aleatoric components remain separate downstream.
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
    print(f"  [sde] stochastic inference: {mc_samples} Brownian paths (epistemic + aleatoric)")

    pv_scale = dataset.pv_scale[None, :]  # (1, N)
    loc_ids = dataset.loc_ids

    # Equal-tail probabilities for the requested coverage. ``z`` is retained as
    # a backward-compatible override of the nominal coverage.
    if z is None:
        z = NormalDist().inv_cdf(1.0 - (1.0 - coverage_target) / 2.0)
    elif not np.isfinite(z) or z <= 0.0:
        raise ValueError(f"z must be finite and > 0 when provided, got {z}.")
    lower_probability = NormalDist().cdf(-float(z))
    upper_probability = NormalDist().cdf(float(z))
    print(
        "  [sde] primary PI: equal-tail quantiles of the Gaussian mixture "
        f"({upper_probability - lower_probability:.1%} nominal coverage)"
    )

    locs, times, ytrue, solar_targets = [], [], [], []
    means, tot_stds, epi_stds, ale_stds = [], [], [], []
    lower_pis, upper_pis = [], []
    for x, _y, k in loader:
        x = x.to(device)
        k = k.numpy()
        B = len(k)
        # Per Brownian path: predictive mean mu_s and aleatoric std sigma_s
        # (physical units).
        mu_samples = np.empty((mc_samples, B, len(loc_ids)), dtype=np.float64)
        sigma_samples = np.empty((mc_samples, B, len(loc_ids)), dtype=np.float64)
        for s in range(mc_samples):
            _ghi, mu, sigma = model(x, ei, ew, None, stochastic=True)[:3]
            mu_samples[s] = mu.cpu().numpy() * pv_scale            # physical mean
            sigma_samples[s] = sigma.cpu().numpy() * pv_scale      # physical aleatoric
        mean = mu_samples.mean(axis=0)                             # (B, N) predictive mean
        epi_var = mu_samples.var(axis=0)                          # epistemic: Var_s(mu_s)
        ale_var = (sigma_samples ** 2).mean(axis=0)               # aleatoric: E_s[sigma_s^2]
        epi_std = np.sqrt(epi_var)
        ale_std = np.sqrt(ale_var)
        total_std = np.sqrt(epi_var + ale_var)                    # law of total variance
        lower_pi = _gaussian_mixture_quantile(
            mu_samples, sigma_samples, lower_probability
        )
        upper_pi = _gaussian_mixture_quantile(
            mu_samples, sigma_samples, upper_probability
        )
        y_true = dataset.y_true_all[k]  # (B, N) physical
        solar_target = dataset.solar_irradiance_poa_target_all[k]  # (B, N) W/m2
        ts = dataset.target_time_all[k].values  # (B,)
        N = mean.shape[1]
        locs.append(np.tile(loc_ids, B))
        times.append(np.repeat(ts, N))
        ytrue.append(y_true.reshape(-1))
        solar_targets.append(solar_target.reshape(-1))
        means.append(mean.reshape(-1))
        tot_stds.append(total_std.reshape(-1))
        epi_stds.append(epi_std.reshape(-1))
        ale_stds.append(ale_std.reshape(-1))
        lower_pis.append(lower_pi.reshape(-1))
        upper_pis.append(upper_pi.reshape(-1))

    y_true = np.concatenate(ytrue).astype(np.float64)
    solar_target = np.concatenate(solar_targets).astype(np.float64)
    y_mean = np.concatenate(means).astype(np.float64)
    y_std = np.concatenate(tot_stds).astype(np.float64)
    y_epi = np.concatenate(epi_stds).astype(np.float64)
    y_ale = np.concatenate(ale_stds).astype(np.float64)
    # Primary interval: exact equal-tail quantiles of the finite Gaussian
    # mixture represented by the sampled SDE paths.
    y_lower_pi = np.concatenate(lower_pis).astype(np.float64)
    y_upper_pi = np.concatenate(upper_pis).astype(np.float64)
    # Secondary diagnostic: single moment-matched Gaussian approximation.
    y_lower_gaussian = y_mean - z * y_std
    y_upper_gaussian = y_mean + z * y_std
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
            "epistemic_std": y_epi,
            "aleatoric_std": y_ale,
            "lower_pi": y_lower_pi,
            "upper_pi": y_upper_pi,
            "lower_gaussian": y_lower_gaussian,
            "upper_gaussian": y_upper_gaussian,
            "lower_raw": y_lower_pi,
            "upper_raw": y_upper_pi,
            "y_pred_lower": y_lower_pi,
            "y_pred_upper": y_upper_pi,
            "error": error,
            "abs_error": np.abs(error),
            "squared_error": error ** 2,
        }
    )
