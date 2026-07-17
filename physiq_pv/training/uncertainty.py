"""SDE-Net inference: deterministic drift mean + stochastic predictive intervals.

Uncertainty has the two sources of Kong et al. (2020): running the model M
times with ``stochastic=True`` samples M Brownian paths, each yielding an
aleatoric head (mu_s, sigma_s). The *epistemic* uncertainty is
Var_s(mu_s). For a Gaussian head the *aleatoric* variance is E_s[sigma_s^2];
for a location-scale Student-t it is
E_s[sigma_s^2 * nu / (nu - 2)]. Their sum is the total variance by the law of
total variance.

A mixture across Brownian paths is generally not a single Gaussian or
Student-t. Student-t predictive intervals are therefore empirical quantiles of
samples from the full Brownian-path/aleatoric mixture. No dropout is involved.
"""
from __future__ import annotations

from statistics import NormalDist

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
    z: float | None = None,
    nll_dist: str = "student_t",
    student_t_nu: float = 5.0,
    student_t_samples_per_path: int = 64,
) -> pd.DataFrame:
    """SDE inference: `mc_samples` Brownian paths -> two-source predictive intervals.

    Each path yields an aleatoric head (mu_s, sigma_s). For Student-t,
    ``sigma_s`` is a scale rather than a standard deviation. The reported
    variance decomposition is analytic, while the predictive interval uses
    empirical quantiles from ``mc_samples * student_t_samples_per_path`` draws
    from the resulting mixture. Gaussian mode retains the historical
    moment-matched band ``mean +/- z * total_std``.
    """
    if mc_samples < 2:
        raise ValueError(f"mc_samples must be >= 2 for SDE sampling, got {mc_samples}.")
    if not 0.0 < coverage_target < 1.0:
        raise ValueError(f"coverage_target must be in (0, 1), got {coverage_target}.")
    if nll_dist not in {"gaussian", "student_t"}:
        raise ValueError(
            "nll_dist must be 'gaussian' or 'student_t', "
            f"got {nll_dist!r}."
        )
    if nll_dist == "student_t":
        if not np.isfinite(student_t_nu) or student_t_nu <= 2.0:
            raise ValueError(
                "student_t_nu must be finite and > 2 for finite predictive "
                f"variance, got {student_t_nu}."
            )
        if student_t_samples_per_path < 1:
            raise ValueError(
                "student_t_samples_per_path must be >= 1, "
                f"got {student_t_samples_per_path}."
            )
        if z is not None:
            raise ValueError(
                "An explicit z is only supported for Gaussian inference; "
                "Student-t intervals are empirical mixture quantiles."
            )
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

    q = 1.0 - (1.0 - coverage_target) / 2.0
    gaussian_z = NormalDist().inv_cdf(q)
    if nll_dist == "gaussian" and z is None:
        z = gaussian_z
    if nll_dist == "student_t":
        print(
            f"  [sde] Student-t(nu={student_t_nu:g}) PI: empirical mixture "
            f"quantiles from {mc_samples * student_t_samples_per_path} draws "
            "per prediction"
        )
    else:
        print(
            f"  [sde] Gaussian PI: mean +/- {z:.3f} * total_std "
            "(total = epistemic + aleatoric)"
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
        mixture_samples = (
            np.empty(
                (
                    mc_samples * student_t_samples_per_path,
                    B,
                    len(loc_ids),
                ),
                dtype=np.float64,
            )
            if nll_dist == "student_t"
            else None
        )
        for s in range(mc_samples):
            _ghi, mu, sigma = model(x, ei, ew, None, stochastic=True)[:3]
            mu_samples[s] = mu.cpu().numpy() * pv_scale            # physical mean
            sigma_samples[s] = sigma.cpu().numpy() * pv_scale      # physical aleatoric
            if mixture_samples is not None:
                conditional = torch.distributions.StudentT(
                    df=float(student_t_nu), loc=mu, scale=sigma
                )
                lo = s * student_t_samples_per_path
                hi = lo + student_t_samples_per_path
                mixture_samples[lo:hi] = (
                    conditional.sample((student_t_samples_per_path,))
                    .cpu()
                    .numpy()
                    * pv_scale[None, :, :]
                )
        mean = mu_samples.mean(axis=0)                             # (B, N) predictive mean
        epi_var = mu_samples.var(axis=0)                          # epistemic: Var_s(mu_s)
        ale_var = (sigma_samples ** 2).mean(axis=0)
        if nll_dist == "student_t":
            ale_var *= student_t_nu / (student_t_nu - 2.0)
        epi_std = np.sqrt(epi_var)
        ale_std = np.sqrt(ale_var)
        total_std = np.sqrt(epi_var + ale_var)                    # law of total variance
        if mixture_samples is not None:
            alpha = 1.0 - coverage_target
            lower_pi = np.quantile(mixture_samples, alpha / 2.0, axis=0)
            upper_pi = np.quantile(mixture_samples, 1.0 - alpha / 2.0, axis=0)
        else:
            lower_pi = mean - float(z) * total_std
            upper_pi = mean + float(z) * total_std
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
    y_lower_pi = np.concatenate(lower_pis).astype(np.float64)
    y_upper_pi = np.concatenate(upper_pis).astype(np.float64)
    y_lower_gaussian = y_mean - gaussian_z * y_std
    y_upper_gaussian = y_mean + gaussian_z * y_std
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
