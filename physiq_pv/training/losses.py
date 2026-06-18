"""Training point-losses and uncertainty penalties for the PVGIS ST-GNN run.

Pure tensor functions extracted from the dataset module: the Huber/MSE point
loss, the under-dispersion penalty and the SDE-proxy OOD penalty.
"""
from __future__ import annotations

import numpy as np
import torch


LOSS_TYPES = ("mse", "huber")


def make_loss_fn(loss_type: str = "mse", huber_delta: float = 1.0) -> torch.nn.Module:
    """Build the training point-loss module.

    loss_type="mse"   -> torch.nn.MSELoss() (historical default; bit-identical run).
    loss_type="huber" -> torch.nn.HuberLoss(delta=huber_delta): quadratic for
                         |err| <= delta, linear beyond, so large residuals get a
                         smaller gradient than under MSE (more robust on extremes).
    The same module supervises pred_pv and, when enabled, the auxiliary kt head.
    NOTE: the PV target is normalised (scaled by p99 daytime energy), so residuals
    are typically << 1; a delta near the residual scale (~0.1) is where Huber
    actually departs from MSE. delta=1.0 is the API default and behaves close to
    MSE on this target.
    """
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {LOSS_TYPES}, got {loss_type!r}.")
    if loss_type == "huber":
        if not np.isfinite(huber_delta) or huber_delta <= 0.0:
            raise ValueError(f"huber_delta must be finite and > 0, got {huber_delta}.")
        return torch.nn.HuberLoss(delta=float(huber_delta))
    return torch.nn.MSELoss()


def under_dispersion_penalty(
    y_true: torch.Tensor,
    y_pred_mean: torch.Tensor,
    y_pred_std: torch.Tensor,
    k: float,
) -> torch.Tensor:
    """Elementwise under-dispersion penalty relu(|y - mean| - k*std)^2 (no reduction).

    Large when the absolute error exceeds k standard deviations (high error +
    low MC std -> over-confident); ~0 once the MC std is wide enough to cover the
    error. Pure function of the mean/std tensors; no anomaly labels involved.
    """
    err = (y_true - y_pred_mean).abs()
    return torch.relu(err - k * y_pred_std) ** 2


def sde_proxy_penalty(
    y_pred_std: torch.Tensor,
    anomaly_mask: torch.Tensor,
    std_min_ood: float,
) -> tuple:
    """SDE-Net-style proxy losses on the MC std (the diffusion proxy).

    Returns (in_loss, out_loss) scalars:
        in_loss  = mean(std[normal]^2)                       minimise in-dist std
        out_loss = mean(relu(std_min_ood - std[anomaly])^2)  keep OOD std >= floor
    `anomaly_mask` True marks OOD/anomalous (rare_or_extreme) cells; the rest are
    in-distribution. Empty subsets contribute 0. No anomaly label is used as a
    model input/target — only to split in-distribution vs OOD here.
    """
    mask = anomaly_mask.to(torch.bool)
    normal_m = ~mask
    std2 = y_pred_std ** 2
    in_loss = (
        std2[normal_m].mean() if bool(normal_m.any()) else y_pred_std.new_zeros(())
    )
    relu_out = torch.relu(std_min_ood - y_pred_std) ** 2
    out_loss = (
        relu_out[mask].mean() if bool(mask.any()) else y_pred_std.new_zeros(())
    )
    return in_loss, out_loss


