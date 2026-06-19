"""Training point-loss for the PVGIS ST-GNN run.

The SDE-Net diffusion objective (low g in-distribution, high g out-of-distribution)
lives in the training loop; this module only builds the Huber/MSE point loss on the
drift prediction.
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


def make_aleatoric_loss() -> torch.nn.Module:
    """Gaussian negative log-likelihood for the heteroscedastic PV head.

    Kong et al. (2020) model the drift net's regression output as a Gaussian
    N(mu, sigma^2): mu is the prediction, sigma^2 the *aleatoric* (data) noise.
    Training minimises the Gaussian NLL
        0.5 * [ (y - mu)^2 / sigma^2 + log sigma^2 ]
    so the head learns a per-sample variance instead of MSE's fixed noise. The
    caller passes var = exp(logvar); `eps` floors it for numerical stability.
    Replaces the MSE/Huber point loss on pred_pv when use_aleatoric is on.
    """
    return torch.nn.GaussianNLLLoss(full=False, eps=1e-6, reduction="mean")

