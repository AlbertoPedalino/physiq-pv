"""Training losses for the PVGIS ST-GNN run.

The PV prediction path is trained with the heteroscedastic Gaussian negative
log-likelihood of SDE-Net's regression experiment (Kong et al., 2020, supp.
S.4.2): ``log(sigma^2) + (y - mean)^2 / sigma^2``.  This is the drift net's
aleatoric-uncertainty objective; the auxiliary irradiance (kt) head keeps a
plain MSE.  Both are separate from SDE-Net's BCE diffusion objective, which
lives in the training loop.
"""
from __future__ import annotations

import torch


def gaussian_nll(
    target: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """Element-wise heteroscedastic Gaussian NLL (Kong et al. regression form).

    Returns ``log(sigma^2) + (target - mean)^2 / sigma^2`` with no reduction, so
    callers can apply their own averaging. ``sigma`` must be strictly positive
    (the PV head adds ``+1e-3``). Equivalent up to an additive constant and
    factor of two to the Gaussian NLL; matches ``yearmsd_nll_loss``.
    """
    return torch.log(sigma ** 2) + (target - mean) ** 2 / (sigma ** 2)


def make_loss_fn(reduction: str = "mean") -> torch.nn.Module:
    """Build the auxiliary point-loss module: torch.nn.MSELoss().

    Used for the optional irradiance (kt) head only. The PV prediction path uses
    ``gaussian_nll`` instead; neither replaces SDE-Net's BCE diffusion objective.
    """
    return torch.nn.MSELoss(reduction=reduction)
