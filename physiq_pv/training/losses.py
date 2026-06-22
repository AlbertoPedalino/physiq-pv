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
    target: torch.Tensor,
    mean: torch.Tensor,
    sigma: torch.Tensor,
    beta: float = 0.0,
) -> torch.Tensor:
    """Element-wise heteroscedastic Gaussian NLL (Kong et al. regression form):
    ``log(sigma^2) + (target - mean)^2 / sigma^2`` (no reduction; ``sigma > 0``).

    ``beta`` adds the beta-NLL weighting of Seitzer et al. (2022): each element
    is scaled by ``stopgrad(sigma)^(2*beta)``. ``beta=0`` is the plain NLL;
    ``beta=0.5`` restores MSE-like gradients on the mean (no variance runaway).
    """
    nll = torch.log(sigma ** 2) + (target - mean) ** 2 / (sigma ** 2)
    if beta > 0.0:
        nll = sigma.detach() ** (2.0 * beta) * nll
    return nll


def make_loss_fn(reduction: str = "mean") -> torch.nn.Module:
    """Build the auxiliary point-loss module: torch.nn.MSELoss().

    Used for the optional irradiance (kt) head only. The PV prediction path uses
    ``gaussian_nll`` instead; neither replaces SDE-Net's BCE diffusion objective.
    """
    return torch.nn.MSELoss(reduction=reduction)
