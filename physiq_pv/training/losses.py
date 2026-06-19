"""Training point-loss for the PVGIS ST-GNN run.

Monaco et al. (2025), "SDE U-Net" — the SDE-Net adaptation we follow — trains the
drift on a plain MSE point loss; the epistemic band is the spread of the
stochastic (Brownian) samples, not a learned variance head. So this module only
builds the MSE point loss on the drift prediction. The SDE-Net diffusion
objective (low g in-distribution, high g out-of-distribution) lives in the
training loop.
"""
from __future__ import annotations

import torch


def make_loss_fn(reduction: str = "mean") -> torch.nn.Module:
    """Build the training point-loss module: torch.nn.MSELoss().

    Monaco et al. (2025) use MSE on the SDE terminal state ("we opted for a MSE
    as loss function to emphasize larger errors"). The same module supervises
    pred_pv and, when enabled, the auxiliary kt head.
    """
    return torch.nn.MSELoss(reduction=reduction)
