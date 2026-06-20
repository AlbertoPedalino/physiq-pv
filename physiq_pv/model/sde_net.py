"""Paper-faithful vector components of SDE-Net (Kong, Sun & Zhang, 2020).

The authors' public implementation has three important conventions which are
easy to lose when adapting SDE-Net to a new backbone:

* the diffusion network returns one *scalar per input example* (not one value
  per latent channel or node);
* that scalar is fixed at the initial state and multiplies every Brownian
  increment in the Euler--Maruyama trajectory; and
* diffusion is trained separately from the prediction path. The paper writes a
  difference of ID/OOD diffusion scores; the authors' public code implements
  that separation with an ID/pseudo-OOD BCE discriminator.

``PaperSDEBlock`` preserves those dynamics for a vector or a batched
node-by-feature latent tensor. ``YearMSDSDENet`` is a direct, dependency-free
reimplementation of the regression architecture in the authors' repository.
It is included both as an executable reference and as a regression baseline
for adaptations such as the PV ST-GNN.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorDrift(nn.Module):
    """The YearMSD drift network from the authors' implementation.

    Its public ``forward(t, x)`` interface follows the neural-SDE notation.
    The vector experiment does not use ``t`` internally; image experiments use
    a time-concatenated convolution instead.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, t: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return self.relu(self.fc(x))


class ScalarDiffusion(nn.Module):
    """SDE-Net's scalar sigmoid diffusion discriminator.

    For a ``(B, D)`` latent vector this is exactly the YearMSD diffusion MLP:
    ``Linear(D, 2D) -> ReLU -> Linear(2D, 1) -> Sigmoid``.  PV latents have a
    node axis (``B, N, D``); their node summary is mean-pooled before the same
    MLP.  This mirrors the spatial pooling used by the authors' image models
    while retaining the paper's one-diffusion-value-per-example invariant.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.fc2 = nn.Linear(2 * hidden_dim, 1)

    @staticmethod
    def _example_summary(x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f"SDE latent must have batch and feature axes, got {x.shape}.")
        if x.ndim == 2:
            return x
        # (B, ..., D) -> (B, D); all non-feature axes belong to one example.
        return x.flatten(start_dim=1, end_dim=-2).mean(dim=1)

    def forward(self, t: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        summary = self._example_summary(x)
        return torch.sigmoid(self.fc2(self.relu(self.fc1(summary))))


class PaperSDEBlock(nn.Module):
    """Euler--Maruyama SDE block with the conventions of SDE-Net.

    ``sigma`` is the paper's maximum Brownian multiplier: ``ScalarDiffusion``
    is a sigmoid in ``[0, 1]``, so ``sigma * g`` lies in ``[0, sigma]``. The
    original experiments lower it only during early training for stability,
    then raise it to the experiment value (0.5 for YearMSD).
    """

    def __init__(
        self,
        dim: int,
        n_steps: int = 4,
        sigma: float = 0.5,
        time_horizon: float = 4.0,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}.")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"sigma must be finite and > 0, got {sigma}.")
        if not math.isfinite(time_horizon) or time_horizon <= 0.0:
            raise ValueError(
                f"time_horizon must be finite and > 0, got {time_horizon}."
            )
        self.dim = dim
        self.n_steps = n_steps
        self.time_horizon = float(time_horizon)
        self.deltat = self.time_horizon / self.n_steps
        self.sigma = float(sigma)
        self.drift = VectorDrift(dim)
        self.diffusion_net = ScalarDiffusion(dim)

    def diffusion(self, x0: torch.Tensor) -> torch.Tensor:
        """Return the paper's unscaled discriminator probability, shape ``(B, 1)``."""
        return self.diffusion_net(0.0, x0)

    def diffusion_scale(self, x0: torch.Tensor) -> torch.Tensor:
        """Return ``sigma * g(x0)`` (one Brownian scale per example)."""
        return self.sigma * self.diffusion(x0)

    def forward(
        self, x0: torch.Tensor, *, stochastic: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate from ``x0`` and return ``(x_T, sigma * g(x0))``.

        SDE-Net evaluates uncertainty with ``stochastic=True`` and averages
        multiple Brownian paths. ``stochastic=False`` is retained only as a
        deterministic drift diagnostic for the surrounding forecasting code.
        """
        if x0.ndim < 2 or x0.shape[-1] != self.dim:
            raise ValueError(
                f"Expected latent (..., {self.dim}) with a batch axis, got {tuple(x0.shape)}."
            )
        diffusion_scale = self.diffusion_scale(x0)
        # (B, 1) -> (B, 1, ..., 1), broadcasting over every state coordinate.
        scale = diffusion_scale.reshape(x0.shape[0], *([1] * (x0.ndim - 1)))
        out = x0
        for i in range(self.n_steps):
            t = self.time_horizon * float(i) / self.n_steps
            out = out + self.drift(t, out) * self.deltat
            if stochastic:
                out = out + scale * math.sqrt(self.deltat) * torch.randn_like(out)
        return out, diffusion_scale


class YearMSDSDENet(nn.Module):
    """Direct reimplementation of ``YearMSD/models/sdenet.py`` from SDE-Net.

    Defaults reproduce the authors' regression experiment: 90 inputs, 50
    latent units, four Euler--Maruyama steps over ``[0, 4]``, and ``sigma=0.5``.
    ``training_diffusion=True`` exposes the detached diffusion discriminator,
    exactly as used by their alternating BCE optimisation step.
    """

    def __init__(
        self,
        input_dim: int = 90,
        hidden_dim: int = 50,
        layer_depth: int = 4,
        sigma: float = 0.5,
        time_horizon: float = 4.0,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("input_dim and hidden_dim must both be >= 1.")
        if layer_depth < 1:
            raise ValueError("layer_depth must be >= 1.")
        self.layer_depth = layer_depth
        self.downsampling_layers = nn.Linear(input_dim, hidden_dim)
        self.drift = VectorDrift(hidden_dim)
        self.diffusion = ScalarDiffusion(hidden_dim)
        self.fc_layers = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(hidden_dim, 2))
        self.deltat = float(time_horizon) / layer_depth
        self.time_horizon = float(time_horizon)
        self.sigma = float(sigma)

    def forward(
        self, x: torch.Tensor, *, training_diffusion: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = self.downsampling_layers(x)
        if training_diffusion:
            return self.diffusion(0.0, out.detach())

        diffusion_term = self.sigma * self.diffusion(0.0, out)
        for i in range(self.layer_depth):
            t = self.time_horizon * float(i) / self.layer_depth
            out = out + self.drift(t, out) * self.deltat
            out = out + diffusion_term * math.sqrt(self.deltat) * torch.randn_like(out)
        final_out = self.fc_layers(out)
        mean = final_out[:, 0]
        sigma = F.softplus(final_out[:, 1]) + 1e-3
        return mean, sigma


def diffusion_bce_loss(g_in: torch.Tensor, g_ood: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BCE form used by Kong et al.'s public SDE-Net training scripts.

    The paper's Algorithm 1 presents the equivalent intent as minimising the
    in-distribution diffusion score and maximising the pseudo-OOD score.
    """
    loss_in = F.binary_cross_entropy(g_in, torch.zeros_like(g_in))
    loss_ood = F.binary_cross_entropy(g_ood, torch.ones_like(g_ood))
    return loss_in + loss_ood, loss_in, loss_ood


def yearmsd_nll_loss(
    target: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """The heteroscedastic regression loss used by the authors' YearMSD script."""
    return torch.mean(torch.log(sigma ** 2) + (target - mean) ** 2 / (sigma ** 2))
