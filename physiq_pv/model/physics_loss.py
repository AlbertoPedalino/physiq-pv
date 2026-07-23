from __future__ import annotations

import torch


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def physics_loss_full(
    pred_poa: torch.Tensor,
    pred_pv: torch.Tensor,
    true_poa: torch.Tensor,
    true_pv: torch.Tensor,
    pr_proxy: torch.Tensor,
    poa_scale: torch.Tensor,
    lam: float = 0.1,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Combine POA, PV and dimensionally consistent normalized-physics losses.

    ``pred_pv`` is normalized by each plant's training-only PV p99. Therefore
    POA must also be normalized by its training-only p99 before applying the
    fitted performance-ratio proxy:

        pred_pv_norm ~= pr_proxy * (pred_poa / poa_scale)
    """
    if sample_weight is None:
        sample_weight = torch.ones_like(true_pv)

    poa_scale = poa_scale.to(
        device=pred_poa.device, dtype=pred_poa.dtype
    ).clamp_min(1e-6)
    pred_poa_norm = pred_poa / poa_scale

    l_poa = _weighted_mean((pred_poa - true_poa).pow(2), sample_weight)
    l_pv = _weighted_mean((pred_pv - true_pv).pow(2), sample_weight)
    l_physics = _weighted_mean(
        (pred_pv - pr_proxy * pred_poa_norm).pow(2),
        sample_weight,
    )

    total = l_poa + l_pv + lam * l_physics
    return total, {
        "l_poa": l_poa.item(),
        "l_pv": l_pv.item(),
        "l_physics": l_physics.item(),
    }
