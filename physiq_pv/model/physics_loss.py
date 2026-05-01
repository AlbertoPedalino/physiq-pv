import torch

_EPS = 1e-6


def quality_weight(
    qs: torch.Tensor,
    qs_weight_exponent: float = 0.5,
    qs_weight_floor: float = 0.2,
) -> torch.Tensor:
    """Soft QS weight in [floor, 1]; low-QS samples are not gated out."""
    qs_clamped = qs.clamp(0.0, 1.0)
    weight = qs_weight_floor + (1.0 - qs_weight_floor) * qs_clamped.pow(qs_weight_exponent)
    return weight.detach()


def physics_loss_full(
    pred_ghi: torch.Tensor,   # (B, N)
    pred_pv: torch.Tensor,    # (B, N)
    true_ghi: torch.Tensor,   # (B, N)
    true_pv: torch.Tensor,    # (B, N)
    eta_T: torch.Tensor,      # (B, N) nominal thermal efficiency
    qs: torch.Tensor,         # (B, N) quality score
    lam: float = 0.1,
    qs_weight_exponent: float = 0.5,
    qs_weight_floor: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    L = L_ghi + L_pv + lam * L_physics.

    L_physics = MSE(pred_pv / (|pred_ghi| + eps), eta_T).
    sample_weight = floor + (1 - floor) * QS^exponent.
    Low-QS samples are down-weighted, not gated out.

    Returns (total_loss, {l_ghi, l_pv, l_physics}).
    """
    weight = quality_weight(qs, qs_weight_exponent, qs_weight_floor)

    l_ghi = (weight * (pred_ghi - true_ghi).pow(2)).mean()
    l_pv = (weight * (pred_pv - true_pv).pow(2)).mean()

    pred_eta = pred_pv / (pred_ghi.abs() + _EPS)
    l_physics = (weight * (pred_eta - eta_T).pow(2)).mean()

    total = l_ghi + l_pv + lam * l_physics
    return total, {
        "l_ghi": l_ghi.item(),
        "l_pv": l_pv.item(),
        "l_physics": l_physics.item(),
    }
