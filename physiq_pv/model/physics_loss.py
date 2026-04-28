import torch

_EPS = 1e-6


def physics_loss_full(
    pred_ghi: torch.Tensor,   # (B, N)
    pred_pv: torch.Tensor,    # (B, N)
    true_ghi: torch.Tensor,   # (B, N)
    true_pv: torch.Tensor,    # (B, N)
    eta_T: torch.Tensor,      # (B, N) nominal thermal efficiency
    qs: torch.Tensor,         # (B, N) quality score
    lam: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    L = L_ghi + L_pv + λ * L_physics

    L_physics = MSE(pred_pv / (|pred_ghi| + eps), eta_T)
    sample_weight = QS^0.2   (down-weights low-quality samples without zeroing them)

    Returns (total_loss, {l_ghi, l_pv, l_physics}).
    """
    weight = qs.pow(0.2).detach()  # (B, N), no grad through QS

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
