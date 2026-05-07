import torch


def physics_loss_full(
    pred_ghi: torch.Tensor,   # (B, N)
    pred_pv: torch.Tensor,    # (B, N)
    true_ghi: torch.Tensor,   # (B, N)
    true_pv: torch.Tensor,    # (B, N)
    eta_T: torch.Tensor,      # (B, N) nominal thermal efficiency
    lam: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    L = L_ghi + L_pv + lam * L_physics.

    Multiplicative formulation (avoids division-by-zero at night):
        L_physics = MSE(pred_pv, eta_T * pred_ghi)

    Equivalent constraint to pred_pv / pred_ghi ~= eta_T but well-defined
    when pred_ghi -> 0 (night, ghi_cs = 0 under clear-sky parametrization).

    Returns (total_loss, {l_ghi, l_pv, l_physics}).
    """
    l_ghi = (pred_ghi - true_ghi).pow(2).mean()
    l_pv = (pred_pv - true_pv).pow(2).mean()

    l_physics = (pred_pv - eta_T * pred_ghi).pow(2).mean()

    total = l_ghi + l_pv + lam * l_physics
    return total, {
        "l_ghi": l_ghi.item(),
        "l_pv": l_pv.item(),
        "l_physics": l_physics.item(),
    }
