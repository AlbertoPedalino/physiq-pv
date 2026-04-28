import torch
import torch.nn as nn
import torch.nn.functional as F

from physiq_pv.continual.replay_buffer import ReplayBuffer


class QualityGatedUpdater:
    """
    Quality-gated continual learning update (DER++ adapted for regression).

    Policy:
      - Always store current sample in replay buffer.
      - Weight update only when mean QS of current batch > qs_threshold.
      - DER++ alpha term: MSE between current model output on replayed samples
        and the stored old predictions (representation stabilisation).
      - DER++ beta term: MSE between current model output on replayed samples
        and the stored ground-truth targets (task retention).

    Reference: aimagelab/mammoth (DER++)
    Hyperparams: alpha=0.2, beta=1.0 per DER++ ablation (Buzzega et al. 2020)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        buffer: ReplayBuffer,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        qs_threshold: float = 0.5,
        alpha_der: float = 0.2,
        beta_der: float = 1.0,
        replay_batch: int = 32,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.buffer = buffer
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.qs_threshold = qs_threshold
        self.alpha = alpha_der
        self.beta = beta_der
        self.replay_batch = replay_batch
        self.max_grad_norm = max_grad_norm

    def step(
        self,
        x: torch.Tensor,        # (B, N, seq_len, C)
        y_pv: torch.Tensor,     # (B, N)
        pred_pv: torch.Tensor,  # (B, N) — current prediction (detached)
        loss: torch.Tensor,     # scalar, computed externally
        qs_mean: float,
    ) -> bool:
        """
        Store sample. If QS > threshold, run DER++ update.
        Returns True if weights were updated.
        """
        self.buffer.add_batch(x, y_pv, pred_pv)

        if qs_mean <= self.qs_threshold:
            return False

        total_loss = loss

        if len(self.buffer) >= self.replay_batch:
            rx, ry, r_old_pred = self.buffer.sample(self.replay_batch)
            device = next(self.model.parameters()).device
            rx        = rx.to(device)
            ry        = ry.to(device)
            r_old_pred = r_old_pred.to(device)
            ei = self.edge_index.to(device)
            ew = self.edge_weight.to(device)

            _, r_curr_pv = self.model(rx, ei, ew)
            # alpha: match stored (old) model predictions
            der_alpha = F.mse_loss(r_curr_pv, r_old_pred.detach())
            # beta: match stored ground-truth targets
            der_beta  = F.mse_loss(r_curr_pv, ry.detach())
            total_loss = total_loss + self.alpha * der_alpha + self.beta * der_beta

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return True
