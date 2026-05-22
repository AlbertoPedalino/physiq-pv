import torch
import torch.nn as nn
import torch.nn.functional as F

from physiq_pv.continual.replay_buffer import ReplayBuffer


class QualityGatedUpdater:
    """
    Quality-weighted continual learning update (DER++ adapted for regression).

    Two distinct quality signals are consumed; they are NOT interchangeable:

      * UPDATE-GATE signal --- ``suspicion_mean`` (fleet-level, in [0, 1]).
        Comes from qs_forensics. Drives the soft Bernoulli gate:
            p_update = 1 - suspicion_mean
        High suspicion => the update is likely to be skipped. Implements
        the *Action* decision: "should we trust this update?".

      * MEMORY-WEIGHTING signal --- ``qs_per_sample`` (per-(batch, node),
        in [0, 1]). Comes from feature channels 5..9 (m1..m5). Drives
        QS-weighted replay sampling inside the buffer. Implements the
        *Memory* prioritisation: "which past samples should we revisit?".

    Legacy hard floor: ``qs_threshold`` (scalar, default None). If the
    positional ``qs_mean`` argument falls below this threshold, the update
    is skipped regardless of suspicion. Kept for backward compatibility.

    Gate chain (in order):
      1. legacy qs_threshold hard floor (skip if qs_mean <= threshold);
      2. soft suspicion Bernoulli gate;
      3. otherwise apply DER++ update.

    The sample is always added to the replay buffer before the gate fires
    so that *memory grows even when the gradient step is skipped*.

    DER++ terms (Buzzega et al., NeurIPS 2020; default alpha=0.2, beta=1.0):
      * alpha: MSE between current model output on replayed samples and the
        stored old predictions (representation stabilisation).
      * beta:  MSE between current model output on replayed samples and
        the stored ground-truth targets (task retention).

    Known limit (G6, see docs/CL_COMPONENTS.md): the replay path currently
    distils only the PV head. The GHI head is NOT included in the buffer or
    in the replay loss. Extending to dual-head distillation is tracked as
    future work; offline training uses the dual-head loss unchanged.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        buffer: ReplayBuffer,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        qs_threshold: float | None = None,
        alpha_der: float = 0.2,
        beta_der: float = 1.0,
        replay_batch: int = 32,
        max_grad_norm: float = 1.0,
        rng: torch.Generator | None = None,
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
        self.rng = rng if rng is not None else torch.Generator()

    def step(
        self,
        x: torch.Tensor,        # (B, N, seq_len, C)
        y_pv: torch.Tensor,     # (B, N)
        pred_pv: torch.Tensor,  # (B, N) - current prediction (detached)
        loss: torch.Tensor,     # scalar, computed externally
        qs_mean: float,
        suspicion_mean: float | None = None,
        qs_per_sample: torch.Tensor | None = None,  # (B,) or (B, N)
    ) -> bool:
        """
        Store sample and run DER++ update unless gating blocks it.

        Gating chain:
          1. qs_threshold hard floor (legacy) -> skip if qs_mean <= threshold.
          2. soft suspicion gate -> Bernoulli skip with prob = suspicion_mean.
          3. otherwise update.

        qs_per_sample (optional) feeds the replay buffer for QS-weighted
        sampling. If omitted, qs_mean is broadcast to all batch elements.

        Returns True if weights were updated.
        """
        buf_qs = qs_per_sample if qs_per_sample is not None else qs_mean
        self.buffer.add_batch(x, y_pv, pred_pv, qs=buf_qs)

        if self.qs_threshold is not None and qs_mean <= self.qs_threshold:
            return False

        if suspicion_mean is not None and 0.0 <= suspicion_mean <= 1.0:
            draw = torch.rand((), generator=self.rng).item()
            if draw < float(suspicion_mean):
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
