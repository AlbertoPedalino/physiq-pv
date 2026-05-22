import numpy as np
import torch


# Per-sample probability floor for QS-weighted sampling.
#
# Rationale: probabilities are computed as p_i ~ max(qs_i, _QS_FLOOR) / sum.
# Without a floor a sample stored with qs=0 (e.g. night artefact or sensor
# failure) would have exactly zero probability of being resampled, which
# permanently silences it. Such a sample still carries information about
# what "broken" looks like and we want it to be reachable, just rarely.
# 1e-3 puts ~3 orders of magnitude between a fully-trusted sample (qs=1)
# and a fully-distrusted one — large enough for clear prioritisation,
# small enough not to dominate when the buffer is mixed-quality.
_QS_FLOOR = 1e-3


class ReplayBuffer:
    """
    DER++ replay buffer for regression tasks with QS-weighted sampling.

    Stores (x, y, pred, qs) tuples where:
      - x:    input features at storage time (shape (N, seq_len, C));
      - y:    ground-truth target (PV-only, see TODO below);
      - pred: model output at storage time (DER++ distillation target);
      - qs:   per-sample aggregate Quality Score in [0, 1], 1 = high.

    Sampling is non-uniform: probability proportional to qs (floored by
    ``qs_floor``, default ``_QS_FLOOR``, so zero-QS samples are not
    permanently silenced). Falls back to uniform when no QS info has been
    provided.

    TODO (G6 — dual-head replay): extend to (x, y_ghi, y_pv, pred_ghi_old,
    pred_pv_old) so that DER++ can distil the GHI head as well. This is a
    non-trivial change because it affects every ``add_batch`` call site
    (train.py, online_loop._retrain_window via the updater) and the
    QualityGatedUpdater loss assembly. Tracked as future work in
    docs/CL_COMPONENTS.md. The buffer currently protects only the PV head.

    Reference: aimagelab/mammoth derpp.py
    """

    def __init__(
        self,
        capacity: int = 1000,
        rng: "np.random.Generator | None" = None,
        qs_floor: float = _QS_FLOOR,
    ):
        self.capacity = capacity
        self._buf: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = []
        self._ptr = 0
        # Inject a numpy Generator for deterministic sampling. Falls back to a
        # default Generator (system entropy) for backward compatibility.
        self._rng = rng if rng is not None else np.random.default_rng()
        # Per-instance probability floor; defaults to module-level _QS_FLOOR
        # so existing callers see unchanged behaviour.
        self.qs_floor = float(qs_floor)

    def add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pred: torch.Tensor,
        qs: float = 1.0,
    ) -> None:
        """Store a single sample (no batch dim)."""
        item = (
            x.detach().cpu(),
            y.detach().cpu(),
            pred.detach().cpu(),
            float(qs),
        )
        if len(self._buf) < self.capacity:
            self._buf.append(item)
        else:
            self._buf[self._ptr % self.capacity] = item
        self._ptr += 1

    def add_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pred: torch.Tensor,
        qs: torch.Tensor | np.ndarray | float | None = None,
    ) -> None:
        """
        Store each element of a batch individually.

        qs may be:
          - None  -> all samples stored with qs=1.0 (legacy uniform fallback).
          - scalar (float / 0-d tensor) -> broadcast to every element.
          - 1-d (B,) -> one QS per batch element, broadcast across nodes.
          - 2-d (B, N) -> one QS per (batch, node).
        """
        B = x.shape[0]
        if qs is None:
            qs_arr = np.ones(B, dtype=np.float32)
        elif isinstance(qs, torch.Tensor):
            qs_arr = qs.detach().cpu().numpy()
        else:
            qs_arr = np.asarray(qs)

        for i in range(B):
            qs_i = qs_arr if np.isscalar(qs_arr) else qs_arr[i] if qs_arr.ndim >= 1 else qs_arr
            qs_scalar = float(np.nanmean(qs_i)) if hasattr(qs_i, "__len__") else float(qs_i)
            if not np.isfinite(qs_scalar):
                qs_scalar = 1.0
            self.add(x[i], y[i], pred[i], qs=qs_scalar)

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample n items with probability proportional to stored QS (floored).
        Returns (x, y, pred) stacked tensors.
        """
        n = min(n, len(self._buf))
        if n == 0:
            raise ValueError("buffer is empty")

        qs_arr = np.array([item[3] for item in self._buf], dtype=np.float64)
        qs_arr = np.clip(qs_arr, self.qs_floor, None)
        total = qs_arr.sum()
        if total > 0:
            p = qs_arr / total
            idx = self._rng.choice(len(self._buf), size=n, replace=False, p=p)
        else:
            idx = self._rng.choice(len(self._buf), size=n, replace=False)

        xs, ys, ps, _ = zip(*[self._buf[i] for i in idx])
        return torch.stack(xs), torch.stack(ys), torch.stack(ps)

    def mean_qs(self) -> float:
        if not self._buf:
            return float("nan")
        return float(np.mean([item[3] for item in self._buf]))

    def __len__(self) -> int:
        return len(self._buf)
