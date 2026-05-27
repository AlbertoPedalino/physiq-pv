import numpy as np
import torch


class SimpleReplayBuffer:
    """
    FIFO ring buffer with uniform random sampling for replay-based
    continual adaptation in PV forecasting.

    Each element stores the full sample tuple from PVDataset:
    (x, y_ghi, y_pv, eta, ghi_cs).

    Design choice: FIFO ring buffer over reservoir sampling because temporal
    locality matters for PV forecasting — evicting the oldest samples first
    is a reasonable default when the data distribution shifts over time.
    """

    def __init__(self, capacity: int = 5000, seed: int = 42):
        self.capacity = capacity
        self._buf: list[tuple[torch.Tensor, ...]] = []
        self._ptr = 0
        self._rng = np.random.default_rng(seed)
        self._total_added = 0

    def add(
        self,
        x: torch.Tensor,
        y_ghi: torch.Tensor,
        y_pv: torch.Tensor,
        eta: torch.Tensor,
        ghi_cs: torch.Tensor,
    ) -> None:
        item = (
            x.detach().cpu(),
            y_ghi.detach().cpu(),
            y_pv.detach().cpu(),
            eta.detach().cpu(),
            ghi_cs.detach().cpu(),
        )
        if len(self._buf) < self.capacity:
            self._buf.append(item)
        else:
            self._buf[self._ptr % self.capacity] = item
        self._ptr += 1
        self._total_added += 1

    def add_batch(
        self,
        x: torch.Tensor,
        y_ghi: torch.Tensor,
        y_pv: torch.Tensor,
        eta: torch.Tensor,
        ghi_cs: torch.Tensor,
    ) -> None:
        for i in range(x.shape[0]):
            self.add(x[i], y_ghi[i], y_pv[i], eta[i], ghi_cs[i])

    def sample(self, n: int) -> tuple[torch.Tensor, ...]:
        """Uniform random sample without replacement. Returns stacked tensors."""
        n = min(n, len(self._buf))
        if n == 0:
            raise ValueError("buffer is empty")
        idx = self._rng.choice(len(self._buf), size=n, replace=False)
        items = [self._buf[i] for i in idx]
        return (
            torch.stack([it[0] for it in items]),
            torch.stack([it[1] for it in items]),
            torch.stack([it[2] for it in items]),
            torch.stack([it[3] for it in items]),
            torch.stack([it[4] for it in items]),
        )

    @property
    def total_added(self) -> int:
        return self._total_added

    def __len__(self) -> int:
        return len(self._buf)
