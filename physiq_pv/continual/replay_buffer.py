import numpy as np
import torch


class ReplayBuffer:
    """
    DER++ replay buffer for regression tasks.
    Stores (x, y, pred) triples where pred = model output at storage time.
    Circular overwrite once capacity is reached.

    Reference: aimagelab/mammoth derpp.py
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._buf: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._ptr = 0

    def add(self, x: torch.Tensor, y: torch.Tensor, pred: torch.Tensor) -> None:
        """Store a single sample (no batch dim)."""
        item = (x.detach().cpu(), y.detach().cpu(), pred.detach().cpu())
        if len(self._buf) < self.capacity:
            self._buf.append(item)
        else:
            self._buf[self._ptr % self.capacity] = item
        self._ptr += 1

    def add_batch(self, x: torch.Tensor, y: torch.Tensor, pred: torch.Tensor) -> None:
        """Store each element of a batch individually."""
        for i in range(x.shape[0]):
            self.add(x[i], y[i], pred[i])

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random sample of n items. Returns (x, y, pred) stacked tensors."""
        n = min(n, len(self._buf))
        idx = np.random.choice(len(self._buf), size=n, replace=False)
        xs, ys, ps = zip(*[self._buf[i] for i in idx])
        return torch.stack(xs), torch.stack(ys), torch.stack(ps)

    def __len__(self) -> int:
        return len(self._buf)
