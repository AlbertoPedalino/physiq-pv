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
        self._peak_scores: list[float] = []
        self._ptr = 0
        self._rng = np.random.default_rng(seed)
        self._total_added = 0

    @staticmethod
    def _peak_score(y_pv: torch.Tensor) -> float:
        y = y_pv.detach().float().cpu()
        if y.numel() == 0:
            return float("-inf")
        finite = torch.isfinite(y)
        if not bool(finite.any()):
            return float("-inf")
        return float(y[finite].max().item())

    def _stack_indices(self, idx: "np.ndarray | list[int]") -> tuple[torch.Tensor, ...]:
        items = [self._buf[int(i)] for i in idx]
        return (
            torch.stack([it[0] for it in items]),
            torch.stack([it[1] for it in items]),
            torch.stack([it[2] for it in items]),
            torch.stack([it[3] for it in items]),
            torch.stack([it[4] for it in items]),
        )

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
            self._peak_scores.append(self._peak_score(y_pv))
        else:
            idx = self._ptr % self.capacity
            self._buf[idx] = item
            self._peak_scores[idx] = self._peak_score(y_pv)
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
        return self._stack_indices(idx)

    def sample_peak_aware(
        self,
        n: int,
        peak_fraction: float = 0.0,
        over_100_fraction: float = 0.0,
        peak_threshold: float = 0.6,
        over_100_threshold: float = 1.0,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, int]]:
        """Sample replay with optional quotas for rare high-production targets."""
        n = min(n, len(self._buf))
        if n == 0:
            raise ValueError("buffer is empty")

        peak_fraction = max(0.0, float(peak_fraction))
        over_100_fraction = max(0.0, float(over_100_fraction))
        if peak_fraction + over_100_fraction > 1.0:
            scale = peak_fraction + over_100_fraction
            peak_fraction /= scale
            over_100_fraction /= scale

        n_over = min(n, int(round(n * over_100_fraction)))
        n_peak = min(n - n_over, int(round(n * peak_fraction)))

        scores = np.asarray(self._peak_scores, dtype=float)
        all_idx = np.arange(len(self._buf))
        over_pool = all_idx[scores >= over_100_threshold]
        peak_pool = all_idx[(scores >= peak_threshold) & (scores < over_100_threshold)]

        selected: list[int] = []
        selected_set: set[int] = set()

        def take(pool: np.ndarray, k: int) -> None:
            if k <= 0:
                return
            available = np.asarray([int(i) for i in pool if int(i) not in selected_set], dtype=int)
            if available.size == 0:
                return
            chosen = self._rng.choice(available, size=min(k, available.size), replace=False)
            for item in chosen:
                value = int(item)
                selected.append(value)
                selected_set.add(value)

        take(over_pool, n_over)
        take(peak_pool, n_peak)

        remaining = n - len(selected)
        if remaining > 0:
            fill_pool = np.asarray([int(i) for i in all_idx if int(i) not in selected_set], dtype=int)
            if fill_pool.size > 0:
                chosen = self._rng.choice(fill_pool, size=min(remaining, fill_pool.size), replace=False)
                selected.extend(int(i) for i in chosen)

        selected_arr = np.asarray(selected, dtype=int)
        selected_scores = scores[selected_arr]
        stats = {
            "n_replay_peak_samples": int(
                ((selected_scores >= peak_threshold) & (selected_scores < over_100_threshold)).sum()
            ),
            "n_replay_over_100_samples": int((selected_scores >= over_100_threshold).sum()),
            "n_replay_low_samples": int(
                (selected_scores < peak_threshold).sum()
            ),
        }
        return self._stack_indices(selected_arr), stats

    @property
    def total_added(self) -> int:
        return self._total_added

    def __len__(self) -> int:
        return len(self._buf)
