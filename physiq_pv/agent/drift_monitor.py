import numpy as np
from collections import deque
from scipy import stats


class KSDriftMonitor:
    """
    KS-based drift monitor inspired by ADWIN-style window comparison.

    NOT a full incremental ADWIN implementation (Bifet & Gavaldà, 2007).
    Instead: a sliding buffer of length 2*window_size is split in two halves
    (older vs newer) and compared via scipy's two-sample Kolmogorov-Smirnov
    test. Drift is reported when p < significance.

    Trade-off vs true ADWIN:
      + simple, no extra dependency, deterministic, easy to interpret;
      - batch (re-runs KS on every update), not truly incremental;
      - reacts only after the full new half-window has filled, so latency
        is O(window_size).

    For honest naming the class is `KSDriftMonitor`. The historical alias
    `ADWINDriftMonitor` is kept for backward compatibility with existing
    imports (cycle.py, notebooks, docs).
    """

    def __init__(self, window_size: int = 720, significance: float = 0.001):
        self.window_size = window_size
        self.significance = significance
        self._buf: deque[float] = deque(maxlen=window_size * 2)

    def update(self, qs_value: float) -> bool:
        """Add one observation. Returns True if drift detected."""
        if not np.isnan(qs_value):
            self._buf.append(qs_value)
        return self._check()

    def update_batch(self, qs_values: np.ndarray) -> np.ndarray:
        """Process 1-D array of QS values. Returns bool mask of drift timesteps."""
        flags = np.zeros(len(qs_values), dtype=bool)
        for i, v in enumerate(qs_values):
            flags[i] = self.update(float(v))
        return flags

    def _check(self) -> bool:
        buf = np.array(self._buf)
        if len(buf) < self.window_size:
            return False
        half = len(buf) // 2
        w1, w2 = buf[:half], buf[half:]
        _, p = stats.ks_2samp(w1, w2)
        return bool(p < self.significance)

    def reset(self) -> None:
        self._buf.clear()


# Backward-compatible alias. Existing imports keep working without churn,
# but new code should prefer `KSDriftMonitor` to avoid suggesting that this
# class implements the true incremental ADWIN algorithm.
ADWINDriftMonitor = KSDriftMonitor
