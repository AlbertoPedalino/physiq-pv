import numpy as np
from collections import deque
from scipy import stats


class ADWINDriftMonitor:
    """
    ADWIN-style drift detector on QS(t).
    Uses two-sample KS test on consecutive half-windows.

    Large ks-statistic + p < significance → distribution shift detected.
    Reference: ADWIN (Bifet & Gavalda, 2007) — scipy approximation.
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
