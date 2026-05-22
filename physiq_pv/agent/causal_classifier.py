"""
Fault causal classifier — MultiROCKET transform + RidgeClassifierCV.

MultiROCKET reference: Tan et al. 2022 (arXiv:2102.00457)
Kernels: MiniROCKET-style fixed alternating weights, exponential dilation,
4 pooling features per kernel (PPV, max, mean, std).

To swap for angus924/hydra (Hydra+MultiROCKET) once installed:
    from hydra import Hydra, SparseScaler
    Replace MultiRocketTransform with Hydra(input_length=T, k=8, g=64)

Fault labels
------------
0  normal
1  gradual_degradation
2  sensor_failure
3  regional_cloud_event
4  soiling_cycle
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FAULT_NORMAL = 0
FAULT_DEGRADATION = 1
FAULT_SENSOR = 2
FAULT_CLOUD = 3
FAULT_SOILING = 4

FAULT_NAMES = {
    FAULT_NORMAL: "normal",
    FAULT_DEGRADATION: "gradual_degradation",
    FAULT_SENSOR: "sensor_failure",
    FAULT_CLOUD: "regional_cloud_event",
    FAULT_SOILING: "soiling_cycle",
}

_KERNEL_LENGTHS = np.array([7, 9, 11])


class MultiRocketTransform:
    """
    MultiROCKET feature extractor.
    Kernels use alternating +1/-1 weights (MiniROCKET style) with random bias
    and exponentially sampled dilation. Four pooling features per kernel:
    PPV, max, mean, std → 4*n_kernels total features per time series.
    """

    def __init__(self, n_kernels: int = 84, random_state: int = 42):
        self.n_kernels = n_kernels
        self.random_state = random_state
        self._kernels: list[tuple[np.ndarray, float, int]] = []

    def fit(self, X: np.ndarray) -> "MultiRocketTransform":
        """X: (N, T)"""
        rng = np.random.default_rng(self.random_state)
        T = X.shape[1]
        self._kernels = []
        for _ in range(self.n_kernels):
            k_len = int(rng.choice(_KERNEL_LENGTHS))
            w = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(k_len)])
            w -= w.mean()
            bias = float(rng.uniform(-1.0, 1.0))
            max_exp = max(int(np.floor(np.log2((T - 1) / max(k_len - 1, 1)))), 0)
            dilation = int(2 ** int(rng.integers(0, max_exp + 1)))
            self._kernels.append((w.astype(np.float32), bias, dilation))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: (N, T) -> (N, 4*n_kernels)"""
        N = X.shape[0]
        out = np.empty((N, 4 * len(self._kernels)), dtype=np.float32)
        for ki, (w, bias, dilation) in enumerate(self._kernels):
            # Build dilated kernel
            dilated = np.zeros(dilation * (len(w) - 1) + 1, dtype=np.float32)
            dilated[::dilation] = w
            flipped = dilated[::-1].copy()
            # Apply to each series
            convs = np.stack(
                [np.convolve(X[i].astype(np.float32), flipped, mode="valid") + bias
                 for i in range(N)]
            )  # (N, L')
            base = ki * 4
            out[:, base + 0] = (convs > 0).mean(axis=1)   # PPV
            out[:, base + 1] = convs.max(axis=1)           # max
            out[:, base + 2] = convs.mean(axis=1)          # mean
            out[:, base + 3] = convs.std(axis=1)           # std
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class MultiRocketClassifier(BaseEstimator, ClassifierMixin):
    """MultiROCKET transform + RidgeClassifierCV. Input X: (N, T)."""

    def __init__(self, n_kernels: int = 84, random_state: int = 42):
        self.n_kernels = n_kernels
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiRocketClassifier":
        self.transform_ = MultiRocketTransform(self.n_kernels, self.random_state)
        feats = self.transform_.fit_transform(X)
        self.pipeline_ = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", RidgeClassifierCV(alphas=[1e-3, 1e-2, 0.1, 1.0, 10.0])),
        ])
        self.pipeline_.fit(feats, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline_.predict(self.transform_.transform(X))


def make_classifier(**kwargs) -> MultiRocketClassifier:
    return MultiRocketClassifier(**kwargs)


def _softmax_confidence(decision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Ridge decision_function output to pseudo-probabilities via softmax.
    Returns (predicted_indices, confidence_per_sample).

    Binary edge case (sklearn convention):
        RidgeClassifierCV.decision_function returns a 1-D array of signed
        margins when there are exactly two classes. The sign convention is
        ``positive -> classes_[1]``, ``negative -> classes_[0]``. We stack
        ``[-decision, decision]`` so that column 0 maps to classes_[0] and
        column 1 maps to classes_[1]. The returned argmax is therefore
        directly aligned with ``self._classes``.

        For ``decision.ndim == 1`` the caller is expected to have built the
        classifier on exactly two classes. If only one class is present in
        training data sklearn raises during fit; we never reach here.
    """
    if decision.ndim == 1:
        # Binary case: stack to (N, 2) keeping classes_ ordering.
        p = np.vstack([-decision, decision]).T
    else:
        p = decision
    shifted = p - p.max(axis=1, keepdims=True)
    exp_p = np.exp(shifted)
    denom = exp_p.sum(axis=1, keepdims=True)
    # Guard against zero denom from extreme inputs (shouldn't happen after
    # the max-shift, but cheap to keep).
    denom = np.where(denom > 0, denom, 1.0)
    proba = exp_p / denom
    return proba.argmax(axis=1), proba.max(axis=1)


class CausalClassifier:
    """
    Wrapper around MultiRocketClassifier providing the agentic diagnosis interface.

    When not yet fitted (no labeled data), falls back to rule-based heuristics.
    After labeling session with supervisor, call fit() to replace the fallback.

    Usage in agentic cycle:
        label, confidence = classifier.diagnose(qs_sequence)  # (T,) array
    """

    def __init__(self, seq_len: int = 720, uncertain_threshold: float = 0.4):
        self.seq_len = seq_len
        self.uncertain_threshold = uncertain_threshold
        self._clf: MultiRocketClassifier | None = None
        self._classes: np.ndarray | None = None

    # ---------------------------------------------------------------------- #
    # Public interface
    # ---------------------------------------------------------------------- #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CausalClassifier":
        """
        Train on labeled QS(t) sequences.
        X : (n_samples, seq_len)
        y : (n_samples,) string labels
        """
        self._clf = MultiRocketClassifier(n_kernels=84)
        self._clf.fit(X, y)
        self._classes = self._clf.pipeline_.classes_
        return self

    def diagnose(self, qs_sequence: np.ndarray) -> tuple[str, float]:
        """
        Classify one QS(t) time series.
        Returns (label, confidence).  confidence ∈ [0,1].
        Falls back to rule-based if not yet fitted.
        """
        if self._clf is None:
            return self._rule_based(qs_sequence), 1.0

        seq = self._prepare(qs_sequence)
        if seq is None:
            return "uncertain", 0.0

        feats = self._clf.transform_.transform(seq[np.newaxis, :])
        scaler = self._clf.pipeline_.named_steps["scaler"]
        ridge = self._clf.pipeline_.named_steps["clf"]
        decision = ridge.decision_function(scaler.transform(feats))
        idx, confidence = _softmax_confidence(decision)
        confidence = float(confidence[0])

        if confidence < self.uncertain_threshold:
            return "uncertain", confidence
        return str(self._classes[int(idx[0])]), confidence

    # ---------------------------------------------------------------------- #
    # Training data helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def make_synthetic_training_data(
        qs,  # xr.DataArray (plant, time)
        window: int = 720,
        stride: int = 360,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract labeled QS(t) windows from synthetic dataset.

        Labels assigned from known fault-injection times in synthetic_generator:
          plant 0              → gradual_degradation (full series)
          plant 1, t > 4200   → sensor_failure
          plants 2-5, t∈[6000,6600) → regional_cloud_event
          plant 6              → soiling_cycle
          others               → normal

        Returns X (n_samples, window), y (n_samples,) string array.
        """
        T = qs.shape[1]
        X_list: list[np.ndarray] = []
        y_list: list[str] = []

        for start in range(window, T - window, stride):
            t_start, t_end = start - window, start

            for p in range(qs.shape[0]):
                seq = qs.isel(plant=p, time=slice(t_start, t_end)).values
                if float(np.isfinite(seq).mean()) < 0.3:
                    continue

                if p == 0:
                    label = FAULT_NAMES[FAULT_DEGRADATION]
                elif p == 1:
                    label = FAULT_NAMES[FAULT_SENSOR] if start > 4200 else FAULT_NAMES[FAULT_NORMAL]
                elif 2 <= p <= 5:
                    label = FAULT_NAMES[FAULT_CLOUD] if 6000 <= start < 6600 else FAULT_NAMES[FAULT_NORMAL]
                elif p == 6:
                    label = FAULT_NAMES[FAULT_SOILING]
                else:
                    label = FAULT_NAMES[FAULT_NORMAL]

                X_list.append(np.nan_to_num(seq, nan=0.5).astype(np.float32))
                y_list.append(label)

        if not X_list:
            return np.empty((0, window), dtype=np.float32), np.empty(0, dtype=object)

        X_arr = np.array(X_list)
        y_arr = np.array(y_list)

        # Undersample normal to 2x the largest fault class to reduce bias
        fault_mask = y_arr != FAULT_NAMES[FAULT_NORMAL]
        n_fault = int(fault_mask.sum())
        normal_mask = ~fault_mask
        n_normal_keep = min(int(normal_mask.sum()), max(n_fault * 2, 20))
        normal_idx = np.where(normal_mask)[0]
        rng = np.random.default_rng(42)
        keep_idx = rng.choice(normal_idx, size=n_normal_keep, replace=False)
        final_idx = np.concatenate([np.where(fault_mask)[0], keep_idx])
        rng.shuffle(final_idx)
        return X_arr[final_idx], y_arr[final_idx]

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _prepare(self, qs_seq: np.ndarray) -> np.ndarray | None:
        """Trim/pad to seq_len, fill NaN → 0.5."""
        valid_mask = np.isfinite(qs_seq)
        if valid_mask.sum() < self.seq_len // 4:
            return None
        seq = qs_seq[-self.seq_len:]
        if len(seq) < self.seq_len:
            fill = float(np.nanmean(seq)) if np.any(valid_mask) else 0.5
            seq = np.concatenate([np.full(self.seq_len - len(seq), fill), seq])
        return np.nan_to_num(seq, nan=0.5).astype(np.float32)

    def _rule_based(self, qs_seq: np.ndarray) -> str:
        """Heuristic fallback — matches patterns from FAULT_NAMES."""
        valid = qs_seq[~np.isnan(qs_seq)]
        if len(valid) < 10:
            return FAULT_NAMES[FAULT_NORMAL]

        n = len(valid)
        recent = valid[-min(200, n):]
        early = valid[:min(200, n)]

        if n > 200 and float(recent.mean()) < 0.3 and float(early.mean()) > 0.5:
            return FAULT_NAMES[FAULT_SENSOR]

        slope = float(np.polyfit(np.arange(n), valid, 1)[0])
        if slope < -3e-4:
            return FAULT_NAMES[FAULT_DEGRADATION]

        if n > 720:
            lag = min(720, n - 1)
            corr = float(np.corrcoef(valid[:-lag], valid[lag:])[0, 1])
            if corr > 0.3:
                return FAULT_NAMES[FAULT_SOILING]

        return FAULT_NAMES[FAULT_NORMAL]
