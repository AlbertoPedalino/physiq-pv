"""
Mondrian split conformal prediction for QS prediction bands.

One MAPIE SplitConformalRegressor per QS bin for per-regime conditional coverage.
MAPIE does not expose Mondrian CP natively, so Mondrian stratification is applied
manually: the calibration set is partitioned into n_bins by QS percentile, and
an independent SplitConformalRegressor is fitted per bin.

Reference: MAPIE >= 1.3.0 (scikit-learn-contrib/MAPIE), Venn-Abers / Mondrian CP
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from mapie.regression import SplitConformalRegressor


class _PassthroughRegressor(BaseEstimator, RegressorMixin):
    """Identity wrapper — MAPIE sees a pre-fitted estimator (prefit=True)."""
    fitted_ = True

    def fit(self, X, y): return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(X).ravel()


class MondriancpQS:
    """
    Mondrian split CP for QS prediction intervals.

    Usage:
        cp = MondriancpQS(alpha=0.1, n_bins=3)
        cp.fit(qs_pred_calib, qs_true_calib)
        qs_pred, lower, upper = cp.predict(qs_pred_test)
    """

    def __init__(self, alpha: float = 0.1, n_bins: int = 3):
        self.alpha = alpha
        self.n_bins = n_bins
        self._bin_edges: np.ndarray | None = None
        self._cps: dict[int, SplitConformalRegressor] = {}

    def _assign_bins(self, qs: np.ndarray) -> np.ndarray:
        return np.digitize(qs, self._bin_edges[1:-1])  # 0 … n_bins-1

    def fit(self, qs_pred: np.ndarray, qs_true: np.ndarray) -> "MondriancpQS":
        """
        qs_pred: model QS predictions on calibration set (N,)
        qs_true: ground-truth QS values (N,)
        """
        self._bin_edges = np.percentile(qs_true, np.linspace(0, 100, self.n_bins + 1))
        self._bin_edges[0] -= 1e-6
        bins = self._assign_bins(qs_true)

        for b in range(self.n_bins):
            mask = bins == b
            # Fall back to full set if bin is too small for reliable CP
            if mask.sum() < 4:
                mask = np.ones(len(qs_true), dtype=bool)
            X_b = qs_pred[mask].reshape(-1, 1)
            y_b = qs_true[mask]
            cp = SplitConformalRegressor(
                estimator=_PassthroughRegressor(),
                prefit=True,
                confidence_level=1.0 - self.alpha,
            )
            cp.conformalize(X_b, y_b)
            self._cps[b] = cp

        return self

    def predict(
        self, qs_pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (qs_pred, lower, upper)."""
        assert self._bin_edges is not None, "Call fit() first."
        clipped = np.clip(qs_pred, self._bin_edges[0], self._bin_edges[-1])
        bins = self._assign_bins(clipped)

        lower = np.empty_like(qs_pred)
        upper = np.empty_like(qs_pred)

        for b in range(self.n_bins):
            mask = bins == b
            if not mask.any():
                continue
            X_b = qs_pred[mask].reshape(-1, 1)
            _, intervals = self._cps[b].predict_interval(X_b)
            lower[mask] = intervals[:, 0, 0]
            upper[mask] = intervals[:, 1, 0]

        lower = np.clip(lower, 0.0, 1.0)
        upper = np.clip(upper, 0.0, 1.0)
        return qs_pred, lower, upper

    @property
    def interval_widths(self) -> dict[int, float]:
        """Mean interval width per bin."""
        return {
            b: float(
                np.diff(
                    np.array(self._cps[b].predict_interval(
                        np.array([[0.5]]))[1][:, :, 0])
                ).item()
            )
            for b in self._cps
        }


class MondriaNCP:
    """
    Mondrian split CP for PV forecasting output, stratified by QS bands.

    Stratification by data quality — not by predicted value (cf. Renkema 2024).
    Low-QS data → wide intervals; high-QS data → tight intervals.
    Finite-sample correction guarantees per-stratum marginal coverage.

    Reference: Boström et al. COPA 2021 (Mondrian CP).
    """

    def __init__(self, n_bins: int = 3, confidence_level: float = 0.9):
        self.n_bins = n_bins
        self.confidence_level = confidence_level
        self._quantiles: dict[int, float] = {}
        self._band_counts: dict[int, int] = {}

    def qs_to_band(self, qs: float | np.ndarray) -> np.ndarray:
        """Map QS ∈ [0,1] → band ∈ {0, …, n_bins-1}."""
        thresholds = np.linspace(0.0, 1.0 + 1e-9, self.n_bins + 1)
        return np.clip(np.digitize(np.asarray(qs), thresholds[1:]), 0, self.n_bins - 1)

    def calibrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        qs_cal: np.ndarray,
    ) -> "MondriaNCP":
        """
        y_true, y_pred: (N,) PV values on calibration set.
        qs_cal: (N,) QS values for the same samples.
        """
        scores = np.abs(np.asarray(y_true) - np.asarray(y_pred))
        bands = self.qs_to_band(qs_cal)
        global_q = float(np.quantile(scores, self.confidence_level))

        for b in range(self.n_bins):
            mask = bands == b
            n = int(mask.sum())
            self._band_counts[b] = n
            if n > 0:
                adjusted = min(self.confidence_level * (n + 1) / n, 1.0)
                self._quantiles[b] = float(np.quantile(scores[mask], adjusted))
            else:
                self._quantiles[b] = global_q

        return self

    def predict_interval(
        self,
        y_pred: np.ndarray,
        qs_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (lower, upper) arrays. Interval width ∝ QS band quantile."""
        if not self._quantiles:
            raise RuntimeError("Call calibrate() first.")
        y_pred = np.asarray(y_pred)
        hw = np.array([self._quantiles[int(b)] for b in self.qs_to_band(qs_test)])
        return y_pred - hw, y_pred + hw

    def coverage_report(self) -> dict[int, dict]:
        """Summary of per-band calibration."""
        return {
            b: {
                "n_cal": self._band_counts.get(b, 0),
                "quantile": self._quantiles.get(b, None),
                "qs_range": (b / self.n_bins, (b + 1) / self.n_bins),
            }
            for b in range(self.n_bins)
        }
