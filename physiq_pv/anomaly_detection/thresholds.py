"""Training-only anomaly thresholds.

All functions in this module consume anomaly scores only.  Ground-truth event
labels are deliberately absent from the API so they cannot leak into threshold
selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdSpec:
    method: str
    value: float
    quantile: float | None = None
    iqr_k: float | None = None
    n_train_scores: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _finite(scores) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot fit an anomaly threshold without finite training scores.")
    return values


def fit_threshold(
    train_scores,
    *,
    method: str = "iqr",
    quantile: float = 0.995,
    iqr_k: float = 1.5,
) -> ThresholdSpec:
    """Fit a high-score-is-anomalous threshold using training scores only."""
    values = _finite(train_scores)
    if method == "iqr":
        if not np.isfinite(iqr_k) or iqr_k < 0:
            raise ValueError(f"iqr_k must be finite and >= 0, got {iqr_k}.")
        q1, q3 = np.quantile(values, [0.25, 0.75])
        value = float(q3 + iqr_k * (q3 - q1))
        return ThresholdSpec("iqr", value, iqr_k=float(iqr_k), n_train_scores=values.size)
    if method == "quantile":
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}.")
        value = float(np.quantile(values, quantile))
        return ThresholdSpec(
            "quantile", value, quantile=float(quantile), n_train_scores=values.size
        )
    raise ValueError(f"Unknown threshold method {method!r}; expected 'iqr' or 'quantile'.")


def apply_threshold(scores, spec: ThresholdSpec) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    return np.isfinite(values) & (values >= spec.value)


def fit_entity_iqr_thresholds(
    train_entity_scores,
    *,
    iqr_k: float = 1.5,
    scale: float = 0.8,
) -> np.ndarray:
    """Fit Equation 15 independently for every entity using training only."""
    values = np.asarray(train_entity_scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Entity scores must have shape [windows, entities].")
    if not np.isfinite(iqr_k) or iqr_k < 0:
        raise ValueError("iqr_k must be finite and non-negative.")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive.")
    thresholds = np.empty(values.shape[1], dtype=np.float64)
    for entity in range(values.shape[1]):
        finite = values[np.isfinite(values[:, entity]), entity]
        if finite.size == 0:
            raise ValueError(f"Entity {entity} has no finite training scores.")
        q1, q3 = np.quantile(finite, [0.25, 0.75])
        thresholds[entity] = scale * (q3 + iqr_k * (q3 - q1))
    return thresholds


def apply_entity_thresholds(scores, thresholds) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    limits = np.asarray(thresholds, dtype=np.float64)
    if values.ndim != 2 or limits.shape != (values.shape[1],):
        raise ValueError("Require scores [windows, entities] and one threshold per entity.")
    return np.isfinite(values) & (values >= limits[None, :])
