"""Label-unaware climate anomaly detection for PVGIS time series."""

from .thresholds import (
    ThresholdSpec,
    apply_entity_thresholds,
    apply_threshold,
    fit_entity_iqr_thresholds,
    fit_threshold,
)

__all__ = [
    "ThresholdSpec",
    "apply_entity_thresholds",
    "apply_threshold",
    "fit_entity_iqr_thresholds",
    "fit_threshold",
]
