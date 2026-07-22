"""Unsupervised anomaly detectors for PVGIS time series."""

from physiq_pv.anomaly_detection.m2ad import (
    M2AD,
    M2ADLSTM,
    M2ADScores,
    compute_discrepancy,
)
from physiq_pv.anomaly_detection.m2ad_calibration import M2ADCalibrator
from physiq_pv.anomaly_detection.m2ad_errors import M2ADDiscrepancy
from physiq_pv.anomaly_detection.m2ad_forecaster import (
    M2ADForecaster,
    M2ADWindowDataset,
)
from physiq_pv.anomaly_detection.m2ad_preprocessing import M2ADPreprocessor

__all__ = [
    "M2AD",
    "M2ADCalibrator",
    "M2ADDiscrepancy",
    "M2ADForecaster",
    "M2ADLSTM",
    "M2ADPreprocessor",
    "M2ADScores",
    "M2ADWindowDataset",
    "compute_discrepancy",
]
