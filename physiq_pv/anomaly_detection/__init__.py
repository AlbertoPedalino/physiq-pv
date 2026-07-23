"""Unsupervised anomaly detectors for PVGIS time series."""

from physiq_pv.anomaly_detection.catch import (
    CATCH,
    CATCHPreprocessor,
    CATCHScorer,
    CATCHScores,
    CATCHTrainer,
    CATCHTrainingConfig,
    CATCHWindowDataset,
    temporal_train_validation_split,
)
from physiq_pv.anomaly_detection.catch_model import (
    CATCHModel,
    CATCHModelOutput,
    ChannelMaskGenerator,
    ResidualFlattenHead,
    frequency_point_error,
)

__all__ = [
    "CATCH",
    "CATCHModel",
    "CATCHModelOutput",
    "CATCHPreprocessor",
    "CATCHScorer",
    "CATCHScores",
    "CATCHTrainer",
    "CATCHTrainingConfig",
    "CATCHWindowDataset",
    "ChannelMaskGenerator",
    "ResidualFlattenHead",
    "frequency_point_error",
    "temporal_train_validation_split",
]
