"""Typed outputs from the STGAN training/scoring pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class STGANResult:
    test_timestamps: pd.DatetimeIndex
    location_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    test_scores: np.ndarray
    test_feature_scores: np.ndarray
    test_generator_scores: np.ndarray
    test_discriminator_scores: np.ndarray
    metadata: dict
