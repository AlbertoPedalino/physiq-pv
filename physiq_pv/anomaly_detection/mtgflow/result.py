"""Typed outputs produced by the MTGFlow training/scoring pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MTGFlowResult:
    """Global and entity-level MTGFlow scores with their window boundaries."""

    train_timestamps: pd.DatetimeIndex
    train_scores: np.ndarray
    test_timestamps: pd.DatetimeIndex
    test_scores: np.ndarray
    metadata: dict
    train_entity_scores: np.ndarray | None = None
    test_entity_scores: np.ndarray | None = None
    entity_names: tuple[str, ...] = ()
    train_window_starts: pd.DatetimeIndex | None = None
    test_window_starts: pd.DatetimeIndex | None = None
