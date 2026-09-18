"""Typed outputs from the STGAN training/scoring pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

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
    _store: object = field(default=None, repr=False, compare=False)

    def close(self):
        """Release memmaps and anonymous storage. Arrays must not be used after close."""
        if self._store is not None:
            self._store.close()
            self._store = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
