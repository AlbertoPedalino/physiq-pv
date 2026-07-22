"""Public facade for the modular, paper-aligned CATCH detector.

Canonical policy:

* explicit equations in Wu et al. (ICLR 2025) take precedence;
* repository architecture defaults fill details omitted by the paper;
* train-only preprocessing and thresholding intentionally prevent leakage.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from physiq_pv.anomaly_detection.catch_data import (
    CATCHPreprocessor,
    CATCHWindowDataset,
    temporal_train_validation_split,
)
from physiq_pv.anomaly_detection.catch_model import CATCHModel
from physiq_pv.anomaly_detection.catch_scoring import CATCHScorer, CATCHScores
from physiq_pv.anomaly_detection.catch_training import (
    CATCHTrainer,
    CATCHTrainingConfig,
)


class CATCH:
    """Compose CATCH preprocessing, model, training, scoring, and calibration."""

    def __init__(
        self,
        sensor_names: Sequence[str],
        *,
        seq_len: int = 192,
        patch_size: int = 16,
        patch_stride: int = 8,
        inference_patch_size: int = 32,
        inference_patch_stride: int = 1,
        cf_dim: int = 64,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 2,
        head_dim: int = 64,
        d_ff: int = 256,
        dropout: float = 0.2,
        head_dropout: float = 0.1,
        head_layers: int = 3,
        temperature: float = 0.07,
        affine_revin: bool = False,
        mask_source: str = "projected",
        frequency_loss_weight: float = 0.005,
        clustering_weight: float = 0.005,
        regularization_weight: float = 0.0025,
        score_frequency_weight: float = 0.05,
        contamination: float = 0.01,
        batch_size: int = 128,
        learning_rate: float = 1e-4,
        mask_learning_rate: float = 1e-5,
        epochs: int = 3,
        validation_split: float = 0.2,
        patience: int = 3,
        training_window_stride: int = 1,
        scoring_window_stride: int = 1,
        model_steps_per_mask: int = 10,
        gradient_clip: float = 1.0,
        device: str = "cpu",
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.sensor_names = [str(name) for name in sensor_names]
        if not self.sensor_names or len(set(self.sensor_names)) != len(self.sensor_names):
            raise ValueError("sensor_names must be non-empty and unique")
        if not 0.0 < contamination < 1.0:
            raise ValueError("contamination must be in (0, 1)")
        if not 0.0 <= validation_split < 1.0:
            raise ValueError("validation_split must be in [0, 1)")
        if epochs < 1 or batch_size < 1 or patience < 1:
            raise ValueError("epochs, batch_size, and patience must be >= 1")
        if model_steps_per_mask < 1:
            raise ValueError("model_steps_per_mask must be >= 1")
        if scoring_window_stride > seq_len:
            raise ValueError("scoring_window_stride must not exceed seq_len")

        self.n_channels = len(self.sensor_names)
        self.seq_len = int(seq_len)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.inference_patch_size = int(min(inference_patch_size, seq_len))
        self.inference_patch_stride = int(inference_patch_stride)
        self.cf_dim = int(cf_dim)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.d_ff = int(d_ff)
        self.dropout = float(dropout)
        self.head_dropout = float(head_dropout)
        self.head_layers = int(head_layers)
        self.temperature = float(temperature)
        self.affine_revin = bool(affine_revin)
        self.mask_source = mask_source
        self.score_frequency_weight = float(score_frequency_weight)
        self.contamination = float(contamination)
        self.batch_size = int(batch_size)
        self.validation_split = float(validation_split)
        self.training_window_stride = int(training_window_stride)
        self.scoring_window_stride = int(scoring_window_stride)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self.training_config = CATCHTrainingConfig(
            frequency_loss_weight=float(frequency_loss_weight),
            clustering_weight=float(clustering_weight),
            regularization_weight=float(regularization_weight),
            batch_size=self.batch_size,
            learning_rate=float(learning_rate),
            mask_learning_rate=float(mask_learning_rate),
            epochs=int(epochs),
            patience=int(patience),
            model_steps_per_mask=int(model_steps_per_mask),
            gradient_clip=float(gradient_clip),
            seed=self.seed,
        )
        self.preprocessor = CATCHPreprocessor(self.n_channels)
        self.model: Optional[CATCHModel] = None
        self.trainer: Optional[CATCHTrainer] = None
        self.scorer: Optional[CATCHScorer] = None
        self.history: list[dict[str, float]] = []
        self.is_fitted = False

    @classmethod
    def compact(cls, sensor_names: Sequence[str], **kwargs) -> "CATCH":
        """Small development preset; never presented as paper/repo equivalent."""

        defaults = {
            "cf_dim": 8,
            "d_model": 8,
            "n_layers": 1,
            "n_heads": 2,
            "head_dim": 4,
            "d_ff": 16,
            "head_layers": 1,
        }
        defaults.update(kwargs)
        return cls(sensor_names, **defaults)

    def _new_model(self) -> CATCHModel:
        return CATCHModel(
            self.n_channels,
            seq_len=self.seq_len,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            cf_dim=self.cf_dim,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            d_ff=self.d_ff,
            dropout=self.dropout,
            head_dropout=self.head_dropout,
            head_layers=self.head_layers,
            temperature=self.temperature,
            affine_revin=self.affine_revin,
            mask_source=self.mask_source,
        ).to(self.device)

    def _new_scorer(self) -> CATCHScorer:
        if self.model is None:
            raise RuntimeError("model has not been initialized")
        return CATCHScorer(
            self.model,
            self.preprocessor,
            self.sensor_names,
            inference_patch_size=self.inference_patch_size,
            inference_patch_stride=self.inference_patch_stride,
            score_frequency_weight=self.score_frequency_weight,
            batch_size=self.batch_size,
            window_stride=self.scoring_window_stride,
            device=self.device,
        )

    def fit(self, train_segments: Sequence[np.ndarray]) -> "CATCH":
        """Fit on chronological train portions and calibrate without labels."""

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        fitting_segments, validation_segments = temporal_train_validation_split(
            train_segments,
            validation_split=self.validation_split,
            min_length=self.seq_len,
        )
        scaled_train = self.preprocessor.fit_transform(fitting_segments)
        scaled_validation = (
            self.preprocessor.transform(validation_segments)
            if validation_segments
            else scaled_train
        )
        train_dataset = CATCHWindowDataset(
            scaled_train,
            self.seq_len,
            stride=self.training_window_stride,
        )
        validation_dataset = CATCHWindowDataset(
            scaled_validation,
            self.seq_len,
            stride=self.training_window_stride,
        )
        self.n_train_windows = len(train_dataset)
        self.train_split_lengths = tuple(len(segment) for segment in fitting_segments)
        self.validation_split_lengths = tuple(
            len(segment) for segment in validation_segments
        )

        self.model = self._new_model()
        self.trainer = CATCHTrainer(
            self.model,
            self.training_config,
            device=self.device,
            verbose=self.verbose,
        )
        self.history = self.trainer.fit(train_dataset, validation_dataset)
        self.scorer = self._new_scorer()
        train_scores = self.scorer.score_scaled_segments(scaled_train)
        self.threshold = float(
            np.quantile(train_scores.global_scores, 1.0 - self.contamination)
        )
        self.is_fitted = True
        return self

    def score_segments(self, segments: Sequence[np.ndarray]) -> CATCHScores:
        """Score disjoint segments using frozen train-only state."""

        if not self.is_fitted or self.scorer is None:
            raise RuntimeError("call fit before score_segments")
        scaled = self.preprocessor.transform(segments)
        return self.scorer.score_scaled_segments(
            scaled, threshold=self.threshold
        )

    def scores_frame(
        self,
        scores: CATCHScores,
        timestamps_by_segment: Sequence[Sequence],
        *,
        entity: Optional[str] = None,
    ) -> pd.DataFrame:
        if not self.is_fitted or self.scorer is None:
            raise RuntimeError("call fit before scores_frame")
        return self.scorer.scores_frame(
            scores,
            timestamps_by_segment,
            threshold=self.threshold,
            entity=entity,
        )


__all__ = [
    "CATCH",
    "CATCHPreprocessor",
    "CATCHScorer",
    "CATCHScores",
    "CATCHTrainer",
    "CATCHTrainingConfig",
    "CATCHWindowDataset",
    "temporal_train_validation_split",
]
