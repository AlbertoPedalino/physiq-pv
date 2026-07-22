"""Composable, leakage-safe implementation of Alnegheimish et al.'s M2AD.

This module is the public facade.  The independent building blocks live in:

* :mod:`m2ad_preprocessing` -- interpolation and train-only scaling;
* :mod:`m2ad_forecaster` -- windowing, LSTM training, and prediction;
* :mod:`m2ad_errors` -- point/area discrepancies and EWMA;
* :mod:`m2ad_calibration` -- sensor GMMs, Fisher aggregation, and Gamma threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from physiq_pv.anomaly_detection.m2ad_calibration import M2ADCalibrator
from physiq_pv.anomaly_detection.m2ad_errors import (
    M2ADDiscrepancy,
    compute_discrepancy,
)
from physiq_pv.anomaly_detection.m2ad_forecaster import (
    M2ADForecaster,
    M2ADLSTM,
    M2ADWindowDataset,
)
from physiq_pv.anomaly_detection.m2ad_preprocessing import M2ADPreprocessor


# Backwards-compatible private alias used by existing synthetic tests.
_WindowDataset = M2ADWindowDataset
_EPS = 1e-12


@dataclass(frozen=True)
class M2ADScores:
    """Aligned numerical outputs from :meth:`M2AD.score_segments`."""

    segment_ids: np.ndarray
    target_indices: np.ndarray
    observed: np.ndarray
    predicted: np.ndarray
    errors: np.ndarray
    sensor_p_values: np.ndarray
    sensor_fisher: np.ndarray
    weighted_contributions: np.ndarray
    global_scores: np.ndarray
    gamma_p_values: np.ndarray
    is_anomaly: np.ndarray


class M2AD:
    """High-level composition of the four independent M2AD stages.

    The class itself owns no neural or statistical fitting logic.  It wires
    together :class:`M2ADPreprocessor`, :class:`M2ADForecaster`,
    :class:`M2ADDiscrepancy`, and :class:`M2ADCalibrator` and exposes a compact
    ``fit``/``score_segments`` interface.
    """

    def __init__(
        self,
        sensor_names: Sequence[str],
        *,
        window_size: int = 120,
        horizon: int = 1,
        hidden_size: int = 80,
        n_layers: int = 2,
        dropout: float = 0.2,
        error: str = "area",
        area_half_window: int = 2,
        ewma_com: Optional[float] = 10.0,
        n_components: int | str | Mapping[str, int] = "bic",
        max_components: int = 3,
        covariance_type: str = "spherical",
        sensor_weights: Optional[Sequence[float]] = None,
        significance: float = 0.001,
        feature_range: tuple[float, float] = (-1.0, 1.0),
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        epochs: int = 30,
        validation_split: float = 0.2,
        patience: int = 5,
        min_delta: float = 0.0,
        device: str = "cpu",
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.sensor_names = [str(name) for name in sensor_names]
        if not self.sensor_names or len(set(self.sensor_names)) != len(self.sensor_names):
            raise ValueError("sensor_names must be non-empty and unique")
        self.n_sensors = len(self.sensor_names)
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        self.error = error
        self.feature_range = tuple(float(value) for value in feature_range)

        self.preprocessor = M2ADPreprocessor(
            self.n_sensors, feature_range=self.feature_range
        )
        self.forecaster = M2ADForecaster(
            self.n_sensors,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            validation_split=validation_split,
            patience=patience,
            min_delta=min_delta,
            device=device,
            seed=seed,
            verbose=verbose,
        )
        self.discrepancy = M2ADDiscrepancy(
            error=error,
            area_half_window=area_half_window,
            ewma_com=ewma_com,
        )
        self.calibrator = M2ADCalibrator(
            self.sensor_names,
            error=error,
            n_components=n_components,
            max_components=max_components,
            covariance_type=covariance_type,
            sensor_weights=sensor_weights,
            significance=significance,
            seed=seed,
        )
        self.is_fitted = False

    # Compatibility/readability properties: state remains owned by components.
    @property
    def model(self):
        return self.forecaster.model

    @property
    def history(self) -> list[dict[str, float]]:
        return self.forecaster.history

    @property
    def data_min(self) -> np.ndarray:
        return self.preprocessor.data_min

    @property
    def data_max(self) -> np.ndarray:
        return self.preprocessor.data_max

    @property
    def selected_components(self) -> dict[str, int]:
        return self.calibrator.selected_components

    @property
    def gamma_shape(self) -> float:
        return self.calibrator.gamma_shape

    @property
    def gamma_scale(self) -> float:
        return self.calibrator.gamma_scale

    @property
    def threshold(self) -> float:
        return self.calibrator.threshold

    def fit(self, train_segments: Sequence[np.ndarray]) -> "M2AD":
        """Fit each stage in order using training segments only."""
        scaled = self.preprocessor.fit_transform(train_segments)
        dataset = M2ADWindowDataset(scaled, self.window_size, self.horizon)
        self.forecaster.fit(dataset)
        observed, predicted, segment_ids, _ = self.forecaster.predict(dataset)
        errors = self.discrepancy.fit_transform(observed, predicted, segment_ids)
        self.calibrator.fit(errors)
        self.n_train_windows = len(dataset)
        self.is_fitted = True
        return self

    def score_segments(self, segments: Sequence[np.ndarray]) -> M2ADScores:
        """Score disjoint inference segments without refitting any stage."""
        if not self.is_fitted:
            raise RuntimeError("call fit before score_segments")
        scaled = self.preprocessor.transform(segments)
        dataset = M2ADWindowDataset(scaled, self.window_size, self.horizon)
        observed, predicted, segment_ids, target_indices = self.forecaster.predict(
            dataset
        )
        errors = self.discrepancy.transform(observed, predicted, segment_ids)
        (
            p_values,
            fisher,
            contributions,
            global_scores,
            gamma_p_values,
            anomalies,
        ) = self.calibrator.score(errors)
        return M2ADScores(
            segment_ids=segment_ids,
            target_indices=target_indices,
            observed=observed,
            predicted=predicted,
            errors=errors,
            sensor_p_values=p_values,
            sensor_fisher=fisher,
            weighted_contributions=contributions,
            global_scores=global_scores,
            gamma_p_values=gamma_p_values,
            is_anomaly=anomalies,
        )

    def scores_frame(
        self,
        scores: M2ADScores,
        timestamps_by_segment: Sequence[Sequence],
        *,
        entity: Optional[str] = None,
    ) -> pd.DataFrame:
        """Convert numerical scores to an interpretable timestamped table."""
        if len(timestamps_by_segment) <= int(np.max(scores.segment_ids)):
            raise ValueError("timestamps_by_segment does not cover every scored segment")
        timestamps = np.asarray(
            [
                timestamps_by_segment[int(segment_id)][int(target_index)]
                for segment_id, target_index in zip(
                    scores.segment_ids, scores.target_indices
                )
            ]
        )
        top_index = np.argmax(scores.weighted_contributions, axis=1)
        row_index = np.arange(len(top_index))
        safe_scores = np.maximum(scores.global_scores, _EPS)
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(timestamps),
                "global_score": scores.global_scores,
                "gamma_p_value": scores.gamma_p_values,
                "threshold": self.threshold,
                "is_anomaly": scores.is_anomaly,
                "top_sensor": np.asarray(self.sensor_names, dtype=object)[top_index],
                "top_contribution": (
                    scores.weighted_contributions[row_index, top_index] / safe_scores
                ),
            }
        )
        if entity is not None:
            frame.insert(0, "location", str(entity))
        observed_raw = self.preprocessor.inverse_transform(scores.observed)
        predicted_raw = self.preprocessor.inverse_transform(scores.predicted)
        for sensor_index, sensor in enumerate(self.sensor_names):
            prefix = sensor.replace(" ", "_")
            frame[f"observed__{prefix}"] = observed_raw[:, sensor_index]
            frame[f"predicted__{prefix}"] = predicted_raw[:, sensor_index]
            frame[f"error__{prefix}"] = scores.errors[:, sensor_index]
            frame[f"p_value__{prefix}"] = scores.sensor_p_values[:, sensor_index]
            frame[f"contribution__{prefix}"] = (
                scores.weighted_contributions[:, sensor_index] / safe_scores
            )
        return frame
