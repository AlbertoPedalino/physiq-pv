"""Point-aligned time/frequency scoring for CATCH."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from physiq_pv.anomaly_detection.catch_data import (
    CATCHPreprocessor,
    CATCHWindowDataset,
)
from physiq_pv.anomaly_detection.catch_model import (
    CATCHModel,
    frequency_point_error,
)


_EPS = 1e-12


@dataclass(frozen=True)
class CATCHScores:
    """Point-aligned CATCH outputs concatenated across disjoint segments."""

    segment_ids: np.ndarray
    target_indices: np.ndarray
    observed: np.ndarray
    reconstructed: np.ndarray
    time_scores: np.ndarray
    frequency_scores: np.ndarray
    channel_scores: np.ndarray
    global_scores: np.ndarray
    is_anomaly: np.ndarray


class CATCHScorer:
    """Independent window reconstruction, overlap aggregation, and formatting."""

    def __init__(
        self,
        model: CATCHModel,
        preprocessor: CATCHPreprocessor,
        sensor_names: Sequence[str],
        *,
        inference_patch_size: int,
        inference_patch_stride: int,
        score_frequency_weight: float,
        batch_size: int,
        window_stride: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.sensor_names = [str(name) for name in sensor_names]
        self.n_channels = len(self.sensor_names)
        self.inference_patch_size = int(inference_patch_size)
        self.inference_patch_stride = int(inference_patch_stride)
        self.score_frequency_weight = float(score_frequency_weight)
        self.batch_size = int(batch_size)
        self.window_stride = int(window_stride)
        self.device = device

    def score_scaled_segments(
        self,
        scaled: Sequence[np.ndarray],
        *,
        threshold: float = float("inf"),
    ) -> CATCHScores:
        dataset = CATCHWindowDataset(
            scaled, self.model.seq_len, stride=self.window_stride
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        reconstruction_sums = [np.zeros_like(segment, dtype=np.float64) for segment in scaled]
        time_sums = [np.zeros_like(segment, dtype=np.float64) for segment in scaled]
        frequency_sums = [np.zeros_like(segment, dtype=np.float64) for segment in scaled]
        counts = [np.zeros(len(segment), dtype=np.float64) for segment in scaled]

        self.model.eval()
        with torch.no_grad():
            for batch, segment_ids, starts in loader:
                batch = batch.to(self.device)
                output = self.model(batch)
                time_error = (batch - output.reconstruction).square()
                frequency_error = frequency_point_error(
                    output.reconstruction,
                    batch,
                    patch_size=self.inference_patch_size,
                    patch_stride=self.inference_patch_stride,
                )
                reconstruction = output.reconstruction.cpu().numpy()
                time_values = time_error.cpu().numpy()
                frequency_values = frequency_error.cpu().numpy()
                segment_values = segment_ids.numpy()
                start_values = starts.numpy()
                offsets = np.arange(self.model.seq_len)
                for segment_id in np.unique(segment_values):
                    selected = segment_values == segment_id
                    point_indices = start_values[selected, None] + offsets[None, :]
                    for channel in range(self.n_channels):
                        np.add.at(
                            reconstruction_sums[segment_id][:, channel],
                            point_indices.ravel(),
                            reconstruction[selected, :, channel].ravel(),
                        )
                        np.add.at(
                            time_sums[segment_id][:, channel],
                            point_indices.ravel(),
                            time_values[selected, :, channel].ravel(),
                        )
                        np.add.at(
                            frequency_sums[segment_id][:, channel],
                            point_indices.ravel(),
                            frequency_values[selected, :, channel].ravel(),
                        )
                    np.add.at(
                        counts[segment_id], point_indices.ravel(), np.ones(point_indices.size)
                    )

        segment_ids_out: list[np.ndarray] = []
        indices_out: list[np.ndarray] = []
        observed_out: list[np.ndarray] = []
        reconstructed_out: list[np.ndarray] = []
        time_out: list[np.ndarray] = []
        frequency_out: list[np.ndarray] = []
        for segment_id, segment in enumerate(scaled):
            if np.any(counts[segment_id] == 0):
                raise RuntimeError("scoring windows did not cover every timestamp")
            denominator = counts[segment_id][:, None]
            segment_ids_out.append(np.full(len(segment), segment_id, dtype=np.int64))
            indices_out.append(np.arange(len(segment), dtype=np.int64))
            observed_out.append(self.preprocessor.inverse_transform(segment))
            reconstructed_out.append(
                self.preprocessor.inverse_transform(
                    reconstruction_sums[segment_id] / denominator
                )
            )
            time_out.append(time_sums[segment_id] / denominator)
            frequency_out.append(frequency_sums[segment_id] / denominator)

        time_scores = np.concatenate(time_out)
        frequency_scores = np.concatenate(frequency_out)
        channel_scores = time_scores + self.score_frequency_weight * frequency_scores
        global_scores = channel_scores.mean(axis=1)
        return CATCHScores(
            segment_ids=np.concatenate(segment_ids_out),
            target_indices=np.concatenate(indices_out),
            observed=np.concatenate(observed_out),
            reconstructed=np.concatenate(reconstructed_out),
            time_scores=time_scores,
            frequency_scores=frequency_scores,
            channel_scores=channel_scores,
            global_scores=global_scores,
            is_anomaly=global_scores > threshold,
        )

    def scores_frame(
        self,
        scores: CATCHScores,
        timestamps_by_segment: Sequence[Sequence],
        *,
        threshold: float,
        entity: Optional[str] = None,
    ) -> pd.DataFrame:
        timestamps = np.asarray(
            [
                timestamps_by_segment[int(segment_id)][int(target_index)]
                for segment_id, target_index in zip(
                    scores.segment_ids, scores.target_indices
                )
            ]
        )
        top_index = np.argmax(scores.channel_scores, axis=1)
        safe_global = np.maximum(scores.channel_scores.sum(axis=1), _EPS)
        row_index = np.arange(len(scores.global_scores))
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(timestamps),
                "global_score": scores.global_scores,
                "time_score": scores.time_scores.mean(axis=1),
                "frequency_score": scores.frequency_scores.mean(axis=1),
                "threshold": threshold,
                "is_anomaly": scores.is_anomaly,
                "top_sensor": np.asarray(self.sensor_names, dtype=object)[top_index],
                "top_contribution": (
                    scores.channel_scores[row_index, top_index] / safe_global
                ),
            }
        )
        if entity is not None:
            frame.insert(0, "location", str(entity))
        for channel, sensor in enumerate(self.sensor_names):
            prefix = sensor.replace(" ", "_")
            frame[f"observed__{prefix}"] = scores.observed[:, channel]
            frame[f"reconstructed__{prefix}"] = scores.reconstructed[:, channel]
            frame[f"time_error__{prefix}"] = scores.time_scores[:, channel]
            frame[f"frequency_error__{prefix}"] = scores.frequency_scores[:, channel]
            frame[f"contribution__{prefix}"] = (
                scores.channel_scores[:, channel] / safe_global
            )
        return frame
