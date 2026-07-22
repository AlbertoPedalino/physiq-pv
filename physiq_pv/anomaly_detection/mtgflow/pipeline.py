"""Leakage-free training, checkpointing, and scoring for MTGFlow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..common import (
    chronological_frame,
    detector_features,
    runtime_environment,
    seed_everything,
    validate_numeric_features,
)
from .config import REFERENCE_CONFIG, REFERENCE_SEEDS
from .model import MTGFlow
from .result import MTGFlowResult


def load_mtgflow_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[MTGFlow, dict]:
    """Reconstruct a trained model and its scaler/config from a trusted bundle."""
    import torch

    resolved = Path(checkpoint_path).resolve()
    torch_device = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    payload = torch.load(resolved, map_location=torch_device, weights_only=False)
    if payload.get("format_version") != 2 or payload.get("model_class") != "MTGFlow":
        raise ValueError(f"Unsupported MTGFlow checkpoint: {resolved}")
    model = MTGFlow(**payload["model_config"]).to(torch_device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def _regular_window_starts(
    timestamps: pd.DatetimeIndex,
    *,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """Return strided starts without crossing a missing/irregular sample."""
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be positive.")
    candidates = np.arange(0, len(timestamps) - window_size + 1, stride, dtype=int)
    if candidates.size == 0 or window_size == 1:
        return candidates
    deltas = np.diff(timestamps.asi8)
    positive = deltas[deltas > 0]
    if positive.size == 0:
        raise ValueError("Cannot infer a positive sampling cadence.")
    unique, counts = np.unique(positive, return_counts=True)
    cadence = unique[np.argmax(counts)]
    irregular = deltas != cadence
    prefix = np.concatenate(([0], np.cumsum(irregular, dtype=np.int64)))
    valid = prefix[candidates + window_size - 1] == prefix[candidates]
    return candidates[valid]


def fit_and_score_mtgflow(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    validation: pd.DataFrame | None = None,
    epochs: int = REFERENCE_CONFIG.epochs,
    window_size: int = REFERENCE_CONFIG.window_size,
    train_stride: int = REFERENCE_CONFIG.train_stride,
    score_stride: int = REFERENCE_CONFIG.score_stride,
    batch_size: int = REFERENCE_CONFIG.batch_size,
    lr: float = REFERENCE_CONFIG.learning_rate,
    weight_decay: float = REFERENCE_CONFIG.weight_decay,
    n_blocks: int = REFERENCE_CONFIG.n_blocks,
    hidden_size: int = REFERENCE_CONFIG.hidden_size,
    n_hidden: int = REFERENCE_CONFIG.n_hidden,
    attention_dropout: float = REFERENCE_CONFIG.attention_dropout,
    device: str = "cuda",
    seed: int = REFERENCE_SEEDS[0],
    checkpoint_path: str | Path | None = None,
) -> MTGFlowResult:
    """Fit MTGFlow base using the paper equations and training data only.

    When supplied, ``validation`` is checked for schema/numerical validity but
    excluded from normalisation, optimisation and threshold calibration.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least one.")
    seed_everything(seed)

    import torch
    from torch.nn.utils import clip_grad_value_
    from torch.utils.data import DataLoader, Dataset

    train = chronological_frame(train)
    test = chronological_frame(test)
    if validation is not None:
        validation = chronological_frame(validation)
    feature_names = detector_features(train)
    splits = (train, test) if validation is None else (train, validation, test)
    for split in splits:
        missing = sorted(set(feature_names) - set(split.columns))
        if missing:
            raise ValueError(f"MTGFlow split is missing features: {missing}")
        validate_numeric_features(split, feature_names, method="MTGFlow")

    # Equation 5, fitted on training only to prevent future-data leakage.
    mean = train[feature_names].mean().to_numpy(dtype=np.float64, copy=True)
    std = train[feature_names].std(ddof=0).to_numpy(dtype=np.float64, copy=True)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0

    def values(frame: pd.DataFrame) -> np.ndarray:
        array = frame[feature_names].to_numpy(dtype=np.float32)
        return ((array - mean) / std).astype(np.float32)

    class Windows(Dataset):
        def __init__(self, frame: pd.DataFrame, stride: int):
            self.data = values(frame)
            self.times = pd.DatetimeIndex(frame["timestamp"])
            self.starts = _regular_window_starts(
                self.times, window_size=window_size, stride=stride
            )

        def __len__(self) -> int:
            return len(self.starts)

        def __getitem__(self, item: int) -> torch.Tensor:
            start = self.starts[item]
            block = self.data[start : start + window_size]
            return torch.from_numpy(block.T[:, :, None])

        @property
        def endpoints(self) -> pd.DatetimeIndex:
            return self.times[self.starts + window_size - 1]

        @property
        def startpoints(self) -> pd.DatetimeIndex:
            return self.times[self.starts]

    train_fit_ds = Windows(train, train_stride)
    train_score_ds = Windows(train, score_stride)
    test_score_ds = Windows(test, score_stride)
    if min(len(train_fit_ds), len(train_score_ds), len(test_score_ds)) == 0:
        raise ValueError("MTGFlow split has no complete regular window.")

    train_loader = DataLoader(train_fit_ds, batch_size=batch_size, shuffle=True)
    torch_device = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model_config = {
        "n_blocks": n_blocks,
        "input_size": 1,
        "hidden_size": hidden_size,
        "n_hidden": n_hidden,
        "window_size": window_size,
        "n_entities": len(feature_names),
        "attention_dropout": attention_dropout,
    }
    model = MTGFlow(**model_config).to(torch_device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    for _epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = -model(batch.to(torch_device))
            loss.backward()
            clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()

    checkpoint_resolved = None
    if checkpoint_path is not None:
        checkpoint_resolved = Path(checkpoint_path).resolve()
        checkpoint_resolved.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 2,
                "model_class": "MTGFlow",
                "model_state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "model_config": model_config,
                "normalization": {
                    "kind": "training_only_zscore",
                    "mean": mean,
                    "std": std,
                },
                "features": feature_names,
                "training": {
                    "epochs": epochs,
                    "train_stride": train_stride,
                    "score_stride": score_stride,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "weight_decay": weight_decay,
                    "seed": seed,
                    "checkpoint_selection": "final_fixed_epoch",
                },
                "environment": runtime_environment(),
            },
            checkpoint_resolved,
        )

    def score(
        dataset: Windows,
    ) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, np.ndarray, np.ndarray]:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        global_result: list[np.ndarray] = []
        entity_result: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                entity_log_prob, _, _ = model.likelihood_components(
                    batch.to(torch_device)
                )
                # Paper Eq. 14: each entity contributes 1/K of its negative
                # log-likelihood, and the global score is their sum.
                entity_score = -entity_log_prob / len(feature_names)
                entity_result.append(entity_score.cpu().numpy())
                global_result.append(entity_score.sum(dim=1).cpu().numpy())
        return (
            dataset.startpoints,
            dataset.endpoints,
            np.concatenate(global_result),
            np.concatenate(entity_result),
        )

    train_start_ts, train_ts, train_score, train_entity_score = score(train_score_ds)
    test_start_ts, test_ts, test_score, test_entity_score = score(test_score_ds)
    reference_other_mts_defaults = all(
        (
            epochs == REFERENCE_CONFIG.epochs,
            window_size == REFERENCE_CONFIG.window_size,
            train_stride == REFERENCE_CONFIG.train_stride,
            score_stride == REFERENCE_CONFIG.score_stride,
            batch_size == REFERENCE_CONFIG.batch_size,
            lr == REFERENCE_CONFIG.learning_rate,
            weight_decay == REFERENCE_CONFIG.weight_decay,
            n_blocks == REFERENCE_CONFIG.n_blocks,
            hidden_size == REFERENCE_CONFIG.hidden_size,
            n_hidden == REFERENCE_CONFIG.n_hidden,
            attention_dropout == REFERENCE_CONFIG.attention_dropout,
        )
    )
    return MTGFlowResult(
        train_ts,
        train_score,
        test_ts,
        test_score,
        {
            "backend": "mtgflow_base",
            "alignment_reference": "paper_v2_and_official_repository",
            "runtime_dependency_on_official_repo": False,
            "normalization": "training_only_zscore",
            "validation_supplied": validation is not None,
            "validation_used_for_training": False,
            "validation_time_range": (
                None
                if validation is None
                else [
                    validation["timestamp"].iloc[0].isoformat(),
                    validation["timestamp"].iloc[-1].isoformat(),
                ]
            ),
            "train_time_range": [
                train["timestamp"].iloc[0].isoformat(),
                train["timestamp"].iloc[-1].isoformat(),
            ],
            "test_time_range": [
                test["timestamp"].iloc[0].isoformat(),
                test["timestamp"].iloc[-1].isoformat(),
            ],
            "checkpoint_selection": "final_fixed_epoch",
            "test_labels_used": False,
            "train_stride": train_stride,
            "score_stride": score_stride,
            "window_score_semantics": "whole_window_assigned_to_window_end",
            "configuration_profile": (
                "reference_other_mts_defaults"
                if reference_other_mts_defaults
                else "custom_configuration"
            ),
            "scoring_profile": (
                "reference_window_sampling"
                if window_size == REFERENCE_CONFIG.window_size
                and score_stride == REFERENCE_CONFIG.score_stride
                else (
                    "dense_hourly_window_adaptation"
                    if window_size == REFERENCE_CONFIG.window_size and score_stride == 1
                    else "custom_window_sampling"
                )
            ),
            "entity_score_definition": "negative_log_likelihood_divided_by_K",
            "global_score_definition": "sum_of_entity_contributions",
            "epochs": epochs,
            "window_size": window_size,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "n_blocks": n_blocks,
            "hidden_size": hidden_size,
            "n_hidden": n_hidden,
            "attention_dropout": attention_dropout,
            "device": str(torch_device),
            "seed": seed,
            "features": feature_names,
            "checkpoint": None if checkpoint_resolved is None else str(checkpoint_resolved),
            "environment": runtime_environment(),
        },
        train_entity_scores=train_entity_score,
        test_entity_scores=test_entity_score,
        entity_names=tuple(feature_names),
        train_window_starts=train_start_ts,
        test_window_starts=test_start_ts,
    )
