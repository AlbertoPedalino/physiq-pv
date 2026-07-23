"""Reusable PVGIS experiment pipeline for the CATCH detector."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from physiq_pv.anomaly_detection.catch import CATCH
from physiq_pv.data.pvgis_catch import (
    DEFAULT_CATCH_SENSORS,
    available_locations,
    extract_location_segments,
    prepare_pvgis_years,
)
from physiq_pv.data.pvgis_dataset import load_pvgis_year, load_pvgis_years
from physiq_pv.reporting.catch_outputs import write_catch_outputs


DEFAULT_OUT_DIR = "outputs/pvgis_catch_2005_2019"


@dataclass(frozen=True)
class PVGISCATCHConfig:
    """Configuration shared by the Python API and command-line runner."""

    pvgis_dir: str
    train_years: tuple[int, ...] = tuple(range(2005, 2019))
    test_year: int = 2019
    file_template: str = "piedmont_pvgis_{year}.nc"
    out_dir: str = DEFAULT_OUT_DIR
    sensors: tuple[str, ...] = tuple(DEFAULT_CATCH_SENSORS)
    max_locations: int | None = None
    seq_len: int = 192
    patch_size: int = 16
    patch_stride: int = 8
    inference_patch_size: int = 32
    inference_patch_stride: int = 1
    cf_dim: int = 64
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 2
    head_dim: int = 64
    d_ff: int = 256
    dropout: float = 0.2
    head_dropout: float = 0.1
    head_layers: int = 3
    temperature: float = 0.07
    mask_source: str = "projected"
    frequency_loss_weight: float = 0.005
    clustering_weight: float = 0.005
    regularization_weight: float = 0.0025
    score_frequency_weight: float = 0.05
    contamination: float = 0.01
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-4
    mask_learning_rate: float = 1e-5
    validation_split: float = 0.2
    patience: int = 3
    training_window_stride: int = 1
    scoring_window_stride: int | None = None
    model_steps_per_mask: int | None = None
    gradient_clip: float | None = None
    lr_adjustment: str = "type1"
    minimum_oom_batch_size: int = 8
    device: str = "cpu"
    seed: int = 42
    verbose: bool = True

    def __post_init__(self) -> None:
        if not self.pvgis_dir:
            raise ValueError("pvgis_dir is required")
        if not self.train_years or len(set(self.train_years)) != len(self.train_years):
            raise ValueError("train_years must be non-empty and unique")
        if self.test_year in self.train_years:
            raise ValueError("test_year must not be included in train_years")
        if not self.sensors or len(set(self.sensors)) != len(self.sensors):
            raise ValueError("sensors must be non-empty and unique")
        if self.max_locations is not None and self.max_locations < 1:
            raise ValueError("max_locations must be >= 1")
        if self.seq_len < 2 or not 1 <= self.patch_size <= self.seq_len:
            raise ValueError("patch_size must be in [1, seq_len]")
        if self.patch_stride < 1 or self.inference_patch_stride < 1:
            raise ValueError("patch strides must be >= 1")
        if min(self.cf_dim, self.d_model, self.n_heads, self.head_dim) < 1:
            raise ValueError("model dimensions must be >= 1")
        if self.mask_source not in {"projected", "raw"}:
            raise ValueError("mask_source must be 'projected' or 'raw'")
        if not 0.0 < self.contamination < 1.0:
            raise ValueError("contamination must be in (0, 1)")
        if self.model_steps_per_mask is not None and self.model_steps_per_mask < 1:
            raise ValueError("model_steps_per_mask must be >= 1")
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be > 0 when provided")
        if self.lr_adjustment not in {"type1", "constant"}:
            raise ValueError("lr_adjustment must be 'type1' or 'constant'")
        if self.minimum_oom_batch_size < 1:
            raise ValueError("minimum_oom_batch_size must be >= 1")

    def detector(self) -> CATCH:
        return CATCH(
            self.sensors,
            seq_len=self.seq_len,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            inference_patch_size=self.inference_patch_size,
            inference_patch_stride=self.inference_patch_stride,
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
            mask_source=self.mask_source,
            frequency_loss_weight=self.frequency_loss_weight,
            clustering_weight=self.clustering_weight,
            regularization_weight=self.regularization_weight,
            score_frequency_weight=self.score_frequency_weight,
            contamination=self.contamination,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            mask_learning_rate=self.mask_learning_rate,
            validation_split=self.validation_split,
            patience=self.patience,
            training_window_stride=self.training_window_stride,
            scoring_window_stride=self.scoring_window_stride,
            model_steps_per_mask=self.model_steps_per_mask,
            gradient_clip=self.gradient_clip,
            lr_adjustment=self.lr_adjustment,
            minimum_oom_batch_size=self.minimum_oom_batch_size,
            device=self.device,
            seed=self.seed,
            verbose=self.verbose,
        )

    def metadata(self, n_locations: int) -> dict:
        return {
            "method": "CATCH",
            "paper": "Wu et al., ICLR 2025",
            "implementation_policy": (
                "paper-first; official repository for unspecified details; "
                "leakage-safe PVGIS protocol"
            ),
            "repository": "https://github.com/decisionintelligence/CATCH",
            "repository_reference_commit": (
                "3647c69be5eb56649b072596cf89098e689e20c3"
            ),
            "label_unaware": True,
            "threshold_calibration": "training-only quantile",
            "train_years": list(self.train_years),
            "test_year": int(self.test_year),
            "sensors": list(self.sensors),
            "n_locations": int(n_locations),
            "seq_len": int(self.seq_len),
            "patch_size": int(self.patch_size),
            "patch_stride": int(self.patch_stride),
            "inference_patch_size": int(self.inference_patch_size),
            "inference_patch_stride": int(self.inference_patch_stride),
            "cf_dim": int(self.cf_dim),
            "d_model": int(self.d_model),
            "n_layers": int(self.n_layers),
            "n_heads": int(self.n_heads),
            "head_dim": int(self.head_dim),
            "d_ff": int(self.d_ff),
            "head_layers": int(self.head_layers),
            "dropout": float(self.dropout),
            "mask_source": self.mask_source,
            "frequency_loss_weight": float(self.frequency_loss_weight),
            "clustering_weight": float(self.clustering_weight),
            "regularization_weight": float(self.regularization_weight),
            "score_frequency_weight": float(self.score_frequency_weight),
            "contamination": float(self.contamination),
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate),
            "mask_learning_rate": float(self.mask_learning_rate),
            "validation_split": float(self.validation_split),
            "training_window_stride": int(self.training_window_stride),
            "scoring_window_stride": int(
                self.seq_len
                if self.scoring_window_stride is None
                else self.scoring_window_stride
            ),
            "scoring_window_policy": (
                "repository non-overlap with final-window tail coverage"
            ),
            "model_steps_per_mask": self.model_steps_per_mask,
            "model_steps_policy": (
                "explicit override"
                if self.model_steps_per_mask is not None
                else "repository min(max(number_of_batches//10, 1), 100)"
            ),
            "gradient_clip": self.gradient_clip,
            "lr_adjustment": self.lr_adjustment,
            "minimum_oom_batch_size": int(self.minimum_oom_batch_size),
            "seed": int(self.seed),
            "device": self.device,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }


@dataclass(frozen=True)
class LocationCATCHResult:
    scores: pd.DataFrame
    summary: dict


def load_protocol_datasets(
    config: PVGISCATCHConfig,
) -> tuple[dict[int, xr.Dataset], dict[int, xr.Dataset], dict[int, xr.Dataset]]:
    """Load complete train/test years and validate a shared sensor schema."""

    train_map = load_pvgis_years(
        config.pvgis_dir, list(config.train_years), config.file_template
    )
    missing = sorted(set(config.train_years) - set(train_map))
    if missing:
        for dataset in train_map.values():
            dataset.close()
        raise FileNotFoundError(
            "CATCH requires the complete requested training interval; "
            f"missing years: {missing}"
        )
    test_path = Path(config.pvgis_dir) / config.file_template.format(year=config.test_year)
    try:
        test_dataset = load_pvgis_year(str(test_path))
        all_map = prepare_pvgis_years(
            {**train_map, config.test_year: test_dataset}, config.sensors
        )
    except Exception:
        for dataset in train_map.values():
            dataset.close()
        if "test_dataset" in locals():
            test_dataset.close()
        raise
    return (
        {year: all_map[year] for year in config.train_years},
        {config.test_year: all_map[config.test_year]},
        all_map,
    )


def fit_score_location(
    train_map: dict[int, xr.Dataset],
    test_map: dict[int, xr.Dataset],
    location,
    config: PVGISCATCHConfig,
) -> LocationCATCHResult:
    train_segments, _ = extract_location_segments(train_map, location, config.sensors)
    test_segments, test_timestamps = extract_location_segments(
        test_map, location, config.sensors
    )
    detector = config.detector().fit(train_segments)
    scores = detector.scores_frame(
        detector.score_segments(test_segments),
        test_timestamps,
        entity=str(location),
    )
    if detector.model is None or detector.trainer is None:
        raise RuntimeError("CATCH model or trainer missing after fit")
    summary = {
        "location": str(location),
        "n_train_windows": int(detector.n_train_windows),
        "n_test_points": int(len(scores)),
        "n_anomalies": int(scores["is_anomaly"].sum()),
        "anomaly_fraction": float(scores["is_anomaly"].mean()),
        "threshold": float(detector.threshold),
        "epochs_completed": int(len(detector.history)),
        "effective_batch_size": int(detector.trainer.effective_batch_size),
        "effective_model_steps_per_mask": int(
            detector.trainer.effective_model_steps_per_mask
        ),
        "best_valid_loss": float(
            min(record["valid_loss"] for record in detector.history)
        ),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in detector.model.parameters())
        ),
    }
    return LocationCATCHResult(scores=scores, summary=summary)


def run_pvgis_catch(config: PVGISCATCHConfig) -> dict[str, Path]:
    """Fit one CATCH model per PVGIS location and write stable outputs."""

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    print(
        f"[1/4] Loading PVGIS train={config.train_years[0]}-{config.train_years[-1]} "
        f"test={config.test_year}"
    )
    train_map, test_map, all_map = load_protocol_datasets(config)
    locations = available_locations(all_map)
    if config.max_locations is not None:
        locations = locations[: config.max_locations]
    score_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    print(
        f"[2/4] Fitting one label-unaware CATCH model for each of "
        f"{len(locations)} locations"
    )
    try:
        for index, location in enumerate(locations, start=1):
            print(f"  [{index}/{len(locations)}] location={location}")
            result = fit_score_location(train_map, test_map, location, config)
            score_frames.append(result.scores)
            summary_rows.append(result.summary)
    finally:
        for dataset in all_map.values():
            dataset.close()

    print("[3/4] Writing scores, detections, intervals, summary, and metadata")
    paths = write_catch_outputs(
        config.out_dir,
        pd.concat(score_frames, ignore_index=True),
        pd.DataFrame(summary_rows),
        config.metadata(len(locations)),
    )
    print(f"[4/4] Done: {config.out_dir}")
    return paths
