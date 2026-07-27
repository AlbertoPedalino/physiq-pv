"""Reusable PVGIS orchestration for M2AD, independent of argparse/notebooks."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import xarray as xr

from physiq_pv.anomaly_detection.m2ad import M2AD
from physiq_pv.data.pvgis_dataset import load_pvgis_year, load_pvgis_years
from physiq_pv.data.pvgis_m2ad import (
    DEFAULT_M2AD_SENSORS,
    available_locations,
    extract_location_segments,
    prepare_pvgis_years,
)
from physiq_pv.reporting.m2ad_outputs import (
    canonical_detector_scores,
    write_m2ad_outputs,
)


DEFAULT_OUT_DIR = "outputs/pvgis_m2ad_2005_2019"


@dataclass(frozen=True)
class PVGISM2ADConfig:
    """Typed experiment configuration shared by CLI, notebook, and Python API."""

    pvgis_dir: str
    train_years: tuple[int, ...] = tuple(range(2005, 2019))
    export_train_years: tuple[int, ...] | None = None
    test_year: int = 2019
    file_template: str = "piedmont_pvgis_{year}.nc"
    out_dir: str = DEFAULT_OUT_DIR
    sensors: tuple[str, ...] = tuple(DEFAULT_M2AD_SENSORS)
    max_locations: int | None = None
    window_size: int = 120
    horizon: int = 1
    hidden_size: int = 80
    n_layers: int = 2
    dropout: float = 0.2
    error: str = "area"
    area_half_window: int = 2
    ewma_com: float = 10.0
    gmm_components: int | str = "bic"
    max_components: int = 3
    significance: float = 0.001
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    validation_split: float = 0.2
    patience: int = 5
    seed: int = 42
    device: str = "cpu"
    verbose: bool = True

    def __post_init__(self) -> None:
        if not self.pvgis_dir:
            raise ValueError("pvgis_dir is required")
        if not self.train_years or len(set(self.train_years)) != len(self.train_years):
            raise ValueError("train_years must be non-empty and unique")
        if self.test_year in self.train_years:
            raise ValueError("test_year must not be included in train_years")
        if self.export_train_years is not None:
            if not self.export_train_years:
                raise ValueError("export_train_years must not be empty")
            missing_export_years = sorted(
                set(self.export_train_years) - set(self.train_years)
            )
            if missing_export_years:
                raise ValueError(
                    "export_train_years must be included in train_years; "
                    f"missing: {missing_export_years}"
                )
        if not self.sensors or len(set(self.sensors)) != len(self.sensors):
            raise ValueError("sensors must be non-empty and unique")
        if self.max_locations is not None and self.max_locations < 1:
            raise ValueError("max_locations must be >= 1")
        if self.window_size < 1 or self.horizon < 1:
            raise ValueError("window_size and horizon must be >= 1")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1")
        if not 0.0 < self.significance < 1.0:
            raise ValueError("significance must be in (0, 1)")

    def detector(self) -> M2AD:
        """Build one fresh detector; useful for custom per-asset workflows."""
        return M2AD(
            self.sensors,
            window_size=self.window_size,
            horizon=self.horizon,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            dropout=self.dropout,
            error=self.error,
            area_half_window=self.area_half_window,
            ewma_com=self.ewma_com,
            n_components=self.gmm_components,
            max_components=self.max_components,
            significance=self.significance,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            validation_split=self.validation_split,
            patience=self.patience,
            device=self.device,
            seed=self.seed,
            verbose=self.verbose,
        )

    @property
    def resolved_export_train_years(self) -> tuple[int, ...]:
        if self.export_train_years is not None:
            return self.export_train_years
        return self.train_years[-min(3, len(self.train_years)) :]

    def metadata(self, n_locations: int) -> dict:
        return {
            "method": "M2AD",
            "paper": "Alnegheimish et al., AISTATS 2025",
            "label_unaware": True,
            "train_years": list(self.train_years),
            "export_train_years": list(self.resolved_export_train_years),
            "test_year": int(self.test_year),
            "sensors": list(self.sensors),
            "n_locations": int(n_locations),
            "window_size": int(self.window_size),
            "horizon": int(self.horizon),
            "hidden_size": int(self.hidden_size),
            "n_layers": int(self.n_layers),
            "dropout": float(self.dropout),
            "error": self.error,
            "area_half_window": int(self.area_half_window),
            "ewma_com": float(self.ewma_com),
            "gmm_components": self.gmm_components,
            "max_components": int(self.max_components),
            "significance": float(self.significance),
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate),
            "validation_split": float(self.validation_split),
            "patience": int(self.patience),
            "seed": int(self.seed),
            "device": self.device,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }


@dataclass(frozen=True)
class LocationM2ADResult:
    scores: pd.DataFrame
    train_scores: pd.DataFrame
    summary: dict


def load_protocol_datasets(
    config: PVGISM2ADConfig,
) -> tuple[dict[int, xr.Dataset], dict[int, xr.Dataset], dict[int, xr.Dataset]]:
    """Load and validate complete train/test years; caller owns dataset closing."""
    train_map = load_pvgis_years(
        config.pvgis_dir, list(config.train_years), config.file_template
    )
    missing = sorted(set(config.train_years) - set(train_map))
    if missing:
        for dataset in train_map.values():
            dataset.close()
        raise FileNotFoundError(
            "M2AD requires the complete requested training interval; "
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
    export_train_map: dict[int, xr.Dataset],
    test_map: dict[int, xr.Dataset],
    location,
    config: PVGISM2ADConfig,
) -> LocationM2ADResult:
    """Fit and score one asset; no filesystem output and no global state."""
    train_segments, _ = extract_location_segments(
        train_map, location, config.sensors
    )
    export_train_segments, export_train_timestamps = extract_location_segments(
        export_train_map, location, config.sensors
    )
    test_segments, test_timestamps = extract_location_segments(
        test_map, location, config.sensors
    )
    detector = config.detector().fit(train_segments)
    train_scores = detector.scores_frame(
        detector.score_segments(export_train_segments),
        export_train_timestamps,
        entity=str(location),
    )
    scores = detector.scores_frame(
        detector.score_segments(test_segments),
        test_timestamps,
        entity=str(location),
    )
    summary = {
        "location": str(location),
        "n_train_windows": int(detector.n_train_windows),
        "n_export_train_points": int(len(train_scores)),
        "n_export_train_anomalies": int(train_scores["is_anomaly"].sum()),
        "n_test_windows": int(len(scores)),
        "n_anomalies": int(scores["is_anomaly"].sum()),
        "anomaly_fraction": float(scores["is_anomaly"].mean()),
        "gamma_shape": float(detector.gamma_shape),
        "gamma_scale": float(detector.gamma_scale),
        "threshold": float(detector.threshold),
        "gmm_components": detector.selected_components,
        "epochs_completed": int(len(detector.history)),
        "best_valid_loss": float(
            min(record["valid_loss"] for record in detector.history)
        ),
    }
    return LocationM2ADResult(
        scores=scores,
        train_scores=canonical_detector_scores(train_scores),
        summary=summary,
    )


def run_pvgis_m2ad(config: PVGISM2ADConfig) -> dict[str, Path]:
    """Execute the multi-location protocol and write its stable output contract."""
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
    train_score_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    export_train_map = {
        year: train_map[year] for year in config.resolved_export_train_years
    }
    print(
        f"[2/4] Fitting one label-unaware M2AD model for each of "
        f"{len(locations)} locations"
    )
    try:
        for index, location in enumerate(locations, start=1):
            print(f"  [{index}/{len(locations)}] location={location}")
            result = fit_score_location(
                train_map, export_train_map, test_map, location, config
            )
            score_frames.append(result.scores)
            train_score_frames.append(result.train_scores)
            summary = dict(result.summary)
            # Stable scalar CSV representation; the Python result stays structured.
            summary["gmm_components"] = json.dumps(
                summary["gmm_components"], sort_keys=True
            )
            summary_rows.append(summary)
    finally:
        for dataset in all_map.values():
            dataset.close()

    print("[3/4] Writing scores, detections, intervals, summary, and metadata")
    scores = pd.concat(score_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    paths = write_m2ad_outputs(
        config.out_dir,
        scores,
        pd.concat(train_score_frames, ignore_index=True),
        summary,
        config.metadata(len(locations)),
    )
    print(f"[4/4] Done: {config.out_dir}")
    return paths
