"""Shared validation and reproducibility utilities for anomaly detection."""

from __future__ import annotations

import random
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def detector_features(frame: pd.DataFrame) -> list[str]:
    """Return model inputs, excluding timestamp and reporting-only columns."""
    forbidden = {
        "pv_power_output",
        "label",
        "labels",
        "is_anomaly",
        "anomaly_label",
        "attack",
    }
    leaked = forbidden.intersection(frame.columns)
    if leaked:
        raise ValueError(
            "Supervision or forecast targets must not enter MTGFlow: "
            f"{sorted(leaked)}"
        )
    excluded = {"timestamp", "is_daytime"}
    columns = [column for column in frame.columns if column not in excluded]
    if not columns:
        raise ValueError("Prepared frame has no detector features.")
    return columns


def validate_numeric_features(
    frame: pd.DataFrame,
    features: list[str],
    *,
    method: str,
    allow_nan: bool = False,
) -> None:
    """Fail early on values unsupported by an official implementation."""
    values = frame[features].to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError(f"{method} input contains infinite feature values.")
    if not allow_nan and np.isnan(values).any():
        raise ValueError(f"{method} input contains missing feature values.")


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed all RNGs and, by default, require deterministic Torch kernels."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(deterministic)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def runtime_environment(project_root: str | Path | None = None) -> dict:
    """Capture the software/hardware identity needed to reproduce one run."""
    try:
        import torch

        torch_version = torch.__version__
        cuda_runtime = torch.version.cuda
        cudnn_version = (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        )
        cuda_available = torch.cuda.is_available()
        gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
    except ImportError:
        torch_version = None
        cuda_runtime = None
        cudnn_version = None
        cuda_available = False
        gpu_names = []
        deterministic_algorithms = None

    revision = None
    git_dirty = None
    root = Path(project_root).resolve() if project_root is not None else Path.cwd()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        revision = completed.stdout.strip() or None
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        git_dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch_version,
        "cuda_runtime": cuda_runtime,
        "cudnn": cudnn_version,
        "cuda_available": cuda_available,
        "gpu_names": gpu_names,
        "deterministic_algorithms": deterministic_algorithms,
        "git_commit": revision,
        "git_dirty": git_dirty,
    }


def chronological_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a strictly increasing, duplicate-free timestamp sequence."""
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result = result.sort_values("timestamp").reset_index(drop=True)
    timestamps = pd.DatetimeIndex(result["timestamp"])
    if timestamps.has_duplicates:
        raise ValueError("Detector input contains duplicate timestamps.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Detector timestamps must be monotonically increasing.")
    return result
