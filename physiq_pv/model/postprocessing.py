from collections.abc import Mapping
from typing import Any

import numpy as np


def apply_pv_calibration_np(
    pred_pv: np.ndarray,
    calibration: Mapping[str, Any] | None = None,
    floor: float = 0.0,
) -> np.ndarray:
    """Apply optional linear PV calibration and enforce non-negative output."""
    out = np.asarray(pred_pv, dtype=np.float64)
    if calibration and calibration.get("enabled", False):
        slope = float(calibration.get("slope", 1.0))
        intercept = float(calibration.get("intercept", 0.0))
        out = slope * out + intercept
    return np.clip(out, floor, None)
