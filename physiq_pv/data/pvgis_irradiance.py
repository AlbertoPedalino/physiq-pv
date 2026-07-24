"""PVGIS plane-of-array irradiance fallback."""
from __future__ import annotations

import numpy as np
import xarray as xr

POA_VAR = "solar_irradiance_poa"
DIRECT_TILTED_VAR = "direct_irradiance_tilted"
DIFFUSE_TILTED_VAR = "diffuse_irradiance_tilted"
NEGATIVE_IRRADIANCE_TOLERANCE_WM2 = 20.0


def _has_signal(da: xr.DataArray) -> bool:
    vals = np.asarray(da.values, dtype=np.float32)
    if vals.size == 0 or not np.isfinite(vals).any():
        return False
    return float(np.nanmax(vals)) > 0.0


def with_effective_poa(ds: xr.Dataset) -> xr.Dataset:
    """Always reconstruct POA from the PVGIS tilted direct and diffuse fields."""
    missing = [
        name for name in (DIRECT_TILTED_VAR, DIFFUSE_TILTED_VAR) if name not in ds
    ]
    if missing:
        raise ValueError(
            "PVGIS POA reconstruction requires tilted irradiance components; "
            f"missing: {missing}."
        )

    def _sanitize_component(name: str) -> xr.DataArray:
        component = ds[name]
        values = np.asarray(component.values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"PVGIS variable {name!r} contains non-finite values.")
        minimum = float(values.min(initial=0.0))
        if minimum < -NEGATIVE_IRRADIANCE_TOLERANCE_WM2:
            raise ValueError(
                f"PVGIS variable {name!r} has implausible negative irradiance "
                f"({minimum:.3f} W/m²; allowed numerical tolerance is "
                f"{NEGATIVE_IRRADIANCE_TOLERANCE_WM2:g} W/m²)."
            )
        sanitized = component.clip(min=0.0)
        sanitized.attrs = dict(component.attrs)
        if minimum < 0.0:
            sanitized.attrs["negative_values_clipped_to_zero"] = True
            sanitized.attrs["minimum_before_clipping_wm2"] = minimum
        return sanitized

    direct = _sanitize_component(DIRECT_TILTED_VAR)
    diffuse = _sanitize_component(DIFFUSE_TILTED_VAR)

    poa = direct + diffuse
    poa.name = POA_VAR
    poa.attrs = {
        "long_name": "Reconstructed plane-of-array irradiance",
        "units": "W m-2",
        "source": f"{DIRECT_TILTED_VAR} + {DIFFUSE_TILTED_VAR}",
    }
    if not _has_signal(poa):
        raise ValueError(
            "PVGIS tilted direct + diffuse irradiance has no positive signal."
        )
    return ds.assign({POA_VAR: poa})
