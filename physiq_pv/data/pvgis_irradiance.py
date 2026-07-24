"""PVGIS plane-of-array irradiance fallback."""
from __future__ import annotations

import numpy as np
import xarray as xr

POA_VAR = "solar_irradiance_poa"
DIRECT_TILTED_VAR = "direct_irradiance_tilted"
DIFFUSE_TILTED_VAR = "diffuse_irradiance_tilted"


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

    direct = ds[DIRECT_TILTED_VAR]
    diffuse = ds[DIFFUSE_TILTED_VAR]
    for name, component in (
        (DIRECT_TILTED_VAR, direct),
        (DIFFUSE_TILTED_VAR, diffuse),
    ):
        values = np.asarray(component.values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"PVGIS variable {name!r} contains non-finite values.")
        if float(values.min(initial=0.0)) < 0.0:
            raise ValueError(f"PVGIS variable {name!r} contains negative irradiance.")

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
