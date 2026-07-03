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
    """Use stored POA, or replace empty POA with tilted direct + diffuse."""
    if POA_VAR in ds and _has_signal(ds[POA_VAR]):
        return ds

    if DIRECT_TILTED_VAR in ds and DIFFUSE_TILTED_VAR in ds:
        poa = ds[DIRECT_TILTED_VAR] + ds[DIFFUSE_TILTED_VAR]
        poa.name = POA_VAR
        poa.attrs = {"long_name": "Solar Irradiance on Plane of Array", "units": "W m-2"}
        if _has_signal(poa):
            return ds.assign({POA_VAR: poa})

    raise ValueError(
        f"PVGIS dataset has no usable {POA_VAR!r}; expected it directly or as "
        f"{DIRECT_TILTED_VAR!r}+{DIFFUSE_TILTED_VAR!r}."
    )
