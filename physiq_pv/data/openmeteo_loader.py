"""
Open-Meteo weather data loader for PhysiQ-PV.

Reads Open-Meteo Historical Forecast or live forecast NetCDF files and merges
them with the Sentinel energy dataset, producing an xr.Dataset compatible
with PVDataset.

The primary solar variable is ``shortwave_radiation`` (GHI, W/m2), mapped
internally to ``solar_irradiance_poa`` for downstream compatibility.
When ``direct_normal_irradiance`` and ``diffuse_radiation`` are present in the
NetCDF they are passed through so PVDataset can use them instead of Erbs.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial.distance import cdist


def merge_with_openmeteo(
    ds: xr.Dataset,
    openmeteo_path: str | Path,
) -> xr.Dataset:
    """Merge Sentinel energy data with Open-Meteo weather variables.

    Parameters
    ----------
    ds : xr.Dataset
        Sentinel hourly dataset (from ``load_sentinel_hourly``).
    openmeteo_path : str | Path
        Path to an Open-Meteo NetCDF file.  Expected variables:
        ``temperature_2m``, ``wind_speed_10m``, ``shortwave_radiation``.
        Optional: ``direct_normal_irradiance``, ``diffuse_radiation``.

    Returns
    -------
    xr.Dataset
        Input dataset with weather variables added:
        ``temperature_2m``, ``solar_irradiance_poa`` (mapped from
        ``shortwave_radiation``), ``wind_speed_10m``, and optionally
        ``direct_normal_irradiance``, ``diffuse_radiation``.
    """
    openmeteo_path = Path(openmeteo_path)
    ds_om = xr.open_dataset(openmeteo_path)

    required = ["temperature_2m", "shortwave_radiation"]
    for var in required:
        if var not in ds_om:
            raise ValueError(
                f"Open-Meteo NetCDF missing required variable '{var}'. "
                f"Available: {list(ds_om.data_vars)}"
            )
    if "wind_speed_10m" not in ds_om:
        warnings.warn(
            "Open-Meteo NetCDF missing 'wind_speed_10m'; filling with 3.0 m/s",
            stacklevel=2,
        )

    has_dni = "direct_normal_irradiance" in ds_om
    has_dhi = "diffuse_radiation" in ds_om
    if has_dni and has_dhi:
        print("  [openmeteo] DNI + DHI available -> will use direct values (no Erbs)")
    elif has_dni or has_dhi:
        warnings.warn(
            "Open-Meteo has only one of DNI/DHI; both needed for direct use. "
            "Falling back to Erbs decomposition.",
            stacklevel=2,
        )
    else:
        print("  [openmeteo] DNI/DHI not in NetCDF -> Erbs fallback will be used")

    # --- Spatial matching ---
    plant_lats = ds.coords["latitude"].values.astype(float)
    plant_lons = ds.coords["longitude"].values.astype(float)
    valid_coords = np.isfinite(plant_lats) & np.isfinite(plant_lons)
    if not valid_coords.all():
        fleet_lat = float(np.nanmean(plant_lats))
        fleet_lon = float(np.nanmean(plant_lons))
        n_missing = int((~valid_coords).sum())
        print(f"  [openmeteo] {n_missing} plants missing coords -> fleet-mean fallback")
        plant_lats = np.where(valid_coords, plant_lats, fleet_lat)
        plant_lons = np.where(valid_coords, plant_lons, fleet_lon)

    om_lats = ds_om.coords["lat"].values if "lat" in ds_om.coords else ds_om.coords["latitude"].values
    om_lons = ds_om.coords["lon"].values if "lon" in ds_om.coords else ds_om.coords["longitude"].values

    plant_coords = np.column_stack([plant_lats, plant_lons])
    om_coords = np.column_stack([om_lats, om_lons])
    closest = np.argmin(cdist(plant_coords, om_coords, metric="euclidean"), axis=1)

    loc_dim = "location" if "location" in ds_om.dims else list(ds_om.dims)[0]
    print(
        f"  [openmeteo] Matching {ds.sizes['plant']} plants to "
        f"{ds_om.sizes[loc_dim]} Open-Meteo grid points"
    )

    # --- Temporal alignment ---
    t_sentinel = ds.coords["time"].values
    t_om = ds_om.coords["time"].values
    print(f"  [openmeteo] Time: Sentinel {len(t_sentinel)}h, Open-Meteo {len(t_om)}h")

    # Clip Sentinel to Open-Meteo temporal coverage. Open-Meteo may not span the
    # full Sentinel year (e.g. starts in March); training on steps outside the
    # weather coverage would extrapolate. PVDataset does not filter NaN weather
    # (it zero-fills), so drop the uncovered steps here instead.
    in_cov = (t_sentinel >= t_om.min()) & (t_sentinel <= t_om.max())
    n_out = int((~in_cov).sum())
    if n_out > 0:
        print(
            f"  [openmeteo] Clipping {n_out}/{len(t_sentinel)} Sentinel steps "
            f"outside Open-Meteo coverage [{pd.Timestamp(t_om.min()).date()} .. "
            f"{pd.Timestamp(t_om.max()).date()}]"
        )
        ds = ds.isel(time=np.where(in_cov)[0])
        t_sentinel = ds.coords["time"].values

    N_plants = ds.sizes["plant"]
    N_times = ds.sizes["time"]

    temperature_arr = np.full((N_plants, N_times), np.nan, dtype=np.float32)
    solar_arr = np.full((N_plants, N_times), np.nan, dtype=np.float32)
    wind_arr = np.full((N_plants, N_times), np.nan, dtype=np.float32)
    dni_arr = np.full((N_plants, N_times), np.nan, dtype=np.float32) if has_dni else None
    dhi_arr = np.full((N_plants, N_times), np.nan, dtype=np.float32) if has_dhi else None

    for i in range(N_plants):
        loc_idx = closest[i]

        temp_vals = ds_om["temperature_2m"].isel(**{loc_dim: loc_idx}).values
        solar_vals = ds_om["shortwave_radiation"].isel(**{loc_dim: loc_idx}).values

        om_df = pd.DataFrame(
            {"temperature_2m": temp_vals, "shortwave_radiation": solar_vals},
            index=t_om,
        )

        if "wind_speed_10m" in ds_om:
            om_df["wind_speed_10m"] = ds_om["wind_speed_10m"].isel(**{loc_dim: loc_idx}).values
        if has_dni:
            om_df["dni"] = ds_om["direct_normal_irradiance"].isel(**{loc_dim: loc_idx}).values
        if has_dhi:
            om_df["dhi"] = ds_om["diffuse_radiation"].isel(**{loc_dim: loc_idx}).values

        # tolerance keeps out-of-coverage timestamps (e.g. Sentinel Jan/Feb when
        # Open-Meteo starts in March) as NaN instead of silently snapping them to
        # the nearest in-range hour. NaN weather samples are dropped downstream.
        aligned = om_df.reindex(t_sentinel, method="nearest", tolerance=pd.Timedelta("1h"))

        temperature_arr[i, :] = aligned["temperature_2m"].values
        solar_arr[i, :] = aligned["shortwave_radiation"].values
        wind_arr[i, :] = (
            aligned["wind_speed_10m"].values
            if "wind_speed_10m" in aligned
            else 3.0
        )
        if has_dni and dni_arr is not None:
            dni_arr[i, :] = aligned["dni"].values
        if has_dhi and dhi_arr is not None:
            dhi_arr[i, :] = aligned["dhi"].values

    # --- Assign to dataset ---
    ds["temperature_2m"] = xr.DataArray(temperature_arr, dims=["plant", "time"])
    # Map shortwave_radiation -> solar_irradiance_poa for downstream compat.
    ds["solar_irradiance_poa"] = xr.DataArray(solar_arr, dims=["plant", "time"])
    ds["wind_speed_10m"] = xr.DataArray(wind_arr, dims=["plant", "time"])

    if has_dni and dni_arr is not None:
        ds["direct_normal_irradiance"] = xr.DataArray(dni_arr, dims=["plant", "time"])
    if has_dhi and dhi_arr is not None:
        ds["diffuse_radiation"] = xr.DataArray(dhi_arr, dims=["plant", "time"])

    ds.attrs["weather_source"] = "openmeteo"
    print(
        "  [openmeteo] Merged: temperature_2m, solar_irradiance_poa "
        "(from shortwave_radiation), wind_speed_10m"
        + (", direct_normal_irradiance" if has_dni else "")
        + (", diffuse_radiation" if has_dhi else "")
    )
    return ds
