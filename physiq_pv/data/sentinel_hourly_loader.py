"""
Load hourly Sentinel/SCADA energy data from individual UPN CSV files.

Data path: /data/SentinelPV/energy_data/piemonte_energy_data/single_ups/
File format: YYYY_UPN_XXXXXXX_01.csv
  - Columns: date (DD/MM/YY HH:MM), ENERGIA (kW)
  - Multiple readings per timestamp (3x per hour from different inverters/sensors)
  - Period: 2019-03-01 to 2019-12-31 (Jun/Aug absent from source files)

Returns xr.Dataset with:
  - ENERGIA (plant, time) — hourly energy production [kW]
  - Coordinates: plant_id, time, lat, lon, eta_base (from plant_mapping)
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple


def load_sentinel_hourly(
    sentinel_dir: str = "/data/SentinelPV/energy_data/piemonte_energy_data/single_ups",
    year: int = 2019,
    plant_mapping_path: Optional[str] = None,
    energy_coords_path: Optional[str] = None,
    upn_list: Optional[List[str]] = None,
    source_timezone: str = "Europe/Rome",
) -> xr.Dataset:
    """
    Load hourly Sentinel energy data from individual UPN CSV files.

    Parameters
    ----------
    sentinel_dir : str
        Root directory containing YYYY_UPN_*.csv files
    year : int
        Year to load (default 2019)
    plant_mapping_path : str, optional
        Path to plant_mapping.csv (for plant_id ↔ UPN mapping)
    energy_coords_path : str, optional
        Path to energy_with_coordinates.csv (for lat/lon, Potenza di picco)
    upn_list : list[str], optional
        List of UPN codes to load. If None, loads all available.
    source_timezone : str
        Timezone represented by naive SCADA timestamps. Values are converted
        to UTC and stored timezone-naive for xarray/NetCDF compatibility.

    Returns
    -------
    xr.Dataset
        Variables: ENERGIA (plant, time)
        Coords: plant_id, time, latitude, longitude, eta_base
    """

    sentinel_path = Path(sentinel_dir)
    if not sentinel_path.exists():
        raise FileNotFoundError(f"Sentinel directory not found: {sentinel_dir}")

    # Load plant mapping (plant_id ↔ UPN)
    plant_map = {}
    upn_to_coords = {}
    eta_base_map = {}

    if plant_mapping_path:
        pm = pd.read_csv(plant_mapping_path)
        for _, row in pm.iterrows():
            if pd.notna(row.get("Codice UP")):
                plant_map[row["Codice UP"]] = row.get("plant_id", None)
                eta_base_map[row["Codice UP"]] = row.get("eta_base", 0.15)
                if pd.notna(row.get("Latitude")):
                    upn_to_coords[row["Codice UP"]] = (
                        row["Latitude"],
                        row["Longitude"],
                    )

    if energy_coords_path:
        ec = pd.read_csv(energy_coords_path)
        for _, row in ec.iterrows():
            upn = row.get("Codice UP", None)
            if upn and pd.notna(upn):
                if upn not in upn_to_coords and pd.notna(row.get("Latitude")):
                    upn_to_coords[upn] = (row["Latitude"], row["Longitude"])

    # Find all CSV files for given year
    pattern = f"{year}_UPN_*.csv"
    csv_files = sorted(sentinel_path.glob(pattern))
    print(f"  Found {len(csv_files)} Sentinel files for year {year}")

    if not csv_files:
        raise FileNotFoundError(
            f"No Sentinel CSV files found in {sentinel_dir} matching {pattern}"
        )

    # Extract UPN codes from filenames
    upn_codes = []
    for f in csv_files:
        # Filename format: 2019_UPN_0110065_01.csv
        parts = f.stem.split("_")
        if len(parts) >= 3:
            upn_code = parts[2]
            upn = f"UPN_{upn_code}_01"
            upn_codes.append((upn, f))

    # Filter if upn_list provided
    if upn_list:
        upn_codes = [(upn, f) for upn, f in upn_codes if upn in upn_list]

    print(f"  Loading {len(upn_codes)} UPN plants...")

    # Load data from each UPN file
    all_data = []
    all_upns = []
    all_plant_ids = []
    all_lats = []
    all_lons = []
    all_eta_base = []
    all_dfs = []  # Keep dataframes for proper time alignment

    for upn, csv_path in upn_codes:
        try:
            df = pd.read_csv(csv_path)

            # Parse date column (format: DD/MM/YY HH:MM)
            timestamp = pd.to_datetime(
                df["date"], format="%d/%m/%y %H:%M"
            )
            # The source has no UTC offset. The ambiguous autumn DST hour is
            # interpreted as standard time. A nonexistent spring timestamp is
            # kept missing instead of being merged into the following hour.
            df["timestamp"] = (
                timestamp.dt.tz_localize(
                    source_timezone,
                    ambiguous=False,
                    nonexistent="NaT",
                )
                .dt.tz_convert("UTC")
                .dt.tz_localize(None)
            )

            # Handle multiple readings per timestamp: take median
            df_agg = (
                df.groupby("timestamp")["ENERGIA"].median().reset_index()
            )

            # Sort by timestamp
            df_agg = df_agg.sort_values("timestamp").reset_index(drop=True)

            all_dfs.append(df_agg)
            all_upns.append(upn)
            all_plant_ids.append(plant_map.get(upn, len(all_upns) - 1))

            # Get coordinates if available
            if upn in upn_to_coords:
                lat, lon = upn_to_coords[upn]
                all_lats.append(lat)
                all_lons.append(lon)
            else:
                all_lats.append(np.nan)
                all_lons.append(np.nan)

            # Get eta_base
            all_eta_base.append(eta_base_map.get(upn, 0.15))

        except Exception as e:
            print(f"  Error loading {csv_path}: {e}")
            continue

    if not all_dfs:
        raise RuntimeError("No data loaded from Sentinel files")

    # Materialise the complete hourly grid. Missing SCADA hours must remain
    # explicit NaNs; otherwise a 24-row model window can silently span days or
    # months while being interpreted as 24 consecutive hours.
    observed_timestamps = pd.DatetimeIndex(
        np.unique(np.concatenate([df["timestamp"].values for df in all_dfs]))
    ).sort_values()
    unique_timestamps = pd.date_range(
        start=observed_timestamps[0],
        end=observed_timestamps[-1],
        freq="h",
    )
    off_grid = observed_timestamps.difference(unique_timestamps)
    if len(off_grid):
        raise ValueError(
            "Sentinel timestamps do not share one exact hourly grid; "
            f"first off-grid value: {off_grid[0]}"
        )
    
    print(
        f"  Hourly grid: {len(unique_timestamps)} steps "
        f"({len(observed_timestamps)} observed timestamps)"
    )

    # Reindex all plants to common time grid
    N_plants = len(all_dfs)
    T = len(unique_timestamps)
    energia_array = np.full((N_plants, T), np.nan, dtype=np.float32)

    for i, df_agg in enumerate(all_dfs):
        # Create Series with timestamp index for reindexing
        ser = pd.Series(df_agg["ENERGIA"].values, index=df_agg["timestamp"].values)
        ser = ser.reindex(unique_timestamps)
        energia_array[i, :] = ser.values

    print(f"  Stacked: {N_plants} plants x {T} timesteps (aligned to common time grid)")

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "ENERGIA": (["plant", "time"], energia_array),
        },
        coords={
            "plant": np.arange(N_plants),
            "time": unique_timestamps,
            "plant_id": ("plant", np.array(all_plant_ids)),
            "latitude": ("plant", np.array(all_lats, dtype=np.float32)),
            "longitude": ("plant", np.array(all_lons, dtype=np.float32)),
            "eta_base": ("plant", np.array(all_eta_base, dtype=np.float32)),
        },
    )

    # Add metadata
    ds.attrs["source"] = "Sentinel/SCADA Piemonte"
    ds.attrs["year"] = year
    ds.attrs["resolution"] = "hourly"
    ds.attrs["region"] = "Piemonte, Italy"
    ds.attrs["source_timezone"] = source_timezone
    ds.attrs["time_standard"] = "UTC"
    ds.attrs["n_plants"] = N_plants
    ds.attrs["period_start"] = str(unique_timestamps[0])
    ds.attrs["period_end"] = str(unique_timestamps[-1])

    print(f"  Dataset: {ds.sizes['plant']} plants x {ds.sizes['time']} hours")

    return ds


def merge_with_weather(
    ds: xr.Dataset,
    pvgis_path: str = "data/piedmont_pvgis_2019.nc",
) -> xr.Dataset:
    """
    Merge Sentinel energy data with weather variables.

    Requires:
      - ds: Sentinel hourly dataset (from load_sentinel_hourly)
      - Weather NetCDF with: temperature_2m, wind_speed_10m,
        direct_irradiance_tilted and diffuse_irradiance_tilted

    Returns:
        xr.Dataset with: ENERGIA, temperature_2m, solar_irradiance_poa, wind_speed_10m
    """
    try:
        ds_pvgis = xr.open_dataset(pvgis_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"PVGIS weather file is required for real training: {pvgis_path}"
        ) from exc

    # The source NetCDF contains a solar_irradiance_poa variable, but it is
    # known to be empty. Always reconstruct global plane-of-array irradiance
    # from the valid PVGIS tilted beam and diffuse components.
    tilted_vars = ("direct_irradiance_tilted", "diffuse_irradiance_tilted")
    missing_tilted = [var for var in tilted_vars if var not in ds_pvgis]
    if missing_tilted:
        raise ValueError(
            "Cannot reconstruct solar_irradiance_poa: missing PVGIS variables "
            + ", ".join(missing_tilted)
        )
    poa_pvgis = (
        ds_pvgis["direct_irradiance_tilted"]
        + ds_pvgis["diffuse_irradiance_tilted"]
    )

    # Match each Sentinel plant to nearest PVGIS location using lat/lon
    print(f"  Matching {ds.sizes['plant']} plants to PVGIS grid ({ds_pvgis.sizes['location']} locations)...")
    
    # Get plant coordinates
    plant_lats = ds.coords["latitude"].values
    plant_lons = ds.coords["longitude"].values
    
    # Get PVGIS grid coordinates
    pvgis_lats = ds_pvgis.coords["lat"].values  
    pvgis_lons = ds_pvgis.coords["lon"].values
    
    plant_coords = np.column_stack([plant_lats, plant_lons])
    pvgis_coords = np.column_stack([pvgis_lats, pvgis_lons])
    if not np.isfinite(plant_coords).all():
        raise ValueError(
            "All Sentinel plants require finite latitude/longitude coordinates"
        )
    if not np.isfinite(pvgis_coords).all():
        raise ValueError("PVGIS grid contains invalid coordinates")
    
    # Find the closest location with great-circle distance rather than
    # Euclidean degrees, whose longitude scale changes with latitude.
    plant_rad = np.radians(plant_coords)
    pvgis_rad = np.radians(pvgis_coords)
    dlat = plant_rad[:, None, 0] - pvgis_rad[None, :, 0]
    dlon = plant_rad[:, None, 1] - pvgis_rad[None, :, 1]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(plant_rad[:, None, 0])
        * np.cos(pvgis_rad[None, :, 0])
        * np.sin(dlon / 2.0) ** 2
    )
    distances = 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    closest_locations = np.argmin(distances, axis=1)
    
    # Align PVGIS time to Sentinel time (reindex PVGIS to Sentinel time grid)
    t_sentinel = pd.DatetimeIndex(ds.coords["time"].values)
    t_pvgis = pd.DatetimeIndex(ds_pvgis.coords["time"].values)
    if t_sentinel.tz is not None:
        t_sentinel = t_sentinel.tz_convert("UTC").tz_localize(None)
    if t_pvgis.tz is not None:
        t_pvgis = t_pvgis.tz_convert("UTC").tz_localize(None)
    if t_pvgis.has_duplicates or not t_pvgis.is_monotonic_increasing:
        raise ValueError("PVGIS time coordinate must be unique and increasing")
    
    print(f"  Time alignment: Sentinel {len(t_sentinel)} hours, PVGIS {len(t_pvgis)} hours")
    
    # Extract PVGIS variables for matched locations, reindex to Sentinel times
    # For each plant, get the corresponding PVGIS location
    N_plants = ds.sizes['plant']
    N_times = ds.sizes['time']
    
    # Create arrays for weather variables
    temperature_array = np.full((N_plants, N_times), np.nan, dtype=np.float32)
    irradiance_array = np.full((N_plants, N_times), np.nan, dtype=np.float32)
    
    for i in range(N_plants):
        loc_idx = closest_locations[i]
        
        # Extract PVGIS data for this location
        temp_pvgis = ds_pvgis["temperature_2m"].isel(location=loc_idx).values
        irr_pvgis = poa_pvgis.isel(location=loc_idx).values
        
        # Reindex PVGIS data to Sentinel time grid
        pvgis_df = pd.DataFrame({
            'temperature_2m': temp_pvgis,
            'solar_irradiance_poa': irr_pvgis,
        }, index=t_pvgis)
        
        # Reindex to Sentinel times and interpolate if needed
        sentinel_df = pvgis_df.reindex(
            t_sentinel,
            method="nearest",
            tolerance=pd.Timedelta("31min"),
        )
        if sentinel_df.isna().any().any():
            raise ValueError(
                "PVGIS/Sentinel time alignment exceeded the 31-minute tolerance"
            )
        
        temperature_array[i, :] = sentinel_df['temperature_2m'].values
        irradiance_array[i, :] = sentinel_df['solar_irradiance_poa'].values
    
    # Add weather variables to Sentinel dataset
    ds["temperature_2m"] = xr.DataArray(
        temperature_array,
        dims=["plant", "time"],
    )
    ds["solar_irradiance_poa"] = xr.DataArray(
        irradiance_array,
        dims=["plant", "time"],
    )
    
    # Also add wind speed
    wind_array = np.full((N_plants, N_times), np.nan, dtype=np.float32)
    for i in range(N_plants):
        loc_idx = closest_locations[i]
        wind_pvgis = ds_pvgis["wind_speed_10m"].isel(location=loc_idx).values
        wind_df = pd.DataFrame({'wind_speed_10m': wind_pvgis}, index=t_pvgis)
        wind_reindexed = wind_df.reindex(
            t_sentinel,
            method="nearest",
            tolerance=pd.Timedelta("31min"),
        )
        if wind_reindexed.isna().any().any():
            raise ValueError(
                "PVGIS/Sentinel wind alignment exceeded the 31-minute tolerance"
            )
        wind_array[i, :] = wind_reindexed['wind_speed_10m'].values
    
    ds["wind_speed_10m"] = xr.DataArray(
        wind_array,
        dims=["plant", "time"],
    )

    # Plane-of-array beam/diffuse components from PVGIS; their sum is POA.
    for var in tilted_vars:
        arr = np.full((N_plants, N_times), np.nan, dtype=np.float32)
        for i in range(N_plants):
            loc_idx = closest_locations[i]
            comp_pvgis = ds_pvgis[var].isel(location=loc_idx).values
            comp_df = pd.DataFrame({var: comp_pvgis}, index=t_pvgis)
            comp_reindexed = comp_df.reindex(
                t_sentinel,
                method="nearest",
                tolerance=pd.Timedelta("31min"),
            )
            if comp_reindexed.isna().any().any():
                raise ValueError(
                    f"PVGIS/Sentinel {var} alignment exceeded the "
                    "31-minute tolerance"
                )
            arr[i, :] = comp_reindexed[var].values
        ds[var] = xr.DataArray(arr, dims=["plant", "time"])
    ds.attrs["pvgis_tilt_angle"] = float(
        ds_pvgis.attrs.get("tilt_angle", 30.0)
    )
    ds.attrs["pvgis_azimuth_angle"] = float(
        ds_pvgis.attrs.get("azimuth_angle", 180.0)
    )
    ds.attrs["weather_time_standard"] = "UTC"
    print("  Merged weather variables: temperature_2m, reconstructed solar_irradiance_poa, "
          "wind_speed_10m, direct_irradiance_tilted, diffuse_irradiance_tilted")

    return ds


if __name__ == "__main__":
    # Example: Load 2019 Sentinel data
    ds = load_sentinel_hourly(
        year=2019,
        plant_mapping_path="data/plant_mapping.csv",
        energy_coords_path="data/energy_with_coordinates.csv",
    )

    # Merge with weather
    ds = merge_with_weather(ds, pvgis_path="data/piedmont_pvgis_2019.nc")

    print("\nFinal dataset:")
    print(ds)

