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
import re
from pathlib import Path
from typing import Optional, List, Tuple


def _upn_key(value: object) -> str:
    """Canonical key for UPN strings, robust to leading zeros and separators."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().upper()
    match = re.search(r"UPN[_\s-]*(\d+)[_\s-]*(\d+)", text)
    if match:
        return f"UPN_{int(match.group(1))}_{int(match.group(2))}"
    match = re.search(r"(\d{4,})[_\s-]+(\d+)", text)
    if match:
        return f"UPN_{int(match.group(1))}_{int(match.group(2))}"
    return text


def load_sentinel_hourly(
    sentinel_dir: str = "/data/SentinelPV/energy_data/piemonte_energy_data/single_ups",
    year: int = 2019,
    plant_mapping_path: Optional[str] = None,
    energy_coords_path: Optional[str] = None,
    upn_list: Optional[List[str]] = None,
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
                key = _upn_key(row["Codice UP"])
                plant_map[key] = row.get("plant_id", None)
                eta_base_map[key] = row.get("eta_base", 0.15)
                if pd.notna(row.get("Latitude")):
                    upn_to_coords[key] = (
                        row["Latitude"],
                        row["Longitude"],
                    )

    if energy_coords_path:
        ec = pd.read_csv(energy_coords_path)
        for _, row in ec.iterrows():
            upn = row.get("Codice UP", None)
            if upn and pd.notna(upn):
                key = _upn_key(upn)
                if key not in upn_to_coords and pd.notna(row.get("Latitude")):
                    upn_to_coords[key] = (row["Latitude"], row["Longitude"])

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
        wanted = {_upn_key(upn) for upn in upn_list}
        upn_codes = [(upn, f) for upn, f in upn_codes if _upn_key(upn) in wanted]

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
            df["timestamp"] = pd.to_datetime(df["date"], format="%d/%m/%y %H:%M")

            # Handle multiple readings per timestamp: take median
            df_agg = (
                df.groupby("timestamp")["ENERGIA"].median().reset_index()
            )

            # Sort by timestamp
            df_agg = df_agg.sort_values("timestamp").reset_index(drop=True)

            all_dfs.append(df_agg)
            all_upns.append(upn)
            key = _upn_key(upn)
            all_plant_ids.append(plant_map.get(key, len(all_upns) - 1))

            # Get coordinates if available
            if key in upn_to_coords:
                lat, lon = upn_to_coords[key]
                all_lats.append(lat)
                all_lons.append(lon)
            else:
                all_lats.append(np.nan)
                all_lons.append(np.nan)

            # Get eta_base
            all_eta_base.append(eta_base_map.get(key, 0.15))

        except Exception as e:
            print(f"  Error loading {csv_path}: {e}")
            continue

    if not all_dfs:
        raise RuntimeError("No data loaded from Sentinel files")

    # Create unified time index from all plants
    all_timestamps = []
    for df in all_dfs:
        all_timestamps.extend(df["timestamp"].values)
    unique_timestamps = np.unique(np.concatenate([df["timestamp"].values for df in all_dfs]))
    unique_timestamps = pd.DatetimeIndex(unique_timestamps).sort_values()
    
    print(f"  Unique timestamps across all plants: {len(unique_timestamps)}")

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
            "upn": ("plant", np.array(all_upns, dtype=object)),
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
      - Weather NetCDF with: temperature_2m, solar_irradiance_poa, wind_speed_10m

    Returns:
        xr.Dataset with: ENERGIA, temperature_2m, solar_irradiance_poa, wind_speed_10m
    """
    try:
        ds_pvgis = xr.open_dataset(pvgis_path)
    except FileNotFoundError:
        print(f"  PVGIS file not found: {pvgis_path}")
        print("  Computing clear-sky irradiance via pvlib (PVGIS-free fallback)...")
        from pvlib.location import Location

        N, T = ds.sizes["plant"], ds.sizes["time"]
        times_pd = pd.DatetimeIndex(ds.coords["time"].values)
        if times_pd.tz is None:
            times_pd = times_pd.tz_localize("UTC")

        plant_lats = ds.coords["latitude"].values.astype(float)
        plant_lons = ds.coords["longitude"].values.astype(float)
        fleet_lat = float(np.nanmean(plant_lats))
        fleet_lon = float(np.nanmean(plant_lons))

        # Seasonal temperature heuristic (mid-latitude: peaks ~Aug, min ~Jan)
        doy = times_pd.day_of_year.values.astype(float)
        temp_seasonal = (10.0 + 12.0 * np.sin(np.pi * (doy - 80) / 180)).astype(np.float32)

        irradiance_array = np.zeros((N, T), dtype=np.float32)
        temperature_array = np.tile(temp_seasonal, (N, 1)).astype(np.float32)
        wind_array = np.full((N, T), 3.0, dtype=np.float32)

        for i in range(N):
            lat = float(plant_lats[i]) if np.isfinite(plant_lats[i]) else fleet_lat
            lon = float(plant_lons[i]) if np.isfinite(plant_lons[i]) else fleet_lon
            loc = Location(lat, lon, tz="UTC")
            cs = loc.get_clearsky(times_pd)  # Ineichen model: ghi, dni, dhi columns
            irradiance_array[i, :] = cs["ghi"].values.astype(np.float32)

        ds["temperature_2m"] = xr.DataArray(temperature_array, dims=["plant", "time"])
        ds["solar_irradiance_poa"] = xr.DataArray(irradiance_array, dims=["plant", "time"])
        ds["wind_speed_10m"] = xr.DataArray(wind_array, dims=["plant", "time"])
        print("  Clear-sky fallback applied (GHI as POA proxy, seasonal temp heuristic)")
        return ds

    # Match each Sentinel plant to nearest PVGIS location using lat/lon
    print(f"  Matching {ds.sizes['plant']} plants to PVGIS grid ({ds_pvgis.sizes['location']} locations)...")
    
    # Get plant coordinates
    plant_lats = ds.coords["latitude"].values.astype(float)
    plant_lons = ds.coords["longitude"].values.astype(float)
    valid_plant_coords = np.isfinite(plant_lats) & np.isfinite(plant_lons)
    if not valid_plant_coords.all():
        fleet_lat = float(np.nanmean(plant_lats))
        fleet_lon = float(np.nanmean(plant_lons))
        n_missing = int((~valid_plant_coords).sum())
        print(
            f"  Warning: {n_missing} plants missing coordinates; "
            "using fleet-mean coordinates for PVGIS nearest-location matching"
        )
        plant_lats = np.where(valid_plant_coords, plant_lats, fleet_lat)
        plant_lons = np.where(valid_plant_coords, plant_lons, fleet_lon)
    
    # Get PVGIS grid coordinates
    pvgis_lats = ds_pvgis.coords["lat"].values  
    pvgis_lons = ds_pvgis.coords["lon"].values
    
    # Find nearest PVGIS location for each plant
    from scipy.spatial.distance import cdist
    
    plant_coords = np.column_stack([plant_lats, plant_lons])
    pvgis_coords = np.column_stack([pvgis_lats, pvgis_lons])
    
    # Find closest location index for each plant
    distances = cdist(plant_coords, pvgis_coords, metric='euclidean')
    closest_locations = np.argmin(distances, axis=1)
    
    # Align PVGIS time to Sentinel time (reindex PVGIS to Sentinel time grid)
    t_sentinel = ds.coords["time"].values
    t_pvgis = ds_pvgis.coords["time"].values
    
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
        irr_pvgis = ds_pvgis["solar_irradiance_poa"].isel(location=loc_idx).values
        
        # Reindex PVGIS data to Sentinel time grid
        pvgis_df = pd.DataFrame({
            'temperature_2m': temp_pvgis,
            'solar_irradiance_poa': irr_pvgis,
        }, index=t_pvgis)
        
        # Reindex to Sentinel times and interpolate if needed
        sentinel_df = pvgis_df.reindex(t_sentinel, method='nearest')
        
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
        wind_reindexed = wind_df.reindex(t_sentinel, method='nearest')
        wind_array[i, :] = wind_reindexed['wind_speed_10m'].values
    
    ds["wind_speed_10m"] = xr.DataArray(
        wind_array,
        dims=["plant", "time"],
    )
    print("  Merged weather variables: temperature_2m, solar_irradiance_poa, wind_speed_10m")

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

