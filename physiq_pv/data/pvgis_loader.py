"""Stub: load PVGIS NetCDF. Swap path → real data; synthetic used when path=None."""
import xarray as xr
from physiq_pv.data.synthetic_generator import generate_synthetic_dataset


def load_pvgis(path: str | None = None, seed: int = 42) -> xr.Dataset:
    """
    Load PVGIS reference data.
    path=None  → synthetic dataset (development mode)
    path=str   → xr.open_dataset(path)  (real NetCDF from PVGIS API)

    Expected real format:
        dims: (plant, time)
        variables: pvgis_ref (kWh), solar_irradiance_poa (W/m²),
                   temperature_2m (°C), wind_speed_10m (m/s), lat, lon
    """
    if path is not None:
        return xr.open_dataset(path)
    return generate_synthetic_dataset(seed=seed)
