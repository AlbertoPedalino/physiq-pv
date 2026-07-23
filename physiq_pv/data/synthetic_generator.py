import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

N_PLANTS = 20
T_STEPS = 8760  # 1 year, hourly


def _coords(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.uniform(44.0, 45.5, N_PLANTS), rng.uniform(7.7, 9.0, N_PLANTS)


def _clear_sky_profile(t: np.ndarray, peak_kw: float = 10.0) -> np.ndarray:
    day = t / 24
    hour = t % 24
    seasonal = 0.5 + 0.5 * np.cos(2 * np.pi * (day - 172) / 365)
    daily = np.where(
        (hour >= 6) & (hour <= 18),
        np.sin(np.pi * (hour - 6) / 12),
        0.0,
    )
    return peak_kw * seasonal * daily


def _meteo(t: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    day = t / 24
    hour = t % 24
    daytime = (hour >= 6) & (hour <= 18)
    temp = (
        10 + 10 * np.cos(2 * np.pi * (day - 200) / 365)
        + 5 * np.where(daytime, np.sin(np.pi * (hour - 6) / 12), 0.0)
        + rng.normal(0, 1, len(t))
    )
    solar = np.maximum(0.0, _clear_sky_profile(t, peak_kw=1000.0) * (1 + rng.normal(0, 0.1, len(t))))
    wind = np.abs(rng.normal(3.0, 1.5, len(t)))
    return temp, solar, wind


def generate_synthetic_dataset(seed: int = 42, output_path: str | None = None) -> xr.Dataset:
    """
    Generate synthetic PV dataset with 4 injected fault scenarios:
      Plant 0: gradual degradation 1.5%/year
      Plant 1: sudden sensor failure at t=4000
      Plants 2-5: regional cloud event t=[6000,6500)
      Plant 6: cyclic soiling, period=720h

    Returns xr.Dataset with dims (plant, time) and variables:
      ENERGIA, temperature_2m, solar_irradiance_poa,
      direct_irradiance_tilted, diffuse_irradiance_tilted,
      wind_speed_10m, lat, lon, eta_base
    """
    rng = np.random.default_rng(seed)
    lats, lons = _coords(seed)
    t = np.arange(T_STEPS, dtype=float)
    eta_base = rng.uniform(0.75, 0.85, N_PLANTS)

    energia = np.zeros((N_PLANTS, T_STEPS))
    temp_arr = np.zeros((N_PLANTS, T_STEPS))
    solar_arr = np.zeros((N_PLANTS, T_STEPS))
    wind_arr = np.zeros((N_PLANTS, T_STEPS))

    for p in range(N_PLANTS):
        temp_arr[p], solar_arr[p], wind_arr[p] = _meteo(t, rng)
        ref = _clear_sky_profile(t)
        eta = eta_base[p] * rng.normal(1.0, 0.05, T_STEPS)

        if p == 0:
            eta *= 1.0 - 0.015 * (t / T_STEPS)
        elif p == 1:
            eta[4000:] *= 0.3
        elif 2 <= p <= 5:
            eta[6000:6500] *= 0.2
        elif p == 6:
            eta *= 1.0 - 0.15 * np.abs(np.sin(2 * np.pi * t / 720))

        energia[p] = np.maximum(0.0, ref * eta)

    times = pd.date_range("2023-01-01", periods=T_STEPS, freq="h")
    ds = xr.Dataset(
        {
            "ENERGIA": (["plant", "time"], energia),
            "temperature_2m": (["plant", "time"], temp_arr),
            "solar_irradiance_poa": (["plant", "time"], solar_arr),
            "direct_irradiance_tilted": (
                ["plant", "time"],
                0.8 * solar_arr,
            ),
            "diffuse_irradiance_tilted": (
                ["plant", "time"],
                0.2 * solar_arr,
            ),
            "wind_speed_10m": (["plant", "time"], wind_arr),
            "lat": (["plant"], lats),
            "lon": (["plant"], lons),
            "eta_base": (["plant"], eta_base),
        },
        coords={"plant": np.arange(N_PLANTS), "time": times},
        attrs={"pvgis_tilt_angle": 30.0, "pvgis_azimuth_angle": 180.0},
    )
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(output_path)
    return ds


if __name__ == "__main__":
    ds = generate_synthetic_dataset(output_path="data/synthetic.nc")
    print(ds)
