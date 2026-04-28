"""Stub: load energy production CSV from Sentinel/SCADA. Uses synthetic when path=None."""
import xarray as xr
import pandas as pd
import numpy as np
from physiq_pv.data.synthetic_generator import generate_synthetic_dataset


def load_sentinel(path: str | None = None, seed: int = 42) -> xr.Dataset:
    """
    Load real energy data (ENERGIA variable).
    path=None  → synthetic dataset
    path=str   → CSV with columns [timestamp, plant_id, ENERGIA_kWh]

    Returns xr.Dataset with ENERGIA (plant, time) merged into base dataset.
    """
    if path is None:
        return generate_synthetic_dataset(seed=seed)

    df = pd.read_csv(path, parse_dates=["timestamp"])
    # Pivot: rows=timestamp, cols=plant_id
    pivot = df.pivot(index="timestamp", columns="plant_id", values="ENERGIA_kWh")
    plants = pivot.columns.to_numpy()
    times = pivot.index
    energia = pivot.values.T  # (N_plants, T)

    ds = xr.Dataset(
        {"ENERGIA": (["plant", "time"], energia)},
        coords={"plant": plants, "time": times},
    )
    return ds
