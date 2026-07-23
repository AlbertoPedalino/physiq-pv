import numpy as np
import pandas as pd


def load_kwp(plant_mapping_path: str, energy_coords_path: str, n_plants: int) -> np.ndarray:
    """
    Returns kwp[plant_id] (kW peak) for 0..n_plants-1, loaded from GSE registry CSVs.
    Plants with no match in registry get NaN. The canonical POA v2 training
    pipeline normalizes PV with training-only p99 and does not consume kWp;
    this utility remains available for external reporting.
    """
    pm = pd.read_csv(plant_mapping_path)[["plant_id", "Codice Censimp Impianto"]]
    ec = pd.read_csv(energy_coords_path)[["Codice Censimp Impianto", "Potenza di picco (kW)"]]
    merged = pm.merge(ec, on="Codice Censimp Impianto", how="left")

    kwp = np.full(n_plants, np.nan, dtype=np.float64)
    for _, row in merged.iterrows():
        pid = int(row["plant_id"])
        val = row["Potenza di picco (kW)"]
        if 0 <= pid < n_plants and not pd.isna(val):
            kwp[pid] = float(val)
    return kwp
