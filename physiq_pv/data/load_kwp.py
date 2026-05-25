import numpy as np
import pandas as pd


def load_kwp(plant_mapping_path: str, energy_coords_path: str, n_plants: int) -> np.ndarray:
    """
    Returns kwp[plant_id] (kW peak) for 0..n_plants-1, loaded from GSE registry CSVs.
    Plants with no match in registry get NaN → dataset.py falls back to p99 inference.
    """
    pm = pd.read_csv(plant_mapping_path)
    ec = pd.read_csv(energy_coords_path)

    if "plant_id" not in pm.columns:
        raise ValueError("plant_mapping.csv must contain 'plant_id'")

    join_candidates = [
        col
        for col in ("Codice Censimp Impianto", "Codice UP")
        if col in pm.columns and col in ec.columns
    ]
    if not join_candidates:
        raise ValueError(
            "No common join column found between plant_mapping and energy_coords"
        )

    kwp_col = "Potenza di picco (kW)"
    if kwp_col not in ec.columns:
        raise ValueError(f"energy_coords missing required column '{kwp_col}'")

    best_join = join_candidates[0]
    best_non_null = -1
    for join_col in join_candidates:
        merged_test = pm[["plant_id", join_col]].merge(
            ec[[join_col, kwp_col]], on=join_col, how="left"
        )
        non_null = int(merged_test[kwp_col].notna().sum())
        if non_null > best_non_null:
            best_non_null = non_null
            best_join = join_col

    merged = pm[["plant_id", best_join]].merge(
        ec[[best_join, kwp_col]], on=best_join, how="left"
    )

    kwp = np.full(n_plants, np.nan, dtype=np.float64)
    for _, row in merged.iterrows():
        pid = int(row["plant_id"])
        val = row[kwp_col]
        if 0 <= pid < n_plants and not pd.isna(val):
            kwp[pid] = float(val)
    return kwp
