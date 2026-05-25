"""
Build the plant mapping and merge coordinates into the energy registry.

The plant mapping is built from the Sentinel hourly file list so it covers the
full deployed fleet, not only a small hand-picked subset.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

PLANTS_FILE = Path("data") / "list_of_plants_in_piedmont_2017_2019_with_coordinates.xlsx"
ENERGY_FILE = Path("/data/SentinelPV/energy_data/Piemonte.xlsx")
SENTINEL_DIR = Path("/data/SentinelPV/energy_data/piemonte_energy_data/single_ups")
OUTPUT_DIR = Path("data")


def setup_output_dir() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


def _upn_key(value: object) -> str:
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


def load_source_plants() -> pd.DataFrame:
    print(f"\nLoading source workbook: {PLANTS_FILE}")
    plants = pd.read_excel(PLANTS_FILE)
    print(f"  rows={len(plants)} cols={list(plants.columns)}")
    return plants


def load_sentinel_upns() -> pd.DataFrame:
    print(f"\nScanning Sentinel files in: {SENTINEL_DIR}")
    csv_files = sorted(SENTINEL_DIR.glob("2019_UPN_*.csv"))
    upns: list[str] = []
    for csv_file in csv_files:
        parts = csv_file.stem.split("_")
        if len(parts) >= 4:
            upns.append(f"UPN_{parts[2]}_{parts[3]}")
    unique_upns = list(dict.fromkeys(_upn_key(upn) for upn in upns))
    print(f"  files={len(csv_files)} unique_upns={len(unique_upns)}")
    return pd.DataFrame({"Codice UP": unique_upns})


def _first_non_null(series: pd.Series):
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else np.nan


def build_complete_plant_mapping(plants: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["Codice UP", "Codice Censimp Impianto", "Latitude", "Longitude", "Potenza di picco (kW)"]
    missing = [col for col in required_cols if col not in plants.columns]
    if missing:
        raise ValueError(f"Workbook missing required columns: {missing}")

    source = plants[required_cols].copy()
    source["Codice UP"] = source["Codice UP"].map(_upn_key)
    source = (
        source.groupby("Codice UP", as_index=False)
        .agg({
            "Codice Censimp Impianto": _first_non_null,
            "Latitude": _first_non_null,
            "Longitude": _first_non_null,
            "Potenza di picco (kW)": _first_non_null,
        })
        .sort_values("Codice UP")
        .reset_index(drop=True)
    )

    sentinel_upns = load_sentinel_upns()
    mapping = sentinel_upns.merge(source, on="Codice UP", how="left", indicator=True)
    mapping.insert(1, "plant_id", np.arange(len(mapping), dtype=int))
    mapping["eta_base"] = 0.15

    matched = int((mapping["_merge"] == "both").sum())
    missing = int((mapping["_merge"] != "both").sum())
    print(f"  mapping rows={len(mapping)} matched={matched} missing={missing}")
    if missing:
        print("  first missing UPNs:")
        print(mapping.loc[mapping["_merge"] != "both", ["Codice UP"]].head(20).to_string(index=False))

    mapping = mapping.drop(columns=["_merge"])
    return mapping


def load_energy_data() -> pd.DataFrame | None:
    print(f"\nLoading energy registry: {ENERGY_FILE}")
    try:
        energy = pd.read_excel(ENERGY_FILE)
        print(f"  rows={len(energy)} cols={list(energy.columns)}")
        return energy
    except Exception as exc:
        print(f"  failed to read energy registry: {exc}")
        return None


def merge_data() -> None:
    plants = load_source_plants()
    mapping = build_complete_plant_mapping(plants)
    mapping_output = OUTPUT_DIR / "plant_mapping.csv"
    mapping.to_csv(mapping_output, index=False)
    print(f"\nSaved complete plant mapping: {mapping_output}")

    energy = load_energy_data()
    if energy is None:
        print("\nSkipping energy merge because the energy registry could not be loaded.")
        return

    print("\nMerging energy registry with coordinates...")
    coords = mapping[["Codice UP", "Latitude", "Longitude"]].drop_duplicates(subset=["Codice UP"])
    merged = energy.merge(coords, on="Codice UP", how="left")
    print(f"  merged rows={len(merged)}")
    print(f"  rows with coordinates={int(merged['Latitude'].notna().sum())}")
    print(f"  rows without coordinates={int(merged['Latitude'].isna().sum())}")

    output_energy = OUTPUT_DIR / "energy_data_piemonte.xlsx"
    energy.to_excel(output_energy, index=False)
    print(f"  saved {output_energy}")

    output_merged = OUTPUT_DIR / "energy_with_coordinates.xlsx"
    merged.to_excel(output_merged, index=False)
    print(f"  saved {output_merged}")

    output_merged_csv = OUTPUT_DIR / "energy_with_coordinates.csv"
    merged.to_csv(output_merged_csv, index=False)
    print(f"  saved {output_merged_csv}")


if __name__ == "__main__":
    setup_output_dir()
    merge_data()
