"""
Script per associare le coordinate agli impianti.
Eseguire con: uv run python merge_coordinates.py
"""

import pandas as pd
from pathlib import Path

# Path dei file
PLANTS_FILE = "list_of_plants_in_piedmont_2017_2019_with_coordinates.xlsx"
ENERGY_FILE = "/data/SentinelPV/energy_data/Piemonte.xlsx"
ENERGY_CSV_DIR = "/data/SentinelPV/energy_data/exported_energy_data_clean"
OUTPUT_DIR = Path("data")

def setup_output_dir():
    """Crea la cartella data se non esiste"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"✓ Directory output: {OUTPUT_DIR}")

def load_plants_coordinates():
    """Carica il file con coordinate degli impianti"""
    print(f"\n📍 Caricando coordinate da {PLANTS_FILE}...")
    df = pd.read_excel(PLANTS_FILE)
    print(f"  Colonne: {df.columns.tolist()}")
    print(f"  Shape: {df.shape}")
    print(f"  Primissime righe:")
    print(df.head(2))
    return df

def load_energy_data():
    """Carica i dati di energia"""
    print(f"\n⚡ Caricando dati di energia da {ENERGY_FILE}...")
    try:
        df = pd.read_excel(ENERGY_FILE)
        print(f"  Colonne: {df.columns.tolist()}")
        print(f"  Shape: {df.shape}")
        print(f"  Primissime righe:")
        print(df.head(2))
        return df
    except Exception as e:
        print(f"  ⚠️ Errore: {e}")
        return None

def merge_data():
    """Merge tra coordinate e dati di energia"""
    # Carica i dati
    plants = load_plants_coordinates()
    energy = load_energy_data()
    
    if energy is None:
        print("\n⚠️ Non riesco a caricare i dati di energia")
        return
    
    # Merge: aggiungi Latitude/Longitude ai dati di energia
    print(f"\n🔗 Facendo il merge...")
    coords_cols = ["Codice UP", "Latitude", "Longitude"]
    coords = plants[coords_cols].drop_duplicates(subset=["Codice UP"])
    print(f"  Coordinate uniche: {len(coords)}")
    
    # Merge sulla colonna Codice UP
    merged = energy.merge(coords, on="Codice UP", how="left")
    print(f"  Shape dopo merge: {merged.shape}")
    
    # Controlla quanti impianti hanno coordinate
    has_coords = merged["Latitude"].notna().sum()
    missing_coords = merged["Latitude"].isna().sum()
    print(f"  ✓ Impianti con coordinate: {has_coords}")
    print(f"  ⚠️ Impianti senza coordinate: {missing_coords}")
    
    # Salva i dati
    print(f"\n💾 Salvando dati...")
    output_plants = OUTPUT_DIR / "plants_with_coordinates.xlsx"
    plants.to_excel(output_plants, index=False)
    print(f"  ✓ {output_plants}")
    
    output_energy = OUTPUT_DIR / "energy_data_piemonte.xlsx"
    energy.to_excel(output_energy, index=False)
    print(f"  ✓ {output_energy}")
    
    # Salva il file MERGED
    output_merged = OUTPUT_DIR / "energy_with_coordinates.xlsx"
    merged.to_excel(output_merged, index=False)
    print(f"  ✓ {output_merged} (MERGED)")
    
    output_merged_csv = OUTPUT_DIR / "energy_with_coordinates.csv"
    merged.to_csv(output_merged_csv, index=False)
    print(f"  ✓ {output_merged_csv} (CSV)")
    
    print(f"\n✅ Merge completato!")
    print(f"File pronti in: {OUTPUT_DIR}")

if __name__ == "__main__":
    setup_output_dir()
    merge_data()
