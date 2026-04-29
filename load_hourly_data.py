"""
Script per caricare i dati di energia ORARI e associare le coordinate.
Legge da: /data/SentinelPV/energy_data/exported_energy_data_clean/
Output: data/hourly_energy_with_coordinates.csv
"""

import pandas as pd
import re
from pathlib import Path
from glob import glob

OUTPUT_DIR = Path("data")
HOURLY_DATA_DIR = "/data/SentinelPV/energy_data/exported_energy_data_clean"

def load_coordinates():
    """Carica le coordinate da energy_with_coordinates.csv"""
    print("📍 Caricando coordinate...")
    coords_file = OUTPUT_DIR / "energy_with_coordinates.csv"
    if not coords_file.exists():
        print("  ⚠️ File non trovato! Esegui prima merge_coordinates.py")
        return None
    
    coords = pd.read_csv(coords_file)
    # Estrai solo le colonne necessarie
    coords = coords[["Codice UP", "Latitude", "Longitude", "Codice Censimp Impianto"]].drop_duplicates()
    print(f"  ✓ Caricate {len(coords)} coordinate uniche")
    return coords

def extract_upn_from_filename(filename):
    """Estrae UPN dal nome file (es: 2019_UPN_2021228_01.csv -> UPN_2021228_01)"""
    match = re.search(r'(UPN_\d+_\d+)', filename)
    if match:
        return match.group(1)
    return None

def load_hourly_files():
    """Carica e combina tutti i file CSV orari"""
    print(f"\n⏰ Caricando dati orari da {HOURLY_DATA_DIR}...")
    
    csv_files = glob(f"{HOURLY_DATA_DIR}/*.csv")
    print(f"  Trovati {len(csv_files)} file CSV")
    
    hourly_data = []
    
    for i, csv_file in enumerate(csv_files[:100]):  # Carica primi 100 per test
        if i % 20 == 0:
            print(f"  Processati {i}/{len(csv_files)}...")
        
        try:
            filename = Path(csv_file).name
            upn = extract_upn_from_filename(filename)
            
            df = pd.read_csv(csv_file)
            df["Codice UP"] = upn
            hourly_data.append(df)
        except Exception as e:
            print(f"  ⚠️ Errore in {filename}: {e}")
    
    if not hourly_data:
        print("  ⚠️ Nessun file caricato!")
        return None
    
    combined = pd.concat(hourly_data, ignore_index=True)
    print(f"  ✓ Combinati {len(combined)} record orari")
    return combined

def merge_hourly_with_coordinates(hourly, coords):
    """Merge dati orari con coordinate"""
    print(f"\n🔗 Merging dati orari con coordinate...")
    
    # Merge sulla colonna Codice UP
    merged = hourly.merge(coords, on="Codice UP", how="left")
    
    has_coords = merged["Latitude"].notna().sum()
    missing_coords = merged["Latitude"].isna().sum()
    
    print(f"  ✓ Record con coordinate: {has_coords}")
    print(f"  ⚠️ Record senza coordinate: {missing_coords}")
    print(f"  Shape finale: {merged.shape}")
    
    return merged

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Carica coordinate
    coords = load_coordinates()
    if coords is None:
        return
    
    # Carica dati orari
    hourly = load_hourly_files()
    if hourly is None:
        return
    
    # Merge
    merged = merge_hourly_with_coordinates(hourly, coords)
    
    # Salva
    print(f"\n💾 Salvando...")
    output_file = OUTPUT_DIR / "hourly_energy_with_coordinates.csv"
    merged.to_csv(output_file, index=False)
    print(f"  ✓ {output_file}")
    
    print(f"\n✅ Done!")
    print(f"Colonne disponibili:")
    print(f"  {merged.columns.tolist()}")

if __name__ == "__main__":
    main()
