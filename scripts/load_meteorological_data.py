"""
Script per associare i dati Open Meteo agli impianti basandosi su vicinanza geografica.
Input: hourly_energy_with_coordinates.csv + file .nc Open Meteo
Output: data/energy_with_meteorology.csv
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("data")
METEO_DIR = "/data/SentinelPV/open_meteo/old_run_history"

def load_energy_data():
    """Carica dati di energia con coordinate"""
    print("📊 Caricando dati di energia...")
    energy_file = OUTPUT_DIR / "hourly_energy_with_coordinates.csv"
    df = pd.read_csv(energy_file)
    print(f"  ✓ Caricate {len(df)} righe")
    return df

def load_meteorological_data(num_files=5):
    """Carica dati Open Meteo dai file .nc disponibili"""
    print(f"\n🌦️  Caricando dati meteorologici...")
    
    nc_files = sorted(glob(f"{METEO_DIR}/*.nc"))[:num_files]
    print(f"  Usando {len(nc_files)} file (.nc)")
    
    all_meteo = []
    
    for nc_file in nc_files:
        try:
            ds = xr.open_dataset(nc_file)
            
            # Estrai coordinate e dati
            locations = ds.coords['location'].values
            lats = ds.coords['lat'].values
            lons = ds.coords['lon'].values
            times = ds.coords['time'].values
            
            # Crea DataFrame per questo file
            meteo_data = []
            for loc_idx in range(len(locations)):
                for time_idx, time in enumerate(times):
                    row = {
                        'time': pd.Timestamp(time),
                        'meteo_lat': float(lats[loc_idx]),
                        'meteo_lon': float(lons[loc_idx]),
                        'temperature_2m': float(ds['temperature_2m'].values[loc_idx, time_idx]),
                        'wind_speed_10m': float(ds['wind_speed_10m'].values[loc_idx, time_idx]),
                        'relative_humidity_2m': float(ds['relative_humidity_2m'].values[loc_idx, time_idx]),
                        'cloud_cover': float(ds['cloud_cover'].values[loc_idx, time_idx]) if 'cloud_cover' in ds else np.nan,
                        'shortwave_radiation': float(ds['shortwave_radiation'].values[loc_idx, time_idx]) if 'shortwave_radiation' in ds else np.nan,
                    }
                    meteo_data.append(row)
            
            all_meteo.append(pd.DataFrame(meteo_data))
            print(f"  ✓ Caricato {Path(nc_file).name}")
            ds.close()
        except Exception as e:
            print(f"  ⚠️ Errore in {Path(nc_file).name}: {e}")
    
    if all_meteo:
        combined_meteo = pd.concat(all_meteo, ignore_index=True)
        print(f"  ✓ Totale record meteorologici: {len(combined_meteo)}")
        return combined_meteo
    return None

def find_nearest_meteo_station(energy_df, meteo_df):
    """Associa ogni impianto alla stazione meteo più vicina"""
    print(f"\n📍 Associando stazioni meteo ai {len(energy_df)} record orari...")
    
    # Coordinate uniche degli impianti
    unique_plants = energy_df[['Latitude', 'Longitude']].drop_duplicates().values
    
    # Coordinate uniche delle stazioni meteo
    unique_meteo = meteo_df[['meteo_lat', 'meteo_lon']].drop_duplicates().values
    
    print(f"  Impianti unici: {len(unique_plants)}")
    print(f"  Stazioni meteo uniche: {len(unique_meteo)}")
    
    # Calcola distanze e trova la stazione più vicina per ogni impianto
    distances = cdist(unique_plants, unique_meteo, metric='euclidean')
    nearest_indices = np.argmin(distances, axis=1)
    
    # Crea mapping: per ogni impianto, associa la stazione meteo più vicina
    meteo_mapping = []
    for i, plant_coords in enumerate(unique_plants):
        meteo_idx = nearest_indices[i]
        meteo_station = unique_meteo[meteo_idx]
        dist = distances[i, meteo_idx]
        
        meteo_mapping.append({
            'Latitude': float(plant_coords[0]),
            'Longitude': float(plant_coords[1]),
            'meteo_lat': float(meteo_station[0]),
            'meteo_lon': float(meteo_station[1]),
            'distance_km': float(dist * 111)  # Approssimazione: 1 grado ≈ 111 km
        })
    
    mapping_df = pd.DataFrame(meteo_mapping)
    print(f"  Distanza media: {mapping_df['distance_km'].mean():.2f} km")
    
    # Merge sulla base delle coordinate geografiche
    energy_df = energy_df.merge(mapping_df, on=['Latitude', 'Longitude'], how='left')
    
    print(f"  ✓ Associazioni completate")
    return energy_df, meteo_df

def merge_energy_with_meteo(energy_df, meteo_df):
    """Merge finale: energia + meteorologia per data/ora e stazione"""
    print(f"\n🔗 Merging energia con meteorologia...")
    
    # Normalizza colonna data
    energy_df['date'] = pd.to_datetime(energy_df['date'], format='%d/%m/%y %H:%M', errors='coerce')
    meteo_df['time'] = pd.to_datetime(meteo_df['time'])
    
    # Arrotonda meteo_lat e meteo_lon per il merge
    meteo_df['meteo_lat_round'] = meteo_df['meteo_lat'].round(2)
    meteo_df['meteo_lon_round'] = meteo_df['meteo_lon'].round(2)
    
    energy_df['meteo_lat_round'] = energy_df['meteo_lat'].round(2)
    energy_df['meteo_lon_round'] = energy_df['meteo_lon'].round(2)
    
    # Merge su data/ora e coordinate stazione meteo
    merged = energy_df.merge(
        meteo_df,
        left_on=['date', 'meteo_lat_round', 'meteo_lon_round'],
        right_on=['time', 'meteo_lat_round', 'meteo_lon_round'],
        how='left'
    )
    
    # Pulisci colonne duplicate
    merged = merged.drop(['meteo_lat_round', 'meteo_lon_round'], axis=1)
    if 'time' in merged.columns and 'date' in merged.columns:
        merged = merged.drop(['time'], axis=1)
    
    records_with_meteo = merged['temperature_2m'].notna().sum()
    records_without_meteo = merged['temperature_2m'].isna().sum()
    
    print(f"  ✓ Record con dati meteorologici: {records_with_meteo}")
    print(f"  ⚠️ Record senza dati meteorologici: {records_without_meteo}")
    
    return merged

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Carica energia
    energy = load_energy_data()
    
    # Carica meteo
    meteo = load_meteorological_data(num_files=5)
    if meteo is None:
        return
    
    # Associa stazioni meteo agli impianti
    energy, meteo = find_nearest_meteo_station(energy, meteo)
    
    # Merge completo
    merged = merge_energy_with_meteo(energy, meteo)
    
    # Salva
    print(f"\n💾 Salvando...")
    output_file = OUTPUT_DIR / "energy_with_meteorology.csv"
    merged.to_csv(output_file, index=False)
    print(f"  ✓ {output_file}")
    
    print(f"\n✅ Done!")
    print(f"Colonne disponibili:")
    cols = merged.columns.tolist()
    for i, col in enumerate(cols, 1):
        print(f"  {i}. {col}")
    
    print(f"\nShape: {merged.shape}")
    print(f"\nPrime righe:")
    print(merged.head(3))

if __name__ == "__main__":
    main()
