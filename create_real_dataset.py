"""
Script per creare dataset reale con:
- Energia oraria effettiva (hourly_energy_with_coordinates.csv)
- Meteorologia (Open Meteo)
- PVGIS piemontese (grid_5km.xlsx)
Output: Dataset xarray compatibile con ST-GNN
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("data")
METEO_DIR = "/data/SentinelPV/open_meteo/old_run_history"
PVGIS_FILE = "/data/SentinelPV/pvgis_data/grid_5km.xlsx"
PVGIS_NC = Path("data/piedmont_pvgis_2019.nc")

def load_hourly_energy():
    """Carica energia oraria con coordinate"""
    print("⏰ Caricando energia oraria...")
    energy_file = OUTPUT_DIR / "hourly_energy_with_coordinates.csv"
    df = pd.read_csv(energy_file)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  ✓ {len(df)} record orari")
    print(f"  Data range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Impianti: {df['Codice UP'].nunique()}")
    return df

def load_meteorology_data(num_files=10):
    """Carica dati meteorologici Open Meteo"""
    print(f"\n🌦️  Caricando meteorologia Open Meteo...")
    
    nc_files = sorted(glob(f"{METEO_DIR}/*.nc"))[:num_files]
    print(f"  Usando {len(nc_files)} file")
    
    all_meteo = []
    
    for nc_file in nc_files:
        try:
            ds = xr.open_dataset(nc_file)
            
            locations = ds.coords['location'].values
            lats = ds.coords['lat'].values
            lons = ds.coords['lon'].values
            times = ds.coords['time'].values
            
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
                        'shortwave_radiation': float(ds['shortwave_radiation'].values[loc_idx, time_idx]) if 'shortwave_radiation' in ds else np.nan,
                    }
                    meteo_data.append(row)
            
            all_meteo.append(pd.DataFrame(meteo_data))
            print(f"  ✓ {Path(nc_file).name}")
            ds.close()
        except Exception as e:
            print(f"  ⚠️ {Path(nc_file).name}: {e}")
    
    if all_meteo:
        combined = pd.concat(all_meteo, ignore_index=True)
        print(f"  ✓ {len(combined)} record meteorologici")
        return combined
    return None

def load_pvgis_piemontese():
    """Carica PVGIS piemontese 2019 e griglia di coordinate"""
    print(f"\n📊 Caricando PVGIS 2019 + griglia...")
    
    # 1. Carica NetCDF PVGIS 2019
    pvgis_ds = None
    if PVGIS_NC.exists():
        try:
            pvgis_ds = xr.open_dataset(PVGIS_NC)
            print(f"  ✓ PVGIS 2019 NetCDF: dims={dict(pvgis_ds.dims)}")
            print(f"    Variabili: {list(pvgis_ds.data_vars.keys())}")
        except Exception as e:
            print(f"  ⚠️ Errore lettura PVGIS 2019: {e}")
    else:
        print(f"  ⚠️ {PVGIS_NC} non trovato")
    
    # 2. Carica griglia coordinate PVGIS
    try:
        grid_df = pd.read_excel(PVGIS_FILE)
        print(f"  ✓ Griglia PVGIS: {len(grid_df)} celle")
        print(f"    Colonne: {grid_df.columns.tolist()}")
        return pvgis_ds, grid_df
    except Exception as e:
        print(f"  ⚠️ Errore lettura griglia: {e}")
        return pvgis_ds, None

def associate_meteorology_to_plants(energy_df, meteo_df, pvgis_grid_df, pvgis_ds=None):
    """Associa meteorologia e PVGIS agli impianti per vicinanza geografica"""
    print(f"\n📍 Associando meteorologia e PVGIS agli impianti...")
    
    from scipy.spatial.distance import cdist
    
    # Coordinate uniche impianti e meteo
    plants_coords = energy_df[['Latitude', 'Longitude']].drop_duplicates().values
    meteo_coords = meteo_df[['meteo_lat', 'meteo_lon']].drop_duplicates().values
    pvgis_coords = pvgis_grid_df[['lat', 'lon']].values
    
    # Trova stazione meteo + PVGIS più vicina per ogni impianto
    distances_meteo = cdist(plants_coords, meteo_coords, metric='euclidean')
    distances_pvgis = cdist(plants_coords, pvgis_coords, metric='euclidean')
    
    nearest_meteo_indices = np.argmin(distances_meteo, axis=1)
    nearest_pvgis_indices = np.argmin(distances_pvgis, axis=1)
    
    mapping = []
    for i, plant in enumerate(plants_coords):
        meteo_idx = nearest_meteo_indices[i]
        pvgis_idx = nearest_pvgis_indices[i]
        
        meteo = meteo_coords[meteo_idx]
        pvgis = pvgis_coords[pvgis_idx]
        
        mapping.append({
            'Latitude': float(plant[0]),
            'Longitude': float(plant[1]),
            'meteo_lat': float(meteo[0]),
            'meteo_lon': float(meteo[1]),
            'pvgis_id': int(pvgis_grid_df.iloc[pvgis_idx]['ID']),
            'pvgis_lat': float(pvgis[0]),
            'pvgis_lon': float(pvgis[1]),
            'dist_meteo_km': float(distances_meteo[i, meteo_idx] * 111),
            'dist_pvgis_km': float(distances_pvgis[i, pvgis_idx] * 111),
        })
    
    mapping_df = pd.DataFrame(mapping)
    print(f"  Distanza media da meteo: {mapping_df['dist_meteo_km'].mean():.2f} km")
    print(f"  Distanza media da PVGIS: {mapping_df['dist_pvgis_km'].mean():.2f} km")
    
    energy_df = energy_df.merge(mapping_df, on=['Latitude', 'Longitude'], how='left')
    
    # Merge con meteorologia su coordinate meteo e ora
    meteo_df['meteo_lat_round'] = meteo_df['meteo_lat'].round(3)
    meteo_df['meteo_lon_round'] = meteo_df['meteo_lon'].round(3)
    energy_df['meteo_lat_round'] = energy_df['meteo_lat'].round(3)
    energy_df['meteo_lon_round'] = energy_df['meteo_lon'].round(3)
    
    # Round orario (pandas 2.x: 'h' instead of 'H')
    energy_df['time_rounded'] = energy_df['date'].dt.floor('h')
    meteo_df['time_rounded'] = meteo_df['time'].dt.floor('h')
    
    merged = energy_df.merge(
        meteo_df[['time_rounded', 'meteo_lat_round', 'meteo_lon_round', 
                  'temperature_2m', 'wind_speed_10m', 'relative_humidity_2m', 'shortwave_radiation']],
        left_on=['time_rounded', 'meteo_lat_round', 'meteo_lon_round'],
        right_on=['time_rounded', 'meteo_lat_round', 'meteo_lon_round'],
        how='left'
    )
    
    merged = merged.drop(['meteo_lat_round', 'meteo_lon_round', 'time_rounded'], axis=1)
    
    with_meteo = merged['temperature_2m'].notna().sum()
    print(f"  ✓ {with_meteo} / {len(merged)} record con meteorologia")
    
    return merged, pvgis_ds if pvgis_ds is not None else None

def create_xarray_dataset(energy_with_meteo, pvgis_grid_df, pvgis_ds=None):
    """Crea xarray dataset con ENERGIA, pvgis_ref, temperature_2m, eta_base per compute_qs"""
    print(f"\n🔧 Creando xarray dataset...")
    
    # Pulisci e prepara energia
    energy_with_meteo = energy_with_meteo.copy()
    energy_with_meteo['date'] = pd.to_datetime(energy_with_meteo['date'])
    
    # Crea ID univoco per impianto
    plant_mapping = energy_with_meteo[['Codice UP', 'Latitude', 'Longitude', 'Codice Censimp Impianto', 'pvgis_id']].drop_duplicates().reset_index(drop=True)
    plant_mapping['plant_id'] = range(len(plant_mapping))
    plant_mapping['eta_base'] = 0.15  # Default 15% per impianti solari
    
    energy_with_meteo = energy_with_meteo.merge(plant_mapping[['Codice UP', 'plant_id', 'eta_base']], on='Codice UP', how='left')
    
    # Pivot energy
    energy_pivot = energy_with_meteo.pivot_table(
        index='date',
        columns='plant_id',
        values='ENERGIA',
        aggfunc='mean'
    )
    
    # Variabili meteorologiche - create FIRST as fallback
    temp_pivot = energy_with_meteo.pivot_table(
        index='date',
        columns='plant_id',
        values='temperature_2m',
        aggfunc='mean'
    )
    
    radiation_pivot = energy_with_meteo.pivot_table(
        index='date',
        columns='plant_id',
        values='shortwave_radiation',
        aggfunc='mean'
    )
    
    # PVGIS reference: carica da NetCDF se disponibile
    pvgis_ref_pivot = None
    if pvgis_ds is not None:
        try:
            # Usa solar_irradiance_poa (Global Horizontal Irradiance equivalent)
            ghi = pvgis_ds['solar_irradiance_poa'].values  # (time, location)
            pvgis_times = pd.to_datetime(pvgis_ds['time'].values)
            pvgis_locs = pvgis_ds['location'].values
            
            # Map piante a celle PVGIS
            plant_pvgis_map = plant_mapping[['plant_id', 'pvgis_id']].set_index('plant_id')['pvgis_id'].to_dict()
            
            # Pivot PVGIS nel formato (time, plant)
            pvgis_data = []
            for t_idx, t in enumerate(pvgis_times):
                row_dict = {'date': t}
                for plant_id in energy_pivot.columns:
                    pvgis_cell = plant_pvgis_map.get(plant_id, 0)
                    if pvgis_cell < len(pvgis_locs):
                        # Converti solar_irradiance_poa [W/m2] a potenza [kW]\n                        # Assumed: 1kWp system at 20% efficiency = 200W/m2 peak\n                        irr_val = float(ghi[t_idx, pvgis_cell]) * 0.0001  # ~0.1 kW per 1000 W/m2
                        row_dict[plant_id] = irr_val
                    else:
                        row_dict[plant_id] = np.nan
                pvgis_data.append(row_dict)
            
            pvgis_ref_df = pd.DataFrame(pvgis_data).set_index('date')
            pvgis_ref_pivot = pvgis_ref_df
            print(f"  ✓ PVGIS reference caricato: {pvgis_ref_pivot.shape}")
        except Exception as e:
            print(f"  ⚠️ Errore caricamento PVGIS ref: {e}")
            pvgis_ref_pivot = None
    
    # Se PVGIS non disponibile, usa proxy
    if pvgis_ref_pivot is None:
        print(f"  ℹ️ Usando shortwave_radiation come proxy per pvgis_ref")
        pvgis_ref_pivot = radiation_pivot * 0.15  # ~15% efficiency conversion
    
    # Align all pivots to common time index
    common_time = energy_pivot.index
    energy_pivot = energy_pivot.loc[common_time]
    temp_pivot = temp_pivot.loc[common_time]
    pvgis_ref_pivot = pvgis_ref_pivot.loc[common_time]
    
    # Creare eta_base array (constant per plant)
    eta_base_array = plant_mapping.set_index('plant_id')['eta_base'].values
    
    # Crea xarray Dataset con nomi variabili per compute_qs
    ds = xr.Dataset(
        {
            'ENERGIA': (['time', 'plant'], energy_pivot.values),  # Nome richiesto da compute_qs
            'pvgis_ref': (['time', 'plant'], pvgis_ref_pivot.values),  # Nome richiesto da compute_qs
            'temperature_2m': (['time', 'plant'], temp_pivot.values),  # Nome richiesto da compute_qs
        },
        coords={
            'time': energy_pivot.index,
            'plant': energy_pivot.columns,
            'latitude': ('plant', plant_mapping.set_index('plant_id')['Latitude'].values),
            'longitude': ('plant', plant_mapping.set_index('plant_id')['Longitude'].values),
            'eta_base': ('plant', eta_base_array),  # Nome richiesto da compute_qs
        }
    )
    
    print(f"  ✓ Dataset creato: {ds.sizes}")
    print(f"    Timestamp: {ds.sizes['time']}")
    print(f"    Impianti: {ds.sizes['plant']}")
    print(f"    Variabili per QS: {list(ds.data_vars.keys())}")
    print(f"    Coordinate: {list(ds.coords.keys())}")
    
    return ds, plant_mapping

def save_dataset(ds, plant_mapping):
    """Salva dataset"""
    print(f"\n💾 Salvando...")
    
    # NetCDF
    output_nc = OUTPUT_DIR / "real_data_dataset.nc"
    ds.to_netcdf(output_nc)
    print(f"  ✓ {output_nc}")
    
    # Plant mapping CSV
    output_mapping = OUTPUT_DIR / "plant_mapping.csv"
    plant_mapping.to_csv(output_mapping, index=False)
    print(f"  ✓ {output_mapping}")
    
    return output_nc, output_mapping

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 1. Carica energie oraria
    energy = load_hourly_energy()
    
    # 2. Carica meteorologia
    meteo = load_meteorology_data(num_files=10)
    if meteo is None:
        print("⚠️ Meteorologia non disponibile")
        return
    
    # 3. Carica PVGIS
    pvgis_ds, pvgis_grid = load_pvgis_piemontese()
    
    # 4. Associa meteorologia agli impianti
    energy_with_meteo, pvgis_ds_out = associate_meteorology_to_plants(energy, meteo, pvgis_grid, pvgis_ds)
    
    # 5. Crea xarray dataset
    ds, plant_mapping = create_xarray_dataset(energy_with_meteo, pvgis_grid, pvgis_ds_out)
    
    # 6. Salva
    save_dataset(ds, plant_mapping)
    
    print(f"\n✅ Dataset pronto per compute_qs!")
    print(f"\nVariabili richieste da compute_qs:")
    print(f"  ✓ ENERGIA (energia oraria)")
    print(f"  ✓ pvgis_ref (riferimento PVGIS)")
    print(f"  ✓ temperature_2m (meteorologia)")
    print(f"  ✓ eta_base (coord - efficienza base)")
    print(f"\nUsa nel tuo main.py:")
    print(f"  from physiq_pv.data.quality_score import compute_qs")
    print(f"  import xarray as xr")
    print(f"  ds = xr.open_dataset('data/real_data_dataset.nc')")
    print(f"  qs = compute_qs(ds)  # Per ogni (plant, time)")

if __name__ == "__main__":
    main()
