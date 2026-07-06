"""
Script semplificato per creare dataset reale con integrazione PVGIS.

Struttura:
1. ENERGIA: Energia oraria effettiva piemontese (2019 only - PVGIS aligned)
2. pvgis_ref: Riferimento PVGIS 2019 per QS
3. temperature_2m: Temperatura da PVGIS 2019
4. eta_base: Coordinata - efficienza base per ogni impianto

Output: real_data_dataset.nc compatibile con compute_qs()

Note: PVGIS piemontese disponibile SOLO per 2019, quindi usiamo solo 
      i dati di energia del 2019 per alignment temporale accurato.
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("data")
PVGIS_NC = Path("data/piedmont_pvgis_2019.nc")

def main():
    print("📊 Creando dataset reale con ENERGIA + PVGIS")
    
    # 1. Carica ENERGIA oraria
    print("\n⏰ Caricando energia oraria...")
    energy_file = OUTPUT_DIR / "hourly_energy_with_coordinates.csv"
    energy_df = pd.read_csv(energy_file)
    energy_df['date'] = pd.to_datetime(energy_df['date'])
    
    # FILTRO: Usa solo 2019 (quando PVGIS è disponibile)
    energy_df = energy_df[energy_df['date'].dt.year == 2019].copy()
    print(f"  ✓ {len(energy_df)} record orari (2019 only - PVGIS alignment)")
    print(f"  Data range: {energy_df['date'].min()} to {energy_df['date'].max()}")
    print(f"  Impianti: {energy_df['Codice UP'].nunique()}")
    
    # 2. Creare plant mapping
    plant_mapping = energy_df[['Codice UP', 'Latitude', 'Longitude', 'Codice Censimp Impianto']].drop_duplicates().reset_index(drop=True)
    plant_mapping['plant_id'] = range(len(plant_mapping))
    plant_mapping['eta_base'] = 0.80  # Performance ratio (real/pvgis_ref), not module efficiency
    print(f"\n  ✓ Plant mapping: {len(plant_mapping)} impianti unici")
    
    # 3. Pivot ENERGIA per (time, plant)
    energy_pivot = energy_df.merge(plant_mapping[['Codice UP', 'plant_id']], on='Codice UP').pivot_table(
        index='date',
        columns='plant_id',
        values='ENERGIA',
        aggfunc='mean'
    )
    print(f"  ✓ Energy pivot: {energy_pivot.shape}")
    
    # 4. Carica PVGIS 2019
    print(f"\n📡 Caricando PVGIS 2019...")
    if not PVGIS_NC.exists():
        print(f"  ❌ {PVGIS_NC} non trovato")
        return
    
    pvgis_ds = xr.open_dataset(PVGIS_NC)
    print(f"  ✓ PVGIS caricato: {dict(pvgis_ds.dims)}")
    print(f"    Variabili: {list(pvgis_ds.data_vars.keys())}")
    
    # 5. Estrai pv_power_output (W) e temperature da PVGIS
    # NOTA: pv_power_output è il riferimento fisicamente coerente con ENERGIA
    pv_ref = pvgis_ds['pv_power_output'].values  # (location=1149, time=8760) in W
    temps = pvgis_ds['temperature_2m'].values      # (location=1149, time=8760)
    pvgis_times = pd.to_datetime(pvgis_ds['time'].values)
    print(f"  Forme: pv_ref={pv_ref.shape}, temps={temps.shape}, times={len(pvgis_times)}")
    
    # Map piante a celle PVGIS usando nearest neighbor geografico
    from scipy.spatial.distance import cdist
    
    print(f"\n  🗺️ Mapping piante → PVGIS con nearest neighbor geografico...")
    
    # Coordinate PVGIS
    pvgis_lats = pvgis_ds['lat'].values
    pvgis_lons = pvgis_ds['lon'].values
    pvgis_coords = np.column_stack([pvgis_lats, pvgis_lons])
    
    # Coordinate piante
    plant_coords = plant_mapping[['Latitude', 'Longitude']].values
    
    # Nearest neighbor: per ogni pianta, trova la cella PVGIS più vicina
    distances = cdist(plant_coords, pvgis_coords, metric='euclidean')
    nearest_indices = np.argmin(distances, axis=1)
    
    plant_to_pvgis = plant_mapping[['plant_id', 'Latitude', 'Longitude']].copy()
    plant_to_pvgis['pvgis_idx'] = nearest_indices
    
    # Verifica qualità del mapping
    min_dist = np.min(distances, axis=1)
    max_dist = np.max(distances, axis=1)
    print(f"    ✓ Mapping completato!")
    print(f"      Distance range: [{min_dist.min():.4f}, {max_dist.max():.4f}] degrees")
    print(f"      Mean distance: {min_dist.mean():.4f} degrees (~{min_dist.mean()*111:.1f} km)")
    
    # 6. Crea pivot PVGIS (time, plant)
    # Nota: PVGIS ha 8760 steps (1 anno), ENERGIA ha 7334 steps (3 anni parziali)
    # Useremo PVGIS per il pattern annuale, ripetendolo ciclicamente
    
    print(f"  Allineando: ENERGIA {energy_pivot.shape}, PVGIS 8760 steps")
    
    pvgis_ref_data_list = []
    temp_data_list = []
    
    for energy_idx, energy_time in enumerate(energy_pivot.index):
        # Mappa diretta: PVGIS 2019 ha 8760 steps per giorno dell'anno
        # energy_time è nel 2019, PVGIS_times è nel 2019 → direct index match
        # Trova indice PVGIS più vicino al timestamp di energia
        time_diffs = np.abs((pvgis_times - pd.Timestamp(energy_time)).total_seconds() / 3600)  # in ore
        pvgis_idx_time = int(np.argmin(time_diffs))
        
        pvgis_row = {'date': energy_time}
        temp_row = {'date': energy_time}
        
        for plant_id in energy_pivot.columns:
            pvgis_idx_loc = int(plant_to_pvgis[plant_to_pvgis['plant_id'] == plant_id]['pvgis_idx'].values[0])
            
            # Accesso: [location, time]
            # Use pv_power_output directly (in W, same scale as ENERGIA)
            pv_val = float(pv_ref[pvgis_idx_loc, pvgis_idx_time]) / 1000.0  # Convert W to kW
            temp_val = float(temps[pvgis_idx_loc, pvgis_idx_time])
            
            pvgis_row[plant_id] = pv_val
            temp_row[plant_id] = temp_val
        
        pvgis_ref_data_list.append(pvgis_row)
        temp_data_list.append(temp_row)
    
    pvgis_ref_pivot = pd.DataFrame(pvgis_ref_data_list).set_index('date')
    temp_pivot = pd.DataFrame(temp_data_list).set_index('date')
    print(f"  ✓ PVGIS pivots: pvgis_ref={pvgis_ref_pivot.shape}, temp={temp_pivot.shape}")
    
    # 7. Align su common time index
    common_idx = energy_pivot.index
    energy_pivot = energy_pivot.reindex(common_idx)
    pvgis_ref_pivot = pvgis_ref_pivot.reindex(common_idx)
    temp_pivot = temp_pivot.reindex(common_idx)
    
    # 8. Crea xarray Dataset
    eta_base_array = plant_mapping.set_index('plant_id')['eta_base'].values
    
    ds = xr.Dataset(
        {
            'ENERGIA': (['plant', 'time'], energy_pivot.T.values),
            'pvgis_ref': (['plant', 'time'], pvgis_ref_pivot.T.values),
            'temperature_2m': (['plant', 'time'], temp_pivot.T.values),
        },
        coords={
            'time': energy_pivot.index,
            'plant': energy_pivot.columns,
            'latitude': ('plant', plant_mapping.set_index('plant_id')['Latitude'].values),
            'longitude': ('plant', plant_mapping.set_index('plant_id')['Longitude'].values),
            'eta_base': ('plant', eta_base_array),
        }
    )
    
    # 9. Salva
    output_nc = OUTPUT_DIR / "real_data_dataset.nc"
    ds.to_netcdf(output_nc)
    print(f"\n✅ Dataset salvato: {output_nc}")
    print(f"  Dimensioni: {ds.sizes}")
    print(f"  Variabili: {list(ds.data_vars.keys())}")
    print(f"  Coordinate: {list(ds.coords.keys())}")
    
    # Salva anche plant mapping
    plant_mapping_out = OUTPUT_DIR / "plant_mapping.csv"
    plant_mapping.to_csv(plant_mapping_out, index=False)
    print(f"  Plant mapping: {plant_mapping_out}")
    
    # 10. Istruzioni per uso
    print(f"\n📖 Uso nel main.py:")
    print(f"""
    from physiq_pv.data.quality_score import compute_qs
    import xarray as xr
    
    # Carica dataset reale
    ds = xr.open_dataset('data/real_data_dataset.nc')
    
    # Calcola QS per ogni (plant, time) ✓
    qs = compute_qs(ds)  # shape (plant, time)
    
    # Agente usa QS per ogni finestra nel ciclo online
    # → QS è applicato a ogni singolo dato!
    """)
    
    pvgis_ds.close()

if __name__ == "__main__":
    main()
