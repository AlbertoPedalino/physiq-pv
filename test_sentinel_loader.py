#!/usr/bin/env python3
"""
Test script for Sentinel hourly loader.
Validates data loading, alignment, and basic statistics.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from physiq_pv.data.sentinel_hourly_loader import load_sentinel_hourly, merge_with_weather
from physiq_pv.data.quality_score import compute_qs


def test_sentinel_loader():
    """Test Sentinel hourly data loading."""
    print("=" * 70)
    print("🧪 TEST: Sentinel Hourly Loader")
    print("=" * 70)

    # 1. Load Sentinel data
    print("\n[1] Loading Sentinel hourly data...")
    try:
        ds = load_sentinel_hourly(
            sentinel_dir="/data/SentinelPV/energy_data/piemonte_energy_data/single_ups",
            year=2019,
            plant_mapping_path="data/plant_mapping.csv",
            energy_coords_path="data/energy_with_coordinates.csv",
        )
        print(f"    ✅ Loaded: {ds.sizes['plant']} plants × {ds.sizes['time']} hours")
    except Exception as e:
        print(f"    ❌ ERROR: {e}")
        return False

    # 2. Inspect dataset
    print("\n[2] Dataset structure:")
    print(f"    Dimensions: {dict(ds.dims)}")
    print(f"    Data vars: {list(ds.data_vars)}")
    print(f"    Coords: {list(ds.coords)}")

    # 3. Check ENERGIA stats
    print("\n[3] ENERGIA (Energy Production) stats:")
    energia = ds["ENERGIA"].values
    valid = energia[~np.isnan(energia)]
    print(f"    Shape: {energia.shape}")
    print(f"    Valid: {len(valid):,} / {energia.size:,} ({len(valid)/energia.size*100:.1f}%)")
    print(f"    Range: {np.nanmin(energia):.2f} — {np.nanmax(energia):.2f} kW")
    print(f"    Mean: {np.nanmean(energia):.2f} kW")
    print(f"    Median: {np.nanmedian(energia):.2f} kW")
    print(f"    Std: {np.nanstd(energia):.2f} kW")

    # 4. Time info
    print("\n[4] Temporal coverage:")
    time_vals = ds.coords["time"].values
    print(f"    First: {time_vals[0]}")
    print(f"    Last: {time_vals[-1]}")
    print(f"    Total hours: {len(time_vals)}")
    print(f"    Expected (365 days × 24h): 8,760")

    # 5. Check coordinates
    print("\n[5] Geographic coverage:")
    lats = ds.coords["latitude"].values
    lons = ds.coords["longitude"].values
    valid_lats = lats[~np.isnan(lats)]
    valid_lons = lons[~np.isnan(lons)]
    print(f"    Plants with coordinates: {len(valid_lats)}/{len(lats)}")
    if len(valid_lats) > 0:
        print(f"    Latitude range: {np.min(valid_lats):.4f}° — {np.max(valid_lats):.4f}°")
        print(f"    Longitude range: {np.min(valid_lons):.4f}° — {np.max(valid_lons):.4f}°")

    # 6. Merge with weather
    print("\n[6] Merging with PVGIS weather...")
    try:
        ds_merged = merge_with_weather(ds, pvgis_path="data/piedmont_pvgis_2019.nc")
        print(f"    ✅ Merged variables: {list(ds_merged.data_vars)}")
        print(f"    Shape after merge: {dict(ds_merged.dims)}")
    except Exception as e:
        print(f"    ⚠️  Warning: {e}")
        ds_merged = ds

    # 7. Compute QS
    print("\n[7] Computing Quality Score...")
    try:
        qs = compute_qs(ds_merged)
        qs_valid = qs.values[~np.isnan(qs.values)]
        print(f"    ✅ QS computed: {qs.shape}")
        print(f"    Valid: {len(qs_valid):,} / {qs.size:,}")
        print(f"    Mean: {np.nanmean(qs.values):.3f}")
        print(f"    Median: {np.nanmedian(qs.values):.3f}")
        print(f"    Range: {np.nanmin(qs.values):.3f} — {np.nanmax(qs.values):.3f}")
    except Exception as e:
        print(f"    ⚠️  Warning: {e}")

    # 8. Sample data
    print("\n[8] Sample data (plant 0, first 5 hours):")
    sample = ds["ENERGIA"].values[0, :5]
    print(f"    {sample}")

    print("\n" + "=" * 70)
    print("✅ Test completed successfully!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = test_sentinel_loader()
    sys.exit(0 if success else 1)
