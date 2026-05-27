"""
Validate Open-Meteo -> PVDataset pipeline on a small data slice.

Usage:
    python scripts/check_openmeteo_pipeline.py \
        --openmeteo-path data/openmeteo_piedmont_2019.nc \
        --max-plants 5 --max-time-steps 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Open-Meteo pipeline")
    parser.add_argument("--openmeteo-path", required=True)
    parser.add_argument("--sentinel-dir",
                        default="/data/SentinelPV/energy_data/piemonte_energy_data/single_ups")
    parser.add_argument("--plant-mapping", default="data/plant_mapping.csv")
    parser.add_argument("--energy-coords", default="data/energy_with_coordinates.csv")
    parser.add_argument("--max-plants", type=int, default=5)
    parser.add_argument("--max-time-steps", type=int, default=200)
    parser.add_argument("--year", type=int, default=2019)
    args = parser.parse_args()

    if not Path(args.openmeteo_path).exists():
        print(f"ERROR: Open-Meteo file not found: {args.openmeteo_path}")
        sys.exit(1)
    if not Path(args.sentinel_dir).exists():
        print(f"ERROR: Sentinel dir not found: {args.sentinel_dir}")
        print("This check requires real Sentinel data on the server.")
        sys.exit(1)

    from physiq_pv.data.sentinel_hourly_loader import load_sentinel_hourly
    from physiq_pv.data.openmeteo_loader import merge_with_openmeteo
    from physiq_pv.data.quality_score import compute_qs
    from physiq_pv.data.dataset import PVDataset, N_FEATURES, get_feature_names

    print("[1] Loading Sentinel data...")
    pm = args.plant_mapping if Path(args.plant_mapping).exists() else None
    ec = args.energy_coords if Path(args.energy_coords).exists() else None
    ds = load_sentinel_hourly(
        sentinel_dir=args.sentinel_dir, year=args.year,
        plant_mapping_path=pm, energy_coords_path=ec,
    )
    ds = ds.isel(
        plant=slice(0, args.max_plants),
        time=slice(0, args.max_time_steps),
    )
    print(f"  plants={ds.sizes['plant']}, time={ds.sizes['time']}")

    print("\n[2] Merging with Open-Meteo...")
    ds = merge_with_openmeteo(ds, openmeteo_path=args.openmeteo_path)

    from main import _normalize_dataset
    ds = _normalize_dataset(ds)

    print(f"  Variables after merge: {list(ds.data_vars)}")
    has_dni = "direct_normal_irradiance" in ds
    has_dhi = "diffuse_radiation" in ds
    print(f"  DNI present: {has_dni}")
    print(f"  DHI present: {has_dhi}")

    print("\n[3] Computing QS + building PVDataset...")
    _qs, m_comp = compute_qs(ds, debug=True)
    dataset = PVDataset(
        ds, m_comp, seq_len=24,
        weather_source="openmeteo_historical_forecast",
        feature_set="openmeteo_operational",
    )

    print(f"  feature_set: {dataset.feature_set}")
    print(f"  weather_source: {dataset.weather_source}")
    print(f"  dni_dhi_source: {dataset._dni_dhi_source}")
    print(f"  feats shape: {dataset.feats.shape}")
    print(f"  N_FEATURES: {N_FEATURES}")
    assert dataset.feats.shape[-1] == N_FEATURES

    feature_names = get_feature_names("openmeteo_operational")
    print(f"  feature_names: {feature_names}")

    print("\n[4] Per-feature stats:")
    feats = dataset.feats  # (T, N, 16)
    for i, name in enumerate(feature_names):
        col = feats[:, :, i]
        finite = col[np.isfinite(col)]
        n_nan = int(np.sum(~np.isfinite(col)))
        if len(finite) > 0:
            print(f"  [{i:2d}] {name:<35s}  min={finite.min():8.4f}  max={finite.max():8.4f}  mean={finite.mean():8.4f}  nan={n_nan}")
        else:
            print(f"  [{i:2d}] {name:<35s}  ALL NaN ({n_nan})")

    print("\n[5] Sample item:")
    x, y_ghi, y_pv, eta, ghi_cs = dataset[0]
    print(f"  x: {x.shape}")
    print(f"  y_ghi: {y_ghi.shape}")
    print(f"  y_pv: {y_pv.shape}")
    print(f"  eta: {eta.shape}")
    print(f"  ghi_cs: {ghi_cs.shape}")
    assert x.shape[-1] == N_FEATURES

    print("\nPIPELINE CHECK PASSED")


if __name__ == "__main__":
    main()
