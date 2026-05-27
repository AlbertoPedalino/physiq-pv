"""
Compare PVGIS vs Open-Meteo weather variables on overlapping time/location.

Usage:
    python scripts/compare_weather_features.py \
        --pvgis-path data/piedmont_pvgis_2019.nc \
        --openmeteo-path data/openmeteo_piedmont_2019.nc \
        --out reports/weather_feature_comparison.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xarray as xr
except ImportError:
    print("ERROR: xarray required")
    sys.exit(1)


def _stats(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> dict:
    mask = np.isfinite(a) & np.isfinite(b)
    a_f, b_f = a[mask], b[mask]
    if len(a_f) < 10:
        return {"n": len(a_f), "note": "too few finite pairs"}
    diff = a_f - b_f
    corr = float(np.corrcoef(a_f, b_f)[0, 1]) if np.std(a_f) > 0 and np.std(b_f) > 0 else np.nan
    return {
        "n": int(len(a_f)),
        "corr": corr,
        "bias": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        f"{name_a}_min": float(a_f.min()),
        f"{name_a}_max": float(a_f.max()),
        f"{name_a}_mean": float(a_f.mean()),
        f"{name_b}_min": float(b_f.min()),
        f"{name_b}_max": float(b_f.max()),
        f"{name_b}_mean": float(b_f.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PVGIS vs Open-Meteo")
    parser.add_argument("--pvgis-path", required=True)
    parser.add_argument("--openmeteo-path", required=True)
    parser.add_argument("--max-locations", type=int, default=50)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    for p in (args.pvgis_path, args.openmeteo_path):
        if not Path(p).exists():
            print(f"ERROR: file not found: {p}")
            sys.exit(1)

    ds_pv = xr.open_dataset(args.pvgis_path)
    ds_om = xr.open_dataset(args.openmeteo_path)

    lines: list[str] = ["# Weather Feature Comparison: PVGIS vs Open-Meteo\n"]

    lines.append(f"PVGIS: {args.pvgis_path}")
    lines.append(f"  dims: {dict(ds_pv.dims)}")
    lines.append(f"  vars: {list(ds_pv.data_vars)}")
    lines.append(f"Open-Meteo: {args.openmeteo_path}")
    lines.append(f"  dims: {dict(ds_om.dims)}")
    lines.append(f"  vars: {list(ds_om.data_vars)}\n")

    comparisons = [
        ("solar_irradiance_poa", "shortwave_radiation", "Solar (W/m2)"),
        ("temperature_2m", "temperature_2m", "Temperature (C)"),
        ("wind_speed_10m", "wind_speed_10m", "Wind (m/s)"),
    ]

    loc_dim_pv = "location" if "location" in ds_pv.dims else list(ds_pv.dims)[0]
    loc_dim_om = "location" if "location" in ds_om.dims else list(ds_om.dims)[0]
    n_locs = min(
        args.max_locations,
        ds_pv.sizes[loc_dim_pv],
        ds_om.sizes[loc_dim_om],
    )

    t_pv = pd.DatetimeIndex(ds_pv.coords["time"].values)
    t_om = pd.DatetimeIndex(ds_om.coords["time"].values)
    t_common = t_pv.intersection(t_om)
    lines.append(f"Common timesteps: {len(t_common)}")
    lines.append(f"Locations compared: {n_locs}\n")

    if len(t_common) == 0:
        lines.append("ERROR: no overlapping timesteps. Cannot compare.")
        report = "\n".join(lines)
        print(report)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(report, encoding="utf-8")
        return

    for var_pv, var_om, label in comparisons:
        lines.append(f"## {label}")
        if var_pv not in ds_pv:
            lines.append(f"  PVGIS variable `{var_pv}` not found. Skipped.\n")
            continue
        if var_om not in ds_om:
            lines.append(f"  Open-Meteo variable `{var_om}` not found. Skipped.\n")
            continue

        all_a, all_b = [], []
        for loc_i in range(n_locs):
            a_series = ds_pv[var_pv].isel(**{loc_dim_pv: loc_i}).sel(time=t_common).values
            b_series = ds_om[var_om].isel(**{loc_dim_om: loc_i}).sel(time=t_common).values
            all_a.append(a_series)
            all_b.append(b_series)

        a_all = np.concatenate(all_a)
        b_all = np.concatenate(all_b)
        s = _stats(a_all, b_all, "pvgis", "openmeteo")

        for k, v in s.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")

        if isinstance(s.get("corr"), float) and s["corr"] < 0.9:
            lines.append(f"  WARNING: correlation {s['corr']:.3f} < 0.9 — distributions may differ significantly.")
        lines.append("")

    ds_pv.close()
    ds_om.close()

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\nReport saved to {args.out}")


if __name__ == "__main__":
    main()
