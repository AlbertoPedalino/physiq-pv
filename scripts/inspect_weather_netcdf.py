"""
Inspect a weather NetCDF file for PhysiQ-PV compatibility.

Usage:
    python scripts/inspect_weather_netcdf.py --path data/piedmont_pvgis_2019.nc
    python scripts/inspect_weather_netcdf.py --path data/openmeteo.nc --out report.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import xarray as xr
except ImportError:
    print("ERROR: xarray required. pip install xarray netcdf4")
    sys.exit(1)

EXPECTED_VARS = [
    "solar_irradiance_poa",
    "temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "global_tilted_irradiance",
    "cloud_cover",
    "relative_humidity_2m",
]


def inspect(path: str) -> str:
    ds = xr.open_dataset(path)
    lines: list[str] = []
    lines.append(f"# NetCDF Inspection: {Path(path).name}\n")

    lines.append("## Dimensions")
    for dim, size in ds.dims.items():
        lines.append(f"- `{dim}`: {size}")

    lines.append("\n## Coordinates")
    for name, coord in ds.coords.items():
        lines.append(f"- `{name}`: dtype={coord.dtype}, shape={coord.shape}")

    lines.append("\n## Global Attributes")
    for k, v in ds.attrs.items():
        lines.append(f"- `{k}`: {v}")

    lines.append("\n## Data Variables")
    for name, var in ds.data_vars.items():
        vals = var.values.astype(float)
        finite = vals[np.isfinite(vals)]
        n_nan = int(np.sum(~np.isfinite(vals)))
        lines.append(f"\n### `{name}`")
        lines.append(f"- dims: {var.dims}")
        lines.append(f"- shape: {var.shape}")
        lines.append(f"- dtype: {var.dtype}")
        if var.attrs:
            for ak, av in var.attrs.items():
                lines.append(f"- attr `{ak}`: {av}")
        if len(finite) > 0:
            lines.append(f"- min: {finite.min():.4f}")
            lines.append(f"- max: {finite.max():.4f}")
            lines.append(f"- mean: {finite.mean():.4f}")
            lines.append(f"- std: {finite.std():.4f}")
        lines.append(f"- NaN/Inf count: {n_nan} / {vals.size} ({n_nan/max(vals.size,1)*100:.1f}%)")

    if "time" in ds.coords:
        times = ds.coords["time"].values
        lines.append(f"\n## Time Range")
        lines.append(f"- start: {times[0]}")
        lines.append(f"- end: {times[-1]}")
        lines.append(f"- steps: {len(times)}")

    lines.append("\n## Variable Checklist")
    present = set(ds.data_vars)
    for var in EXPECTED_VARS:
        status = "PRESENT" if var in present else "MISSING"
        lines.append(f"- `{var}`: **{status}**")

    extra = present - set(EXPECTED_VARS)
    if extra:
        lines.append(f"\nExtra variables not in checklist: {sorted(extra)}")

    lines.append("\n## Interpretation")
    if "solar_irradiance_poa" in present and "shortwave_radiation" not in present:
        lines.append("- This appears to be a **PVGIS legacy** file.")
        lines.append("- `solar_irradiance_poa` present — check attrs/units to confirm if GHI or true POA.")
    elif "shortwave_radiation" in present and "solar_irradiance_poa" not in present:
        lines.append("- This appears to be an **Open-Meteo** file.")
        lines.append("- `shortwave_radiation` = GHI (W/m2).")
    elif "shortwave_radiation" in present and "solar_irradiance_poa" in present:
        lines.append("- Both `solar_irradiance_poa` and `shortwave_radiation` present — merged or hybrid file.")

    if "direct_normal_irradiance" in present and "diffuse_radiation" in present:
        lines.append("- DNI + DHI available -> PVDataset can use direct values (no Erbs).")
    else:
        lines.append("- DNI/DHI missing -> Erbs fallback will be used.")

    ds.close()
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect weather NetCDF")
    parser.add_argument("--path", required=True)
    parser.add_argument("--out", default=None, help="Save report to file")
    args = parser.parse_args()

    if not Path(args.path).exists():
        print(f"ERROR: file not found: {args.path}")
        sys.exit(1)

    report = inspect(args.path)
    print(report)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\nReport saved to {args.out}")


if __name__ == "__main__":
    main()
