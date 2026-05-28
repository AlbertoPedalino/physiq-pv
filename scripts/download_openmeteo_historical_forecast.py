"""
Download Open-Meteo Historical Forecast data for PhysiQ-PV plants.

Preferred source for operational training: Historical Forecast API
(retroactive NWP model runs, same pipeline as live Forecast API).

Usage:
    python scripts/download_openmeteo_historical_forecast.py \
        --plants-path data/energy_with_coordinates.csv \
        --start-date 2019-03-01 --end-date 2019-12-31 \
        --out data/openmeteo_piedmont_2019.nc \
        --source historical_forecast

    # Dry run (no download):
    python scripts/download_openmeteo_historical_forecast.py \
        --plants-path data/energy_with_coordinates.csv \
        --start-date 2019-03-01 --end-date 2019-12-31 \
        --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time as time_mod
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

try:
    import xarray as xr
except ImportError:
    xr = None  # type: ignore[assignment]

API_ENDPOINTS = {
    "historical_forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "forecast": "https://api.open-meteo.com/v1/forecast",
    "archive": "https://archive-api.open-meteo.com/v1/archive",
}

REQUIRED_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
]

OPTIONAL_VARS = [
    "cloud_cover",
    "relative_humidity_2m",
    "direct_radiation",
]


def _load_plant_coords(plants_path: str) -> np.ndarray:
    """Load unique (lat, lon) pairs from plant coordinate file.

    Supports CSV with columns Latitude/Longitude or lat/lon.
    """
    df = pd.read_csv(plants_path)
    lat_col = next((c for c in df.columns if c.lower() in ("latitude", "lat")), None)
    lon_col = next((c for c in df.columns if c.lower() in ("longitude", "lon")), None)
    if lat_col is None or lon_col is None:
        raise ValueError(
            f"Cannot find lat/lon columns in {plants_path}. "
            f"Available: {list(df.columns)}"
        )
    coords = df[[lat_col, lon_col]].dropna().drop_duplicates().values
    return coords


def _deduplicate_grid(coords: np.ndarray, precision: int = 2) -> np.ndarray:
    """Round to grid precision and deduplicate."""
    rounded = np.round(coords, precision)
    unique = np.unique(rounded, axis=0)
    return unique


def _build_url(
    endpoint: str,
    lats: list[float],
    lons: list[float],
    start_date: str,
    end_date: str,
    hourly_vars: list[str],
    timezone: str = "UTC",
) -> str:
    lat_str = ",".join(f"{la:.4f}" for la in lats)
    lon_str = ",".join(f"{lo:.4f}" for lo in lons)
    var_str = ",".join(hourly_vars)
    return (
        f"{endpoint}"
        f"?latitude={lat_str}"
        f"&longitude={lon_str}"
        f"&hourly={var_str}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        f"&timezone={timezone}"
    )


def _fetch_batch(
    url: str,
    max_retries: int = 8,
    sleep_seconds: float = 1.0,
    max_backoff: float = 60.0,
) -> dict:
    if requests is None:
        raise ImportError("requests library required: pip install requests")
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code == 429:
                # Open-Meteo rate limit is per-minute; honor Retry-After if sent,
                # else exponential backoff capped at max_backoff (minute window reset).
                retry_after = resp.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = min(sleep_seconds * (2 ** attempt), max_backoff)
                else:
                    wait = min(sleep_seconds * (2 ** attempt), max_backoff)
                print(f"  Rate limited (attempt {attempt+1}/{max_retries}), waiting {wait:.0f}s...")
                time_mod.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait = min(sleep_seconds * (2 ** attempt), max_backoff)
                print(f"  Request failed ({e}), retry in {wait:.0f}s...")
                time_mod.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


def _parse_response(data: dict, lats: list[float], lons: list[float]) -> list[dict]:
    """Parse Open-Meteo JSON response into per-location dicts."""
    if isinstance(data, list):
        locations = data
    elif "hourly" in data:
        locations = [data]
    else:
        raise ValueError(f"Unexpected response format: {list(data.keys())}")

    results = []
    for i, loc_data in enumerate(locations):
        hourly = loc_data.get("hourly", {})
        times = hourly.get("time", [])
        loc_result = {
            "lat": lats[i] if i < len(lats) else loc_data.get("latitude", np.nan),
            "lon": lons[i] if i < len(lons) else loc_data.get("longitude", np.nan),
            "time": pd.to_datetime(times),
        }
        for var in REQUIRED_VARS + OPTIONAL_VARS:
            if var in hourly:
                loc_result[var] = np.array(hourly[var], dtype=np.float32)
        results.append(loc_result)
    return results


def _to_netcdf(
    all_results: list[dict],
    out_path: str,
    source: str,
    start_date: str,
    end_date: str,
    variables: list[str],
) -> None:
    if xr is None:
        raise ImportError("xarray required: pip install xarray netcdf4")

    times = all_results[0]["time"]
    n_locs = len(all_results)
    n_times = len(times)

    lats = np.array([r["lat"] for r in all_results], dtype=np.float64)
    lons = np.array([r["lon"] for r in all_results], dtype=np.float64)

    data_vars = {}
    for var in variables:
        arr = np.full((n_locs, n_times), np.nan, dtype=np.float32)
        for i, r in enumerate(all_results):
            if var in r:
                vals = r[var]
                arr[i, :len(vals)] = vals[:n_times]
        data_vars[var] = (["location", "time"], arr)

    ds = xr.Dataset(
        data_vars,
        coords={
            "lat": ("location", lats),
            "lon": ("location", lons),
            "time": times,
        },
        attrs={
            "source": f"Open-Meteo {source} API",
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",
            "variables": json.dumps(variables),
            "n_locations": n_locs,
            "created_at": datetime.utcnow().isoformat(),
        },
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    print(f"  Saved: {out_path} ({n_locs} locations x {n_times} timesteps)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Open-Meteo Historical Forecast for PhysiQ-PV",
    )
    parser.add_argument("--plants-path", required=True,
                        help="CSV with plant coordinates (Latitude, Longitude)")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out", default="data/openmeteo_piedmont_2019.nc")
    parser.add_argument("--source", default="historical_forecast",
                        choices=list(API_ENDPOINTS.keys()))
    parser.add_argument("--timezone", default="UTC")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Locations per API request")
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--grid-precision", type=int, default=2,
                        help="Decimal places for lat/lon dedup (2 = ~1km)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    endpoint = API_ENDPOINTS[args.source]
    out_path = Path(args.out)

    if out_path.exists() and not args.force and not args.dry_run:
        print(f"Output already exists: {out_path}")
        print("Use --force to overwrite.")
        sys.exit(1)

    print(f"[config] source={args.source}")
    print(f"[config] endpoint={endpoint}")
    print(f"[config] dates={args.start_date} -> {args.end_date}")
    print(f"[config] out={args.out}")

    coords = _load_plant_coords(args.plants_path)
    print(f"[plants] {len(coords)} raw coordinate pairs from {args.plants_path}")

    grid = _deduplicate_grid(coords, precision=args.grid_precision)
    print(f"[plants] {len(grid)} unique grid points (precision={args.grid_precision})")

    hourly_vars = list(REQUIRED_VARS)
    n_batches = int(np.ceil(len(grid) / args.batch_size))
    print(f"[plan] {n_batches} batches x {args.batch_size} locations")
    print(f"[plan] variables: {hourly_vars}")

    if args.dry_run:
        print("\n[dry-run] Would download:")
        for b in range(n_batches):
            start = b * args.batch_size
            end = min(start + args.batch_size, len(grid))
            batch_lats = grid[start:end, 0].tolist()
            batch_lons = grid[start:end, 1].tolist()
            url = _build_url(
                endpoint, batch_lats, batch_lons,
                args.start_date, args.end_date, hourly_vars, args.timezone,
            )
            print(f"  batch {b+1}/{n_batches}: {end-start} locations, URL length={len(url)}")
        print(f"\n[dry-run] Total: {len(grid)} locations, {len(hourly_vars)} vars")
        print("[dry-run] No data downloaded.")
        return

    if requests is None:
        print("ERROR: requests library not installed. pip install requests")
        sys.exit(1)

    all_results: list[dict] = []
    for b in range(n_batches):
        start = b * args.batch_size
        end = min(start + args.batch_size, len(grid))
        batch_lats = grid[start:end, 0].tolist()
        batch_lons = grid[start:end, 1].tolist()

        url = _build_url(
            endpoint, batch_lats, batch_lons,
            args.start_date, args.end_date, hourly_vars, args.timezone,
        )
        print(f"  batch {b+1}/{n_batches} ({end-start} locations)...", end=" ", flush=True)

        data = _fetch_batch(url, max_retries=args.max_retries, sleep_seconds=args.sleep_seconds)
        batch_results = _parse_response(data, batch_lats, batch_lons)
        all_results.extend(batch_results)
        print(f"OK ({len(batch_results)} parsed)")

        if b < n_batches - 1:
            time_mod.sleep(args.sleep_seconds)

    present_vars = set()
    for r in all_results:
        present_vars.update(k for k in r if k not in ("lat", "lon", "time"))

    missing = set(REQUIRED_VARS) - present_vars
    if missing:
        print(f"WARNING: missing required variables: {missing}")
        print("Erbs fallback will be used for DNI/DHI if those are missing.")

    save_vars = [v for v in REQUIRED_VARS if v in present_vars]
    _to_netcdf(all_results, args.out, args.source, args.start_date, args.end_date, save_vars)
    print(f"\nDone. Variables saved: {save_vars}")


if __name__ == "__main__":
    main()
