"""
Run the PVGIS climatology anomaly pipeline.

Compares a PVGIS target year against a multi-year PVGIS climatology built in
memory from separate annual PVGIS NetCDF files, and flags rare / extreme
meteo-solar conditions relative to that climatology.

This is climatology-based PVGIS analysis only: it does NOT analyse real plants,
does NOT use observed plant energy, and does NOT touch the training pipeline.

Example (server):

    PYTHONPATH=$PWD python scripts/run_pvgis_climatology_anomaly.py \\
      --year 2019 \\
      --pvgis-path /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance/piedmont_pvgis_2019.nc \\
      --pvgis-climatology-dir /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance \\
      --climatology-start-year 2005 \\
      --climatology-end-year 2023 \\
      --out-dir outputs/pvgis_anomaly_2019_2005_2023_w15 \\
      --quantile 0.975 \\
      --climatology-window-days 15 \\
      --min-score-denominator 1.0
"""

from __future__ import annotations

import argparse

from physiq_pv.data.pvgis_climatology_anomaly import (
    DEFAULT_VARIABLES,
    build_climatology,
    load_climatology_files,
    load_target_pvgis,
    score_target_against_climatology,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PVGIS target year vs multi-year PVGIS climatology anomaly scoring."
    )
    p.add_argument("--year", type=int, required=True, help="Target PVGIS year (e.g. 2019).")
    p.add_argument("--pvgis-path", required=True, help="Target-year PVGIS NetCDF file.")
    p.add_argument(
        "--pvgis-climatology-dir",
        required=True,
        help="Directory holding the separate annual PVGIS NetCDF files.",
    )
    p.add_argument("--climatology-start-year", type=int, required=True)
    p.add_argument("--climatology-end-year", type=int, required=True)
    p.add_argument("--out-dir", default="outputs/pvgis_anomaly")
    p.add_argument(
        "--quantile",
        type=float,
        default=0.975,
        help="Upper quantile of the climatology band; q_low = 1 - quantile. "
        "Exploratory threshold, not a definitive scientific value.",
    )
    p.add_argument(
        "--min-climatology-years",
        type=int,
        default=3,
        help="Minimum pooled samples per (location, calendar_day, hour) bin; "
        "below this a point is labelled insufficient_climatology.",
    )
    p.add_argument(
        "--climatology-window-days",
        type=int,
        default=15,
        help="Calendar-day half-window (+/-N days) for pooling climatology samples "
        "around each target day, across all years. 0 = exact-day matching.",
    )
    p.add_argument(
        "--min-score-denominator",
        type=float,
        default=1.0,
        help="Floor for the anomaly-score denominator; avoids huge scores in "
        "near-zero climatology bands (night / sunrise / sunset).",
    )
    p.add_argument(
        "--file-template",
        default="piedmont_pvgis_{year}.nc",
        help="Filename template for annual climatology files.",
    )
    p.add_argument(
        "--include-target-year-in-climatology",
        action="store_true",
        help="Include the target year in the climatology (default: leave-one-out).",
    )
    p.add_argument("--variables", nargs="+", default=DEFAULT_VARIABLES)
    p.add_argument("--top-n", type=int, default=20, help="Rows in the report top table.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[1/5] Loading target PVGIS year {args.year}: {args.pvgis_path}")
    target_ds = load_target_pvgis(args.pvgis_path)

    exclude = None if args.include_target_year_in_climatology else args.year
    print(
        f"[2/5] Loading climatology {args.climatology_start_year}-{args.climatology_end_year} "
        f"from {args.pvgis_climatology_dir}"
    )
    clim_datasets = load_climatology_files(
        args.pvgis_climatology_dir,
        args.climatology_start_year,
        args.climatology_end_year,
        file_template=args.file_template,
        exclude_year=exclude,
    )

    print(
        f"[3/5] Building in-memory climatology "
        f"(quantile={args.quantile}, window=+/-{args.climatology_window_days}d)"
    )
    climatology = build_climatology(
        clim_datasets,
        args.variables,
        args.quantile,
        window_days=args.climatology_window_days,
    )

    print(
        f"[4/5] Scoring target year against climatology "
        f"(min_years={args.min_climatology_years}, min_denom={args.min_score_denominator})"
    )
    result = score_target_against_climatology(
        target_ds,
        climatology,
        args.variables,
        quantile=args.quantile,
        min_years=args.min_climatology_years,
        min_score_denominator=args.min_score_denominator,
        window_days=args.climatology_window_days,
        climatology_years=sorted(clim_datasets),
    )

    print(f"[5/5] Writing outputs to {args.out_dir}")
    paths = write_outputs(result, args.out_dir, top_n=args.top_n)

    print("\nDone.")
    print(f"  variables analyzed : {', '.join(result.meta['variables_analyzed']) or '(none)'}")
    print(f"  total flagged      : {result.meta['total_flagged']}")
    for key in ("scores", "labels", "summary", "report"):
        print(f"  {key:8s} -> {paths[key]}")

    target_ds.close()
    for ds in clim_datasets.values():
        ds.close()


if __name__ == "__main__":
    main()
