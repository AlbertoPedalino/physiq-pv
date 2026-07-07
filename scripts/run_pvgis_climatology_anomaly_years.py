"""
Parametric multi-year PVGIS climatology anomaly scoring + aggregation.

Wraps scripts/run_pvgis_climatology_anomaly.run_single_year, so the anomaly
logic stays in one place. For each target year Y, the climatology is always
restricted to years strictly before Y: [start, min(requested_end, Y - 1)].

Example (server):

    PYTHONPATH=$PWD python scripts/run_pvgis_climatology_anomaly_years.py \\
      --years 2016,2017,2018 \\
      --pvgis-dir /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance \\
      --climatology-start-year 2005 --climatology-end-year 2018 \\
      --quantile 0.975 --climatology-window-days 15 --min-climatology-years 3 \\
      --variables solar_irradiance_poa pv_power_output temperature_2m wind_speed_10m \\
      --out-root outputs \\
      --overwrite

Produces:
    outputs/pvgis_anomaly_2016_2005_2015_w15_q0975/pvgis_climatology_scores.csv
    outputs/pvgis_anomaly_2017_2005_2016_w15_q0975/pvgis_climatology_scores.csv
    outputs/pvgis_anomaly_2018_2005_2017_w15_q0975/pvgis_climatology_scores.csv
    outputs/pvgis_anomaly_train_2016_2018_2005_past_w15_q0975/pvgis_climatology_scores.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Repo root on sys.path so the script runs standalone.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physiq_pv.data.pvgis_anomaly_scores import DEFAULT_VARIABLES  # noqa: E402
from scripts.run_pvgis_climatology_anomaly import (  # noqa: E402
    effective_climatology_end_year,
    run_single_year,
)

SCORES_FILENAME = "pvgis_climatology_scores.csv"


def parse_years(values: List[str]) -> List[int]:
    """Parse years from comma-separated and/or space-separated CLI tokens."""
    years: List[int] = []
    for token in values:
        for part in str(token).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                years.append(int(part))
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"--years: '{part}' is not an integer year."
                )
    if not years:
        raise argparse.ArgumentTypeError("--years requires at least one year.")
    return sorted(set(years))


def quantile_tag(quantile: float) -> str:
    """Filesystem tag for the quantile: 0.975 -> q0975."""
    return f"q{int(round(quantile * 1000)):04d}"


def year_dir_name(
    year: int,
    climatology_start_year: int,
    climatology_end_year: int,
    climatology_window_days: int,
    quantile: float,
) -> str:
    """Per-year output folder name."""
    return (
        f"pvgis_anomaly_{year}_{climatology_start_year}_{climatology_end_year}"
        f"_w{climatology_window_days}_{quantile_tag(quantile)}"
    )


def year_out_dir(
    out_root: str,
    year: int,
    climatology_start_year: int,
    climatology_end_year: int,
    climatology_window_days: int,
    quantile: float,
) -> Path:
    """Per-year output directory. Pass the already effective past-only end year."""
    return Path(out_root) / year_dir_name(
        year,
        climatology_start_year,
        climatology_end_year,
        climatology_window_days,
        quantile,
    )


def default_aggregate_dir(out_root: str, years: List[int], climatology_start_year: int,
                          climatology_end_year: int, climatology_window_days: int,
                          quantile: float) -> Path:
    """Default aggregate folder for rolling past-only yearly scores."""
    _ = climatology_end_year
    lo, hi = min(years), max(years)
    name = (
        f"pvgis_anomaly_train_{lo}_{hi}_{climatology_start_year}_past"
        f"_w{climatology_window_days}_{quantile_tag(quantile)}"
    )
    return Path(out_root) / name


def year_pvgis_path(pvgis_dir: str, file_template: str, year: int) -> Path:
    return Path(pvgis_dir) / file_template.format(year=year)


def aggregate_annual_scores(
    annual_scores: Dict[int, Path],
    out_csv: Path,
) -> pd.DataFrame:
    """Concatenate per-year scores CSVs and tag each row with its target year."""
    frames = []
    for year in sorted(annual_scores):
        path = Path(annual_scores[year])
        if not path.exists():
            raise FileNotFoundError(
                f"annual scores CSV for {year} not found for aggregation: {path}"
            )
        df = pd.read_csv(path)
        df["source_anomaly_year"] = int(year)
        df["source_anomaly_year_file"] = str(path)
        frames.append(df)
    if not frames:
        raise ValueError("no annual scores to aggregate.")

    combined = pd.concat(frames, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    return combined


def summarize_aggregate(combined: pd.DataFrame) -> Dict[str, object]:
    """Print and return a small summary of the aggregated scores."""
    years = sorted(int(y) for y in pd.unique(combined["source_anomaly_year"]))
    ts_min = ts_max = None
    if "timestamp" in combined.columns and len(combined):
        ts = pd.to_datetime(combined["timestamp"], errors="coerce")
        ts_min, ts_max = ts.min(), ts.max()

    info = {
        "rows": int(len(combined)),
        "columns": list(combined.columns),
        "timestamp_min": ts_min,
        "timestamp_max": ts_max,
        "years": years,
    }
    print(f"  aggregated rows    : {info['rows']:,}")
    print(f"  columns ({len(info['columns'])})      : {info['columns']}")
    print(f"  timestamp min      : {ts_min}")
    print(f"  timestamp max      : {ts_max}")
    print(f"  years present      : {years}")
    return info


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--years",
        nargs="+",
        required=True,
        help="Years to score. Accepts comma-separated or space-separated values.",
    )
    p.add_argument(
        "--pvgis-dir",
        "--pvgis_dir",
        required=True,
        help="Directory with the annual PVGIS NetCDF files.",
    )
    p.add_argument(
        "--file-template",
        "--file_template",
        default="piedmont_pvgis_{year}.nc",
        help="Annual filename template.",
    )
    p.add_argument(
        "--climatology-start-year",
        "--climatology_start_year",
        type=int,
        required=True,
    )
    p.add_argument(
        "--climatology-end-year",
        "--climatology_end_year",
        type=int,
        required=True,
        help="Maximum candidate climatology year. Each target year still uses "
        "only years strictly before itself.",
    )
    p.add_argument(
        "--quantile",
        type=float,
        default=0.975,
        help="Upper climatology quantile.",
    )
    p.add_argument(
        "--climatology-window-days",
        "--climatology_window_days",
        type=int,
        default=15,
        help="Calendar-day half-window for pooling.",
    )
    p.add_argument(
        "--min-climatology-years",
        "--min_climatology_years",
        type=int,
        default=3,
        help="Minimum samples per bin before insufficient_climatology.",
    )
    p.add_argument(
        "--min-score-denominator",
        "--min_score_denominator",
        type=float,
        default=1.0,
        help="Floor for the anomaly-score denominator.",
    )
    p.add_argument(
        "--variables",
        nargs="+",
        default=list(DEFAULT_VARIABLES),
        help="Variables to analyse.",
    )
    p.add_argument(
        "--out-root",
        "--out_root",
        default="outputs",
        help="Root dir for the per-year output folders.",
    )
    p.add_argument(
        "--aggregate-out-dir",
        "--aggregate_out_dir",
        default=None,
        help="Folder for the aggregated CSV. Default uses a *_past_* name.",
    )
    p.add_argument(
        "--top-n",
        "--top_n",
        type=int,
        default=20,
        help="Rows in each per-year report top table.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate a year even if its scores CSV exists. The existing "
        "folder is backed up before regeneration.",
    )
    p.add_argument(
        "--no-aggregate",
        "--no_aggregate",
        action="store_true",
        help="Generate per-year outputs only; skip the aggregated CSV.",
    )

    args = p.parse_args(argv)
    try:
        args.years = parse_years(args.years)
    except argparse.ArgumentTypeError as exc:
        p.error(str(exc))
    return args


def _fail(msg: str) -> None:
    raise SystemExit(f"error: {msg}")


def generate_years(args: argparse.Namespace) -> Dict[int, Path]:
    """Score each requested year and return {year: scores_csv}."""
    annual_scores: Dict[int, Path] = {}
    for year in args.years:
        try:
            climatology_end_year = effective_climatology_end_year(
                year,
                args.climatology_start_year,
                args.climatology_end_year,
            )
        except ValueError as exc:
            _fail(str(exc))

        pvgis_path = year_pvgis_path(args.pvgis_dir, args.file_template, year)
        if not pvgis_path.exists():
            _fail(f"PVGIS file for {year} not found: {pvgis_path}")

        out_dir = year_out_dir(
            args.out_root,
            year,
            args.climatology_start_year,
            climatology_end_year,
            args.climatology_window_days,
            args.quantile,
        )
        scores_csv = out_dir / SCORES_FILENAME

        if scores_csv.exists() and not args.overwrite:
            print(f"[year {year}] exists, skipping (use --overwrite): {scores_csv}")
            annual_scores[year] = scores_csv
            continue
        if out_dir.exists() and args.overwrite:
            backup = out_dir.with_name(f"{out_dir.name}.bak-{int(time.time())}")
            print(f"[year {year}] --overwrite: backing up {out_dir} -> {backup}")
            out_dir.rename(backup)

        print(f"[year {year}] scoring past-only climatology -> {out_dir}")
        run_single_year(
            year=year,
            pvgis_path=str(pvgis_path),
            climatology_dir=args.pvgis_dir,
            climatology_start_year=args.climatology_start_year,
            climatology_end_year=climatology_end_year,
            out_dir=str(out_dir),
            quantile=args.quantile,
            min_climatology_years=args.min_climatology_years,
            climatology_window_days=args.climatology_window_days,
            min_score_denominator=args.min_score_denominator,
            file_template=args.file_template,
            variables=args.variables,
            top_n=args.top_n,
        )
        if not scores_csv.exists():
            _fail(f"annual scores CSV was not produced for {year}: {scores_csv}")
        annual_scores[year] = scores_csv
    return annual_scores


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    print(
        f"[plan] years={args.years} pvgis_dir={args.pvgis_dir} "
        f"past_only_end=min({args.climatology_end_year}, target_year - 1)"
    )
    annual_scores = generate_years(args)

    if args.no_aggregate:
        print("\n[no-aggregate] per-year scores only:")
        for year, path in sorted(annual_scores.items()):
            print(f"  {year} -> {path}")
        print("\nDone.")
        return

    agg_dir = (
        Path(args.aggregate_out_dir)
        if args.aggregate_out_dir
        else default_aggregate_dir(
            args.out_root,
            args.years,
            args.climatology_start_year,
            args.climatology_end_year,
            args.climatology_window_days,
            args.quantile,
        )
    )
    agg_csv = agg_dir / SCORES_FILENAME
    print(f"\n[aggregate] -> {agg_csv}")
    combined = aggregate_annual_scores(annual_scores, agg_csv)
    info = summarize_aggregate(combined)

    missing = sorted(set(args.years) - set(info["years"]))
    if missing:
        _fail(
            f"aggregated CSV is missing requested years {missing}; "
            f"present={info['years']}."
        )

    print(f"\nDone. Aggregated scores -> {agg_csv}")


if __name__ == "__main__":
    main()
