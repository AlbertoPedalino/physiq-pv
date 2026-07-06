"""
One-shot setup: download Open-Meteo data and validate the training input path.

Usage:
    python scripts/setup_openmeteo_data.py                    # full run
    python scripts/setup_openmeteo_data.py --dry-run          # preview only
    python scripts/setup_openmeteo_data.py --skip-download     # validate only
    python scripts/setup_openmeteo_data.py --start-date 2019-03-01 --end-date 2019-06-30
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], label: str, allow_fail: bool = False) -> bool:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=str(_ROOT))
    if result.returncode != 0 and not allow_fail:
        print(f"\nFAILED: {label} (exit {result.returncode})")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup Open-Meteo data for PhysiQ-PV")
    parser.add_argument("--plants-path", default="data/energy_with_coordinates.csv")
    parser.add_argument("--start-date", default="2019-03-01")
    parser.add_argument("--end-date", default="2019-12-31")
    parser.add_argument("--out", default="data/openmeteo_piedmont_2019.nc")
    parser.add_argument("--source", default="historical_forecast",
                        choices=["historical_forecast", "archive", "forecast"])
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    steps_ok = []

    if args.skip_download:
        print(f"\n[skip] Download skipped (--skip-download)")
        steps_ok.append(("download", None))
    elif Path(args.out).exists() and not args.force:
        print(f"\n[skip] Output already exists: {args.out} (use --force to re-download)")
        steps_ok.append(("download", None))
    else:
        download_cmd = [
            sys.executable, "scripts/download_openmeteo_historical_forecast.py",
            "--plants-path", args.plants_path,
            "--start-date", args.start_date,
            "--end-date", args.end_date,
            "--out", args.out,
            "--source", args.source,
            "--batch-size", str(args.batch_size),
        ]
        if args.dry_run:
            download_cmd.append("--dry-run")
        if args.force:
            download_cmd.append("--force")

        ok = _run(download_cmd, "Download Open-Meteo Historical Forecast")
        steps_ok.append(("download", ok))

        if args.dry_run:
            print("\n[dry-run] Stopping after download preview.")
            _summary(steps_ok)
            return

        if not ok:
            print("\nDownload failed. Fix errors and retry.")
            _summary(steps_ok)
            sys.exit(1)

    sentinel_dir = "/data/SentinelPV/energy_data/piemonte_energy_data/single_ups"
    if Path(args.out).exists() and Path(sentinel_dir).exists():
        ok = _run(
            [sys.executable, "scripts/check_openmeteo_pipeline.py",
             "--openmeteo-path", args.out,
             "--max-plants", "5",
             "--max-time-steps", "200"],
            "Validate Open-Meteo -> PVDataset pipeline",
            allow_fail=True,
        )
        steps_ok.append(("validate_pipeline", ok))
    else:
        print("\n[skip] Pipeline validation (need Open-Meteo file + Sentinel data)")
        steps_ok.append(("validate_pipeline", None))

    _summary(steps_ok)


def _summary(steps: list[tuple[str, bool | None]]) -> None:
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, ok in steps:
        if ok is None:
            status = "SKIPPED"
        elif ok:
            status = "OK"
        else:
            status = "FAILED"
        print(f"  {name:<25s} {status}")
    print()


if __name__ == "__main__":
    main()
