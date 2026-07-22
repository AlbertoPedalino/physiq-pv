"""Prepare label-unaware PVGIS climate data for MTGFlow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from physiq_pv.anomaly_detection.pvgis_climate import prepare_pvgis_climate_data


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pvgis-dir", required=True)
    parser.add_argument("--out-dir", default="outputs/pvgis_climate_anomaly/prepared")
    parser.add_argument("--file-template", default="piedmont_pvgis_{year}.nc")
    parser.add_argument("--train-start", type=int, default=2005)
    parser.add_argument("--train-end", type=int, default=2018)
    parser.add_argument(
        "--validation-year",
        type=int,
        help="Optional schema-only holdout; excluded from training and thresholds.",
    )
    parser.add_argument("--test-year", type=int, default=2019)
    parser.add_argument("--target-variable", default="pv_power_output")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-locations", type=int)
    parser.add_argument("--location-batch-size", type=int, default=8)
    parser.add_argument(
        "--seasonal-normalization",
        action="store_true",
        help="Opt into the PVGIS seasonal adaptation; the reference protocol uses z-score only.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest = prepare_pvgis_climate_data(
        pvgis_dir=args.pvgis_dir,
        out_dir=args.out_dir,
        train_start=args.train_start,
        train_end=args.train_end,
        validation_year=args.validation_year,
        test_year=args.test_year,
        file_template=args.file_template,
        target_variable=args.target_variable,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        max_locations=args.max_locations,
        location_batch_size=args.location_batch_size,
        seasonal_normalization=args.seasonal_normalization,
    )
    print(f"Prepared {len(manifest)} locations in {args.out_dir}")


if __name__ == "__main__":
    main()
