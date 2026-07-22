"""Thin argparse entrypoint for the reusable PVGIS M2AD pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from physiq_pv.data.pvgis_m2ad import DEFAULT_M2AD_SENSORS
from physiq_pv.experiments.pvgis_m2ad_pipeline import (
    DEFAULT_OUT_DIR,
    PVGISM2ADConfig,
    run_pvgis_m2ad,
)
from physiq_pv.reporting.m2ad_outputs import (
    merge_detection_intervals as _intervals,
)


def parse_years(value: str) -> list[int]:
    """Parse ``2005-2018`` or comma-separated years/ranges."""
    years: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(
                    f"invalid descending year range: {token}"
                )
            years.extend(range(start, end + 1))
        else:
            years.append(int(token))
    if not years:
        raise argparse.ArgumentTypeError("at least one training year is required")
    if len(set(years)) != len(years):
        raise argparse.ArgumentTypeError("training years contain duplicates")
    return years


def parse_sensors(value: str) -> list[str]:
    sensors = [item.strip() for item in value.split(",") if item.strip()]
    if not sensors or len(set(sensors)) != len(sensors):
        raise argparse.ArgumentTypeError(
            "sensors must be a non-empty unique comma-separated list"
        )
    return sensors


def parse_components(value: str) -> int | str:
    if value.lower() == "bic":
        return "bic"
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError(
            "gmm-components must be >= 1 or 'bic'"
        )
    return count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Label-unaware M2AD for PVGIS "
            "(default train 2005-2018, test 2019)."
        )
    )
    parser.add_argument("--pvgis-dir", required=True)
    parser.add_argument(
        "--train-years", type=parse_years, default=parse_years("2005-2018")
    )
    parser.add_argument("--test-year", type=int, default=2019)
    parser.add_argument("--file-template", default="piedmont_pvgis_{year}.nc")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--sensors", type=parse_sensors, default=list(DEFAULT_M2AD_SENSORS)
    )
    parser.add_argument(
        "--max-locations",
        type=int,
        default=None,
        help="Debug/smoke limit; default scores every location.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=120,
        help="History length; 120 hourly points = five days (paper case study).",
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=80)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--error", choices=("point", "area"), default="area")
    parser.add_argument("--area-half-window", type=int, default=2)
    parser.add_argument("--ewma-com", type=float, default=10.0)
    parser.add_argument("--gmm-components", type=parse_components, default="bic")
    parser.add_argument("--max-components", type=int, default=3)
    parser.add_argument("--significance", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> PVGISM2ADConfig:
    """Translate the transport-specific argparse namespace into typed config."""
    return PVGISM2ADConfig(
        pvgis_dir=args.pvgis_dir,
        train_years=tuple(args.train_years),
        test_year=args.test_year,
        file_template=args.file_template,
        out_dir=args.out_dir,
        sensors=tuple(args.sensors),
        max_locations=args.max_locations,
        window_size=args.window_size,
        horizon=args.horizon,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        error=args.error,
        area_half_window=args.area_half_window,
        ewma_com=args.ewma_com,
        gmm_components=args.gmm_components,
        max_components=args.max_components,
        significance=args.significance,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        validation_split=args.validation_split,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        verbose=not args.quiet,
    )


def run_from_args(
    args: argparse.Namespace, parser: Optional[argparse.ArgumentParser] = None
) -> dict[str, Path]:
    try:
        config = config_from_args(args)
    except ValueError as exc:
        if parser is not None:
            parser.error(str(exc))
        raise
    return run_pvgis_m2ad(config)


def main() -> None:
    parser = build_arg_parser()
    run_from_args(parser.parse_args(), parser)


if __name__ == "__main__":
    main()
