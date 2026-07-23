"""Command-line entrypoint for the reusable PVGIS CATCH pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from physiq_pv.data.pvgis_catch import DEFAULT_CATCH_SENSORS
from physiq_pv.experiments.pvgis_catch_pipeline import (
    DEFAULT_OUT_DIR,
    PVGISCATCHConfig,
    run_pvgis_catch,
)


def parse_years(value: str) -> list[int]:
    years: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"invalid descending year range: {token}")
            years.extend(range(start, end + 1))
        else:
            years.append(int(token))
    if not years or len(set(years)) != len(years):
        raise argparse.ArgumentTypeError("training years must be non-empty and unique")
    return years


def parse_sensors(value: str) -> list[str]:
    sensors = [item.strip() for item in value.split(",") if item.strip()]
    if not sensors or len(set(sensors)) != len(sensors):
        raise argparse.ArgumentTypeError("sensors must be non-empty and unique")
    return sensors


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label-unaware CATCH frequency-patching detector for PVGIS."
    )
    parser.add_argument("--pvgis-dir", required=True)
    parser.add_argument("--train-years", type=parse_years, default=parse_years("2005-2018"))
    parser.add_argument("--test-year", type=int, default=2019)
    parser.add_argument("--file-template", default="piedmont_pvgis_{year}.nc")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--sensors", type=parse_sensors, default=list(DEFAULT_CATCH_SENSORS))
    parser.add_argument("--max-locations", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--inference-patch-size", type=int, default=32)
    parser.add_argument("--inference-patch-stride", type=int, default=1)
    parser.add_argument("--cf-dim", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--head-layers", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--mask-source", choices=("projected", "raw"), default="projected")
    parser.add_argument("--frequency-loss-weight", type=float, default=0.005)
    parser.add_argument("--clustering-weight", type=float, default=0.005)
    parser.add_argument("--regularization-weight", type=float, default=0.0025)
    parser.add_argument("--score-frequency-weight", type=float, default=0.05)
    parser.add_argument("--contamination", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mask-lr", type=float, default=1e-5)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--training-window-stride", type=int, default=1)
    parser.add_argument(
        "--scoring-window-stride",
        type=int,
        default=None,
        help="defaults to seq-len, matching the repository's non-overlap scoring",
    )
    parser.add_argument(
        "--model-steps-per-mask",
        type=int,
        default=None,
        help="defaults to the update cadence derived by the official repository",
    )
    parser.add_argument("--gradient-clip", type=float, default=None)
    parser.add_argument(
        "--lr-adjustment", choices=("type1", "constant"), default="type1"
    )
    parser.add_argument("--minimum-oom-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quiet", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> PVGISCATCHConfig:
    return PVGISCATCHConfig(
        pvgis_dir=args.pvgis_dir,
        train_years=tuple(args.train_years),
        test_year=args.test_year,
        file_template=args.file_template,
        out_dir=args.out_dir,
        sensors=tuple(args.sensors),
        max_locations=args.max_locations,
        seq_len=args.seq_len,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        inference_patch_size=args.inference_patch_size,
        inference_patch_stride=args.inference_patch_stride,
        cf_dim=args.cf_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        head_layers=args.head_layers,
        temperature=args.temperature,
        mask_source=args.mask_source,
        frequency_loss_weight=args.frequency_loss_weight,
        clustering_weight=args.clustering_weight,
        regularization_weight=args.regularization_weight,
        score_frequency_weight=args.score_frequency_weight,
        contamination=args.contamination,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        mask_learning_rate=args.mask_lr,
        validation_split=args.validation_split,
        patience=args.patience,
        training_window_stride=args.training_window_stride,
        scoring_window_stride=args.scoring_window_stride,
        model_steps_per_mask=args.model_steps_per_mask,
        gradient_clip=args.gradient_clip,
        lr_adjustment=args.lr_adjustment,
        minimum_oom_batch_size=args.minimum_oom_batch_size,
        device=args.device,
        seed=args.seed,
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
    return run_pvgis_catch(config)


def main() -> None:
    parser = build_arg_parser()
    run_from_args(parser.parse_args(), parser)


if __name__ == "__main__":
    main()
