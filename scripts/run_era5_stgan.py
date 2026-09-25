"""Local ERA5 preparation, STGAN execution, and independent geographical events.

No network or automatic training: explicit prepare/train/events subcommands.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from physiq_pv.era5.cube import CubeGrid
from physiq_pv.era5.data import load_prepared, prepare_era5
from physiq_pv.era5.events import EventConfig, process_events
from physiq_pv.anomaly_detection.stgan import STGANCNNConfig, fit_and_score_stgan


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Read local raw NetCDFs; no download")
    prepare.add_argument("--input-dir", type=Path, default=Path("/home/apedalino/physiq_pv/data/era5"))
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--start-year", type=int, default=1980)
    prepare.add_argument("--train-end-year", type=int, default=2002,
                         help="Last training year; calibration starts the following year")
    prepare.add_argument("--calibration-end-year", type=int, default=2004,
                         help="Last calibration year; test starts the following year")
    prepare.add_argument("--end-year", default="latest")
    prepare.add_argument("--area", type=float, nargs=4, default=[60,-15,20,50], metavar=("N","W","S","E"))
    prepare.add_argument("--chunk-size", type=int, default=32)
    prepare.add_argument("--missing-policy", choices=("static_mask","error"), default="static_mask")
    train = commands.add_parser("train", help="Explicitly start training/scoring of prepared data")
    train.add_argument("--prepared-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--seed", type=int, default=20)
    train.add_argument("--epochs", type=int, default=6)
    train.add_argument("--batch-size", type=int, default=None,
                       help="Training batch: default 1 global timestamp for GAT, 256 patches for ConvGRU")
    train.add_argument("--score-batch-size", type=int, default=None,
                       help="Scoring batch: default 1 global timestamp for GAT, 1024 patches for ConvGRU")
    train.add_argument("--spatial-encoder", choices=("gat", "convgru"), default="gat")
    train.add_argument("--gat-hidden-dim", type=int, default=16, help="Hidden features per attention head")
    train.add_argument("--gat-heads", type=int, default=4)
    train.add_argument("--gat-layers", type=int, choices=(2,), default=2)
    train.add_argument("--discriminator-chunk-size", type=int, default=256)
    train.add_argument("--trend-chunk-size", type=int, default=256)
    train.add_argument("--recent-steps", type=int, default=1)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--kernel-size", type=int, choices=(1,3,5), default=3)
    train.add_argument("--shuffle-mode", choices=("global","block","legacy"), default="global")
    train.add_argument("--dropout-enabled", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--dropout-p", type=float, default=0.2)
    train.add_argument("--mc-dropout-enabled", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--mc-samples", type=int, default=20)
    train.add_argument("--save-raw-mc", action=argparse.BooleanOptionalAction, default=False,
                       help="Retain raw MC components for debug; default discards temporary draws after aggregation.")
    events = commands.add_parser("events", help="Post-process saved scores; no model/training")
    events.add_argument("--run-dir", type=Path, required=True)
    events.add_argument("--output-dir", type=Path, required=True)
    events.add_argument("--top-percent", type=float, default=1)
    events.add_argument("--absolute-threshold", type=float)
    events.add_argument("--threshold-scope", choices=("global","frame"), default="global")
    events.add_argument("--kernel-size", type=int, default=3)
    events.add_argument("--opening-iterations", type=int, default=1)
    events.add_argument("--closing-iterations", type=int, default=1)
    events.add_argument("--min-cells", type=int, default=1)
    events.add_argument("--link-policy", choices=("overlap","dilated_overlap","centroid","any"), default="dilated_overlap")
    events.add_argument("--min-overlap", type=float, default=.1)
    events.add_argument("--dilation-cells", type=int, default=1)
    events.add_argument("--centroid-distance-km", type=float, default=100)
    events.add_argument("--chunk-size", type=int, default=32)
    return root


def parse_args(argv=None):
    args = parser().parse_args(argv)
    if args.command == "train":
        if args.batch_size is None:
            args.batch_size = 1 if args.spatial_encoder == "gat" else 256
        if args.score_batch_size is None:
            args.score_batch_size = 1 if args.spatial_encoder == "gat" else 1024
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "prepare":
        cubes, _, metadata = prepare_era5(args.input_dir,args.output_dir,
            start_year=args.start_year,train_end_year=args.train_end_year,score_end_year=args.end_year,
            calibration_end_year=args.calibration_end_year,
            area=args.area,chunk_size=args.chunk_size,missing_policy=args.missing_policy)
        cubes.close()
    elif args.command == "train":
        output = args.output_dir
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use a new/empty STGAN run directory")
        cubes, grid, preparation = load_prepared(args.prepared_dir)
        output.mkdir(parents=True,exist_ok=True)
        config = STGANCNNConfig(epochs=args.epochs,batch_size=args.batch_size,
            score_batch_size=args.score_batch_size,num_workers=args.num_workers,
            kernel_size=args.kernel_size,trend_steps=56,grid_crs="EPSG:4326",
            dropout_enabled=args.dropout_enabled,dropout_p=args.dropout_p,
            mc_dropout_enabled=args.mc_dropout_enabled,mc_samples=args.mc_samples,
            save_raw_mc=args.save_raw_mc,
            spatial_encoder=args.spatial_encoder, gat_hidden_dim=args.gat_hidden_dim,
            gat_heads=args.gat_heads, gat_layers=args.gat_layers,
            discriminator_chunk_size=args.discriminator_chunk_size, recent_steps=args.recent_steps,
            trend_chunk_size=args.trend_chunk_size,
            score_storage="memmap",shuffle_mode=args.shuffle_mode)
        options = asdict(config)
        options["lr"] = options.pop("learning_rate")
        try:
            with fit_and_score_stgan(cubes.train,cubes.test,train_timestamps=cubes.train_timestamps,
                calibration=cubes.calibration,calibration_timestamps=cubes.calibration_timestamps,
                test_timestamps=cubes.test_timestamps,location_names=cubes.location_names,
                feature_names=cubes.feature_names,latitudes=cubes.latitudes,longitudes=cubes.longitudes,
                device=args.device,seed=args.seed,checkpoint_path=output/"model.pt",
                score_dir=output/"scores",timestep_hours=3,dataset_name="era5",
                angular_grid_spacing=.5,grid_audit_knn=False,**options) as result:
                np.save(output/"test_timestamps.npy",result.test_timestamps.as_unit("ns").asi8)
                metadata = {"status":"complete","grid":grid.to_dict(),"preparation":preparation,
                            "backend":result.metadata,"config":asdict(config),
                            "scores_file":"scores/anomaly_mean.npy",
                            "anomaly_mean_file":"scores/anomaly_mean.npy",
                            "anomaly_std_file":"scores/anomaly_std.npy"}
                (output/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        finally:
            cubes.close()
    else:
        run = json.loads((args.run_dir/"metadata.json").read_text(encoding="utf-8"))
        if run.get("status") != "complete":
            raise ValueError("Scoring run is incomplete")
        scores = np.load(args.run_dir/run.get("anomaly_mean_file", run["scores_file"]),mmap_mode="r")
        uncertainty = None
        try:
            if "anomaly_std_file" in run:
                uncertainty = np.load(args.run_dir/run["anomaly_std_file"],mmap_mode="r")
            options = {name:getattr(args,name) for name in asdict(EventConfig()) if hasattr(args,name)}
            metadata = process_events(scores,pd.DatetimeIndex(np.load(args.run_dir/"test_timestamps.npy")),
                CubeGrid(**run["grid"]),args.output_dir,EventConfig(**options), uncertainty=uncertainty)
        finally:
            scores._mmap.close()
            if uncertainty is not None:
                uncertainty._mmap.close()
    print(json.dumps(metadata,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
