"""Local ERA5 preparation, STGAN execution, and independent geographical events.

No automatic training: explicit prepare/train/climatology/events subcommands.
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
from physiq_pv.anomaly_detection.stgan.pca_reference import PLOT, build_pca_reference, load_pca_reference
from physiq_pv.anomaly_detection.stgan.mmd import build_mmd_reference
from physiq_pv.era5.events import EventConfig, process_events
from physiq_pv.era5.seasonality import climatology_zscores
from physiq_pv.anomaly_detection.stgan import STGANCNNConfig, fit_and_score_stgan
from physiq_pv.anomaly_detection.stgan.config import REFERENCE_CONFIG
from physiq_pv.anomaly_detection.stgan.data import ContextArray


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Read local raw NetCDFs; no download")
    prepare.add_argument("--input-dir", type=Path, default=Path("/home/apedalino/physiq_pv/data/era5"))
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--start-year", type=int, default=1980)
    prepare.add_argument("--train-end-year", type=int, default=2003,
                         help="Last training year")
    prepare.add_argument("--validation-end-year", type=int, default=2004,
                         help="Last validation year; the test starts the next year")
    prepare.add_argument("--end-year", default="latest")
    prepare.add_argument("--area", type=float, nargs=4, default=[60,-15,20,50], metavar=("N","W","S","E"))
    prepare.add_argument("--chunk-size", type=int, default=32)
    prepare.add_argument("--missing-policy", choices=("static_mask","error"), default="static_mask")
    train = commands.add_parser("train", help="Train through 2004 (through 2003 with --validation-holdout), then score paper-style fused anomaly scores from 2005")
    train.add_argument("--prepared-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--resume-from", type=Path,
                       help="Epoch checkpoint (model_epoch_N.pt) to continue from; same options as the "
                            "interrupted run. --output-dir may be the directory containing it")
    train.add_argument("--device", default="cuda")
    train.add_argument("--precision", choices=("fp32", "bf16"), default="fp32",
                       help="FP32 baseline or native CUDA BF16 mixed precision for training and scoring")
    train.add_argument("--seed", type=int, default=20)
    train.add_argument("--epochs", type=int, default=6)
    train.add_argument("--lr", "--learning-rate", "--generator-learning-rate", dest="lr", type=float,
                       default=REFERENCE_CONFIG.learning_rate, help="Generator Adam learning rate")
    discriminator_rate = train.add_mutually_exclusive_group()  # One way to give D's rate per run.
    discriminator_rate.add_argument("--discriminator-lr-ratio", type=float,
                       default=REFERENCE_CONFIG.discriminator_lr_ratio,
                       help="D learning rate / G learning rate; 1 preserves equal rates")
    discriminator_rate.add_argument("--discriminator-learning-rate", type=float, default=None,
                       help="D Adam learning rate, independent of G; use instead of --discriminator-lr-ratio")
    train.add_argument("--validation-holdout", action=argparse.BooleanOptionalAction, default=False,
                       help="Train through 2003 and keep 2004 as validation: per-epoch monitoring and validation objective only")
    train.add_argument("--monitoring-timestamps", type=int, default=REFERENCE_CONFIG.monitoring_timestamps,
                       help="Evenly spaced validation timestamps (every location) checked after each epoch; 0 = off")
    train.add_argument("--monitoring-feature-mmd-every-n-epochs", type=int,
                       default=REFERENCE_CONFIG.monitoring_feature_mmd_every_n_epochs,
                       help="Epochs between two MMD evaluations on the validation subset; 0 = never")
    train.add_argument("--monitoring-feature-mmd-samples", type=int, default=REFERENCE_CONFIG.monitoring_feature_mmd_samples,
                       help="Feature vectors per set in the MMD")
    train.add_argument("--discriminator-generator-update-ratio",
                       default=REFERENCE_CONFIG.discriminator_generator_update_ratio,
                       help="Optimizer steps per batch as D:G (1:1, 2:1, 1:2); not the learning-rate ratio")
    train.add_argument("--generator-reconstruction-weight", type=float,
                       default=REFERENCE_CONFIG.generator_reconstruction_weight,
                       help="Weight of reconstruction relative to G's adversarial loss")
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
    train.add_argument("--time-encoding", choices=("onehot", "cyclic"), default="onehot",
                       help="onehot: paper weekday+hour. cyclic: sine/cosine of per-cell local solar time "
                            "and of the position in the year; requires a new training run")
    train.add_argument("--normalization", choices=("minmax", "seasonal"), default="minmax",
                       help="minmax: train-only feature min-max. seasonal: first remove the mean annual and "
                            "diurnal cycle of each cell (train-only mean/std per day of year and time of day); "
                            "requires a new training run")
    train.add_argument("--seasonal-window-days", type=int, default=15,
                       help="Days pooled on each side of a day of year for the seasonal mean/std")
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--kernel-size", type=int, choices=(1,3,5), default=3)
    train.add_argument("--shuffle-mode", choices=("global","block","legacy"), default="global")
    train.add_argument("--dropout-enabled", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--dropout-p", type=float, default=0.2)
    train.add_argument("--mc-dropout-enabled", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--mc-samples", type=int, default=20)
    train.add_argument("--save-raw-mc", action=argparse.BooleanOptionalAction, default=False,
                       help="Retain raw MC components for debug; default discards temporary draws after aggregation.")
    train.add_argument("--pca-reference-dir", type=Path,
                       help="Fixed PCA feature space written by the pca-reference command: loaded and verified, "
                            "never refitted by a run")
    train.add_argument("--mmd-reference-dir", type=Path,
                       help="Reference written by the mmd-reference command, with the --pca-reference-dir it was "
                            "built on: enables validation/pca_mmd_* every epoch")
    train.add_argument("--mmd-objective-window", type=int, default=REFERENCE_CONFIG.mmd_objective_window,
                       help="Epochs averaged in validation/pca_mmd_rolling_mean")
    reference = commands.add_parser(
        "pca-reference", help="Fit once, before any run, the PCA feature space of complete training fields; "
                              "reads the training partition only")
    reference.add_argument("--prepared-dir", type=Path, required=True)
    reference.add_argument("--output-dir", type=Path, required=True)
    reference.add_argument("--pca-samples", type=int, default=4096,
                           help="Training timestamps, evenly spaced over the whole training period, the PCA is fitted on")
    reference.add_argument("--pca-components", type=int, default=100,
                           help="Leading components to store (fewer when the rank is lower); not a choice of how many to use")
    mmd = commands.add_parser(
        "mmd-reference", help="Build once, before any run, the fixed validation subsets and kernel bandwidth of the "
                              "validation MMD on a saved PCA reference; reads the validation partition only")
    mmd.add_argument("--prepared-dir", type=Path, required=True)
    mmd.add_argument("--pca-reference-dir", type=Path, required=True, help="Loaded and verified, never refitted")
    mmd.add_argument("--output-dir", type=Path, required=True)
    mmd.add_argument("--pca-components", type=int, default=100, help="Leading saved components the MMD uses")
    mmd.add_argument("--subsets", type=int, default=5)
    mmd.add_argument("--subset-size", type=int, default=512, help="Validation timestamps (complete fields) per subset")
    mmd.add_argument("--subset-seed", type=int, default=0)
    climatology = commands.add_parser(
        "climatology", help="Leave-one-year-out z-scores per location x month x UTC hour; writes a run for `events`")
    climatology.add_argument("--run-dir", type=Path, required=True)
    climatology.add_argument("--output-dir", type=Path, required=True)
    climatology.add_argument("--score-component", choices=("combined", "generator", "discriminator"),
                             default="combined")
    climatology.add_argument("--min-samples", type=int, default=30,
                             help="Minimum reference values per location/month/hour; fewer gives NaN")
    climatology.add_argument("--chunk-size", type=int, default=64)
    events = commands.add_parser("events", help="Post-process saved scores; no model/training")
    events.add_argument("--run-dir", type=Path, required=True)
    events.add_argument("--output-dir", type=Path, required=True)
    events.add_argument("--score-component", choices=("combined", "generator", "discriminator"),
                        default="combined", help="Combined paper score by default; raw components for diagnostics")
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


def score_files(run, component):
    """Mean and MC-std score files of one component in a complete run manifest."""
    if run.get("status") != "complete":
        raise ValueError("Scoring run is incomplete")
    if component == "combined":
        mean_file = run.get("anomaly_mean_file", run.get("scores_file"))
        std_file = run.get("anomaly_std_file")
    else:
        mean_file, std_file = run.get(f"{component}_mean_file"), run.get(f"{component}_std_file")
    if mean_file is None:
        raise ValueError(f"Run does not contain {component} component scores")
    return mean_file, std_file


def run_training(*, prepared_dir, output_dir, config, device="cuda", seed=20, on_epoch=None,
                 resume_from=None, pca_reference_dir=None, mmd_reference_dir=None):
    """Shared CLI/W&B entrypoint; preserve the ERA5 train/test and export protocol."""
    output = Path(output_dir)
    if resume_from is not None and not Path(resume_from).is_file():
        raise ValueError(f"Resume checkpoint not found: {resume_from}")
    # An interrupted run may be continued in place, from its own epoch checkpoint.
    in_place = resume_from is not None and Path(resume_from).resolve().parent == output.resolve()
    if output.exists() and any(output.iterdir()) and not in_place:
        raise ValueError("Use a new/empty STGAN run directory")
    if in_place and (output/"metadata.json").exists():
        raise ValueError("STGAN run directory is already complete")
    cubes, grid, preparation = load_prepared(prepared_dir)
    if (preparation.get("train_end_year"), preparation.get("validation_end_year"),
            preparation.get("test_start_year")) != (2003, 2004, 2005):
        cubes.close()
        raise ValueError("ERA5 paper protocol requires a cache with train through 2003, validation 2004 and test from 2005")
    # With a validation holdout 2004 stays out of training: monitoring and validation objective only.
    # Otherwise 1980-2004 trains, as in the runs that precede the validation split.
    holdout = config.validation_holdout
    train_data = cubes.train if holdout else ContextArray(cubes.train, cubes.validation)
    train_timestamps = (cubes.train_timestamps if holdout
                        else cubes.train_timestamps.append(cubes.validation_timestamps))
    validation = (dict(validation=cubes.validation, validation_timestamps=cubes.validation_timestamps)
                  if holdout else {})
    output.mkdir(parents=True,exist_ok=True)
    options = asdict(config)
    options["lr"] = options.pop("learning_rate")
    options.pop("validation_holdout")
    try:
        with fit_and_score_stgan(train_data,cubes.test,train_timestamps=train_timestamps,
            test_timestamps=cubes.test_timestamps,location_names=cubes.location_names,
            feature_names=cubes.feature_names,latitudes=cubes.latitudes,longitudes=cubes.longitudes,
            device=device,seed=seed,checkpoint_path=output/"model.pt",resume_from=resume_from,
            score_dir=output/"scores",timestep_hours=3,dataset_name="era5",
            angular_grid_spacing=.5,grid_audit_knn=False,score_mode="paper",on_epoch=on_epoch,pca_reference=pca_reference_dir,mmd_reference=mmd_reference_dir,**validation,**options) as result:
            np.save(output/"test_timestamps.npy",result.test_timestamps.as_unit("ns").asi8)
            metadata = {"status":"complete","grid":grid.to_dict(),"preparation":preparation,
                        "backend":result.metadata,"config":asdict(config),
                        "effective_train_end_year":2003 if holdout else 2004,
                        "validation_years":[2004] if holdout else None,
                        "scores_file":"scores/anomaly_mean.npy",
                        "anomaly_mean_file":"scores/anomaly_mean.npy",
                        "anomaly_std_file":"scores/anomaly_std.npy",
                        "generator_mean_file":"scores/generator_mean.npy",
                        "generator_std_file":"scores/generator_std.npy",
                        "discriminator_mean_file":"scores/discriminator_mean.npy",
                        "discriminator_std_file":"scores/discriminator_std.npy",
                        "component_covariance_file":"scores/component_covariance.npy"}
            (output/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    finally:
        cubes.close()
    return metadata


def main(argv=None):
    args = parse_args(argv)
    if args.command == "prepare":
        cubes, _, metadata = prepare_era5(args.input_dir,args.output_dir,
            start_year=args.start_year,train_end_year=args.train_end_year,score_end_year=args.end_year,
            validation_end_year=args.validation_end_year,
            area=args.area,chunk_size=args.chunk_size,missing_policy=args.missing_policy)
        cubes.close()
    elif args.command == "train":
        config = STGANCNNConfig(epochs=args.epochs,batch_size=args.batch_size, precision=args.precision,
            learning_rate=args.lr, discriminator_lr_ratio=args.discriminator_lr_ratio,
            discriminator_learning_rate=args.discriminator_learning_rate,
            validation_holdout=args.validation_holdout, monitoring_timestamps=args.monitoring_timestamps,
            monitoring_feature_mmd_every_n_epochs=args.monitoring_feature_mmd_every_n_epochs,
            monitoring_feature_mmd_samples=args.monitoring_feature_mmd_samples,
            mmd_objective_window=args.mmd_objective_window,
            discriminator_generator_update_ratio=args.discriminator_generator_update_ratio,
            generator_reconstruction_weight=args.generator_reconstruction_weight,
            score_batch_size=args.score_batch_size,num_workers=args.num_workers,
            kernel_size=args.kernel_size,trend_steps=56,grid_crs="EPSG:4326",
            dropout_enabled=args.dropout_enabled,dropout_p=args.dropout_p,
            mc_dropout_enabled=args.mc_dropout_enabled,mc_samples=args.mc_samples,
            save_raw_mc=args.save_raw_mc,
            spatial_encoder=args.spatial_encoder, gat_hidden_dim=args.gat_hidden_dim,
            gat_heads=args.gat_heads, gat_layers=args.gat_layers,
            discriminator_chunk_size=args.discriminator_chunk_size, recent_steps=args.recent_steps,
            time_encoding=args.time_encoding,
            normalization=args.normalization, seasonal_window_days=args.seasonal_window_days,
            trend_chunk_size=args.trend_chunk_size,
            score_storage="memmap",shuffle_mode=args.shuffle_mode)
        metadata = run_training(prepared_dir=args.prepared_dir, output_dir=args.output_dir,
                                config=config, device=args.device, seed=args.seed,
                                resume_from=args.resume_from, pca_reference_dir=args.pca_reference_dir,
                                mmd_reference_dir=args.mmd_reference_dir)
    elif args.command == "pca-reference":
        cubes, _, preparation = load_prepared(args.prepared_dir)
        try:
            if (preparation.get("train_end_year"), preparation.get("validation_end_year"),
                    preparation.get("test_start_year")) != (2003, 2004, 2005):
                raise ValueError("The PCA reference requires a cache with train through 2003, validation 2004 "
                                 "and test from 2005")
            # The training partition only: validation and test are never read.
            reference = build_pca_reference(cubes.train, cubes.train_timestamps, feature_names=cubes.feature_names,
                location_names=cubes.location_names, latitudes=cubes.latitudes, longitudes=cubes.longitudes,
                output_dir=args.output_dir, pca_samples=args.pca_samples, n_components=args.pca_components)
        finally:
            cubes.close()
        metadata = {**reference.summary(), "plot": str(reference.directory / PLOT)}
    elif args.command == "mmd-reference":
        cubes, _, preparation = load_prepared(args.prepared_dir)
        try:
            if (preparation.get("train_end_year"), preparation.get("validation_end_year"),
                    preparation.get("test_start_year")) != (2003, 2004, 2005):
                raise ValueError("The MMD reference requires a cache with train through 2003, validation 2004 "
                                 "and test from 2005")
            # The validation partition only: training and test are never read, the PCA is loaded as it is.
            reference = build_mmd_reference(cubes.validation, cubes.validation_timestamps,
                load_pca_reference(args.pca_reference_dir), feature_names=cubes.feature_names,
                location_names=cubes.location_names, latitudes=cubes.latitudes, longitudes=cubes.longitudes,
                output_dir=args.output_dir, n_subsets=args.subsets, subset_size=args.subset_size,
                subset_seed=args.subset_seed, n_components=args.pca_components)
        finally:
            cubes.close()
        metadata = reference.summary()
    elif args.command == "climatology":
        run = json.loads((args.run_dir/"metadata.json").read_text(encoding="utf-8"))
        score_file, std_file = score_files(run, args.score_component)
        output = args.output_dir
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use a new/empty climatology run directory")
        (output/"scores").mkdir(parents=True, exist_ok=True)
        timestamps = pd.DatetimeIndex(np.load(args.run_dir/"test_timestamps.npy"))
        scores = np.load(args.run_dir/score_file, mmap_mode="r")
        uncertainty = None if std_file is None else np.load(args.run_dir/std_file, mmap_mode="r")
        try:
            normalization = climatology_zscores(
                scores, timestamps, output/"scores"/"anomaly_mean.npy", uncertainty=uncertainty,
                uncertainty_output=None if uncertainty is None else output/"scores"/"anomaly_std.npy",
                min_samples=args.min_samples, chunk_size=args.chunk_size)
        finally:
            scores._mmap.close()
            if uncertainty is not None:
                uncertainty._mmap.close()
        np.save(output/"test_timestamps.npy", timestamps.as_unit("ns").asi8)
        metadata = {"status": "complete", "grid": run["grid"], "score_kind": "climatology_zscore",
                    "source_run_dir": str(args.run_dir), "source_score_component": args.score_component,
                    "source_score_file": score_file, "source_std_file": std_file,
                    "normalization": normalization, "anomaly_mean_file": "scores/anomaly_mean.npy",
                    "anomaly_std_file": None if uncertainty is None else "scores/anomaly_std.npy"}
        (output/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    else:
        run = json.loads((args.run_dir/"metadata.json").read_text(encoding="utf-8"))
        component = args.score_component
        score_file, std_file = score_files(run, component)
        scores = np.load(args.run_dir/score_file,mmap_mode="r")
        uncertainty = None
        try:
            if std_file is not None:
                uncertainty = np.load(args.run_dir/std_file,mmap_mode="r")
            options = {name:getattr(args,name) for name in asdict(EventConfig()) if hasattr(args,name)}
            metadata = process_events(scores,pd.DatetimeIndex(np.load(args.run_dir/"test_timestamps.npy")),
                CubeGrid(**run["grid"]),args.output_dir,EventConfig(**options), uncertainty=uncertainty)
            metadata["score_component"] = component
            metadata["source_score_file"] = score_file
            (args.output_dir/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        finally:
            scores._mmap.close()
            if uncertainty is not None:
                uncertainty._mmap.close()
    print(json.dumps(metadata,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
