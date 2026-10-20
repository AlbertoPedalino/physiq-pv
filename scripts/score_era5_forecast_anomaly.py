"""The anomaly the generator forecasts: how far from the seasonal normal it expects the next field to be.

The generator reads the history up to a timestamp and gives the field of the next one. The score of the
run uses its error, known once the next field is there; the forecast itself is known before. This script
reads, with an epoch checkpoint and one pass with dropout off, for every target timestamp T of the period:
- the forecast anomaly: the weighted mean over the variables of (forecast - normal)^2, in the units and with
  the weights of the reconstruction error, the normal being the seasonal normal (0 of the standardisation);
- the observed anomaly: the same for the observed field of T.
Two runs are written, both over the timestamps of the period and both known at the timestamp they are set
at, as a score that looks ahead must be: the forecast anomaly of T + 1 at T (`--output-dir`), and the
observed anomaly of T at T (`--persistence-dir`), the reference that says the next field is as this one.
`climatology` reads them as any score.

    python scripts/score_era5_forecast_anomaly.py --run-dir outputs/run_scored --checkpoint outputs/run/model_epoch_6.pt \
        --prepared-dir data/era5_prepared --period test --output-dir outputs/test_forecast \
        --persistence-dir outputs/test_persistence
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

# The package next to the script, not the one of the working directory (see score_era5_stgan_run.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

import physiq_pv.anomaly_detection.stgan as stgan
from physiq_pv.anomaly_detection.stgan.precision import autocast_context
from physiq_pv.anomaly_detection.stgan.scoring import scoring_mode
from physiq_pv.anomaly_detection.stgan.shared_cache import MANIFEST
from physiq_pv.era5.data import load_prepared
from scripts import analyze_era5_stgan_checkpoints as checkpoints

PROGRESS_EVERY = 2000  # timestamps


def validation_dataset(cubes, payload, cache_dir):
    """The validation years with their real history, read from the shared cache as the test is."""
    if payload.get("cnn_training_mode") != "full_grid" and payload["model_class"] != "STGAN_GAT":
        raise ValueError("The checkpoint must score one whole timestamp per batch (GAT or full-grid ConvGRU).")
    dataset_type = stgan.STGANGraphDataset if payload["model_class"] == "STGAN_GAT" else stgan.STGANFullGridDataset
    data = np.load(Path(cache_dir) / "validation.npy", mmap_mode="r")
    built = json.loads((Path(cache_dir) / MANIFEST).read_text(encoding="utf-8"))["built"]
    window = payload["window_config"]
    context = len(data) - len(cubes.validation_timestamps)
    if context < window["trend_steps"] or context > len(cubes.train_timestamps):
        raise ValueError("The cache does not hold the validation period with the context of the run")
    times = cubes.train_timestamps[len(cubes.train_timestamps) - context:].append(cubes.validation_timestamps)
    grid = stgan.build_spatial_grid(cubes.latitudes, cubes.longitudes, patch_size=payload["model_config"]["patch_size"],
                                    grid_crs="EPSG:4326", angular_spacing=.5, audit_knn=False)
    return dataset_type(data, times, grid, feature_minimum=np.asarray(built["minimum"]),
                        feature_scale=np.asarray(built["scale"]), recent_steps=window["recent_steps"],
                        trend_steps=window["trend_steps"], stride=1, normalized=False,
                        time_encoding=payload.get("time_encoding", "onehot"), longitudes=cubes.longitudes)


def anomalies(model, dataset, ahead, observed, *, device="cpu", precision="fp32"):
    """Write the observed anomaly of every target timestamp of `dataset` in `observed` [targets, cells], and the
    forecast anomaly of target i + 1 in `ahead[i]` (the last row stays NaN): the forecast known at i."""
    weights = getattr(model, "reconstruction_feature_weights", None)
    weights = np.ones(len(dataset.minimum)) if weights is None else weights.detach().float().cpu().numpy()
    normal = (0.0 - np.asarray(dataset.minimum, dtype=np.float64)) / np.asarray(dataset.scale, dtype=np.float64) * 2 - 1
    ahead[-1] = np.nan
    started = perf_counter()
    with scoring_mode(model, False), torch.no_grad():
        for position in range(len(dataset)):
            if position and position % PROGRESS_EVERY == 0:
                elapsed = perf_counter() - started
                print(f"[forecast-anomaly] {position}/{len(dataset)} elapsed={elapsed / 60:.1f} min "
                      f"eta={elapsed / position * (len(dataset) - position) / 60:.1f} min", flush=True)
            batch = dataset.fetch_batch([position])
            recent, trend, mask, calendar, cells = (values.to(device) for values in batch[:5])
            with autocast_context(precision, device):
                seen, generated = model.reconstructed_cells(recent, trend, mask, calendar, cells)
            seen, generated = (values.float().cpu().numpy().astype(np.float64) for values in (seen, generated))
            sample, locations = batch[-2].numpy().reshape(-1), batch[-1].numpy().reshape(-1)
            if not (sample == position).all():  # one whole timestamp per batch, the cells of that target
                raise RuntimeError("The dataset does not give one whole timestamp per batch")
            observed[position, locations] = ((seen - normal) ** 2 * weights).mean(axis=1)
            if position:
                ahead[position - 1, locations] = ((generated - normal) ** 2 * weights).mean(axis=1)


def open_run(directory, shape):
    """The score array [timestamps, cells] of a new run, on disk."""
    directory = Path(directory)
    (directory / "scores").mkdir(parents=True)
    return np.lib.format.open_memmap(directory / "scores" / "anomaly_mean.npy", mode="w+", dtype=np.float32, shape=shape)


def write_run(directory, timestamps, grid, period, kind, source):
    directory = Path(directory)
    np.save(directory / "test_timestamps.npy", timestamps.as_unit("ns").asi8)
    (directory / "metadata.json").write_text(json.dumps({
        "status": "complete", "scored_period": period, "grid": grid, "score_kind": kind, **source,
        "anomaly_mean_file": "scores/anomaly_mean.npy", "anomaly_std_file": None}, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Complete run of the model (its grid and cache)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--period", choices=("test", "validation"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True, help="Run of the forecast anomaly")
    parser.add_argument("--persistence-dir", type=Path, required=True, help="Run of the observed anomaly")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    for directory in (args.output_dir, args.persistence_dir):
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Use a new/empty directory: {directory}")
    manifest = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    seasonal = manifest.get("backend", {}).get("seasonal_normalization")
    if not seasonal:
        raise ValueError("The run has no seasonal normalization: there is no seasonal normal to measure from")
    cache_dir = Path(seasonal["mean_file"]).parent
    model, payload = stgan.load_stgan_checkpoint(args.checkpoint, device=args.device)
    precision = payload.get("precision", "fp32") if torch.device(args.device).type == "cuda" else "fp32"
    train_end = (manifest.get("config") or {}).get("train_end_year")
    cubes = load_prepared(args.prepared_dir, train_end_year=train_end)[0] if train_end else load_prepared(args.prepared_dir)[0]
    try:
        dataset = (checkpoints.test_dataset(cubes, payload, cache_dir=cache_dir) if args.period == "test"
                   else validation_dataset(cubes, payload, cache_dir))
        timestamps = dataset.target_timestamps
        shape = (len(dataset), dataset.n_locations)
        # Known at T: the forecast of T + 1, read from the history up to T; and the anomaly observed at T.
        ahead, observed = open_run(args.output_dir, shape), open_run(args.persistence_dir, shape)
        try:
            anomalies(model, dataset, ahead, observed, device=args.device, precision=precision)
        finally:
            for array in (ahead, observed):
                array.flush()
                array._mmap.close()
    finally:
        cubes.close()
    source = {"source_run_dir": str(args.run_dir), "checkpoint": str(args.checkpoint), "dropout": "off"}
    write_run(args.output_dir, timestamps, manifest["grid"], args.period, "forecast_anomaly_of_the_next_timestamp", source)
    write_run(args.persistence_dir, timestamps, manifest["grid"], args.period, "observed_anomaly", source)
    print(f"[forecast-anomaly] {args.period}: {shape[0]} timestamps, {shape[1]} cells -> {args.output_dir}, "
          f"{args.persistence_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
