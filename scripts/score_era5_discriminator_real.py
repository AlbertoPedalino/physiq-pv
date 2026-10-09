"""Add discriminator_real.npy to a finished ERA5 STGAN run, without training or Monte Carlo draws.

The discriminator component saved by a run is D(observation) - mean D(reconstruction). Runs
scored before discriminator_real.npy existed kept only that difference. This script loads the
final checkpoint of such a run and computes D(observation) on the test period: D has no dropout,
so the value is the one every scoring draw used. Nothing else in the run directory is changed.

    python scripts/score_era5_discriminator_real.py \
        --prepared-dir data/era5_prepared --run-dir outputs/era5_gat_mc_paper_recent16_run1
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from physiq_pv.era5.data import load_prepared
from physiq_pv.anomaly_detection.stgan import (
    STGANGraphDataset, STGANWindowDataset, build_spatial_grid, load_stgan_checkpoint,
)
from physiq_pv.anomaly_detection.stgan.data import ContextArray, prepend_training_context_to_test
from physiq_pv.anomaly_detection.stgan.loading import close_loader, make_loader
from physiq_pv.anomaly_detection.stgan.precision import autocast_context
from physiq_pv.anomaly_detection.stgan.scoring import scoring_mode

PROGRESS_SECONDS = 60


def score_observations(model, dataset, output_path, *, device="cpu", batch_size=1, num_workers=0,
                       precision="fp32"):
    """Write D's score of every observation of `dataset` as [timestamps, locations] and return it."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32,
                                       shape=(len(dataset.targets), dataset.n_locations))
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False, device=device, num_workers=num_workers)
    try:
        total, started, logged = len(loader), perf_counter(), None
        with scoring_mode(model, False), torch.no_grad():
            for index, batch in enumerate(loader, start=1):
                recent, _, mask, _, observed = (x.to(device, non_blocking=True) for x in batch[:5])
                with autocast_context(precision, device):
                    real = model.observation_scores(recent, mask, observed)
                real = real.float().cpu().numpy()
                if not np.isfinite(real).all():
                    raise FloatingPointError("Non-finite discriminator output; nothing usable was written.")
                time, location = batch[-2].numpy(), batch[-1].numpy()
                scores[time, location] = real.reshape(time.shape)
                now = perf_counter()
                if logged is None or now - logged >= PROGRESS_SECONDS or index == total:
                    logged, elapsed = now, now - started
                    print(f"[discriminator-real] batch={index}/{total} elapsed={elapsed / 3600:.2f}h "
                          f"eta={elapsed / index * (total - index) / 3600:.2f}h", flush=True)
        scores.flush()
        return scores
    except BaseException:
        del scores
        output_path.unlink(missing_ok=True)
        raise
    finally:
        close_loader(loader)


def test_dataset(cubes, payload, *, patch_size):
    """The test dataset of the run: raw values, the run's min-max, the last training steps as context."""
    grid = build_spatial_grid(cubes.latitudes, cubes.longitudes, patch_size=patch_size,
                              grid_crs="EPSG:4326", angular_spacing=.5, audit_knn=False)
    # The steps that precede the test are the end of 2004 whether or not 2004 was held out.
    context = ContextArray(cubes.train, cubes.validation)
    context_times = cubes.train_timestamps.append(cubes.validation_timestamps)
    window = payload["window_config"]
    data, times = prepend_training_context_to_test(
        context, cubes.test, train_timestamps=context_times, test_timestamps=cubes.test_timestamps,
        context_steps=window["trend_steps"], materialize=False)
    # Checkpoints written before time_encoding existed used the one-hot calendar.
    time_encoding = payload.get("time_encoding", "onehot")
    normalization = payload["normalization"]
    kind = normalization.get("kind", "training_only_feature_minmax_to_minus_one_one")
    if "seasonal" in kind:
        raise ValueError("Runs with seasonal normalization need their standardised cache; not supported here.")
    dataset_type = STGANGraphDataset if payload["model_class"] == "STGAN_GAT" else STGANWindowDataset
    return dataset_type(data, times, grid, feature_minimum=np.asarray(normalization["minimum"]),
                        feature_scale=np.asarray(normalization["scale"]),
                        recent_steps=window["recent_steps"], trend_steps=window["trend_steps"], stride=1,
                        normalized=False, time_encoding=time_encoding, longitudes=cubes.longitudes)


def check_against_saved(scores, score_dir, model, dataset, *, device, precision, timestamps, mc_samples):
    """Compare with what the run saved: the implied D(reconstruction) must be a probability, and
    a few timestamps rescored with Monte Carlo draws must give the saved component again."""
    saved = np.load(Path(score_dir) / "discriminator_mean.npy", mmap_mode="r")
    if saved.shape != scores.shape:
        raise ValueError(f"Saved component has shape {saved.shape}, the new array {scores.shape}.")
    rows = np.linspace(0, len(scores) - 1, min(len(scores), 2000)).astype(int)
    generated = np.asarray(scores[rows], dtype=np.float64) - np.asarray(saved[rows], dtype=np.float64)
    outside = float(((generated < -1e-3) | (generated > 1 + 1e-3)).mean())
    print(f"[discriminator-real] implied D(reconstruction): min={generated.min():.4f} max={generated.max():.4f} "
          f"share outside [0,1]={outside:.5f} (0 expected)", flush=True)
    if not timestamps:
        return outside, None
    spread = np.load(Path(score_dir) / "discriminator_std.npy", mmap_mode="r") \
        if (Path(score_dir) / "discriminator_std.npy").exists() else None
    differences, noise = [], []
    with scoring_mode(model, True), torch.no_grad():
        for position in np.linspace(0, len(dataset) - 1, timestamps).astype(int):
            batch = dataset.fetch_batch([int(position)])
            recent, trend, mask, calendar, observed = (x.to(device) for x in batch[:5])
            with autocast_context(precision, device):
                draws = model.score_draws(recent, trend, mask, calendar, observed, mc_samples)
            component = draws[:, :, 1].float().mean(0).cpu().numpy()
            time, location = batch[-2].numpy().reshape(-1), batch[-1].numpy().reshape(-1)
            differences.append(np.abs(component - saved[time, location]))
            if spread is not None:
                noise.append(spread[time, location] / np.sqrt(mc_samples))
    difference = float(np.concatenate(differences).mean())
    expected = float(np.concatenate(noise).mean()) if noise else float("nan")
    print(f"[discriminator-real] {timestamps} timestamps rescored with {mc_samples} draws: mean |new - saved| "
          f"of the component={difference:.5f}; Monte Carlo noise expected about {expected:.5f}", flush=True)
    return outside, difference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="Finished run: model.pt and scores/")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1, help="Timestamps per batch")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--discriminator-chunk-size", type=int, default=10611,
                        help="GAT: centers D processes together; memory and speed only")
    parser.add_argument("--check-timestamps", type=int, default=3,
                        help="Timestamps rescored with Monte Carlo draws to compare with the saved component; 0 skips")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    score_dir = args.run_dir / "scores"
    output = score_dir / "discriminator_real.npy"
    if not (args.run_dir / "model.pt").is_file() or not (score_dir / "discriminator_mean.npy").is_file():
        raise SystemExit("Not a finished run: model.pt and scores/discriminator_mean.npy are required.")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} already exists; pass --overwrite to compute it again.")
    model, payload = load_stgan_checkpoint(args.run_dir / "model.pt", device=args.device)
    if hasattr(model, "discriminator_chunk_size"):
        model.discriminator_chunk_size = args.discriminator_chunk_size
    precision = payload.get("precision", "fp32")
    cubes, _, _ = load_prepared(args.prepared_dir)
    try:
        dataset = test_dataset(cubes, payload, patch_size=payload["model_config"]["patch_size"])
        print(f"[discriminator-real] model={payload['model_class']} precision={precision} "
              f"recent_steps={payload['window_config']['recent_steps']} timestamps={len(dataset.targets)} "
              f"locations={dataset.n_locations}", flush=True)
        scores = score_observations(model, dataset, output, device=args.device, batch_size=args.batch_size,
                                    num_workers=args.num_workers, precision=precision)
        mc_samples = payload.get("mc_config", {}).get("mc_samples", 20)
        check_against_saved(scores, score_dir, model, dataset, device=args.device, precision=precision,
                            timestamps=args.check_timestamps, mc_samples=mc_samples)
        print(f"[discriminator-real] wrote {output}", flush=True)
    finally:
        cubes.close()


if __name__ == "__main__":
    main()
