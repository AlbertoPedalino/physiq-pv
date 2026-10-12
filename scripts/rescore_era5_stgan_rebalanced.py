"""Monte Carlo spread of the rebalanced reconstruction error of an ERA5 STGAN run.

`rebalance_era5_stgan_scores.py` builds its score from the errors by variable that the scoring
saved, which are already means over the dropout draws: that score has no spread. This script
scores again with the run's checkpoint, puts the errors of every draw on the common scale (the
same one, fitted on the saved errors of the run) and takes the mean and the standard deviation of
the rebalanced score over the draws, as the scoring does for the original score.

Without `--output-dir` it scores `--timestamps` test timestamps, evenly spaced, and prints

- the spread by band of the score, in absolute value and as a share of the score;
- how far the mean over the draws is from the score built on the mean errors (with `rarity` the
  scale is not linear, so the two differ);
- the time a complete scoring would take.

With `--output-dir` it scores every timestamp and writes a run with the mean and the spread, which
`climatology`, `events` and the report accept. The model must score one whole timestamp per batch
(GAT, or the full-grid ConvGRU).

    python scripts/rescore_era5_stgan_rebalanced.py --prepared-dir data/era5_prepared \
        --run-dir outputs/run --epoch 6 --scale rarity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

import physiq_pv.anomaly_detection.stgan as stgan
from physiq_pv.anomaly_detection.stgan.precision import autocast_context
from physiq_pv.anomaly_detection.stgan.scoring import scoring_mode
from physiq_pv.era5.data import load_prepared
from scripts import analyze_era5_stgan_checkpoints as checkpoints
from scripts import rebalance_era5_stgan_scores as rebalance

BANDS = (0, 50, 90, 95, 99, 99.9, 100)  # percentiles of the score
PROGRESS_SECONDS = 300


def fitted_scales(run_dir, scale, features, scale_stride, names):
    """The columns kept and what `transform` needs, fitted on the errors by variable saved by the run."""
    unknown = [name for name in features or [] if name not in names]
    if unknown:
        raise ValueError(f"Unknown variables: {unknown}; the run has {list(names)}")
    columns = [names.index(name) for name in features] if features else list(range(len(names)))
    errors = np.load(Path(run_dir) / "scores" / "feature_scores.npy", mmap_mode="r")
    try:
        if errors.shape[2] != len(names):
            raise ValueError("The variables of the run do not match its errors by variable")
        sample = np.asarray(errors[::scale_stride][:, :, columns], dtype=np.float32).reshape(-1, len(columns))
    finally:
        errors._mmap.close()
    return columns, rebalance.fit(sample[np.isfinite(sample).all(axis=1)], scale)


def transform(errors, scale, fitted):
    """`rebalance.transform` on a tensor [..., variables], in float64."""
    errors = errors.double()
    if scale == "none":
        return errors
    if scale == "mean":
        return errors / torch.as_tensor(fitted, dtype=torch.float64, device=errors.device)
    result = torch.empty_like(errors)
    for index, (values, probabilities) in enumerate(fitted):
        values = torch.as_tensor(values, dtype=torch.float64, device=errors.device)
        probabilities = torch.as_tensor(probabilities, dtype=torch.float64, device=errors.device)
        if len(values) == 1:
            result[..., index] = -torch.log1p(-probabilities[0])
            continue
        # Linear interpolation between the points of the table, constant outside it, as numpy.interp.
        column = errors[..., index].clamp(values[0], values[-1]).contiguous()
        upper = torch.searchsorted(values, column).clamp(1, len(values) - 1)
        weight = (column - values[upper - 1]) / (values[upper] - values[upper - 1])
        probability = probabilities[upper - 1] + weight * (probabilities[upper] - probabilities[upper - 1])
        result[..., index] = -torch.log1p(-probability)
    return result


def rescore(model, dataset, positions, *, scale, columns, fitted, mc_samples=20, device="cpu", precision="fp32",
            on_row=None):
    """Mean and standard deviation over the draws of the rebalanced score, and the score built on the
    mean errors of the same draws, each as [timestamps, locations]."""
    shape = (len(positions), dataset.n_locations)
    mean, spread, of_mean_errors = (np.full(shape, np.nan, dtype=np.float32) for _ in range(3))
    with scoring_mode(model, True), torch.no_grad():
        for row, position in enumerate(positions):
            batch = dataset.fetch_batch([int(position)])
            recent, trend, mask, calendar, observed = (values.to(device) for values in batch[:5])
            with autocast_context(precision, device):
                draws = model.score_draws(recent, trend, mask, calendar, observed, mc_samples)
            errors = draws.float()[:, :, 2:][:, :, columns]
            scores = transform(errors, scale, fitted).mean(dim=2)
            cells = batch[-1].numpy().reshape(-1)
            if len(cells) != dataset.n_locations:
                raise ValueError("A batch must cover the whole grid.")
            mean[row, cells] = scores.mean(dim=0).cpu().numpy()
            spread[row, cells] = scores.std(dim=0, correction=0).cpu().numpy()
            of_mean_errors[row, cells] = transform(errors.mean(dim=0), scale, fitted).mean(dim=1).cpu().numpy()
            if on_row is not None:
                on_row(row)
    if not all(np.isfinite(values).all() for values in (mean, spread, of_mean_errors)):
        raise FloatingPointError("Non-finite model output.")
    return mean, spread, of_mean_errors


def band_table(mean, spread):
    """Rows (band, mean score, mean spread, spread over score) and the rank correlation of score and spread."""
    mean, spread = np.asarray(mean, dtype=np.float64).ravel(), np.asarray(spread, dtype=np.float64).ravel()
    edges = np.percentile(mean, BANDS)
    rows = []
    for index in range(len(BANDS) - 1):
        inside = (mean >= edges[index]) & (mean <= edges[index + 1])
        # With few points a narrow band can hold none.
        score, deviation = (mean[inside].mean(), spread[inside].mean()) if inside.any() else (np.nan, np.nan)
        rows.append((f"{BANDS[index]:g}-{BANDS[index + 1]:g}", score, deviation, deviation / score))
    order = lambda values: np.argsort(np.argsort(values))
    return rows, float(np.corrcoef(order(mean), order(spread))[0, 1])


def print_report(mean, spread, of_mean_errors):
    rows, correlation = band_table(mean, spread)
    print("\n=== Monte Carlo spread of the rebalanced score by score band ===")
    print(f"{'score percentiles':<20}{'mean score':>12}{'mean spread':>13}{'spread / score':>16}")
    for band, score, deviation, ratio in rows:
        print(f"{band:<20}{score:>12.4g}{deviation:>13.4g}{ratio:>16.4g}")
    print(f"rank correlation between score and spread: {correlation:+.3f}")
    difference = np.abs(mean - of_mean_errors).ravel()
    top = lambda values: values.ravel() >= np.percentile(values, 99)
    print("\n=== mean over the draws against the score built on the mean errors ===")
    print(f"correlation {np.corrcoef(mean.ravel(), of_mean_errors.ravel())[0, 1]:.4f} | mean difference "
          f"{difference.mean():.4g} (mean score {mean.mean():.4g}) | of the highest 1% of one, "
          f"{100 * top(of_mean_errors)[top(mean)].mean():.1f}% is in the highest 1% of the other")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run with model_epoch_<n>.pt and scores/feature_scores.npy")
    parser.add_argument("--epoch", type=int, required=True, help="The epoch checkpoint that produced the scores of the run")
    parser.add_argument("--scale", choices=[name for name in rebalance.SCALES if name != "none"], default="rarity")
    parser.add_argument("--features", nargs="+", help="Only these variables; default: all of them")
    parser.add_argument("--scale-stride", type=int, default=10, help="The scales are read on one timestamp every this many")
    parser.add_argument("--timestamps", type=int, default=400,
                        help="Test timestamps, evenly spaced over the period; ignored with --output-dir")
    parser.add_argument("--output-dir", type=Path, help="Score every timestamp and write the result as a run")
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.mc_samples < 2:
        raise ValueError("The spread needs at least two draws")
    if args.output_dir is not None and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use a new/empty output directory")
    manifest = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    cubes = load_prepared(args.prepared_dir)[0]
    try:
        names = list(cubes.feature_names)
        columns, fitted = fitted_scales(args.run_dir, args.scale, args.features, args.scale_stride, names)
        model, payload = stgan.load_stgan_checkpoint(args.run_dir / f"model_epoch_{args.epoch}.pt", device=args.device)
        if hasattr(model, "discriminator_chunk_size"):
            model.discriminator_chunk_size = 10611  # GAT: memory and speed only
        dataset = checkpoints.test_dataset(cubes, payload)
        positions = (np.arange(len(dataset)) if args.output_dir is not None
                     else np.linspace(0, len(dataset) - 1, min(args.timestamps, len(dataset))).astype(int))
        started, logged = perf_counter(), [perf_counter()]

        def progress(row):
            if perf_counter() - logged[0] >= PROGRESS_SECONDS:
                logged[0], elapsed = perf_counter(), perf_counter() - started
                print(f"[rescore] timestamp={row + 1}/{len(positions)} elapsed={elapsed / 3600:.2f}h "
                      f"eta={elapsed / (row + 1) * (len(positions) - row - 1) / 3600:.2f}h", flush=True)

        mean, spread, of_mean_errors = rescore(
            model, dataset, positions, scale=args.scale, columns=columns, fitted=fitted, mc_samples=args.mc_samples,
            device=args.device, precision=payload.get("precision", "fp32"), on_row=progress)
        elapsed = perf_counter() - started
        print(f"[rescore] scale={args.scale} variables={len(columns)} epoch={args.epoch} timestamps={len(positions)} "
              f"locations={dataset.n_locations} mc_samples={args.mc_samples} | {elapsed / len(positions):.2f} s per timestamp, "
              f"{elapsed / len(positions) * len(dataset) / 3600:.1f} h for all the {len(dataset)} timestamps")
        timestamps = cubes.test_timestamps
    finally:
        cubes.close()
    print_report(mean, spread, of_mean_errors)
    if args.output_dir is not None:
        if len(timestamps) != len(mean):
            raise ValueError("The scored timestamps are not the test timestamps of the run")
        (args.output_dir / "scores").mkdir(parents=True)
        np.save(args.output_dir / "scores" / "anomaly_mean.npy", mean)
        np.save(args.output_dir / "scores" / "anomaly_std.npy", spread)
        np.save(args.output_dir / "test_timestamps.npy", timestamps.as_unit("ns").asi8)
        result = {"status": "complete", "grid": manifest["grid"], "score_kind": "rebalanced_reconstruction_error_mc",
                  "source_run_dir": str(args.run_dir), "epoch": args.epoch, "scale": args.scale,
                  "scale_stride": args.scale_stride, "features": [names[index] for index in columns],
                  "mc_samples": args.mc_samples, "anomaly_mean_file": "scores/anomaly_mean.npy",
                  "anomaly_std_file": "scores/anomaly_std.npy"}
        (args.output_dir / "metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[rescore] written: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
