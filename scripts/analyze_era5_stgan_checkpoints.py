"""How the generator and the discriminator treat the most anomalous observations, at every epoch checkpoint.

A finished run saves the components of its score. A run that is still training, or that was
stopped, has only its epoch checkpoints: this script scores a sample of test timestamps with each
of them, as the final scoring does (the mean of `--mc-samples` draws with the generator's dropout
on), and prints, per epoch,

- the distribution of the reconstruction error, of D(observation) and of D(reconstruction);
- how much of the discriminator component D(observation) - D(reconstruction) comes from each term;
- what the three quantities are on the observations with the highest score: whether D finds them
  unusual, or only tells real from generated;
- how many of the highest scores of an epoch are still the highest at the next one.

The combined score is the sum of the two components after a min-max fitted on the sample, so its
values are not those of the final scoring. D has no dropout: D(observation) is the same with or
without draws, while D(reconstruction) depends on them, because the reconstruction does.
`--mc-samples 0` gives one deterministic pass, as the validation monitoring of the training. The
model must score one whole timestamp per batch (GAT, or the full-grid ConvGRU).

    python scripts/analyze_era5_stgan_checkpoints.py \
        --prepared-dir data/era5_prepared --run-dir outputs/stgan_convgru_fullgrid_recent16_cyclic_fp32
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

import physiq_pv.anomaly_detection.stgan as stgan
from physiq_pv.anomaly_detection.stgan.data import ContextArray, prepend_training_context_to_test
from physiq_pv.anomaly_detection.stgan.precision import autocast_context
from physiq_pv.anomaly_detection.stgan.scoring import scoring_mode
from physiq_pv.era5.data import load_prepared

TOP_PERCENTS = (5, 1, 0.1)


def test_dataset(cubes, payload):
    """The test dataset of the run: raw values, the run's min-max, the last training steps as context."""
    if payload["model_class"] == "STGAN_GAT":
        dataset_type = stgan.STGANGraphDataset
    elif payload.get("cnn_training_mode") == "full_grid":
        dataset_type = stgan.STGANFullGridDataset
    else:
        raise ValueError("The checkpoint must score one whole timestamp per batch (GAT or full-grid ConvGRU).")
    normalization = payload["normalization"]
    if "seasonal" in normalization.get("kind", ""):
        raise ValueError("Runs with seasonal normalization need their standardised cache; not supported here.")
    grid = stgan.build_spatial_grid(cubes.latitudes, cubes.longitudes, patch_size=payload["model_config"]["patch_size"],
                                    grid_crs="EPSG:4326", angular_spacing=.5, audit_knn=False)
    # The steps that precede the test are the end of 2004 whether or not 2004 was held out.
    window = payload["window_config"]
    data, times = prepend_training_context_to_test(
        ContextArray(cubes.train, cubes.validation), cubes.test,
        train_timestamps=cubes.train_timestamps.append(cubes.validation_timestamps),
        test_timestamps=cubes.test_timestamps, context_steps=window["trend_steps"], materialize=False)
    return dataset_type(data, times, grid, feature_minimum=np.asarray(normalization["minimum"]),
                        feature_scale=np.asarray(normalization["scale"]), recent_steps=window["recent_steps"],
                        trend_steps=window["trend_steps"], stride=1, normalized=False,
                        time_encoding=payload.get("time_encoding", "onehot"), longitudes=cubes.longitudes)


def checkpoint_outputs(model, dataset, positions, *, device="cpu", precision="fp32", mc_samples=0):
    """Reconstruction error, D(observation) and D(reconstruction) as [timestamps, locations].

    With `mc_samples` the error and D(reconstruction) are means over that many dropout draws, as
    in the final scoring; with 0 they come from one pass with dropout off.
    """
    shape = (len(positions), dataset.n_locations)
    error, real, generated = (np.full(shape, np.nan) for _ in range(3))
    with scoring_mode(model, mc_samples > 0), torch.no_grad():
        for row, position in enumerate(positions):
            batch = dataset.fetch_batch([int(position)])
            recent, trend, mask, calendar, observed = (values.to(device) for values in batch[:5])
            with autocast_context(precision, device):
                observation = model.observation_scores(recent, mask, observed)
                if mc_samples:
                    draws = model.score_draws(recent, trend, mask, calendar, observed, mc_samples).float().mean(0)
                    reconstruction, component = draws[:, 0], draws[:, 1]
                else:
                    reconstruction, component = model.monitoring_outputs(recent, trend, mask, calendar, observed)[:2]
            cells = batch[-1].numpy().reshape(-1)
            if len(cells) != dataset.n_locations:
                raise ValueError("A batch must cover the whole grid.")
            error[row, cells] = reconstruction.double().cpu().numpy().reshape(-1)
            real[row, cells] = observation.double().cpu().numpy().reshape(-1)
            generated[row, cells] = real[row, cells] - component.double().cpu().numpy().reshape(-1)
    if not all(np.isfinite(values).all() for values in (error, real, generated)):
        raise FloatingPointError("Non-finite model output.")
    return {"error": error.ravel(), "real": real.ravel(), "generated": generated.ravel()}


def combined_score(outputs):
    """The sum of the two components after a min-max fitted on the sample."""
    component = outputs["real"] - outputs["generated"]
    return sum((values - values.min()) / max(values.max() - values.min(), 1e-12) for values in (outputs["error"], component))


def summary(outputs):
    """The numbers printed for one checkpoint."""
    error, real, generated = outputs["error"], outputs["real"], outputs["generated"]
    component, score = real - generated, combined_score(outputs)
    result = {
        "error": np.percentile(error, (50, 95, 99, 100)), "real": np.percentile(real, (50, 75, 95, 99)),
        "generated": np.percentile(generated, (50, 25, 5, 1)), "component": np.percentile(component, (50, 95, 99)),
        "real_above_half": float((real > 0.5).mean()),
        "variance": (float(real.var()), float(generated.var()), float(component.var())),
        "correlation_with_score": (float(np.corrcoef(score, error)[0, 1]), float(np.corrcoef(score, component)[0, 1])),
        "correlation_between_components": float(np.corrcoef(error, component)[0, 1]), "top": {}}
    real_median, generated_median = np.median(real), np.median(generated)
    for name, values in (("score", score), ("component", component)):
        for percent in TOP_PERCENTS:
            top = values >= np.percentile(values, 100 - percent)
            # Which term of the component left its usual value more: D(observation) rising or D(reconstruction) falling.
            observation_moved_more = (real[top] - real_median) > (generated_median - generated[top])
            result["top"][name, percent] = (float(real[top].mean()), float(generated[top].mean()),
                                            float(error[top].mean()), float(observation_moved_more.mean()))
    return result


def print_summary(epoch, result):
    def numbers(values, pattern="{:.4f}"):
        return " ".join(pattern.format(value) for value in values)
    print(f"\n=== epoch {epoch} ===")
    print(f"reconstruction error  p50/p95/p99/max: {numbers(result['error'])}")
    print(f"D(observation)        p50/p75/p95/p99: {numbers(result['real'])} | above 0.5: {100 * result['real_above_half']:.2f}%")
    print(f"D(reconstruction)     p50/p25/p5/p1:   {numbers(result['generated'])}")
    print(f"component             p50/p95/p99:     {numbers(result['component'])} | variance: observation "
          f"{result['variance'][0]:.4f}, reconstruction {result['variance'][1]:.4f}, component {result['variance'][2]:.4f}")
    print(f"correlation with the combined score: reconstruction error {result['correlation_with_score'][0]:.2f}, "
          f"component {result['correlation_with_score'][1]:.2f} | between the two {result['correlation_between_components']:.2f}")
    medians = (result["real"][0], result["generated"][0], result["error"][0])
    print(f"medians: D(observation) {medians[0]:.4f} | D(reconstruction) {medians[1]:.4f} | reconstruction error {medians[2]:.4f}")
    for name in ("score", "component"):
        for percent in TOP_PERCENTS:
            real, generated, error, moved = result["top"][name, percent]
            print(f"top {percent:g}% of the {name}: D(observation) {real:.3f} | D(reconstruction) {generated:.3f} | "
                  f"reconstruction error {error:.4f} | observation term moved more in {100 * moved:.0f}%")


def print_stability(by_epoch):
    """Of the highest 1% of an epoch, the share that is in the highest 1% and 5% of the next one."""
    epochs = sorted(by_epoch)
    quantities = {"reconstruction error": lambda o: o["error"], "component": lambda o: o["real"] - o["generated"],
                  "combined score": combined_score}
    print("\n=== stability between consecutive epochs (two unrelated rankings would give 1% and 5%) ===")
    for name, quantity in quantities.items():
        values = {epoch: quantity(by_epoch[epoch]) for epoch in epochs}
        for earlier, later in zip(epochs[:-1], epochs[1:]):
            top = values[earlier] >= np.percentile(values[earlier], 99)
            kept = [float((values[later][top] >= np.percentile(values[later], 100 - percent)).mean()) for percent in (1, 5)]
            print(f"{name}, epoch {earlier} -> {later}: of the top 1%, {100 * kept[0]:.1f}% stays in the top 1%, "
                  f"{100 * kept[1]:.1f}% in the top 5%")


def epoch_of(path):
    return int(re.search(r"_epoch_(\d+)\.pt$", path.name).group(1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory with model_epoch_<n>.pt")
    parser.add_argument("--timestamps", type=int, default=400, help="Test timestamps, evenly spaced over the period")
    parser.add_argument("--epochs", type=int, nargs="+", help="Only these epochs; default: every checkpoint")
    parser.add_argument("--mc-samples", type=int, default=20,
                        help="Dropout draws averaged per timestamp, as in the final scoring; 0: one pass with dropout off")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    paths = sorted(args.run_dir.glob("model_epoch_*.pt"), key=epoch_of)
    if args.epochs:
        paths = [path for path in paths if epoch_of(path) in args.epochs]
    if not paths:
        raise ValueError(f"No epoch checkpoint in {args.run_dir}")
    cubes = load_prepared(args.prepared_dir)[0]
    by_epoch = {}
    try:
        for path in paths:
            model, payload = stgan.load_stgan_checkpoint(path, device=args.device)
            if hasattr(model, "discriminator_chunk_size"):
                model.discriminator_chunk_size = 10611  # GAT: memory and speed only
            dataset = test_dataset(cubes, payload)
            positions = np.linspace(0, len(dataset) - 1, min(args.timestamps, len(dataset))).astype(int)
            by_epoch[epoch_of(path)] = checkpoint_outputs(model, dataset, positions, device=args.device,
                                                          precision=payload.get("precision", "fp32"),
                                                          mc_samples=args.mc_samples)
            print(f"[checkpoints] epoch={epoch_of(path)} scored: timestamps={len(positions)} "
                  f"locations={dataset.n_locations} mc_samples={args.mc_samples}", flush=True)
            del model
    finally:
        cubes.close()
    for epoch in sorted(by_epoch):
        print_summary(epoch, summary(by_epoch[epoch]))
    if len(by_epoch) > 1:
        print_stability(by_epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
