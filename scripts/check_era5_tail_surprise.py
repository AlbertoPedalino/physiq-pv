"""Where the model flags the trail of an event, does it flag the end of the event as a surprise?

The model's score is the error of a forecast one step ahead. When an event leaves a cell, a model that
expects it to go on forecasts weather that is still unusual while the observation is back near normal:
a large error at the end of the event, which is a real change of the weather but not an anomaly. This
check scores a sample of test timestamps with an epoch checkpoint, one pass with dropout off, and
reads at every cell, in the standardised units of the seasonal normalization (0 is the seasonal normal
of the cell), the observation, the forecast and the observation of the step before. Each point is read on
the variable that carries most of its weighted squared error (the main variable), and the check says
how often:
- ending: the observation is nearer the normal than the step before, |obs| < |prev|;
- beyond: the forecast is farther from the normal than the observation, |forecast| > |obs|;
- persistent: the forecast is nearer the step before than the observation, |forecast - prev| < |forecast - obs|;
- all three together;
- obs >1, obs >2: the observation is still more than 1 (2) standard deviations from the normal, an event
  that weakens but is still there;
- fcst only: the observation is within 1 of the normal and the forecast is not, the event is over but the
  model still forecasts it.

The points are in groups as in `check_era5_edge_lead.py`: at distance 1 to `--max-distance` cells from
the area the other scores flag at the same timestamp, the cells the model flags and the others do not
(model only) and the cells no score flags (neither); for each, by what the others do at the same cell
within `--window-hours`: front (after, not before), trail (before, not after), both, none. The cells the
model and another score flag together (inside) are a reference. If the trail of the model is the surprise
of the end, its points are ending, beyond and persistent more often than its fronts, than the inside and
than the trail of neither, which is also an event leaving but one the model does not flag.

    python scripts/check_era5_tail_surprise.py --prepared-dir data/era5_prepared --run-dir outputs/run \
        --epoch 3 --score-dir outputs/run_gen_clim --by-dir outputs/run_simple_max --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))  # also runs as a copy outside the repository, started from its root
import numpy as np
import pandas as pd
from scipy import ndimage
import torch

import physiq_pv.anomaly_detection.stgan as stgan
from physiq_pv.anomaly_detection.stgan.precision import autocast_context
from physiq_pv.anomaly_detection.stgan.scoring import scoring_mode
from physiq_pv.era5.data import load_prepared
from scripts import analyze_era5_stgan_checkpoints as checkpoints
from scripts.check_era5_dust_lead import THRESHOLD_STRIDE, score_file

MEASURES = ("ending", "beyond", "persistent", "all three", "obs >1", "obs >2", "fcst only")
KINDS = ("front", "trail", "both", "none")
GROUPS = ("model only", "neither")
NAME, WIDTH = 24, 11
PROGRESS_EVERY = 25  # timestamps


def forecasts(model, dataset, positions, *, device="cpu", precision="fp32"):
    """Per point (timestamp, cell) and variable, in the units of the dataset's data: the observation of the step
    before, the observation and the forecast, [points, variables]; the weighted squared error the model sees,
    [points, variables]; and the sample and the cell of every point."""
    weights = getattr(model, "reconstruction_feature_weights", None)
    previous, observed, forecast, errors, samples, places = [], [], [], [], [], []
    started = perf_counter()
    with scoring_mode(model, False), torch.no_grad():
        for index, position in enumerate(positions, start=1):
            if index % PROGRESS_EVERY == 0:
                elapsed = perf_counter() - started
                print(f"[tail-surprise] forecast {index}/{len(positions)} elapsed={elapsed / 60:.1f} min "
                      f"eta={elapsed / index * (len(positions) - index) / 60:.1f} min", flush=True)
            batch = dataset.fetch_batch([int(position)])
            recent, trend, mask, calendar, cells = (values.to(device) for values in batch[:5])
            with autocast_context(precision, device):
                seen, generated = model.reconstructed_cells(recent, trend, mask, calendar, cells)
            seen, generated = seen.float().cpu().numpy(), generated.float().cpu().numpy()
            sample, locations = batch[-2].numpy().reshape(-1), batch[-1].numpy().reshape(-1)
            target = dataset.targets[sample]
            now = np.asarray(dataset.data[target, locations], dtype=np.float64)
            # The cells of the model and the rows of the data must be the same points.
            if not np.allclose(seen, dataset._normalise(now), atol=1e-4):
                raise ValueError("The cells of the model do not match the data at the same points")
            square = (generated - seen) ** 2
            if weights is not None:
                square = square * weights.detach().float().cpu().numpy()
            previous.append(np.asarray(dataset.data[target - 1, locations], dtype=np.float64))
            observed.append(now)
            forecast.append((generated.astype(np.float64) + 1.0) / 2.0 * dataset.scale + dataset.minimum)
            errors.append(square)
            samples.append(sample)
            places.append(locations)
    arrays = [np.concatenate(values) for values in (previous, observed, forecast, errors)]
    if not all(np.isfinite(values).all() for values in arrays):
        raise FloatingPointError("Non-finite model output.")
    return (*arrays, np.concatenate(samples), np.concatenate(places))


def measures(previous, observed, forecast):
    """[points, MEASURES] booleans on one value per point."""
    ending = np.abs(observed) < np.abs(previous)
    beyond = np.abs(forecast) > np.abs(observed)
    persistent = np.abs(forecast - previous) < np.abs(forecast - observed)
    far = np.abs(observed)
    return np.column_stack((ending, beyond, persistent, ending & beyond & persistent, far > 1, far > 2,
                            (far <= 1) & (np.abs(forecast) > 1)))


def limit_of(path, top_percent):
    manifest = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    scores = np.load(path / score_file(manifest, "combined"), mmap_mode="r")
    try:
        return float(np.nanpercentile(np.asarray(scores[::THRESHOLD_STRIDE]), 100 - top_percent))
    finally:
        scores._mmap.close()


def flags_at(path, rows, limit):
    """Flags [len(rows), cells] of the score of the run at the rows of its timestamps."""
    manifest = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    scores = np.load(path / score_file(manifest, "combined"), mmap_mode="r")
    try:
        with np.errstate(invalid="ignore"):
            return np.asarray(scores[rows]) > limit  # NaN is not flagged
    finally:
        scores._mmap.close()


def classes(model_flags, other_flags, centre, window, rows, cols, shape, max_distance):
    """Per cell at the timestamp `centre` of the stacks [time, cell]: group (0 model only, 1 neither, 2 inside,
    -1 other), kind (index in KINDS) by what the others do within `window` steps before and after, and
    whether the cell is at distance 1 to `max_distance` from the area the others flag at that timestamp."""
    model, other = model_flags[centre], other_flags[centre]
    before = other_flags[centre - window:centre].any(axis=0)
    after = other_flags[centre + 1:centre + 1 + window].any(axis=0)
    kind = np.where(after & ~before, 0, np.where(before & ~after, 1, np.where(before & after, 2, 3)))
    group = np.full(model.shape, -1)
    group[model & ~other] = 0
    group[~model & ~other] = 1
    group[model & other] = 2
    if other.any():
        lattice = np.ones(shape, dtype=bool)
        lattice[rows[other], cols[other]] = False
        distance = ndimage.distance_transform_cdt(lattice, metric="chessboard")[rows, cols]
    else:
        distance = np.full(model.shape, np.iinfo(np.int32).max)
    near = (distance >= 1) & (distance <= max_distance)
    return group, kind, near


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run with model_epoch_<n>.pt, seasonal normalization")
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the model's score over the test period")
    parser.add_argument("--by-dir", type=Path, nargs="+", required=True, help="Runs with the other scores")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--by-top-percent", type=float, help="Level of the other scores; default --top-percent")
    parser.add_argument("--window-hours", type=int, default=3, help="Hours before and after for front and trail")
    parser.add_argument("--max-distance", type=int, default=1, help="Cells from the area of the others")
    parser.add_argument("--timestamps", type=int, default=400, help="Test timestamps, evenly spaced over the period")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.max_distance < 1:
        raise ValueError("max-distance must be at least 1")
    by_top = args.top_percent if args.by_top_percent is None else args.by_top_percent
    manifest = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    seasonal = manifest.get("backend", {}).get("seasonal_normalization")
    cache_dir = Path(seasonal["mean_file"]).parent if seasonal else None
    model, payload = stgan.load_stgan_checkpoint(args.run_dir / f"model_epoch_{args.epoch}.pt", device=args.device)
    if "seasonal" not in payload["normalization"].get("kind", ""):
        raise ValueError("The seasonal normal is 0 only in a run with the seasonal normalization")
    if hasattr(model, "discriminator_chunk_size"):
        model.discriminator_chunk_size = 10611  # GAT: memory and speed only
    precision = payload.get("precision", "fp32") if torch.device(args.device).type == "cuda" else "fp32"
    scored = pd.DatetimeIndex(np.load(args.score_dir / "test_timestamps.npy"))
    step_hours = int(round((scored[1] - scored[0]) / pd.Timedelta(hours=1)))
    if args.window_hours % step_hours or args.window_hours < step_hours:
        raise ValueError(f"The window must be a multiple of {step_hours} hours")
    window = args.window_hours // step_hours
    cubes = load_prepared(args.prepared_dir)[0]
    try:
        names = list(cubes.feature_names)
        dataset = checkpoints.test_dataset(cubes, payload, cache_dir=cache_dir)
        times = dataset.target_timestamps
        # Only timestamps with the whole window inside the scored period.
        usable = np.nonzero((scored.get_indexer(times) >= window) & (scored.get_indexer(times) < len(scored) - window))[0]
        positions = usable[np.linspace(0, len(usable) - 1, min(args.timestamps, len(usable))).astype(int)]
        previous, observed, forecast, errors, samples, places = forecasts(
            model, dataset, positions, device=args.device, precision=precision)
        target_times = times[samples]
    finally:
        cubes.close()
    grid = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    shape = (len(grid["latitudes"]), len(grid["longitudes"]))
    for path in args.by_dir:
        if not pd.DatetimeIndex(np.load(path / "test_timestamps.npy")).equals(scored):
            raise ValueError(f"The scores are not over the same timestamps: {path}")
    model_limit = limit_of(args.score_dir, args.top_percent)
    other_limits = [limit_of(path, by_top) for path in args.by_dir]
    main_variable = errors.argmax(axis=1)
    point = np.arange(len(main_variable))
    marks = measures(previous[point, main_variable], observed[point, main_variable], forecast[point, main_variable])
    # Sums of the measures and counts [group, kind]; the inside has no kind.
    sums, counts = np.zeros((len(GROUPS), len(KINDS), len(MEASURES))), np.zeros((len(GROUPS), len(KINDS)))
    inside_sum, inside_count = np.zeros(len(MEASURES)), 0
    variables = np.zeros((len(KINDS), len(names)))  # main variables of the model only, by kind
    for timestamp in np.unique(target_times):
        centre = scored.get_loc(timestamp)
        block = np.arange(centre - window, centre + window + 1)
        model_flags = flags_at(args.score_dir, block, model_limit)
        other_flags = np.logical_or.reduce([flags_at(path, block, limit) for path, limit in zip(args.by_dir, other_limits)])
        group, kind, near = classes(model_flags, other_flags, window, window, rows, cols, shape, args.max_distance)
        chosen = np.nonzero(target_times == timestamp)[0]
        cells = places[chosen]
        for g in range(len(GROUPS)):
            for k in range(len(KINDS)):
                picked = chosen[(group[cells] == g) & (kind[cells] == k) & near[cells]]
                sums[g, k] += marks[picked].sum(axis=0)
                counts[g, k] += len(picked)
                if g == 0:
                    variables[k] += np.bincount(main_variable[picked], minlength=len(names))
        picked = chosen[group[cells] == 2]
        inside_sum += marks[picked].sum(axis=0)
        inside_count += len(picked)

    paths = [args.score_dir, *args.by_dir]
    common = len(os.path.commonprefix([path.name for path in paths]))
    short = [path.name[common:] or path.name for path in paths]
    print(f"[tail-surprise] epoch={args.epoch} timestamps={len(positions)} points={len(main_variable)} | flagged: the "
          f"highest {args.top_percent:g}% of {short[0]}, {by_top:g}% of {' or '.join(short[1:])} | distance 1-"
          f"{args.max_distance} cell(s) from the area of the others | window {args.window_hours} h | dropout off")
    print("\n% of the points, on the variable with the largest weighted squared error of each point")
    print(f"{'group, kind':<{NAME}}{'points':>{WIDTH}}" + "".join(f"{name:>{WIDTH}}" for name in MEASURES))
    rows_out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for g, group_name in enumerate(GROUPS):
            for k, kind_name in enumerate(KINDS):
                shares = 100 * sums[g, k] / counts[g, k]
                rows_out[f"{group_name}, {kind_name}"] = (int(counts[g, k]), *shares)
                print(f"{f'{group_name}, {kind_name}':<{NAME}}{int(counts[g, k]):>{WIDTH}d}"
                      + "".join(f"{share:>{WIDTH}.1f}" for share in shares))
        shares = 100 * inside_sum / inside_count
        rows_out["inside"] = (inside_count, *shares)
        print(f"{'inside (model and other)':<{NAME}}{inside_count:>{WIDTH}d}" + "".join(f"{share:>{WIDTH}.1f}" for share in shares))
    print("ending: |obs| < |prev|; beyond: |forecast| > |obs|; persistent: |forecast - prev| < |forecast - obs|; "
          "obs >1, >2: |obs| above 1, 2; fcst only: |obs| <= 1 < |forecast|; units of the seasonal standardisation, "
          "0 = seasonal normal; prev = the observation one step before")
    print("\nmain variables of the model only, % of the kind (the three most frequent)")
    for k, kind_name in enumerate(KINDS):
        total = variables[k].sum()
        top = np.argsort(variables[k])[::-1][:3]
        print(f"{kind_name:<{NAME}}" + ", ".join(f"{names[v]} {100 * variables[k, v] / max(total, 1):.1f}" for v in top))
    return rows_out


if __name__ == "__main__":
    main()
