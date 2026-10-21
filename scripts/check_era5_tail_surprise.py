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

With `--components-dir`, a run scored with dropout draws over the same period, the check also gives for
every group and kind the median spread of the draws of G (over its error) and of D at the points, for all of
them, for those back near the normal with an unusual forecast and for those still unusual: whether the draws
disagree more where the model flags the end of an event as a surprise. `--checkpoint` reads the model from
an epoch checkpoint of another directory, as that of the sweep run whose scoring is kept apart.

With the components, `--confidence-out` puts every flag of the model in a class of confidence, 1 to 5, by the
spread of the draws of G among the flags with a similar error of G (the quintiles of the spread within bands of
the error), writes the limits of the classes, and gives the share of the points still unusual and of those
with the anomaly in the forecast only in every class; `--confidence-from` reads limits written before, so
that they are fitted on the validation years and applied to the test.

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
from scripts.check_era5_dust_lead import THRESHOLD_STRIDE, score_file, score_limit

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
        return score_limit(scores, top_percent)
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
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Complete run with the seasonal normalization (its metadata names the cache)")
    model_file = parser.add_mutually_exclusive_group(required=True)
    model_file.add_argument("--epoch", type=int, help="Read model_epoch_<n>.pt of --run-dir")
    model_file.add_argument("--checkpoint", type=Path, help="Or this epoch checkpoint, e.g. of the sweep run scored apart")
    parser.add_argument("--components-dir", type=Path,
                        help="Run scored with dropout draws over the same period: the spread of G and D at the points")
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the model's score over the test period")
    parser.add_argument("--by-dir", type=Path, nargs="+", required=True, help="Runs with the other scores")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--by-top-percent", type=float, help="Level of the other scores; default --top-percent")
    parser.add_argument("--window-hours", type=int, default=3, help="Hours before and after for front and trail")
    parser.add_argument("--max-distance", type=int, default=1, help="Cells from the area of the others")
    parser.add_argument("--timestamps", type=int, default=400, help="Test timestamps, evenly spaced over the period")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence-out", type=Path,
                        help="Write here the limits of the classes of confidence, fitted on these points (validation)")
    parser.add_argument("--confidence-from", type=Path, help="Read the limits of the classes of confidence from here")
    parser.add_argument("--period", choices=("test", "validation"), default="test",
                        help="The years the forecasts are read on: those of --score-dir")
    args = parser.parse_args(argv)
    if args.max_distance < 1:
        raise ValueError("max-distance must be at least 1")
    by_top = args.top_percent if args.by_top_percent is None else args.by_top_percent
    manifest = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    seasonal = manifest.get("backend", {}).get("seasonal_normalization")
    cache_dir = Path(seasonal["mean_file"]).parent if seasonal else None
    checkpoint = args.checkpoint if args.checkpoint is not None else args.run_dir / f"model_epoch_{args.epoch}.pt"
    model, payload = stgan.load_stgan_checkpoint(checkpoint, device=args.device)
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
    train_end = (manifest.get("config") or {}).get("train_end_year")
    cubes = load_prepared(args.prepared_dir, train_end_year=train_end)[0] if train_end else load_prepared(args.prepared_dir)[0]
    try:
        names = list(cubes.feature_names)
        if args.period == "test":
            dataset = checkpoints.test_dataset(cubes, payload, cache_dir=cache_dir)
        else:  # the validation years with their real history, from the shared cache
            from scripts.score_era5_forecast_anomaly import validation_dataset

            dataset = validation_dataset(cubes, payload, cache_dir)
        times = dataset.target_timestamps
        # Only timestamps with the whole window inside the scored period.
        usable = np.nonzero((scored.get_indexer(times) >= window) & (scored.get_indexer(times) < len(scored) - window))[0]
        if not len(usable):
            raise ValueError(f"No {args.period} timestamp falls in the period of --score-dir: give the matching --period")
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
    spreads = components_at(args.components_dir, target_times, places) if args.components_dir is not None else None
    members = {}  # the points of every group and kind, and of the inside, for the medians of the spreads
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
                members.setdefault((g, k), []).append(picked)
                if g == 0:
                    variables[k] += np.bincount(main_variable[picked], minlength=len(names))
        picked = chosen[group[cells] == 2]
        inside_sum += marks[picked].sum(axis=0)
        inside_count += len(picked)
        members.setdefault("inside", []).append(picked)

    paths = [args.score_dir, *args.by_dir]
    common = len(os.path.commonprefix([path.name for path in paths]))
    short = [path.name[common:] or path.name for path in paths]
    print(f"[tail-surprise] model={checkpoint.name} timestamps={len(positions)} points={len(main_variable)} | flagged: the "
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
    if spreads is not None:
        rows_out["spread"] = spread_table(spreads, members, marks)
        if args.confidence_out is not None or args.confidence_from is not None:
            rows_out["confidence"] = confidence_table(spreads, members, marks, args.confidence_out, args.confidence_from)
    return rows_out


SPREAD_COLUMNS = ("G spr/err", "  if fcst only", "  if obs >1", "D spread", "  if fcst only", "  if obs >1")


def components_at(components_dir, target_times, places):
    """At every point (timestamp, cell): the spread of the dropout draws of the reconstruction error over its mean
    (G) and the spread of the discriminator component (D), read in a run scored with dropout draws."""
    times = pd.DatetimeIndex(np.load(components_dir / "test_timestamps.npy"))
    rows = times.get_indexer(target_times)
    if (rows < 0).any():
        raise ValueError("The run of the components does not cover the timestamps read")
    unique, inverse = np.unique(rows, return_inverse=True)
    read = {}
    for name in ("generator_mean", "generator_std", "discriminator_std"):
        array = np.load(components_dir / "scores" / f"{name}.npy", mmap_mode="r")
        try:
            read[name] = np.asarray(array[unique], dtype=np.float64)[inverse, places]
        finally:
            array._mmap.close()
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"G": read["generator_std"] / read["generator_mean"], "D": read["discriminator_std"],
                "G spread": read["generator_std"], "G error": read["generator_mean"]}


def spread_table(spreads, members, marks):
    """Medians of the spreads of G and D for every group and kind, for all its points, for those whose observation
    is back near the normal while the forecast is not (fcst only) and for those still unusual (obs >1)."""
    only, unusual = marks[:, MEASURES.index("fcst only")], marks[:, MEASURES.index("obs >1")]
    print("\nmedians of the dropout spread at the points: G, spread of the reconstruction error over its mean; D, spread "
          "of D(real) - D(generated); for all the points, those back near normal with an unusual forecast (fcst only) "
          "and those still unusual (obs >1)")
    print(f"{'group, kind':<{NAME}}" + "".join(f"{name:>{WIDTH + 3}}" for name in SPREAD_COLUMNS))
    result = {}
    keys = [(g, k) for g in range(len(GROUPS)) for k in range(len(KINDS))] + ["inside"]
    for key in keys:
        points = np.concatenate(members.get(key, [np.array([], dtype=np.int64)]))
        label = key if key == "inside" else f"{GROUPS[key[0]]}, {KINDS[key[1]]}"
        values = []
        for network in ("G", "D"):
            for subset in (np.ones(len(points), dtype=bool), only[points], unusual[points]):
                chosen = spreads[network][points[subset]]
                values.append(float(np.nanmedian(chosen)) if np.isfinite(chosen).any() else np.nan)
        result[label] = values
        print(f"{label:<{NAME}}" + "".join(f"{value:>{WIDTH + 3}.4g}" for value in values))
    # How well the spread tells, among the flags of the model, a forecast unusual on weather back near normal from
    # weather still unusual: the area under the ROC curve, 0.5 for no separation, 1 for a perfect one.
    print("\narea under the ROC curve of the spread, fcst only against obs >1 (0.5: the spread does not tell them apart)")
    print(f"{'flags':<{NAME}}{'fcst only':>{WIDTH + 3}}{'obs >1':>{WIDTH + 3}}{'AUC of G':>{WIDTH + 3}}{'AUC of D':>{WIDTH + 3}}")
    sets = {"model only, all kinds": [(0, k) for k in range(len(KINDS))], "model only, trail": [(0, KINDS.index("trail"))],
            "model only, front": [(0, KINDS.index("front"))], "inside (model and other)": ["inside"]}
    for label, keys in sets.items():
        points = np.concatenate([np.concatenate(members.get(key, [np.array([], dtype=np.int64)])) for key in keys])
        positive, negative = points[only[points]], points[unusual[points]]
        aucs = [roc_auc(spreads[network][positive], spreads[network][negative]) for network in ("G", "D")]
        result[f"AUC, {label}"] = (len(positive), len(negative), *aucs)
        print(f"{label:<{NAME}}{len(positive):>{WIDTH + 3}d}{len(negative):>{WIDTH + 3}d}"
              + "".join(f"{value:>{WIDTH + 3}.3f}" for value in aucs))
    # Does a spread tell them apart beyond the error of G? The error itself, and every spread within bands of equal
    # error (check_era5_components_by_group.auc_at_equal).
    from scripts.check_era5_components_by_group import auc_at_equal

    names = ("G spread", "G", "D")
    print("\nthe same, by the error of G and by each spread at equal error of G (0.5: nothing beyond the error)")
    print(f"{'flags':<{NAME}}{'G error':>{WIDTH + 3}}{'G spr, eq err':>{WIDTH + 3}}{'G s/e, eq err':>{WIDTH + 3}}"
          f"{'D spr, eq err':>{WIDTH + 3}}")
    for label, keys in sets.items():
        points = np.concatenate([np.concatenate(members.get(key, [np.array([], dtype=np.int64)])) for key in keys])
        positive, negative = points[only[points]], points[unusual[points]]
        chosen = np.concatenate((positive, negative))
        hit = np.arange(len(chosen)) < len(positive)
        error = spreads["G error"][chosen]
        finite = np.isfinite(error) & np.logical_and.reduce([np.isfinite(spreads[name][chosen]) for name in names])
        values = [roc_auc(spreads["G error"][positive], spreads["G error"][negative])]
        values += [auc_at_equal(hit[finite], spreads[name][chosen][finite], error[finite])
                   if 0 < hit[finite].sum() < finite.sum() else np.nan for name in names]
        result[f"AUC at equal error, {label}"] = tuple(values)
        print(f"{label:<{NAME}}" + "".join(f"{value:>{WIDTH + 3}.3f}" for value in values))
    return result


CONFIDENCE_BANDS, CONFIDENCE_CLASSES = 20, 5


def fit_confidence(error, spread, bands=CONFIDENCE_BANDS, classes=CONFIDENCE_CLASSES):
    """Limits of the classes of confidence: the quantiles of the error of G that cut it in `bands` bands, and in
    each band the quantiles of the spread of G that cut it in `classes` classes of equal size."""
    error_edges = np.quantile(error, np.linspace(0, 1, bands + 1)[1:-1])
    band = np.searchsorted(error_edges, error, side="right")
    spread_edges = np.stack([np.quantile(spread[band == b], np.linspace(0, 1, classes + 1)[1:-1]) if (band == b).any()
                             else np.full(classes - 1, np.nan) for b in range(bands)])
    return {"error_edges": error_edges, "spread_edges": spread_edges}


def confidence_classes(error, spread, limits):
    """Class of confidence of every point, 1 (the draws of G agree most for its error) to CONFIDENCE_CLASSES (they
    disagree most)."""
    band = np.searchsorted(limits["error_edges"], error, side="right")
    edges = limits["spread_edges"][band]
    return 1 + (spread[:, None] > edges).sum(axis=1)


def confidence_table(spreads, members, marks, out_path=None, from_path=None):
    """For the flags of the model, the share of the points still unusual (obs >1) and of those back near normal with
    an unusual forecast (fcst only) in every class of confidence; the limits are fitted on these points and written
    to `out_path`, or read from `from_path` (fitted on the validation, applied to the test)."""
    only, unusual = marks[:, MEASURES.index("fcst only")], marks[:, MEASURES.index("obs >1")]
    sets = {"model only": [(0, k) for k in range(len(KINDS))], "inside (model and other)": ["inside"]}
    points = {label: np.concatenate([np.concatenate(members.get(key, [np.array([], dtype=np.int64)])) for key in keys])
              for label, keys in sets.items()}
    every = np.concatenate(list(points.values()))
    every = every[np.isfinite(spreads["G error"][every]) & np.isfinite(spreads["G spread"][every])]
    if from_path is not None:
        data = np.load(from_path)
        limits = {"error_edges": data["error_edges"], "spread_edges": data["spread_edges"]}
        origin = f"limits read from {from_path}"
    else:
        limits = fit_confidence(spreads["G error"][every], spreads["G spread"][every])
        origin = "limits fitted on these points"
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **limits)
        origin += f", written to {out_path}"
    print(f"\nclasses of confidence of the flags of the model, by the spread of G at its error (1: the draws agree most; "
          f"{CONFIDENCE_CLASSES}: they disagree most; {origin})")
    print(f"{'flags, class':<{NAME}}{'points':>{WIDTH + 3}}{'obs >1 %':>{WIDTH + 3}}{'fcst only %':>{WIDTH + 3}}")
    result = {}
    for label, chosen in (*points.items(), ("all flags of the model", every)):
        chosen = chosen[np.isfinite(spreads["G error"][chosen]) & np.isfinite(spreads["G spread"][chosen])]
        level = confidence_classes(spreads["G error"][chosen], spreads["G spread"][chosen], limits)
        for c in range(1, CONFIDENCE_CLASSES + 1):
            inside = chosen[level == c]
            row = (len(inside), 100 * unusual[inside].mean() if len(inside) else np.nan,
                   100 * only[inside].mean() if len(inside) else np.nan)
            result[(label, c)] = row
            print(f"{f'{label}, {c}':<{NAME}}{row[0]:>{WIDTH + 3}d}{row[1]:>{WIDTH + 3}.1f}{row[2]:>{WIDTH + 3}.1f}")
    return result


def roc_auc(positive, negative):
    """The probability that a value of `positive` is above one of `negative` (ties count half): the area under the
    ROC curve of the values as a score for positive. NaN values are left out."""
    from scipy.stats import rankdata

    positive, negative = positive[np.isfinite(positive)], negative[np.isfinite(negative)]
    if not len(positive) or not len(negative):
        return np.nan
    ranks = rankdata(np.concatenate((positive, negative)))
    return (ranks[:len(positive)].sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative))


if __name__ == "__main__":
    main()
