"""Can the anomaly of the next hours be told now? A score of where an anomaly forms or arrives.

At a cell not flagged now by the second score at its event level (`--event-percent`, e.g. its highest 1%),
the target is a flag of that score at the cell within `--lead-hours`: a new anomaly there. Two kinds of
cells are read:
- formation: calm cells, with no event flag within QUIET_RADIUS cells in the QUIET_HOURS before; an
  anomaly there forms;
- movement: cells within NEAR_CELLS of an event now; an anomaly there arrives.
What is known at the timestamp of the cell, nothing later, makes its features:
- model, model_near, model_rise: the first score (the model) at the cell, its highest over the 3 x 3
  cells around, its change from the timestamp before; simple, simple_near, simple_rise: the same for the
  second score; agree: the lower of model and simple, high when both are;
- distance: cells to the nearest event now (capped at NEAR_CELLS + 1 for the calm cells);
- downwind: the cell is where the events now go in `--lead-hours` with the wind at 850 hPa (`--wind-dir`);
- along_motion: the cell is where the events now go if they move on as they came in the step before.
Logistic regressions on standardised features are fitted on the cells of `--fit-dir` (runs over the
validation years) and read on those of `--score-dir` (the test): A, one score for both kinds with a
indicator of the kind; B, one score for each kind; each with the features of the model and without them
(model, model_near, model_rise, agree), so that the difference is what the model adds. References: the
simple score alone, the model alone, downwind alone. For every kind: the area under the ROC curve, the
average precision, and the precision and the share of new anomalies caught with the alarms on the
highest `--alarm-percent` of its cells.

The calm and near cells are many: all those followed by an anomaly are kept and the others are kept at
`--negative-share`, each standing for 1 / negative-share cells in every share and metric.

    python scripts/nowcast_era5_anomalies.py --fit-model-dir outputs/val_gen_clim --fit-dir outputs/val_simple_max \
        --score-model-dir outputs/test_gen_clim --score-dir outputs/test_simple_max --wind-dir data/era5/pressure_850
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))  # also runs as a copy outside the repository, started from its root
import numpy as np
import pandas as pd
from scipy import ndimage

from scripts.check_era5_dust_lead import score_file, score_limit
from scripts.check_era5_edge_direction import wind_of_month
from scripts.check_era5_score_against_unseen import read_month, unusual_values

# The referee of what counts as a new anomaly: the simple score at its event level, or weather none of the
# scores reads, the CAPE (which favours a detector of the convective side), the gust (which favours one of the
# wind) or either of the two, unusual at the cell where it was not.
TARGETS = {"simple": None, "cape": (0,), "gust": (1,), "cape_or_gust": (0, 1)}

FEATURES = ("model", "model_near", "model_rise", "simple", "simple_near", "simple_rise", "agree", "distance",
            "downwind", "along_motion")
MODEL_FEATURES = ("model", "model_near", "model_rise", "agree")
KINDS = ("formation", "movement")
QUIET_HOURS, QUIET_RADIUS, NEAR_CELLS, REACH = 24, 5, 2, 4
METERS_PER_DEGREE = 111_000.0
NAME, WIDTH = 34, 11


def open_score(path):
    manifest = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return np.load(path / score_file(manifest, "combined"), mmap_mode="r"), manifest


def shifted(mask, row_shift, col_shift, shape):
    """[shape] of the cells of `mask` moved by the shifts [shape] (rounded), then widened by one cell."""
    rows, cols = np.nonzero(mask)
    to_rows = np.clip(np.rint(rows + row_shift[rows, cols]).astype(int), 0, shape[0] - 1)
    to_cols = np.clip(np.rint(cols + col_shift[rows, cols]).astype(int), 0, shape[1] - 1)
    moved = np.zeros(shape, dtype=bool)
    moved[to_rows, to_cols] = True
    return ndimage.binary_dilation(moved, structure=np.ones((3, 3), dtype=bool))


def motion_shift(event_now, event_before, shape):
    """Row and column shift [shape] of every event cell now: the shift of the centre of its event from the centre of
    the event flags of the step before within REACH cells of it."""
    labels, count = ndimage.label(event_now, structure=np.ones((3, 3), dtype=bool))
    row_shift, col_shift = np.zeros(shape), np.zeros(shape)
    if not count:
        return row_shift, col_shift
    grid_rows, grid_cols = np.indices(shape)
    size = count + 1
    weight = np.bincount(labels.ravel(), minlength=size).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        now_row = np.bincount(labels.ravel(), weights=grid_rows.ravel(), minlength=size) / weight
        now_col = np.bincount(labels.ravel(), weights=grid_cols.ravel(), minlength=size) / weight
        _, (near_rows, near_cols) = ndimage.distance_transform_cdt(~event_now, metric="chessboard", return_indices=True)
        distance = ndimage.distance_transform_cdt(~event_now, metric="chessboard")
        before = event_before & (distance <= REACH)
        owner = labels[near_rows, near_cols][before]
        count_before = np.bincount(owner, minlength=size).astype(float)
        before_row = np.bincount(owner, weights=grid_rows[before], minlength=size) / count_before
        before_col = np.bincount(owner, weights=grid_cols[before], minlength=size) / count_before
    shift_row = np.nan_to_num(now_row - before_row)
    shift_col = np.nan_to_num(now_col - before_col)
    shift_row[0] = shift_col[0] = 0
    return shift_row[labels], shift_col[labels]


def unseen_flags(unseen_dir, timestamps, latitudes, longitudes, percent):
    """[timestamps, cells, 2] where the CAPE and the gust are in their highest `percent` for the cell, the calendar
    month and the time of day over the period, as in check_era5_score_against_unseen.py."""
    result = np.zeros((len(timestamps), len(latitudes), 2), dtype=bool)
    for month in range(1, 13):
        positions = np.nonzero(timestamps.month == month)[0]
        if not len(positions):
            continue
        parts = [read_month(unseen_dir, year, month, latitudes, longitudes) for year in sorted(set(timestamps.year[positions]))]
        names = parts[0][2]
        found = parts[0][0].append([part[0] for part in parts[1:]])
        index = found.get_indexer(timestamps[positions])
        if (index < 0).any():
            raise ValueError(f"The unseen variables do not cover every timestamp of month {month}")
        values = np.concatenate([part[1] for part in parts])[index][:, :, [names.index("cape"), names.index("i10fg")]]
        result[positions] = unusual_values(values, timestamps.hour[positions].to_numpy(), percent)[0]
    return result


def sample_cells(model_path, simple_path, wind_dir, event_percent, lead_hours, stride, negative_share, seed,
                 target_name="simple", unseen_dir=None, unusual_percent=5.0):
    """DataFrame of the sampled cells: features, target, kind and weight."""
    model, model_manifest = open_score(model_path)
    simple, _ = open_score(simple_path)
    try:
        timestamps = pd.DatetimeIndex(np.load(model_path / "test_timestamps.npy"))
        if not pd.DatetimeIndex(np.load(simple_path / "test_timestamps.npy")).equals(timestamps):
            raise ValueError("The two scores are not over the same timestamps")
        event_limit = score_limit(simple, event_percent)
        grid = model_manifest["grid"]
        rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
        shape = (len(grid["latitudes"]), len(grid["longitudes"]))
        latitudes, longitudes = np.asarray(grid["latitudes"], dtype=float), np.asarray(grid["longitudes"], dtype=float)
        step_hours = (timestamps[1] - timestamps[0]) / pd.Timedelta(hours=1)
        quiet, lead = int(QUIET_HOURS / step_hours), int(lead_hours / step_hours)
        seconds = lead_hours * 3600.0
        cells_per_metre_north = 1 / (METERS_PER_DEGREE * abs(latitudes[1] - latitudes[0]))
        cells_per_metre_east = 1 / (METERS_PER_DEGREE * np.cos(np.radians(latitudes)) * abs(longitudes[1] - longitudes[0]))
        rng = np.random.default_rng(seed)

        def lattice(values, fill=0.0):
            frame = np.full(shape, fill, dtype=np.float64 if values.dtype.kind == "f" else values.dtype)
            frame[rows, cols] = values
            return frame

        def near_max(values):
            return ndimage.maximum_filter(lattice(np.nan_to_num(values, nan=-np.inf), fill=-np.inf), size=3)[rows, cols]

        positions = np.arange(quiet, len(timestamps) - lead, stride)
        referee = TARGETS[target_name]
        unseen = None if referee is None else unseen_flags(unseen_dir, timestamps, latitudes[rows], longitudes[cols],
                                                           unusual_percent)[:, :, list(referee)]
        parts = []
        for month in range(1, 13):
            chosen = positions[timestamps.month[positions] == month]
            if not len(chosen):
                continue
            wind = None
            if wind_dir:  # the reader takes every timestamp of the month; the sampled ones are taken from them
                whole = np.nonzero(timestamps.month == month)[0]
                wind = wind_of_month(wind_dir, timestamps, whole, latitudes[rows], longitudes[cols])[
                    np.searchsorted(whole, chosen)]
            for index, t in enumerate(chosen):
                window = np.asarray(simple[t - quiet:t + lead + 1], dtype=np.float64)
                events = window > event_limit  # [quiet + 1 + lead, cells]; row quiet is now
                event_now = events[quiet]
                lattice_now = lattice(event_now, fill=False)
                distance = ndimage.distance_transform_cdt(~lattice_now, metric="chessboard")[rows, cols] if \
                    lattice_now.any() else np.full(len(rows), 99)
                before = events[:quiet].any(axis=0)  # the union of the frames, widened once: the same cells
                around_before = ndimage.binary_dilation(
                    lattice(before, fill=False), structure=np.ones((2 * QUIET_RADIUS + 1,) * 2, dtype=bool)
                )[rows, cols] if before.any() else np.zeros(len(rows), dtype=bool)
                formation = ~event_now & ~around_before & (distance > QUIET_RADIUS)
                movement = ~event_now & (distance >= 1) & (distance <= NEAR_CELLS)
                candidates = formation | movement
                if unseen is None:
                    target = events[quiet + 1:].any(axis=0)
                else:  # unusual weather in the hours after, where it is not unusual now
                    unusual_now = unseen[t].any(axis=1)
                    candidates &= ~unusual_now
                    target = unseen[t + 1:t + lead + 1].any(axis=(0, 2))
                if not candidates.any():
                    continue
                keep = candidates & (target | (rng.random(len(rows)) < negative_share))
                if not keep.any():
                    continue
                model_now, model_before = (np.asarray(model[t - k], dtype=np.float64) for k in (0, 1))
                simple_now, simple_before = window[quiet], window[quiet - 1]
                downwind = np.zeros(len(rows), dtype=bool)
                if wind is not None and lattice_now.any():
                    u, v = lattice(wind[index, :, 0].astype(np.float64)), lattice(wind[index, :, 1].astype(np.float64))
                    downwind = shifted(lattice_now, -v * seconds * cells_per_metre_north,
                                       u * seconds * cells_per_metre_east[:, None], shape)[rows, cols] & ~event_now
                along = np.zeros(len(rows), dtype=bool)
                if lattice_now.any():
                    row_shift, col_shift = motion_shift(lattice_now, lattice(events[quiet - 1], fill=False), shape)
                    along = shifted(lattice_now, row_shift * lead, col_shift * lead, shape)[rows, cols] & ~event_now
                frame = {"model": model_now, "model_near": near_max(model_now), "model_rise": model_now - model_before,
                         "simple": simple_now, "simple_near": near_max(simple_now), "simple_rise": simple_now - simple_before,
                         "agree": np.minimum(model_now, simple_now), "distance": np.minimum(distance, NEAR_CELLS + 1),
                         "downwind": downwind, "along_motion": along}
                part = pd.DataFrame({name: np.asarray(values, dtype=np.float32)[keep] for name, values in frame.items()})
                part["target"] = target[keep]
                part["kind"] = np.where(formation[keep], KINDS[0], KINDS[1])
                part["weight"] = np.where(target[keep], 1.0, 1.0 / negative_share).astype(np.float32)
                parts.append(part)
    finally:
        model._mmap.close()
        simple._mmap.close()
    if not parts:
        raise ValueError("No calm or near cell was read")
    data = pd.concat(parts, ignore_index=True)
    return data[np.isfinite(data[list(FEATURES)]).all(axis=1)].reset_index(drop=True)


def design(data, features, with_kind):
    columns = [data[name].to_numpy(dtype=np.float64) for name in features]
    if with_kind:
        columns.append((data["kind"] == KINDS[1]).to_numpy(dtype=np.float64))
    return np.column_stack(columns)


def fit(data, features, with_kind):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if data["target"].nunique() < 2:  # nothing to learn from: no new anomaly, or nothing else, among these cells
        return None
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(design(data, features, with_kind), data["target"], logisticregression__sample_weight=data["weight"])
    return model


def metrics(target, score, weight, alarm_percent):
    """Area under the ROC curve, average precision, and the precision and the share of positives caught by the
    alarms on the highest `alarm_percent` of the cells (by weight)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    if target.all() or not target.any() or not np.isfinite(score).all():
        return (np.nan,) * 4
    order = np.argsort(-score, kind="stable")
    cumulative = np.cumsum(weight[order])
    alarms = order[cumulative <= alarm_percent / 100 * cumulative[-1]]
    caught = (weight[alarms] * target[alarms]).sum()
    precision = caught / weight[alarms].sum() if len(alarms) else np.nan
    recall = caught / (weight * target).sum()
    return (roc_auc_score(target, score, sample_weight=weight), average_precision_score(target, score, sample_weight=weight),
            100 * precision, 100 * recall)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fit-model-dir", type=Path, required=True, help="The model's score over the validation years")
    parser.add_argument("--fit-dir", type=Path, required=True, help="The simple score over the validation years")
    parser.add_argument("--score-model-dir", type=Path, required=True, help="The model's score over the test")
    parser.add_argument("--score-dir", type=Path, required=True, help="The simple score over the test")
    parser.add_argument("--wind-dir", type=Path, help="Monthly ERA5 files at 850 hPa with u and v")
    parser.add_argument("--target", choices=sorted(TARGETS), default="simple",
                        help="What a new anomaly is: the simple score at its event level, or the CAPE, the gust, either "
                             "of them unusual where it was not (--unseen-dir)")
    parser.add_argument("--unseen-dir", type=Path, help="Monthly files of the CAPE and the gust, for those targets")
    parser.add_argument("--unusual-percent", type=float, default=5, help="Unusual: this highest share of the CAPE or gust")
    parser.add_argument("--event-percent", type=float, default=1)
    parser.add_argument("--lead-hours", type=int, default=3)
    parser.add_argument("--stride", type=int, default=3, help="Timestamps read: one every this many")
    parser.add_argument("--negative-share", type=float, default=0.02)
    parser.add_argument("--alarm-percent", type=float, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, help="Also write the metrics here")
    args = parser.parse_args(argv)
    if args.target != "simple" and args.unseen_dir is None:
        raise ValueError(f"--target {args.target} needs --unseen-dir")
    sample = lambda model_dir, simple_dir, seed: sample_cells(model_dir, simple_dir, args.wind_dir, args.event_percent,
                                                           args.lead_hours, args.stride, args.negative_share, seed,
                                                           args.target, args.unseen_dir, args.unusual_percent)
    fitted_on, read_on = sample(args.fit_model_dir, args.fit_dir, args.seed), sample(args.score_model_dir, args.score_dir,
                                                                                      args.seed + 1)
    without = tuple(name for name in FEATURES if name not in MODEL_FEATURES)
    models = {}
    for label, features in (("with the model", FEATURES), ("without the model", without)):
        models[f"A, {label}"] = (fit(fitted_on, features, True), features, True, None)
        for kind in KINDS:
            subset = fitted_on[fitted_on["kind"] == kind]
            models[f"B, {label}"] = models.get(f"B, {label}", {})
            models[f"B, {label}"][kind] = (fit(subset, features, False), features, False)
    print(f"[nowcast] fitted on {args.fit_model_dir.name} and {args.fit_dir.name}, read on {args.score_model_dir.name} and "
          f"{args.score_dir.name} | new anomaly at the cell within {args.lead_hours} h: "
          + (f"a flag of the simple score at its highest {args.event_percent:g}%" if args.target == "simple" else
             f"{args.target.replace('_', ' ')} in its highest {args.unusual_percent:g}% where it was not")
          + f" | events of the simple score at {args.event_percent:g}% | one timestamp every {args.stride}, other cells "
          f"kept at {args.negative_share:g}")
    print(f"\n{'cells (weighted)':<{NAME}}{'fit':>{WIDTH}}{'% new':>{WIDTH}}{'read':>{WIDTH}}{'% new':>{WIDTH}}")
    for kind in KINDS:
        values = []
        for data in (fitted_on, read_on):
            chosen = data[data["kind"] == kind]
            weight = chosen["weight"].to_numpy()
            values += [weight.sum(), 100 * (weight * chosen["target"]).sum() / weight.sum()]
        print(f"{kind:<{NAME}}{values[0]:>{WIDTH}.3g}{values[1]:>{WIDTH}.3f}{values[2]:>{WIDTH}.3g}{values[3]:>{WIDTH}.3f}")
    rows_out = []
    header = ("ROC AUC", "avg prec", "prec@alarm", "caught %")
    for kind in KINDS:
        chosen = read_on[read_on["kind"] == kind]
        target, weight = chosen["target"].to_numpy(), chosen["weight"].to_numpy(dtype=np.float64)
        print(f"\n{kind}: new anomalies at {100 * (weight * target).sum() / weight.sum():.3f}% of the cells; alarms on the "
              f"highest {args.alarm_percent:g}%")
        print(f"{'score':<{NAME}}" + "".join(f"{name:>{WIDTH}}" for name in header))
        scores = {}
        for label, entry in models.items():
            if label.startswith("A"):
                model, features, with_kind, _ = entry
            else:
                model, features, with_kind = entry[kind]
            scores[label] = (np.full(len(chosen), np.nan) if model is None else
                             model.predict_proba(design(chosen, features, with_kind))[:, 1])
        scores["simple score alone"] = chosen["simple"].to_numpy()
        scores["model alone"] = chosen["model"].to_numpy()
        scores["downwind alone"] = chosen["downwind"].to_numpy() + 1e-6 * chosen["simple"].to_numpy()
        for label, score in scores.items():
            values = metrics(target, np.asarray(score, dtype=np.float64), weight, args.alarm_percent)
            rows_out.append({"kind": kind, "score": label, **dict(zip(header, values))})
            print(f"{label:<{NAME}}" + "".join(f"{value:>{WIDTH}.3f}" for value in values))
    print("\ncoefficients on standardised features (A has the kind as its last term; B one column per kind)")
    print(f"{'feature':<{NAME}}{'A':>{WIDTH}}" + "".join(f"{'B ' + kind[:8]:>{WIDTH}}" for kind in KINDS))
    last_step = lambda pipeline: None if pipeline is None else pipeline[-1]
    a_model = last_step(models["A, with the model"][0])
    b_models = [last_step(models["B, with the model"][kind][0]) for kind in KINDS]
    coefficient = lambda model, index: model.coef_[0][index] if model is not None and index < model.coef_.shape[1] else np.nan
    for index, name in enumerate([*FEATURES, "kind: movement"]):
        values = [coefficient(a_model, index)] + [coefficient(m, index) if index < len(FEATURES) else np.nan
                                                  for m in b_models]
        print(f"{name:<{NAME}}" + "".join(f"{value:>{WIDTH}.3f}" for value in values))
    print("A: one logistic regression for both kinds; B: one for each; with/without the model: with or without model, "
          "model_near, model_rise and agree; prec@alarm: % of the alarms followed by a new anomaly; caught: % of the new "
          "anomalies under an alarm")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_out).to_csv(args.csv, index=False)
        print(f"written {args.csv}")
    return rows_out


if __name__ == "__main__":
    main()
