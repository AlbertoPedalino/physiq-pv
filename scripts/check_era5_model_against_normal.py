"""What does the score of the model flag that the error of the seasonal normal does not?

Two scores over the same period and cells: that of a trained run and that of no model
(`score_era5_seasonal_normal.py`), both read as the checks read them, for instance after the
`climatology` step. A value is flagged by a score when it is in its highest `--top-percent`. The
flagged values are in three groups: flagged by the model alone, by both, by the normal alone. For
every group, and for all the values as a reference, the check says
- how many values it has;
- where its values stand in each score, as the median percentile: a value the other score puts
  just below the threshold is nearly flagged by it too, one in the middle is something else;
- how many are flagged again at the next timestamp, in the same cell, by each score: a value
  flagged alone in time is more likely to be noise than one that lasts;
- how many are south of SPLIT degrees, and in which season;
- what the error of the model and the error of the normal are made of there, by variable, with the
  weights of the run.

The errors by variable are those the scoring saved (scores/feature_scores.npy). If the run of the
normal no longer has that file, its errors are computed again from the shared cache it names.
When the score of the normal was rebalanced (`rebalance_era5_stgan_scores.py --scale mean`),
`--normal-scale-dir` names that run: the error of the normal is then read as that score reads it,
each variable over its mean, and not with the weights of the run.
Everything is read on one timestamp every `--stride`, an odd number so that every time of day is read.

    python scripts/check_era5_model_against_normal.py --model-run outputs/run_clim \
        --normal-run outputs/run_no_model_clim --model-errors-dir outputs/run --normal-errors-dir outputs/run_no_model
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

from physiq_pv.anomaly_detection.stgan.shared_cache import MANIFEST
from scripts.check_era5_dust_lead import COMPONENTS, THRESHOLD_STRIDE, score_file
from scripts.check_era5_score_by_place import SEASONS, SPLIT

CHUNK = 64  # timestamps read at a time
GROUPS = ("model alone", "both", "normal alone", "all values")
NAME, WIDTH = 44, 14


def saved_errors(directory):
    """Reader of the squared error by variable a scoring saved in `directory`, and what to close after it."""
    array = np.load(directory / "scores" / "feature_scores.npy", mmap_mode="r")
    return (lambda read: np.asarray(array[read], dtype=np.float32)), array


def normal_errors(manifest, length):
    """Reader of the error of the seasonal normal computed from the cache `manifest` names, as
    `score_era5_seasonal_normal.py` writes it, and what to close after it."""
    cache = Path(manifest.get("cache_dir", ""))
    if not manifest.get("cache_dir") or not (cache / MANIFEST).is_file() or not (cache / "test.npy").is_file():
        raise ValueError("The run of the normal has no errors by variable and its cache is not there")
    scale = np.asarray(json.loads((cache / MANIFEST).read_text(encoding="utf-8"))["built"]["scale"], dtype=np.float64)
    standardised = np.load(cache / "test.npy", mmap_mode="r")
    first = len(standardised) - length  # the cache holds a few timestamps of context before the scored period
    previous = manifest.get("previous_timestamps") or 0  # the answer was the value that many timestamps before
    if first < previous:
        standardised._mmap.close()
        raise ValueError("The cache and the run are not over the same period")

    def read(rows):
        block = np.asarray(standardised[first + rows.start:first + rows.stop:rows.step], dtype=np.float64)
        if previous:
            block = block - np.asarray(standardised[first + rows.start - previous:first + rows.stop - previous:rows.step],
                                       dtype=np.float64)
        return ((2.0 * block / scale) ** 2).astype(np.float32)
    return read, standardised


def thresholds(model, normal, top_percent):
    """Threshold of each score and its sorted sample, on the values of every THRESHOLD_STRIDE-th timestamp
    that are finite in both."""
    a, b = np.asarray(model[::THRESHOLD_STRIDE]), np.asarray(normal[::THRESHOLD_STRIDE])
    finite = np.isfinite(a) & np.isfinite(b)
    samples = [np.sort(values[finite]) for values in (a, b)]
    if not len(samples[0]):
        raise ValueError("The two scores have no value that is finite in both")
    return [float(np.percentile(sample, 100 - top_percent)) for sample in samples], samples


def compare(model, normal, errors, months, south, top_percent, stride):
    """Counts and sums by group of `GROUPS`, over one timestamp every `stride`. `errors` are the two readers of
    the error by variable, of the model and of the normal."""
    limits, samples = thresholds(model, normal, top_percent)
    groups, variables = len(GROUPS), None
    result = {"count": np.zeros(groups), "south": np.zeros(groups), "season": np.zeros((groups, len(SEASONS))),
              "again": np.zeros((groups, 2)), "with_next": np.zeros(groups), "percentile": [[[], []] for _ in GROUPS]}
    for start in range(0, len(model), CHUNK * stride):
        read = slice(start, min(start + CHUNK * stride, len(model)), stride)
        after = slice(read.start + 1, read.stop + 1, stride)
        scores = [np.asarray(model[read]), np.asarray(normal[read])]
        later = [np.asarray(model[after]), np.asarray(normal[after])]
        finite = np.isfinite(scores[0]) & np.isfinite(scores[1])
        with np.errstate(invalid="ignore"):
            flagged = [finite & (values > limit) for values, limit in zip(scores, limits)]
        masks = (flagged[0] & ~flagged[1], flagged[0] & flagged[1], ~flagged[0] & flagged[1], finite)
        blocks = [reader(read) for reader in errors]
        if variables is None:
            variables = blocks[0].shape[2]
            result["error"] = np.zeros((groups, 2, variables))
        month = np.asarray(months[read])
        for group, mask in enumerate(masks):
            result["count"][group] += mask.sum()
            result["south"][group] += mask[:, south].sum()
            for season, of_season in enumerate(SEASONS.values()):
                result["season"][group, season] += mask[np.isin(month, of_season)].sum()
            for which in range(2):
                result["error"][group, which] += blocks[which][mask].sum(axis=0, dtype=np.float64)
                if group < 3:  # the percentile of all the values is 50 by construction
                    result["percentile"][group][which].append(
                        np.searchsorted(samples[which], scores[which][mask]) / len(samples[which]))
            # The last timestamp of the period has no next one.
            has_next = mask[:len(later[0])]
            result["with_next"][group] += has_next.sum()
            for which in range(2):
                with np.errstate(invalid="ignore"):
                    result["again"][group, which] += (later[which] > limits[which])[has_next].sum()
    result["percentile"] = [[100 * float(np.median(np.concatenate(values))) if values and sum(map(len, values)) else
                             float("nan") for values in group] for group in result["percentile"]]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-run", type=Path, required=True, help="Run with the score of the model")
    parser.add_argument("--normal-run", type=Path, required=True, help="Run with the score of no model, read the same way")
    parser.add_argument("--model-errors-dir", type=Path, required=True, help="Training run, with scores/feature_scores.npy")
    parser.add_argument("--normal-errors-dir", type=Path, required=True,
                        help="Run written by score_era5_seasonal_normal.py; without its errors by variable they are computed again")
    parser.add_argument("--normal-scale-dir", type=Path,
                        help="Run of the normal rebalanced with --scale mean: its error is read over the means saved there")
    parser.add_argument("--score-component", choices=COMPONENTS, default="combined")
    parser.add_argument("--top-percent", type=float, default=5, help="Flagged: this highest share of each score")
    parser.add_argument("--stride", type=int, default=9, help="Timestamps read: one every this many")
    args = parser.parse_args(argv)
    if args.stride < 1 or args.stride % 2 == 0:
        raise ValueError("stride must be odd and at least 1")
    manifests = [json.loads((path / "metadata.json").read_text(encoding="utf-8")) for path in (args.model_run, args.normal_run)]
    grid = manifests[0]["grid"]
    latitudes = np.asarray(grid["latitudes"], dtype=float)[np.asarray(grid["rows"])]
    timestamps = pd.DatetimeIndex(np.load(args.model_run / "test_timestamps.npy"))
    if not pd.DatetimeIndex(np.load(args.normal_run / "test_timestamps.npy")).equals(timestamps):
        raise ValueError("The two scores are not over the same timestamps")
    source = json.loads((args.model_errors_dir / "metadata.json").read_text(encoding="utf-8"))
    of_normal = json.loads((args.normal_errors_dir / "metadata.json").read_text(encoding="utf-8"))
    opened = [np.load(args.model_run / score_file(manifests[0], args.score_component), mmap_mode="r"),
              np.load(args.normal_run / score_file(manifests[1], "combined"), mmap_mode="r")]
    try:
        model, normal = opened
        if model.shape != normal.shape or model.shape != (len(timestamps), len(latitudes)):
            raise ValueError("The two scores are not over the same timestamps and cells")
        readers = []
        for directory, fallback in ((args.model_errors_dir, None), (args.normal_errors_dir, of_normal)):
            if fallback is not None and not (directory / "scores" / "feature_scores.npy").is_file():
                reader, array = normal_errors(fallback, len(timestamps))
            else:
                reader, array = saved_errors(directory)
            opened.append(array)
            readers.append(reader)
        names = list(source.get("preparation", {}).get("features") or [])
        saved = source.get("backend", {}).get("reconstruction_feature_weights")
        result = compare(model, normal, readers, timestamps.month.to_numpy(), np.nonzero(latitudes < SPLIT)[0],
                         args.top_percent, args.stride)
    finally:
        for array in opened:
            array._mmap.close()
    names = names or [f"feature_{index}" for index in range(result["error"].shape[2])]
    weights = np.array([saved[name] for name in names]) if saved else np.ones(len(names))
    how = "with the weights of the run" if saved else "the run has no weights"
    scales = [(weights, how), (weights, how)]
    if args.normal_scale_dir is not None:
        means = json.loads((args.normal_scale_dir / "metadata.json").read_text(encoding="utf-8")).get("feature_means")
        if not means or set(means) != set(names):
            raise ValueError("normal-scale-dir is not a run rebalanced with --scale mean over all the variables")
        scales[1] = (1 / np.array([means[name] for name in names]), "each variable over its mean")
    count = result["count"]

    def row(label, values, form="{:.1f}"):
        print(f"{label:<{NAME}}" + "".join(
            f"{'-' if isinstance(value, float) and np.isnan(value) else form.format(value):>{WIDTH}}" for value in values))

    print(f"[model-against-normal] {timestamps[0].date()} to {timestamps[-1].date()}, one timestamp every {args.stride} | "
          f"flagged: the highest {args.top_percent:g}% of each score")
    row("", GROUPS, "{}")
    row("values", count, "{:.0f}")
    row("% of all the values", 100 * count / count[3])
    row("% of the values flagged by the model", [*(100 * count[:3] / (count[0] + count[1])), float("nan")])
    print("\nmedian percentile of the values of the group in each score")
    row("in the score of the model", [group[0] for group in result["percentile"]])
    row("in the score of the normal", [group[1] for group in result["percentile"]])
    print(f"\n% flagged again at the next timestamp, same cell ({args.top_percent:g} if the time did not matter)")
    with np.errstate(invalid="ignore", divide="ignore"):
        row("by the model", 100 * result["again"][:, 0] / result["with_next"])
        row("by the normal", 100 * result["again"][:, 1] / result["with_next"])
        print("\n% of the values of the group by place and season")
        row(f"south of {SPLIT}N", 100 * result["south"] / count)
        for season, name in enumerate(SEASONS):
            row(name, 100 * result["season"][:, season] / count)
        for which, title in enumerate(("error of the model", "error of the normal")):
            sums = result["error"][:, which] * scales[which][0]
            shares = 100 * sums / sums.sum(axis=1, keepdims=True)
            print(f"\nwhat the {title} is made of, % by variable ({100 / len(names):.1f} each if equal), "
                  f"{scales[which][1]}")
            row("", GROUPS, "{}")
            for index, name in enumerate(names):
                row(name, shares[:, index])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
