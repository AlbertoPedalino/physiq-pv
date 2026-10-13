"""Recompute the reconstruction error of a finished ERA5 STGAN run with every variable weighing the same.

The reconstruction error of the score is the mean of the squared errors of the variables after a
min-max of the inputs. A variable that moves across its whole range, as the cloud cover does, then
makes most of that mean and the others are not heard. The scoring saves the squared error of every
cell by variable (scores/feature_scores.npy): this script puts the errors of the variables on a
common scale and averages them, with no new scoring and no change to the model.

- `--scale mean`: the error of each variable is divided by its mean, so every variable gives the
  same share of the total. A variable with rare and very large errors still makes most of the
  highest scores;
- `--scale rarity`: the error of each variable is replaced by how rare it is for that variable,
  -log of the share of its errors that are larger. Every variable then has the same distribution,
  in the bulk and in the highest values;
- `--scale none`: the plain mean of the variables, to compare;
- `--scale weights`: the error of each variable times the weight the run gave it in its loss and in
  its score (`--reconstruction-weights`). The table then says how much of the error of that run
  each variable makes, which its saved errors by variable, kept without weights, do not say.

The score of a cell is the mean of its variables on that scale. With `--combine max` it is the
highest of them instead: one very unusual variable is then enough, where the mean needs several.
With `--combine top` it is the mean of the `--top-k` highest of them, between the two: an anomaly
that only a few variables carry is not diluted by the others, and no single variable decides alone.

The scales are read on one timestamp every `--scale-stride` of the scored period itself, as the
min-max of the score is. `--features` keeps only some variables. The result is a run that
`climatology`, `events` and the other checks accept. Nothing in the source run is changed.

    python scripts/rebalance_era5_stgan_scores.py \
        --run-dir outputs/run --output-dir outputs/run_balanced
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CHUNK = 256  # timestamps read at a time
SCALES = ("mean", "rarity", "none", "weights")
COMBINES = ("mean", "max", "top")
RAREST = 1e-6  # the smallest share of larger errors that is told apart
# Where the distribution of the errors of a variable is read: evenly up to 99%, then ever closer to 100%.
PROBABILITIES = np.concatenate((np.linspace(0, 0.99, 991)[:-1], 1 - np.logspace(-2, np.log10(RAREST), 400)))
DIAGNOSTIC_STRIDE = 4  # the shares are measured on one point every this many of the sample


def fit(sample, scale):
    """What `transform` needs, from a [points, variables] sample of the squared errors."""
    if scale == "none":
        return None
    if scale == "mean":
        means = np.mean(sample, axis=0, dtype=np.float64)
        if not (np.isfinite(means).all() and (means > 0).all()):
            raise ValueError("A variable has no positive error to scale by")
        return means
    tables = []
    for index in range(sample.shape[1]):
        quantiles = np.quantile(sample[:, index], PROBABILITIES)
        # Equal values (many errors can be zero) take the highest of their probabilities.
        values, first = np.unique(quantiles[::-1], return_index=True)
        tables.append((values, PROBABILITIES[len(quantiles) - 1 - first]))
    return tables


def transform(block, scale, fitted):
    """The errors [..., variables] on the common scale, in the same shape."""
    if scale == "none":
        return np.asarray(block, dtype=np.float32)
    if scale == "mean":
        return (block / fitted).astype(np.float32)
    if scale == "weights":
        return (block * fitted).astype(np.float32)
    result = np.empty(block.shape, dtype=np.float32)
    for index, (values, probabilities) in enumerate(fitted):
        result[..., index] = -np.log1p(-np.interp(block[..., index], values, probabilities))
    return result


def combined(rescaled, combine, top_k=3):
    """The score of the errors [..., variables] on the common scale: their mean, the highest of them, or the
    mean of the `top_k` highest."""
    if combine == "mean":
        return rescaled.mean(axis=-1, dtype=np.float64)
    if combine == "max":
        return rescaled.max(axis=-1)
    return np.partition(rescaled, -top_k, axis=-1)[..., -top_k:].mean(axis=-1, dtype=np.float64)


def shares(sample, rescaled, combine="mean", top_k=3):
    """Per variable: its share of the plain sum, of the rescaled sum, and of the rescaled sum over the
    points with the highest 1% of the new score."""
    score = combined(rescaled, combine, top_k)
    top = score >= np.percentile(score, 99)
    plain, balanced, highest = (values.sum(axis=0, dtype=np.float64) for values in (sample, rescaled, rescaled[top]))
    return plain / plain.sum(), balanced / balanced.sum(), highest / highest.sum()


def rebalance(run_dir, output_dir, *, scale="mean", features=None, scale_stride=10, combine="mean", top_k=3):
    """Write the rebalanced reconstruction error of `run_dir` as a complete run in `output_dir`; return its manifest."""
    run_dir, output = Path(run_dir), Path(output_dir)
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}")
    if combine not in COMBINES:
        raise ValueError(f"combine must be one of {COMBINES}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty output directory")
    manifest = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("The run must be complete")
    source = run_dir / "scores" / "feature_scores.npy"
    if not source.is_file():
        raise ValueError(f"The run has no errors by variable: {source}")
    errors = np.load(source, mmap_mode="r")
    try:
        names = list(manifest.get("preparation", {}).get("features") or
                     [f"feature_{index}" for index in range(errors.shape[2])])
        if len(names) != errors.shape[2]:
            raise ValueError("The variables of the run do not match its errors by variable")
        unknown = [name for name in features or [] if name not in names]
        if unknown:
            raise ValueError(f"Unknown variables: {unknown}; the run has {names}")
        columns = [names.index(name) for name in features] if features else list(range(len(names)))
        kept = [names[index] for index in columns]
        if combine == "top" and not 1 <= top_k <= len(kept):
            raise ValueError(f"top_k must be between 1 and the number of variables ({len(kept)})")
        weights = None
        if scale == "weights":
            saved = manifest.get("backend", {}).get("reconstruction_feature_weights")
            if not saved:
                raise ValueError("The run gave no weights to its variables")
            weights = np.array([saved[name] for name in kept], dtype=np.float64)
        sample = np.asarray(errors[::scale_stride][:, :, columns], dtype=np.float32).reshape(-1, len(columns))
        sample = sample[np.isfinite(sample).all(axis=1)]
        fitted = weights if scale == "weights" else fit(sample, scale)
        sample = sample[::DIAGNOSTIC_STRIDE]
        before, after, highest = shares(sample, transform(sample, scale, fitted), combine, top_k)
        del sample
        (output / "scores").mkdir(parents=True)
        score = np.lib.format.open_memmap(output / "scores" / "anomaly_mean.npy", mode="w+", dtype=np.float32,
                                          shape=errors.shape[:2])
        for start in range(0, len(score), CHUNK):
            block = np.asarray(errors[start:start + CHUNK][:, :, columns], dtype=np.float32)
            score[start:start + CHUNK] = combined(transform(block, scale, fitted), combine, top_k)
        score.flush()
        score._mmap.close()
    finally:
        errors._mmap.close()
    np.save(output / "test_timestamps.npy", np.load(run_dir / "test_timestamps.npy"))
    result = {"status": "complete", "grid": manifest["grid"], "score_kind": "rebalanced_reconstruction_error",
              "source_run_dir": str(run_dir), "source_score_file": "scores/feature_scores.npy", "scale": scale,
              "scale_stride": scale_stride, "combine": combine, "top_k": top_k if combine == "top" else None,
              "features": kept,
              "feature_means": dict(zip(kept, map(float, fitted))) if scale == "mean" else None,
              "feature_weights": dict(zip(kept, map(float, fitted))) if scale == "weights" else None,
              "share_of_the_plain_error": dict(zip(kept, map(float, before))),
              "share_of_the_new_score": dict(zip(kept, map(float, after))),
              "share_of_the_new_score_in_its_top_1_percent": dict(zip(kept, map(float, highest))),
              "anomaly_mean_file": "scores/anomaly_mean.npy", "anomaly_std_file": None}
    (output / "metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parser():
    root = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    root.add_argument("--run-dir", type=Path, required=True, help="Complete run with scores/feature_scores.npy")
    root.add_argument("--output-dir", type=Path, required=True)
    root.add_argument("--scale", choices=SCALES, default="mean",
                      help="The common scale: the error over its mean, how rare the error is for its variable, none, "
                           "or the weights of the run")
    root.add_argument("--combine", choices=COMBINES, default="mean",
                      help="The score of a cell: the mean of its variables, the most unusual of them, or the mean "
                           "of the --top-k most unusual")
    root.add_argument("--top-k", type=int, default=3, help="With --combine top: how many of the most unusual variables")
    root.add_argument("--features", nargs="+", help="Only these variables; default: all of them")
    root.add_argument("--scale-stride", type=int, default=10, help="The scales are read on one timestamp every this many")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    result = rebalance(args.run_dir, args.output_dir, scale=args.scale, features=args.features,
                       scale_stride=args.scale_stride, combine=args.combine, top_k=args.top_k)
    print(f"[rebalance] scale={args.scale} combine={args.combine} variables={len(result['features'])} "
          f"output={args.output_dir}")
    print(f"{'variable':<36}{'share before':>14}{'share after':>13}{'after, top 1%':>15}")
    for name in sorted(result["features"], key=lambda name: -result["share_of_the_plain_error"][name]):
        print(f"{name:<36}{100 * result['share_of_the_plain_error'][name]:>13.1f}%"
              f"{100 * result['share_of_the_new_score'][name]:>12.1f}%"
              f"{100 * result['share_of_the_new_score_in_its_top_1_percent'][name]:>14.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
