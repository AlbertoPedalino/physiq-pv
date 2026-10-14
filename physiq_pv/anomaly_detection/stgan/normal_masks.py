"""Values far from normal weather, flagged with no model, once and before any run.

The data are those of the shared cache of a run with the seasonal normalization: z, the departure
of every variable from its seasonal normal in standard deviations of the training years. In the
units of the model the departure of a variable is 2 z / scale (scale: the min-max range of the
training years), as in `scripts/score_era5_seasonal_normal.py`. Two distances per value (cell x
timestamp), each the mean over the variables of the squared departure, with the weights of the
reconstruction error:
- from the seasonal normal: how far the value is from the normal of its place, day and hour;
- from the timestamp before: how fast it changes, which the first one cannot see when the value
  stays near its normal while it changes.
Each distance is standardised by its mean and standard deviation over the training years, by cell,
calendar month and time of day, so that a place or a season whose weather is livelier does not
fill the flags. A value is flagged when either standardised distance is above the level exceeded by
`top_percent` of the training values. Every quantity is fitted on the training years: the
validation and the test only receive the flags.

What is written, for the validation of every run of a sweep:
- the flag of every validation value (`validation_mask.npy`), which a run leaves out of its
  validation statistics;
- the validation inputs with every flagged value replaced by the seasonal normal, zero in the
  standardised data (`validation_inputs.npy`, same rows as the validation file of the cache,
  context included), which a run may read as the history of its validation windows;
- the share of flagged values of every training and validation timestamp, which the PCA and the
  MMD references use to keep the fields of normal weather.
No model, no label and no test data are involved.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .pca_reference import _sha256
from .shared_cache import MANIFEST

FORMAT = 1
CHUNK = 256  # timestamps read at a time
THRESHOLD_STEP = 10  # the levels are read on one training timestamp every this many
SMALL_FILES = ("validation_mask.npy", "validation_share.npy", "train_share.npy", "statistics.npz")
INPUTS = "validation_inputs.npy"
DISTANCES = ("seasonal_normal", "previous_value")
VALIDATION_ANOMALIES = ("kept", "excluded", "excluded_and_replaced")  # what a run does with the flagged values


def slot_of(timestamps, hours):
    """Calendar month x time of day of every timestamp, as one index; `hours` are the times of day in use."""
    timestamps = pd.DatetimeIndex(timestamps)
    position = np.searchsorted(hours, timestamps.hour)
    if (position >= len(hours)).any() or (np.asarray(hours)[np.minimum(position, len(hours) - 1)] != timestamps.hour).any():
        raise ValueError("A time of day that the training years do not have")
    return (timestamps.month.to_numpy() - 1) * len(hours) + position


def distances(now, before, scale, weights):
    """The two squared distances [time, cell] of standardised values `now` [time, cell, variable] in the
    units of the model: from the seasonal normal (zero) and from the values `before`."""
    unit = 2.0 / np.asarray(scale, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    normal = (np.square(now * unit) * weights).mean(axis=-1)
    change = (np.square((now - before) * unit) * weights).mean(axis=-1)
    return np.stack((normal, change))


def _blocks(standardised, first, count):
    """(row offset, now, before) over `count` timestamps from row `first`, each with the row before it."""
    for start in range(0, count, CHUNK):
        stop = min(start + CHUNK, count)
        rows = slice(first + start, first + stop)
        yield start, (np.asarray(standardised[rows], dtype=np.float64),
                      np.asarray(standardised[rows.start - 1:rows.stop - 1], dtype=np.float64))


def fit_statistics(train, slots, n_slots, scale, weights):
    """Mean and standard deviation [distance, slot, cell] of the two distances over the training years.
    The first training timestamp has no timestamp before it and is left out."""
    cells = train.shape[1]
    sums, squares = (np.zeros((2, n_slots, cells)) for _ in range(2))
    counts = np.zeros(n_slots)
    for start, (now, before) in _blocks(train, 1, len(train) - 1):
        values = distances(now, before, scale, weights)
        local = slots[1 + start:1 + start + len(now)]
        for slot in np.unique(local):
            chosen = local == slot
            sums[:, slot] += values[:, chosen].sum(axis=1)
            squares[:, slot] += np.square(values[:, chosen]).sum(axis=1)
            counts[slot] += chosen.sum()
    if (counts < 2).any():
        raise ValueError("Some month and time of day have fewer than two training timestamps")
    mean = sums / counts[None, :, None]
    spread = np.sqrt(np.maximum(squares / counts[None, :, None] - np.square(mean), 0.0))
    # A cell whose distance never changes (none in ERA5) would divide by zero.
    spread = np.maximum(spread, 1e-12 * np.maximum(np.abs(mean), 1.0))
    return mean, spread


def standardised_distances(now, before, slots, mean, spread, scale, weights):
    values = distances(now, before, scale, weights)
    return (values - mean[:, slots]) / spread[:, slots]


def flag_share(standardised, first, count, slots, mean, spread, levels, scale, weights, mask=None):
    """Share of flagged values of each of `count` timestamps from row `first`; their flags in `mask`."""
    share = np.empty(count, dtype=np.float32)
    flagged = np.zeros(2)
    for start, (now, before) in _blocks(standardised, first, count):
        values = standardised_distances(now, before, slots[start:start + len(now)], mean, spread, scale, weights)
        above = values > np.asarray(levels)[:, None, None]
        flagged += above.mean(axis=2).sum(axis=1)
        flags = above.any(axis=0)
        share[start:start + len(now)] = flags.mean(axis=1)
        if mask is not None:
            mask[start:start + len(now)] = flags
    return share, flagged / max(count, 1)


def build_normal_masks(train, validation, train_timestamps, validation_timestamps, *, context, scale, weights,
                       output_dir, top_percent=5.0, cache=None):
    """Fit on `train` [time, cell, variable] (standardised) and flag `validation`, whose first `context` rows
    are the end of the training years. Writes the files of the module docstring in `output_dir`."""
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty directory: masks are built once and never rewritten")
    train_timestamps, validation_timestamps = pd.DatetimeIndex(train_timestamps), pd.DatetimeIndex(validation_timestamps)
    if len(train) != len(train_timestamps) or len(validation) != context + len(validation_timestamps) or context < 1:
        raise ValueError("The data do not match their timestamps (validation: context rows, then the validation)")
    if not 0 < top_percent < 100:
        raise ValueError("top_percent must be between 0 and 100")
    hours = np.unique(train_timestamps.hour)
    n_slots = 12 * len(hours)
    train_slots = slot_of(train_timestamps, hours)
    validation_slots = slot_of(validation_timestamps, hours)
    print(f"[normal-masks] statistics by cell, month and time of day on {len(train)} training timestamps", flush=True)
    mean, spread = fit_statistics(train, train_slots, n_slots, scale, weights)
    rows = np.arange(1, len(train), THRESHOLD_STEP)
    sample = np.concatenate([standardised_distances(
        np.asarray(train[rows[start:start + CHUNK]], dtype=np.float64),
        np.asarray(train[rows[start:start + CHUNK] - 1], dtype=np.float64),
        train_slots[rows[start:start + CHUNK]], mean, spread, scale, weights).astype(np.float32)
        for start in range(0, len(rows), CHUNK)], axis=1)
    levels = np.percentile(sample.reshape(2, -1), 100 - top_percent, axis=1)
    del sample
    print(f"[normal-masks] levels of the highest {top_percent:g}% in the training years: "
          + " ".join(f"{name}={level:.4g}" for name, level in zip(DISTANCES, levels)), flush=True)
    train_share, train_flagged = flag_share(train, 1, len(train) - 1, train_slots[1:], mean, spread, levels, scale, weights)
    train_share = np.concatenate(([np.nan], train_share)).astype(np.float32)
    output.mkdir(parents=True)
    mask = np.zeros((len(validation_timestamps), validation.shape[1]), dtype=np.bool_)
    validation_share, validation_flagged = flag_share(validation, context, len(validation_timestamps), validation_slots,
                                                      mean, spread, levels, scale, weights, mask=mask)
    inputs = np.lib.format.open_memmap(output / INPUTS, mode="w+", dtype=np.float32, shape=validation.shape)
    inputs[:context] = np.asarray(validation[:context], dtype=np.float32)
    for start in range(0, len(validation_timestamps), CHUNK):
        block = np.array(validation[context + start:context + start + CHUNK], dtype=np.float32)
        block[mask[start:start + CHUNK]] = 0.0  # the seasonal normal, every variable of a flagged value
        inputs[context + start:context + start + len(block)] = block
    inputs.flush()
    inputs_bytes = int(inputs.nbytes)
    del inputs
    np.save(output / "validation_mask.npy", mask)
    np.save(output / "validation_share.npy", validation_share)
    np.save(output / "train_share.npy", train_share)
    np.savez(output / "statistics.npz", mean=mean.astype(np.float32), spread=spread.astype(np.float32), hours=hours)
    files = {name: _sha256(output / name) for name in SMALL_FILES}
    metadata = {
        "format": FORMAT, "top_percent": float(top_percent), "distances": list(DISTANCES),
        "levels": {name: float(level) for name, level in zip(DISTANCES, levels)},
        "flag": "either_standardised_distance_above_its_level",
        "standardisation": "mean_and_std_by_cell_calendar_month_and_time_of_day_over_the_training_years",
        "units": "2 z / scale squared, weighted mean over the variables",
        "weights": [float(value) for value in weights], "scale": [float(value) for value in scale],
        "train": {"start": str(train_timestamps[0]), "end": str(train_timestamps[-1]),
                  "timestamps": int(len(train_timestamps)),
                  "flagged_share_by_distance": {name: float(value) for name, value in zip(DISTANCES, train_flagged)},
                  "flagged_share": float(np.nanmean(train_share))},
        "validation": {"start": str(validation_timestamps[0]), "end": str(validation_timestamps[-1]),
                       "timestamps": int(len(validation_timestamps)), "context_rows": int(context),
                       "flagged_share_by_distance": {name: float(value) for name, value in zip(DISTANCES, validation_flagged)},
                       "flagged_share": float(mask.mean())},
        "inputs": {"file": INPUTS, "bytes": inputs_bytes, "replacement": "zero_the_seasonal_normal_every_variable"},
        "cache": cache, "model_used": False, "labels_used": False, "test_data_used": False, "files": files}
    metadata["fingerprint"] = hashlib.sha256(json.dumps(
        {key: metadata[key] for key in ("format", "levels", "files", "cache")}, sort_keys=True).encode("utf-8")).hexdigest()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[normal-masks] flagged share: training {metadata['train']['flagged_share']:.4f}, validation "
          f"{metadata['validation']['flagged_share']:.4f} ({metadata['validation']['flagged_share_by_distance']}) "
          f"fingerprint={metadata['fingerprint']}", flush=True)
    return load_normal_masks(output)


class NormalMasks:
    """What a run and the references read: the metadata, the small arrays and the path of the inputs."""

    def __init__(self, directory, metadata):
        self.directory, self.metadata = Path(directory).resolve(), metadata
        self.validation_mask = np.load(self.directory / "validation_mask.npy")
        self.validation_share = np.load(self.directory / "validation_share.npy")
        self.train_share = np.load(self.directory / "train_share.npy")

    @property
    def fingerprint(self):
        return self.metadata["fingerprint"]

    @property
    def inputs_path(self):
        return self.directory / INPUTS

    def summary(self):
        return {"directory": str(self.directory), "fingerprint": self.fingerprint,
                "top_percent": self.metadata["top_percent"], "levels": self.metadata["levels"],
                "validation_flagged_share": self.metadata["validation"]["flagged_share"],
                "cache": self.metadata.get("cache")}


def load_normal_masks(directory):
    directory = Path(directory)
    path = directory / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"No normal masks in {directory}: build them once with `run_era5_stgan.py normal-masks`.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("format") != FORMAT or set(metadata.get("files", {})) != set(SMALL_FILES):
        raise ValueError(f"Unsupported normal masks format in {directory}")
    for name, expected in metadata["files"].items():
        if _sha256(directory / name) != expected:
            raise ValueError(f"Normal masks file {name} does not match its checksum")
    inputs = directory / INPUTS
    if not inputs.is_file() or inputs.stat().st_size < metadata["inputs"]["bytes"]:
        raise ValueError(f"The validation inputs of the normal masks are missing or incomplete: {inputs}")
    return NormalMasks(directory, metadata)


def normal_rows(timestamps, share, count, *, share_quantile=0.5, seed=0):
    """`count` positions of `timestamps` of normal weather, as many in every calendar month: drawn at random
    (fixed `seed`) among the timestamps of the month whose share of flagged values is at most the
    `share_quantile` of that month. Sorted."""
    timestamps, share = pd.DatetimeIndex(timestamps), np.asarray(share, dtype=np.float64)
    if len(share) != len(timestamps) or not 0 < share_quantile <= 1 or count < 12:
        raise ValueError("Need one share per timestamp, a quantile in (0, 1] and at least one timestamp per month")
    generator = np.random.Generator(np.random.PCG64(seed))
    per_month = np.full(12, count // 12)
    per_month[:count % 12] += 1
    chosen = []
    for month in range(1, 13):
        positions = np.nonzero((timestamps.month == month) & np.isfinite(share))[0]
        if not len(positions):
            raise ValueError(f"No timestamp in month {month}")
        eligible = positions[share[positions] <= np.quantile(share[positions], share_quantile)]
        if len(eligible) < per_month[month - 1]:
            raise ValueError(f"Month {month}: {len(eligible)} normal timestamps, {per_month[month - 1]} needed")
        chosen.append(generator.choice(eligible, per_month[month - 1], replace=False))
    return np.sort(np.concatenate(chosen)).astype(np.int64)


def cache_identity(directory):
    """What names a shared cache: its directory and the inputs it was built from."""
    directory = Path(directory).resolve()
    inputs = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))["inputs"]
    return {"directory": str(directory), "inputs_sha256": hashlib.sha256(
        json.dumps(inputs, sort_keys=True).encode("utf-8")).hexdigest()}
