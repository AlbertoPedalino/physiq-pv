"""Scores of published spatial-outlier tests, which compare every cell with its neighbours, written as runs.

None of them is trained: each is a statistic of the values of a cell and of its neighbours, read on the
same standardised variables the generator reads (the distance from the seasonal normal). The neighbours
of a cell are the cells next to it on the grid, at most eight, fewer at the edge of the domain.

- shekhar_spatial, shekhar_temporal, shekhar_spatiotemporal: the test of Shekhar, Lu and Zhang (KDD 2001).
  For a variable, S(x) is the value at x minus the mean of the values of its neighbours, and x is unusual
  when |(S(x) - mu) / sigma| is large, mu and sigma being the mean and the standard deviation of S. The
  paper defines three neighbourhoods (its Fig. 1), and so three tests: the cells next to x at the same
  time (spatial); the same cell at the timestamps before and after (temporal); the cells next to x and
  the cell itself at the timestamps before and after, and the cells next to x at the same time
  (spatio-temporal). As in the paper, mu and sigma are computed on one data set and used on the data
  that come after: here on `--train-timestamps` timestamps of the training years. The test is for one
  variable: the score of a value is the largest |(S - mu) / sigma| over the variables (our choice).
- chen_multivariate: Algorithm 2 of Chen, Lu, Kou and Chen (GeoInformatica 2008), for several variables
  at once. For every variable, the median of the values of the neighbours at the same time; the vector
  of the differences of the cell from those medians; its squared Mahalanobis distance from the centre of
  the differences of all the cells, with the robust location and covariance of the minimum covariance
  determinant (MCD) of those differences. As in the paper the data set is one spatial snapshot: the MCD
  is estimated anew at every timestamp, over the cells of that timestamp (`--workers` processes share
  the timestamps; about two seconds each).

A value with a variable that is not finite has no score; neighbours that are not finite are left out.
The timestamp after the last of the period is not in the cache: there the temporal tests read the one
before only. Every test is written as a run in its own directory under `--output-dir`, so that
`climatology`, the thresholds and the checks read it as they read the run.

    python scripts/score_era5_spatial_outliers.py --run-dir outputs/stgan_good_sweep_2_scored \
        --output-dir outputs/spatial_good2_test --workers 16
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import time
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path.cwd()))  # after the copy next to the script: a copy run from the checkout imports itself
import numpy as np
from sklearn.covariance import MinCovDet

from physiq_pv.anomaly_detection.stgan.loading import release_pages
from physiq_pv.anomaly_detection.stgan.shared_cache import MANIFEST

AROUND = [(row, column) for row in (-1, 0, 1) for column in (-1, 0, 1) if (row, column) != (0, 0)]
NEIGHBOURS = {  # (timestamp, row, column) offsets of the neighbours of Shekhar et al. (2001), Fig. 1
    "shekhar_spatial": [(0, row, column) for row, column in AROUND],
    "shekhar_temporal": [(-1, 0, 0), (1, 0, 0)],
    "shekhar_spatiotemporal": [(step, row, column) for step in (-1, 0, 1) for row in (-1, 0, 1)
                               for column in (-1, 0, 1) if (step, row, column) != (0, 0, 0)],
}
METHODS = (*NEIGHBOURS, "chen_multivariate")
CHUNK = 32  # timestamps read at a time


def lattice_of(grid):
    """Height and width of the grid and the row and column of every cell."""
    rows, columns = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    return int(rows.max()) + 1, int(columns.max()) + 1, rows, columns


def on_lattice(values, layout):
    """Values [timestamps, cells, variables] on the grid [timestamps, height + 2, width + 2, variables], with a
    border and the places without a cell not finite."""
    height, width, rows, columns = layout
    grid = np.full((values.shape[0], height + 2, width + 2, values.shape[2]), np.nan)
    grid[:, rows + 1, columns + 1] = np.where(np.isfinite(values), values, np.nan)
    return grid


def neighbour_values(grid, offsets, steps, layout):
    """The values of the neighbours [neighbours, steps, cells, variables] of the cells at the timestamps `steps`
    of `grid` (on_lattice); a neighbour off the grid or off the timestamps read is not finite."""
    _, _, rows, columns = layout
    result = np.full((len(offsets), len(steps), len(rows), grid.shape[3]), np.nan)
    for index, (step, row, column) in enumerate(offsets):
        times = np.asarray(steps) + step
        inside = (times >= 0) & (times < len(grid))
        result[index, inside] = grid[times[inside]][:, rows + 1 + row, columns + 1 + column]
    return result


def shekhar_statistic(values, steps, offsets, layout):
    """S(x) = f(x) - mean of f over the neighbours, per variable: [steps, cells, variables]."""
    around = neighbour_values(on_lattice(values, layout), offsets, steps, layout)
    count = np.isfinite(around).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, np.nansum(around, axis=0) / count, np.nan)
    return values[steps] - mean


def chen_differences(values, steps, layout):
    """f(x) - median of f over the neighbours at the same timestamp, per variable: [steps, cells, variables]."""
    around = neighbour_values(on_lattice(values, layout), NEIGHBOURS["shekhar_spatial"], steps, layout)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a cell without finite neighbours gives NaN
        median = np.nanmedian(around, axis=0)
    return values[steps] - median


def robust_distance(differences, seed=0):
    """Squared robust Mahalanobis distance of every row of [cells, variables] from the MCD centre of the finite
    rows; NaN where a row is not finite."""
    finite = np.isfinite(differences).all(axis=1)
    result = np.full(len(differences), np.nan)
    if finite.sum() > differences.shape[1] + 1:
        mcd = MinCovDet(random_state=seed).fit(differences[finite])
        result[finite] = mcd.mahalanobis(differences[finite])
    return result


def _robust_distance(task):
    return robust_distance(*task)


def test_parameters(train, layout, timestamps, rng):
    """mu and sigma of S per variable for every Shekhar test, from `timestamps` training timestamps at random
    (each read with the one before and after): {method: (mu, sigma)}."""
    rows = np.sort(rng.choice(np.arange(1, len(train) - 1), size=min(timestamps, len(train) - 2), replace=False))
    variables = train.shape[2]
    sums = {name: np.zeros((3, variables)) for name in NEIGHBOURS}
    for start in range(0, len(rows), CHUNK):
        for row in rows[start:start + CHUNK]:
            values = np.asarray(train[row - 1:row + 2], dtype=np.float64)
            for name, offsets in NEIGHBOURS.items():
                statistic = shekhar_statistic(values, [1], offsets, layout)[0]
                finite = np.isfinite(statistic)
                sums[name] += [finite.sum(axis=0), np.where(finite, statistic, 0).sum(axis=0),
                               np.where(finite, statistic ** 2, 0).sum(axis=0)]
        release_pages(train)
    parameters = {}
    for name, (count, total, squares) in sums.items():
        mean = total / count
        parameters[name] = (mean, np.sqrt(np.maximum(squares / count - mean ** 2, 0)))
    return parameters


def score(run_dir, output_dir, methods=METHODS, *, train_timestamps=2000, workers=1, seed=0):
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    manifest = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    seasonal = manifest.get("backend", {}).get("seasonal_normalization")
    if not seasonal:
        raise ValueError("The run has no seasonal normalization: there are no standardised data to read")
    cache = Path(seasonal["mean_file"]).parent
    period = manifest.get("scored_period", "test")
    if not all((cache / name).is_file() for name in (MANIFEST, "train.npy", f"{period}.npy")):
        raise ValueError(f"The shared cache of the run is not there: {cache}")
    for name in methods:
        if (output_dir / name).exists() and any((output_dir / name).iterdir()):
            raise ValueError(f"Use a new/empty output directory: {output_dir / name}")
    grid = manifest["grid"]
    layout = lattice_of(grid)
    names = list(manifest.get("preparation", {}).get("features") or [])
    timestamps = np.load(run_dir / "test_timestamps.npy")
    train = np.load(cache / "train.npy", mmap_mode="r")
    scored = np.load(cache / f"{period}.npy", mmap_mode="r")
    written, seconds = {}, dict.fromkeys(methods, 0.0)
    pool = Pool(workers) if "chen_multivariate" in methods and workers > 1 else None
    try:
        first = len(scored) - len(timestamps)  # the cache holds a few timestamps of context before the period
        if first < 1 or train.shape[1:] != scored.shape[1:] or scored.shape[1] != len(layout[2]):
            raise ValueError("The cache and the run are not over the same cells and variables, or there is no "
                             "timestamp before the period")
        shekhar = [name for name in methods if name in NEIGHBOURS]
        parameters = test_parameters(train, layout, train_timestamps, np.random.default_rng(seed)) if shekhar else {}
        for name in methods:
            (output_dir / name / "scores").mkdir(parents=True, exist_ok=True)
            written[name] = np.lib.format.open_memmap(output_dir / name / "scores" / "anomaly_mean.npy", mode="w+",
                                                      dtype=np.float32, shape=(len(timestamps), scored.shape[1]))
        report = max(1, len(timestamps) // CHUNK // 20)
        for step, start in enumerate(range(0, len(timestamps), CHUNK)):
            stop = min(start + CHUNK, len(timestamps))
            # One timestamp before and after the chunk, for the temporal neighbours.
            low, high = first + start - 1, min(first + stop + 1, len(scored))
            values = np.asarray(scored[low:high], dtype=np.float64)
            steps = np.arange(1, 1 + stop - start)
            for name in shekhar:
                began = time.perf_counter()
                mean, deviation = parameters[name]
                with np.errstate(invalid="ignore", divide="ignore"):
                    z = np.abs((shekhar_statistic(values, steps, NEIGHBOURS[name], layout) - mean) / deviation)
                complete = np.isfinite(values[steps]).all(axis=2)
                written[name][start:stop] = np.where(complete, np.nanmax(np.where(np.isfinite(z), z, -np.inf), axis=2),
                                                     np.nan)
                seconds[name] += time.perf_counter() - began
            if "chen_multivariate" in methods:
                began = time.perf_counter()
                tasks = [(differences, seed) for differences in chen_differences(values, steps, layout)]
                distances = pool.map(_robust_distance, tasks) if pool is not None else [robust_distance(*task) for task in tasks]
                written["chen_multivariate"][start:stop] = np.stack(distances)
                seconds["chen_multivariate"] += time.perf_counter() - began
            release_pages(scored, *written.values())
            if step % report == 0:
                print(f"[spatial] {stop}/{len(timestamps)} timestamps | " + " ".join(
                    f"{name} {value:.0f}s" for name, value in seconds.items()), flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        for array in written.values():
            array.flush()
        for array in (train, scored, *written.values()):
            array._mmap.close()
    for name in methods:
        np.save(output_dir / name / "test_timestamps.npy", timestamps)
        details = {"method": name, "seconds": seconds[name], "seed": seed}
        if name in NEIGHBOURS:
            details.update(neighbours=[list(offset) for offset in NEIGHBOURS[name]], train_timestamps=train_timestamps,
                           mu=parameters[name][0].tolist(), sigma=parameters[name][1].tolist())
        (output_dir / name / "metadata.json").write_text(json.dumps(
            {"status": "complete", "scored_period": period, "grid": grid, "score_kind": f"spatial_outlier_{name}",
             "spatial_outlier": details, "source_run_dir": str(run_dir), "cache_dir": str(cache),
             "preparation": {"features": names}, "anomaly_mean_file": "scores/anomaly_mean.npy",
             "anomaly_std_file": None}, indent=2), encoding="utf-8")
    return {name: output_dir / name for name in methods}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Scored run with the seasonal normalization")
    parser.add_argument("--output-dir", type=Path, required=True, help="One run per test is written under it")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--train-timestamps", type=int, default=2000,
                        help="Training timestamps on which mu and sigma of the Shekhar tests are computed")
    parser.add_argument("--workers", type=int, default=1, help="Processes that share the MCD of the timestamps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    runs = score(args.run_dir, args.output_dir, args.methods, train_timestamps=args.train_timestamps,
                 workers=args.workers, seed=args.seed)
    print(f"[spatial] runs in {args.output_dir}: {', '.join(runs)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
