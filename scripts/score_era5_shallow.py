"""Scores of shallow detectors, each fitted on a sample of the training years and written as a run.

Each detector reads one cell at one timestamp: the vector of the variables in the units of the model, with
the weights of the run (2 z / scale times the square root of the weight, z the standardised value), so that
the squared distances are those of the error of the seasonal normal in `score_era5_seasonal_normal.py`. It
sees neither the cells around nor the timestamps before: what the generator adds over them is what it reads
in space and time. The detectors are seven of the statistical ones of TSB-AD (Liu and Paparrizos 2024), with
the defaults of PyOD where PyOD has the detector; a higher score is more unusual:
- pca: PCA with all the components, on the variables standardised first: the sum over the components of the
  distance from each, over the share of the variance it explains;
- kmeans: the distance from the nearest of `--kmeans-clusters` centres;
- cblof: CBLOF with 8 clusters, alpha 0.9, beta 5, unweighted: the distance from the centre of its cluster
  when that cluster is large, else from the nearest centre of a large cluster;
- mcd: the squared Mahalanobis distance with the robust covariance of the minimum covariance determinant;
- hbos: HBOS with 10 bins per variable, alpha 0.1, tol 0.5: minus the sum of the logarithms of the
  histogram densities;
- copod: COPOD, the tail probabilities of every variable with the empirical distributions of the training
  sample (PyOD adds the scored values to that sample, which would make a score depend on the values scored
  with it); past the sample the tail of a variable stays at 1 / (n + 1), and a variable whose skewness is
  near zero is read on the tail its sign gives, as in PyOD;
- iforest: Isolation Forest with 100 trees of 256 values each.

The training sample is `--train-sample` vectors with every variable finite, drawn from `--train-timestamps`
timestamps of the training years at random; every detector is fitted on the same sample. A vector with a
variable that is not finite has no score. Every detector is written as a run in its own directory under
`--output-dir`, so that `climatology`, the thresholds and the checks read it as they read the run. The
scored period is that of `--run-dir`: the test, or the validation years of a run scored over them.

    python scripts/score_era5_shallow.py --run-dir outputs/stgan_good_sweep_2_scored \
        --output-dir outputs/shallow_good2_test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))  # also runs as a copy outside the repository, started from its root
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import skew
from sklearn.cluster import KMeans
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from physiq_pv.anomaly_detection.stgan.loading import release_pages
from physiq_pv.anomaly_detection.stgan.shared_cache import MANIFEST

METHODS = ("pca", "kmeans", "cblof", "mcd", "hbos", "copod", "iforest")
CHUNK = 32  # timestamps scored at a time
CBLOF_CLUSTERS, CBLOF_ALPHA, CBLOF_BETA = 8, 0.9, 5
HBOS_BINS, HBOS_ALPHA, HBOS_TOL = 10, 0.1, 0.5


class PrincipalComponents:
    """PCA as in PyOD (all components, weighted, standardised first)."""

    def __init__(self, seed):
        self.model = PCA(random_state=seed)

    def fit(self, x):
        self.mean, self.std = x.mean(axis=0), x.std(axis=0)
        self.model.fit((x - self.mean) / self.std)
        return self

    def score(self, x):
        return (cdist((x - self.mean) / self.std, self.model.components_) / self.model.explained_variance_ratio_).sum(axis=1)


class NearestCentre:
    def __init__(self, clusters, seed):
        self.model = KMeans(n_clusters=clusters, n_init="auto", random_state=seed)

    def fit(self, x):
        self.model.fit(x)
        return self

    def score(self, x):
        return self.model.transform(x).min(axis=1)


class CBLOF:
    """CBLOF as in PyOD (use_weights=False)."""

    def __init__(self, seed):
        self.model = KMeans(n_clusters=CBLOF_CLUSTERS, n_init="auto", random_state=seed)

    def fit(self, x):
        labels = self.model.fit_predict(x)
        size = np.bincount(labels, minlength=CBLOF_CLUSTERS)
        order = np.argsort(-size)
        by_alpha = [i for i in range(1, CBLOF_CLUSTERS) if size[order[:i]].sum() >= len(x) * CBLOF_ALPHA]
        by_beta = [i for i in range(1, CBLOF_CLUSTERS) if size[order[i]] == 0 or size[order[i - 1]] / size[order[i]] >= CBLOF_BETA]
        both = sorted(set(by_alpha) & set(by_beta))
        split = (both or by_alpha or by_beta or [None])[0]
        if split is None:
            raise ValueError("CBLOF: no split of the clusters into large and small ones")
        self.large = np.zeros(CBLOF_CLUSTERS, dtype=bool)
        self.large[order[:split]] = True
        return self

    def score(self, x):
        distances = self.model.transform(x)
        labels = distances.argmin(axis=1)
        own = distances[np.arange(len(x)), labels]
        return np.where(self.large[labels], own, distances[:, self.large].min(axis=1))


class RobustMahalanobis:
    def __init__(self, seed):
        self.model = MinCovDet(random_state=seed)

    def fit(self, x):
        self.model.fit(x)
        return self

    def score(self, x):
        return self.model.mahalanobis(x)


class HBOS:
    """HBOS as in PyOD (static bins)."""

    def fit(self, x):
        self.hist, self.edges = zip(*(np.histogram(column, bins=HBOS_BINS, density=True) for column in x.T))
        return self

    def score(self, x):
        total = np.zeros(len(x))
        for column, hist, edges in zip(x.T, self.hist, self.edges):
            index = np.digitize(column, edges)
            width = edges[1] - edges[0]
            log = np.log(hist[np.clip(index - 1, 0, HBOS_BINS - 1)] + HBOS_ALPHA)
            below, above = index == 0, index == HBOS_BINS + 1
            far = (below & (edges[0] - column > width * HBOS_TOL)) | (above & (column - edges[-1] > width * HBOS_TOL))
            log = np.where(below & ~far, np.log(hist[0] + HBOS_ALPHA), log)
            log = np.where(above & ~far, np.log(hist[-1] + HBOS_ALPHA), log)
            total += np.where(far, np.log(HBOS_ALPHA), log)
        return -total


class COPOD:
    """COPOD as in PyOD, with the empirical distributions of the training sample alone."""

    def fit(self, x):
        self.sorted = np.sort(x, axis=0)
        self.sign = np.sign(skew(x, axis=0))
        return self

    def score(self, x):
        count = len(self.sorted)
        # Share of the training sample at or below (left) and at or above (right) every value, never 0.
        left = np.stack([np.searchsorted(column, value, side="right") for column, value in zip(self.sorted.T, x.T)], axis=1)
        right = count - np.stack([np.searchsorted(column, value, side="left") for column, value in zip(self.sorted.T, x.T)], axis=1)
        tail_left = -np.log((left + 1) / (count + 1))
        tail_right = -np.log((right + 1) / (count + 1))
        tail_skew = tail_left * -1 * np.sign(self.sign - 1) + tail_right * np.sign(self.sign + 1)
        return np.maximum(tail_skew, (tail_left + tail_right) / 2).sum(axis=1)


class Isolation:
    def __init__(self, seed):
        self.model = IsolationForest(n_estimators=100, max_samples=256, n_jobs=-1, random_state=seed)

    def fit(self, x):
        self.model.fit(x)
        return self

    def score(self, x):
        return -self.model.score_samples(x)


def detector(name, *, seed, kmeans_clusters):
    return {"pca": lambda: PrincipalComponents(seed), "kmeans": lambda: NearestCentre(kmeans_clusters, seed),
            "cblof": lambda: CBLOF(seed), "mcd": lambda: RobustMahalanobis(seed), "hbos": HBOS, "copod": COPOD,
            "iforest": lambda: Isolation(seed)}[name]()


def vectors(block, factor):
    """The vectors [values, variables] of a block [timestamps, cells, variables] of standardised values, in the
    units of the model with the weights of the run."""
    return (np.asarray(block, dtype=np.float64) * factor).reshape(-1, block.shape[-1])


def training_sample(train, factor, size, timestamps, rng):
    """`size` vectors with every variable finite, from `timestamps` training timestamps drawn at random."""
    rows = np.sort(rng.choice(len(train), size=min(timestamps, len(train)), replace=False))
    values = vectors(train[rows], factor)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < size:
        raise ValueError(f"Only {len(values)} finite training vectors in {len(rows)} timestamps: use more timestamps")
    return values[rng.choice(len(values), size=size, replace=False)]


def score(run_dir, output_dir, methods=METHODS, *, train_sample=200_000, train_timestamps=2000, seed=0,
          kmeans_clusters=20):
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
    scale = np.asarray(json.loads((cache / MANIFEST).read_text(encoding="utf-8"))["built"]["scale"], dtype=np.float64)
    names = list(manifest.get("preparation", {}).get("features") or [f"feature_{index}" for index in range(len(scale))])
    saved = manifest.get("backend", {}).get("reconstruction_feature_weights")
    weights = np.array([saved[name] for name in names]) if saved else np.ones(len(names))
    factor = 2.0 / scale * np.sqrt(weights)
    timestamps = np.load(run_dir / "test_timestamps.npy")
    rng = np.random.default_rng(seed)
    train = np.load(cache / "train.npy", mmap_mode="r")
    scored = np.load(cache / f"{period}.npy", mmap_mode="r")
    written = {}
    try:
        first = len(scored) - len(timestamps)  # the cache holds a few timestamps of context before the period
        if first < 0 or scored.shape[2] != len(names) or train.shape[1:] != scored.shape[1:]:
            raise ValueError("The cache and the run are not over the same period, cells and variables")
        sample = training_sample(train, factor, train_sample, train_timestamps, rng)
        release_pages(train)  # the file pages read stay in the page cache, not in this process
        fitted, fit_seconds = {}, {}
        for name in methods:
            start = time.perf_counter()
            fitted[name] = detector(name, seed=seed, kmeans_clusters=kmeans_clusters).fit(sample)
            fit_seconds[name] = time.perf_counter() - start
            print(f"[shallow] {name} fitted on {len(sample)} vectors in {fit_seconds[name]:.1f} s", flush=True)
        cells = scored.shape[1]
        for name in methods:
            (output_dir / name / "scores").mkdir(parents=True, exist_ok=True)
            written[name] = np.lib.format.open_memmap(output_dir / name / "scores" / "anomaly_mean.npy", mode="w+",
                                                      dtype=np.float32, shape=(len(timestamps), cells))
        score_seconds = dict.fromkeys(methods, 0.0)
        report = max(1, len(timestamps) // CHUNK // 10)
        for step, start in enumerate(range(0, len(timestamps), CHUNK)):
            stop = min(start + CHUNK, len(timestamps))
            values = vectors(scored[first + start:first + stop], factor)
            finite = np.isfinite(values).all(axis=1)
            for name in methods:
                began = time.perf_counter()
                result = np.full(len(values), np.nan)
                if finite.any():
                    result[finite] = fitted[name].score(values[finite])
                written[name][start:stop] = result.reshape(stop - start, cells)
                score_seconds[name] += time.perf_counter() - began
            release_pages(scored, *written.values())
            if step % report == 0:
                print(f"[shallow] {stop}/{len(timestamps)} timestamps | " + " ".join(
                    f"{name} {seconds:.0f}s" for name, seconds in score_seconds.items()), flush=True)
    finally:
        for array in written.values():
            array.flush()
        for array in (train, scored, *written.values()):
            array._mmap.close()
    options = {"train_sample": len(sample), "train_timestamps": train_timestamps, "seed": seed,
               "kmeans_clusters": kmeans_clusters}
    for name in methods:
        np.save(output_dir / name / "test_timestamps.npy", timestamps)
        (output_dir / name / "metadata.json").write_text(json.dumps(
            {"status": "complete", "scored_period": period, "grid": manifest["grid"], "score_kind": f"shallow_{name}",
             "shallow": {"method": name, **options, "fit_seconds": fit_seconds[name],
                         "score_seconds": score_seconds[name]},
             "source_run_dir": str(run_dir), "cache_dir": str(cache), "preparation": {"features": names},
             "backend": {"reconstruction_feature_weights": saved}, "anomaly_mean_file": "scores/anomaly_mean.npy",
             "anomaly_std_file": None}, indent=2), encoding="utf-8")
    return {name: output_dir / name for name in methods}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Scored run with the seasonal normalization")
    parser.add_argument("--output-dir", type=Path, required=True, help="One run per detector is written under it")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--train-sample", type=int, default=200_000, help="Training vectors every detector is fitted on")
    parser.add_argument("--train-timestamps", type=int, default=2000, help="Training timestamps the sample is drawn from")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kmeans-clusters", type=int, default=20)
    args = parser.parse_args(argv)
    runs = score(args.run_dir, args.output_dir, args.methods, train_sample=args.train_sample,
                 train_timestamps=args.train_timestamps, seed=args.seed, kmeans_clusters=args.kmeans_clusters)
    print(f"[shallow] runs in {args.output_dir}: {', '.join(runs)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
