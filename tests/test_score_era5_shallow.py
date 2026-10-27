"""The shallow detectors must flag the unusual vectors, each written as a run the other steps read."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import score_era5_shallow as script

TIMES = pd.date_range("2005-01-01", "2005-01-31 21:00", freq="3h")
GRID = {"latitudes": [40.0, 39.5], "longitudes": [10.0, 10.5, 11.0, 11.5], "rows": [0] * 4 + [1] * 4,
        "cols": [0, 1, 2, 3] * 2, "spacing": 0.5}
NAMES = ["temperature", "rain", "wind"]
SCALE = [4.0, 10.0, 8.0]
WEIGHTS = {"temperature": 0.5, "rain": 2.0, "wind": 0.5}
CONTEXT = 4
UNUSUAL = (slice(100, 104), 5)  # timestamps and cell where the variables take an unusual combination


class ShallowScoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        # Rain and wind go together in the training years and in the scored period, except at UNUSUAL where the
        # rain is high and the wind low: each is within its usual range, their combination is not.
        def correlated(length):
            common = rng.normal(size=(length, 8))
            return np.stack((rng.normal(size=(length, 8)), common + 0.2 * rng.normal(size=(length, 8)),
                             common + 0.2 * rng.normal(size=(length, 8))), axis=2).astype(np.float32)
        self.train = correlated(3000)
        self.scored = correlated(CONTEXT + len(TIMES))
        self.scored[CONTEXT + UNUSUAL[0].start:CONTEXT + UNUSUAL[0].stop, UNUSUAL[1], 1:] = [2.0, -2.0]
        self.scored[CONTEXT + 50, 2, 0] = np.nan
        self.cache = self.root / "cache" / "abc"
        self.cache.mkdir(parents=True)
        np.save(self.cache / "train.npy", self.train)
        np.save(self.cache / "test.npy", self.scored)
        (self.cache / "cache.json").write_text(json.dumps({"inputs": {}, "built": {"scale": SCALE}}), encoding="utf-8")
        self.run = self.root / "run"
        self.run.mkdir()
        np.save(self.run / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
        (self.run / "metadata.json").write_text(json.dumps(
            {"status": "complete", "grid": GRID, "preparation": {"features": NAMES},
             "backend": {"reconstruction_feature_weights": WEIGHTS,
                         "seasonal_normalization": {"mean_file": str(self.cache / "seasonal_mean.npy")}}}),
            encoding="utf-8")

    def test_every_detector_flags_the_unusual_combination(self):
        output = self.root / "shallow"
        runs = script.score(self.run, output, train_sample=5000, train_timestamps=1000, kmeans_clusters=5)
        self.assertEqual(list(runs), list(script.METHODS))
        for name, run in runs.items():
            scores = np.load(run / "scores" / "anomaly_mean.npy")
            self.assertEqual(scores.shape, (len(TIMES), 8), name)
            # A vector with a variable that is not finite has no score, the others all have one.
            self.assertTrue(np.isnan(scores[50, 2]), name)
            self.assertEqual(np.isnan(scores).sum(), 1, name)
            # The unusual combination is in the highest 5% of the scores of the detectors that read the variables
            # together; HBOS and COPOD read each variable alone and cannot see it. The PCA of PyOD is left out: its
            # distances from the components depend on their arbitrary signs.
            rank = (scores[UNUSUAL][:, None] > scores[np.isfinite(scores)][None]).mean(axis=1)
            if name in ("hbos", "copod"):
                self.assertLess(rank.mean(), 0.95, name)
            elif name != "pca":
                self.assertTrue((rank > 0.95).all(), (name, rank))
            manifest = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual((manifest["status"], manifest["grid"], manifest["anomaly_mean_file"]),
                             ("complete", GRID, "scores/anomaly_mean.npy"))
            self.assertEqual(manifest["shallow"]["method"], name)
            np.testing.assert_array_equal(np.load(run / "test_timestamps.npy"), TIMES.as_unit("ns").asi8)

    def test_a_variable_far_out_of_its_range_is_flagged(self):
        self.scored[CONTEXT + 10, 1, 0] = 8.0
        np.save(self.cache / "test.npy", self.scored)
        runs = script.score(self.run, self.root / "far", train_sample=5000, train_timestamps=1000, kmeans_clusters=5)
        for name, run in runs.items():
            scores = np.load(run / "scores" / "anomaly_mean.npy")
            # HBOS: see test_hbos_reads_a_value_far_out_as_an_empty_bin. COPOD: past the training sample the tail of a
            # variable stays at 1 / (n + 1), and a symmetric variable may be read on its other tail only.
            # IForest: past the sample every value is isolated about as fast as its most extreme ones, so the value is
            # high but not at the top.
            if name not in ("hbos", "copod"):
                top = 5 if name == "iforest" else 1
                self.assertGreaterEqual(scores[10, 1], np.nanpercentile(scores, 100 - top), name)

    def test_hbos_reads_a_value_far_out_as_an_empty_bin(self):
        x = np.random.default_rng(3).normal(size=(2000, 1))
        model = script.HBOS().fit(x)
        hist, edges = model.hist[0], model.edges[0]
        width = edges[1] - edges[0]
        # Inside the histogram: minus the logarithm of the height of its bin plus alpha.
        self.assertAlmostEqual(model.score(np.array([[0.0]]))[0], -np.log(hist[np.digitize(0.0, edges) - 1] + 0.1))
        # Just past an edge (within tol bins) the edge bin counts; far past it, an empty bin.
        self.assertAlmostEqual(model.score(np.array([[edges[-1] + 0.4 * width]]))[0], -np.log(hist[-1] + 0.1))
        self.assertAlmostEqual(model.score(np.array([[edges[0] - 100.0]]))[0], -np.log(0.1))

    def test_vectors_are_in_the_units_of_the_model_with_the_weights(self):
        block = np.ones((2, 3, 3))
        factor = 2.0 / np.array(SCALE) * np.sqrt(list(WEIGHTS.values()))
        np.testing.assert_allclose(script.vectors(block, factor), np.tile(factor, (6, 1)))

    def test_refuses_an_output_already_written(self):
        script.score(self.run, self.root / "once", ["pca"], train_sample=1000, train_timestamps=500)
        with self.assertRaises(ValueError):
            script.score(self.run, self.root / "once", ["pca"], train_sample=1000, train_timestamps=500)


if __name__ == "__main__":
    unittest.main()
