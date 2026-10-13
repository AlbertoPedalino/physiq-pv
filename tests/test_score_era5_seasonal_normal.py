"""The score of no model must be the error of the seasonal normal, written as a run the other steps read."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import rebalance_era5_stgan_scores as rebalance
from scripts import score_era5_seasonal_normal as script

TIMES = pd.date_range("2005-01-01", "2005-03-31 21:00", freq="3h")
GRID = {"latitudes": [40.0, 39.5], "longitudes": [10.0, 10.5, 11.0], "rows": [0, 0, 0, 1, 1, 1],
        "cols": [0, 1, 2, 0, 1, 2], "spacing": 0.5}
NAMES = ["temperature", "rain", "wind"]
SCALE = [4.0, 10.0, 8.0]
WEIGHTS = {"temperature": 0.5, "rain": 2.0, "wind": 0.5}
CONTEXT = 4  # timestamps of training history the cache keeps before the scored period


class SeasonalNormalScoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        self.standardised = rng.normal(size=(CONTEXT + len(TIMES), 6, 3)).astype(np.float32)
        self.cache = self.root / "cache" / "abc"
        self.cache.mkdir(parents=True)
        np.save(self.cache / "test.npy", self.standardised)
        (self.cache / "cache.json").write_text(json.dumps({"inputs": {}, "built": {"minimum": [-2.0, -5.0, -4.0],
                                                                                    "scale": SCALE}}), encoding="utf-8")

    def write_run(self, name, *, status="complete", seasonal=True, weights=WEIGHTS):
        run = self.root / name
        run.mkdir()
        np.save(run / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
        backend = {"reconstruction_feature_weights": weights}
        if seasonal:
            backend["seasonal_normalization"] = {"mean_file": str(self.cache / "seasonal_mean.npy")}
        (run / "metadata.json").write_text(json.dumps({"status": status, "grid": GRID, "preparation": {"features": NAMES},
                                                       "backend": backend}), encoding="utf-8")
        return run

    def test_the_error_of_the_seasonal_normal_in_the_units_of_the_model(self):
        run = self.write_run("run")
        self.assertEqual(script.main(["--run-dir", str(run), "--output-dir", str(self.root / "none")]), 0)
        errors = np.load(self.root / "none" / "scores" / "feature_scores.npy")
        score = np.load(self.root / "none" / "scores" / "anomaly_mean.npy")
        # The context before the scored period is left out; the error of the normal is 2 z / scale.
        expected = (2.0 * self.standardised[CONTEXT:].astype(np.float64) / np.array(SCALE)) ** 2
        np.testing.assert_allclose(errors, expected, rtol=1e-5)
        np.testing.assert_allclose(score, (expected * np.array(list(WEIGHTS.values()))).mean(axis=2), rtol=1e-5)
        manifest = json.loads((self.root / "none" / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual((manifest["status"], manifest["grid"], manifest["anomaly_std_file"]), ("complete", GRID, None))
        self.assertEqual(manifest["backend"]["reconstruction_feature_weights"], WEIGHTS)
        np.testing.assert_array_equal(np.load(self.root / "none" / "test_timestamps.npy"), TIMES.as_unit("ns").asi8)
        # The rebalancing reads it as a run, with its weights and with some of its variables.
        as_run = rebalance.rebalance(self.root / "none", self.root / "weighted", scale="weights")
        np.testing.assert_allclose(np.load(self.root / "weighted" / "scores" / "anomaly_mean.npy"), score, rtol=1e-4)
        self.assertEqual(as_run["features"], NAMES)
        one = rebalance.rebalance(self.root / "none", self.root / "one", scale="rarity", features=["temperature"],
                                  scale_stride=1)
        self.assertEqual(one["features"], ["temperature"])
        # Without weights in the run every variable counts the same.
        plain = self.write_run("plain", weights=None)
        script.score(plain, self.root / "plain_none")
        np.testing.assert_allclose(np.load(self.root / "plain_none" / "scores" / "anomaly_mean.npy"), expected.mean(axis=2),
                                   rtol=1e-5)

    def test_refusals(self):
        run = self.write_run("run")
        for name, message, options in ((self.write_run("running", status="running"), "must be complete", {}),
                                       (self.write_run("minmax", seasonal=False), "no seasonal normalization", {}),
                                       (run, "cache of the run is not there", {"cache_dir": self.root / "nowhere"})):
            with self.assertRaisesRegex(ValueError, message):
                script.score(name, self.root / ("out_" + message[:6].replace(" ", "_")), **options)
        script.score(run, self.root / "done")
        with self.assertRaisesRegex(ValueError, "new/empty"):
            script.score(run, self.root / "done")
        # A cache shorter than the scored period is not the cache of this run.
        np.save(self.cache / "test.npy", self.standardised[:10])
        with self.assertRaisesRegex(ValueError, "not over the same"):
            script.score(run, self.root / "short")
        self.assertFalse((self.root / "short").exists())


if __name__ == "__main__":
    unittest.main()
