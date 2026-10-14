"""The combined score must be the highest or the mean of the scores, written as a run the checks read."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import combine_era5_scores as script

TIMES = pd.date_range("2005-01-01", periods=300, freq="3h")
GRID = {"latitudes": [45.0], "longitudes": [7.0, 7.5], "rows": [0, 0], "cols": [0, 1], "spacing": 0.5}


class CombineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        self.first, self.second = (rng.normal(size=(len(TIMES), 2)).astype(np.float32) for _ in range(2))
        self.first[3, 0] = np.nan
        self.first[4, 1] = self.second[4, 1] = np.nan
        self.runs = [self.write_run("a", self.first), self.write_run("b", self.second)]

    def write_run(self, name, scores, times=TIMES):
        run = self.root / name
        (run / "scores").mkdir(parents=True)
        np.save(run / "scores" / "anomaly_mean.npy", scores)
        np.save(run / "test_timestamps.npy", times.as_unit("ns").asi8)
        (run / "metadata.json").write_text(json.dumps({"status": "complete", "grid": GRID}), encoding="utf-8")
        return run

    def test_highest_and_mean(self):
        self.assertEqual(script.main(["--run-dirs", *map(str, self.runs), "--output-dir", str(self.root / "max")]), 0)
        result = np.load(self.root / "max" / "scores" / "anomaly_mean.npy")
        expected = np.fmax(self.first, self.second)
        np.testing.assert_allclose(result, expected, equal_nan=True)
        self.assertEqual(result[3, 0], self.second[3, 0])  # where one is missing the other decides
        self.assertTrue(np.isnan(result[4, 1]))  # where both are missing the result is missing
        manifest = json.loads((self.root / "max" / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual((manifest["status"], manifest["grid"], manifest["score_kind"]), ("complete", GRID, "max_of_scores"))
        script.combine(self.runs, self.root / "mean", how="mean")
        mean = np.load(self.root / "mean" / "scores" / "anomaly_mean.npy")
        np.testing.assert_allclose(mean[10], (self.first[10] + self.second[10]) / 2, rtol=1e-6)

    def test_refusals(self):
        shorter = self.write_run("short", self.first[:-5], TIMES[:-5])
        with self.assertRaisesRegex(ValueError, "same timestamps"):
            script.combine([self.runs[0], shorter], self.root / "x")
        with self.assertRaisesRegex(ValueError, "at least two"):
            script.combine(self.runs[:1], self.root / "y")
        script.combine(self.runs, self.root / "z")
        with self.assertRaisesRegex(ValueError, "new/empty"):
            script.combine(self.runs, self.root / "z")


if __name__ == "__main__":
    unittest.main()
