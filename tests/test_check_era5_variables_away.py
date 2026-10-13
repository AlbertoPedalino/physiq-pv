"""The check must tell a score that flags many variables a little away from one that flags one variable far away."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import check_era5_variables_away as script

TIMES = pd.date_range("2005-01-01", "2005-12-31 21:00", freq="3h")
GRID = {"latitudes": [45.0, 44.5], "longitudes": [10.0, 10.5, 11.0], "rows": [0, 0, 0, 1, 1, 1], "cols": [0, 1, 2, 0, 1, 2],
        "spacing": 0.5}
NAMES = ["rain", "pressure", "wind", "cloud"]
CONTEXT = 4  # timestamps of training history the cache keeps before the scored period
SUMMER, WINTER = np.isin(TIMES.month, (6, 7, 8)), np.isin(TIMES.month, (12, 1, 2))
MANY, ONE = slice(0, 2), slice(4, 6)  # the cells where many variables are a little away, and where one is far
# A third of the cells for a quarter of the year is high in each score.
TOP = "8"


class VariablesAwayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        standardised = rng.normal(scale=0.1, size=(CONTEXT + len(TIMES), 6, 4)).astype(np.float32)
        # In summer every variable is a standard deviation and a half away in some cells; in winter the rain alone
        # is five away in others.
        standardised[CONTEXT:][SUMMER, MANY] = -1.5
        standardised[CONTEXT:][WINTER, ONE, 0] = 5
        self.cache = self.root / "cache"
        self.cache.mkdir()
        np.save(self.cache / "test.npy", standardised)
        (self.cache / "cache.json").write_text(json.dumps({"inputs": {}, "built": {"scale": [1] * 4}}), encoding="utf-8")
        together, alone = (rng.random((len(TIMES), 6)).astype(np.float32) * 0.1 for _ in range(2))
        together[SUMMER, MANY] += 1
        alone[WINTER, ONE] += 1
        self.run = self.write_run("run")
        self.paths = [self.write_run("run_model", together), self.write_run("run_normal", alone)]

    def write_run(self, name, scores=None, *, times=TIMES, seasonal=True):
        (self.root / name / "scores").mkdir(parents=True)
        if scores is not None:
            np.save(self.root / name / "scores" / "anomaly_mean.npy", scores)
        np.save(self.root / name / "test_timestamps.npy", times.as_unit("ns").asi8)
        backend = {"seasonal_normalization": {"mean_file": str(self.cache / "seasonal_mean.npy")}} if seasonal else {}
        (self.root / name / "metadata.json").write_text(json.dumps(
            {"status": "complete", "grid": GRID, "preparation": {"features": NAMES}, "backend": backend}), encoding="utf-8")
        return self.root / name

    def run_script(self, *arguments, run=None, paths=None):
        paths = paths or self.paths
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--run-dir", str(run or self.run), "--score-dir", str(paths[0]), "--by-dirs",
                                          *map(str, paths[1:]), "--top-percent", TOP, "--stride", "3",
                                          *map(str, arguments)]), 0)
        text = output.getvalue()
        return text, [{line[:script.NAME].strip(): [float("nan" if value == "-" else value) for value in line[script.NAME:].split()]
                       for line in part.splitlines()
                       if line[script.NAME:].translate(str.maketrans("", "", ". -")).isdigit()} for part in text.split("\n\n")]

    def test_many_variables_a_little_away_and_one_far_away(self):
        text, (summary, by_variable) = self.run_script()
        self.assertIn("score: model, combined | others: normal", text)
        alone, others = 0, 2  # the columns of the values the first score flags alone, and the other alone
        # Where the first score flags, the four variables are away and none is far.
        self.assertAlmostEqual(summary["variables away, mean of 4"][alone], 4, delta=0.2)
        self.assertAlmostEqual(summary["variables far, mean of 4"][alone], 0, delta=0.05)
        self.assertGreater(summary["% of the values with no variable far"][alone], 97)
        self.assertAlmostEqual(summary["the farthest variable, median distance"][alone], 1.5, delta=0.15)
        # Where the other flags, one variable is far and the rest are at their normal.
        self.assertAlmostEqual(summary["variables away, mean of 4"][others], 1, delta=0.2)
        self.assertAlmostEqual(summary["variables far, mean of 4"][others], 1, delta=0.2)
        self.assertLess(summary["% of the values with no variable far"][others], 10)
        self.assertAlmostEqual(summary["the farthest variable, median distance"][others], 5, delta=0.15)
        # Elsewhere nothing is away.
        self.assertLess(summary["variables away, mean of 4"][3], 0.1)
        self.assertGreater(by_variable["cloud"][alone], 90)
        self.assertLess(by_variable["cloud"][others], 10)
        self.assertGreater(by_variable["rain"][others], 90)

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "odd"):
            self.run_script("--stride", "2")
        with self.assertRaisesRegex(ValueError, "no seasonal normalization"):
            self.run_script(run=self.write_run("minmax", seasonal=False))
        shorter = self.write_run("shorter", np.zeros((len(TIMES) - 8, 6), dtype=np.float32), times=TIMES[:-8])
        with self.assertRaisesRegex(ValueError, "same timestamps"):
            self.run_script(paths=[self.paths[0], shorter])


if __name__ == "__main__":
    unittest.main()
