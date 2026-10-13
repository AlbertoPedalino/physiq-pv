"""The check against the change of the data must tell a score that follows the fast changes from one that does not."""
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

from scripts import check_era5_score_against_change as script

TIMES = pd.date_range("2005-01-01", "2005-12-31 21:00", freq="3h")
GRID = {"latitudes": [55.0, 45.0, 35.0, 25.0], "longitudes": [10.0, 10.5, 11.0],
        "rows": np.repeat(np.arange(4), 3).tolist(), "cols": np.tile(np.arange(3), 4).tolist(), "spacing": 0.5}
NAMES = ["rain", "pressure", "wind"]
CONTEXT = 4  # timestamps of training history the cache keeps before the scored period
SUMMER, WINTER = np.isin(TIMES.month, (6, 7, 8)), np.isin(TIMES.month, (12, 1, 2))
FAST, FAR = slice(9, 12), slice(0, 3)  # the cells where the weather changes fast, and where it is far and steady
# A quarter of the cells for a quarter of the year is high in each score.
TOP = "6.25"


class ScoreAgainstChangeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        standardised = rng.normal(scale=0.01, size=(CONTEXT + len(TIMES), 12, 3)).astype(np.float32)
        # In summer the rain and the pressure jump at every timestamp in some cells; in winter they are far
        # from the normal, and still, in others.
        standardised[CONTEXT:][SUMMER, FAST, :2] = rng.normal(size=(SUMMER.sum(), 3, 2))
        standardised[CONTEXT:][WINTER, FAR, :2] = 3
        # The wind changes a hundred times as much as the rest everywhere: it must not hide the others.
        standardised[..., 2] = rng.normal(scale=100, size=standardised.shape[:2])
        self.cache = self.root / "cache"
        self.cache.mkdir()
        np.save(self.cache / "test.npy", standardised)
        (self.cache / "cache.json").write_text(json.dumps({"inputs": {}, "built": {"scale": [1, 1, 1]}}), encoding="utf-8")
        self.run_dir = self.write_run("run")
        quick, far = (rng.random((len(TIMES), 12)).astype(np.float32) * 0.1 for _ in range(2))
        quick[SUMMER, FAST] += 1
        far[WINTER, FAR] += 1
        self.scores = [self.write_run("run_quick", scores=quick), self.write_run("run_far", scores=far)]

    def write_run(self, name, *, scores=None, times=TIMES, seasonal=True):
        (self.root / name / "scores").mkdir(parents=True)
        if scores is not None:
            np.save(self.root / name / "scores" / "anomaly_mean.npy", scores)
        np.save(self.root / name / "test_timestamps.npy", times.as_unit("ns").asi8)
        backend = {"seasonal_normalization": {"mean_file": str(self.cache / "seasonal_mean.npy")}} if seasonal else {}
        (self.root / name / "metadata.json").write_text(json.dumps(
            {"status": "complete", "grid": GRID, "preparation": {"features": NAMES}, "backend": backend}), encoding="utf-8")
        return self.root / name

    def run_script(self, *arguments, run_dir=None, scores=None):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--run-dir", str(run_dir or self.run_dir), "--score-dirs",
                                          *map(str, scores or self.scores), "--top-percent", TOP, "--stride", "3",
                                          *map(str, arguments)]), 0)
        text = output.getvalue()
        # One table for every lag: the band of the change, then the share flagged by each score.
        tables = [{line.split(",")[0].split()[0]: [float(value) for value in line[script.NAME:].split()[1:]]
                   for line in part.splitlines() if line[:1].isdigit()} for part in text.split("\n\n")[1:]]
        return text, tables

    def test_the_score_that_follows_the_fast_changes(self):
        text, (three_hours, a_day) = self.run_script()
        self.assertIn("change of the data in 3 hours", text)
        self.assertIn("change of the data in 24 hours", text)
        self.assertIn("quick", text.splitlines()[3])
        self.assertIn("far", text.splitlines()[3])
        for table in (three_hours, a_day):
            # Where the data change fastest the first score flags nearly everything and the second nearly
            # nothing: only the first and last timestamps of the far and steady season are a change.
            for band in ("99-100", "95-99"):
                self.assertGreater(table[band][0], 75)
                self.assertLess(table[band][1], 15)
            # Where they change least the first flags nothing; the far and steady values are there.
            self.assertLess(table["0-50"][0], 1)
            self.assertGreater(table["0-50"][1], 5)

    def test_one_lag_and_one_score(self):
        text, tables = self.run_script("--lags", "2", scores=self.scores[:1])
        self.assertEqual(len(tables), 1)
        self.assertIn("in 6 hours", text)
        self.assertIn("run_quick", text.splitlines()[3])

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "odd"):
            self.run_script("--stride", "2")
        with self.assertRaisesRegex(ValueError, "no seasonal normalization"):
            self.run_script(run_dir=self.write_run("minmax", seasonal=False))
        shorter = self.write_run("shorter", scores=np.zeros((len(TIMES) - 8, 12), dtype=np.float32), times=TIMES[:-8])
        with self.assertRaisesRegex(ValueError, "not over the timestamps"):
            self.run_script(scores=[shorter])
        (self.cache / "test.npy").unlink()
        with self.assertRaisesRegex(ValueError, "cache of the run is not there"):
            self.run_script()


if __name__ == "__main__":
    unittest.main()
