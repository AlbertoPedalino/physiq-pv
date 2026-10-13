"""The check against unseen variables must tell a score that flags unusual weather from one that does not."""
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
import xarray as xr

from scripts import check_era5_score_against_unseen as script

TIMES = pd.date_range("2005-01-01", "2006-12-31 21:00", freq="3h")
LATITUDES, LONGITUDES = [45.0, 44.5], [10.0, 10.5, 11.0]
GRID = {"latitudes": LATITUDES, "longitudes": LONGITUDES, "rows": [0, 0, 0, 1, 1, 1], "cols": [0, 1, 2, 0, 1, 2],
        "spacing": 0.5}
# One day a month the gust is strong everywhere and the first score is high; another day the second score is high.
STORMY, OTHER = TIMES.day == 10, TIMES.day == 20
# A day a month of two years is 2 of the 60 values of a cell, month and time of day, and 3.3% of the values.
TOP, UNUSUAL = "3.2", "5"


class ScoreAgainstUnseenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        gust = rng.random((len(TIMES), 6)).astype(np.float32)
        gust[STORMY] += 10
        self.still = rng.random((len(TIMES), 6)).astype(np.float32)  # a variable that follows no score
        self.unseen = self.root / "unseen"
        self.write_files(gust)
        first, second = (rng.random((len(TIMES), 6)).astype(np.float32) * 0.1 for _ in range(2))
        first[STORMY] += 1
        second[OTHER] += 1
        self.paths = [self.write_run("run_model", first), self.write_run("run_other", second)]

    def write_files(self, gust, *, skip=None):
        for year in (2005, 2006):
            for month in range(1, 13):
                if (year, month) == skip:
                    continue
                chosen = (TIMES.year == year) & (TIMES.month == month)
                (self.unseen / str(year)).mkdir(parents=True, exist_ok=True)
                # The files hold the latitudes from the north and one cell more than the score in longitude.
                fields = {name: (("valid_time", "latitude", "longitude"),
                                 np.concatenate((values[chosen].reshape(-1, 2, 3), np.zeros((chosen.sum(), 2, 1), np.float32)),
                                                axis=2))
                          for name, values in (("i10fg", gust), ("cape", self.still))}
                xr.Dataset(fields, coords={"valid_time": TIMES[chosen], "latitude": LATITUDES,
                                           "longitude": [*LONGITUDES, 11.5]}).to_netcdf(
                    self.unseen / str(year) / f"era5_independent_{year}_{month:02d}.nc")

    def write_run(self, name, scores, *, times=TIMES):
        (self.root / name / "scores").mkdir(parents=True)
        np.save(self.root / name / "scores" / "anomaly_mean.npy", scores)
        np.save(self.root / name / "test_timestamps.npy", times.as_unit("ns").asi8)
        (self.root / name / "metadata.json").write_text(json.dumps({"status": "complete", "grid": GRID}), encoding="utf-8")
        return self.root / name

    def run_script(self, *arguments, paths=None):
        paths = paths or self.paths
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--unseen-dir", str(self.unseen), "--score-dir", str(paths[0]), "--by-dirs",
                                          *map(str, paths[1:]), "--top-percent", TOP, "--unusual-percent", UNUSUAL,
                                          *map(str, arguments)]), 0)
        text = output.getvalue()
        tables = [{line[:script.NAME].strip(): [float("nan") if value == "-" else float(value)
                                                for value in line[script.NAME:].split()]
                   for line in part.splitlines()[2:]} for part in text.split("\n\n")[1:]]
        return text, tables

    def test_the_score_that_flags_unusual_weather(self):
        text, (groups, flagged) = self.run_script()
        self.assertIn("score: model | others: other", text)
        self.assertEqual(text.split("\n\n")[1].splitlines()[1].split(), ["values", "cape", "i10fg"])
        # What the first score flags alone is the stormy days: the gust is unusual in all of them.
        alone = groups["score alone, none of the others"]
        self.assertAlmostEqual(alone[0], STORMY.sum() * 6, delta=STORMY.sum() * 6 * 0.1)
        self.assertGreater(alone[2], 85)
        # What the second score flags is not: the gust is unusual there no more than anywhere.
        self.assertLess(groups["others, not the score"][2], 5)
        np.testing.assert_allclose(groups["all values"][1:], [5, 5], atol=1.5)
        # The variable that follows no score is unusual in every group as often as anywhere.
        self.assertLess(alone[1], 10)
        # The two scores hardly ever flag the same value here.
        self.assertLess(groups["score and at least one other"][0], alone[0] * 0.02)
        # Of the values with an unusual gust the first score flags the stormy ones, two in three; the second few.
        self.assertGreater(flagged["model"][1], 45)
        self.assertLess(flagged["other"][1], 8)
        self.assertLess(flagged["model"][0], 8)

    def test_refusals(self):
        shorter = self.write_run("shorter", np.zeros((len(TIMES) - 8, 6), dtype=np.float32), times=TIMES[:-8])
        with self.assertRaisesRegex(ValueError, "same timestamps"):
            self.run_script(paths=[self.paths[0], shorter])
        (self.unseen / "2006" / "era5_independent_2006_03.nc").unlink()
        with self.assertRaisesRegex(ValueError, "No file of the unseen variables for 2006-03"):
            self.run_script()


if __name__ == "__main__":
    unittest.main()
