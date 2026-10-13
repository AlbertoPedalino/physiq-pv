"""The list of the days must put first the days on which the score flags the place and the others do not."""
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

from scripts import list_era5_score_alone_days as script

TIMES = pd.date_range("2005-01-01", "2005-12-31 21:00", freq="3h")
# Two cells in the place asked about, at 45N, and four cells far from it.
GRID = {"latitudes": [45.0, 30.0], "longitudes": [7.0, 8.0, 20.0], "rows": [0, 0, 0, 1, 1, 1], "cols": [0, 1, 2, 0, 1, 2],
        "spacing": 0.5}
NAMES = ["rain", "pressure", "wind"]
ALONE, BOTH, ELSEWHERE = (TIMES.normalize() == day for day in ("2005-07-10", "2005-03-05", "2005-09-20"))
ZONE = ["--zone", "Place", "44", "46", "6", "9"]
TOP = "1"


class ScoreAloneDaysTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        first, second = (rng.random((len(TIMES), 6)).astype(np.float32) * 0.1 for _ in range(2))
        # One day the first score is high over the place alone, one day with the other, one day far from the place.
        first[np.ix_(ALONE, [0, 1])] += 1
        first[np.ix_(BOTH, [0, 1])] += 1
        second[np.ix_(BOTH, [0, 1])] += 1
        first[np.ix_(ELSEWHERE, [3, 4, 5])] += 1
        self.paths = [self.write_run("run_model", first), self.write_run("run_other", second)]
        errors = np.zeros((len(TIMES), 6, 3), dtype=np.float32)
        errors[..., 0], errors[..., 2] = 3, 1
        self.errors = self.write_run("run", errors=errors)

    def write_run(self, name, scores=None, *, errors=None):
        (self.root / name / "scores").mkdir(parents=True)
        if scores is not None:
            np.save(self.root / name / "scores" / "anomaly_mean.npy", scores)
        if errors is not None:
            np.save(self.root / name / "scores" / "feature_scores.npy", errors)
        np.save(self.root / name / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
        (self.root / name / "metadata.json").write_text(json.dumps(
            {"status": "complete", "grid": GRID, "preparation": {"features": NAMES}}), encoding="utf-8")
        return self.root / name

    def run_script(self, *arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--score-dir", str(self.paths[0]), "--by-dirs", str(self.paths[1]),
                                          "--top-percent", TOP, *map(str, arguments)]), 0)
        return output.getvalue().splitlines()

    def test_the_day_the_score_flags_the_place_alone_comes_first(self):
        lines = self.run_script(*ZONE, "--feature-errors-dir", self.errors, "--days-shown", 3)
        self.assertIn("Place 44-46N 6-9E, 2 cells", lines[0])
        day, alone, of_score, of_other = lines[3].split()[:4]
        self.assertEqual((day, alone, of_score), ("2005-07-10", "100%", "100%"))
        self.assertLess(float(of_other.rstrip("%")), 20)
        # The error behind the score there, by variable.
        self.assertIn("rain 75%, wind 25%", lines[3])
        # The day both scores flag the place is not a day of the first score alone.
        both = next(line for line in self.run_script(*ZONE, "--days-shown", 400) if line.startswith("2005-03-05"))
        self.assertEqual(both.split()[1:4], ["0%", "100%", "100%"])
        # The day the score is high far from the place does not count for the place.
        self.assertFalse(any(line.startswith("2005-09-20") for line in lines))

    def test_months_and_refusals(self):
        lines = self.run_script(*ZONE, "--months", 3, 4)
        self.assertTrue(all(line[5:7] in ("03", "04") for line in lines[3:]))
        with self.assertRaisesRegex(ValueError, "holds no cell"):
            self.run_script("--zone", "Nowhere", "0", "1", "0", "1")


if __name__ == "__main__":
    unittest.main()
