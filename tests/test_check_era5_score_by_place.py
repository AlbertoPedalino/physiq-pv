"""The check by place must say where the score flags and which variables make the error there."""
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

from scripts import check_era5_score_by_place as script

TIMES = pd.date_range("2005-01-01", "2005-12-31 21:00", freq="3h")
LATITUDES = [55.0, 45.0, 35.0, 25.0]
GRID = {"latitudes": LATITUDES, "longitudes": [10.0, 10.5, 11.0], "rows": np.repeat(np.arange(4), 3).tolist(),
        "cols": np.tile(np.arange(3), 4).tolist(), "spacing": 0.5}
NAMES = ["rain", "pressure", "wind"]
SUMMER, WINTER = np.isin(TIMES.month, (6, 7, 8)), np.isin(TIMES.month, (12, 1, 2))
SOUTH, NORTH = slice(9, 12), slice(0, 3)  # the cells at 25N and at 55N


class ScoreByPlaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        # The score is high at 25N in summer and at 55N in winter, and low everywhere else.
        self.scores = rng.random((len(TIMES), 12)).astype(np.float32) * 0.1
        self.scores[SUMMER, SOUTH] += 1
        self.scores[WINTER, NORTH] += 1
        # In the south the error is all rain, in the north the pressure is three times the wind.
        self.errors = np.zeros((len(TIMES), 12, 3), dtype=np.float32)
        self.errors[:, 6:, 0] = 1
        self.errors[:, :6, 1] = 3
        self.errors[:, :6, 2] = 1

    def write_run(self, name, *, weights=None, errors=True):
        (self.root / name / "scores").mkdir(parents=True)
        np.save(self.root / name / "scores" / "anomaly_mean.npy", self.scores)
        if errors:
            np.save(self.root / name / "scores" / "feature_scores.npy", self.errors)
        np.save(self.root / name / "test_timestamps.npy", TIMES[:len(self.scores)].as_unit("ns").asi8)
        (self.root / name / "metadata.json").write_text(json.dumps(
            {"status": "complete", "grid": GRID, "preparation": {"features": NAMES},
             **({"backend": {"reconstruction_feature_weights": weights}} if weights else {})}), encoding="utf-8")
        return self.root / name

    def run_script(self, *arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main([*map(str, arguments)]), 0)
        return output.getvalue()

    def rows(self, text):
        return {line[:script.NAME].strip(): line[script.NAME:].split() for line in text.splitlines()
                if line[:script.NAME].strip()}

    def test_where_the_score_flags(self):
        run = self.write_run("run")
        # A quarter of the cells for a quarter of the year, twice: an eighth of the values is high.
        text = self.run_script("--run-dir", run, "--top-percent", "12.5")
        table = {line.split()[0]: [float(value) for value in line.split()[2:]] for line in text.splitlines()
                 if line[:1].isdigit()}
        self.assertEqual(list(table), ["50-60N", "40-50N", "30-40N", "20-30N"])
        self.assertEqual([line.split()[1] for line in text.splitlines() if line[:1].isdigit()], ["3"] * 4)
        np.testing.assert_allclose(table["20-30N"], [0] * 5 + [100] * 3 + [0] * 4, atol=1)
        np.testing.assert_allclose(table["50-60N"], [100] * 2 + [0] * 9 + [100], atol=1)
        np.testing.assert_allclose(table["40-50N"] + table["30-40N"], 0, atol=1)
        self.assertNotIn("made of", text)

    def test_what_the_error_of_the_flagged_values_is_made_of(self):
        run = self.write_run("run")
        rows = self.rows(self.run_script("--run-dir", run, "--top-percent", "12.5", "--feature-errors-dir", run,
                                         "--feature-stride", "3"))
        # Columns: south of 37N in the four seasons, then north of it. Flagged: the south in summer, the north in winter.
        counts = [int(value) for value in rows["flagged values read"]]
        self.assertEqual([count > 0 for count in counts], [False, False, True, False, True, False, False, False])
        self.assertAlmostEqual(counts[2], 3 * SUMMER.sum() / 3, delta=3)
        self.assertEqual((rows["rain"][2], rows["pressure"][2], rows["wind"][2]), ("100.0", "0.0", "0.0"))
        self.assertEqual((rows["rain"][4], rows["pressure"][4], rows["wind"][4]), ("0.0", "75.0", "25.0"))
        self.assertEqual(rows["rain"][0], "nan")
        # The weights of the run change what the error is made of: the wind as heavy as the pressure.
        weighted = self.write_run("weighted", weights={"rain": 1.0, "pressure": 0.5, "wind": 1.5})
        text = self.run_script("--run-dir", run, "--top-percent", "12.5", "--feature-errors-dir", weighted)
        self.assertIn("with the weights of the run", text)
        self.assertEqual((self.rows(text)["pressure"][4], self.rows(text)["wind"][4]), ("50.0", "50.0"))

    def test_refusals(self):
        run = self.write_run("run")
        self.scores = self.scores[:-8]
        shorter = self.write_run("shorter", errors=False)
        with self.assertRaisesRegex(ValueError, "not over the same"):
            self.run_script("--run-dir", shorter, "--feature-errors-dir", run)
        with self.assertRaisesRegex(ValueError, "feature-stride"):
            self.run_script("--run-dir", run, "--feature-stride", "0")


if __name__ == "__main__":
    unittest.main()
