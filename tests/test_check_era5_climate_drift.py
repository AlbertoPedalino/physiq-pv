"""The drift check must be 0 over the training years and follow a trend into the scored ones."""
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

from physiq_pv.era5.data import FEATURE_NAMES
from scripts import check_era5_climate_drift as script

ROWS, COLS = np.indices((3, 3)).reshape(2, -1)
PERIODS = {"train": pd.date_range("1990-01-01", "1999-12-31 21:00", freq="3h"),
           "validation": pd.date_range("2000-01-01", "2000-12-31 21:00", freq="3h"),
           "test": pd.date_range("2001-01-01", "2006-12-31 21:00", freq="3h")}
TREND = 0.1  # standard deviations of the noise a year, of the 2 m temperature alone, from 1995


class DriftTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        for name, times in PERIODS.items():
            # A seasonal and a daily cycle larger than the noise, in every variable.
            cycle = 2 * np.sin(2 * np.pi * times.dayofyear / 365.25) + 5 * np.sin(2 * np.pi * times.hour / 24)
            values = (cycle.to_numpy()[:, None, None] + rng.normal(size=(len(times), 9, len(FEATURE_NAMES)))).astype(np.float32)
            values[:, :, 0] += (TREND * (times.year - 1995)).to_numpy()[:, None]
            np.save(self.root / f"{name}.npy", values)
            np.save(self.root / f"{name}_timestamps.npy", times.as_unit("ns").asi8)
        (self.root / "metadata.json").write_text(json.dumps({
            "status": "complete", "features": list(FEATURE_NAMES), "start_year": 1990, "train_end_year": 1999,
            "validation_start_year": 2000, "validation_end_year": 2000, "test_start_year": 2001, "score_end_year": 2006,
            "grid": {"latitudes": [45.0, 44.5, 44.0], "longitudes": [7.0, 7.5, 8.0], "rows": ROWS.tolist(),
                     "cols": COLS.tolist(), "spacing": 0.5}}), encoding="utf-8")

    def run_script(self, *arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--prepared-dir", str(self.root), "--cell-stride", "1", *arguments]), 0)
        return output.getvalue()

    def table(self, text, season):
        """year -> the numbers of its row, and the row of the last 5 scored years, of one season."""
        lines = text.split(f"\n{season}\n")[1].split("\n\n")[0].splitlines()[1:]
        rows = {line.split()[0]: [float(value) for value in line.split()[1:] if value != "scored"] for line in lines}
        return {int(year): row for year, row in rows.items() if year.isdigit()}, rows["last"][1:]

    def test_the_trend_of_one_variable_is_read_by_year_and_the_others_stay_at_zero(self):
        text = self.run_script("--features", "2m_temperature", "total_precipitation")
        self.assertIn("normal: 1990-1999", text)
        years, last = self.table(text, "all months")
        self.assertEqual(list(years), list(range(1990, 2007)))
        self.assertEqual(text.count("  scored"), 3 * 7)
        # The noise has a standard deviation of 1; the trend of the training years and the part of the
        # seasonal cycle inside a month add a little to the spread of a month.
        spread = np.sqrt(1 + np.var(TREND * (np.arange(1990, 2000) - 1995)) + 0.1)
        for year, (temperature, precipitation) in years.items():
            self.assertAlmostEqual(temperature, TREND * (year - 1994.5) / spread, delta=0.1)
            self.assertAlmostEqual(precipitation, 0, delta=0.08)
        self.assertAlmostEqual(np.mean([row[0] for year, row in years.items() if year < 2000]), 0, delta=0.02)
        self.assertAlmostEqual(last[0], TREND * (2004 - 1994.5) / spread, delta=0.1)
        self.assertGreater(years[2006][0], years[2001][0] + 0.3)
        # A line fitted on the training years takes the trend away, for the domain and for every cell.
        block = text.split("what a line fitted on 1990-1999 leaves in the last 5 scored years")[1]
        rows = {line[:32].strip(): [float(value) for value in line[32:].split()] for line in block.splitlines()[2:]}
        self.assertAlmostEqual(rows["slope"][0], TREND / spread, delta=0.01)
        self.assertAlmostEqual(rows["slope"][1], 0, delta=0.01)
        self.assertAlmostEqual(rows["none, all months"][0], last[0], delta=0.01)
        for name in ("domain", "cell"):
            for season in script.SEASONS:
                self.assertAlmostEqual(rows[f"{name}, {season}"][0], 0, delta=0.12)
            self.assertLess(rows[f"{name}, a cell in a year"][0], 0.4 * rows["none, a cell in a year"][0])
        # A variable without a trend: a line by cell follows the noise of its cell and leaves a little more.
        self.assertGreater(rows["cell, a cell in a year"][1], rows["domain, a cell in a year"][1])
        self.assertLess(rows["cell, the same in training"][1], rows["none, the same in training"][1])
        # The same trend in summer and in winter.
        for season in ("June to August", "December to February"):
            self.assertAlmostEqual(self.table(text, season)[0][2006][0], years[2006][0], delta=0.25)

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "Unknown variables"):
            self.run_script("--features", "sea_ice")
        with self.assertRaisesRegex(ValueError, "odd"):
            self.run_script("--time-stride", "8")
        self.assertIn("all months", self.run_script("--time-stride", "1", "--features", "temperature"))


if __name__ == "__main__":
    unittest.main()
