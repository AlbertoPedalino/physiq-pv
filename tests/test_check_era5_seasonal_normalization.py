"""The normalization check must find the floor a variable needs and the drift a line removes."""
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
from scripts import check_era5_seasonal_normalization as script

ROWS, COLS = np.indices((3, 3)).reshape(2, -1)
PERIODS = {"train": pd.date_range("1990-01-01", "1997-12-31 21:00", freq="3h"),
           "validation": pd.date_range("1998-01-01", "1998-12-31 21:00", freq="3h"),
           "test": pd.date_range("1999-01-01", "2003-12-31 21:00", freq="3h")}
RAIN, WARMING = FEATURE_NAMES.index("total_precipitation"), FEATURE_NAMES.index("2m_temperature")
TREND = 0.1  # of the noise a year


class NormalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        for name, times in PERIODS.items():
            values = rng.normal(size=(len(times), 9, len(FEATURE_NAMES))).astype(np.float32)
            values[:, :, WARMING] += (TREND * (times.year + (times.dayofyear - 1) / 365.25 - 1994)).to_numpy()[:, None]
            # The rain falls outside the summer in the training years; in the others a little falls in summer too.
            summer = np.isin(times.month, (6, 7, 8))
            rain = rng.exponential(size=(len(times), 9)).astype(np.float32)
            rain[summer] = 0 if name == "train" else 0.5 * (rng.random((int(summer.sum()), 9)) < 0.02)
            values[:, :, RAIN] = rain
            np.save(self.root / f"{name}.npy", values)
            np.save(self.root / f"{name}_timestamps.npy", times.as_unit("ns").asi8)
        (self.root / "metadata.json").write_text(json.dumps({
            "status": "complete", "features": list(FEATURE_NAMES), "start_year": 1990, "train_end_year": 1997,
            "validation_start_year": 1998, "validation_end_year": 1998, "test_start_year": 1999, "score_end_year": 2003,
            "grid": {"latitudes": [45.0, 44.5, 44.0], "longitudes": [7.0, 7.5, 8.0], "rows": ROWS.tolist(),
                     "cols": COLS.tolist(), "spacing": 0.5}}), encoding="utf-8")

    def run_script(self, *arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--prepared-dir", str(self.root), "--cell-stride", "1", *arguments]), 0)
        return output.getvalue()

    def rows(self, text, name):
        """floor -> replaced %, no spread %, low, high, then square, largest, beyond %, if cut of validation and scored."""
        lines = text.split(f"\n{name}\n")[1].split("\n\n")[0].splitlines()[2:]
        return {float(line.split()[0]): [float(value) for value in line.replace("..", " ").replace("|", " ").split()[1:]]
                for line in lines}

    def test_the_floor_a_variable_needs_and_the_line_that_removes_its_drift(self):
        text = self.run_script()
        self.assertIn("normal: 1990-1997", text)
        self.assertIn("validation 1998, scored 1999-2003", text)
        rain, steady = self.rows(text, "total_precipitation"), self.rows(text, "temperature")
        self.assertEqual(list(rain), list(script.FLOORS))
        # The middle of the summer has no spread of the rain in the training years, whatever the floor: the
        # window of days gives one to the days near its ends.
        for floor in script.FLOORS:
            self.assertTrue(5 < rain[floor][1] < 15, rain[floor])
            self.assertLess(rain[floor][0], 12)
        self.assertGreater(rain[0.3][0], rain[0.01][0])
        # With the lowest floor a rain of 0.5 is far beyond the training range, in the validation year and after.
        for start in (4, 8):
            self.assertGreater(rain[0.01][start], 3)
            self.assertGreater(rain[0.01][start + 1], 3)
            self.assertGreater(rain[0.01][start + 2], 0.05)
            # Cut at the training range it is much nearer the training years; with a higher floor it is anyway.
            self.assertLess(rain[0.01][start + 3], 0.5 * rain[0.01][start])
            self.assertLess(rain[0.3][start], 1.5)
            self.assertLess(rain[0.3][start + 1], 1.5)
        # A variable with a spread everywhere is the same at every floor.
        for floor in script.FLOORS:
            self.assertEqual(steady[floor][:2], [0.0, 0.0])
            self.assertAlmostEqual(steady[floor][4], 1, delta=0.1)
            self.assertAlmostEqual(steady[floor][8], 1, delta=0.1)
        needed = dict(line.rsplit("   own spread", 1)[0].rsplit(None, 1) for line in text.split(
            "the smallest floor that keeps the validation year")[1].split("\n\n")[0].splitlines()[1:] if "own spread" in line)
        self.assertEqual(float(needed["temperature"]), 0.01)
        self.assertGreater(float(needed["total_precipitation"]), 0.01)
        for transform in script.TRANSFORMS:
            self.assertIn(f"total_precipitation, {transform}", text)
        # The drift of the last years and the line of the training years.
        drift = {line[:40].strip(): [float(value) for value in line[40:].split()]
                 for line in text.split("drift at floor 0.1")[1].splitlines()[2:]}
        self.assertEqual(len(drift), len(FEATURE_NAMES) + len(script.TRANSFORMS))
        slope, as_it_is, line_away = drift["2m_temperature"][:3]
        self.assertAlmostEqual(slope, TREND, delta=0.02)
        self.assertGreater(as_it_is, 0.5)
        self.assertLess(abs(line_away), 0.12)
        self.assertLess(abs(drift["temperature"][1]), 0.1)
        self.assertLess(abs(drift["temperature"][0]), 0.01)

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "odd"):
            self.run_script("--time-stride", "2")
        with self.assertRaisesRegex(ValueError, "positive"):
            self.run_script("--floors", "0", "0.1")
        with self.assertRaisesRegex(ValueError, "one of the floors"):
            self.run_script("--trend-floor", "0.2")


if __name__ == "__main__":
    unittest.main()
