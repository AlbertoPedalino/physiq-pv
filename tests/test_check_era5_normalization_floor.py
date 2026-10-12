"""The floor check must show a variable that leaves its training scale where its spread is almost 0."""
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
from scripts import check_era5_normalization_floor as script

ROWS, COLS = np.indices((3, 3)).reshape(2, -1)
PERIODS = {"train": pd.date_range("1990-01-01", "1997-12-31 21:00", freq="3h"),
           "validation": pd.date_range("1998-01-01", "1998-12-31 21:00", freq="3h"),
           "test": pd.date_range("1999-01-01", "2002-12-31 21:00", freq="3h")}
RAIN, STEADY = FEATURE_NAMES.index("total_precipitation"), FEATURE_NAMES.index("2m_temperature")


class FloorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        for name, times in PERIODS.items():
            values = rng.normal(size=(len(times), 9, len(FEATURE_NAMES))).astype(np.float32)
            # The rain falls in winter alone in the training years; in the others a little falls in summer too.
            summer = np.isin(times.month, (6, 7, 8))
            rain = rng.exponential(size=(len(times), 9)).astype(np.float32)
            rain[summer] = 0 if name == "train" else 0.5 * (rng.random((int(summer.sum()), 9)) < 0.02)
            values[:, :, RAIN] = rain
            np.save(self.root / f"{name}.npy", values)
            np.save(self.root / f"{name}_timestamps.npy", times.as_unit("ns").asi8)
        (self.root / "metadata.json").write_text(json.dumps({
            "status": "complete", "features": list(FEATURE_NAMES), "start_year": 1990, "train_end_year": 1997,
            "validation_start_year": 1998, "validation_end_year": 1998, "test_start_year": 1999, "score_end_year": 2002,
            "grid": {"latitudes": [45.0, 44.5, 44.0], "longitudes": [7.0, 7.5, 8.0], "rows": ROWS.tolist(),
                     "cols": COLS.tolist(), "spacing": 0.5}}), encoding="utf-8")

    def run_script(self, *arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--prepared-dir", str(self.root), "--cell-stride", "1", "--features",
                                          "total_precipitation", "2m_temperature", *arguments]), 0)
        return output.getvalue()

    def rows(self, text, name):
        """floor -> at the floor %, low, high, beyond %, largest ratio, training square, scored square."""
        lines = text.split(f"\n{name}\n")[1].split("\n\n")[0].splitlines()[1:]
        return {float(line.split()[0]): [float(value) for value in line.replace("..", " ").split()[1:]] for line in lines}

    def test_a_low_floor_lets_a_little_summer_rain_leave_the_scale(self):
        text = self.run_script()
        self.assertIn("normal: 1990-1997", text)
        rain, steady = self.rows(text, "total_precipitation"), self.rows(text, "2m_temperature")
        self.assertEqual(list(rain), list(script.FLOORS))
        # A quarter of the year has no spread of its own, whatever the floor below the usual spread.
        for floor in (0.01, 0.03, 0.1, 0.3):
            self.assertAlmostEqual(rain[floor][0], 25, delta=1)
        # With the lowest floor a rain of 0.5 is 0.5 / (0.01 x the spread of the rain): far beyond the training range.
        self.assertGreater(rain[0.01][4], 3)
        self.assertGreater(rain[0.01][3], 0.1)
        self.assertGreater(rain[0.01][6], 2 * rain[0.01][5])
        # With a higher floor the same rain stays inside it and the two periods are on one scale.
        self.assertLess(rain[0.3][4], 1.5)
        self.assertLess(rain[0.3][3], 0.05)
        self.assertAlmostEqual(rain[0.3][6], rain[0.3][5], delta=0.1)
        squares = [rain[floor][6] for floor in script.FLOORS]
        self.assertEqual(squares, sorted(squares, reverse=True))
        # A variable with a spread everywhere is the same at every floor below it.
        for floor in (0.01, 0.03, 0.1, 0.3):
            self.assertEqual(steady[floor][0], 0)
            self.assertAlmostEqual(steady[floor][5], 1, delta=0.01)
            self.assertAlmostEqual(steady[floor][6], 1, delta=0.05)
        self.assertGreater(steady[1.0][0], 20)  # a floor as large as the usual spread replaces it in many places
        # The rain makes more of the mean square of the scored years with the lowest floor.
        shares = text.split("after the min-max, %\n")[1].splitlines()[1:]
        by_floor = {float(line.split()[0]): [float(value) for value in line.split()[1:]] for line in shares}
        self.assertGreater(by_floor[0.01][0], by_floor[0.3][0])
        for values in by_floor.values():
            self.assertAlmostEqual(sum(values), 100, delta=0.2)

    def test_refusals_and_options(self):
        with self.assertRaisesRegex(ValueError, "Unknown variables"):
            self.run_script("--features", "sea_ice")
        with self.assertRaisesRegex(ValueError, "odd"):
            self.run_script("--time-stride", "2")
        with self.assertRaisesRegex(ValueError, "positive"):
            self.run_script("--floors", "0")
        text = self.run_script("--floors", "0.5", "--time-stride", "3")
        self.assertEqual(list(self.rows(text, "total_precipitation")), [0.5])


if __name__ == "__main__":
    unittest.main()
