"""The floor check on the training years must read only those years and find the variable that needs a floor."""
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
from scripts import check_era5_floor_on_training_years as script

ROWS, COLS = np.indices((3, 3)).reshape(2, -1)
PERIODS = {"train": pd.date_range("1990-01-01", "1997-12-31 21:00", freq="3h"),
           "validation": pd.date_range("1998-01-01", "1998-12-31 21:00", freq="3h"),
           "test": pd.date_range("1999-01-01", "2000-12-31 21:00", freq="3h")}
STEADY, WAKING = FEATURE_NAMES.index("temperature"), FEATURE_NAMES.index("volumetric_soil_water_layer_1")
HELD_OUT = 3  # 1996-1998 with the validation year, 1995-1997 without


class FloorOnTrainingYearsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        for name, times in PERIODS.items():
            values = rng.normal(size=(len(times), 9, len(FEATURE_NAMES))).astype(np.float32)
            # One variable hardly varies in summer in the first training years and varies as usual later.
            quiet = np.isin(times.month, (6, 7, 8)) & (times.year < 1995)
            values[quiet, :, WAKING] *= 1e-3
            # Values no training year holds: the check must not read the scored years.
            if name == "test":
                values[:] = 1e6
            np.save(self.root / f"{name}.npy", values)
            np.save(self.root / f"{name}_timestamps.npy", times.as_unit("ns").asi8)
        (self.root / "metadata.json").write_text(json.dumps({
            "status": "complete", "features": list(FEATURE_NAMES), "start_year": 1990, "train_end_year": 1997,
            "validation_start_year": 1998, "validation_end_year": 1998, "test_start_year": 1999, "score_end_year": 2000,
            "grid": {"latitudes": [45.0, 44.5, 44.0], "longitudes": [7.0, 7.5, 8.0], "rows": ROWS.tolist(),
                     "cols": COLS.tolist(), "spacing": 0.5}}), encoding="utf-8")

    def run_script(self, *arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--prepared-dir", str(self.root), "--cell-stride", "1",
                                          "--held-out-years", str(HELD_OUT), *arguments]), 0)
        return output.getvalue()

    def needed(self, text, name):
        line = next(line for line in text.splitlines() if line.startswith(f"{name} ") or line.startswith(f"{name},"))
        return line[40:48].strip()

    def test_only_the_training_years_are_read_and_split(self):
        text = self.run_script()
        self.assertIn("normal fitted on 1990-1995", text)
        self.assertIn("held out 1996-1998", text)
        self.assertIn("scored years not read", text)
        # A variable on one scale in all the training years needs no more than the smallest floor.
        self.assertEqual(self.needed(text, FEATURE_NAMES[STEADY]), format(script.FLOORS[0], "g"))
        self.assertEqual(self.needed(text, "total_precipitation"), format(script.FLOORS[0], "g"))

    def test_a_variable_that_wakes_up_in_the_held_out_years_needs_a_larger_floor(self):
        text = self.run_script()
        self.assertNotEqual(self.needed(text, FEATURE_NAMES[WAKING]), format(script.FLOORS[0], "g"))

    def test_the_validation_year_can_be_left_out(self):
        text = self.run_script("--without-validation-year")
        self.assertIn("normal fitted on 1990-1994", text)
        self.assertIn("held out 1995-1997", text)

    def test_the_held_out_years_must_leave_years_to_fit(self):
        with self.assertRaises(ValueError):
            script.main(["--prepared-dir", str(self.root), "--cell-stride", "1", "--held-out-years", "8"])


if __name__ == "__main__":
    unittest.main()
