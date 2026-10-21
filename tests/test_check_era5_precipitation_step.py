"""The precipitation-step check must find tiny values that appear only after the training years."""
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
from scripts import check_era5_precipitation_step as script

ROWS, COLS = np.indices((3, 3)).reshape(2, -1)
PERIODS = {"train": pd.date_range("1996-01-01", "1999-12-31 21:00", freq="3h"),
           "validation": pd.date_range("2000-01-01", "2000-12-31 21:00", freq="3h"),
           "test": pd.date_range("2001-01-01", "2002-12-31 21:00", freq="3h")}
COLUMN = FEATURE_NAMES.index("total_precipitation")


class PrecipitationStepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        for name, times in PERIODS.items():
            values = rng.normal(size=(len(times), 9, len(FEATURE_NAMES))).astype(np.float32)
            # Rain in a fifth of the hours, exact zeros otherwise; from 2000 the dry hours hold a tiny value.
            rain = np.where(rng.random((len(times), 9)) < 0.2, rng.exponential(1e-3, (len(times), 9)), 0.0)
            if name != "train":
                rain = np.where(rain == 0, 1e-7, rain)
            values[:, :, COLUMN] = rain
            np.save(self.root / f"{name}.npy", values)
            np.save(self.root / f"{name}_timestamps.npy", times.as_unit("ns").asi8)
        (self.root / "metadata.json").write_text(json.dumps({
            "status": "complete", "features": list(FEATURE_NAMES), "start_year": 1996, "train_end_year": 1999,
            "validation_start_year": 2000, "validation_end_year": 2000, "test_start_year": 2001, "score_end_year": 2002,
            "grid": {"latitudes": [45.0, 44.5, 44.0], "longitudes": [7.0, 7.5, 8.0], "rows": ROWS.tolist(),
                     "cols": COLS.tolist(), "spacing": 0.5}}), encoding="utf-8")

    def test_tiny_values_after_the_training_years_are_counted_and_dated(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--prepared-dir", str(self.root), "--cell-stride", "1"]), 0)
        text = output.getvalue()
        self.assertIn("training 1996-1999", text)
        rows = {line.split()[0]: line.split() for line in text.split("values by year")[1].split("\n\n")[0].splitlines()[2:]}
        self.assertLess(float(rows["1999"][3]), 1.0)  # only the smallest rain in training
        self.assertGreater(float(rows["2000"][3]), 70.0)  # the former dry hours
        self.assertEqual(rows["2001"][-1], "scored")
        months = dict(line.split()[:2] for line in text.split("by month")[1].split("\n\n")[0].splitlines()[1:])
        self.assertLess(abs(float(months["1999-12"])), 0.5)
        self.assertGreater(float(months["2000-01"]), 0.5)


if __name__ == "__main__":
    unittest.main()
