"""The check of the normalization of a run must make what the run makes and show what its options change."""
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
from scripts import check_era5_run_normalization as script

ROWS, COLS = np.indices((3, 3)).reshape(2, -1)
PERIODS = {"train": pd.date_range("1990-01-01", "1997-12-31 21:00", freq="3h"),
           "validation": pd.date_range("1998-01-01", "1998-12-31 21:00", freq="3h"),
           "test": pd.date_range("1999-01-01", "2003-12-31 21:00", freq="3h")}
RAIN, WARMING = "total_precipitation", "2m_temperature"
TREND = 0.1  # of the noise a year


class RunNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "prepared"
        self.root.mkdir()
        rng = np.random.default_rng(0)
        for name, times in PERIODS.items():
            values = rng.normal(size=(len(times), 9, len(FEATURE_NAMES))).astype(np.float32)
            years = times.year + (times.dayofyear - 1) / 365.25
            values[:, :, FEATURE_NAMES.index(WARMING)] += (TREND * (years - 1994)).to_numpy()[:, None]
            # The rain falls outside the summer in the training years; in the others a little falls in summer too.
            summer = np.isin(times.month, (6, 7, 8))
            rain = rng.exponential(size=(len(times), 9)).astype(np.float32)
            rain[summer] = 0 if name == "train" else 0.5 * (rng.random((int(summer.sum()), 9)) < 0.02)
            values[:, :, FEATURE_NAMES.index(RAIN)] = rain
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

    def tables(self, text):
        """The three tables as {variable: the rest of its line}."""
        blocks = [text.split(title)[1].split("\n\n")[0].splitlines() for title in
                  ("what is applied", "drift left", "expected share of the reconstruction error")]
        return [{line[:script.NAME].strip(): line[script.NAME:] for line in block if line[:script.NAME].strip() in FEATURE_NAMES}
                for block in blocks]

    def test_the_options_of_the_run_put_the_unseen_years_back_on_the_training_scale(self):
        plain = self.run_script()
        chosen = self.run_script("--seasonal-std-floors", f"{RAIN}=0.3", "--seasonal-fourth-root-features", RAIN,
                                 "--seasonal-trend", "linear", "--seasonal-trend-features", WARMING)
        self.assertIn("normal: 1990-1997", plain)
        self.assertIn("cells 9 of 9", plain)
        applied, drift, shares = self.tables(plain)
        self.assertEqual(len(applied), len(FEATURE_NAMES))
        # As it is: the summer rain of the unseen years leaves the scale and the warming is read as an anomaly.
        self.assertTrue(applied[RAIN].rstrip().endswith("!!"))
        self.assertEqual(applied[RAIN].split()[:2], ["0.01", "+0.0000"])
        self.assertFalse(applied["temperature"].rstrip().endswith("!!"))
        self.assertGreater(float(drift[WARMING].split()[0]), 0.5)
        self.assertLess(abs(float(drift["temperature"].split()[0])), 0.1)
        # The rain makes more of the expected error in the scored years than in the training ones.
        weight, training, scored, weighted = map(float, shares[RAIN].split())
        self.assertGreater(scored, 1.5 * training)
        self.assertGreater(weighted, 1.5 * 100 / len(FEATURE_NAMES))
        self.assertAlmostEqual(sum(float(line.split()[1]) for line in shares.values()), 100, delta=0.5)
        # With the options: the rain on its fourth root with a higher floor, the line of the warming taken away.
        applied, drift, shares = self.tables(chosen)
        self.assertEqual(applied[RAIN].split()[:3], ["yes", "0.3", "+0.0000"])
        self.assertFalse(applied[RAIN].rstrip().endswith("!!"))
        self.assertAlmostEqual(float(applied[WARMING].split()[1]), TREND, delta=0.02)
        self.assertEqual(applied["temperature"].split()[:2], ["0.01", "+0.0000"])
        for season in range(3):
            self.assertLess(abs(float(drift[WARMING].split()[season])), 0.15)
        weight, training, scored, weighted = map(float, shares[RAIN].split())
        self.assertLess(weighted, 1.5 * 100 / len(FEATURE_NAMES))
        # The sample is not left behind.
        self.assertEqual(sorted(path.name for path in self.root.parent.iterdir()), ["prepared"])

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "Unknown variables"):
            self.run_script("--seasonal-fourth-root-features", "rain")
        with self.assertRaisesRegex(ValueError, "needs --seasonal-trend linear"):
            self.run_script("--seasonal-trend-features", WARMING)
        with self.assertRaisesRegex(ValueError, "VARIABLE=FLOOR"):
            self.run_script("--seasonal-std-floors", RAIN)


if __name__ == "__main__":
    unittest.main()
