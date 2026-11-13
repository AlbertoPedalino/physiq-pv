"""The period of the prepared ERA5 variables: a daily cycle sampled every 3 h gives its first autocorrelation peak and
its periodogram peak at 24 h with a large prominence; noise gives no prominent peak; a constant variable is left out."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import check_era5_data_period as script


class DataPeriodTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_daily_cycle_noise_and_constant(self):
        times, cells = 8 * 200, 30
        timestamps = pd.date_range("2005-01-01", periods=times, freq="3h")
        rng = np.random.default_rng(0)
        phase = rng.uniform(0, 2 * np.pi, cells)
        data = np.zeros((times, cells, 3), dtype=np.float32)
        data[:, :, 0] = 280 + 5 * np.sin(2 * np.pi * np.arange(times)[:, None] / 8 + phase) + rng.standard_normal((times, cells))
        data[:, :, 1] = rng.standard_normal((times, cells))
        data[:, :, 2] = 1.0  # constant: no autocorrelation to compute
        root = Path(self.tmp.name)
        np.save(root / "test.npy", data)
        np.save(root / "test_timestamps.npy", timestamps.as_unit("ns").asi8)
        (root / "metadata.json").write_text(json.dumps({"features": ["t2m", "noise", "flat"]}), encoding="utf-8")
        script.main(["--prepared-dir", str(root), "--cells", "20", "--max-lag-hours", "48", "--csv", str(root / "p.csv")])
        table = pd.read_csv(root / "p.csv", index_col="variable")
        self.assertEqual(table.loc["t2m", "first_peak_hours"], 24)
        self.assertAlmostEqual(table.loc["t2m", "periodogram_peak_hours"], 24, delta=0.5)
        self.assertGreater(table.loc["t2m", "first_peak_prominence"], 1.0)
        self.assertEqual(table.loc["t2m", "cells"], 20)
        self.assertAlmostEqual(table.loc["t2m", "acf_0h"], 1.0)
        prominence = table.loc["noise", "first_peak_prominence"]
        self.assertTrue(np.isnan(prominence) or prominence < 0.1)
        self.assertEqual(table.loc["flat", "cells"], 0)
        self.assertTrue(np.isnan(table.loc["flat", "periodogram_peak_hours"]))


if __name__ == "__main__":
    unittest.main()
