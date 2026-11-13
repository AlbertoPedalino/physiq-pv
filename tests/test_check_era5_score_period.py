"""The period of a score: a daily cycle sampled every 3 h gives its first autocorrelation peak and its periodogram peak
at 24 h; a score without a cycle has no peak at 24 h."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import check_era5_score_period as script


def write_run(directory, scores, timestamps):
    (directory / "scores").mkdir(parents=True)
    np.save(directory / "scores" / "anomaly_mean.npy", scores.astype(np.float32))
    np.save(directory / "test_timestamps.npy", timestamps.as_unit("ns").asi8)
    (directory / "metadata.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")


class ScorePeriodTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_daily_cycle_and_noise(self):
        times, cells = 8 * 200, 30
        timestamps = pd.date_range("2005-01-01", periods=times, freq="3h")
        rng = np.random.default_rng(0)
        phase = rng.uniform(0, 2 * np.pi, cells)
        daily = np.sin(2 * np.pi * np.arange(times)[:, None] / 8 + phase) + 0.5 * rng.standard_normal((times, cells))
        daily[5, 3] = np.nan  # a gap later in a cell: counted as its mean
        noise = rng.standard_normal((times, cells))
        root = Path(self.tmp.name)
        write_run(root / "daily", daily, timestamps)
        write_run(root / "noise", noise, timestamps)
        script.main(["--runs", f"daily={root / 'daily'}", f"noise={root / 'noise'}", "--cells", "20",
                     "--max-lag-hours", "48", "--csv", str(root / "period.csv")])
        table = pd.read_csv(root / "period.csv", index_col="detector")
        self.assertEqual(table.loc["daily", "first_peak_hours"], 24)
        self.assertAlmostEqual(table.loc["daily", "periodogram_peak_hours"], 24, delta=0.5)
        self.assertEqual(table.loc["daily", "gaps"], 0)
        self.assertEqual(table.loc["daily", "cells"], 20)
        self.assertAlmostEqual(table.loc["daily", "acf_0h"], 1.0)
        self.assertGreater(table.loc["daily", "acf_24h"], 0.5)
        self.assertLess(table.loc["daily", "acf_12h"], -0.5)
        self.assertLess(abs(table.loc["noise", "acf_24h"]), 0.1)
        self.assertGreater(table.loc["daily", "first_peak_prominence"], 1.0)  # from -0.67 at 12 h to 0.67 at 24 h
        noise_prominence = table.loc["noise", "first_peak_prominence"]
        self.assertTrue(np.isnan(noise_prominence) or noise_prominence < 0.1)


if __name__ == "__main__":
    unittest.main()
