"""Seasonal recurrence report and leave-one-year-out climatological z-scores."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physiq_pv.era5.seasonality import annual_report, climatology_activity, seasonality_metrics


def seasonal_cube(years=4, h=2, w=3, seed=0):
    """3-hourly scores whose spread is 5x larger at noon in April-May."""
    timestamps = pd.date_range(f"{2005}-01-01", f"{2005 + years}-01-01", freq="3h", inclusive="left")
    rng = np.random.default_rng(seed)
    spread = np.where(timestamps.month.isin((4, 5)) & (timestamps.hour == 12), 5.0, 1.0)
    scores = 1.0 + spread[:, None, None] * rng.standard_normal((len(timestamps), h, w))
    scores[:, 0, 0] = np.nan  # a cell without data stays out of every count
    return scores.astype(np.float32), timestamps


class ClimatologyTests(unittest.TestCase):
    def test_z_scores_use_only_other_years(self):
        scores, timestamps = seasonal_cube(years=3, h=1, w=2)
        activity, info = climatology_activity(scores, timestamps, top_percent=5, min_samples=2,
                                              start="2005-01-01", end="2008-01-01")
        # Brute force z for one cell at one timestamp, then compare counts over that frame.
        index = 1000
        time = timestamps[index]
        same = ((timestamps.month == time.month) & (timestamps.hour == time.hour)
                & (timestamps.year != time.year))
        reference = scores[same, 0, 1].astype(np.float64)
        z = (scores[index, 0, 1] - reference.mean()) / reference.std(ddof=1)
        row = activity.loc[activity.time_index == index].iloc[0]
        self.assertEqual(row.n_valid, 1)
        self.assertEqual(row.n_above_threshold, int(z > info["threshold_z"]))
        self.assertAlmostEqual(activity.n_above_threshold.sum() / activity.n_valid.sum(), .05, delta=.005)

    def test_normalization_removes_the_seasonal_peak(self):
        scores, timestamps = seasonal_cube()
        raw_threshold = np.nanpercentile(scores, 99)
        raw = pd.DataFrame({"timestamp": timestamps,
                            "n_valid": np.isfinite(scores).sum(axis=(1, 2)),
                            "n_above_threshold": (np.nan_to_num(scores, nan=-np.inf) > raw_threshold).sum(axis=(1, 2))})
        normalized, info = climatology_activity(scores, timestamps, top_percent=1)
        self.assertEqual(info["years"], [2005, 2006, 2007, 2008])
        with tempfile.TemporaryDirectory() as directory:
            raw_report = annual_report(raw, Path(directory) / "raw", permutations=50)
            clim_report = annual_report(normalized, Path(directory) / "clim", permutations=50)
            for figure in raw_report["figures"] + clim_report["figures"]:
                plt.close(figure)
            self.assertTrue((Path(directory) / "clim" / "summary.txt").exists())
        raw_metrics, clim_metrics = seasonality_metrics(raw_report), seasonality_metrics(clim_report)
        self.assertGreater(raw_metrics["lift mese massimo"], 3)
        self.assertGreater(raw_metrics["lift ora massimo"], 3)
        self.assertLess(clim_metrics["lift mese massimo"], 1.5)
        self.assertLess(clim_metrics["lift ora massimo"], 1.5)
        self.assertLess(raw_metrics["p ricorrenza"], .05)

    def test_report_lists_event_start_months(self):
        scores, timestamps = seasonal_cube(years=3)
        activity, _ = climatology_activity(scores, timestamps)
        starts = pd.to_datetime(["2005-04-02", "2006-04-10", "2007-12-01", "2030-01-01"])
        with tempfile.TemporaryDirectory() as directory:
            report = annual_report(activity, directory, event_starts=starts, header=["prova"],
                                   permutations=10)
            for figure in report["figures"]:
                plt.close(figure)
            summary = (Path(directory) / "summary.txt").read_text(encoding="utf-8")
        events = report["tables"]["Mese di inizio degli eventi"]
        self.assertEqual(events.loc["Apr", "eventi_iniziati"], 2)
        self.assertEqual(events.eventi_iniziati.sum(), 3)
        self.assertIn("prova", summary)
        self.assertAlmostEqual(report["tables"]["Tasso per mese"]["quota_osservazioni_%"].sum(), 100)


if __name__ == "__main__":
    unittest.main()
