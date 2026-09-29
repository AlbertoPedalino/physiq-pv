"""Climatological z-score runs, their events, and the seasonal recurrence report."""
import contextlib
import io
import json
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

from physiq_pv.era5.cube import CubeGrid
from physiq_pv.era5.seasonality import annual_report, climatology_zscores, seasonality_metrics
from physiq_pv.era5.visualization import anomaly_activity, create_synthetic_demo, plot_anomaly_prevalence
from scripts.run_era5_stgan import main as run_era5


def seasonal_scores(years=3, n=6, seed=0):
    """3-hourly (T, N) scores whose spread is 5x larger at noon in April-May."""
    timestamps = pd.date_range("2005-01-01", f"{2005 + years}-01-01", freq="3h", inclusive="left")
    rng = np.random.default_rng(seed)
    spread = np.where(timestamps.month.isin((4, 5)) & (timestamps.hour == 12), 5.0, 1.0)
    scores = 1.0 + spread[:, None] * rng.standard_normal((len(timestamps), n))
    scores[:, 0] = np.nan  # a location without data stays NaN
    return scores.astype(np.float32), timestamps


def raw_activity(scores, timestamps, top_percent=1):
    threshold = np.nanpercentile(scores, 100 - top_percent)
    return pd.DataFrame({"timestamp": timestamps, "n_valid": np.isfinite(scores).sum(axis=1),
                         "n_above_threshold": (np.nan_to_num(scores, nan=-np.inf) > threshold).sum(axis=1)})


def close_figures(report):
    for figure in report["figures"]:
        plt.close(figure)


class ClimatologyZScoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_z_scores_use_only_other_years(self):
        scores, timestamps = seasonal_scores(n=3)
        uncertainty = np.full_like(scores, .5)
        info = climatology_zscores(scores, timestamps, self.root / "z.npy", uncertainty=uncertainty,
                                   uncertainty_output=self.root / "z_std.npy", min_samples=2)
        z, z_std = np.load(self.root / "z.npy"), np.load(self.root / "z_std.npy")
        index = 1000
        time = timestamps[index]
        same = ((timestamps.month == time.month) & (timestamps.hour == time.hour)
                & (timestamps.year != time.year))
        reference = scores[same, 2].astype(np.float64)
        self.assertAlmostEqual(z[index, 2], (scores[index, 2] - reference.mean()) / reference.std(ddof=1), places=4)
        self.assertAlmostEqual(z_std[index, 2], .5 / reference.std(ddof=1), places=4)
        self.assertTrue(np.isnan(z[:, 0]).all() and np.isnan(z_std[:, 0]).all())
        self.assertEqual(info["years"], [2005, 2006, 2007])
        self.assertEqual(info["finite_values"], z.size - len(z))

    def test_small_groups_become_nan(self):
        scores, timestamps = seasonal_scores(years=2, n=2)
        info = climatology_zscores(scores, timestamps, self.root / "z.npy", min_samples=10_000)
        self.assertTrue(np.isnan(np.load(self.root / "z.npy")).all())
        self.assertEqual(info["finite_values"], 0)
        with self.assertRaises(ValueError):
            climatology_zscores(scores[:100], timestamps[:100], self.root / "one_year.npy")

    def test_script_writes_a_run_that_events_can_process(self):
        h, w = 2, 3
        scores, timestamps = seasonal_scores(n=h * w)
        rows, cols = np.indices((h, w))
        grid = CubeGrid(45 - np.arange(h) * .5, 7 + np.arange(w) * .5, rows.ravel(), cols.ravel())
        run = self.root / "run"
        (run / "scores").mkdir(parents=True)
        np.save(run / "scores" / "anomaly_mean.npy", scores)
        np.save(run / "scores" / "anomaly_std.npy", np.full_like(scores, .1))
        np.save(run / "test_timestamps.npy", timestamps.as_unit("ns").asi8)
        (run / "metadata.json").write_text(json.dumps({
            "status": "complete", "grid": grid.to_dict(), "anomaly_mean_file": "scores/anomaly_mean.npy",
            "anomaly_std_file": "scores/anomaly_std.npy"}), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            run_era5(["climatology", "--run-dir", str(run), "--output-dir", str(self.root / "zrun")])
            run_era5(["events", "--run-dir", str(self.root / "zrun"), "--output-dir", str(self.root / "zevents"),
                      "--opening-iterations", "0", "--closing-iterations", "0"])
        manifest = json.loads((self.root / "zrun" / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["score_kind"], "climatology_zscore")
        self.assertEqual(manifest["anomaly_std_file"], "scores/anomaly_std.npy")
        with self.assertRaises(ValueError):  # never overwrite an existing run
            run_era5(["climatology", "--run-dir", str(run), "--output-dir", str(self.root / "zrun")])

        activity = anomaly_activity(self.root / "zevents")
        self.assertEqual(len(activity), len(timestamps))
        self.assertAlmostEqual(activity.n_above_threshold.sum() / activity.n_valid.sum(), .01, delta=.002)
        raw_report = annual_report(raw_activity(scores, timestamps), self.root / "raw_report", permutations=20)
        z_report = annual_report(activity, self.root / "z_report", permutations=20)
        close_figures(raw_report)
        close_figures(z_report)
        raw_metrics, z_metrics = seasonality_metrics(raw_report), seasonality_metrics(z_report)
        self.assertGreater(raw_metrics["lift mese massimo"], 3)
        self.assertGreater(raw_metrics["lift ora massimo"], 3)
        self.assertLess(z_metrics["lift mese massimo"], 1.6)
        self.assertLess(z_metrics["lift ora massimo"], 1.6)


class ActivityAndReportTests(unittest.TestCase):
    def test_activity_matches_event_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            demo = create_synthetic_demo(Path(directory) / "demo")
            activity = anomaly_activity(demo)
            cube = np.load(demo / "anomaly_mean_cube.npy")
            labels = np.load(demo / "cluster_labels.npy")
            figure = plot_anomaly_prevalence(activity, "2005-01-01 03:00")
            plt.close(figure)
        np.testing.assert_array_equal(activity.n_above_threshold, (cube > 1).sum(axis=(1, 2)))
        np.testing.assert_array_equal(activity.n_event_cells, (labels > 0).sum(axis=(1, 2)))
        self.assertTrue((activity.threshold == 1).all())

    def test_report_lists_event_start_months(self):
        scores, timestamps = seasonal_scores()
        starts = pd.to_datetime(["2005-04-02", "2006-04-10", "2007-12-01", "2030-01-01"])
        with tempfile.TemporaryDirectory() as directory:
            report = annual_report(raw_activity(scores, timestamps), directory, event_starts=starts,
                                   header=["prova"], permutations=10)
            close_figures(report)
            summary = (Path(directory) / "summary.txt").read_text(encoding="utf-8")
        events = report["tables"]["Mese di inizio degli eventi"]
        self.assertEqual(events.loc["Apr", "eventi_iniziati"], 2)
        self.assertEqual(events.eventi_iniziati.sum(), 3)
        self.assertIn("prova", summary)
        self.assertAlmostEqual(report["tables"]["Tasso per mese"]["quota_osservazioni_%"].sum(), 100)


if __name__ == "__main__":
    unittest.main()
