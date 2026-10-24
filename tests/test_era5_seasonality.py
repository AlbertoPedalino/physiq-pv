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
from physiq_pv.era5.seasonality import annual_report, climatology_zscores, reference_zscores, seasonality_metrics
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


def day_window_z(value, time, values, times, half):
    """The z-score of `value` at `time` against `values` at `times` (same hour, the reference years): mean of
    the days within `half` of its day of year, around the year; spread of each of those days around its own
    such mean."""
    days = times.dayofyear.to_numpy() - 1
    values = np.asarray(values, dtype=np.float64)

    def near(center):
        distance = np.abs(days - center)
        return np.minimum(distance, 366 - distance) <= half

    def mean(center):
        return values[near(center)].mean()

    center = time.dayofyear - 1
    squares = sum(((values[days == day % 366] - mean(day % 366))**2).sum()
                  for day in range(center - half, center + half + 1) if (days == day % 366).any())
    return (value - mean(center)) / np.sqrt(squares / (near(center).sum() - 1))


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

    def test_reference_z_scores_use_only_the_reference_without_its_excluded_values(self):
        scores, timestamps = seasonal_scores(years=3, n=3)
        reference, scored = timestamps.year < 2007, timestamps.year == 2007
        excluded = np.zeros((reference.sum(), 3), dtype=bool)
        excluded[::7, 2] = True
        uncertainty = np.full((scored.sum(), 3), .5, dtype=np.float32)
        info = reference_zscores(scores[scored], timestamps[scored], scores[reference], timestamps[reference],
                                 self.root / "z.npy", excluded=excluded, uncertainty=uncertainty,
                                 uncertainty_output=self.root / "z_std.npy", min_samples=2)
        z, z_std = np.load(self.root / "z.npy"), np.load(self.root / "z_std.npy")
        index = 1000
        time = timestamps[scored][index]
        same = ((timestamps[reference].month == time.month) & (timestamps[reference].hour == time.hour)
                & ~excluded[:, 2])
        values = scores[reference][same, 2].astype(np.float64)
        self.assertAlmostEqual(z[index, 2], (scores[scored][index, 2] - values.mean()) / values.std(ddof=1), places=4)
        self.assertAlmostEqual(z_std[index, 2], .5 / values.std(ddof=1), places=4)
        self.assertTrue(np.isnan(z[:, 0]).all())
        self.assertEqual(info["reference_values_left_out"], int(excluded[:, 2].sum()))
        self.assertEqual((info["reference_start"][:4], info["reference_end"][:4]), ("2005", "2006"))
        # The scored period is not in its own climatology: the same values scored twice give the same z.
        twice = np.concatenate([scores[scored], scores[scored]])
        reference_zscores(twice, timestamps[scored].append(timestamps[scored]), scores[reference],
                          timestamps[reference], self.root / "z2.npy", excluded=excluded, min_samples=2)
        np.testing.assert_allclose(np.load(self.root / "z2.npy")[:len(z)], z)
        with self.assertRaisesRegex(ValueError, "excluded"):
            reference_zscores(scores[scored], timestamps[scored], scores[reference], timestamps[reference],
                              self.root / "z3.npy", excluded=excluded[:10])

    def test_leave_one_year_out_without_the_excluded_values(self):
        scores, timestamps = seasonal_scores(n=3)
        excluded = np.zeros(scores.shape, dtype=bool)
        excluded[::5, 2] = True
        info = climatology_zscores(scores, timestamps, self.root / "z.npy", min_samples=2, excluded=excluded)
        z = np.load(self.root / "z.npy")
        index = 1000  # an excluded value is still scored
        self.assertTrue(excluded[index, 2])
        time = timestamps[index]
        same = ((timestamps.month == time.month) & (timestamps.hour == time.hour) & (timestamps.year != time.year)
                & ~excluded[:, 2])
        reference = scores[same, 2].astype(np.float64)
        self.assertAlmostEqual(z[index, 2], (scores[index, 2] - reference.mean()) / reference.std(ddof=1), places=4)
        self.assertEqual(info["reference_values_left_out"], int(excluded.sum()))

    def test_day_of_year_climatology_pools_a_window_around_each_day(self):
        scores, timestamps = seasonal_scores(n=3)
        index = 1000
        time = timestamps[index]
        other = (timestamps.hour == time.hour) & (timestamps.year != time.year)
        # Without a window: the same day of year and hour of the other years.
        climatology_zscores(scores, timestamps, self.root / "z0.npy", min_samples=2, period="day", window_days=0)
        reference = scores[other & (timestamps.dayofyear == time.dayofyear), 2].astype(np.float64)
        self.assertAlmostEqual(np.load(self.root / "z0.npy")[index, 2],
                               (scores[index, 2] - reference.mean()) / reference.std(ddof=1), places=4)
        info = climatology_zscores(scores, timestamps, self.root / "z.npy", min_samples=2, period="day",
                                   window_days=3)
        self.assertAlmostEqual(np.load(self.root / "z.npy")[index, 2],
                               day_window_z(scores[index, 2], time, scores[other, 2], timestamps[other], 3), places=4)
        self.assertEqual((info["period"], info["window_days"]), ("day", 3))
        with self.assertRaisesRegex(ValueError, "period"):
            climatology_zscores(scores, timestamps, self.root / "bad.npy", period="week")

    def test_day_of_year_window_wraps_around_the_year(self):
        scores, timestamps = seasonal_scores(years=3, n=2)
        reference, scored = timestamps.year < 2007, timestamps.year == 2007
        reference_zscores(scores[scored], timestamps[scored], scores[reference], timestamps[reference],
                          self.root / "z.npy", min_samples=2, period="day", window_days=2)
        index = 1  # 1 January 2007, 03 UTC: the window reaches the end of December
        time = timestamps[scored][index]
        same_hour = timestamps[reference].hour == time.hour
        self.assertAlmostEqual(np.load(self.root / "z.npy")[index, 1],
                               day_window_z(scores[scored][index, 1], time, scores[reference][same_hour, 1],
                                            timestamps[reference][same_hour], 2), places=4)

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
