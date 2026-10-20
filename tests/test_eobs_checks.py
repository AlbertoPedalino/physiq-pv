"""E-OBS rain on the cells of a run, and the check of the flags against it."""
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
import xarray as xr

from scripts import check_era5_score_against_eobs as check
from scripts import download_eobs

TIMES = pd.date_range("2005-01-01", "2005-12-31 21:00", freq="3h")
DAYS = pd.date_range("2005-01-01", "2005-12-31", freq="D")
LATITUDES, LONGITUDES = [45.0, 44.5, 44.0], [7.0, 7.5, 8.0, 8.5]
ROWS, COLS = np.indices((3, 4)).reshape(2, -1)
GRID = {"latitudes": LATITUDES, "longitudes": LONGITUDES, "rows": ROWS.tolist(), "cols": COLS.tolist(), "spacing": 0.5}
ALONE_DAYS = DAYS.day == 10  # the first score alone flags cell 0 on these days
SHARED_DAYS = DAYS.day == 20  # both flag cell 11 on these days


def eobs_file(path, days, values, latitudes, longitudes):
    xr.Dataset({"rr": (("time", "latitude", "longitude"), values)},
               coords={"time": days, "latitude": latitudes, "longitude": longitudes}).to_netcdf(path)


class RainOnCellsTests(unittest.TestCase):
    def test_spreads_of_the_day_where_the_score_flags(self):
        times = pd.date_range("2005-01-01", periods=16, freq="3h")
        days = pd.DatetimeIndex(["2005-01-01", "2005-01-02"])
        score = np.zeros((16, 2))
        score[[1, 3], 0] = 5  # flagged twice on the first day at cell 0
        spread = np.zeros((16, 2))
        spread[1, 0], spread[3, 0], spread[5, 0] = 1, 3, 100  # the unflagged timestamp does not count
        components = {"generator_mean": np.ones((16, 2)), "generator_std": spread,
                      "discriminator_std": np.full((16, 2), 0.5)}
        values = check.daily_spreads(components, score, 1.0, times, days)
        self.assertAlmostEqual(values[0, 0, 0], 2.0)
        self.assertAlmostEqual(values[0, 0, 2], 0.5)
        self.assertTrue(np.isnan(values[0, 1]).all() and np.isnan(values[1]).all())

    def test_mean_of_the_points_inside_each_cell(self):
        with tempfile.TemporaryDirectory() as temp:
            # E-OBS points at 0.25 degrees: 44.875, 45.125 around the cell at 45.0; 6.875, 7.125 around 7.0.
            latitudes, longitudes = np.array([44.875, 45.125]), np.array([6.875, 7.125, 7.375, 7.625])
            values = np.array([[[1.0, 3.0, np.nan, np.nan], [5.0, 7.0, np.nan, 10.0]]] * 3, dtype=np.float32)
            days = pd.date_range("2005-01-01", periods=3)
            eobs_file(Path(temp) / "a.nc", days, values, latitudes, longitudes)
            rain, found = download_eobs.values_on_cells([Path(temp) / "a.nc"], "rr", np.array([45.0, 45.0, 44.0]),
                                                      np.array([7.0, 7.5, 7.0]), 0.5, days[0], days[-1])
            self.assertTrue(found.equals(days))
            np.testing.assert_allclose(rain[0, 0], 4.0)  # mean of 1, 3, 5, 7
            np.testing.assert_allclose(rain[0, 1], 10.0)  # only one point measured
            self.assertTrue(np.isnan(rain[0, 2]))  # no point inside the cell

    def test_overlapping_periods_and_the_files_on_disk(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "eobs" / "raw").mkdir(parents=True)
            latitudes, longitudes = np.arange(43.875, 45.2, 0.25), np.arange(6.875, 8.7, 0.25)
            early, late = pd.date_range("2004-12-30", "2005-01-05"), pd.date_range("2005-01-04", "2005-01-10")
            shape = (len(latitudes), len(longitudes))
            eobs_file(root / "eobs" / "raw" / "eobs_precipitation_amount_early.nc", early, np.ones((len(early), *shape), np.float32), latitudes, longitudes)
            eobs_file(root / "eobs" / "raw" / "eobs_precipitation_amount_late.nc", late, np.full((len(late), *shape), 2, np.float32), latitudes, longitudes)
            run = root / "run"
            run.mkdir()
            (run / "metadata.json").write_text(json.dumps({"grid": GRID}), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                download_eobs.main(["--grid-dir", str(run), "--output-dir", str(root / "eobs"), "--skip-download",
                                    "--years", "2005", "2005"])
            rain = np.load(root / "eobs" / "rain.npy")
            days = pd.DatetimeIndex(np.load(root / "eobs" / "days.npy").astype("datetime64[D]"))
            self.assertEqual((days[0], days[-1], len(days)), (pd.Timestamp("2005-01-01"), pd.Timestamp("2005-01-10"), 10))
            self.assertEqual(rain.shape, (10, 12))
            np.testing.assert_allclose(rain[:3], 1)
            np.testing.assert_allclose(rain[3:], 2)  # the later file wins where the periods overlap


class ObservedRainCheckTests(unittest.TestCase):
    def test_the_flags_of_the_score_alone_fall_on_rainy_days(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rng = np.random.default_rng(0)
            # Zero outside the flags, so that the highest share holds the flags and nothing else.
            model, other = (np.zeros((len(TIMES), 12), dtype=np.float32) for _ in range(2))
            day_of = TIMES.normalize()
            model[np.isin(day_of, DAYS[ALONE_DAYS]) & (TIMES.hour == 12), 0] += 1
            model[np.isin(day_of, DAYS[SHARED_DAYS]) & (TIMES.hour == 12), 11] += 1
            other[np.isin(day_of, DAYS[SHARED_DAYS]) & (TIMES.hour == 12), 11] += 1
            runs = []
            for name, scores in (("run_model", model), ("run_other", other)):
                (root / name / "scores").mkdir(parents=True)
                np.save(root / name / "scores" / "anomaly_mean.npy", scores)
                np.save(root / name / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
                (root / name / "metadata.json").write_text(json.dumps({"status": "complete", "grid": GRID}), encoding="utf-8")
                runs.append(root / name)
            # It rains hard at cell 0 on the days of the score alone; elsewhere ordinary rain, some cells unmeasured.
            rain = rng.gamma(0.5, 2.0, (len(DAYS), 12)).astype(np.float32)
            rain[ALONE_DAYS, 0] += 100
            rain[:, 6] = np.nan
            (root / "eobs").mkdir()
            np.save(root / "eobs" / "rain.npy", rain)
            np.save(root / "eobs" / "days.npy", DAYS.values.astype("datetime64[D]").astype(np.int64))
            output = io.StringIO()
            with redirect_stdout(output):
                rows = check.main(["--rain-dir", str(root / "eobs"), "--score-dir", str(runs[0]),
                                   "--by-dirs", str(runs[1]), "--top-percent", "0.5"])
            self.assertIn("cells with stations: 11 of 12", output.getvalue())
            self.assertGreater(rows["score alone"][0], 90)  # same day
            self.assertLess(rows["score and another"][0], 20)
            self.assertLess(rows["ring around score alone"][0], 20)
            self.assertAlmostEqual(rows["all cell-days"][0], 5, delta=2)
            self.assertLess(rows["score alone"][1], 20)  # the next day is ordinary
            ratio, low, high = rows["score alone / all cell-days"][0]
            self.assertTrue(low <= ratio <= high)
            self.assertGreater(low, 5)  # every year has the rainy flags of the score alone: far above 1
            self.assertIn("draws of whole years (1 years)", output.getvalue())

    def test_alone_far_from_the_flags_of_the_others(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, other = (np.zeros((len(TIMES), 12), dtype=np.float32) for _ in range(2))
            day_of = TIMES.normalize()
            noon = TIMES.hour == 12
            edge_days, before_days = DAYS.day == 5, DAYS.day == 25
            # The first score flags cell 0 alone (day 10), next to the other at cell 1 (day 5) and the day before
            # the other flags cell 0 (day 25).
            model[np.isin(day_of, DAYS[ALONE_DAYS | edge_days | before_days]) & noon, 0] += 1
            other[np.isin(day_of, DAYS[edge_days]) & noon, 1] += 1
            other[np.isin(day_of, DAYS[before_days] + pd.Timedelta(days=1)) & noon, 0] += 1
            runs = []
            for name, scores in (("run_model", model), ("run_other", other)):
                (root / name / "scores").mkdir(parents=True)
                np.save(root / name / "scores" / "anomaly_mean.npy", scores)
                np.save(root / name / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
                (root / name / "metadata.json").write_text(json.dumps({"status": "complete", "grid": GRID}), encoding="utf-8")
                runs.append(root / name)
            # It rains hard at cell 0 only on the days the first score flags far from the other.
            rain = np.random.default_rng(2).gamma(0.5, 2.0, (len(DAYS), 12)).astype(np.float32)
            rain[ALONE_DAYS, 0] += 100
            (root / "eobs").mkdir()
            np.save(root / "eobs" / "rain.npy", rain)
            np.save(root / "eobs" / "days.npy", DAYS.values.astype("datetime64[D]").astype(np.int64))
            arguments = ["--rain-dir", str(root / "eobs"), "--score-dir", str(runs[0]), "--by-dirs", str(runs[1]),
                         "--top-percent", "0.5", "--by-top-percent", "0.5"]
            output = io.StringIO()
            with redirect_stdout(output):
                same_cell = check.main(arguments)
                cells_only = check.main([*arguments, "--apart-cells", "1"])
                apart = check.main([*arguments, "--apart-cells", "1", "--apart-days", "1"])
            self.assertIn("alone: no other flag within 1 cell(s) and 1 day(s)", output.getvalue())
            self.assertAlmostEqual(same_cell["score alone"][0], 100 / 3, delta=10)  # days 5, 10 and 25
            self.assertAlmostEqual(cells_only["score alone"][0], 50, delta=10)  # days 10 and 25
            self.assertGreater(apart["score alone"][0], 90)  # day 10 only

    def test_a_drop_of_the_pressure_where_the_score_flags_alone(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, other = (np.zeros((len(TIMES), 12), dtype=np.float32) for _ in range(2))
            day_of = TIMES.normalize()
            model[np.isin(day_of, DAYS[ALONE_DAYS]) & (TIMES.hour == 12), 0] += 1
            runs = []
            for name, scores in (("run_model", model), ("run_other", other)):
                (root / name / "scores").mkdir(parents=True)
                np.save(root / name / "scores" / "anomaly_mean.npy", scores)
                np.save(root / name / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
                (root / name / "metadata.json").write_text(json.dumps({"status": "complete", "grid": GRID}), encoding="utf-8")
                runs.append(root / name)
            # The pressure is low at cell 0 on the days of the score alone, after an ordinary day: a drop.
            rng = np.random.default_rng(1)
            pressure = (1013 + rng.normal(size=(len(DAYS), 12))).astype(np.float32)
            pressure[ALONE_DAYS, 0] -= 30
            (root / "eobs").mkdir()
            np.save(root / "eobs" / "pressure.npy", pressure)
            np.save(root / "eobs" / "days.npy", DAYS.values.astype("datetime64[D]").astype(np.int64))
            arguments = ["--eobs-dir", str(root / "eobs"), "--variable", "pressure", "--tail", "low",
                         "--score-dir", str(runs[0]), "--by-dirs", str(runs[1]), "--top-percent", "0.5"]
            output = io.StringIO()
            with redirect_stdout(output):
                drop = check.main([*arguments, "--change-days", "1"])
                level = check.main(arguments)
            self.assertIn("unusual change of the pressure over 1 day(s): lowest 5%", output.getvalue())
            self.assertGreater(drop["score alone"][0], 90)
            self.assertLess(drop["score alone"][1], 20)  # the day after the pressure rises again
            self.assertGreater(level["score alone"][0], 90)
            self.assertAlmostEqual(level["all cell-days"][0], 5, delta=2)


if __name__ == "__main__":
    unittest.main()
