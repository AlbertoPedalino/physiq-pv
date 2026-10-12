"""The dust check must read both ADS layouts, resume its downloads and find only a planted lead."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import xarray as xr

from scripts import check_era5_dust_lead as script

LATITUDES = np.arange(59.5, 20, -1.5)
LONGITUDES = np.arange(-14.5, 50, 1.5)


def write_plain(path, times, values):
    xr.Dataset({"duaod550": (("time", "latitude", "longitude"), values)},
               coords={"time": times, "latitude": LATITUDES, "longitude": LONGITUDES}).to_netcdf(path)


class Client:
    """Refuses the zipped format, like a dataset that only serves plain NetCDF."""
    def __init__(self, fields):
        self.fields, self.calls = fields, []

    def retrieve(self, name, request, target):
        self.calls.append((name, request["date"][0], request["data_format"], request["area"]))
        if request["data_format"] == "netcdf_zip":
            raise RuntimeError("400 invalid data_format")
        write_plain(target, *self.fields[int(request["date"][0][:4])])


class DailyMeansTests(unittest.TestCase):
    def test_both_layouts_give_the_same_daily_means(self):
        rng = np.random.default_rng(0)
        times = pd.date_range("2006-01-01", "2006-01-10 21:00", freq="3h")
        values = rng.random((len(times), len(LATITUDES), len(LONGITUDES))).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_plain(root / "plain.nc", times, values)
            xr.Dataset({"duaod550": (("forecast_period", "forecast_reference_time", "latitude", "longitude"), values[None])},
                       coords={"forecast_period": [np.timedelta64(0, "h")], "forecast_reference_time": times,
                               "latitude": LATITUDES, "longitude": LONGITUDES,
                               "valid_time": (("forecast_period", "forecast_reference_time"), times.values[None])}
                       ).to_netcdf(root / "inner.nc")
            with zipfile.ZipFile(root / "zipped.zip", "w") as archive:
                archive.write(root / "inner.nc", "data_sfc.nc")
            plain, zipped = script.daily_means(root / "plain.nc"), script.daily_means(root / "zipped.zip")
        for a, b in zip(plain, zipped):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(plain[0], values.reshape(10, 8, len(LATITUDES), len(LONGITUDES)).mean(axis=1), atol=1e-6)
        self.assertEqual(str(plain[1][0]), "2006-01-01")
        self.assertEqual(len(plain[1]), 10)


class DownloadTests(unittest.TestCase):
    def test_refused_format_falls_back_and_saved_years_are_skipped(self):
        rng = np.random.default_rng(1)
        fields = {}
        for year in (2005, 2006):
            times = pd.date_range(f"{year}-01-01", f"{year}-01-05 21:00", freq="3h")
            fields[year] = (times, rng.random((len(times), len(LATITUDES), len(LONGITUDES))).astype(np.float32))
        client = Client(fields)
        with tempfile.TemporaryDirectory() as temp, patch.object(script, "create_client", return_value=client), \
                redirect_stdout(io.StringIO()):
            script.download_dust(temp, (2005, 2006), [59.5, -14.5, 20.5, 49.5])
            first = list(client.calls)
            script.download_dust(temp, (2005, 2006), [59.5, -14.5, 20.5, 49.5])
            dust, days, latitudes, _ = script.load_dust(temp)
            leftovers = [path.name for path in Path(temp).glob("*.tmp")]
        # The zipped format is tried once, then the accepted one is kept; nothing is asked twice.
        self.assertEqual([(call[1][:4], call[2]) for call in first],
                         [("2005", "netcdf_zip"), ("2005", "netcdf"), ("2006", "netcdf")])
        self.assertEqual(first[0][0], script.DATASET)
        self.assertEqual(first[0][3], [59.5, -14.5, 20.5, 49.5])
        self.assertEqual(client.calls, first)
        self.assertEqual(leftovers, [])
        self.assertEqual(dust.shape, (10, len(LATITUDES), len(LONGITUDES)))
        self.assertEqual([str(day.date()) for day in days[[0, -1]]], ["2005-01-01", "2006-01-05"])
        np.testing.assert_array_equal(latitudes, LATITUDES)

    def test_no_client_is_created_when_every_year_is_saved(self):
        with tempfile.TemporaryDirectory() as temp:
            np.savez(Path(temp) / "dust_2005.npz", dust=np.zeros((1, 1, 1), np.float32),
                     days=np.array(["2005-01-01"], "datetime64[D]"), latitudes=[0.0], longitudes=[0.0])
            with patch.object(script, "create_client", side_effect=AssertionError("no network expected")), \
                    redirect_stdout(io.StringIO()):
                script.download_dust(temp, (2005,), [1, 0, 0, 1])


def series(seed, planted):
    """Eight years of daily score share and dust; with `planted` the share rises the day before each arrival."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2005-01-01", "2012-12-31", freq="D")
    share = rng.gamma(2.0, 2.5, len(days))
    dust = rng.gamma(2.0, 0.02, len(days))
    arrivals = np.sort(rng.choice(np.arange(30, len(days) - 30, 40), 30, replace=False) + rng.integers(0, 10, 30))
    dust[arrivals] += 1
    if planted:
        share[arrivals - 1] += 20
    return share, dust, days.month.to_numpy(), arrivals


class LeadTests(unittest.TestCase):
    def test_planted_lead_is_found(self):
        share, dust, months, arrivals = series(2, planted=True)
        result = script.lead_test(share, dust, months, rng=np.random.default_rng(0))
        # 1% of 2922 days is 30 days: the 30 planted arrivals, and nothing else.
        np.testing.assert_array_equal(result["arrivals"], arrivals)
        self.assertLess(result["p_value"], 0.001)
        self.assertGreater(result["percentiles"].mean(), 95)
        lag = dict(zip(script.LAGS, result["episode_profile"]))
        self.assertGreater(lag[-1], 4 * lag[0])
        self.assertLess(abs(lag[0] - result["comparison_profile"][0]), 2)

    def test_window_moves_the_days_where_the_score_is_read(self):
        """The share rises on the arrival day only: the default window misses it, `(0, 0)` finds it."""
        rng = np.random.default_rng(5)
        days = pd.date_range("2005-01-01", "2012-12-31", freq="D")
        share, dust = rng.gamma(2.0, 2.5, len(days)), rng.gamma(2.0, 0.02, len(days))
        arrivals = np.sort(rng.choice(np.arange(30, len(days) - 30, 40), 30, replace=False) + rng.integers(0, 10, 30))
        dust[arrivals] += 1
        share[arrivals] += 20
        months = days.month.to_numpy()
        before = script.lead_test(share, dust, months, rng=np.random.default_rng(0))
        same_day = script.lead_test(share, dust, months, window=(0, 0), rng=np.random.default_rng(0))
        self.assertGreater(before["p_value"], 0.05)
        self.assertLess(same_day["p_value"], 0.001)
        np.testing.assert_array_equal(same_day["observed"], share[arrivals])
        # The default window is the three days before the arrival.
        np.testing.assert_array_equal(before["observed"], np.max([share[arrivals - k] for k in (1, 2, 3)], axis=0))
        np.testing.assert_array_equal(script.lead_test(share, dust, months, window=(-3, -1),
                                                       rng=np.random.default_rng(0))["null"], before["null"])
        with self.assertRaisesRegex(ValueError, "earlier to a later"):
            script.lead_test(share, dust, months, window=(1, 0))

    def test_dust_peak_is_the_highest_value_of_the_episode(self):
        days = pd.date_range("2005-01-01", "2006-12-31", freq="D")
        dust = np.zeros(len(days))
        dust[[100, 103, 300]] = [1, 3, 2]  # 103 belongs to the episode that arrives on day 100
        result = script.lead_test(np.ones(len(days)), dust, days.month.to_numpy(), dust_quantile=0.996, draws=10,
                                  rng=np.random.default_rng(0))
        np.testing.assert_array_equal(result["arrivals"], [100, 300])
        np.testing.assert_array_equal(result["dust_peak"], [3, 2])

    def test_independent_series_give_uniform_p_values(self):
        p_values = []
        for seed in range(60):
            share, dust, months, _ = series(seed, planted=False)
            p_values.append(script.lead_test(share, dust, months, draws=400, rng=np.random.default_rng(seed))["p_value"])
        p_values = np.array(p_values)
        self.assertLess((p_values < 0.05).mean(), 0.15)
        self.assertGreater((p_values < 0.5).mean(), 0.3)
        self.assertLess((p_values < 0.5).mean(), 0.7)

    def test_arrivals_within_the_gap_are_one_episode(self):
        days = pd.date_range("2005-01-01", "2006-12-31", freq="D")
        dust = np.zeros(len(days))
        dust[[100, 101, 105, 300]] = 1  # 105 is within 7 days of 101: same episode
        result = script.lead_test(np.ones(len(days)), dust, days.month.to_numpy(), dust_quantile=0.995, draws=10,
                                  rng=np.random.default_rng(0))
        np.testing.assert_array_equal(result["arrivals"], [100, 300])

    def test_dust_bands_count_the_days_with_a_high_score(self):
        """The score is high on the dustiest half per cent of the days and nowhere else."""
        rng = np.random.default_rng(9)
        days = pd.date_range("2005-01-01", "2010-12-31", freq="D")
        dust, share = rng.random(len(days)), rng.random(len(days))
        dustiest = dust >= np.percentile(dust, 99.5)
        share[dustiest] += 10
        counts = script.dust_band_counts(share, dust, days.month.to_numpy())
        self.assertEqual(counts[:, 0].sum(), len(days))
        self.assertEqual(len(counts), len(script.DUST_BANDS) - 1)
        self.assertEqual(tuple(counts[-1]), (dustiest.sum(), dustiest.sum()))
        # A tenth of the days of every month is high, whatever the dust: the other bands keep the rest.
        self.assertLess(counts[0, 1] / counts[0, 0], 0.12)
        self.assertRegex(script.band_line(counts), rf"^0-50: \d+% of \d+ \| .* \| 99\.5-100: 100% of {dustiest.sum()}$")

    def test_departure_is_the_distance_from_the_usual_value_of_the_month_and_hour(self):
        rng = np.random.default_rng(10)
        timestamps = pd.date_range("2005-01-01", "2008-12-31 21:00", freq="3h")
        # A daily cycle and a noise ten times smaller: only the day with the added 5 is far from usual.
        level = (np.sin(2 * np.pi * timestamps.hour.to_numpy() / 24) * 10 + rng.normal(size=len(timestamps)))[:, None]
        unusual = timestamps.normalize() == pd.Timestamp("2006-07-15")
        level[unusual] += 5
        result = script.departure(level, timestamps, ["2m_temperature"])["2m_temperature"]
        self.assertEqual(len(result), len(pd.unique(timestamps.normalize())))
        self.assertEqual(result.idxmax(), pd.Timestamp("2006-07-15"))
        self.assertGreater(result.max(), 3)
        self.assertLess(result.median(), 1)

    def test_nan_scores_are_not_counted_as_below_the_threshold(self):
        days = pd.DatetimeIndex(["2005-01-01", "2005-01-01", "2005-01-02"])
        scores = np.array([[3.0, np.nan], [0.0, 3.0], [np.nan, 3.0]])
        share = script.daily_share_above(scores, days, np.array([0, 1]), 1.0)
        np.testing.assert_allclose(share.to_numpy(), [100 * 2 / 3, 100.0])
        with self.assertRaisesRegex(ValueError, "No finite score"):
            script.daily_share_above(np.full((1, 2), np.nan), days[:1], np.array([0, 1]), 1.0)


class MainTests(unittest.TestCase):
    def test_only_the_region_with_the_planted_lead_stands_out(self):
        rng = np.random.default_rng(3)
        latitudes, longitudes = np.arange(59.5, 20, -1.0), np.arange(-14.5, 50, 1.0)
        rows, cols = (x.ravel() for x in np.meshgrid(np.arange(len(latitudes)), np.arange(len(longitudes)), indexing="ij"))
        lat, lon = latitudes[rows], longitudes[cols]
        days = pd.date_range("2005-01-01", "2012-12-31", freq="D")
        scores = rng.normal(size=(len(days), len(rows))).astype(np.float32)
        dust = rng.gamma(2.0, 0.02, (len(days), len(LATITUDES), len(LONGITUDES))).astype(np.float32)
        arrivals = np.sort(rng.choice(np.arange(30, len(days) - 30, 40), 30, replace=False) + rng.integers(0, 10, 30))
        _, south, north, west, east = script.REGIONS[0]
        dust[np.ix_(arrivals, (LATITUDES >= south) & (LATITUDES <= north), (LONGITUDES >= west) & (LONGITUDES <= east))] += 1
        scores[np.ix_(arrivals - 1, (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east))] += 2
        with tempfile.TemporaryDirectory() as temp:
            run, dust_dir = Path(temp) / "run", Path(temp) / "dust"
            (run / "scores").mkdir(parents=True)
            dust_dir.mkdir()
            np.save(run / "scores" / "anomaly_mean.npy", scores)
            np.save(run / "test_timestamps.npy", days.as_unit("ns").asi8)
            (run / "metadata.json").write_text(json.dumps({"grid": {
                "latitudes": latitudes.tolist(), "longitudes": longitudes.tolist(),
                "rows": rows.tolist(), "cols": cols.tolist(), "spacing": 1.0}}), encoding="utf-8")
            for year in range(2005, 2013):
                of_year = days.year == year
                np.savez(dust_dir / f"dust_{year}.npz", dust=dust[of_year], days=days[of_year].values.astype("datetime64[D]"),
                         latitudes=LATITUDES, longitudes=LONGITUDES)
            output = io.StringIO()
            with patch.object(script, "create_client", side_effect=AssertionError("no network expected")), \
                    redirect_stdout(output):
                code = script.main(["--run-dir", str(run), "--dust-dir", str(dust_dir), "--last-year", "2012"])
        self.assertEqual(code, 0)
        p_values = [float(value) for value in re.findall(r"p = ([\d.]+)", output.getvalue())]
        self.assertEqual(len(p_values), len(script.REGIONS) + 1)
        self.assertLess(p_values[0], 0.001)
        self.assertTrue(all(value > 0.01 for value in p_values[1:-1]), p_values)
        self.assertIn(f"{script.REGIONS[0][0]} | score read in {script.REGIONS[0][0]} | cells=", output.getvalue())
        self.assertIn("episodes above the 95th percentile 100% (5% expected) | "
                      "episodes above the 99th percentile 100% (1% expected)", output.getvalue())
        # Every planted episode is found, so both halves are: the split itself is exercised.
        self.assertRegex(output.getvalue(), r"dustier half of the episodes above the 90th percentile 100% \| other half 100%")

    def test_score_region_source_reads_the_score_upwind_of_the_arrival(self):
        """Dust arrives in Greece; the score rises the day before in northeastern Africa only."""
        rng = np.random.default_rng(4)
        latitudes, longitudes = np.arange(59.5, 20, -1.0), np.arange(-14.5, 50, 1.0)
        rows, cols = (x.ravel() for x in np.meshgrid(np.arange(len(latitudes)), np.arange(len(longitudes)), indexing="ij"))
        lat, lon = latitudes[rows], longitudes[cols]
        days = pd.date_range("2005-01-01", "2012-12-31", freq="D")
        scores = rng.normal(size=(len(days), len(rows))).astype(np.float32)
        dust = rng.gamma(2.0, 0.02, (len(days), len(LATITUDES), len(LONGITUDES))).astype(np.float32)
        arrivals = np.sort(rng.choice(np.arange(30, len(days) - 30, 40), 30, replace=False) + rng.integers(0, 10, 30))
        name, south, north, west, east = script.REGIONS[3]
        source_name, source_south, source_north, source_west, source_east = script.SOURCES[name]
        dust[np.ix_(arrivals, (LATITUDES >= south) & (LATITUDES <= north), (LONGITUDES >= west) & (LONGITUDES <= east))] += 1
        scores[np.ix_(arrivals - 1, (lat >= source_south) & (lat <= source_north)
                      & (lon >= source_west) & (lon <= source_east))] += 2
        with tempfile.TemporaryDirectory() as temp:
            run, dust_dir = Path(temp) / "run", Path(temp) / "dust"
            (run / "scores").mkdir(parents=True)
            dust_dir.mkdir()
            np.save(run / "scores" / "anomaly_mean.npy", scores)
            np.save(run / "test_timestamps.npy", days.as_unit("ns").asi8)
            (run / "metadata.json").write_text(json.dumps({"grid": {
                "latitudes": latitudes.tolist(), "longitudes": longitudes.tolist(),
                "rows": rows.tolist(), "cols": cols.tolist(), "spacing": 1.0}}), encoding="utf-8")
            for year in range(2005, 2013):
                of_year = days.year == year
                np.savez(dust_dir / f"dust_{year}.npz", dust=dust[of_year], days=days[of_year].values.astype("datetime64[D]"),
                         latitudes=LATITUDES, longitudes=LONGITUDES)
            found = {}
            for region in ("arrival", "source"):
                output = io.StringIO()
                with redirect_stdout(output):
                    script.main(["--run-dir", str(run), "--dust-dir", str(dust_dir), "--last-year", "2012",
                                 "--score-region", region])
                found[region] = output.getvalue()
        greece = {region: float(re.search(re.escape(name) + r" \|.*?p = ([\d.]+)", text, re.S).group(1))
                  for region, text in found.items()}
        self.assertGreater(greece["arrival"], 0.01)
        self.assertLess(greece["source"], 0.001)
        self.assertIn(f"{name} | score read in {source_name} | cells=", found["source"])

    def test_analysis_years_keep_only_that_part_of_the_period(self):
        """A lead planted in 2005-2008 only must be found there and not in 2009-2012."""
        rng = np.random.default_rng(6)
        latitudes, longitudes = np.arange(59.5, 20, -1.0), np.arange(-14.5, 50, 1.0)
        rows, cols = (x.ravel() for x in np.meshgrid(np.arange(len(latitudes)), np.arange(len(longitudes)), indexing="ij"))
        lat, lon = latitudes[rows], longitudes[cols]
        days = pd.date_range("2005-01-01", "2012-12-31", freq="D")
        scores = rng.normal(size=(len(days), len(rows))).astype(np.float32)
        dust = rng.gamma(2.0, 0.02, (len(days), len(LATITUDES), len(LONGITUDES))).astype(np.float32)
        arrivals = np.sort(rng.choice(np.arange(30, len(days) - 30, 40), 60, replace=False) + rng.integers(0, 10, 60))
        name, south, north, west, east = script.REGIONS[0]
        dust[np.ix_(arrivals, (LATITUDES >= south) & (LATITUDES <= north), (LONGITUDES >= west) & (LONGITUDES <= east))] += 1
        early = arrivals[days[arrivals].year <= 2008]
        scores[np.ix_(early - 1, (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east))] += 2
        with tempfile.TemporaryDirectory() as temp:
            run, dust_dir = Path(temp) / "run", Path(temp) / "dust"
            (run / "scores").mkdir(parents=True)
            dust_dir.mkdir()
            np.save(run / "scores" / "anomaly_mean.npy", scores)
            np.save(run / "test_timestamps.npy", days.as_unit("ns").asi8)
            (run / "metadata.json").write_text(json.dumps({"grid": {
                "latitudes": latitudes.tolist(), "longitudes": longitudes.tolist(),
                "rows": rows.tolist(), "cols": cols.tolist(), "spacing": 1.0}}), encoding="utf-8")
            for year in range(2005, 2013):
                of_year = days.year == year
                np.savez(dust_dir / f"dust_{year}.npz", dust=dust[of_year], days=days[of_year].values.astype("datetime64[D]"),
                         latitudes=LATITUDES, longitudes=LONGITUDES)
            found = {}
            for years in (("2005", "2008"), ("2009", "2012")):
                output = io.StringIO()
                with redirect_stdout(output):
                    script.main(["--run-dir", str(run), "--dust-dir", str(dust_dir), "--last-year", "2012",
                                 "--analysis-years", *years])
                found[years[0]] = output.getvalue()
        first = {key: float(re.search(r"p = ([0-9.]+)", text).group(1)) for key, text in found.items()}
        self.assertLess(first["2005"], 0.001)
        self.assertGreater(first["2009"], 0.01)
        self.assertIn("analysis years: 2005-2008", found["2005"])
        self.assertIn("from day -3 to day -1 from the arrival", found["2005"])

    def test_score_component_reads_that_part_of_the_score(self):
        """The lead is planted in the reconstruction error only: the combined score must not show it."""
        rng = np.random.default_rng(7)
        latitudes, longitudes = np.arange(59.5, 20, -1.0), np.arange(-14.5, 50, 1.0)
        rows, cols = (x.ravel() for x in np.meshgrid(np.arange(len(latitudes)), np.arange(len(longitudes)), indexing="ij"))
        lat, lon = latitudes[rows], longitudes[cols]
        days = pd.date_range("2005-01-01", "2012-12-31", freq="D")
        combined = rng.normal(size=(len(days), len(rows))).astype(np.float32)
        generator = rng.normal(size=(len(days), len(rows))).astype(np.float32)
        dust = rng.gamma(2.0, 0.02, (len(days), len(LATITUDES), len(LONGITUDES))).astype(np.float32)
        arrivals = np.sort(rng.choice(np.arange(30, len(days) - 30, 40), 30, replace=False) + rng.integers(0, 10, 30))
        _, south, north, west, east = script.REGIONS[0]
        dust[np.ix_(arrivals, (LATITUDES >= south) & (LATITUDES <= north), (LONGITUDES >= west) & (LONGITUDES <= east))] += 1
        generator[np.ix_(arrivals - 1, (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east))] += 2
        with tempfile.TemporaryDirectory() as temp:
            run, dust_dir = Path(temp) / "run", Path(temp) / "dust"
            (run / "scores").mkdir(parents=True)
            dust_dir.mkdir()
            np.save(run / "scores" / "anomaly_mean.npy", combined)
            np.save(run / "scores" / "generator_mean.npy", generator)
            np.save(run / "test_timestamps.npy", days.as_unit("ns").asi8)
            (run / "metadata.json").write_text(json.dumps({
                "anomaly_mean_file": "scores/anomaly_mean.npy", "generator_mean_file": "scores/generator_mean.npy",
                "grid": {"latitudes": latitudes.tolist(), "longitudes": longitudes.tolist(),
                         "rows": rows.tolist(), "cols": cols.tolist(), "spacing": 1.0}}), encoding="utf-8")
            for year in range(2005, 2013):
                of_year = days.year == year
                np.savez(dust_dir / f"dust_{year}.npz", dust=dust[of_year], days=days[of_year].values.astype("datetime64[D]"),
                         latitudes=LATITUDES, longitudes=LONGITUDES)
            found = {}
            for component in ("combined", "generator"):
                output = io.StringIO()
                with redirect_stdout(output):
                    script.main(["--run-dir", str(run), "--dust-dir", str(dust_dir), "--last-year", "2012",
                                 "--score-component", component])
                found[component] = output.getvalue()
            # A run without that part, like a climatology run, is refused instead of read as combined.
            with self.assertRaisesRegex(ValueError, "does not contain discriminator"):
                script.main(["--run-dir", str(run), "--dust-dir", str(dust_dir), "--last-year", "2012",
                             "--score-component", "discriminator"])
        first = {key: float(re.search(r"p = ([0-9.]+)", text).group(1)) for key, text in found.items()}
        self.assertGreater(first["combined"], 0.01)
        self.assertLess(first["generator"], 0.001)
        self.assertIn("score=generator (scores/generator_mean.npy)", found["generator"])
        self.assertIn("score=combined (scores/anomaly_mean.npy)", found["combined"])

    def test_circulation_change_separates_the_flagged_episodes(self):
        """Half of the arrivals come with a pressure jump, and only those raise the score."""
        rng = np.random.default_rng(8)
        latitudes, longitudes = np.arange(59.5, 20, -2.0), np.arange(-14.5, 50, 2.0)
        rows, cols = (x.ravel() for x in np.meshgrid(np.arange(len(latitudes)), np.arange(len(longitudes)), indexing="ij"))
        lat, lon = latitudes[rows], longitudes[cols]
        days = pd.date_range("2005-01-01", "2012-12-31", freq="D")
        features = ["2m_temperature", *script.CIRCULATION_FEATURES]
        scores = rng.normal(size=(len(days), len(rows))).astype(np.float32)
        fields = rng.normal(size=(len(days), len(rows), len(features))).astype(np.float32)
        dust = rng.gamma(2.0, 0.02, (len(days), len(LATITUDES), len(LONGITUDES))).astype(np.float32)
        arrivals = np.sort(rng.choice(np.arange(30, len(days) - 30, 40), 30, replace=False) + rng.integers(0, 10, 30))
        name, south, north, west, east = script.REGIONS[0]
        inside = (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)
        dust[np.ix_(arrivals, (LATITUDES >= south) & (LATITUDES <= north), (LONGITUDES >= west) & (LONGITUDES <= east))] += 1
        abrupt = arrivals[::2]
        scores[np.ix_(abrupt, inside)] += 2
        fields[np.ix_(abrupt, inside, [features.index("mean_sea_level_pressure")])] += 10
        with tempfile.TemporaryDirectory() as temp:
            run, dust_dir, prepared = Path(temp) / "run", Path(temp) / "dust", Path(temp) / "prepared"
            (run / "scores").mkdir(parents=True)
            dust_dir.mkdir()
            prepared.mkdir()
            np.save(run / "scores" / "anomaly_mean.npy", scores)
            np.save(run / "test_timestamps.npy", days.as_unit("ns").asi8)
            (run / "metadata.json").write_text(json.dumps({"grid": {
                "latitudes": latitudes.tolist(), "longitudes": longitudes.tolist(),
                "rows": rows.tolist(), "cols": cols.tolist(), "spacing": 2.0}}), encoding="utf-8")
            np.save(prepared / "test.npy", fields)
            np.save(prepared / "test_timestamps.npy", days.as_unit("ns").asi8)
            (prepared / "metadata.json").write_text(json.dumps({"features": features}), encoding="utf-8")
            for year in range(2005, 2013):
                of_year = days.year == year
                np.savez(dust_dir / f"dust_{year}.npz", dust=dust[of_year], days=days[of_year].values.astype("datetime64[D]"),
                         latitudes=LATITUDES, longitudes=LONGITUDES)
            arguments = ["--run-dir", str(run), "--dust-dir", str(dust_dir), "--last-year", "2012", "--window", "0", "0"]
            plain, with_change = io.StringIO(), io.StringIO()
            with redirect_stdout(plain):
                script.main(arguments)
            with redirect_stdout(with_change):
                script.main([*arguments, "--prepared-dir", str(prepared)])
            by_feature = io.StringIO()
            with redirect_stdout(by_feature):
                script.main([*arguments, "--prepared-dir", str(prepared), "--by-feature"])
            with self.assertRaisesRegex(ValueError, "needs --prepared-dir"):
                script.main([*arguments, "--by-feature"])
            change = script.circulation_change(prepared, days, {name: np.nonzero(inside)[0]})[name]
            np.save(prepared / "test_timestamps.npy", (days + pd.Timedelta(days=1)).as_unit("ns").asi8)
            with self.assertRaisesRegex(ValueError, "do not share the test timestamps"), redirect_stdout(io.StringIO()):
                script.main([*arguments, "--prepared-dir", str(prepared)])
        text = with_change.getvalue()
        first = text[text.index(name + " |"):text.index(script.REGIONS[1][0] + " |")]
        # The 15 arrivals with the jump are the flagged ones and the ones with the stronger change.
        self.assertIn("episodes with the stronger change: score above the 90th percentile 100% | other half", first)
        self.assertLess(float(re.search(r"other half (\d+)%", first.split("stronger change")[1]).group(1)), 35)
        self.assertGreater(float(re.search(r"change and score percentile ([+-][\d.]+)", first).group(1)), 0.5)
        self.assertRegex(text, r"ALL REGIONS by circulation change of the episode: change below its 50th percentile: \d+ episodes")
        # The lines of the plain check are unchanged: the new ones are only added.
        self.assertEqual([line for line in text.splitlines() if "circulation change" not in line
                          and "stronger change" not in line and "over all the episodes" not in line],
                         plain.getvalue().splitlines())
        # One row per variable and measure; the planted jump is in the pressure, in both of its measures.
        table = by_feature.getvalue().split("BY VARIABLE")[1]
        rows = {tuple(line.split()[:2]): line.split()[2:] for line in table.splitlines()
                if line.split()[:1] and line.split()[0] in features}
        self.assertEqual(set(rows), {(feature, measure) for feature in features for measure in ("change", "departure")})
        # Columns: arrival percentile, flagged below the 50th, flagged above the 90th, episodes above, two correlations.
        for measure in ("change", "departure"):
            below, above = (float(value.rstrip("%")) for value in rows["mean_sea_level_pressure", measure][1:3])
            self.assertLess(below, 15)
            self.assertGreater(above, 50)
            self.assertGreater(float(rows["mean_sea_level_pressure", measure][4]), 0.15)
            self.assertLess(float(rows["2m_temperature", measure][2].rstrip("%")), 40)
            self.assertLess(abs(float(rows["2m_temperature", measure][4])), 0.15)
        # Every other line is the one of the run without --by-feature.
        self.assertEqual(by_feature.getvalue().split("\nBY VARIABLE")[0].splitlines()
                         + by_feature.getvalue().split("\nBY VARIABLE")[1].splitlines()[4 + 2 * len(features):],
                         text.splitlines())
        # Daily data: the change of a day is the difference from the day before, the first day has none.
        self.assertTrue(np.isnan(change.iloc[0]))
        self.assertEqual(len(change), len(days))
        self.assertGreater(change[days[abrupt]].min(), 2 * change.drop(days[np.r_[abrupt, abrupt + 1]]).max())

    def test_every_receptor_region_has_a_source_in_northern_africa(self):
        self.assertEqual(set(script.SOURCES), {region[0] for region in script.REGIONS})
        for _, south, north, west, east in script.SOURCES.values():
            self.assertTrue(20.5 <= south < north <= 37 and -14.5 <= west < east <= 49.5)


if __name__ == "__main__":
    unittest.main()
