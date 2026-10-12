"""The event check must anchor every event to its dust arrival and read the score upwind before it."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import check_era5_dust_events as script

LATITUDES, LONGITUDES = np.arange(59.5, 20, -1.0), np.arange(-14.5, 50, 1.0)
DUST_LATITUDES, DUST_LONGITUDES = np.arange(59.5, 20, -1.5), np.arange(-14.5, 50, 1.5)
DAYS = pd.date_range("2018-01-01", "2025-12-31", freq="D")


def box(latitudes, longitudes, name):
    south, north, west, east = script.BOXES[name]
    return (latitudes >= south) & (latitudes <= north), (longitudes >= west) & (longitudes <= east)


class EventTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        rows, cols = (x.ravel() for x in np.meshgrid(np.arange(len(LATITUDES)), np.arange(len(LONGITUDES)), indexing="ij"))
        lat, lon = LATITUDES[rows], LONGITUDES[cols]
        scores = rng.normal(size=(len(DAYS), len(rows))).astype(np.float32)
        dust = rng.gamma(2.0, 0.02, (len(DAYS), len(DUST_LATITUDES), len(DUST_LONGITUDES))).astype(np.float32)
        # The event of 14-20 March 2022: the dust reaches the Iberian Peninsula and France on the 15th,
        # and the score rises over northwestern Africa the day before.
        arrival, before = DAYS.get_loc("2022-03-15"), DAYS.get_loc("2022-03-14")
        for region in (script.IBERIA, script.FRANCE):
            dust[np.ix_(np.arange(arrival, arrival + 4), *box(DUST_LATITUDES, DUST_LONGITUDES, region))] += 1
        south, north, west, east = script.BOXES["Northwest Africa"]
        scores[before, (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)] += 3
        (self.root / "run" / "scores").mkdir(parents=True)
        (self.root / "dust").mkdir()
        np.save(self.root / "run" / "scores" / "anomaly_mean.npy", scores)
        np.save(self.root / "run" / "test_timestamps.npy", DAYS.as_unit("ns").asi8)
        (self.root / "run" / "metadata.json").write_text(json.dumps({"grid": {
            "latitudes": LATITUDES.tolist(), "longitudes": LONGITUDES.tolist(), "rows": rows.tolist(),
            "cols": cols.tolist(), "spacing": 1.0}}), encoding="utf-8")
        for year in range(2018, 2025):  # no dust for 2025
            of_year = DAYS.year == year
            np.savez(self.root / "dust" / f"dust_{year}.npz", dust=dust[of_year],
                     days=DAYS[of_year].values.astype("datetime64[D]"), latitudes=DUST_LATITUDES, longitudes=DUST_LONGITUDES)

    def run_script(self, *arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--run-dir", str(self.root / "run"), "--dust-dir", str(self.root / "dust"),
                                          *arguments]), 0)
        return output.getvalue()

    def test_the_planted_event_is_anchored_and_its_source_is_seen_the_day_before(self):
        text = self.run_script("--dust-percentile", "99")
        event = text.split("===== 2022-03-14 -> 2022-03-20 =====")[1].split("=====")[0]
        for region in (script.IBERIA, script.FRANCE):
            line = next(line for line in event.splitlines() if line.startswith(region))
            self.assertIn("arrival 2022-03-15", line)
            self.assertIn("Northwest Africa", line)
            self.assertEqual(float(re.search(r"day -1: +(\d+)", line).group(1)), 100)
            self.assertLess(float(re.search(r"day -3: +(\d+)", line).group(1)), 100)
        # Both arrivals count, with the planted day and nothing else.
        summary = text.split("SUMMARY")[1]
        arrivals = int(re.search(r"arrivals with a dust day: (\d+)", summary).group(1))
        self.assertGreaterEqual(arrivals, 2)
        self.assertGreaterEqual(int(re.search(r"on the day before: (\d+)", summary).group(1)), 2)
        self.assertIn("14% expected by chance", summary)
        self.assertRegex(summary, rf"arrival region at or above 95 on the arrival day: \d+ \({(1 - script.ALARM / 100) * arrivals:.1f} "
                                  rf"expected by chance\) \| on the day before: \d+ \(")
        # The dust alone: high on the planted arrival day, as the arrival is where the dust rises, not the day before.
        for region in (script.IBERIA, script.FRANCE):
            line = next(line for line in event[event.index(region):].splitlines() if "the dust alone" in line)
            self.assertGreaterEqual(float(re.search(r"day \+0: +(\d+)", line).group(1)), 95)
            self.assertLess(float(re.search(r"day -1: +(\d+)", line).group(1)), 95)
            self.assertRegex(line, r"arrival region on the day before: +\d+ \|")
        on_the_day, before = map(int, re.search(
            r"the dust alone at or above 95 in the arrival region, on the arrival day: (\d+) \| on the day before: (\d+)",
            summary).groups())
        self.assertGreaterEqual(on_the_day, 2)
        self.assertLess(before, on_the_day)
        # With unrelated noise elsewhere, few other arrivals have a source alarm.
        self.assertLessEqual(int(re.search(r"days before: (\d+)", summary).group(1)), 2 + arrivals // 3)

    def test_events_without_an_arrival_day(self):
        text = self.run_script()
        # The Canary Islands are outside the domain: the score is read upwind, with no arrival day.
        canary = text.split("===== 2020-02-22 -> 2020-02-29 =====")[1].split("=====")[0]
        self.assertEqual([line.split("(")[0].strip() for line in canary.strip().splitlines()], list(script.OUTSIDE))
        self.assertIn("outside the domain: no arrival day", canary)
        self.assertIn("of 8", canary)
        # An event after the last year of dust.
        late = text.split("===== 2025-02-01 -> 2025-02-04 =====")[1].split("=====")[0]
        self.assertIn("no dust data: no arrival day", late)
        self.assertEqual([line.split("(")[0].strip() for line in late.strip().splitlines()],
                         ["Central North Africa", "Northeast Africa"])
        # A higher level leaves events without a dust day; they are not counted.
        strict = self.run_script("--dust-percentile", "99.99")
        self.assertIn("dust below the level in the whole window", strict)
        self.assertLess(int(re.search(r"arrivals with a dust day: (\d+)", strict).group(1)),
                        int(re.search(r"arrivals with a dust day: (\d+)", text).group(1)))

    def write_events(self):
        """An events database with small areas on many days and one large area over the central Sahara,
        outside every fixed box, the day before the planted arrival."""
        rng = np.random.default_rng(1)
        (self.root / "events").mkdir()
        database = sqlite3.connect(self.root / "events" / "events.sqlite")
        database.execute("CREATE TABLE frames(time_index INTEGER PRIMARY KEY, timestamp TEXT, threshold REAL)")
        database.execute("CREATE TABLE clusters(cluster_id INTEGER PRIMARY KEY, timestamp TEXT, cells INTEGER, "
                         "area_km2 REAL, centroid_lat REAL, centroid_lon REAL)")
        database.executemany("INSERT INTO frames VALUES (?,?,?)",
                             [(index, day.isoformat(), 1.0) for index, day in enumerate(DAYS)])
        rows = [(DAYS[day].isoformat(), int(cells), float(cells) * 2500, float(lat), float(lon))
                for day, cells, lat, lon in zip(rng.integers(0, len(DAYS), 4000), rng.integers(1, 60, 4000),
                                                rng.uniform(21, 59, 4000), rng.uniform(-14, 49, 4000))]
        rows.append((pd.Timestamp("2022-03-14 12:00").isoformat(), 900, 900 * 2500.0, 24.0, 10.0))
        database.executemany("INSERT INTO clusters(timestamp, cells, area_km2, centroid_lat, centroid_lon) "
                             "VALUES (?,?,?,?,?)", rows)
        database.commit()
        database.close()

    def test_an_anomalous_area_anywhere_over_northern_africa_is_found(self):
        self.write_events()
        areas = script.africa_areas(self.root / "events")
        self.assertEqual(len(areas), len(DAYS))
        self.assertEqual(areas.cells.idxmax(), pd.Timestamp("2022-03-14"))
        self.assertEqual(tuple(areas.loc["2022-03-14"]), (900, 900 * 2500.0, 24.0, 10.0))
        # Days without an area over northern Africa have none, and areas elsewhere are not counted.
        self.assertTrue((areas.cells == 0).any())
        self.assertTrue(areas.centroid_lat.dropna().between(script.AFRICA[0], script.AFRICA[1]).all())
        plain = self.run_script("--dust-percentile", "99")
        text = self.run_script("--dust-percentile", "99", "--events-dir", str(self.root / "events"))
        event = text.split("===== 2022-03-14 -> 2022-03-20 =====")[1].split("=====")[0]
        lines = [line for line in event.splitlines() if "anywhere over northern Africa" in line]
        self.assertEqual(len(lines), 2)  # one per region of the event
        for line in lines:
            self.assertEqual(float(re.search(r"day -1: +(\d+)", line).group(1)), 100)
            self.assertIn("largest area before the arrival: 2022-03-14, 900 cells, 2250 thousand km2, centre 24.0N 10.0E", line)
        summary = text.split("SUMMARY, anywhere over northern Africa")[1]
        self.assertGreaterEqual(int(re.search(r"days before: (\d+)", summary).group(1)), 2)
        self.assertRegex(summary, r"\(\d+% of all days, \d+\.\d arrivals\)")
        # Apart from the lines of the database and of the reading that includes it, the output is the plain one.
        keep = lambda value: [line for line in value.splitlines() if line and "northern Africa" not in line
                              and "largest area" not in line and "any signal" not in line and "expected: on any day" not in line]
        self.assertEqual(keep(text), keep(plain))

    def write_errors(self):
        """A run with the errors by variable: one variable is far from usual over France on the arrival day."""
        rng = np.random.default_rng(2)
        grid = json.loads((self.root / "run" / "metadata.json").read_text(encoding="utf-8"))["grid"]
        lat, lon = LATITUDES[np.asarray(grid["rows"])], LONGITUDES[np.asarray(grid["cols"])]
        errors = rng.random((len(DAYS), len(lat), 2)).astype(np.float32)
        south, north, west, east = script.BOXES[script.FRANCE]
        errors[DAYS.get_loc("2022-03-15"), (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east), 1] += 5
        (self.root / "errors" / "scores").mkdir(parents=True)
        np.save(self.root / "errors" / "scores" / "feature_scores.npy", errors)
        (self.root / "errors" / "metadata.json").write_text(json.dumps(
            {"preparation": {"features": ["total_cloud_cover", "2m_temperature"]}}), encoding="utf-8")

    def test_any_signal_near_the_arrival_and_the_variables_behind_it(self):
        self.write_errors()
        text = self.run_script("--dust-percentile", "99", "--feature-errors-dir", str(self.root / "errors"))
        event = text.split("===== 2022-03-14 -> 2022-03-20 =====")[1].split("=====")[0]
        france = event[event.index(script.FRANCE):]
        # The score upwind the day before is the highest of its month: the reading near the arrival takes it.
        self.assertEqual(float(re.search(r"any signal near the arrival .*: max +(\d+)", france).group(1)), 100)
        self.assertIn("variables with an unusual error in the region on the arrival day: 2m_temperature 100", france)
        iberia = event[event.index(script.IBERIA):event.index(script.FRANCE)]
        self.assertNotIn("2m_temperature 100", iberia)
        # The reading on every day of the run: with five readings a day, well above 5% and well below all.
        summary = text.split("SUMMARY, any signal near the arrival")[1]
        found, expected, share = re.search(r"at or above 95: (\d+) \(([\d.]+) expected: .* (\d+)% of the times\)", summary).groups()
        self.assertGreaterEqual(int(found), 2)
        self.assertTrue(10 <= int(share) <= 35, share)
        self.assertLess(float(expected), int(found) + 5)
        variables = text.split("SUMMARY, variables")[1]
        self.assertRegex(variables, r"per arrival: \d\.\d \(0\.1 on any day\)")
        self.assertRegex(variables, r"by variable \(\d\.\d expected for each\): .*2m_temperature \d+")
        # The readings of a day as if it were an arrival: its own region around it and the place upwind before it.
        score = {script.FRANCE: pd.Series([10.0, 20, 99, 30, 40], index=DAYS[:5]),
                 "Northwest Africa": pd.Series([97.0, 5, 5, 5, 96], index=DAYS[:5])}
        np.testing.assert_array_equal(script.near_signal(score, script.FRANCE).to_numpy(), [97, 99, 99, 99, 96])
        with self.assertRaisesRegex(ValueError, "do not match the timestamps"):
            script.feature_percentiles(self.root / "errors", DAYS[:10], LATITUDES, LONGITUDES)

    def test_some_years_of_a_run_read_as_a_run_scored_on_those_years(self):
        self.write_errors()
        self.write_events()
        years = (DAYS.year >= 2020) & (DAYS.year <= 2024)
        # The same scores, errors and areas, kept for 2020-2024 only: what a run scored on those years has.
        (self.root / "short" / "scores").mkdir(parents=True)
        (self.root / "short_errors" / "scores").mkdir(parents=True)
        np.save(self.root / "short" / "scores" / "anomaly_mean.npy",
                np.load(self.root / "run" / "scores" / "anomaly_mean.npy")[years])
        np.save(self.root / "short" / "test_timestamps.npy", DAYS[years].as_unit("ns").asi8)
        np.save(self.root / "short_errors" / "scores" / "feature_scores.npy",
                np.load(self.root / "errors" / "scores" / "feature_scores.npy")[years])
        for name, source in (("short", "run"), ("short_errors", "errors")):
            (self.root / name / "metadata.json").write_text((self.root / source / "metadata.json").read_text(
                encoding="utf-8"), encoding="utf-8")
        (self.root / "short_events").mkdir()
        database = sqlite3.connect(self.root / "short_events" / "events.sqlite")
        database.execute("ATTACH DATABASE ? AS source", (str(self.root / "events" / "events.sqlite"),))
        for table in ("frames", "clusters"):
            database.execute(f"CREATE TABLE {table} AS SELECT * FROM source.{table} "
                             "WHERE timestamp >= '2020-01-01' AND timestamp < '2025-01-01'")
        database.commit()
        database.close()
        whole = self.run_script("--dust-percentile", "99", "--years", "2020", "2024",
                                "--feature-errors-dir", str(self.root / "errors"), "--events-dir", str(self.root / "events"))
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main([
                "--run-dir", str(self.root / "short"), "--dust-dir", str(self.root / "dust"), "--dust-percentile", "99",
                "--feature-errors-dir", str(self.root / "short_errors"),
                "--events-dir", str(self.root / "short_events")]), 0)
        self.assertEqual(whole, output.getvalue())
        self.assertIn("run read from 2020-01-01 to 2024-12-31", whole)
        # The events before and after those years are left out, the first one inside them is read.
        headers = re.findall(r"===== (\S+) -> ", whole)
        self.assertEqual(headers[0], "2020-02-03")
        self.assertEqual(headers[-1], "2024-04-22")
        self.assertIn("arrival 2022-03-15", whole)
        # The percentiles are those of the years read, not those of the whole run.
        self.assertNotEqual(whole.split("SUMMARY")[1:], self.run_script(
            "--dust-percentile", "99", "--feature-errors-dir", str(self.root / "errors"),
            "--events-dir", str(self.root / "events")).split("SUMMARY")[1:])
        with self.assertRaisesRegex(ValueError, "no timestamp in 1990-1991"):
            script.main(["--run-dir", str(self.root / "run"), "--dust-dir", str(self.root / "dust"),
                         "--years", "1990", "1991"])

    def test_every_event_is_in_order_and_names_known_regions(self):
        starts = [pd.Timestamp(start) for start, _, _ in script.EVENTS]
        self.assertEqual(starts, sorted(starts))
        for start, end, regions in script.EVENTS:
            self.assertLessEqual(pd.Timestamp(start), pd.Timestamp(end))
            self.assertTrue(set(regions) <= set(script.SOURCES))
        self.assertEqual(set(script.SOURCES.values()) | set(script.OUTSIDE), set(script.BOXES) - set(script.SOURCES))


if __name__ == "__main__":
    unittest.main()
