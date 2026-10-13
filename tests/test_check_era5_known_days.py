"""The documented days of 2019 must be read where they are documented, among the days of their month."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import check_era5_known_days as script

LATITUDES, LONGITUDES = np.arange(50.0, 42.5, -0.5), np.arange(-1.0, 13.5, 0.5)
TIMES = pd.date_range("2017-01-01", "2020-12-31 12:00", freq="12h")


class KnownDaysTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows, cols = (x.ravel() for x in np.meshgrid(np.arange(len(LATITUDES)), np.arange(len(LONGITUDES)), indexing="ij"))
        self.lat, self.lon = LATITUDES[rows], LONGITUDES[cols]
        self.grid = {"latitudes": LATITUDES.tolist(), "longitudes": LONGITUDES.tolist(), "rows": rows.tolist(),
                     "cols": cols.tolist(), "spacing": 0.5}
        self.scores = np.random.default_rng(0).normal(size=(len(TIMES), len(rows))).astype(np.float32)

    def run_script(self, name, scores, times=TIMES, *arguments):
        (self.root / name / "scores").mkdir(parents=True)
        np.save(self.root / name / "scores" / "anomaly_mean.npy", scores)
        np.save(self.root / name / "test_timestamps.npy", times.as_unit("ns").asi8)
        (self.root / name / "metadata.json").write_text(json.dumps({"grid": self.grid}), encoding="utf-8")
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--run-dir", str(self.root / name), *arguments]), 0)
        return output.getvalue()

    def test_a_day_flagged_over_piedmont_is_the_highest_of_its_month_there(self):
        south, north, west, east = script.KNOWN_ZONES["Piedmont"]
        piedmont = (self.lat >= south) & (self.lat <= north) & (self.lon >= west) & (self.lon <= east)
        self.assertTrue(piedmont.any())
        scores = self.scores.copy()
        scores[np.ix_(TIMES.normalize() == pd.Timestamp("2019-10-21"), piedmont)] += 5
        text = self.run_script("run", scores)
        self.assertIn("documented days in it: 26 of 26", text)
        row = next(line for line in text.splitlines() if line.startswith("2019-10-21")).split()
        self.assertEqual(float(row[1]), 100)  # Piedmont
        piedmont_count, wide_count, both, expected = re.search(
            r"of 52: Piedmont (\d+) \| France and northern Italy (\d+) \| both (\d+) \(([\d.]+) expected", text).groups()
        self.assertEqual(int(piedmont_count) + int(wide_count), int(both))
        self.assertGreaterEqual(int(piedmont_count), 1)
        self.assertLessEqual(int(both), 12)  # noise alone: 2.6 expected
        self.assertEqual(expected, "2.6")
        # Without the planted day the reading is that of noise.
        plain = self.run_script("plain", self.scores)
        self.assertLess(float(next(line for line in plain.splitlines() if line.startswith("2019-10-21")).split()[1]), 100)

    def test_other_days_and_other_places(self):
        # A day of 2018 flagged in the north-east corner of the grid, far from Piedmont.
        corner = (self.lat >= 48) & (self.lon >= 10)
        scores = self.scores.copy()
        scores[np.ix_(TIMES.normalize() == pd.Timestamp("2018-04-04"), corner)] += 5
        text = self.run_script("other", scores, TIMES, "--days", "2018-04-03", "2018-04-04", "2031-01-01",
                               "--zone", "Corner", "48", "50", "10", "13", "--zone", "West", "43", "50", "-1", "3")
        self.assertIn("documented days in it: 2 of 3", text)
        header = next(line for line in text.splitlines() if line.startswith("day"))
        self.assertEqual(header.split(), ["day", "Corner", "West"])
        row = next(line for line in text.splitlines() if line.startswith("2018-04-04")).split()
        self.assertEqual(float(row[1]), 100)
        self.assertLess(float(row[2]), 100)
        self.assertIn("of 4: Corner", text)
        self.assertNotIn("Piedmont", text)
        with self.assertRaisesRegex(ValueError, "has no cell"):
            self.run_script("nowhere", scores, TIMES, "--zone", "Nowhere", "70", "75", "0", "5")

    def test_the_days_one_variable_at_a_time(self):
        # Two days of 2018: the second variable alone is far from usual over the corner, the score does not move.
        corner = (self.lat >= 48) & (self.lon >= 10)
        rng = np.random.default_rng(1)
        errors = rng.random((len(TIMES), len(self.lat), 3)).astype(np.float32)
        event = np.isin(TIMES.normalize(), pd.DatetimeIndex(["2018-04-03", "2018-04-04"]))
        errors[np.ix_(event, corner, [1])] += 5
        (self.root / "source" / "scores").mkdir(parents=True)
        np.save(self.root / "source" / "scores" / "feature_scores.npy", errors)
        (self.root / "source" / "metadata.json").write_text(json.dumps(
            {"preparation": {"features": ["wind", "temperature", "rain"]}}), encoding="utf-8")
        arguments = ("--days", "2018-04-03", "2018-04-04", "--zone", "Corner", "48", "50", "10", "13",
                     "--zone", "West", "43", "50", "-1", "3", "--feature-errors-dir", str(self.root / "source"))
        text = self.run_script("variables", self.scores, TIMES, *arguments)
        self.assertIn("days and places at or above 95, of 4: Corner 0", text)
        blocks = text.split("one variable at a time")[1].split("\n\n")[1:]
        rows = [{line.split()[0]: line.split()[1:] for line in block.splitlines()[2:]} for block in blocks]
        self.assertEqual([block.splitlines()[0] for block in blocks], ["Corner", "West"])
        self.assertEqual(rows[0]["temperature"], ["2", "100", "100"])
        self.assertEqual((rows[0]["wind"][0], rows[0]["rain"][0], rows[1]["temperature"][0]), ("0", "0", "0"))
        shorter = self.scores[:-2]
        with self.assertRaisesRegex(ValueError, "not over the same"):
            self.run_script("shorter", shorter, TIMES[:-2], *arguments)

    def test_the_days_in_the_data_themselves(self):
        # A low over the corner on one day of 2018: the pressure is the lowest of its month there, the wind the highest.
        corner = (self.lat >= 48) & (self.lon >= 10)
        rng = np.random.default_rng(2)
        values = rng.normal(size=(len(TIMES), len(self.lat), 3)).astype(np.float32)
        event = TIMES.normalize() == pd.Timestamp("2018-04-04")
        values[np.ix_(event, corner, [0])] -= 8
        values[np.ix_(event, corner, [1])] += 8
        prepared = self.root / "prepared"
        prepared.mkdir()
        np.save(prepared / "test.npy", values)
        np.save(prepared / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
        (prepared / "metadata.json").write_text(json.dumps({"features": ["pressure", "wind", "rain"]}), encoding="utf-8")
        zones = ("--zone", "Corner", "48", "50", "10", "13", "--zone", "West", "43", "50", "-1", "3")
        text = self.run_script("data", self.scores, TIMES, "--days", "2018-04-03", "2018-04-04", *zones,
                               "--prepared-dir", str(prepared))
        blocks = text.split("the data themselves")[1].split("\n\n")[1:]
        self.assertEqual([block.splitlines()[0] for block in blocks], ["Corner", "West"])
        self.assertEqual(blocks[0].splitlines()[1].split(), ["04-03", "04-04"])
        rows = {line[:script.NAME].strip(): line[script.NAME:].split() for line in blocks[0].splitlines()[2:]}
        self.assertLessEqual(float(rows["pressure"][1]), 1)
        self.assertEqual(rows["wind"][1], "100")
        self.assertEqual(rows["variables outside 5-95"][1], "2")
        self.assertNotIn(blocks[1].splitlines()[2].split()[2], ("0", "1", "100"))
        self.assertNotIn("one variable at a time", text)
        # Over many days the data are summed up by variable: on how many days each was low or high.
        many = [str(day.date()) for day in pd.date_range("2018-03-25", "2018-04-08")]
        text = self.run_script("many", self.scores, TIMES, "--days", *many, *zones, "--prepared-dir", str(prepared))
        block = text.split("the data themselves")[1].split("\n\n")[1]
        self.assertEqual(block.splitlines()[1].split(), ["low", "days", "high", "days", "mean"])
        rows = {line[:script.NAME].strip(): line[script.NAME:].split() for line in block.splitlines()[2:]}
        self.assertGreaterEqual(int(rows["pressure"][0]), 1)
        self.assertGreaterEqual(int(rows["wind"][1]), 1)
        with self.assertRaisesRegex(ValueError, "not over the same"):
            self.run_script("shorter", self.scores[:-2], TIMES[:-2], "--days", "2018-04-04", *zones,
                            "--prepared-dir", str(prepared))

    def test_a_run_without_2019_has_no_documented_day(self):
        later = TIMES[TIMES.year == 2020]
        text = self.run_script("later", self.scores[TIMES.year == 2020], later)
        self.assertIn("documented days in it: 0 of 26", text)
        self.assertNotIn("at or above", text)

    def test_the_days_are_in_order_and_in_2019(self):
        days = pd.DatetimeIndex(script.KNOWN_DAYS)
        self.assertTrue(days.is_monotonic_increasing and days.is_unique)
        self.assertEqual(set(days.year), {2019})


if __name__ == "__main__":
    unittest.main()
