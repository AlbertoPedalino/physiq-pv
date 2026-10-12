"""The event check must anchor every event to its dust arrival and read the score upwind before it."""
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

    def test_every_event_is_in_order_and_names_known_regions(self):
        starts = [pd.Timestamp(start) for start, _, _ in script.EVENTS]
        self.assertEqual(starts, sorted(starts))
        for start, end, regions in script.EVENTS:
            self.assertLessEqual(pd.Timestamp(start), pd.Timestamp(end))
            self.assertTrue(set(regions) <= set(script.SOURCES))
        self.assertEqual(set(script.SOURCES.values()) | set(script.OUTSIDE), set(script.BOXES) - set(script.SOURCES))


if __name__ == "__main__":
    unittest.main()
