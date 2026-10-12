"""The onset check must find the onsets of the dust and a score that rises in the hours before them."""
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

from scripts import check_era5_dust_onset_hours as script

LATITUDES, LONGITUDES = np.arange(59.5, 20, -1.0), np.arange(-14.5, 50, 1.0)
TIMES = pd.date_range("2020-01-01", "2021-12-31 21:00", freq="3h")
STARTS = np.arange(20 * 8 + 4, len(TIMES) - 40, 30 * 8)  # an episode every 30 days, from 12 UTC
EPISODE = 16  # timestamps of an episode: more than 5% of the time in all


class OnsetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        rows, cols = (x.ravel() for x in np.meshgrid(np.arange(len(LATITUDES)), np.arange(len(LONGITUDES)), indexing="ij"))
        lat, lon = LATITUDES[rows], LONGITUDES[cols]
        arrival = np.zeros(len(rows), dtype=bool)
        for region in script.SOURCES:
            arrival |= script.inside(lat, lon, region)
        scores = rng.normal(size=(len(TIMES), len(rows))).astype(np.float32)
        dust = (rng.random((len(TIMES), len(rows))) * 1e-3).astype(np.float32)
        for start in STARTS:
            # The dust is highest when it arrives, in every arrival region at once; the score of those
            # regions is high from 9 to 3 hours before and at no other time.
            dust[start:start + EPISODE, arrival] += (1 - 0.01 * np.arange(EPISODE, dtype=np.float32))[:, None]
            scores[start - 3:start, arrival] += 3
        (self.root / "run" / "scores").mkdir(parents=True)
        (self.root / "dust").mkdir()
        np.save(self.root / "run" / "scores" / "anomaly_mean.npy", scores)
        np.save(self.root / "run" / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
        (self.root / "run" / "metadata.json").write_text(json.dumps({"grid": {
            "latitudes": LATITUDES.tolist(), "longitudes": LONGITUDES.tolist(), "rows": rows.tolist(),
            "cols": cols.tolist(), "spacing": 1.0}}), encoding="utf-8")
        fields = dust.reshape(len(TIMES), len(LATITUDES), len(LONGITUDES))
        for year in (2020, 2021):
            of_year = TIMES.year == year
            np.savez(self.root / "dust" / f"eac4_{year}.npz", dust=fields[of_year],
                     times=TIMES[of_year].values.astype("datetime64[h]"), latitudes=LATITUDES, longitudes=LONGITUDES)

    def run_script(self, *arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--run-dir", str(self.root / "run"), "--dust-dir", str(self.root / "dust"),
                                          *arguments]), 0)
        return output.getvalue()

    def table(self, text, title):
        """reading -> {hours from the onset: value} of the table under `title`."""
        lines = text.split(title)[1].split("\n\n")[0].splitlines()[1:]
        lags = [int(value) for value in lines[0].split()[4:]]
        return {name: dict(zip(lags, map(float, line[26:].split())))
                for line in lines[1:] for name in script.READINGS if line.startswith(f"{name:<26}")}

    def test_the_score_is_seen_before_the_onset_and_the_dust_only_from_it(self):
        text = self.run_script()
        self.assertIn(f"all {5 * len(STARTS)}", text)
        for region in script.SOURCES:
            self.assertIn(f"{region} {len(STARTS)}", text)
        alarms = self.table(text, "onsets with an alarm")
        self.assertEqual(list(alarms), list(script.READINGS))
        self.assertEqual(list(alarms["dust, arrival region"])[0::8], [-48, -24, 0, 24])
        score, dust, rise = (alarms[name] for name in script.READINGS[:3])
        for lag in (-9, -6, -3):
            self.assertGreaterEqual(score[lag], 90)
            self.assertLess(dust[lag], 30)
            self.assertLess(rise[lag], 30)
        for lag in (-48, -24, -12, 0, 12):
            self.assertLess(score[lag], 30)
        # The dust and its rise are seen from the onset on, and nothing happens upwind.
        self.assertGreaterEqual(dust[0], 90)
        self.assertGreaterEqual(rise[0], 90)
        self.assertGreaterEqual(dust[12], 90)
        for name in script.READINGS[3:]:
            self.assertLess(max(alarms[name].values()), 30)
        means = self.table(text, "mean percentile")
        self.assertGreater(means["score, arrival region"][-6], 90)
        self.assertTrue(30 < means["score, arrival region"][-24] < 70)
        # In the day before: the score at every onset, the dust at few, and what any timestamp gives.
        found = re.findall(r"^(.{26})\s*(\d+) \((\d+)\)$", text.split("an alarm at any time")[1], flags=re.M)
        before = {name.strip(): int(value) for name, value, _ in found}
        usual = {name.strip(): int(value) for name, _, value in found}
        self.assertEqual(sorted(before), sorted(script.READINGS))
        self.assertGreaterEqual(before["score, arrival region"], 95)
        self.assertLess(before["dust, arrival region"], 30)
        self.assertTrue(all(5 <= value <= 45 for value in usual.values()), usual)

    def test_onsets_need_quiet_hours_and_a_complete_window(self):
        level = pd.Series([0, 0, 0, 0, 9, 9, 0, 9, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0.0],
                          index=pd.date_range("2020-01-01", periods=20, freq="3h"))
        found = script.onsets(level, 80, quiet_hours=9)
        self.assertEqual([str(time) for time in found], ["2020-01-01 12:00:00", "2020-01-02 12:00:00"])
        one_year = self.run_script("--years", "2020", "2020")
        self.assertIn("run read from 2020-01-01 00:00:00 to 2020-12-31 21:00:00", one_year)
        self.assertIn(f"all {5 * int((TIMES[STARTS].year == 2020).sum())}", one_year)
        (self.root / "dust" / "eac4_2021.npz").unlink()
        with self.assertRaisesRegex(ValueError, "does not cover every timestamp"):
            self.run_script()
        self.assertIn("all ", self.run_script("--years", "2020", "2020"))


if __name__ == "__main__":
    unittest.main()
