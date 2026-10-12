"""The share of flagged values must be read by month and by hour, flat when the score has no season."""
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

from scripts import check_era5_score_by_month as script

TIMES = pd.date_range("2020-01-01", "2021-12-31 21:00", freq="3h")


class ByMonthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scores = np.random.default_rng(0).normal(size=(len(TIMES), 40)).astype(np.float32)

    def run_script(self, name, scores):
        (self.root / name / "scores").mkdir(parents=True)
        np.save(self.root / name / "scores" / "anomaly_mean.npy", scores)
        np.save(self.root / name / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
        (self.root / name / "metadata.json").write_text("{}", encoding="utf-8")
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--run-dir", str(self.root / name)]), 0)
        return output.getvalue()

    def rows(self, text):
        return [np.array(line.split()[2:], dtype=float) for line in text.splitlines() if line.startswith("flagged %")]

    def test_a_score_without_a_season_is_flat(self):
        months, hours = self.rows(self.run_script("flat", self.scores))
        self.assertEqual((len(months), len(hours)), (12, 8))
        np.testing.assert_allclose(months, 5, atol=1.5)
        np.testing.assert_allclose(hours, 5, atol=1.0)

    def test_a_season_and_a_time_of_day_are_seen(self):
        scores = self.scores.copy()
        scores[TIMES.month == 7] += 2
        scores[TIMES.hour == 12] += 1
        scores[0, :5] = np.nan  # missing cells are not counted
        text = self.run_script("season", scores)
        months, hours = self.rows(text)
        self.assertEqual(int(months.argmax()), 6)
        self.assertGreater(months[6], 5 * np.delete(months, 6).max())
        self.assertEqual(int(hours.argmax()), 4)
        # The shares are those of the values above the threshold of every tenth timestamp.
        threshold = np.nanpercentile(scores[::10], 95)
        july = scores[TIMES.month == 7]
        self.assertAlmostEqual(months[6], 100 * (july > threshold).sum() / np.isfinite(july).sum(), places=1)
        lowest, highest, ratio = map(float, re.search(
            r"lowest ([\d.]+)% \| highest ([\d.]+)% \| highest over lowest ([\d.]+)", text).groups())
        self.assertAlmostEqual(highest, months[6], places=1)
        self.assertGreater(ratio, 5)
        self.assertLess(lowest, 5)


if __name__ == "__main__":
    unittest.main()
