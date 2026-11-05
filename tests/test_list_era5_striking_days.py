"""The striking days must be those with the largest single patch, one per episode, read in space only."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import list_era5_striking_days as script


class PatchTests(unittest.TestCase):
    def test_largest_patch_joins_diagonals_and_keeps_the_largest(self):
        flagged = np.zeros((6, 6), dtype=bool)
        flagged[0, 0] = flagged[1, 1] = flagged[2, 2] = True  # three cells touching on the diagonal
        flagged[4, 4] = flagged[4, 5] = True
        patch, size = script.largest_patch(flagged)
        self.assertEqual(size, 3)
        self.assertTrue(patch[0, 0] and patch[2, 2] and not patch[4, 4])

    def test_no_flag_no_patch(self):
        patch, size = script.largest_patch(np.zeros((3, 3), dtype=bool))
        self.assertEqual(size, 0)
        self.assertFalse(patch.any())

    def test_days_are_taken_largest_first_and_apart(self):
        days = np.array(["2010-01-01", "2010-01-03", "2010-01-20", "2010-01-26", "2010-02-10"], dtype="datetime64[D]")
        sizes = [50, 80, 30, 40, 0]
        taken = script.choose_days(days, sizes, count=5, gap=7)
        # 3 January first; 1 January is 2 days from it; 26 January, then 20 January is 6 days from it; 10 February
        # has no flag
        self.assertEqual(taken, [1, 3])

    def test_ties_go_to_the_earlier_day(self):
        days = np.array(["2010-03-05", "2010-03-01"], dtype="datetime64[D]")
        self.assertEqual(script.choose_days(days, [10, 10], count=1, gap=7), [1])


class RunTests(unittest.TestCase):
    def test_days_and_cases_are_written(self):
        times = pd.date_range("2010-01-01", "2010-01-31 21:00", freq="3h")
        latitudes, longitudes = [41.0, 40.5, 40.0, 39.5], [10.0, 10.5, 11.0, 11.5, 12.0]
        rows = [row for row in range(4) for _ in range(5)]
        cols = [col for _ in range(4) for col in range(5)]
        rng = np.random.default_rng(0)
        scores = rng.random((len(times), 20)).astype(np.float32) * 0.5
        big = times.get_loc(pd.Timestamp("2010-01-10 06:00"))
        scores[big, :8] = 5.0  # two rows of the grid: the largest patch of the month
        small = times.get_loc(pd.Timestamp("2010-01-25 12:00"))
        scores[small, [0, 1, 5]] = 5.0
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "run"
            (run / "scores").mkdir(parents=True)
            np.save(run / "scores" / "anomaly_mean.npy", scores)
            np.save(run / "test_timestamps.npy", times.as_unit("ns").asi8)
            # The threshold comes from a reference run, as from the validation years: its 95th percentile is
            # about 0.95, above the background of the scores read.
            reference = Path(folder) / "reference"
            (reference / "scores").mkdir(parents=True)
            np.save(reference / "scores" / "anomaly_mean.npy",
                    np.linspace(0, 1, len(times) * 20, dtype=np.float32).reshape(len(times), 20))
            (run / "metadata.json").write_text(json.dumps(
                {"status": "complete", "grid": {"latitudes": latitudes, "longitudes": longitudes, "rows": rows,
                                                "cols": cols}, "anomaly_mean_file": "scores/anomaly_mean.npy",
                 "threshold_reference": str(reference)}),
                encoding="utf-8")
            output = Path(folder) / "out"
            script.main(["--score", f"G={run}", "--top-percent", "5", "--days", "2", "--gap-days", "7",
                         "--output-dir", str(output)])
            cases = pd.read_csv(output / "casi.csv")
            self.assertEqual(list(cases["giorno"]), ["2010-01-10", "2010-01-25"])
            self.assertEqual(cases.loc[0, "celle"], 8)
            self.assertEqual(cases.loc[0, "istante"], "2010-01-10 06:00:00")
            self.assertEqual((cases.loc[0, "lat_min"], cases.loc[0, "lat_max"]), (40.5, 41.0))
            self.assertEqual(len(pd.read_csv(output / "giorni.csv")), 31)
            with np.load(output / "casi.npz") as arrays:
                self.assertEqual(int(arrays["macchia_01"].sum()), 8)


if __name__ == "__main__":
    unittest.main()
