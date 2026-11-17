import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.check_era5_forecast_verification import main, measures, tally, verdict

TIMES, CELLS = 6, 4
# the reference: cell 0 unusual at timestamps 2 and 3, cell 1 at timestamp 4
PATTERN = np.zeros((TIMES, CELLS), dtype=bool)
PATTERN[[2, 3], 0] = True
PATTERN[4, 1] = True


def write_run(directory, values, timestamps):
    directory = Path(directory)
    (directory / "scores").mkdir(parents=True)
    np.save(directory / "scores" / "anomaly_mean.npy", values.astype(np.float32))
    np.save(directory / "test_timestamps.npy", timestamps)
    (directory / "metadata.json").write_text(json.dumps({"anomaly_mean_file": "scores/anomaly_mean.npy"}),
                                             encoding="utf-8")


class TallyTest(unittest.TestCase):
    def test_counts_and_measures(self):
        now = np.array([True, True, False, False])
        before = np.array([True, False, False, False])
        forecast = np.array([True, False, True, False])
        usable = np.array([True, True, True, True])
        counts = tally(now, before, forecast, usable)
        np.testing.assert_array_equal(counts, [1, 1, 1, 1, 1, 0])
        result = measures(counts)
        self.assertAlmostEqual(result["POD"], 0.5)
        self.assertAlmostEqual(result["FAR"], 0.5)
        self.assertAlmostEqual(result["CSI"], 1 / 3)
        self.assertAlmostEqual(result["POD onsets"], 0.0)

    def test_unusable_values_are_left_out(self):
        counts = tally(np.array([True, False]), np.array([False, False]), np.array([True, True]),
                       np.array([False, False]))
        np.testing.assert_array_equal(counts, [0, 0, 0, 0, 0, 0])

    def test_verdict(self):
        table = pd.DataFrame({"POD": [0.6, 0.5, 0.4], "FAR": [0.3, 0.4, 0.2], "CSI": [0.4, 0.35, 0.3]},
                             index=["Pn", "reference_persistence", "Gn_persistence"])
        self.assertEqual(verdict(table, "Pn"), ("reference_persistence", True))


class EndToEndTest(unittest.TestCase):
    def test_perfect_forecast_beats_persistence(self):
        timestamps = pd.date_range("2005-01-01", periods=TIMES, freq="3h").as_unit("ns").asi8
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cube = tmp / "cube"
            (cube / "flags").mkdir(parents=True)
            write_run(cube / "votes", PATTERN.astype(np.float32) * 3, timestamps)
            runs = []
            for name in ("a", "b", "c"):
                np.save(cube / "flags" / f"{name}.npy", np.packbits(PATTERN, axis=1))
                write_run(tmp / name, PATTERN.astype(np.float32), timestamps)
                runs.append(f"{name}={tmp / name}")
            write_run(tmp / "P", PATTERN.astype(np.float32), timestamps)  # right every time
            write_run(tmp / "G", np.zeros((TIMES, CELLS), dtype=np.float32), timestamps)  # never flags
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                main(["--cube", str(cube), "--forecast", f"Pn={tmp / 'P'}", "--persistence", f"Gn={tmp / 'G'}",
                      "--baselines", *runs, "--votes", "2"])
            text = out.getvalue()
            self.assertIn("-> yes", text)
            rows = {line.split()[0]: line.split() for line in text.splitlines()
                    if line.split() and line.split()[0] in ("Pn", "reference_persistence")}
            # columns: flagged %, POD, FAR, CSI, bias, POD onsets, hits, misses, false alarms
            self.assertEqual(rows["Pn"][2:4], ["1.000", "0.000"])
            self.assertEqual(rows["reference_persistence"][2], "0.333")
            self.assertEqual(rows["reference_persistence"][3], "0.667")
            self.assertEqual(rows["reference_persistence"][6], "0.000")


if __name__ == "__main__":
    unittest.main()
