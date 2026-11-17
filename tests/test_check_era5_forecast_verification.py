import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import scripts.check_era5_forecast_verification as verification
from scripts.check_era5_forecast_verification import main, measures, tally, verdict

CELLS = 4
BASELINES = ("a", "b", "c")


def pattern(times):
    """The reference: cell 0 unusual at timestamps 2 and 3, cell 1 at timestamp 4."""
    flags = np.zeros((times, CELLS), dtype=bool)
    flags[[2, 3], 0] = True
    flags[4, 1] = True
    return flags


def write_run(directory, values, timestamps, extra=None):
    directory = Path(directory)
    (directory / "scores").mkdir(parents=True)
    np.save(directory / "scores" / "anomaly_mean.npy", values.astype(np.float32))
    np.save(directory / "test_timestamps.npy", timestamps)
    (directory / "metadata.json").write_text(json.dumps({"anomaly_mean_file": "scores/anomaly_mean.npy", **(extra or {})}),
                                             encoding="utf-8")


def setup(tmp, times=6, forecast=None, timestamps=None, nan_at=None):
    """A cube of the baselines a, b, c and P (the forecast, in the cube) with threshold 0.5, and G (never flags)."""
    tmp = Path(tmp)
    if timestamps is None:
        timestamps = pd.date_range("2005-01-01", periods=times, freq="3h").as_unit("ns").asi8
    reference = pattern(times)
    forecast = reference if forecast is None else forecast
    cube = tmp / "cube"
    (cube / "flags").mkdir(parents=True)
    detectors = {}
    for name in BASELINES:
        np.save(cube / "flags" / f"{name}.npy", np.packbits(reference, axis=1))
        values = reference.astype(np.float32)
        if nan_at is not None and name == "a":
            values[nan_at] = np.nan
        write_run(tmp / name, values, timestamps)
        detectors[name] = f"{tmp / name}@combined"
    np.save(cube / "flags" / "Pn.npy", np.packbits(forecast, axis=1))
    write_run(tmp / "P", forecast.astype(np.float32), timestamps)
    detectors["Pn"] = f"{tmp / 'P'}@combined"
    vote = {"detectors": detectors, "thresholds": {name: 0.5 for name in detectors}, "top_percent": 5}
    write_run(cube / "votes", reference.astype(np.float32) * 3, timestamps, {"vote": vote})
    write_run(tmp / "G", np.zeros((times, CELLS), dtype=np.float32), timestamps)
    return cube


def run(cube, tmp, votes=2, forecast_dir=None):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        main(["--cube", str(cube), "--forecast", f"Pn={forecast_dir or Path(tmp) / 'P'}",
              "--persistence", f"Gn={Path(tmp) / 'G'}", "--votes", str(votes)])
    text = out.getvalue()
    # columns: flagged %, POD, FAR, CSI, bias, POD onsets, hits, misses, false alarms
    rows = {line.split()[0]: line.split()[1:] for line in text.splitlines()
            if line.split() and line.split()[0] in ("Pn", "reference_persistence", "Gn_persistence")}
    return text, rows


class TallyTest(unittest.TestCase):
    def test_counts_and_measures(self):
        now = np.array([True, True, False, False])
        before = np.array([True, False, False, False])
        forecast = np.array([True, False, True, False])
        counts = tally(now, before, forecast, np.ones(4, dtype=bool))
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
        with tempfile.TemporaryDirectory() as tmp:
            text, rows = run(setup(tmp), tmp)
            self.assertIn("-> yes", text)
            self.assertEqual(rows["Pn"][1:3], ["1.000", "0.000"])
            # persistence of the reference: 1 hit (cell 0 at 3), 2 misses, 2 false alarms (cell 0 at 4, cell 1 at 5)
            self.assertEqual(rows["reference_persistence"][1:3], ["0.333", "0.667"])
            self.assertEqual(rows["reference_persistence"][5], "0.000")

    def test_several_blocks(self):
        old, verification.CHUNK = verification.CHUNK, 2
        try:
            with tempfile.TemporaryDirectory() as tmp:
                _, rows = run(setup(tmp, times=7), tmp)
        finally:
            verification.CHUNK = old
        self.assertEqual(rows["reference_persistence"][6:9], ["1.000", "2.000", "2.000"])
        self.assertEqual(rows["Pn"][6:9], ["3.000", "0.000", "0.000"])

    def test_gap_in_the_timestamps(self):
        stamps = pd.DatetimeIndex(["2005-01-01 00:00", "2005-01-01 03:00", "2005-01-01 06:00", "2005-01-01 09:00",
                                   "2005-01-01 15:00", "2005-01-01 18:00"]).as_unit("ns").asi8  # 09 -> 15 is no pair
        with tempfile.TemporaryDirectory() as tmp:
            _, rows = run(setup(tmp, timestamps=stamps), tmp)
        # timestamp 4 is not paired (3 -> 4 is a gap): hits of P 2; persistence: hit at 3, miss at 2, and of its false
        # alarms only cell 1 at 5 stays (4 -> 5 is a pair), cell 0 at 4 goes with the gap
        self.assertEqual(rows["Pn"][6:9], ["2.000", "0.000", "0.000"])
        self.assertEqual(rows["reference_persistence"][6:9], ["1.000", "1.000", "1.000"])

    def test_nan_removes_the_value_for_every_forecast(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, rows = run(setup(tmp, nan_at=(2, 0)), tmp)  # cell 0 at timestamp 2: pairs (1,2) and (2,3) go
        self.assertEqual(rows["Pn"][6:9], ["1.000", "0.000", "0.000"])
        self.assertEqual(rows["reference_persistence"][6:9], ["0.000", "1.000", "2.000"])

    def test_refuses_a_forecast_run_other_than_the_cube(self):
        with tempfile.TemporaryDirectory() as tmp:
            cube = setup(tmp)
            write_run(Path(tmp) / "other", pattern(6).astype(np.float32),
                      pd.date_range("2005-01-01", periods=6, freq="3h").as_unit("ns").asi8)
            with self.assertRaises(ValueError):
                run(cube, tmp, forecast_dir=Path(tmp) / "other")


if __name__ == "__main__":
    unittest.main()
