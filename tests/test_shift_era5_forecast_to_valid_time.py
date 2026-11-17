import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.shift_era5_forecast_to_valid_time import KNOWN_AT, VALID_AT, main


def write_run(directory, kind=KNOWN_AT, spread=True):
    directory = Path(directory)
    (directory / "scores").mkdir(parents=True)
    mean = np.arange(12, dtype=np.float32).reshape(4, 3)
    mean[-1] = np.nan  # as score_era5_forecast_anomaly.py leaves it
    np.save(directory / "scores" / "anomaly_mean.npy", mean)
    np.save(directory / "test_timestamps.npy", np.arange(4, dtype=np.int64))
    run = {"status": "complete", "score_kind": kind, "anomaly_mean_file": "scores/anomaly_mean.npy",
           "anomaly_std_file": None, "grid": {"rows": [0]}}
    if spread:
        np.save(directory / "scores" / "generator_std.npy", mean * 0.1)
        run.update(anomaly_std_file="scores/generator_std.npy", generator_mean_file="scores/anomaly_mean.npy",
                   generator_std_file="scores/generator_std.npy")
    (directory / "metadata.json").write_text(json.dumps(run), encoding="utf-8")
    return mean


class ShiftTest(unittest.TestCase):
    def test_moves_every_file_one_row_later(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "P", Path(tmp) / "out"
            mean = write_run(source)
            self.assertEqual(main(["--run-dir", str(source), "--output-dir", str(target)]), 0)
            shifted = np.load(target / "scores" / "anomaly_mean.npy")
            self.assertTrue(np.isnan(shifted[0]).all())
            np.testing.assert_array_equal(shifted[1:], mean[:-1])
            spread = np.load(target / "scores" / "generator_std.npy")
            np.testing.assert_allclose(spread[1:], mean[:-1] * 0.1)
            np.testing.assert_array_equal(np.load(target / "test_timestamps.npy"), np.arange(4))
            run = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual((run["status"], run["score_kind"]), ("complete", VALID_AT))
            self.assertEqual(run["anomaly_std_file"], "scores/generator_std.npy")

    def test_refuses_a_run_already_moved(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(Path(tmp) / "P", kind=VALID_AT, spread=False)
            with self.assertRaises(ValueError):
                main(["--run-dir", str(Path(tmp) / "P"), "--output-dir", str(Path(tmp) / "out")])


if __name__ == "__main__":
    unittest.main()
