"""The rise of a score must be its growth over the steps asked, written as a run the checks read."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import rise_era5_scores as script
from scripts.check_era5_lead import flags


class RiseTests(unittest.TestCase):
    def test_rise_over_one_and_two_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            times = pd.date_range("2005-01-01", periods=600, freq="3h")
            scores = np.zeros((len(times), 4), dtype=np.float32)
            scores[:, 1] = np.arange(len(times))  # grows by one every step
            scores[300:, 2] = 5  # jumps once
            scores[10, 3] = np.nan
            (root / "run" / "scores").mkdir(parents=True)
            np.save(root / "run" / "scores" / "anomaly_mean.npy", scores)
            np.save(root / "run" / "test_timestamps.npy", times.as_unit("ns").asi8)
            grid = {"latitudes": [45.0, 44.5], "longitudes": [10.0, 10.5], "rows": [0, 0, 1, 1], "cols": [0, 1, 0, 1]}
            (root / "run" / "metadata.json").write_text(json.dumps({"status": "complete", "grid": grid}),
                                                        encoding="utf-8")
            script.main(["--run-dir", str(root / "run"), "--output-dir", str(root / "one")])
            script.main(["--run-dir", str(root / "run"), "--output-dir", str(root / "two"), "--steps", "2"])
            one = np.load(root / "one" / "scores" / "anomaly_mean.npy")
            two = np.load(root / "two" / "scores" / "anomaly_mean.npy")
            self.assertTrue(np.isnan(one[0]).all() and np.isnan(two[:2]).all())
            np.testing.assert_array_equal(one[1:, 1], 1)
            np.testing.assert_array_equal(two[2:, 1], 2)
            self.assertEqual((one[1:, 2] != 0).sum(), 1)
            self.assertEqual(one[300, 2], 5)
            self.assertTrue(np.isnan(one[10, 3]) and np.isnan(one[11, 3]))
            # The checks read it: the jump is among the highest values.
            flagged, read_times, _ = flags(root / "one", 1)
            self.assertTrue(read_times.equals(times))
            self.assertTrue(flagged[300, 2])


if __name__ == "__main__":
    unittest.main()
