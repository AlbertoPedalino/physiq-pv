"""Flags that sit where the cluster will be: a best lag of +6 h for a detector that flags the block two steps ahead."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import build_era5_concordant_cube as cube
from scripts import check_era5_future_lean as script
from tests.test_build_era5_concordant_cube import write_run


class LeanTests(unittest.TestCase):
    def test_lag_of_flags_two_steps_ahead(self):
        own = np.zeros((10, 3, 16), dtype=bool)
        ahead = np.zeros_like(own)
        for t in range(10):
            own[t, 1, t:t + 2] = True
            ahead[t, 1, t + 2:t + 4] = True  # where the cluster is two steps later
        x = np.tile(np.arange(16, dtype=float) * 50.0, (3, 1))
        y = np.zeros((3, 16))
        out = script.lean(own, ahead, x, y, [1, 2], 2)
        self.assertEqual(out["lag"], 2)
        self.assertAlmostEqual(out[2], 100.0)  # two cells toward where it goes
        self.assertAlmostEqual(out["local"], 100.0)
        self.assertEqual(script.lean(own, own, x, y, [1, 2], 2)["lag"], 0)


class RunTests(unittest.TestCase):
    def test_a_detector_flagging_the_future_block(self):
        height, width, times = 6, 20, 12
        rows, cols = np.divmod(np.arange(height * width), width)
        grid = {"rows": rows.tolist(), "cols": cols.tolist(), "latitudes": np.linspace(50, 47.5, height).tolist(),
                "longitudes": np.linspace(5, 14.5, width).tolist()}
        timestamps = pd.date_range("2005-01-01", periods=times, freq="3h")
        moving = np.zeros((times, height, width))
        future = np.zeros_like(moving)
        for t in range(times):
            moving[t, 2:4, t:t + 2] = 10
            future[t, 2:4, t + 2:t + 4] = 10
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            runs = []
            signals = [future, moving, moving, moving, -moving]
            for k, signal in enumerate(signals):
                scores = 0.01 * rng.random((times, height * width)) + signal.reshape(times, -1)
                write_run(folder / f"d{k}", scores, timestamps, grid)
                runs.append(f"d{k}={folder / f'd{k}'}")
            with contextlib.redirect_stdout(io.StringIO()):
                cube.main(["--runs", *runs, "--output", str(folder / "out"), "--top-percent", "4", "--votes", "3",
                           "--closure", "none", "--min-values", "5", "--max-hours", "100"])
                script.main(["--cube", str(folder / "out"), "--kind", "analysed", "--radius-km", "300", "--first", "d0",
                             "--hours", "3", "6", "--max-lag", "6", "--csv", str(folder / "f.csv")])
            table = pd.read_csv(folder / "f.csv")
            big = table.loc[table.groupby("cluster")["hours"].transform("max").idxmax(), "cluster"]
            part = table[table["cluster"] == big].set_index("detector")
            self.assertGreater(part.loc["d0", "best_lag_h"], 0)  # d0 sits where the block will be
            self.assertEqual(part.loc["d1", "best_lag_h"], 0)


if __name__ == "__main__":
    unittest.main()
