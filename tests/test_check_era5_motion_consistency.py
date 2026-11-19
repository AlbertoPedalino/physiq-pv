"""Coherence of the local motion of a concordant cluster: a block moving straight, and a cluster that grows on one side."""
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
from scripts import check_era5_concordant_lead as lead
from scripts import check_era5_motion_consistency as script
from tests.test_build_era5_concordant_cube import write_run


class PieceTests(unittest.TestCase):
    def test_straight_motion_is_coherent(self):
        own = np.zeros((5, 3, 8), dtype=bool)
        for t in range(5):
            own[t, 1, t:t + 2] = True
        x = np.tile(np.arange(8, dtype=float) * 50.0, (3, 1))
        y = np.zeros((3, 8))
        out = script.consistency(own, x, y, 0.2)
        self.assertAlmostEqual(out["coherence"], 1.0)
        self.assertAlmostEqual(out["straightness"], 1.0)
        self.assertAlmostEqual(out["angle_deg"], 0.0, places=5)
        self.assertEqual(out["jumps"], 0.0)

    def test_growth_on_one_side_is_a_jump(self):
        own = np.zeros((3, 6, 6), dtype=bool)
        own[:, 2, 2] = True
        own[1:, 0:2, 4:6] = True  # a part joins from the north-east and stays: the centroid moves, the system does not
        x = np.tile(np.arange(6, dtype=float) * 50.0, (6, 1))
        y = -np.tile(np.arange(6, dtype=float)[:, None] * 50.0, (1, 6))
        out = script.consistency(own, x, y, 0.2)
        self.assertGreater(out["jumps"], 0.4)  # one of the two steps changes the size from 1 to 5 cells


class RunTests(unittest.TestCase):
    def test_runs_with_the_lead_tables(self):
        height, width, times = 6, 16, 10
        rows, cols = np.divmod(np.arange(height * width), width)
        grid = {"rows": rows.tolist(), "cols": cols.tolist(), "latitudes": np.linspace(50, 47.5, height).tolist(),
                "longitudes": np.linspace(5, 12.5, width).tolist()}
        timestamps = pd.date_range("2005-01-01", periods=times, freq="3h")
        moving = np.zeros((times, height, width))
        for t in range(times):
            moving[t, 2:4, t:t + 2] = 10
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            runs = []
            for k in range(4):
                scores = 0.01 * rng.random((times, height * width)) + (moving if k < 3 else -moving).reshape(times, -1)
                write_run(folder / f"d{k}", scores, timestamps, grid)
                runs.append(f"d{k}={folder / f'd{k}'}")
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                cube.main(["--runs", *runs, "--output", str(folder / "out"), "--top-percent", "12", "--votes", "3",
                           "--closure", "none", "--min-values", "5", "--max-hours", "100"])
                for motion in ("mean", "local"):
                    lead.main(["--cube", str(folder / "out"), "--first", "d0", "--kind", "analysed", "--radius-km", "300",
                               "--motion", motion, "--csv", str(folder / f"{motion}.csv")])
                script.main(["--cube", str(folder / "out"), "--kind", "analysed", "--first", "d0",
                             "--lead-mean", str(folder / "mean.csv"), "--lead-local", str(folder / "local.csv"),
                             "--csv", str(folder / "m.csv")])
            table = pd.read_csv(folder / "m.csv")
            big = table.loc[table["values"].idxmax()]
            self.assertGreater(big["coherence"], 0.9)  # the block moves east one cell a timestamp
            self.assertIn("local motion", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
