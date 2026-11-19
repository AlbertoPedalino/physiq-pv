"""Whether the exclusive flags ahead of a concordant cluster are reached by it later, and the land cells."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import build_era5_concordant_cube as cube
from scripts import check_era5_exclusive_reach as script
from tests.test_build_era5_concordant_cube import write_run


class PieceTests(unittest.TestCase):
    def test_reached_later_within_the_steps(self):
        own = np.zeros((5, 1, 4), dtype=bool)
        own[3, 0, 2] = True  # the cluster is at cell 2 only at timestamp 3
        later = script.reached(own, [1, 2])
        self.assertTrue(later[1][2, 0, 2])  # one step before
        self.assertFalse(later[1][1, 0, 2])  # two steps before: not within one step
        self.assertTrue(later[2][1, 0, 2])
        self.assertFalse(later[2][3, 0, 2])  # not later than itself

    def test_bands_of_distance(self):
        np.testing.assert_array_equal(script.band_of(np.array([10.0, 55.0, 120.0, 400.0])), [0, 1, 2, 5])


class RunTests(unittest.TestCase):
    def test_the_column_ahead_is_reached_and_the_land_is_where_the_soil_varies(self):
        height, width, times = 6, 16, 10
        rows, cols = np.divmod(np.arange(height * width), width)
        grid = {"rows": rows.tolist(), "cols": cols.tolist(), "latitudes": np.linspace(50, 47.5, height).tolist(),
                "longitudes": np.linspace(5, 12.5, width).tolist()}
        timestamps = pd.date_range("2005-01-01", periods=times, freq="3h")
        moving = np.zeros((times, height, width))
        ahead = np.zeros_like(moving)
        for t in range(times):  # a block moving east; d0 also flags the column it enters next
            moving[t, 2:4, t:t + 2] = 10
            ahead[t, 2:4, t + 2:t + 3] = 10
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            runs = []
            for k in range(4):
                signal = moving + ahead if k == 0 else moving if k < 3 else -moving - ahead
                scores = 0.01 * rng.random((times, height * width)) + signal.reshape(times, -1)
                write_run(folder / f"d{k}", scores, timestamps, grid)
                runs.append(f"d{k}={folder / f'd{k}'}")
            inputs = np.zeros((times, height * width, 3), dtype=np.float32)
            inputs[:, :, 0] = (3 * (moving > 0)).reshape(times, -1)
            inputs[:, :, 1] = (3 * (ahead > 0)).reshape(times, -1)
            soil = np.zeros((times, height, width), dtype=np.float32)
            soil[:, :, :8] = rng.normal(size=(times, height, 8))  # land on the western half
            inputs[:, :, 2] = soil.reshape(times, -1)
            (folder / "cache").mkdir()
            np.save(folder / "cache" / "test.npy", inputs)
            (folder / "model").mkdir()
            (folder / "model" / "metadata.json").write_text(json.dumps(
                {"backend": {"seasonal_normalization": {"mean_file": str(folder / "cache" / "mean.npy")}},
                 "scored_period": "test",
                 "preparation": {"features": ["a", "b", "volumetric_soil_water_layer_1"]}}), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as out:
                cube.main(["--runs", *runs, "--output", str(folder / "out"), "--top-percent", "12", "--votes", "3",
                           "--closure", "none", "--min-values", "5", "--max-hours", "100"])
                script.main(["--cube", str(folder / "out"), "--kind", "analysed", "--radius-km", "300", "--first", "d0",
                             "--hours", "3", "6", "--every", "1", "--inputs-run", str(folder / "model"),
                             "--csv", str(folder / "r.csv")])
            self.assertIn("land cells (the standardised volumetric_soil_water_layer_1 varies): 48 of 96", out.getvalue())
            rates = pd.read_csv(folder / "r.csv")
            d0 = rates[(rates["who"] == "d0") & (rates["hours"] == 3) & (rates["band"] == "0-55 km")]
            self.assertEqual(len(d0), 1)
            self.assertGreater(d0["reached %"].iloc[0], 80)  # the column ahead is in the cluster three hours later
            land = pd.read_csv(folder / "r_land.csv")
            self.assertTrue((land["on_land"] <= land["exclusive"]).all())

            # Every flag outside the cube: the column ahead is among them, and they are at least as many.
            with contextlib.redirect_stdout(io.StringIO()):
                script.main(["--cube", str(folder / "out"), "--kind", "analysed", "--radius-km", "300", "--first", "d0",
                             "--hours", "3", "--flags", "outside", "--csv", str(folder / "o.csv")])
            outside = pd.read_csv(folder / "o.csv")
            d0o = outside[(outside["who"] == "d0") & (outside["hours"] == 3) & (outside["band"] == "0-55 km")]
            self.assertGreater(d0o["reached %"].iloc[0], 80)
            self.assertGreaterEqual(d0o["values"].iloc[0], d0["values"].iloc[0])


if __name__ == "__main__":
    unittest.main()
