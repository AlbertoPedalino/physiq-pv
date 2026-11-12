"""What is in the excess of a detector: positions against a moving cluster, the inputs of the excess and the reference."""
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
from scripts import check_era5_concordant_excess as script
from tests.test_build_era5_concordant_cube import write_run


class PositionTests(unittest.TestCase):
    def test_before_ahead_behind_after(self):
        own = np.zeros((5, 1, 8), dtype=bool)
        for t in range(1, 4):
            own[t, 0, t + 2] = True  # moving east, timestamps 1..3
        x = np.arange(8, dtype=float)[None, :] * 50.0
        y = np.zeros((1, 8))
        cells = (np.array([0, 2, 2, 4]), np.array([0, 0, 0, 0]), np.array([3, 6, 2, 5]))
        self.assertEqual(script.positions_of(cells, own, x, y, 1, 3).tolist(), [0, 1, 2, 3])

    def test_judge_years_need_twelve_months(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for year, months in ((2005, range(1, 13)), (2006, range(1, 12))):
                (root / str(year)).mkdir()
                for month in months:
                    suffix = "__stepType-instant" if month == 3 else ""  # a month split by the download
                    (root / str(year) / f"era5_judges_{year}_{month:02d}{suffix}.nc").write_bytes(b"")
            self.assertEqual(script.judge_years([root]), [2005])


class ExcessTests(unittest.TestCase):
    def test_the_excess_behind_holds_the_departure(self):
        height, width, times, variables = 6, 16, 10, 3
        rows, cols = np.divmod(np.arange(height * width), width)
        grid = {"rows": rows.tolist(), "cols": cols.tolist(), "latitudes": np.linspace(50, 47.5, height).tolist(),
                "longitudes": np.linspace(5, 12.5, width).tolist()}
        timestamps = pd.date_range("2005-01-01", periods=times, freq="3h")
        moving = np.zeros((times, height, width))
        behind = np.zeros_like(moving)
        for t in range(times):  # a block of 2 x 2 cells moving east; d0 also flags the column it has just left
            moving[t, 2:4, t + 2:t + 4] = 10
            behind[t, 2:4, t + 1] = 10
        rng = np.random.default_rng(2)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            runs = []
            for k in range(4):
                signal = moving + behind if k == 0 else moving if k < 3 else -moving - behind
                scores = 0.01 * rng.random((times, height * width)) + signal.reshape(times, -1)
                write_run(folder / f"d{k}", scores, timestamps, grid)
                runs.append(f"d{k}={folder / f'd{k}'}")
            # Standardised inputs: far from normal in the block and in the column left behind, normal elsewhere.
            inputs = np.zeros((times, height * width, variables), dtype=np.float32)
            inputs[:, :, 0] = 3 * ((moving + behind) > 0).reshape(times, -1)
            (folder / "cache").mkdir()
            np.save(folder / "cache" / "test.npy", inputs)
            (folder / "model").mkdir()
            (folder / "model" / "metadata.json").write_text(json.dumps(
                {"backend": {"seasonal_normalization": {"mean_file": str(folder / "cache" / "mean.npy")}},
                 "scored_period": "test", "preparation": {"features": ["a", "b", "c"]}}), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                cube.main(["--runs", *runs, "--output", str(folder / "out"), "--top-percent", "12", "--votes", "3",
                           "--closure", "none", "--min-values", "5", "--max-hours", "100", "--pad-cells", "3",
                           "--pad-hours", "6"])
                script.main(["--cube", str(folder / "out"), "--detectors", "d0", "--inputs-run", str(folder / "model"),
                             "--hours-after", "0", "3", "--reference-share", "1", "--csv", str(folder / "x.csv")])
            table = pd.read_csv(folder / "x.csv")
            now = table[table["hours"] == 0]
            d0 = now[now["detector"] == "d0"].groupby("position")["values"].sum()
            # The column left behind is excess behind the cluster; ahead there is only the noise d0 flags too.
            self.assertGreater(d0.get("behind", 0), d0.get("ahead", 0))

            def z(det, pos):
                part = now[(now["detector"] == det) & (now["position"] == pos)]
                return np.average(part["max |z|"], weights=part["values"])

            self.assertGreater(z("d0", "behind"), 1.5)
            self.assertLess(z("d0", "ahead"), 0.5)
            self.assertLess(z("reference", "behind"), 0.5)

            # A judge no score reads, high where the column is left behind: the excess behind holds it, the reference not.
            import xarray as xr
            judge_root = folder / "judges"
            for month in range(1, 13):
                (judge_root / "2005").mkdir(parents=True, exist_ok=True)
                if month == 1:
                    times_j = timestamps
                    field = rng.random((times, height, width)).astype(np.float32)
                    field[behind > 0] = 100.0
                else:  # the other months exist (twelve files a year), outside the period
                    times_j = pd.date_range(f"2005-{month:02d}-01", periods=1, freq="3h")
                    field = np.zeros((1, height, width), dtype=np.float32)
                xr.Dataset({"cape": (("valid_time", "latitude", "longitude"), field)},
                           coords={"valid_time": times_j, "latitude": grid["latitudes"],
                                   "longitude": grid["longitudes"]}).to_netcdf(
                    judge_root / "2005" / f"era5_independent_2005_{month:02d}.nc")
            # d0 at different probabilities: a spread so that its flags fall in several classes.
            spread = np.full((times, height * width), 5.0, dtype=np.float32)
            np.save(folder / "d0" / "scores" / "anomaly_std.npy", spread)
            manifest = json.loads((folder / "d0" / "metadata.json").read_text(encoding="utf-8"))
            manifest["anomaly_std_file"] = "scores/anomaly_std.npy"
            (folder / "d0" / "metadata.json").write_text(json.dumps(manifest), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                script.main(["--cube", str(folder / "out"), "--detectors", "d0", "--inputs-run", str(folder / "model"),
                             "--hours-after", "0", "--reference-share", "1", "--probability", "d0",
                             "--unseen-dir", str(judge_root), "--unusual-percent", "20", "--csv", str(folder / "y.csv")])
            judged = pd.read_csv(folder / "y_giudici.csv")
            share = judged.set_index(["detector", "position"])["unusual %"]
            self.assertGreater(share[("d0", "behind")], share[("reference", "behind")])
            self.assertIn("by class of the probability", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
