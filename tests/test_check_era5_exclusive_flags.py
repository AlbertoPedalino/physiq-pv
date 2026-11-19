"""Where the exclusive flags lie around a concordant cluster: distance from the cluster, attached or detached, and the
same along-track error of the exclusive flags as check_era5_concordant_lead.py."""
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
from scripts import check_era5_concordant_lead as lead
from scripts import check_era5_exclusive_flags as script
from tests.test_build_era5_concordant_cube import write_run


class PieceTests(unittest.TestCase):
    def test_distance_to_the_nearest_target_on_the_sphere(self):
        lat = np.radians(np.array([0.0, 0.0, 1.0]))
        lon = np.radians(np.array([0.0, 1.0, 0.0]))
        points = script.unit_vectors(lat, lon)
        km = script.nearest_km(points[1:], points[:1])
        one_degree = np.pi / 180 * script.EARTH_KM
        np.testing.assert_allclose(km, [one_degree, one_degree], rtol=1e-9)

    def test_attached_only_through_a_group_touching_the_cluster(self):
        flags = np.zeros((2, 1, 8), dtype=bool)
        own = np.zeros_like(flags)
        own[:, 0, 1] = True
        flags[:, 0, 1:3] = True  # on the cluster and the cell next to it: one group
        flags[1, 0, 6] = True  # a patch apart
        linked = script.attached(flags, own)
        self.assertTrue(linked[:, 0, 2].all())
        self.assertFalse(linked[1, 0, 6])


class RunTests(unittest.TestCase):
    def test_the_column_ahead_is_attached_and_the_control_matches_the_lead_script(self):
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
            inputs = np.zeros((times, height * width, 2), dtype=np.float32)  # "a" on the block, "b" on the column ahead
            inputs[:, :, 0] = (3 * (moving > 0)).reshape(times, -1)
            inputs[:, :, 1] = (3 * (ahead > 0)).reshape(times, -1)
            (folder / "cache").mkdir()
            np.save(folder / "cache" / "test.npy", inputs)
            (folder / "model").mkdir()
            (folder / "model" / "metadata.json").write_text(json.dumps(
                {"backend": {"seasonal_normalization": {"mean_file": str(folder / "cache" / "mean.npy")}},
                 "scored_period": "test", "preparation": {"features": ["a", "b"]}}), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                cube.main(["--runs", *runs, "--output", str(folder / "out"), "--top-percent", "12", "--votes", "3",
                           "--closure", "none", "--min-values", "5", "--max-hours", "100"])
                lead.main(["--cube", str(folder / "out"), "--first", "d0", "--kind", "analysed", "--exclusive",
                           "--radius-km", "300", "--csv", str(folder / "lead.csv")])
                script.main(["--cube", str(folder / "out"), "--kind", "analysed", "--radius-km", "300",
                             "--inputs-run", str(folder / "model"), "--csv", str(folder / "x.csv")])
            mine = pd.read_csv(folder / "x.csv")
            theirs = pd.read_csv(folder / "lead.csv")
            big = mine.loc[mine["values"].idxmax(), "cluster"]
            d0 = mine[(mine["detector"] == "d0") & (mine["cluster"] == big)].iloc[0]
            # The column ahead touches the block (attached); the noise flagged by d0 alone elsewhere is detached.
            self.assertGreater(d0["attached"], 0)
            self.assertEqual(d0["attached"] + d0["detached"], d0["exclusive"])
            self.assertGreater(d0["along_attached_km"], 20)
            self.assertGreater(d0["band_0_55"], 0)  # the column ahead is one cell (about 50 km) from the cluster
            self.assertEqual(d0["type_cluster"], "a")  # the block is far from normal in "a"
            self.assertEqual(d0["type_exclusive"], "b")  # what d0 alone flags, the column ahead, in "b"
            merged = mine.merge(theirs, on=["cluster", "detector"], suffixes=("", "_lead"))
            both = merged.dropna(subset=["along_alone_km", "along_alone_km_lead"])
            self.assertGreater(len(both), 0)
            np.testing.assert_allclose(both["along_alone_km"], both["along_alone_km_lead"])


if __name__ == "__main__":
    unittest.main()
