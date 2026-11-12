"""Precision and recall against the concordant cube: the cube's own threshold gives back its flags, the recall never
grows with the percentile, and the cube without a detector's vote is counted as in check_era5_concordant_boxes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import build_era5_concordant_cube as cube
from scripts import check_era5_concordant_pr as script


def write_run(directory, scores, timestamps, grid):
    (directory / "scores").mkdir(parents=True)
    np.save(directory / "scores" / "anomaly_mean.npy", scores.astype(np.float32))
    np.save(directory / "test_timestamps.npy", timestamps.as_unit("ns").asi8)
    (directory / "metadata.json").write_text(json.dumps({"status": "complete", "grid": grid}), encoding="utf-8")


class PrecisionRecallTests(unittest.TestCase):
    def run_toy(self, closure):
        height, width, times = 6, 7, 20
        rows, cols = np.divmod(np.arange(height * width), width)
        grid = {"rows": rows.tolist(), "cols": cols.tolist(), "latitudes": np.linspace(50, 47.5, height).tolist(),
                "longitudes": np.linspace(5, 8, width).tolist()}
        timestamps = pd.date_range("2005-01-01", periods=times, freq="3h")
        block = np.zeros((times, height, width))
        block[4:9, 1:4, 2:5] = 10  # three of the four detectors see it
        rng = np.random.default_rng(1)
        folder = Path(self.tmp.name) / closure
        runs = []
        for k in range(4):
            scores = rng.random((times, height * width)) + (0 if k == 3 else block.reshape(times, -1))
            scores[0, 0] = np.nan  # a value one detector does not score
            write_run(folder / f"d{k}", scores, timestamps, grid)
            runs.append(f"d{k}={folder / f'd{k}'}")
        out = folder / f"cube_{closure}"
        cube.main(["--runs", *runs, "--output", str(out), "--top-percent", "10", "--votes", "3", "--closure", closure,
                   "--pad-cells", "1", "--pad-hours", "3", "--min-values", "5"])
        script.main(["--cube", str(out), "--percentiles", "50", "80", "95", "--csv", str(folder / f"pr_{closure}.csv")])
        return out, pd.read_csv(folder / f"pr_{closure}.csv"), height * width

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_cube_threshold_matches_the_cube(self):
        for closure in ("none", "before"):
            out, table, cells = self.run_toy(closure)
            votes = np.load(out / "votes" / "scores" / "anomaly_mean.npy")
            valid = np.isfinite(votes)
            concordant = valid & (np.nan_to_num(votes) >= 3)
            self.assertEqual(sorted(table["percentile"].unique()), [50, 80, 90, 95])  # the cube's 90 is added
            for k in range(4):
                own = np.unpackbits(np.load(out / "flags" / f"d{k}.npy"), axis=1, count=cells).astype(bool)
                row = table[(table["detector"] == f"d{k}") & table["cube"]].iloc[0]
                self.assertEqual(row["percentile"], 90)
                self.assertEqual(row["flagged"], own.sum())
                self.assertAlmostEqual(row["precision"], (own & concordant).sum() / own.sum())
                self.assertAlmostEqual(row["recall"], (own & concordant).sum() / concordant.sum())
                others = valid & ((np.nan_to_num(votes) - own) >= 3)
                self.assertEqual(row["reference_without_own"], others.sum())
                self.assertAlmostEqual(row["recall_without_own"], (own & others).sum() / others.sum())
                recall = table[table["detector"] == f"d{k}"].sort_values("percentile")["recall"].to_numpy()
                self.assertTrue((np.diff(recall) <= 1e-12).all())  # nested flags: the recall never grows

    def test_detector_that_does_not_see_the_block_has_a_lower_recall(self):
        _, table, _ = self.run_toy("none")
        at_cube = table[table["cube"]].set_index("detector")
        self.assertLess(at_cube.loc["d3", "recall"], at_cube.loc["d0", "recall"])
        # Without its vote the three detectors that see the block lose the cube (only two other votes are left).
        self.assertTrue((at_cube.loc[["d0", "d1", "d2"], "reference_without_own"] < at_cube.loc["d0", "reference"]).all())


if __name__ == "__main__":
    unittest.main()
