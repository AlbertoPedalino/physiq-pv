"""The closing keeps every flag and fills the holes; the concordant cube, its deciding votes and its clusters on toy runs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import build_era5_concordant_cube as script


def write_run(directory, scores, timestamps, grid):
    (directory / "scores").mkdir(parents=True)
    np.save(directory / "scores" / "anomaly_mean.npy", scores.astype(np.float32))
    np.save(directory / "test_timestamps.npy", timestamps.as_unit("ns").asi8)
    (directory / "metadata.json").write_text(json.dumps({"status": "complete", "grid": grid}), encoding="utf-8")


class ClosingTests(unittest.TestCase):
    def test_keeps_the_flags_at_the_edges_and_fills_a_hole(self):
        cube = np.zeros((5, 4, 4), dtype=bool)
        cube[:, 0, :] = True  # a row at the edge of the domain, over the whole period
        cube[2, 2, 1] = cube[2, 2, 3] = True  # a gap of one cell between two flags
        closed = script.close(cube, 1, 1, [(0, 5)])
        self.assertTrue(closed[cube].all())  # never removes a flag, at the edges either
        self.assertTrue(closed[2, 2, 2])  # the hole is closed
        self.assertFalse(closed[2, 3, 0])  # nothing far from the flags

    def test_does_not_cross_a_gap_in_the_timestamps(self):
        cube = np.zeros((4, 3, 3), dtype=bool)
        cube[1, 1, 1] = cube[3, 1, 1] = True  # one timestamp apart, but timestamps 2 and 3 are in different runs
        closed = script.close(cube, 1, 0, [(0, 2), (2, 4)])
        self.assertFalse(closed[2, 1, 1])
        self.assertTrue(script.close(cube, 1, 0, [(0, 4)])[2, 1, 1])

    def test_segments_split_at_a_missing_timestamp(self):
        times = pd.DatetimeIndex(["2005-01-01 00:00", "2005-01-01 03:00", "2005-01-01 09:00", "2005-01-01 12:00"])
        self.assertEqual(script.segments(times), [(0, 2), (2, 4)])


class CubeTests(unittest.TestCase):
    def test_vote_deciding_votes_and_clusters(self):
        height, width, times = 6, 6, 12
        rows, cols = np.divmod(np.arange(height * width), width)
        grid = {"rows": rows.tolist(), "cols": cols.tolist(), "latitudes": np.linspace(50, 47.5, height).tolist(),
                "longitudes": np.linspace(5, 7.5, width).tolist()}
        timestamps = pd.date_range("2005-01-01", periods=times, freq="3h")
        block = np.zeros((times, height, width))
        block[3:6, 1:3, 1:3] = 10  # a block that three detectors see; each detector has its own low noise
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            runs = []
            for k in range(4):
                scores = rng.random((times, height * width)) + (0 if k == 3 else block.reshape(times, -1))
                write_run(folder / f"d{k}", scores, timestamps, grid)
                runs.append(f"d{k}={folder / f'd{k}'}")
            script.main(["--runs", *runs, "--output", str(folder / "out"), "--top-percent", "5", "--votes", "3",
                         "--closure", "none", "--pad-cells", "1", "--pad-hours", "3", "--min-values", "5"])
            summary = json.loads((folder / "out" / "summary.json").read_text(encoding="utf-8"))
            table = pd.read_csv(folder / "out" / "clusters.csv")
            self.assertEqual(summary["concordant_percent"], summary["concordant_without_closing_percent"])
            big = table.loc[table["values"].idxmax()]
            self.assertEqual((big["t0"], big["t1"], big["row0"], big["row1"], big["col0"], big["col1"]), (3, 5, 1, 2, 1, 2))
            self.assertEqual((big["pad_t0"], big["pad_t1"], big["pad_row0"], big["pad_row1"]), (2, 6, 0, 3))
            self.assertEqual(big["values"], 12)
            self.assertEqual(big["kind"], "analysed")
            # The deciding votes of every detector, counted again from the files written: exactly three votes, one its own.
            votes = np.load(folder / "out" / "votes" / "scores" / "anomaly_mean.npy")
            total = (votes >= 3).sum()
            self.assertAlmostEqual(summary["concordant_percent"], 100 * total / votes.size)
            for k in range(4):
                own = np.unpackbits(np.load(folder / "out" / "flags" / f"d{k}.npy"), axis=1, count=height * width)
                deciding = ((votes == 3) & own.astype(bool)).sum()
                self.assertAlmostEqual(summary["detectors"][f"d{k}"]["deciding_percent_of_cube"], 100 * deciding / total)
            self.assertGreater(summary["detectors"]["d0"]["deciding_percent_of_cube"],
                               summary["detectors"]["d3"]["deciding_percent_of_cube"])
            self.assertTrue((votes.reshape(times, height, width)[3:6, 1:3, 1:3] >= 3).all())
            flags = np.unpackbits(np.load(folder / "out" / "flags" / "d0.npy"), axis=1, count=height * width).astype(bool)
            self.assertTrue(flags.reshape(times, height, width)[3:6, 1:3, 1:3].all())


if __name__ == "__main__":
    unittest.main()
