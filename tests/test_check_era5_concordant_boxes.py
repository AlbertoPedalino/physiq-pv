"""The zone within a radius in km (--radius-km): for every cluster it holds exactly the cells, at the timestamps of the
cluster only, whose nearest labelled cell (great-circle distance, ties to the lowest cluster) is a cell of that
cluster and lies closer than the radius; checked against a brute-force computation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import build_era5_concordant_cube as cube
from scripts import check_era5_concordant_boxes as script
from scripts.era5_km_zone import haversine_km


def write_run(directory, scores, timestamps, grid):
    (directory / "scores").mkdir(parents=True)
    np.save(directory / "scores" / "anomaly_mean.npy", scores.astype(np.float32))
    np.save(directory / "test_timestamps.npy", timestamps.as_unit("ns").asi8)
    (directory / "metadata.json").write_text(json.dumps({"status": "complete", "grid": grid}), encoding="utf-8")


class RadiusZoneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_zone_matches_brute_force(self):
        height, width, times = 7, 9, 24
        rows, cols = np.divmod(np.arange(height * width), width)
        latitudes = np.linspace(50, 47, height)
        longitudes = np.linspace(5, 9, width)
        grid = {"rows": rows.tolist(), "cols": cols.tolist(), "latitudes": latitudes.tolist(),
                "longitudes": longitudes.tolist()}
        timestamps = pd.date_range("2005-01-01", periods=times, freq="3h")
        block = np.zeros((times, height, width))
        block[4:10, 1:4, 1:4] = 10  # a cluster seen by three of four detectors
        block[6:12, 3:6, 5:8] = 10  # a second one, close to the first
        rng = np.random.default_rng(3)
        root = Path(self.tmp.name)
        runs = []
        for k in range(4):
            scores = rng.random((times, height * width)) + (0 if k == 3 else block.reshape(times, -1))
            write_run(root / f"d{k}", scores, timestamps, grid)
            runs.append(f"d{k}={root / f'd{k}'}")
        out = root / "cube"
        cube.main(["--runs", *runs, "--output", str(out), "--top-percent", "10", "--votes", "3", "--closure", "none",
                   "--pad-cells", "1", "--pad-hours", "3", "--min-values", "5"])
        radius = 100.0
        script.main(["--cube", str(out), "--kind", "all", "--radius-km", str(radius), "--csv", str(root / "b.csv")])
        table = pd.read_csv(root / "b.csv")
        clusters = pd.read_csv(out / "clusters.csv")
        self.assertGreaterEqual(len(clusters), 2)

        # Brute force: label every concordant value as the script does, then for every cell and timestamp the nearest
        # labelled cell closer than the radius.
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        votes = np.load(out / "votes" / "scores" / "anomaly_mean.npy")
        concordant = np.nan_to_num(votes) >= summary["vote"]["votes_needed"]
        labels, sizes = script.label_clusters(concordant, rows, cols, (height, width), summary["overlap"])
        lat_of = latitudes[:, None].repeat(width, 1)
        lon_of = longitudes[None, :].repeat(height, 0)
        expected = np.zeros(len(clusters), dtype=int)
        for t in range(times):
            labelled = np.argwhere(labels[t] > 0)
            if not len(labelled):
                continue
            for r in range(height):
                for c in range(width):
                    d = haversine_km(lat_of[r, c], lon_of[r, c], lat_of[labelled[:, 0], labelled[:, 1]],
                                     lon_of[labelled[:, 0], labelled[:, 1]])
                    close = d < radius
                    if not close.any():
                        continue
                    ids = labels[t][labelled[close, 0], labelled[close, 1]]
                    best = d[close].min()
                    owner = ids[d[close] == best].min()
                    expected[owner - 1] += 1
        got = table[table["detector"] == "d0"].set_index("cluster")["zone"]
        for cluster in clusters["cluster"]:
            self.assertEqual(got.loc[cluster], expected[cluster - 1], f"cluster {cluster}")
        # The concordant values of a zone are those of its cluster, and no zone reaches outside its timestamps.
        self.assertTrue((table["concordant"] == table["values"]).all())


if __name__ == "__main__":
    unittest.main()
