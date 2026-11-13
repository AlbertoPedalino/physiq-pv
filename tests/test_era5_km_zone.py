"""The zone within 300 km at the same timestamp: distances, the width of the disc with latitude, the nearest cluster and
its ties, the radius as a strict bound."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts import era5_km_zone as zone


class KmZoneTests(unittest.TestCase):
    def test_haversine(self):
        self.assertAlmostEqual(zone.haversine_km(45.0, 10.0, 46.0, 10.0), 111.195, places=2)
        self.assertAlmostEqual(zone.haversine_km(60.0, 0.0, 60.0, 0.5), 27.80, places=1)

    def test_disc_widens_towards_the_pole(self):
        latitudes = 60 - 0.5 * np.arange(81)  # 60N to 20N, north first, as the ERA5 grid of the thesis
        offsets = zone.row_offsets(latitudes, 0.5, 300.0)
        at_60, at_20 = offsets[0], offsets[80]
        self.assertEqual(int(at_60[at_60[:, 0] == 0][:, 1].max()), 10)  # 10 x 27.8 km = 278 km < 300 < 11 x 27.8
        self.assertEqual(int(at_20[at_20[:, 0] == 0][:, 1].max()), 5)  # 5 x 52.2 km = 261 km < 300 < 6 x 52.2
        self.assertEqual(int(np.abs(offsets[40][:, 0]).max()), 5)  # 5 x 55.6 km = 278 km < 300 < 6 x 55.6
        self.assertTrue((offsets[40][:, 2] < 300).all())
        self.assertEqual(tuple(offsets[40][0]), (0.0, 0.0, 0.0))  # the cell itself first
        self.assertTrue((np.diff(offsets[40][:, 2]) >= 0).all())
        self.assertTrue((at_60[:, 0] >= 0).all())  # no row north of the first

    def test_nearest_cluster_and_ties(self):
        latitudes = np.full(1, 45.0)  # one row: distances along it only, 0.5 deg = 39.3 km at 45N
        offsets = zone.row_offsets(latitudes, 0.5, 100.0)  # dc up to 2 (78.6 km)
        labels = np.zeros((2, 1, 12), dtype=np.int32)
        labels[0, 0, 2] = 3
        labels[0, 0, 6] = 7
        labels[1, 0, 9] = 5
        owner = zone.nearest_within(labels, offsets)
        self.assertEqual(owner[0, 0, 3], 3)  # 1 cell from 3, 3 cells (too far) from 7
        self.assertEqual(owner[0, 0, 4], 3)  # 2 cells from both: tie to the lower label
        self.assertEqual(owner[0, 0, 5], 7)  # 3 cells from 3 (out of radius), 1 from 7
        self.assertEqual(owner[0, 0, 6], 7)  # its own cell
        self.assertEqual(owner[0, 0, 8], 7)
        self.assertEqual(owner[0, 0, 9], 0)  # 3 cells from 7: 118 km, beyond 100 km
        self.assertEqual(owner[1, 0].tolist(), [0] * 7 + [5] * 5)  # the other timestamp: only cluster 5
        self.assertEqual(owner[0, 0, 0], 3)

    def test_nearest_against_brute_force_on_real_latitudes(self):
        rng = np.random.default_rng(7)
        for latitudes in (60 - 0.5 * np.arange(12), 25.5 - 0.5 * np.arange(12)):
            height, width, radius = len(latitudes), 15, 120.0
            longitudes = 0.5 * np.arange(width)
            labels = ((rng.random((2, height, width)) < 0.06) * rng.integers(1, 8, (2, height, width))).astype(np.int32)
            owner = zone.nearest_within(labels, zone.row_offsets(latitudes, 0.5, radius))
            for t in range(2):
                cells = np.argwhere(labels[t] > 0)
                for r in range(height):
                    for c in range(width):
                        d = zone.haversine_km(latitudes[r], longitudes[c], latitudes[cells[:, 0]], longitudes[cells[:, 1]])
                        close = d < radius
                        if not close.any():
                            self.assertEqual(owner[t, r, c], 0)
                            continue
                        ids = labels[t][cells[close, 0], cells[close, 1]]
                        self.assertEqual(owner[t, r, c], ids[d[close] == d[close].min()].min(), (t, r, c))

    def test_within_is_a_strict_bound(self):
        latitudes = np.full(1, 45.0)
        two_cells_km = zone.haversine_km(45.0, 0.0, 45.0, 1.0)  # not twice one cell: a parallel is no great circle
        offsets = zone.row_offsets(latitudes, 0.5, two_cells_km)  # exactly two cells away is not "closer than"
        mask = np.zeros((1, 1, 7), dtype=bool)
        mask[0, 0, 3] = True
        self.assertEqual(zone.within(mask, offsets)[0, 0].tolist(), [False, False, True, True, True, False, False])


if __name__ == "__main__":
    unittest.main()
