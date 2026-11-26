import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from physiq_pv.anomaly_detection.stgan.mmd import multiscale_mmd2
from physiq_pv.anomaly_detection.stgan.normal_masks import normal_rows
from physiq_pv.anomaly_detection.stgan.shared_cache import MANIFEST
from scripts.check_era5_mmd_subset_size import PairedMMD, normal_positions, remap, run_cache


class PairedMMDTest(unittest.TestCase):
    def test_reproduces_the_pipeline_estimator_on_any_subset(self):
        rng = np.random.default_rng(3)
        real = rng.normal(size=(60, 7))
        fake = real + rng.normal(scale=0.4, size=real.shape) + 0.2
        distance = PairedMMD(real, fake, sigma=2.5, device="cpu")
        for subset in (np.arange(60), np.arange(0, 60, 3), rng.choice(60, 17, replace=False)):
            self.assertAlmostEqual(distance(subset), multiscale_mmd2(real[subset], fake[subset], 2.5), places=12)

    def test_same_distribution_is_near_zero_and_a_shift_is_not(self):
        rng = np.random.default_rng(4)
        real, other = rng.normal(size=(400, 5)), rng.normal(size=(400, 5))
        same = PairedMMD(real, other, 2.0, "cpu")(np.arange(400))
        shifted = PairedMMD(real, other + 1.0, 2.0, "cpu")(np.arange(400))
        self.assertLess(abs(same), 0.01)
        self.assertGreater(shifted, 10 * abs(same))


class RemapTest(unittest.TestCase):
    def test_maps_only_missing_paths_with_a_matching_prefix(self):
        here = Path(__file__).resolve().parent
        self.assertEqual(remap(here, [("/nowhere", "/else")]), here)
        self.assertEqual(remap("/home/user/data/cache", [("/home/user", "/data2/homes/user")]),
                         Path("/data2/homes/user/data/cache"))
        self.assertEqual(remap("/other/cache", [("/home/user", "/data2/homes/user")]), Path("/other/cache"))
        self.assertIsNone(remap(None, []))


class RunCacheTest(unittest.TestCase):
    def test_reuses_the_run_cache_only_when_inputs_differ_by_moved_paths_and_times(self):
        saved = {"version": 1, "sources": [{"file": "/home/u/data/train.npy", "bytes": 10, "modified_ns": 1}]}
        moved = {"version": 1, "sources": [{"file": "/data2/u/data/train.npy", "bytes": 10, "modified_ns": 2}]}
        path_map = [("/home/u", "/data2/u")]
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / MANIFEST).write_text(json.dumps({"inputs": saved, "built": {"k": 1}}), encoding="utf-8")
            use = run_cache(directory, path_map)
            never = lambda target: self.fail("no cache may be built")
            self.assertEqual(use("/root", moved, never), (Path(directory), {"k": 1}, True))
            with self.assertRaises(ValueError):
                use("/root", {**moved, "version": 2}, never)
            with self.assertRaises(ValueError):
                use("/root", {"version": 1, "sources": [{**moved["sources"][0], "bytes": 11}]}, never)


class NormalPositionsTest(unittest.TestCase):
    def test_contains_every_field_the_sweep_could_draw(self):
        timestamps = pd.date_range("2000-01-01", "2004-12-31 21:00", freq="3h")
        share = np.random.default_rng(5).random(len(timestamps))
        normal = normal_positions(timestamps, share)
        drawn = normal_rows(timestamps, share, 2560, seed=0)
        self.assertTrue(np.isin(drawn, normal).all())
        # At most the median of every calendar month: at least half of each month.
        for month in range(1, 13):
            in_month = timestamps.month == month
            self.assertGreaterEqual(np.isin(np.nonzero(in_month)[0], normal).sum(), in_month.sum() // 2)


if __name__ == "__main__":
    unittest.main()
