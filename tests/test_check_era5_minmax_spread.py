import unittest

import numpy as np

from scripts.check_era5_minmax_spread import describe, find_ranges, scaled_sample


class MinmaxSpreadTest(unittest.TestCase):
    def test_find_ranges_anywhere(self):
        meta = {"a": 1, "score": {"normalization": {"r_min": 0, "r_max": 2, "d_min": -1, "d_max": 1, "x": 3}}}
        found = find_ranges(meta)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0][0], "metadata.score.normalization")
        self.assertEqual(found[0][1], {"r_min": 0.0, "r_max": 2.0, "d_min": -1.0, "d_max": 1.0})

    def test_scaled_sample(self):
        scores = np.array([[0.0, 2.0], [1.0, np.nan], [2.0, 1.0]])
        keep = np.array([[True, False], [True, True], [True, True]])
        everything, kept = scaled_sample(scores, 0.0, 2.0, keep, stride=2)  # rows 0 and 2
        np.testing.assert_array_equal(np.sort(everything), [0.0, 0.5, 1.0, 1.0])
        np.testing.assert_array_equal(np.sort(kept), [0.0, 0.5, 1.0])

    def test_describe(self):
        stats = describe(np.arange(101, dtype=float))
        self.assertAlmostEqual(stats["p50"], 50.0)
        self.assertAlmostEqual(stats["iqr"], 50.0)


if __name__ == "__main__":
    unittest.main()
