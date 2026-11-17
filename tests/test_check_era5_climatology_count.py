import unittest

import numpy as np

from scripts.check_era5_climatology_count import summarize


class SummarizeTest(unittest.TestCase):
    def test_rules(self):
        total = np.array([[155, 155, 155, 0]])
        kept = np.array([[150, 120, 25, 0]])
        deviation = np.array([[1.0, 1.0, 1.0, np.nan]])
        share, no_now, no_share = summarize(kept, total, deviation, 30, 0.8)
        np.testing.assert_allclose(share[0, :3], [150 / 155, 120 / 155, 25 / 155])
        self.assertTrue(np.isnan(share[0, 3]))
        np.testing.assert_array_equal(no_now, [[False, False, True, False]])
        np.testing.assert_array_equal(no_share, [[False, True, True, False]])

    def test_no_spread(self):
        share, no_now, no_share = summarize(np.array([[150]]), np.array([[155]]), np.array([[0.0]]), 30, 0.8)
        self.assertTrue(no_now[0, 0] and no_share[0, 0])


if __name__ == "__main__":
    unittest.main()
