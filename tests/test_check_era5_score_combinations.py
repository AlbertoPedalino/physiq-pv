import unittest

import numpy as np

from scripts.check_era5_score_combinations import combinations, compare


class CombinationsTest(unittest.TestCase):
    def test_follow_the_larger_spread(self):
        rng = np.random.default_rng(0)
        g = rng.normal(0, 0.001, 5000)
        d = rng.normal(0, 1.0, 5000)
        rows = compare(g, d, (g.min(), g.max(), d.min(), d.max()), 5)
        self.assertGreater(rows["anogan"]["spearman_D"], 0.9)  # raw sum: D's scale wins
        self.assertLess(abs(rows["rank_mean"]["spearman_G"] - rows["rank_mean"]["spearman_D"]), 0.1)

    def test_shapes(self):
        g, d = np.arange(10.0), np.arange(10.0)[::-1]
        scores = combinations(g, d, (0, 9, 0, 9))
        self.assertEqual(set(scores), {"stgan", "anogan", "tadgan_mean", "tadgan_product", "rank_mean"})
        np.testing.assert_allclose(scores["stgan"], np.ones(10))


if __name__ == "__main__":
    unittest.main()
