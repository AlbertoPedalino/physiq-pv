import unittest

import numpy as np

from scripts.sample_era5_score_ranks import pearson, scaled_ranks


class ScaledRanksTest(unittest.TestCase):
    def test_ranks_span_zero_to_one_and_keep_order(self):
        ranks = scaled_ranks([3.0, 1.0, 2.0])
        np.testing.assert_allclose(ranks, [1.0, 0.0, 0.5])

    def test_ties_are_averaged(self):
        np.testing.assert_allclose(scaled_ranks([1.0, 1.0, 2.0]), [0.25, 0.25, 1.0])

    def test_non_finite_values_get_no_rank(self):
        ranks = scaled_ranks([np.nan, 5.0, 1.0])
        self.assertTrue(np.isnan(ranks[0]))
        np.testing.assert_allclose(ranks[1:], [1.0, 0.0])

    def test_pearson_of_ranks_is_spearman(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=200)
        b = np.exp(a) + 0.01 * rng.normal(size=200)  # monotone: Spearman near 1, Pearson lower
        self.assertGreater(pearson(scaled_ranks(a), scaled_ranks(b)), 0.99)


if __name__ == "__main__":
    unittest.main()
