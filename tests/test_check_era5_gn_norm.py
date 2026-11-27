import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.check_era5_gn_norm import mean_spatial_spearman, shared_flags, threshold, write_sqrt


class GnNormTest(unittest.TestCase):
    def test_sqrt_is_written_elementwise(self):
        with tempfile.TemporaryDirectory() as folder:
            out = write_sqrt(np.array([[4.0, 9.0], [0.0, 16.0]]), Path(folder) / "s.npy", chunk=1)
            np.testing.assert_allclose(out, [[2, 3], [0, 4]])
            del out

    def test_threshold_and_shared_flags(self):
        a = np.arange(100.0).reshape(10, 10)
        b = a.copy()
        b[9, 9] = -1  # one top value of a is not a top value of b
        la, lb = threshold(a, 5), threshold(b, 5)
        n_a, n_b, n_both = shared_flags(a, b, la, lb, chunk=3)
        self.assertEqual(n_a, 5)
        self.assertEqual(n_both, 4)

    def test_spearman_of_a_monotone_transform_is_one(self):
        rng = np.random.default_rng(0)
        g = rng.random((4, 50))
        mean, low = mean_spatial_spearman(g, np.sqrt(g), range(4))
        self.assertAlmostEqual(mean, 1.0)
        self.assertAlmostEqual(low, 1.0)


if __name__ == "__main__":
    unittest.main()
