import unittest

import numpy as np

from scripts.check_era5_g_tail_source import cell_values, find_key, find_weights, patch_mean


class GTailSourceTest(unittest.TestCase):
    def test_patch_mean_matches_masked_cell_mean_on_a_full_grid(self):
        rng = np.random.default_rng(0)
        shape = (4, 5)
        rows, cols = np.divmod(np.arange(20), 5)
        values = rng.normal(size=20)
        grid = values.reshape(shape)
        result = patch_mean(values, rows, cols, shape)
        self.assertAlmostEqual(result[0], grid[:2, :2].mean())  # corner: only the cells inside the grid
        self.assertAlmostEqual(result[6], grid[0:3, 0:3].mean())  # inner cell (1, 1)

    def test_patch_mean_ignores_cells_missing_from_the_grid(self):
        rows, cols = np.array([0, 0]), np.array([0, 1])
        np.testing.assert_allclose(patch_mean(np.array([1.0, 3.0]), rows, cols, (2, 2)), [2.0, 2.0])

    def test_absolute_version_uses_square_roots_of_errors_and_weights(self):
        errors = np.array([[4.0, 9.0]])
        weights = np.array([4.0, 1.0])
        self.assertAlmostEqual(cell_values(errors, weights, "square")[0], (16 + 9) / 2)
        self.assertAlmostEqual(cell_values(errors, weights, "absolute")[0], (2 * 2 + 3 * 1) / 2)

    def test_weights_and_keys_are_found_nested(self):
        meta = {"backend": {"reconstruction_feature_weights": {"a": 1.0, "b": 2.0}, "patch_size": 3}}
        self.assertEqual(list(find_weights(meta).values()), [1.0, 2.0])
        self.assertEqual(find_key(meta, "patch_size"), 3)
        self.assertIsNone(find_weights({"x": [1, 2]}))


if __name__ == "__main__":
    unittest.main()
