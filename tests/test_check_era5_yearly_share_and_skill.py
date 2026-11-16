import unittest

import numpy as np
import pandas as pd

from scripts.check_era5_yearly_share_and_skill import mask_rows, masked_mean, yearly_share


class YearlyShareTest(unittest.TestCase):
    def test_share_by_year(self):
        times = pd.DatetimeIndex(["2005-12-31 21:00", "2006-01-01 00:00", "2006-01-01 03:00"])
        scores = np.array([[1.0, 3.0], [3.0, np.nan], [0.0, 0.0]])
        table = yearly_share(scores, times, limit=2.0)
        self.assertEqual(table.year.tolist(), [2005, 2006])
        self.assertAlmostEqual(table.share_pct[0], 50.0)
        self.assertAlmostEqual(table.share_pct[1], 100 / 3)  # NaN left out of the count
        self.assertEqual(table["values"].tolist(), [2, 3])


class SkillTest(unittest.TestCase):
    def test_masked_mean(self):
        scores = np.array([[1.0, 2.0], [np.nan, 10.0]])
        keep = np.array([[True, True], [True, False]])
        self.assertEqual(masked_mean(scores, keep), (1.5, 2))

    def test_mask_rows(self):
        mask = np.arange(5)
        np.testing.assert_array_equal(mask_rows(mask, 3), [2, 3, 4])
        with self.assertRaises(ValueError):
            mask_rows(mask, 6)


if __name__ == "__main__":
    unittest.main()
