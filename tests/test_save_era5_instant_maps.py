"""The typical timestamp is the one whose correlation is the closest to the median."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts import save_era5_instant_maps as script


class TypicalTests(unittest.TestCase):
    def test_closest_to_the_median(self):
        self.assertEqual(script.typical_index([-0.5, -0.1, -0.24, -0.3, -0.2]), 2)  # median -0.24

    def test_earliest_of_equally_close_and_missing_values_ignored(self):
        self.assertEqual(script.typical_index([np.nan, 1.0, 3.0, 2.0, 2.0]), 3)  # median 2, positions 3 and 4


if __name__ == "__main__":
    unittest.main()
