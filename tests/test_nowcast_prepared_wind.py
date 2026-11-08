"""The nowcast reads the wind at 850 hPa of the validation and test timestamps from the prepared cache."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from physiq_pv.era5.features import FEATURE_NAMES
from scripts import nowcast_era5_anomalies as script
from test_check_era5_known_events_lead import prepared


class PreparedWindTests(unittest.TestCase):
    def test_validation_and_test_timestamps(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            test_times = prepared(folder / "prepared", np.random.default_rng(0))
            columns = [FEATURE_NAMES.index("u_component_of_wind"), FEATURE_NAMES.index("v_component_of_wind")]
            validation = np.load(folder / "prepared" / "validation.npy")
            test = np.load(folder / "prepared" / "test.npy")
            validation_times = pd.date_range("2002-01-01", "2002-12-31 21:00", freq="3h")
            read = script.prepared_wind(folder / "prepared")
            try:
                asked = pd.DatetimeIndex([validation_times[5], test_times[7], validation_times[0]])
                wind = read(asked)
                self.assertEqual(wind.shape, (3, 100, 2))
                np.testing.assert_array_equal(wind[0], validation[5][:, columns])
                np.testing.assert_array_equal(wind[1], test[7][:, columns])
                with self.assertRaises(ValueError):
                    read(pd.DatetimeIndex(["1999-01-01"]))
            finally:
                read.close()  # the files of the cache, before the folder is removed (Windows)


if __name__ == "__main__":
    unittest.main()
