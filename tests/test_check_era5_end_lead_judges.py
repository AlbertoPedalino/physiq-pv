"""The judges of the end lead check: the side of each variable and the clearness of the sky."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import xarray as xr

from scripts import check_era5_end_lead as script

TIMES = pd.date_range("2005-01-01", periods=248, freq="3h")  # January 2005
HEIGHT, WIDTH = 3, 4


class JudgesTests(unittest.TestCase):
    def test_sides_and_clearness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "judges"
            (root / "2005").mkdir(parents=True)
            latitudes, longitudes = 60 - 0.5 * np.arange(HEIGHT), 0.5 * np.arange(WIDTH)
            cells = HEIGHT * WIDTH
            rng = np.random.default_rng(0)
            minimum = rng.normal(270, 3, (len(TIMES), cells)).astype(np.float32)
            minimum[10] = 200  # a very cold timestamp: it must be unusual on the low side
            maximum = rng.normal(280, 3, (len(TIMES), cells)).astype(np.float32)
            day = np.isin(TIMES.hour, (9, 12, 15))
            clear = np.where(day[:, None], 2e6, 0).astype(np.float32) * np.ones((1, cells), dtype=np.float32)
            radiation = (clear * rng.uniform(0.6, 1.0, clear.shape)).astype(np.float32)
            radiation[12 * 8 + 4] = 0.05 * clear[12 * 8 + 4]  # an overcast noon: unusual clearness on the low side
            fields = {"mn2t": minimum, "mx2t": maximum, "ssrdc": clear}
            xr.Dataset({name: (("valid_time", "latitude", "longitude"), field.reshape(-1, HEIGHT, WIDTH))
                        for name, field in fields.items()},
                       coords={"valid_time": TIMES, "latitude": latitudes, "longitude": longitudes}
                       ).to_netcdf(root / "2005" / "era5_judges_2005_01.nc")
            positions = np.arange(len(TIMES))
            cell_lat = np.repeat(latitudes, WIDTH)
            cell_lon = np.tile(longitudes, HEIGHT)
            labels, unusual, defined = script.judges_of_month([root], 1, positions, TIMES, cell_lat, cell_lon, 5,
                                                              {"mn2t"}, set(), radiation)
            self.assertEqual(labels, ["mn2t lowest 5%", "mx2t highest 5%", "clearness lowest 5%"])
            self.assertTrue(unusual[10, :, 0].all())  # the cold timestamp
            self.assertTrue(unusual[12 * 8 + 4, :, 2].all())  # the overcast noon
            self.assertFalse(defined[0, :, 2].any())  # midnight: no clearness
            self.assertTrue(defined[12 * 8 + 4, :, 2].all())


if __name__ == "__main__":
    unittest.main()
