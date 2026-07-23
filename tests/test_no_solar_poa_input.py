import unittest

import numpy as np
import pandas as pd
import xarray as xr

from physiq_pv.data.dataset import FEATURE_NAMES, N_FEATURES, PVDataset


def _dataset(poa_wm2: float) -> xr.Dataset:
    times = pd.date_range("2019-06-01", periods=72, freq="h").to_numpy()
    hour = np.arange(len(times), dtype=float)
    production = np.maximum(0.0, np.sin((hour % 24 - 6.0) * np.pi / 12.0))
    return xr.Dataset(
        data_vars={
            "ENERGIA": (("plant", "time"), production[None, :]),
            "temperature_2m": (
                ("plant", "time"),
                np.full((1, len(times)), 20.0),
            ),
            "wind_speed_10m": (
                ("plant", "time"),
                np.full((1, len(times)), 3.0),
            ),
            "solar_irradiance_poa": (
                ("plant", "time"),
                np.full((1, len(times)), poa_wm2),
            ),
        },
        coords={
            "plant": ["plant-0"],
            "time": times,
            "lat": ("plant", [45.0]),
            "lon": ("plant", [7.0]),
        },
    )


class NoSolarPoaInputTest(unittest.TestCase):
    def test_feature_schema_contains_no_irradiance_channels(self) -> None:
        self.assertEqual(
            FEATURE_NAMES,
            ("temp", "wind", "sin_elev", "cos_elev", "pv_lag"),
        )
        self.assertEqual(N_FEATURES, 5)

    def test_encoder_features_do_not_change_with_poa_values(self) -> None:
        low_poa = PVDataset(_dataset(400.0), seq_len=24)
        high_poa = PVDataset(_dataset(800.0), seq_len=24)

        np.testing.assert_allclose(low_poa.feats, high_poa.feats)
        self.assertFalse(
            np.allclose(low_poa.target_ghi, high_poa.target_ghi),
            "POA must remain an auxiliary target, not an encoder feature",
        )


if __name__ == "__main__":
    unittest.main()
