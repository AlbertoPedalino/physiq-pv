import unittest

import numpy as np
import pandas as pd
import xarray as xr

from physiq_pv.data.dataset import (
    FEATURE_NAMES,
    N_FEATURES,
    POA_INPUT_INDICES,
    PVDataset,
)


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
    @staticmethod
    def _quality(n_steps: int) -> dict[str, np.ndarray]:
        return {
            key: np.full((1, n_steps), 0.8)
            for key in ("m1", "m2", "m3", "m4", "m5")
        }

    def test_feature_schema_matches_poa_enabled_model(self) -> None:
        self.assertEqual(N_FEATURES, 16)
        self.assertEqual(len(FEATURE_NAMES), N_FEATURES)

    def test_encoder_features_do_not_change_with_poa_values(self) -> None:
        low_ds = _dataset(400.0)
        high_ds = _dataset(800.0)
        fit_mask = np.arange(low_ds.sizes["time"]) < 56
        quality = self._quality(low_ds.sizes["time"])
        low_poa = PVDataset(
            low_ds,
            quality,
            seq_len=24,
            fit_time_mask=fit_mask,
            include_poa_inputs=False,
        )
        high_poa = PVDataset(
            high_ds,
            quality,
            seq_len=24,
            fit_time_mask=fit_mask,
            include_poa_inputs=False,
        )

        np.testing.assert_allclose(low_poa.feats, high_poa.feats)
        np.testing.assert_allclose(
            low_poa.feats[..., POA_INPUT_INDICES],
            0.0,
        )
        self.assertFalse(
            np.allclose(low_poa.target_poa, high_poa.target_poa),
            "POA must remain an auxiliary target, not an encoder feature",
        )


if __name__ == "__main__":
    unittest.main()
