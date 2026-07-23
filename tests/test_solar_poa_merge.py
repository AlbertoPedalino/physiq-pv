import numpy as np
import tempfile
import unittest
import xarray as xr
from pathlib import Path

from physiq_pv.data.sentinel_hourly_loader import merge_with_weather


def _sentinel_dataset(times: np.ndarray) -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "ENERGIA": (("plant", "time"), np.zeros((1, len(times)))),
        },
        coords={
            "plant": ["plant-0"],
            "time": times,
            "latitude": ("plant", [45.0]),
            "longitude": ("plant", [7.0]),
        },
    )


def _pvgis_dataset(times: np.ndarray, include_diffuse: bool = True) -> xr.Dataset:
    data_vars = {
        "temperature_2m": (("location", "time"), [[10.0, 11.0, 12.0]]),
        "wind_speed_10m": (("location", "time"), [[2.0, 3.0, 4.0]]),
        "direct_irradiance_tilted": (
            ("location", "time"),
            [[0.0, 300.0, 500.0]],
        ),
        # The source variable is intentionally invalid and must be ignored.
        "solar_irradiance_poa": (("location", "time"), [[0.0, 0.0, 0.0]]),
    }
    if include_diffuse:
        data_vars["diffuse_irradiance_tilted"] = (
            ("location", "time"),
            [[0.0, 50.0, 100.0]],
        )
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "location": [0],
            "time": times,
            "lat": ("location", [45.0]),
            "lon": ("location", [7.0]),
        },
    )


class SolarPoaMergeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.times = np.array(
            ["2019-01-01T00:00", "2019-01-01T01:00", "2019-01-01T02:00"],
            dtype="datetime64[ns]",
        )
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pvgis_path = Path(self.temp_dir.name) / "pvgis.nc"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_merge_reconstructs_poa_from_tilted_components(self) -> None:
        _pvgis_dataset(self.times).to_netcdf(self.pvgis_path)

        merged = merge_with_weather(
            _sentinel_dataset(self.times),
            str(self.pvgis_path),
        )

        np.testing.assert_allclose(
            merged["solar_irradiance_poa"].values,
            [[0.0, 350.0, 600.0]],
        )
        np.testing.assert_allclose(
            merged["solar_irradiance_poa"],
            merged["direct_irradiance_tilted"]
            + merged["diffuse_irradiance_tilted"],
        )

    def test_merge_requires_both_tilted_components(self) -> None:
        _pvgis_dataset(self.times, include_diffuse=False).to_netcdf(
            self.pvgis_path
        )

        with self.assertRaisesRegex(
            ValueError,
            "missing PVGIS variables diffuse_irradiance_tilted",
        ):
            merge_with_weather(
                _sentinel_dataset(self.times),
                str(self.pvgis_path),
            )


if __name__ == "__main__":
    unittest.main()
