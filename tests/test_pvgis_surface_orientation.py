"""Plane orientation read from PVGIS file attributes for the clear-sky reference."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physiq_pv.data.pvgis_dataset import (  # noqa: E402
    _solar_geometry_and_clearsky_poa,
    _surface_orientation,
)


class SurfaceOrientationTest(unittest.TestCase):
    def test_pvlib_spelling_is_kept(self):
        self.assertEqual(
            _surface_orientation({"tilt_angle": 30, "azimuth_angle": 180}), (30.0, 180.0)
        )
        self.assertEqual(
            _surface_orientation({"pvgis_tilt_angle": 20, "pvgis_azimuth_angle": 170}),
            (20.0, 170.0),
        )

    def test_missing_attributes_keep_previous_default(self):
        self.assertEqual(_surface_orientation({}), (30.0, 180.0))

    def test_pvgis_request_spelling_is_read(self):
        # Attributes of the summed-irradiance files: horizontal plane.
        self.assertEqual(_surface_orientation({"tilt": 0, "azimuth": 0}), (0.0, 180.0))
        self.assertEqual(_surface_orientation({"tilt": 35, "azimuth": -90}), (35.0, 90.0))

    def test_pvlib_spelling_wins_over_pvgis_request(self):
        attrs = {"tilt_angle": 30, "azimuth_angle": 180, "tilt": 0, "azimuth": 0}
        self.assertEqual(_surface_orientation(attrs), (30.0, 180.0))

    def test_horizontal_plane_gives_clear_sky_ghi(self):
        times = pd.date_range("2019-12-21 06:10", periods=12, freq="h")
        lats, lons = np.array([45.0]), np.array([7.7])
        _, _, horizontal = _solar_geometry_and_clearsky_poa(
            times, lats, lons, surface_tilt=0.0, surface_azimuth=180.0
        )
        _, _, tilted = _solar_geometry_and_clearsky_poa(
            times, lats, lons, surface_tilt=30.0, surface_azimuth=180.0
        )
        # In winter a south-facing 30-degree plane receives much more clear-sky
        # irradiance than the horizontal one: the two references must differ.
        self.assertGreater(float(tilted.max()), 1.3 * float(horizontal.max()))
        self.assertGreater(float(horizontal.max()), 0.2)


if __name__ == "__main__":
    unittest.main()
