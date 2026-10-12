"""The 3-hourly dust must be saved on the grid asked for, several years at a time, and resume after a failure."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

from scripts import download_cams_dust as script

AREA = (46.0, 6.0, 44.0, 10.0)  # north, west, south, east


def field(times, latitudes, longitudes):
    """Linear in latitude and longitude at every time: a bilinear interpolation gives it back exactly."""
    hours = (times - times[0]) / pd.Timedelta(hours=1)
    return (0.01 * latitudes[None, :, None] + 0.002 * longitudes[None, None, :]
            + 1e-5 * np.asarray(hours)[:, None, None]).astype(np.float32)


class Client:
    """Serves the 0.75 degree points inside the area asked for; `wait` is called while a request is running."""
    def __init__(self, calls, *, failing=(), short=(), wait=None):
        self.calls, self.failing, self.short, self.wait = calls, failing, short, wait

    def retrieve(self, name, request, target):
        year = int(request["date"][0][:4])
        self.calls.append((name, year, request["data_format"], request["area"], request["time"]))
        if request["data_format"] == "netcdf_zip":
            raise RuntimeError("400 invalid data_format")
        if self.wait is not None:
            self.wait()
        if year in self.failing:
            raise RuntimeError("503 try again later")
        north, west, south, east = request["area"]
        axis = lambda low, high: 0.75 * np.arange(np.ceil(low / 0.75), np.floor(high / 0.75) + 1)
        latitudes, longitudes = axis(south, north)[::-1], axis(west, east)
        times = pd.date_range(f"{year}-01-01", f"{year}-12-31 21:00", freq="3h")
        if year in self.short:
            times = times[:-8]
        xr.Dataset({"duaod550": (("time", "latitude", "longitude"), field(times, latitudes, longitudes))},
                   coords={"time": times, "latitude": latitudes, "longitude": longitudes}).to_netcdf(target)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = []

    def download(self, years, **options):
        client = options.pop("client", None)
        output = io.StringIO()
        with patch.object(script, "create_client", side_effect=lambda: client or Client(self.calls)), redirect_stdout(output):
            failed = script.download(self.root, years, area=AREA, **options)
        return failed, output.getvalue()

    def test_every_year_is_saved_every_3_hours_on_the_grid_asked_for(self):
        failed, text = self.download((2003, 2004))
        self.assertEqual(failed, [])
        for year, days in ((2003, 365), (2004, 366)):
            saved = np.load(self.root / f"eac4_{year}.npz")
            times = pd.DatetimeIndex(saved["times"])
            self.assertEqual(saved["dust"].shape, (days * 8, 5, 9))
            self.assertEqual((str(times[0]), str(times[-1])), (f"{year}-01-01 00:00:00", f"{year}-12-31 21:00:00"))
            self.assertTrue((np.diff(times.asi8) == np.diff(times.asi8)[0]).all())
            np.testing.assert_allclose(saved["latitudes"], np.arange(46, 43.9, -0.5))
            np.testing.assert_allclose(saved["longitudes"], np.arange(6, 10.1, 0.5))
            self.assertEqual(len(saved["longitudes"]), 9)
            np.testing.assert_allclose(saved["dust"], field(times, saved["latitudes"], saved["longitudes"]), atol=2e-6)
            # The fields as downloaded are kept, on their own grid, wider than the area on every side.
            self.assertEqual(saved["native_dust"].shape, (days * 8, len(saved["native_latitudes"]), len(saved["native_longitudes"])))
            np.testing.assert_allclose(np.abs(np.diff(saved["native_latitudes"])), 0.75)
            self.assertGreater(saved["native_latitudes"].max(), 46)
            self.assertLess(saved["native_longitudes"].min(), 6)
        asked = [call for call in self.calls if call[2] == "netcdf"]
        self.assertEqual(sorted(call[1] for call in asked), [2003, 2004])
        self.assertEqual(asked[0][0], script.DATASET)
        self.assertEqual(asked[0][3], [46.5, 5.25, 43.5, 10.5])
        self.assertEqual(asked[0][4], ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"])
        self.assertIn("grid=5x9", text)
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["eac4_2003.npz", "eac4_2004.npz"])
        # The saved years are read back as one series of fields.
        dust, times, latitudes, longitudes = script.load_fields(self.root)
        self.assertEqual(dust.shape, ((365 + 366) * 8, 5, 9))
        self.assertEqual((str(times[0]), str(times[-1])), ("2003-01-01 00:00:00", "2004-12-31 21:00:00"))
        np.testing.assert_array_equal(dust[365 * 8:], np.load(self.root / "eac4_2004.npz")["dust"])
        np.testing.assert_allclose(latitudes, np.arange(46, 43.9, -0.5))
        with tempfile.TemporaryDirectory() as empty, self.assertRaisesRegex(ValueError, "No eac4_<year>.npz"):
            script.load_fields(empty)
        # A year already saved is not asked for again.
        before = len(self.calls)
        with patch.object(script, "create_client", side_effect=AssertionError("no network expected")), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(script.download(self.root, (2003, 2004), area=AREA), [])
        self.assertEqual(len(self.calls), before)
        self.assertEqual(output.getvalue().count("already saved"), 2)

    def test_the_years_are_asked_for_at_the_same_time(self):
        together = threading.Barrier(4, timeout=30)
        failed, _ = self.download((2003, 2004, 2005, 2006), workers=4, client=Client(self.calls, wait=together.wait))
        self.assertEqual(failed, [])  # every request was running while the other three were
        self.assertEqual(len(list(self.root.glob("eac4_*.npz"))), 4)
        alone = threading.Barrier(2, timeout=0.5)
        failed, text = self.download((2007, 2008), workers=1, client=Client(self.calls, wait=alone.wait))
        self.assertEqual(failed, [2007, 2008])  # one worker: no two requests at once
        self.assertEqual(text.count("failed:"), 2)

    def test_a_year_that_fails_does_not_stop_the_others_and_is_asked_again(self):
        failed, text = self.download((2003, 2005, 2006), client=Client(self.calls, failing=(2005,), short=(2006,)))
        self.assertEqual(failed, [2005, 2006])
        self.assertIn("year=2005 failed: Dust download failed for 2005", text)
        self.assertIn("year=2006 failed: 2006: 2912 fields instead of one every 3 hours (2920)", text)
        self.assertEqual([path.name for path in self.root.iterdir()], ["eac4_2003.npz"])  # nothing partial is left
        failed, _ = self.download((2003, 2005, 2006))
        self.assertEqual(failed, [])
        self.assertEqual(len(list(self.root.glob("eac4_*.npz"))), 3)
        self.assertEqual(sum(call[1] == 2003 and call[2] == "netcdf" for call in self.calls), 1)

    def test_command_line(self):
        args = script.parser().parse_args(["--output-dir", "out"])
        self.assertEqual((args.first_year, args.last_year, args.workers, args.grid), (2003, 2025, 8, 0.5))
        self.assertEqual(tuple(args.area), script.AREA)
        latitudes, longitudes = script.target_axes(script.AREA, 0.5)
        self.assertEqual((len(latitudes), len(longitudes), len(latitudes) * len(longitudes)), (81, 131, 10611))
        self.assertEqual((latitudes[0], latitudes[-1], longitudes[0], longitudes[-1]), (60, 20, -15, 50))
        # The request is on the lines of the 0.75 degree grid just outside the domain.
        self.assertEqual(script.request_area(script.AREA), [60.75, -15.75, 19.5, 50.25])
        arguments = ["--output-dir", str(self.root), "--first-year", "2003", "--last-year", "2004", "--workers", "2",
                     "--area", *map(str, AREA)]
        output = io.StringIO()
        with patch.object(script, "create_client", side_effect=lambda: Client(self.calls, failing=(2004,))), \
                redirect_stdout(output):
            self.assertEqual(script.main(arguments), 1)
        self.assertIn("saved 1 of 2 years", output.getvalue())
        self.assertIn("failed: 2004", output.getvalue())
        with patch.object(script, "create_client", side_effect=lambda: Client(self.calls)), redirect_stdout(io.StringIO()):
            self.assertEqual(script.main(arguments), 0)
        for wrong in ((46.0, 6.0, 44.2, 10.0), (44.0, 6.0, 46.0, 10.0)):
            with self.assertRaisesRegex(ValueError, "whole number of grid steps"):
                script.download(self.root, (2010,), area=wrong)
        with self.assertRaisesRegex(ValueError, "workers"):
            script.download(self.root, (2010,), workers=0)


class FieldTests(unittest.TestCase):
    def test_bilinear_interpolation_is_the_one_of_scipy_and_refuses_points_outside(self):
        rng = np.random.default_rng(0)
        latitudes, longitudes = np.arange(47.25, 42.7, -0.75), np.arange(5.25, 11, 0.75)
        values = rng.random((6, len(latitudes), len(longitudes))).astype(np.float32)
        target_latitudes, target_longitudes = script.target_axes(AREA, 0.5)
        result = script.to_grid(values, latitudes, longitudes, target_latitudes, target_longitudes)
        points = np.stack(np.meshgrid(target_latitudes, target_longitudes, indexing="ij"), axis=-1)
        for index in range(len(values)):
            expected = RegularGridInterpolator((latitudes[::-1], longitudes), values[index, ::-1].astype(np.float64))(points)
            np.testing.assert_allclose(result[index], expected, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "do not cover the grid"):
            script.to_grid(values, latitudes, longitudes, np.array([48.0, 47.0]), target_longitudes)

    def test_both_layouts_are_read_in_time_order(self):
        rng = np.random.default_rng(1)
        latitudes, longitudes = np.arange(46.5, 43, -0.75), np.arange(5.25, 11, 0.75)
        times = pd.date_range("2006-01-01", "2006-01-03 21:00", freq="3h")
        values = rng.random((len(times), len(latitudes), len(longitudes))).astype(np.float32)
        shuffled = rng.permutation(len(times))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            xr.Dataset({"duaod550": (("time", "latitude", "longitude"), values)},
                       coords={"time": times, "latitude": latitudes, "longitude": longitudes}).to_netcdf(root / "plain.nc")
            xr.Dataset({"duaod550": (("forecast_period", "forecast_reference_time", "latitude", "longitude"),
                                     values[shuffled][None])},
                       coords={"forecast_period": [np.timedelta64(0, "h")], "forecast_reference_time": times[shuffled],
                               "latitude": latitudes, "longitude": longitudes + 360,
                               "valid_time": (("forecast_period", "forecast_reference_time"), times[shuffled].values[None])}
                       ).to_netcdf(root / "inner.nc")
            with zipfile.ZipFile(root / "zipped.zip", "w") as archive:
                archive.write(root / "inner.nc", "data_sfc.nc")
            plain, zipped = script.read_fields(root / "plain.nc"), script.read_fields(root / "zipped.zip")
        for first, second in zip(plain, zipped):
            np.testing.assert_allclose(np.asarray(first, dtype=float), np.asarray(second, dtype=float))
        np.testing.assert_array_equal(plain[0], values)
        self.assertTrue(plain[1].equals(times))


if __name__ == "__main__":
    unittest.main()
