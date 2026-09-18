"""Offline tests with real NetCDF payloads; no CDS credentials or data downloads."""

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch
from zipfile import ZipFile

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import download_era5 as era5


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.area = [30.5, 0, 30, 0.5]
        self.product = era5.PRODUCTS[0]
        self.request = era5.make_request(self.product, 1980, 1, [1, 2], self.area)
        self.target = self.root / "out" / "era5_single_1980_01.nc"

    def fixture(self, filename="source.nc", product=None, request=None, variables=None):
        product = product or self.product
        request = request or self.request
        variables = product.variables if variables is None else variables
        coords = {"valid_time": era5.expected_times(request),
                  "latitude": [30.5, 30.0], "longitude": [0, 0.5]}
        dims = ["valid_time", "latitude", "longitude"]
        shape = [len(coords["valid_time"]), 2, 2]
        if product.pressure:
            coords["pressure_level"] = [850.0]
            dims.insert(1, "pressure_level")
            shape.insert(1, 1)
        ds = xr.Dataset({product.variables[name][0]: (dims, np.ones(shape, dtype=np.float32))
                         for name in variables}, coords=coords)
        path = self.root / filename
        ds.to_netcdf(path, engine="netcdf4")
        ds.close()
        return path

    def client_for(self, source):
        client = Mock()
        client.retrieve.side_effect = lambda dataset, request, target: shutil.copyfile(source, target)
        return client

    def download(self, client, **kwargs):
        return era5.download_month(client, self.product, self.request, self.target,
                                   retry_delay=0, **kwargs)

    def test_requests_calendar_and_defaults(self):
        args = era5.build_parser().parse_args([])
        self.assertEqual(args.start_year, 1980)
        self.assertEqual(args.end_year, "latest")
        self.assertEqual([args.north, args.west, args.south, args.east], [60, -15, 20, 50])
        self.assertEqual(sum(len(p.variables) for p in era5.PRODUCTS), 15)
        self.assertEqual(args.output_dir.as_posix(), "/home/apedalino/physiq_pv/data/era5")
        self.assertEqual(era5.grid_shape([60, -15, 20, 50]), (81, 131))
        self.assertEqual(len(self.request["variable"]), 10)
        selected_times = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]
        self.assertEqual(self.request["time"], selected_times)
        self.assertEqual(self.request["grid"], [0.5, 0.5])
        p = era5.make_request(era5.PRODUCTS[1], 1980, 2, list(range(1, 30)), self.area)
        self.assertEqual(p["day"][-1], "29")
        self.assertEqual(p["pressure_level"], ["850"])
        self.assertEqual(p["time"], selected_times)
        self.assertEqual(p["grid"], [0.5, 0.5])
        self.assertEqual(len(era5.expected_times(p)), 29 * 8)
        self.assertEqual(p["data_format"], "netcdf")
        with self.assertRaises(ValueError):
            era5.validate_area([60.1, -15, 20, 50])
        with self.assertRaises(ValueError):
            era5.validate_area([20, -15, 60, 50])

    def test_catalogue_names_and_common_latest(self):
        def fetch(url):
            product = next(p for p in era5.PRODUCTS if p.dataset in url)
            if url.endswith("/form"):
                return [{"name": name, "details": {"values": values}} for name, values in {
                    "variable": list(product.variables), "product_type": ["reanalysis"],
                    "data_format": ["netcdf"], "download_format": ["unarchived"],
                    "pressure_level": ["850"], "year": [str(y) for y in range(1940, 2027)],
                }.items()]
            end = "2026-09-12T00:00:00Z" if not product.pressure else "2025-12-31T00:00:00Z"
            return {"links": [{"rel": "form", "href": url + "/form"}],
                    "extent": {"temporal": {"interval": [["1940-01-01T00:00:00Z", end]]}}}
        result = era5.inspect_catalogue(fetch)
        self.assertEqual(era5.latest_complete_year(result, date(2026, 9, 18)), 2025)
        result[era5.PRODUCTS[1].dataset]["end"] = "2025-12-30T00:00:00+00:00"
        self.assertEqual(era5.latest_complete_year(result, date(2026, 9, 18)), 2024)
        def invalid(url):
            result = fetch(url)
            if url.endswith("/form"):
                result[0]["details"]["values"].pop()
            return result
        with self.assertRaisesRegex(ValueError, "Invalid CDS"):
            era5.inspect_catalogue(invalid)

    def test_plain_download_skip_force_and_checksum_corruption(self):
        source = self.fixture()
        client = self.client_for(source)
        self.assertEqual(self.download(client), "downloaded")
        self.assertEqual(self.target.read_bytes(), source.read_bytes())
        self.assertEqual(self.download(client), "skipped")
        self.assertEqual(client.retrieve.call_count, 1)
        self.assertEqual(self.download(client, force=True), "downloaded")
        with self.target.open("ab") as stream:
            stream.write(b"corruption")
        self.assertEqual(self.download(client), "downloaded")
        self.assertEqual(client.retrieve.call_count, 3)
        self.assertEqual(self.target.read_bytes(), source.read_bytes())
        self.assertFalse(list(self.target.parent.glob("*.part")))

    def test_preexisting_valid_adopted_and_truncated_redownloaded(self):
        source = self.fixture()
        self.target.parent.mkdir()
        shutil.copyfile(source, self.target)
        client = self.client_for(source)
        self.assertEqual(self.download(client), "skipped")
        client.retrieve.assert_not_called()
        self.target.with_suffix(".json").unlink()
        self.target.write_bytes(source.read_bytes()[:256])
        self.assertEqual(self.download(client), "downloaded")

    def test_split_zip_preserves_raw_bytes_and_resumes(self):
        accum = {"total_precipitation", "surface_solar_radiation_downwards"}
        a = self.fixture("instant.nc", variables=set(era5.SINGLE) - accum)
        b = self.fixture("accum.nc", variables=accum)
        archive_path = self.root / "response.zip"
        with ZipFile(archive_path, "w") as archive:
            archive.write(a, "data_stream-oper_stepType-instant.nc")
            archive.write(b, "data_stream-oper_stepType-accum.nc")
        client = self.client_for(archive_path)
        self.assertEqual(self.download(client), "downloaded")
        self.assertFalse(self.target.exists())  # Never disguise a ZIP as NetCDF.
        receipt = json.loads(self.target.with_suffix(".json").read_text())
        self.assertEqual(len(receipt["files"]), 2)
        self.assertEqual(self.download(client), "skipped")
        for entry, source in zip(receipt["files"], (a, b)):
            raw = self.target.parent / entry["name"]
            self.assertEqual(raw.read_bytes(), source.read_bytes())
            with xr.open_dataset(raw) as ds:
                self.assertEqual(ds.sizes["valid_time"], 16)
        raw.unlink()
        self.assertEqual(self.download(client), "downloaded")

    def test_pressure_850_readable_and_wrong_level_rejected(self):
        product = era5.PRODUCTS[1]
        request = era5.make_request(product, 1980, 1, [1, 2], self.area)
        source = self.fixture(product=product, request=request)
        result = era5.validate_netcdfs([source], product, request)
        self.assertEqual(result["timestamps"], 16)
        with xr.open_dataset(source) as ds:
            wrong = ds.load().assign_coords(pressure_level=[700.0])
        wrong.to_netcdf(self.root / "wrong.nc")
        with self.assertRaisesRegex(ValueError, "850"):
            era5.validate_netcdfs([self.root / "wrong.nc"], product, request)

    def test_accumulated_fields_values_units_and_times_preserved(self):
        source = self.fixture()
        with xr.open_dataset(source) as opened:
            ds = opened.load()
        # Deliberately varying accumulations: sums, differences, scaling, or a
        # time shift must not accidentally pass through an all-ones fixture.
        values = np.arange(16 * 2 * 2, dtype=np.float32).reshape(16, 2, 2)
        ds["tp"].values[:] = values / 1000
        ds["ssrd"].values[:] = values * 3600 + 123
        ds["tp"].attrs.update(units="m", GRIB_stepType="accum")
        ds["ssrd"].attrs.update(units="J m**-2", GRIB_stepType="accum")
        source = self.root / "accumulations.nc"
        ds.to_netcdf(source)
        self.assertEqual(self.download(self.client_for(source)), "downloaded")
        self.assertEqual(self.target.read_bytes(), source.read_bytes())
        with xr.open_dataset(self.target) as actual:
            xr.testing.assert_identical(actual[["tp", "ssrd"]], ds[["tp", "ssrd"]])
            np.testing.assert_array_equal(np.diff(actual.valid_time.values),
                                          np.full(15, np.timedelta64(3, "h")))

    def test_old_resolution_receipts_and_payloads_are_not_skipped(self):
        source = self.fixture()
        self.download(self.client_for(source))
        receipt_path = self.target.with_suffix(".json")
        receipt = json.loads(receipt_path.read_text())
        for field, previous_value in (("grid", [0.25, 0.25]),
                                      ("time", [f"{h:02d}:00" for h in range(24)])):
            old_receipt = json.loads(json.dumps(receipt))
            old_receipt["request"][field] = previous_value
            receipt_path.write_text(json.dumps(old_receipt))
            self.assertFalse(era5.existing_valid(self.target, self.product, self.request))
        receipt_path.unlink()
        with xr.open_dataset(source) as opened:
            ds = opened.load()
        # A receipt-free file must also fail based on actual coordinates/times.
        for dim, values in (("latitude", [30.5, 30.25, 30.0]),
                            ("valid_time", np.arange(np.datetime64("1980-01-01T00"),
                                                     np.datetime64("1980-01-03T00"),
                                                     np.timedelta64(1, "h")))):
            ds.reindex({dim: values}, fill_value=1.0).to_netcdf(self.target)
            self.assertFalse(era5.existing_valid(self.target, self.product, self.request))

    def test_missing_hour_variable_area_and_payload_rejected(self):
        source = self.fixture()
        with xr.open_dataset(source) as opened:
            ds = opened.load()
        wrong = [ds.isel(valid_time=slice(1, None)), ds.drop_vars("zust"),
                 ds.assign_coords(latitude=[31.25, 31.0])]
        missing_payload = ds.copy(deep=True)
        missing_payload["t2m"][0] = np.nan
        wrong.append(missing_payload)
        for i, candidate in enumerate(wrong):
            path = self.root / f"invalid{i}.nc"
            candidate.to_netcdf(path)
            with self.assertRaises(ValueError):
                era5.validate_netcdfs([path], self.product, self.request)

    def test_retry_and_ctrl_c_preserve_committed_files(self):
        source = self.fixture()
        client = self.client_for(source)
        calls = []
        def flaky(dataset, request, target):
            calls.append(target)
            if len(calls) == 1:
                Path(target).write_bytes(b"incomplete")
                raise OSError("temporary network failure")
            shutil.copyfile(source, target)
        client.retrieve.side_effect = flaky
        self.assertEqual(self.download(client, attempts=2), "downloaded")
        original = self.target.read_bytes()
        client.retrieve.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.download(client, force=True)
        self.assertEqual(self.target.read_bytes(), original)
        self.assertTrue(era5.existing_valid(self.target, self.product, self.request))
        self.assertFalse(list(self.target.parent.glob("*.part")))

    def test_bad_response_never_commits_and_zip_traversal_rejected(self):
        source = self.root / "bad.zip"
        with ZipFile(source, "w") as archive:
            archive.writestr("../escape.nc", b"bad")
        with self.assertRaises(RuntimeError):
            self.download(self.client_for(source), attempts=1)
        self.assertFalse(self.target.exists())
        self.assertFalse(self.target.with_suffix(".json").exists())
        self.assertFalse((self.root / "escape.nc").exists())
        self.assertFalse(list(self.target.parent.glob("*.part")))

    def test_single_member_zip_and_failed_force_preserve_old_month(self):
        source = self.fixture()
        archive_path = self.root / "one.zip"
        with ZipFile(archive_path, "w") as archive:
            archive.write(source, "data.nc")
        self.assertEqual(self.download(self.client_for(archive_path)), "downloaded")
        self.assertEqual(self.target.read_bytes(), source.read_bytes())
        receipt = self.target.with_suffix(".json").read_bytes()
        bad = self.root / "error.html"
        bad.write_text("CDS internal error")
        with self.assertRaises(RuntimeError):
            self.download(self.client_for(bad), force=True, attempts=1)
        self.assertEqual(self.target.read_bytes(), source.read_bytes())
        self.assertEqual(self.target.with_suffix(".json").read_bytes(), receipt)

    def test_test_mode_only_requests_two_days_both_products(self):
        catalogue = {p.dataset: {"end": "2026-09-12T00:00:00+00:00", "years": [str(y) for y in range(1940, 2027)]}
                     for p in era5.PRODUCTS}
        args = era5.build_parser().parse_args(["--test-days", "2", "--output-dir", str(self.root / "absent")])
        with patch.object(era5, "inspect_catalogue", return_value=catalogue), \
             patch.object(era5, "create_client"), \
             patch.object(era5, "download_month", return_value="downloaded") as download, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(era5.run(args), 0)
        self.assertEqual(download.call_count, 2)
        for call in download.call_args_list:
            request, target = call.args[2:4]
            self.assertEqual(request["year"], ["1980"])
            self.assertEqual(request["month"], ["01"])
            self.assertEqual(request["day"], ["01", "02"])
            self.assertIn("test_1980_01_02", target.parts)

    def test_credentials_not_read_or_written_when_absent(self):
        with patch.object(era5.Path, "home", return_value=self.root):
            with self.assertRaisesRegex(RuntimeError, "Missing ~/.cdsapirc"):
                era5.create_client()
        self.assertFalse((self.root / ".cdsapirc").exists())

    def test_client_leaves_credentials_to_cdsapi(self):
        module = Mock()
        with patch.dict(sys.modules, {"cdsapi": module}), \
             patch.object(era5.Path, "is_file", return_value=True), \
             patch.object(era5.Path, "read_text", side_effect=AssertionError("Do not read credentials")):
            self.assertIs(era5.create_client(), module.Client.return_value)
        options = module.Client.call_args.kwargs
        self.assertNotIn("url", options)
        self.assertNotIn("key", options)

    def test_client_configuration_errors_do_not_echo_credentials(self):
        module = Mock()
        module.Client.side_effect = ValueError("private configuration contents")
        with patch.dict(sys.modules, {"cdsapi": module}), \
             patch.object(era5.Path, "is_file", return_value=True), \
             self.assertLogs(era5.LOG, level="ERROR") as logs:
            self.assertEqual(era5.main(["--check-api"]), 1)
        self.assertIn("Check url and key in ~/.cdsapirc", " ".join(logs.output))
        self.assertNotIn("private configuration contents", " ".join(logs.output))

    def test_check_api_authenticates_without_download_or_output_writes(self):
        client = Mock()
        output = io.StringIO()
        destination = self.root / "absent"
        # Even download-related options cannot make --check-api submit a job.
        with patch.object(era5, "create_client", return_value=client), \
             patch.object(era5, "inspect_catalogue") as catalogue, \
             patch.object(era5, "free_disk") as disk, \
             patch.object(era5, "download_month") as download, \
             contextlib.redirect_stdout(output):
            self.assertEqual(era5.main(["--check-api", "--test-days", "2", "--force",
                                       "--output-dir", str(destination)]), 0)
        client.client.check_authentication.assert_called_once_with()
        client.retrieve.assert_not_called()
        client.client.submit.assert_not_called()
        catalogue.assert_not_called()
        disk.assert_not_called()
        download.assert_not_called()
        self.assertFalse(destination.exists())
        self.assertIn("authentication OK", output.getvalue())

    def test_check_api_reports_failures_without_echoing_response(self):
        for status in (401, 403, None):
            with self.subTest(status=status):
                client = Mock()
                error = OSError("private response contents")
                error.response = Mock(status_code=status)
                client.client.check_authentication.side_effect = error
                with patch.object(era5, "create_client", return_value=client), \
                     patch.object(era5, "download_month") as download, \
                     self.assertLogs(era5.LOG, level="ERROR") as logs:
                    self.assertEqual(era5.main(["--check-api"]), 1)
                message = " ".join(logs.output)
                self.assertIn("CDS API check failed", message)
                self.assertNotIn("private response contents", message)
                if status:
                    self.assertIn(f"HTTP {status}", message)
                download.assert_not_called()

    def test_missing_configuration_fails_before_network_access(self):
        for options in ([], ["--check-api"]):
            with patch.object(era5.Path, "home", return_value=self.root), \
                 patch.object(era5, "inspect_catalogue") as catalogue, \
                 patch.object(era5, "download_month") as download, \
                 self.assertLogs(era5.LOG, level="ERROR") as logs:
                self.assertEqual(era5.main(options), 1)
            self.assertIn("Missing ~/.cdsapirc", " ".join(logs.output))
            catalogue.assert_not_called()
            download.assert_not_called()

    def test_check_api_rejects_legacy_config_and_conflicting_dry_run(self):
        with self.assertRaisesRegex(RuntimeError, "current CDS URL"):
            era5.check_api(object())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            era5.build_parser().parse_args(["--check-api", "--dry-run"])
        self.assertEqual(error.exception.code, 2)

    def test_dry_run_no_files_no_client_and_test_scope(self):
        catalogue = {p.dataset: {"end": "2026-09-12T00:00:00+00:00", "years": [str(y) for y in range(1940, 2027)]}
                     for p in era5.PRODUCTS}
        for options, expected in (([], "Months: 552; timestamps: 134,416; CDS requests: 1104"),
                                  (["--test-days", "2"], "Months: 1; timestamps: 16; CDS requests: 2")):
            args = era5.build_parser().parse_args(["--dry-run", "--output-dir", str(self.root / "absent"), *options])
            output = io.StringIO()
            with patch.object(era5, "inspect_catalogue", return_value=catalogue), \
                 patch.object(era5, "create_client") as client, contextlib.redirect_stdout(output):
                self.assertEqual(era5.run(args), 0)
            client.assert_not_called()
            self.assertFalse((self.root / "absent").exists())
            self.assertIn(expected, output.getvalue())
            self.assertIn("cells: 81 x 131 = 10,611", output.getvalue())
            self.assertIn("8 timestamps/day", output.getvalue())
            if not options:
                self.assertIn("payload: 0.086 TB", output.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
