import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_era5_case_fields import case_fields  # noqa: E402


class CaseFieldsTest(unittest.TestCase):
    def make(self, root, cells=3):
        run, prepared = Path(root) / "run", Path(root) / "prepared"
        run.mkdir(), prepared.mkdir()
        grid = {"rows": [0, 0, 1][:cells], "cols": [0, 1, 1][:cells], "latitudes": [45.0, 44.5],
                "longitudes": [7.0, 7.5]}
        (run / "metadata.json").write_text(json.dumps({"grid": grid}), encoding="utf-8")
        (prepared / "metadata.json").write_text(json.dumps({"features": ["a", "mean_sea_level_pressure", "b"]}),
                                                encoding="utf-8")
        times = pd.date_range("2017-11-22 18:00", periods=4, freq="3h")
        np.save(prepared / "test_timestamps.npy", times.to_numpy())
        np.save(prepared / "test.npy", np.arange(4 * 3 * 3, dtype=np.float32).reshape(4, 3, 3))
        return run, prepared

    def test_values_of_the_timestamps_and_features(self):
        with tempfile.TemporaryDirectory() as root:
            run, prepared = self.make(root)
            out = case_fields(run, prepared, ["2017-11-23 00:00"], ["mean_sea_level_pressure"])
            self.assertEqual(out["values"].shape, (1, 3, 1))
            np.testing.assert_array_equal(out["values"][0, :, 0], [19, 22, 25])  # timestamp 2, column 1
            np.testing.assert_array_equal(out["rows"], [0, 0, 1])

    def test_missing_timestamp_or_feature(self):
        with tempfile.TemporaryDirectory() as root:
            run, prepared = self.make(root)
            with self.assertRaises(ValueError):
                case_fields(run, prepared, ["2018-01-01 00:00"], ["a"])
            with self.assertRaises(ValueError):
                case_fields(run, prepared, ["2017-11-23 00:00"], ["total_precipitation"])

    def test_cells_must_match_the_grid(self):
        with tempfile.TemporaryDirectory() as root:
            run, prepared = self.make(root, cells=2)
            with self.assertRaises(ValueError):
                case_fields(run, prepared, ["2017-11-23 00:00"], ["a"])


if __name__ == "__main__":
    unittest.main()
