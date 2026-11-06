"""The check must read one variable in the cells of the patches and in the domain, year by year: its input, the share
out of the training range and the error of the generator."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import check_era5_cells_inputs as script

GRID = {"latitudes": [41.0, 40.5], "longitudes": [10.0, 10.5], "rows": [0, 0, 1, 1], "cols": [0, 1, 0, 1]}
NAMES = ["2m_temperature", "volumetric_soil_water_layer_1"]


class CellsTests(unittest.TestCase):
    def test_years_cells_and_domain(self):
        times = pd.date_range("2018-01-01", "2019-12-31 21:00", freq="3h")
        context = 2
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root / "cache"
            cache.mkdir()
            values = np.zeros((context + len(times), 4, 2), dtype=np.float32)
            in_2019 = context + np.flatnonzero(times.year == 2019)
            values[in_2019, 1, 1] = 5.0  # cell 1 (41N 10.5E): soil water far out of the range [-1, 1] in 2019
            np.save(cache / "test.npy", values)
            (cache / "cache.json").write_text(json.dumps({"inputs": {}, "built": {"minimum": [-1.0, -1.0],
                                                                                    "scale": [2.0, 2.0]}}))
            run = root / "run"
            (run / "scores").mkdir(parents=True)
            errors = np.zeros((len(times), 4, 2), dtype=np.float32)
            errors[in_2019 - context, 1, 1] = 3.0
            np.save(run / "scores" / "feature_scores.npy", errors)
            np.save(run / "test_timestamps.npy", times.as_unit("ns").asi8)
            (run / "metadata.json").write_text(json.dumps(
                {"status": "complete", "grid": GRID, "preparation": {"features": NAMES},
                 "backend": {"seasonal_normalization": {"mean_file": str(cache / "seasonal_mean.npy")}}}))
            cases = root / "casi.csv"
            pd.DataFrame({"rango": [1, 2], "lat_min": [41.0, 41.0], "lat_max": [41.0, 41.0], "lon_min": [10.5, 10.5],
                          "lon_max": [10.5, 10.5]}).to_csv(cases, index=False)
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                script.main(["--run-dir", str(run), "--cases", str(cases), "--stride", "1"])
            text = printed.getvalue()
        self.assertIn("41-41N 10.5-10.5E (giorni 1,2, 1 celle)", text)
        place = text.split("41-41N 10.5-10.5E")[1]
        rows = {int(line.split()[0]): [float(value) for value in line.split()[1:]]
                for line in place.strip().splitlines()[2:]}
        self.assertEqual(rows[2018], [0.0, 0.0, 0.0])
        self.assertEqual(rows[2019], [5.0, 100.0, 3.0])
        domain = text.split("dominio (4 celle)")[1].split("41-41N")[0]
        rows = {int(line.split()[0]): [float(value) for value in line.split()[1:]]
                for line in domain.strip().splitlines()[1:]}
        self.assertEqual(rows[2019], [1.25, 25.0, 0.75])


if __name__ == "__main__":
    unittest.main()
