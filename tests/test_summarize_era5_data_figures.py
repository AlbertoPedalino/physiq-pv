"""The data-figure summary must read the 24-hour change on every timestamp and write the three parts."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from physiq_pv.era5.data import FEATURE_NAMES
from scripts import summarize_era5_data_figures as script

ROWS, COLS = np.indices((4, 4)).reshape(2, -1)
PERIODS = {"train": pd.date_range("1996-01-01", "1999-12-31 21:00", freq="3h"),
           "validation": pd.date_range("2000-01-01", "2000-12-31 21:00", freq="3h"),
           "test": pd.date_range("2001-01-01", "2001-12-31 21:00", freq="3h")}
CLOUD = FEATURE_NAMES.index("total_cloud_cover")


class DataFigureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        for name, times in PERIODS.items():
            # Every variable a smooth seasonal cycle in [0, 1]; the cloud cover alone jumps between days.
            season = (0.5 + 0.4 * np.sin(2 * np.pi * times.dayofyear.to_numpy() / 365.25))[:, None, None]
            values = np.repeat(np.repeat(season, 16, axis=1), len(FEATURE_NAMES), axis=2).astype(np.float32)
            values += 0.01 * rng.normal(size=values.shape).astype(np.float32)
            values[:, :, CLOUD] = rng.random((len(times), 16))
            np.save(self.root / f"{name}.npy", values)
            np.save(self.root / f"{name}_timestamps.npy", times.as_unit("ns").asi8)
        (self.root / "metadata.json").write_text(json.dumps({
            "status": "complete", "features": list(FEATURE_NAMES), "start_year": 1996, "train_end_year": 1999,
            "validation_start_year": 2000, "validation_end_year": 2000, "test_start_year": 2001, "score_end_year": 2001,
            "grid": {"latitudes": [45.0, 44.5, 44.0, 43.5], "longitudes": [7.0, 7.5, 8.0, 8.5], "rows": ROWS.tolist(),
                     "cols": COLS.tolist(), "spacing": 0.5}}), encoding="utf-8")

    def test_the_jumping_variable_dominates_the_change_and_every_part_is_written(self):
        output_path = self.root / "out.json"
        text = io.StringIO()
        with redirect_stdout(text):
            self.assertEqual(script.main(["--prepared-dir", str(self.root), "--output", str(output_path)]), 0)
        result = json.loads(output_path.read_text(encoding="utf-8"))
        share = dict(zip(result["change_24h"]["features"], result["change_24h"]["share"]))
        self.assertAlmostEqual(sum(share.values()), 1.0)
        self.assertGreater(share["total_cloud_cover"], 0.5)
        self.assertEqual(result["training"], [1996, 1999])
        cycle = result["cycles"]["cells"]["Pianura Padana"]["2m_temperature"]
        self.assertEqual(len(cycle["monthly_mean"]), 12)
        self.assertEqual(len(cycle["hourly_mean"]["7"]), 8)
        self.assertGreater(max(cycle["monthly_mean"]) - min(cycle["monthly_mean"]), 0.5)
        self.assertEqual(result["correlation"]["cells"], 16)
        self.assertIn("change over 24 hours", text.getvalue())


if __name__ == "__main__":
    unittest.main()
