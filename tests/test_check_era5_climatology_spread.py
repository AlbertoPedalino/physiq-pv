"""The check must rebuild the spread of the score climatology as the climatology run did, and find the groups with
almost none and the flags that fall in them."""
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

from physiq_pv.era5.seasonality import reference_zscores
from scripts import check_era5_climatology_spread as script

GRID = {"latitudes": [41.0, 40.5], "longitudes": [10.0, 10.5], "rows": [0, 0, 1, 1], "cols": [0, 1, 0, 1]}


class SpreadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        self.reference_times = pd.date_range("2000-01-01", "2001-12-31 21:00", freq="3h")
        reference = rng.normal(1.0, 0.5, (len(self.reference_times), 4)).astype(np.float32)
        reference[:, 3] = 1.0 + rng.normal(0, 1e-5, len(self.reference_times))  # cell 3: almost no spread
        self.reference = reference
        validation = root / "validation"
        (validation / "scores").mkdir(parents=True)
        np.save(validation / "scores" / "generator_mean.npy", reference)
        np.save(validation / "test_timestamps.npy", self.reference_times.as_unit("ns").asi8)
        (validation / "metadata.json").write_text(json.dumps(
            {"status": "complete", "grid": GRID, "generator_mean_file": "scores/generator_mean.npy"}), encoding="utf-8")
        times = pd.date_range("2005-01-01", "2005-06-30 21:00", freq="3h")
        scores = rng.normal(1.0, 0.5, (len(times), 4)).astype(np.float32)
        scores[:, 3] = 1.0 + rng.normal(0, 1e-3, len(times))  # a small change: huge against the spread of cell 3
        self.run = root / "clim"
        (self.run / "scores").mkdir(parents=True)
        normalization = reference_zscores(scores, times, reference, self.reference_times,
                                          self.run / "scores" / "anomaly_mean.npy", period="day", window_days=15)
        normalization["reference_run_dir"] = str(validation)
        normalization["normal_masks"] = None
        np.save(self.run / "test_timestamps.npy", times.as_unit("ns").asi8)
        (self.run / "metadata.json").write_text(json.dumps(
            {"status": "complete", "grid": GRID, "reference": "validation", "source_score_component": "generator",
             "source_run_dir": str(root / "test"), "normalization": normalization,
             "anomaly_mean_file": "scores/anomaly_mean.npy"}), encoding="utf-8")
        self.times, self.scores = times, scores

    def test_the_moments_are_those_of_the_climatology(self):
        hours, n, mean, deviation, _ = script.reference_moments(self.reference, self.reference_times, None, "day", 15)
        slot = script._slots(self.times, hours, "day")[0]
        z = np.load(self.run / "scores" / "anomaly_mean.npy")
        np.testing.assert_allclose(z, ((self.scores - mean[slot]) / deviation[slot]).astype(np.float32), rtol=1e-3)

    def test_the_cell_without_spread_takes_the_flags(self):
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            script.main(["--run-dir", str(self.run), "--extreme", "20"])
        text = printed.getvalue()
        self.assertIn("cells with at least one group under it 1 of 4", text)
        # The flags of the run: those of cell 3, enormous, and the highest of the others.
        flagged_under = int(text.split("in groups under the floor ")[1].split(" ")[0])
        self.assertGreater(flagged_under, 0)
        self.assertIn("-> recompute", text)


if __name__ == "__main__":
    unittest.main()
