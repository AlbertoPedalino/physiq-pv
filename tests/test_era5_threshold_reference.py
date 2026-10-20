"""The checks take the threshold of a test score from the same score over the validation years when its run says so."""
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

from scripts import set_era5_threshold_reference as script
from scripts.check_era5_dust_lead import THRESHOLD_STRIDE, score_limit
from scripts.check_era5_lead import flags

GRID = {"latitudes": [45.0, 44.5], "longitudes": [7.0, 7.5], "rows": [0, 0, 1, 1], "cols": [0, 1, 0, 1], "spacing": 0.5}


def run(directory, times, scores, **metadata):
    (directory / "scores").mkdir(parents=True)
    np.save(directory / "scores" / "anomaly_mean.npy", scores.astype(np.float32))
    np.save(directory / "test_timestamps.npy", times.as_unit("ns").asi8)
    (directory / "metadata.json").write_text(json.dumps({"status": "complete", "grid": GRID,
        "score_kind": "climatology_zscore", "source_score_component": "generator",
        "anomaly_mean_file": "scores/anomaly_mean.npy", **metadata}), encoding="utf-8")
    return directory


class ThresholdReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        valid_times = pd.date_range("2000-01-01", "2004-12-31 21:00", freq="3h")
        test_times = pd.date_range("2005-01-01", "2006-12-31 21:00", freq="3h")
        # The test is a little more unusual than the validation: one standard deviation higher.
        self.valid = run(root / "val", valid_times, rng.normal(size=(len(valid_times), 4)))
        self.test = run(root / "test", test_times, rng.normal(1.0, 1.0, size=(len(test_times), 4)))
        self.other = run(root / "other", valid_times, rng.normal(size=(len(valid_times), 4)), score_kind="max_of_scores")

    def limit(self, directory, percent):
        scores = np.load(directory / "scores" / "anomaly_mean.npy", mmap_mode="r")
        try:
            return score_limit(scores, percent)
        finally:
            scores._mmap.close()

    def test_the_threshold_of_the_validation_is_applied_to_the_test(self):
        own = self.limit(self.test, 5)
        validation = self.limit(self.valid, 5)
        self.assertAlmostEqual(own, 1.0 + 1.645, delta=0.1)
        self.assertAlmostEqual(validation, 1.645, delta=0.1)
        with redirect_stdout(io.StringIO()):
            script.main(["--run-dir", str(self.test), "--reference-run-dir", str(self.valid)])
        self.assertEqual(self.limit(self.test, 5), validation)
        # The checks flag what lies above it: about a quarter of the test instead of 5%.
        flagged, _, limit = flags(self.test, 5)
        self.assertEqual(limit, validation)
        self.assertAlmostEqual(flagged.mean(), 0.26, delta=0.03)
        # An array that is not a file of a run keeps its own percentile.
        values = np.arange(1000.0)
        self.assertEqual(score_limit(values, 5), float(np.nanpercentile(values[::THRESHOLD_STRIDE], 95)))
        with redirect_stdout(io.StringIO()):
            script.main(["--run-dir", str(self.test), "--clear"])
        self.assertEqual(self.limit(self.test, 5), own)

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "same kind"):
            script.set_reference(self.test, self.other)
        with self.assertRaisesRegex(ValueError, "years before"):
            script.set_reference(self.valid, self.test)
        metadata = json.loads((self.other / "metadata.json").read_text())
        (self.other / "metadata.json").write_text(json.dumps({**metadata, "grid": {**GRID, "rows": [0, 1, 0, 1]},
                                                              "score_kind": "climatology_zscore"}))
        with self.assertRaisesRegex(ValueError, "same cells"):
            script.set_reference(self.test, self.other)
        self.assertNotIn("threshold_reference", json.loads((self.test / "metadata.json").read_text()))


if __name__ == "__main__":
    unittest.main()
