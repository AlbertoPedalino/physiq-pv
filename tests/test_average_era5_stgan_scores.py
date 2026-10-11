"""The mean of several runs must be a run that `climatology` and `events` accept."""
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

from scripts import average_era5_stgan_scores as script
from scripts import run_era5_stgan as cli

TIMES = pd.date_range("2005-01-01", "2007-12-31 21:00", freq="3h")
GRID = {"latitudes": [40.0, 39.5, 39.0], "longitudes": [10.0, 10.5, 11.0, 11.5],
        "rows": np.repeat(np.arange(3), 4).tolist(), "cols": np.tile(np.arange(4), 3).tolist(), "spacing": 0.5}


class AverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        self.values = {name: rng.random((len(TIMES), 12)).astype(np.float32) for name in ("a", "b", "c")}
        for name, values in self.values.items():
            self.write_run(name, values)

    def write_run(self, name, values, *, timestamps=TIMES, grid=GRID, status="complete"):
        (self.root / name / "scores").mkdir(parents=True, exist_ok=True)
        np.save(self.root / name / "scores" / "anomaly_mean.npy", values)
        np.save(self.root / name / "test_timestamps.npy", timestamps.as_unit("ns").asi8)
        (self.root / name / "metadata.json").write_text(json.dumps(
            {"status": status, "grid": grid, "anomaly_mean_file": "scores/anomaly_mean.npy"}), encoding="utf-8")

    def test_output_is_the_mean_and_feeds_climatology_and_events(self):
        with redirect_stdout(io.StringIO()):
            script.main(["--run-dirs", *(str(self.root / name) for name in "abc"), "--output-dir", str(self.root / "mean")])
            cli.main(["climatology", "--run-dir", str(self.root / "mean"), "--output-dir", str(self.root / "clim"),
                      "--min-samples", "10"])
            cli.main(["events", "--run-dir", str(self.root / "clim"), "--output-dir", str(self.root / "events"),
                      "--top-percent", "5"])
        mean = np.load(self.root / "mean" / "scores" / "anomaly_mean.npy")
        expected = np.mean([self.values[name].astype(np.float64) for name in "abc"], axis=0)
        np.testing.assert_allclose(mean, expected, atol=1e-7)
        np.testing.assert_array_equal(np.load(self.root / "mean" / "test_timestamps.npy"), TIMES.as_unit("ns").asi8)
        manifest = json.loads((self.root / "mean" / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_run_dirs"], [str(self.root / name) for name in "abc"])
        self.assertEqual(manifest["grid"], GRID)
        for step in ("clim", "events"):
            self.assertEqual(json.loads((self.root / step / "metadata.json").read_text(encoding="utf-8"))["status"], "complete")
        # The source runs are left as they were.
        np.testing.assert_array_equal(np.load(self.root / "a" / "scores" / "anomaly_mean.npy"), self.values["a"])

    def test_refuses_runs_that_cannot_be_averaged(self):
        runs = [self.root / "a", self.root / "b"]
        with self.assertRaisesRegex(ValueError, "at least two"):
            script.average_scores(runs[:1], self.root / "one")
        self.write_run("late", self.values["a"], timestamps=TIMES + pd.Timedelta(hours=3))
        self.write_run("moved", self.values["a"], grid={**GRID, "longitudes": [20.0, 20.5, 21.0, 21.5]})
        self.write_run("short", self.values["a"][:-8], timestamps=TIMES[:-8])
        self.write_run("running", self.values["a"], status="running")
        for name, message in (("late", "share timestamps and grid"), ("moved", "share timestamps and grid"),
                              ("short", "share timestamps and grid"), ("running", "must be complete")):
            with self.assertRaisesRegex(ValueError, message):
                script.average_scores([self.root / "a", self.root / name], self.root / ("out_" + name))
            self.assertFalse((self.root / ("out_" + name)).exists())
        script.average_scores(runs, self.root / "mean")
        with self.assertRaisesRegex(ValueError, "new/empty"):
            script.average_scores(runs, self.root / "mean")


if __name__ == "__main__":
    unittest.main()
