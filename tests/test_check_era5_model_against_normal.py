"""The check of the model against the normal must tell apart what each score flags alone and what both flag."""
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

from scripts import check_era5_model_against_normal as script

TIMES = pd.date_range("2005-01-01", "2005-12-31 21:00", freq="3h")
GRID = {"latitudes": [55.0, 45.0, 35.0, 25.0], "longitudes": [10.0, 10.5, 11.0],
        "rows": np.repeat(np.arange(4), 3).tolist(), "cols": np.tile(np.arange(3), 4).tolist(), "spacing": 0.5}
NAMES = ["rain", "pressure", "wind"]
SUMMER, WINTER, SPRING = (np.isin(TIMES.month, months) for months in ((6, 7, 8), (12, 1, 2), (3, 4, 5)))
AT_25N, AT_45N, AT_55N = slice(9, 12), slice(3, 6), slice(0, 3)
CONTEXT = 4  # timestamps of training history the cache keeps before the scored period
SCALE = [4.0, 10.0, 8.0]
# A quarter of the cells for a quarter of the year, twice for each score: an eighth of its values is high.
TOP = "12.5"


class ModelAgainstNormalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        self.model = rng.random((len(TIMES), 12)).astype(np.float32) * 0.1
        self.normal = rng.random((len(TIMES), 12)).astype(np.float32) * 0.1
        # The model alone is high at 25N in summer, the normal alone at 45N in spring, both at 55N in winter.
        self.model[SUMMER, AT_25N] += 1
        self.normal[SPRING, AT_45N] += 1
        self.model[WINTER, AT_55N] += 1
        self.normal[WINTER, AT_55N] += 1
        # The error of the model is all rain at 25N; elsewhere the pressure is three times the wind.
        self.model_errors = np.zeros((len(TIMES), 12, 3), dtype=np.float32)
        self.model_errors[:, AT_25N, 0] = 1
        self.model_errors[:, :9, 1] = 3
        self.model_errors[:, :9, 2] = 1
        # The data are away from the normal in the wind alone.
        self.standardised = np.zeros((CONTEXT + len(TIMES), 12, 3), dtype=np.float32)
        self.standardised[..., 2] = 1
        self.cache = self.root / "cache"
        self.cache.mkdir()
        np.save(self.cache / "test.npy", self.standardised)
        (self.cache / "cache.json").write_text(json.dumps({"inputs": {}, "built": {"scale": SCALE}}), encoding="utf-8")
        self.paths = [self.write_run("model_clim", scores=self.model), self.write_run("normal_clim", scores=self.normal),
                      self.write_run("model", errors=self.model_errors),
                      self.write_run("normal", errors=(2 * self.standardised[CONTEXT:] / np.array(SCALE)) ** 2)]

    def write_run(self, name, *, scores=None, errors=None, weights=None, times=TIMES):
        (self.root / name / "scores").mkdir(parents=True)
        if scores is not None:
            np.save(self.root / name / "scores" / "anomaly_mean.npy", scores)
        if errors is not None:
            np.save(self.root / name / "scores" / "feature_scores.npy", errors.astype(np.float32))
        np.save(self.root / name / "test_timestamps.npy", times.as_unit("ns").asi8)
        (self.root / name / "metadata.json").write_text(json.dumps(
            {"status": "complete", "grid": GRID, "preparation": {"features": NAMES}, "cache_dir": str(self.cache),
             "backend": {"reconstruction_feature_weights": weights}}), encoding="utf-8")
        return self.root / name

    def run_script(self, *arguments, paths=None):
        options = ("--model-run", "--normal-run", "--model-errors-dir", "--normal-errors-dir")
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main([str(item) for pair in zip(options, paths or self.paths) for item in pair]
                                         + ["--top-percent", TOP, "--stride", "3", *map(str, arguments)]), 0)
        # One table for every block of the output, by the label of the row; a column for every group.
        return [{line[:script.NAME].strip(): [float("nan" if value == "-" else value) for value in line[script.NAME:].split()]
                 for line in part.splitlines()
                 if line[:script.NAME].strip() and line[script.NAME:].translate(str.maketrans("", "", ". -")).isdigit()}
                for part in output.getvalue().split("\n\n")]

    def test_the_three_groups(self):
        size, score, again, place, of_model, of_normal = self.run_script()
        # The model flags as much alone as with the normal, and so does the normal.
        np.testing.assert_allclose(size["% of the values flagged by the model"][:3], [50, 50, 50], atol=3)
        # What a score flags is at the top of it; what it does not flag is anywhere in it.
        self.assertGreater(score["in the score of the model"][0], 87)
        self.assertLess(score["in the score of the normal"][0], 60)
        self.assertGreater(score["in the score of the model"][1], 87)
        self.assertGreater(score["in the score of the normal"][1], 87)
        self.assertLess(score["in the score of the model"][2], 60)
        # The high values last a season: the score that flags one flags the next too, the other does not.
        np.testing.assert_allclose(again["by the model"], [100, 100, 0, 12.5], atol=3)
        np.testing.assert_allclose(again["by the normal"], [0, 100, 100, 12.5], atol=3)
        np.testing.assert_allclose(place[f"south of {script.SPLIT}N"], [100, 0, 0, 50], atol=3)
        np.testing.assert_allclose(place["Jun-Aug"][:3], [100, 0, 0], atol=3)
        np.testing.assert_allclose(place["Dec-Feb"][:3], [0, 100, 0], atol=3)
        np.testing.assert_allclose(place["Mar-May"][:3], [0, 0, 100], atol=3)
        # The error of the model is rain where the model alone flags, pressure and wind elsewhere.
        np.testing.assert_allclose(of_model["rain"][:3], [100, 0, 0], atol=3)
        np.testing.assert_allclose(of_model["pressure"][:3], [0, 75, 75], atol=3)
        np.testing.assert_allclose(of_normal["wind"], [100, 100, 100, 100], atol=0.1)

    def test_the_weights_of_the_run(self):
        self.paths[2] = self.write_run("weighted", errors=self.model_errors,
                                       weights={"rain": 1.0, "pressure": 1.0, "wind": 3.0})
        of_model = self.run_script()[4]
        np.testing.assert_allclose(of_model["pressure"][1:3], [50, 50], atol=3)
        np.testing.assert_allclose(of_model["wind"][1:3], [50, 50], atol=3)

    def test_the_error_of_the_normal_on_the_scale_of_a_rebalanced_score(self):
        # The data are three times as far from the normal in the pressure as in the wind, in squared error.
        errors = np.zeros((len(TIMES), 12, 3), dtype=np.float32)
        errors[..., 1], errors[..., 2] = 3, 1
        self.paths[3] = self.write_run("uneven", errors=errors)
        np.testing.assert_allclose(self.run_script()[5]["pressure"], [75] * 4, atol=0.1)
        balanced = self.write_run("balanced")
        manifest = json.loads((balanced / "metadata.json").read_text(encoding="utf-8"))
        (balanced / "metadata.json").write_text(json.dumps(
            {**manifest, "feature_means": {"rain": 1.0, "pressure": 3.0, "wind": 1.0}}), encoding="utf-8")
        tables = self.run_script("--normal-scale-dir", balanced)
        np.testing.assert_allclose(tables[5]["pressure"], [50] * 4, atol=0.1)
        np.testing.assert_allclose(tables[5]["wind"], [50] * 4, atol=0.1)
        # The error of the model keeps the weights of its run.
        np.testing.assert_allclose(tables[4]["pressure"][1:3], [75, 75], atol=3)
        with self.assertRaisesRegex(ValueError, "not a run rebalanced"):
            self.run_script("--normal-scale-dir", self.paths[3])

    def test_the_error_of_the_normal_from_the_cache(self):
        # The data are away from the normal in the pressure alone, and the run of the normal has lost its errors.
        self.standardised[..., 2] = 0
        self.standardised[CONTEXT:, :, 1] = 1
        np.save(self.cache / "test.npy", self.standardised)
        self.paths[3] = self.write_run("lost")
        of_normal = self.run_script()[5]
        np.testing.assert_allclose(of_normal["pressure"], [100, 100, 100, 100], atol=0.1)
        # A run whose answer was the previous value: its error is the change, and only the rain changes.
        self.standardised[CONTEXT::2, :, 0] = 5
        np.save(self.cache / "test.npy", self.standardised)
        before = self.write_run("lost_previous")
        manifest = json.loads((before / "metadata.json").read_text(encoding="utf-8"))
        (before / "metadata.json").write_text(json.dumps({**manifest, "previous_timestamps": 1}), encoding="utf-8")
        self.paths[3] = before
        of_previous = self.run_script()[5]
        np.testing.assert_allclose(of_previous["rain"], [100, 100, 100, 100], atol=1)

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "odd"):
            self.run_script("--stride", "2")
        shorter = self.write_run("shorter", scores=self.normal[:-8], times=TIMES[:-8])
        with self.assertRaisesRegex(ValueError, "same timestamps"):
            self.run_script(paths=[self.paths[0], shorter, *self.paths[2:]])
        (self.cache / "test.npy").unlink()
        with self.assertRaisesRegex(ValueError, "cache is not there"):
            self.run_script(paths=[*self.paths[:3], self.write_run("lost")])


if __name__ == "__main__":
    unittest.main()
