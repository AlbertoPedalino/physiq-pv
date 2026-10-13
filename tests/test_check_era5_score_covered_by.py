"""The coverage check must say how much of what a score flags the other scores flag, alone and together."""
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

from scripts import check_era5_score_covered_by as script

TIMES = pd.date_range("2005-01-01", "2005-12-31 21:00", freq="3h")
GRID = {"latitudes": [55.0, 45.0, 35.0, 25.0], "longitudes": [10.0, 10.5, 11.0],
        "rows": np.repeat(np.arange(4), 3).tolist(), "cols": np.tile(np.arange(3), 4).tolist(), "spacing": 0.5}
NAMES = ["rain", "pressure", "wind"]
WINTER, SPRING, SUMMER, AUTUMN = (np.isin(TIMES.month, months) for months in ((12, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)))
CELLS = [slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12)]
# Each score is high in four places and seasons, a quarter of the cells for a quarter of the year each.
TOP = "25"


class ScoreCoveredByTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)

        def score(*high):
            values = rng.random((len(TIMES), 12)).astype(np.float32) * 0.1
            for season, cells in high:
                values[season, cells] += 1
            return values
        # The score is high alone, with the first other, with the second, and with both.
        alone, first, second, both = (WINTER, CELLS[0]), (SPRING, CELLS[1]), (SUMMER, CELLS[2]), (AUTUMN, CELLS[3])
        self.paths = [self.write_run("run_model", score(alone, first, second, both)),
                      self.write_run("run_normal", score(first, both, (SPRING, CELLS[0]), (WINTER, CELLS[2]))),
                      self.write_run("run_previous", score(second, both, (SUMMER, CELLS[0]), (AUTUMN, CELLS[1])))]
        # The error behind the score is all rain in the cells where it is high alone, pressure elsewhere.
        errors = np.zeros((len(TIMES), 12, 3), dtype=np.float32)
        errors[:, CELLS[0], 0] = 1
        errors[:, 3:, 1] = 1
        self.errors = self.write_run("run", errors=errors)

    def write_run(self, name, scores=None, *, errors=None, times=TIMES):
        (self.root / name / "scores").mkdir(parents=True)
        if scores is not None:
            np.save(self.root / name / "scores" / "anomaly_mean.npy", scores)
        if errors is not None:
            np.save(self.root / name / "scores" / "feature_scores.npy", errors)
        np.save(self.root / name / "test_timestamps.npy", times.as_unit("ns").asi8)
        (self.root / name / "metadata.json").write_text(json.dumps(
            {"status": "complete", "grid": GRID, "preparation": {"features": NAMES}}), encoding="utf-8")
        return self.root / name

    def run_script(self, *arguments, others=None):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--score-dir", str(self.paths[0]), "--by-dirs",
                                          *map(str, others or self.paths[1:]), "--top-percent", TOP, "--stride", "3",
                                          *map(str, arguments)]), 0)
        text = output.getvalue()
        rows = {line[:script.NAME].strip(): [float(value) for value in line[script.NAME:].split()]
                for line in text.splitlines()
                if line[script.NAME:].strip() and line[script.NAME:].replace(".", "").replace(" ", "").isdigit()}
        return text, rows

    def test_what_the_others_flag_alone_and_together(self):
        text, rows = self.run_script()
        self.assertIn("score: model", text)
        np.testing.assert_allclose(rows["normal"], [50], atol=3)
        np.testing.assert_allclose(rows["previous"], [50], atol=3)
        np.testing.assert_allclose(rows["at least one of them"], [75], atol=3)
        np.testing.assert_allclose(rows["none of them"], [25], atol=3)
        # The two others are high in seven places and seasons together, and the score in three of them.
        np.testing.assert_allclose(rows[""], [100 * 3 / 7], atol=3)
        # The high values last a season.
        np.testing.assert_allclose(rows["of those none of them flags"], [100], atol=3)
        np.testing.assert_allclose(rows["of those at least one of them flags"], [100], atol=3)
        self.assertNotIn("what the error", text)

    def test_one_other_score_and_the_error_by_variable(self):
        text, rows = self.run_script("--feature-errors-dir", self.errors, others=self.paths[1:2])
        np.testing.assert_allclose(rows["at least one of them"], [50], atol=3)
        np.testing.assert_allclose(rows["none of them"], [50], atol=3)
        # Where the first other does not flag: the season alone, all rain, and that with the second, all pressure.
        np.testing.assert_allclose(rows["rain"], [50, 0, 25], atol=3)
        np.testing.assert_allclose(rows["pressure"], [50, 100, 75], atol=3)
        np.testing.assert_allclose(rows["wind"], [0, 0, 0], atol=0.1)

    def test_one_part_of_the_score_of_a_training_run(self):
        # The score of the other tests is here the reconstruction error of the run; the sum of its parts is flat.
        run = self.paths[0]
        (run / "scores" / "anomaly_mean.npy").rename(run / "scores" / "generator_mean.npy")
        np.save(run / "scores" / "anomaly_mean.npy", np.zeros((len(TIMES), 12), dtype=np.float32))
        manifest = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        (run / "metadata.json").write_text(json.dumps({**manifest, "generator_mean_file": "scores/generator_mean.npy"}),
                                           encoding="utf-8")
        text, rows = self.run_script("--score-component", "generator")
        self.assertIn("score: model, generator", text)
        np.testing.assert_allclose(rows["at least one of them"], [75], atol=3)
        with self.assertRaisesRegex(ValueError, "does not contain discriminator"):
            self.run_script("--score-component", "discriminator")

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "odd"):
            self.run_script("--stride", "2")
        shorter = self.write_run("shorter", np.zeros((len(TIMES) - 8, 12), dtype=np.float32), times=TIMES[:-8])
        with self.assertRaisesRegex(ValueError, "same timestamps"):
            self.run_script(others=[shorter])


if __name__ == "__main__":
    unittest.main()
