"""The check of the dropout spread must say whether the draws can undo the flags."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts import check_era5_dropout_spread as script

SHAPE = (3000, 20)


class DropoutSpreadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_run(self, name, mean, spread, *, component="generator"):
        run = self.root / name
        (run / "scores").mkdir(parents=True)
        np.save(run / "scores" / f"{component}_mean.npy", mean.astype(np.float32))
        manifest = {"status": "complete", f"{component}_mean_file": f"scores/{component}_mean.npy"}
        if spread is not None:
            np.save(run / "scores" / f"{component}_std.npy", spread.astype(np.float32))
            manifest[f"{component}_std_file"] = f"scores/{component}_std.npy"
        (run / "metadata.json").write_text(json.dumps(manifest), encoding="utf-8")
        return run

    def run_script(self, run, *arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--run-dir", str(run), "--score-component", "generator", "--stride", "1",
                                          *map(str, arguments)]), 0)
        text = output.getvalue()
        rows = {line[:script.NAME].strip(): line[script.NAME:].split() for line in text.splitlines() if line[:1] != "["}
        return text, rows

    def test_a_small_spread_keeps_the_flags_and_a_large_one_undoes_them(self):
        rng = np.random.default_rng(0)
        mean = rng.exponential(size=SHAPE)
        _, small = self.run_script(self.write_run("small", mean, 0.001 * mean))
        self.assertGreater(float(small["flagged, still flagged one spread lower"][0].rstrip("%")), 98)
        self.assertLess(float(small["not flagged, flagged one spread higher"][0].rstrip("%")), 1)
        # The spread grows with the score here: the correlation of their ranks is one.
        self.assertIn("rank correlation between score and spread: +1.000", _)
        _, large = self.run_script(self.write_run("large", mean, np.full(SHAPE, 5.0)))
        self.assertLess(float(large["flagged, still flagged one spread lower"][0].rstrip("%")), 5)
        self.assertGreater(float(large["not flagged, flagged one spread higher"][0].rstrip("%")), 90)
        # The band table: the highest band has the highest mean score.
        self.assertGreater(float(small["99.9-100"][0]), float(small["0-50"][0]))

    def test_the_flags_split_by_another_score(self):
        rng = np.random.default_rng(2)
        mean = rng.exponential(size=SHAPE)
        # The spread is small where another score flags too, and large where the score flags alone.
        other = np.where(np.arange(SHAPE[0])[:, None] % 2 == 0, mean, 0.0)
        spread = np.where(other > 0, 0.001 * mean, 10.0)
        run = self.write_run("model", mean, spread)
        (self.root / "other" / "scores").mkdir(parents=True)
        np.save(self.root / "other" / "scores" / "anomaly_mean.npy", other.astype(np.float32))
        (self.root / "other" / "metadata.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
        text, rows = self.run_script(run, "--by-dirs", self.root / "other")
        self.assertIn("the flags split by the other scores", text)
        self.assertAlmostEqual(float(rows["% of the flags"][0].rstrip("%")) + float(rows["% of the flags"][1].rstrip("%")),
                               100, delta=0.2)
        alone, together = (float(value.rstrip("%")) for value in rows["still flagged one spread lower"])
        self.assertLess(alone, 5)
        self.assertGreater(together, 95)

    def test_no_spread(self):
        mean = np.random.default_rng(1).random(SHAPE)
        text, _ = self.run_script(self.write_run("flat", mean, np.zeros(SHAPE)))
        self.assertIn("every spread is zero", text)
        with self.assertRaisesRegex(ValueError, "kept no spread"):
            self.run_script(self.write_run("none", mean, None))


if __name__ == "__main__":
    unittest.main()
