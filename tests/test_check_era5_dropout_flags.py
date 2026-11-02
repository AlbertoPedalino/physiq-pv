"""The dropout spread of the flags, read with no label: solidity, classes at equal score, position across the events."""
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

from scripts import check_era5_dropout_flags as script

HEIGHT = WIDTH = 20
CELLS = HEIGHT * WIDTH
CONTEXT = 4
TIMES = pd.date_range("2005-01-01", periods=64, freq="3h")
GRID = {"latitudes": (45 - np.arange(HEIGHT) * .5).tolist(), "longitudes": (7 + np.arange(WIDTH) * .5).tolist(),
        "rows": np.repeat(np.arange(HEIGHT), WIDTH).tolist(), "cols": np.tile(np.arange(WIDTH), HEIGHT).tolist()}
cell = lambda row, column: row * WIDTH + column
BLOCK = [cell(r, c) for r in range(2, 12) for c in range(2, 12)]


class DropoutFlagsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        self.inputs = (0.1 * rng.normal(size=(CONTEXT + len(TIMES), CELLS, 3))).astype(np.float32)
        self.score = 0.5 * rng.random((len(TIMES), CELLS)).astype(np.float32)  # all below the threshold of the reference
        self.reference = rng.random((len(TIMES), CELLS)).astype(np.float32)
        self.spread = np.full((len(TIMES), CELLS), 0.01, dtype=np.float32)
        self.normal = 0.5 * rng.random((len(TIMES), CELLS)).astype(np.float32)
        self.raw = rng.random((len(TIMES), CELLS)).astype(np.float32)
        # Half of the flags: the draws agree and a variable is far from normal; the other half: they disagree and not.
        for step in (10, 11, 20, 21):
            self.score[step, BLOCK] = rng.uniform(10, 11, len(BLOCK))
            agree = step < 15
            self.spread[step, BLOCK] = rng.uniform(0.01, 0.02, len(BLOCK)) if agree else rng.uniform(20, 21, len(BLOCK))
            if agree:
                self.inputs[CONTEXT + step, BLOCK, 1] = 5
                self.normal[step, BLOCK] = 50
        cache = self.root / "cache"
        cache.mkdir()
        np.save(cache / "test.npy", self.inputs)
        self.inputs_run = self.root / "model"
        self.inputs_run.mkdir()
        (self.inputs_run / "metadata.json").write_text(json.dumps(
            {"status": "complete", "grid": GRID, "preparation": {"features": ["a", "b", "c"]},
             "backend": {"seasonal_normalization": {"mean_file": str(cache / "seasonal_mean.npy")}}}), encoding="utf-8")

    def write_run(self, name, values, spread=None, reference=None):
        run = self.root / name
        (run / "scores").mkdir(parents=True)
        np.save(run / "scores" / "anomaly_mean.npy", values)
        np.save(run / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
        metadata = {"status": "complete", "grid": GRID, "anomaly_mean_file": "scores/anomaly_mean.npy"}
        if spread is not None:
            np.save(run / "scores" / "anomaly_std.npy", spread)
            metadata["anomaly_std_file"] = "scores/anomaly_std.npy"
        if reference is not None:
            metadata["threshold_reference"] = str(reference)
        (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return run

    def test_the_ratio_of_the_spread_to_the_error(self):
        reference = self.write_run("reference", self.reference)
        events = self.write_run("events", self.score, self.spread, reference)
        normal = self.write_run("normal", self.normal, reference=reference)
        error = 1 + self.score  # the same size within the block: the ratio follows the spread
        np.save(self.root / "raw_spread.npy", self.spread * 0.1)
        raw = self.write_run("raw", error)
        csv = self.root / "dropout.csv"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(script.main(["--events", f"G={events}", "--spreads", f"grezza={self.root / 'raw_spread.npy'}",
                                          "--normal", f"N={normal}", "--raw-error", f"E={raw}", "--ratio-of", "grezza",
                                          "--inputs-run", str(self.inputs_run), "--bands", "2", "--classes", "2",
                                          "--limit", "4", "--csv", str(csv)]), 0)
        table = pd.read_csv(csv).set_index(["spread", "bands", "class"]).sort_index()
        self.assertEqual(sorted({key[:2] for key in table.index}),
                         [("grezza", "raw error"), ("grezza", "score"), ("ratio", "raw error"), ("ratio", "score"), ("score", "score")])
        first, second = table.loc[("ratio", "raw error", 1)], table.loc[("ratio", "raw error", 2)]
        self.assertLess(first["median spread"], 0.01)
        self.assertGreater(second["median spread"], 0.1)
        self.assertGreater(first["normal flagged %"], 95)
        self.assertLess(second["normal flagged %"], 5)
        by_position = pd.read_csv(self.root / "dropout_positions.csv")
        self.assertIn("ratio value", by_position.columns)

    def test_classes_positions_and_solidity(self):
        reference = self.write_run("reference", self.reference)
        events = self.write_run("events", self.score, self.spread, reference)
        normal = self.write_run("normal", self.normal, reference=reference)
        raw = self.root / "raw.npy"
        np.save(raw, self.raw)
        csv = self.root / "dropout.csv"
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--events", f"G={events}", "--spreads", f"raw={raw}", "--normal", f"N={normal}",
                                          "--inputs-run", str(self.inputs_run), "--top-percent", "5", "--bands", "2",
                                          "--classes", "2", "--limit", "4", "--csv", str(csv)]), 0)
        table = pd.read_csv(csv).set_index(["spread", "class"])
        agree, disagree = table.loc[("score", 1)], table.loc[("score", 2)]
        self.assertEqual(agree["values"] + disagree["values"], 4 * len(BLOCK))
        self.assertLess(agree["median spread"], 1)
        self.assertGreater(disagree["median spread"], 4)
        self.assertGreater(agree["normal flagged %"], 95)  # the median of a band can fall between the two groups
        self.assertLess(disagree["normal flagged %"], 5)
        self.assertLess(agree["none beyond 2 %"], 5)
        self.assertGreater(disagree["none beyond 2 %"], 95)
        self.assertAlmostEqual(table.loc[("raw", 1), "normal flagged %"], table.loc[("raw", 2), "normal flagged %"], delta=30)
        by_position = pd.read_csv(self.root / "dropout_positions.csv").set_index("position")
        self.assertEqual(by_position.loc[-1, "values"], 4 * 36)  # the border of a 10 x 10 block
        self.assertIn("raw percentile", by_position.columns)
        # The flags with a large spread do not stay one spread lower: half of them, at the score of the flags.
        self.assertIn("still flagged one spread lower 50.0%", output.getvalue())


if __name__ == "__main__":
    unittest.main()
