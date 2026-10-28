"""The comparison must say how much the detectors agree and what their majority flags."""
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

from scripts import compare_era5_detectors as script

TIMES = pd.date_range("2005-01-01", periods=400, freq="3h")
CELLS = 25
GRID = {"latitudes": [45.0, 44.5, 44.0, 43.5, 43.0], "longitudes": [7.0, 7.5, 8.0, 8.5, 9.0],
        "rows": np.repeat(np.arange(5), 5).tolist(), "cols": np.tile(np.arange(5), 5).tolist(), "spacing": 0.5}


class CompareDetectorsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        base = rng.random((len(TIMES), CELLS))
        # A and B are the same score; C is that score on the first half of the cells and noise on the other; D is
        # noise. Each flags its highest 10%.
        self.scores = {"A": base, "B": base * 2 + 1, "C": base.copy(), "D": rng.random((len(TIMES), CELLS))}
        self.scores["C"][:, CELLS // 2:] = rng.random((len(TIMES), CELLS - CELLS // 2))
        self.scores["D"][3, 4] = np.nan
        self.runs = []
        for name, values in self.scores.items():
            run = self.root / name
            (run / "scores").mkdir(parents=True)
            np.save(run / "scores" / "anomaly_mean.npy", values.astype(np.float32))
            np.save(run / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
            (run / "metadata.json").write_text(json.dumps(
                {"status": "complete", "grid": GRID, "anomaly_mean_file": "scores/anomaly_mean.npy"}), encoding="utf-8")
            self.runs.append(f"{name}={run}")

    def run_script(self, *arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main([*map(str, arguments)]), 0)
        return output.getvalue()

    def table(self, text, title):
        lines = text.split(title)[1].strip().splitlines()[1:5]
        return {line.split()[0]: [None if value == "-" else float(value) for value in line.split()[1:]] for line in lines}

    def test_agreement_of_the_detectors_and_their_majority(self):
        vote = self.root / "vote"
        figure = self.root / "jaccard.png"
        text = self.run_script("--runs", *self.runs, "--top-percent", "10", "--vote-dir", vote, "--figure", figure)
        overlap = self.table(text, "share of the flags of the row also flagged by the column, %")
        self.assertAlmostEqual(overlap["A"][1], 100, delta=1)  # A and B flag the same values
        self.assertAlmostEqual(overlap["A"][3], 10, delta=4)  # D is independent: about 10% by chance
        self.assertGreater(overlap["A"][2], 45)  # C follows A on half of the cells
        jaccard = self.table(text, "Jaccard index of the flags, %")
        self.assertAlmostEqual(jaccard["A"][1], 100, delta=1)
        self.assertAlmostEqual(jaccard["B"][0], jaccard["A"][1])
        rows = {line.split()[0]: [float(value) for value in line.split()[1:]] for line in text.splitlines()
                if line[:1] in "ABCD" and len(line.split()) == 4}
        for name in "ABCD":
            self.assertAlmostEqual(rows[name][0], 10, delta=2)  # each its highest 10%, threshold read on 1 timestamp in 10
        self.assertAlmostEqual(rows["A"][2], 0)  # B always flags with A: A is never alone
        self.assertGreater(rows["D"][2], 50)  # D mostly alone
        # The majority needs C or D beside A and B: C on the half of the cells where it follows A, chance on the
        # other half (about 1 - 0.9 * 0.9 = 19%): about 50 + 50 * 0.19 = 60% of the flags of A.
        self.assertAlmostEqual(rows["A"][1], 60, delta=8)
        self.assertIn("majority: 3 of 4", text)
        # The vote run: the number of detectors flagging each value, not finite where one detector is not.
        count = np.load(vote / "scores" / "anomaly_mean.npy")
        flags = np.stack([self.scores[name] > np.nanpercentile(self.scores[name][::10], 90) for name in "ABCD"])
        expected = flags.sum(axis=0).astype(float)
        expected[3, 4] = np.nan
        np.testing.assert_array_equal(count, expected.astype(np.float32))
        metadata = json.loads((vote / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual((metadata["vote"]["votes_needed"], metadata["grid"]), (3, GRID))
        histogram = [float(value) for value in text.split("values flagged by that many detectors, %")[1].strip()
                     .splitlines()[1].split()]
        self.assertAlmostEqual(sum(histogram), 100, delta=0.05)
        self.assertTrue(figure.is_file())

    def test_needs_three_detectors_and_a_valid_vote(self):
        with self.assertRaises(ValueError):
            self.run_script("--runs", *self.runs[:2])
        with self.assertRaises(ValueError):
            self.run_script("--runs", *self.runs, "--votes", "5")


if __name__ == "__main__":
    unittest.main()
