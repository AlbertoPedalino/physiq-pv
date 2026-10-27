"""The replay must put the runs where the W&B controller puts them and give the failed ones the worst value."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts import replay_era5_sweep_bayes as script

PARAMETERS = {"generator_learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
              "discriminator_generator_update_ratio": {"values": ["1:1", "2:1", "1:2"]},
              "epochs": {"value": 6}}


class ReplayTests(unittest.TestCase):
    def test_points_in_the_unit_cube(self):
        names, to_unit = script.normalizer(PARAMETERS)
        self.assertEqual(names, ["discriminator_generator_update_ratio", "generator_learning_rate"])  # no constant
        self.assertTrue(np.allclose(to_unit({"generator_learning_rate": 1e-3,
                                             "discriminator_generator_update_ratio": "2:1"}), [2 / 3, 0.5]))

    def test_failed_runs_count_as_the_worst(self):
        runs = [{"name": "a", "state": "finished", "point": np.array([0.1]), "history": [5.0, 3.0]},
                {"name": "b", "state": "failed", "point": np.array([0.5]), "history": [2.0]},
                {"name": "c", "state": "crashed", "point": np.array([0.9]), "history": []},
                {"name": "d", "state": "running", "point": np.array([0.3]), "history": [1.0]}]
        points, values, worst = script.gp_data(runs, 4)
        self.assertEqual(worst, 5.0)  # the largest maximum of the histories
        self.assertEqual(values.tolist(), [3.0, 5.0, 5.0])  # finished: its minimum; the running one is left out
        points, values, _ = script.gp_data(runs, 4, skip={"c"})
        self.assertEqual(len(values), 2)

    def test_replay_and_table(self):
        rng = np.random.default_rng(0)
        runs = [{"name": f"run-{index}", "state": "failed" if index == 4 else "finished",
                 "point": rng.random(2), "history": [float(rng.random())]} for index in range(8)]
        rows = script.replay(runs, 3, 0.1)
        self.assertEqual([row["run"] for row in rows], list(range(3, 9)))
        for row in rows:
            self.assertGreaterEqual(row["gain_max"], 0)
            self.assertTrue(0 <= row["percentile_proposed"] <= 100)
        with patch.object(script, "read_sweep", return_value=(["a", "b"], runs, {"goal": "minimize"})):
            printed = io.StringIO()
            with redirect_stdout(printed):
                self.assertEqual(script.main(["--from-run", "5"]), 0)
        self.assertIn("run-5", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
