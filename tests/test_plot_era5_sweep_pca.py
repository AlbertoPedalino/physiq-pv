"""The sweep plot must read the four swept parameters and draw them on their principal plane."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts import plot_era5_sweep_pca as script


class SweepPcaTests(unittest.TestCase):
    def test_coordinates_of_a_run(self):
        config = {"generator_learning_rate": 1e-3, "discriminator_lr_ratio": 10.0,
                  "generator_reconstruction_weight": 100.0, "discriminator_generator_update_ratio": "1:2"}
        self.assertEqual(script.coordinates(config), [-3.0, 1.0, 2.0, -1.0])
        self.assertIsNone(script.coordinates({"generator_learning_rate": 1e-3}))

    def test_the_plane_keeps_the_one_direction_of_the_points(self):
        # Four parameters moving together: one component holds all the variance.
        points = np.outer(np.arange(5.0), [1, 2, -1, 0.5])
        _, plane, variance, loadings = script.principal_plane(points)
        self.assertAlmostEqual(variance[0], 1.0)
        self.assertTrue(np.allclose(plane[:, 1], 0, atol=1e-9))
        self.assertTrue(np.allclose(np.abs(loadings[:, 0]), 0.5))  # standardised: equal weights

    def test_narrowing_of_the_draws(self):
        # The first half at the two ends of every range, the second half all on one value.
        ends = np.array([[low for _, _, (low, _) in script.PARAMETERS], [high for _, _, (_, high) in script.PARAMETERS]])
        points = np.vstack([ends, ends, np.repeat(ends[:1], 4, axis=0)])
        spread = script.narrowing(points)
        self.assertTrue(np.allclose(spread[:3, 0], np.sqrt(3)))  # half of the range over range / sqrt(12)
        self.assertTrue(np.allclose(spread[:, 1], 0))

    def test_first_component_by_chance(self):
        # Independent draws: about a quarter of the variance with many runs, more with few.
        many, _ = script.chance_first_component(2000, trials=20)
        few, high = script.chance_first_component(21, trials=500)
        self.assertLess(abs(many - 0.25), 0.03)
        self.assertGreater(few, 0.32)
        self.assertGreater(high, few)

    def test_plot_and_table(self):
        rows = [(f"run-{index}", "finished" if index != 3 else "failed",
                 [-3.5 + 0.1 * index, 0.1 * index, 2 + 0.05 * index, (-1, 0, 1)[index % 3]],
                 np.nan if index == 3 else 0.01 / (1 + index % 4)) for index in range(8)]
        with tempfile.TemporaryDirectory() as temp, patch.object(script, "read_sweep", return_value=rows):
            output = Path(temp) / "sweep.png"
            printed = io.StringIO()
            with redirect_stdout(printed):
                self.assertEqual(script.main(["--output", str(output)]), 0)
            self.assertTrue(output.is_file())
        text = printed.getvalue()
        self.assertIn("runs with parameters: 8, with the objective: 7", text)
        best = next(line for line in text.splitlines() if line.endswith("best"))
        self.assertIn("run-7", best)  # 0.01 / 4, the lowest of the finished runs


if __name__ == "__main__":
    unittest.main()
