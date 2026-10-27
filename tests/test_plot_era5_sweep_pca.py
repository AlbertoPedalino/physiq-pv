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
