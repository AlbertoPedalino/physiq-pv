"""The spatial-outlier tests must compare each cell with the neighbours their papers define."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet

from scripts import score_era5_spatial_outliers as script

HEIGHT, WIDTH, VARIABLES, CONTEXT = 6, 8, 3, 4
ROWS, COLS = np.indices((HEIGHT, WIDTH)).reshape(2, -1)
PRESENT = np.ones(HEIGHT * WIDTH, dtype=bool)
PRESENT[[10, 27]] = False  # two places of the grid without a cell
GRID = {"latitudes": (45 - np.arange(HEIGHT) * .5).tolist(), "longitudes": (7 + np.arange(WIDTH) * .5).tolist(),
        "rows": ROWS[PRESENT].tolist(), "cols": COLS[PRESENT].tolist(), "spacing": 0.5}
CELLS = int(PRESENT.sum())
TIMES = pd.date_range("2005-01-01", periods=120, freq="3h")
SPIKE = (30, 20)  # timestamp and cell where one variable is far from its neighbours
SHIFT = 60  # timestamp where the whole field moves at once
COMBINED = (90, 33)  # timestamp and cell where every variable is somewhat away from its neighbours


def smooth_fields(steps, rng):
    """Fields that change slowly in space and time, with a little noise: [steps, cells, variables]."""
    time = np.arange(steps)[:, None, None]
    row, column = ROWS[PRESENT][None, :, None], COLS[PRESENT][None, :, None]
    phase = rng.uniform(0, 2 * np.pi, VARIABLES)[None, None, :]
    smooth = np.sin(0.4 * row + 0.3 * column + 0.05 * time + phase)
    return (smooth + 0.05 * rng.normal(size=(steps, CELLS, VARIABLES))).astype(np.float32)


class SpatialOutlierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        self.train = smooth_fields(400, rng)
        self.scored = smooth_fields(CONTEXT + len(TIMES), rng)
        self.scored[CONTEXT + SPIKE[0], SPIKE[1], 0] += 3.0
        self.scored[CONTEXT + SHIFT] += 1.5
        self.scored[CONTEXT + COMBINED[0], COMBINED[1]] += 0.6
        self.scored[CONTEXT + 5, 7, 2] = np.nan
        self.cache = self.root / "cache" / "abc"
        self.cache.mkdir(parents=True)
        np.save(self.cache / "train.npy", self.train)
        np.save(self.cache / "test.npy", self.scored)
        (self.cache / "cache.json").write_text(json.dumps({"inputs": {}, "built": {}}), encoding="utf-8")
        self.run = self.root / "run"
        self.run.mkdir()
        np.save(self.run / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
        (self.run / "metadata.json").write_text(json.dumps(
            {"status": "complete", "grid": GRID, "preparation": {"features": ["a", "b", "c"]},
             "backend": {"seasonal_normalization": {"mean_file": str(self.cache / "seasonal_mean.npy")}}}),
            encoding="utf-8")

    def rank(self, scores, time, cell):
        """Share of the finite scores below the score of (time, cell)."""
        finite = scores[np.isfinite(scores)]
        return (finite < scores[time, cell]).mean()

    def test_the_statistic_is_the_value_minus_the_mean_of_the_neighbours(self):
        layout = script.lattice_of(GRID)
        values = self.scored[:3].astype(np.float64)
        statistic = script.shekhar_statistic(values, [1], script.NEIGHBOURS["shekhar_spatial"], layout)[0]
        where = {(row, column): index for index, (row, column) in enumerate(zip(GRID["rows"], GRID["cols"]))}
        for cell in (0, 9, 15, CELLS - 1):  # a corner, next to a missing place, inside, the last corner
            row, column = GRID["rows"][cell], GRID["cols"][cell]
            neighbours = [where[(row + dr, column + dc)] for dr, dc in script.AROUND if (row + dr, column + dc) in where]
            np.testing.assert_allclose(statistic[cell], values[1, cell] - values[1, neighbours].mean(axis=0))
        self.assertEqual(len(script.AROUND), 8)
        temporal = script.shekhar_statistic(values, [1], script.NEIGHBOURS["shekhar_temporal"], layout)[0]
        np.testing.assert_allclose(temporal, values[1] - (values[0] + values[2]) / 2)

    def test_each_neighbourhood_sees_what_it_should(self):
        runs = script.score(self.run, self.root / "out", train_timestamps=200)
        scores = {name: np.load(run / "scores" / "anomaly_mean.npy") for name, run in runs.items()}
        for name in script.METHODS:
            self.assertEqual(scores[name].shape, (len(TIMES), CELLS))
            self.assertTrue(np.isnan(scores[name][5, 7]), name)  # a variable not finite: no score
        # One variable far from its neighbours: every Shekhar test sees it at the top.
        for name in script.NEIGHBOURS:
            self.assertGreater(self.rank(scores[name], *SPIKE), 0.999, name)
        # The whole field moving at once: the same for the cell and its neighbours, so not a spatial outlier,
        # but the cell departs from itself before and after.
        shifted = lambda name: np.nanmedian(scores[name][SHIFT]) / np.nanmedian(scores[name])
        self.assertLess(shifted("shekhar_spatial"), 1.5)
        self.assertGreater(shifted("shekhar_temporal"), 5)
        # Every variable somewhat away from its neighbours: the multivariate test sees the combination.
        self.assertGreater(self.rank(scores["chen_multivariate"], *COMBINED), 0.99)
        metadata = json.loads((runs["shekhar_spatial"] / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual((metadata["status"], metadata["grid"], metadata["anomaly_mean_file"]),
                         ("complete", GRID, "scores/anomaly_mean.npy"))
        self.assertEqual(len(metadata["spatial_outlier"]["neighbours"]), 8)
        self.assertEqual(len(json.loads((runs["shekhar_spatiotemporal"] / "metadata.json").read_text(
            encoding="utf-8"))["spatial_outlier"]["neighbours"]), 26)

    def test_the_test_parameters_come_from_the_training_years(self):
        layout = script.lattice_of(GRID)
        parameters = script.test_parameters(np.load(self.cache / "train.npy", mmap_mode="r"), layout, 398,
                                            np.random.default_rng(0))
        statistic = script.shekhar_statistic(self.train.astype(np.float64), np.arange(1, 399),
                                             script.NEIGHBOURS["shekhar_spatial"], layout)
        np.testing.assert_allclose(parameters["shekhar_spatial"][0], statistic.mean(axis=(0, 1)), atol=1e-10)
        np.testing.assert_allclose(parameters["shekhar_spatial"][1], statistic.std(axis=(0, 1)), rtol=1e-8)

    def test_the_multivariate_distance_is_that_of_the_mcd(self):
        differences = np.random.default_rng(1).normal(size=(300, 3))
        differences[7] = np.nan
        distance = script.robust_distance(differences, seed=0)
        finite = np.isfinite(differences).all(axis=1)
        expected = MinCovDet(random_state=0).fit(differences[finite]).mahalanobis(differences[finite])
        np.testing.assert_allclose(distance[finite], expected)
        self.assertTrue(np.isnan(distance[7]))

    def test_processes_give_the_same_scores(self):
        one = script.score(self.run, self.root / "one", ["chen_multivariate"])
        two = script.score(self.run, self.root / "two", ["chen_multivariate"], workers=2)
        np.testing.assert_array_equal(np.load(one["chen_multivariate"] / "scores" / "anomaly_mean.npy"),
                                      np.load(two["chen_multivariate"] / "scores" / "anomaly_mean.npy"))


if __name__ == "__main__":
    unittest.main()
