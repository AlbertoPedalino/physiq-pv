"""The combination check must find the values where the variables take together a combination they rarely take."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts import check_era5_combination as script

TIMES = pd.date_range("2005-01-01", "2005-12-31 21:00", freq="3h")
GRID = {"latitudes": [45.0, 44.5], "longitudes": [7.0, 7.5, 8.0], "rows": [0, 0, 0, 1, 1, 1], "cols": [0, 1, 2, 0, 1, 2],
        "spacing": 0.5}
CONTEXT = 4
ODD, FAR = TIMES.day == 10, TIMES.day == 20  # days of a strange combination, and of one variable far away


def correlated(rng, shape):
    """Three variables: the first two move together, the third by itself."""
    common = rng.normal(size=shape)
    return np.stack((common + 0.1 * rng.normal(size=shape), common + 0.1 * rng.normal(size=shape),
                     rng.normal(size=shape)), axis=-1).astype(np.float32)


class CombinationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        cache = self.root / "cache"
        cache.mkdir()
        np.save(cache / "train.npy", correlated(rng, (3000, 6)))
        test = correlated(rng, (CONTEXT + len(TIMES), 6))
        # On some days the first two variables go apart, each by little; on others the third goes far.
        test[CONTEXT:][ODD, :, 0], test[CONTEXT:][ODD, :, 1] = 1.2, -1.2
        test[CONTEXT:][FAR, :, 2] = 6
        np.save(cache / "test.npy", test)
        (cache / "cache.json").write_text(json.dumps({"inputs": {}, "built": {"scale": [1, 1, 1]}}), encoding="utf-8")
        self.run = self.root / "run"
        self.run.mkdir()
        (self.run / "metadata.json").write_text(json.dumps(
            {"status": "complete", "grid": GRID, "preparation": {"features": ["a", "b", "c"]},
             "backend": {"seasonal_normalization": {"mean_file": str(cache / "seasonal_mean.npy")}}}), encoding="utf-8")
        model, other = (rng.random((len(TIMES), 6)).astype(np.float32) * 0.1 for _ in range(2))
        model[ODD] += 1
        other[FAR] += 1
        self.scores = [self.write_run("run_model", model), self.write_run("run_other", other)]

    def write_run(self, name, scores):
        run = self.root / name
        (run / "scores").mkdir(parents=True)
        np.save(run / "scores" / "anomaly_mean.npy", scores)
        np.save(run / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
        (run / "metadata.json").write_text(json.dumps({"status": "complete", "grid": GRID}), encoding="utf-8")
        return run

    def test_the_strange_combination_and_the_far_variable(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(script.main(["--run-dir", str(self.run), "--score-dir", str(self.scores[0]), "--by-dirs",
                                          str(self.scores[1]), "--top-percent", "3"]), 0)
        rows = {line[:script.NAME].strip(): [float(value) for value in line[script.NAME:].split()]
                for line in output.getvalue().splitlines() if line.startswith(("score alone", "others, not", "all values"))}
        alone, others = rows["score alone, none of the others"], rows["others, not the score"]
        # Where the first score flags, the joint state is unusual and the separate one is not: a combination.
        self.assertGreater(alone[1], 90)
        self.assertLess(alone[2], 10)
        self.assertGreater(alone[3], 90)
        # Where the other flags, one variable is far: the separate distance is unusual, no combination.
        self.assertGreater(others[2], 90)
        self.assertLess(others[3], 10)
        self.assertAlmostEqual(rows["all values"][1], 5, delta=2)

    def test_local_covariances_find_what_is_strange_only_where_it_is(self):
        # The first two variables move together in the north row and against each other in the south row. In the
        # south, moving together is the strange combination; the domain as a whole cannot tell it.
        rng = np.random.default_rng(1)
        train = correlated(rng, (30000, 6))
        train[:, 3:, 1] = -train[:, 3:, 1]
        test = correlated(rng, (CONTEXT + len(TIMES), 6))
        test[:, 3:, 1] = -test[:, 3:, 1]
        test[CONTEXT:][ODD, 3:, 0], test[CONTEXT:][ODD, 3:, 1] = 1.2, 1.2
        cache = self.root / "cache"
        np.save(cache / "train.npy", train)
        np.save(cache / "test.npy", test)
        model = np.random.default_rng(2).random((len(TIMES), 6)).astype(np.float32) * 0.1
        model[np.ix_(ODD, [3, 4, 5])] += 1
        np.save(self.scores[0] / "scores" / "anomaly_mean.npy", model)

        class Cubes:
            train_timestamps = pd.date_range("1990-01-01", periods=len(train), freq="3h")

            def close(self):
                pass

        arguments = ["--run-dir", str(self.run), "--score-dir", str(self.scores[0]), "--by-dirs", str(self.scores[1]),
                     "--top-percent", "1.5"]
        results = {}
        for name, extra in (("domain", []), ("local", ["--local-degrees", "0.5", "--prepared-dir", "unused"])):
            output = io.StringIO()
            with patch.object(script, "load_prepared", return_value=(Cubes(), None, {})), redirect_stdout(output):
                self.assertEqual(script.main([*arguments, *extra]), 0)
            results[name] = {line[:script.NAME].strip(): [float(value) for value in line[script.NAME:].split()]
                             for line in output.getvalue().splitlines() if line.startswith("score alone")}
        self.assertGreater(results["local"]["score alone, none of the others"][3], 90)
        self.assertLess(results["domain"]["score alone, none of the others"][3], 50)
        with self.assertRaisesRegex(ValueError, "needs --prepared-dir"):
            script.main([*arguments, "--local-degrees", "5"])

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "multiple of 3"):
            script.main(["--run-dir", str(self.run), "--score-dir", str(self.scores[0]), "--by-dirs", str(self.scores[1]),
                         "--change-hours", "4"])


if __name__ == "__main__":
    unittest.main()
