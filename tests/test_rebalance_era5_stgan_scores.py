"""The rebalanced reconstruction error must give every variable the same weight and be a run `events` accepts."""
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

from scripts import rebalance_era5_stgan_scores as script
from scripts import run_era5_stgan as cli

TIMES = pd.date_range("2005-01-01", "2007-12-31 21:00", freq="3h")
GRID = {"latitudes": [40.0, 39.5, 39.0], "longitudes": [10.0, 10.5, 11.0, 11.5],
        "rows": np.repeat(np.arange(3), 4).tolist(), "cols": np.tile(np.arange(4), 3).tolist(), "spacing": 0.5}
NAMES = ["total_cloud_cover", "2m_temperature", "total_precipitation"]
UNUSUAL = pd.Timestamp("2006-07-15")
TOP = "share_of_the_new_score_in_its_top_1_percent"


def by_day(values):
    return pd.Series(values.mean(axis=1), index=TIMES.normalize()).groupby(level=0).mean()


class RebalanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(0)
        shape = (len(TIMES), 12)
        # The first variable has an error a hundred times larger than the other two. The third also
        # has rare large errors. The second has one unusual day.
        self.errors = np.stack([rng.random(shape), rng.random(shape) * 0.01,
                                rng.random(shape) * 0.01 + (rng.random(shape) > 0.999) * 0.5], axis=2).astype(np.float32)
        self.errors[TIMES.normalize() == UNUSUAL, :, 1] += 0.05
        self.write_run("run", self.errors)

    def write_run(self, name, errors, *, status="complete", features=NAMES, weights=None):
        (self.root / name / "scores").mkdir(parents=True, exist_ok=True)
        if errors is not None:
            np.save(self.root / name / "scores" / "feature_scores.npy", errors)
        np.save(self.root / name / "test_timestamps.npy", TIMES.as_unit("ns").asi8)
        (self.root / name / "metadata.json").write_text(json.dumps(
            {"status": status, "grid": GRID, "preparation": {"features": features},
             **({"backend": {"reconstruction_feature_weights": weights}} if weights is not None else {})}),
            encoding="utf-8")

    def test_mean_scale_gives_equal_shares_and_shows_what_the_plain_error_hides(self):
        output = io.StringIO()
        with redirect_stdout(output), patch.object(script, "DIAGNOSTIC_STRIDE", 1):
            script.main(["--run-dir", str(self.root / "run"), "--output-dir", str(self.root / "balanced"),
                         "--scale-stride", "1"])
            script.main(["--run-dir", str(self.root / "run"), "--output-dir", str(self.root / "plain"), "--scale", "none"])
            cli.main(["climatology", "--run-dir", str(self.root / "balanced"), "--output-dir", str(self.root / "clim"),
                      "--min-samples", "10"])
            cli.main(["events", "--run-dir", str(self.root / "balanced"), "--output-dir", str(self.root / "events"),
                      "--top-percent", "5"])
        manifest = json.loads((self.root / "balanced" / "metadata.json").read_text(encoding="utf-8"))
        balanced = np.load(self.root / "balanced" / "scores" / "anomaly_mean.npy")
        plain = np.load(self.root / "plain" / "scores" / "anomaly_mean.npy")
        means = self.errors.reshape(-1, 3).mean(axis=0, dtype=np.float64)
        np.testing.assert_allclose(balanced, (self.errors / means).mean(axis=2), rtol=1e-4)
        np.testing.assert_allclose(plain, self.errors.mean(axis=2, dtype=np.float64), rtol=1e-5)
        np.testing.assert_allclose(list(manifest["feature_means"].values()), means, rtol=1e-6)
        # Before, the first variable is almost all of the error; after, the three weigh the same.
        self.assertGreater(manifest["share_of_the_plain_error"]["total_cloud_cover"], 0.95)
        np.testing.assert_allclose(list(manifest["share_of_the_new_score"].values()), 1 / 3, atol=1e-4)
        # With the mean as scale the variable with rare large errors takes more of the highest scores.
        self.assertGreater(manifest[TOP]["total_precipitation"], 0.45)
        # The unusual day of the second variable: lost in the plain error, the highest in the rebalanced one.
        self.assertLess(by_day(plain).rank(pct=True)[UNUSUAL], 0.99)
        self.assertEqual(by_day(balanced).idxmax(), UNUSUAL)
        for step in ("clim", "events"):
            self.assertEqual(json.loads((self.root / step / "metadata.json").read_text(encoding="utf-8"))["status"], "complete")
        np.testing.assert_array_equal(np.load(self.root / "balanced" / "test_timestamps.npy"), TIMES.as_unit("ns").asi8)
        self.assertIn("share before", output.getvalue())
        self.assertEqual(output.getvalue().splitlines()[2].split()[0], "total_cloud_cover")
        # The source run is left as it was.
        np.testing.assert_array_equal(np.load(self.root / "run" / "scores" / "feature_scores.npy"), self.errors)

    def test_rarity_gives_every_variable_the_same_distribution(self):
        with patch.object(script, "DIAGNOSTIC_STRIDE", 1):
            manifest = script.rebalance(self.root / "run", self.root / "rarity", scale="rarity", scale_stride=1)
            mean = script.rebalance(self.root / "run", self.root / "mean", scale="mean", scale_stride=1)
        score = np.load(self.root / "rarity" / "scores" / "anomaly_mean.npy")
        flat = self.errors.reshape(-1, 3)
        rescaled = script.transform(flat, "rarity", script.fit(flat, "rarity"))
        np.testing.assert_allclose(score.ravel(), rescaled.mean(axis=1), rtol=1e-5)
        # -log of the share of larger errors: mean 1 and the same quantiles for every variable.
        np.testing.assert_allclose(rescaled.mean(axis=0), 1, atol=0.02)
        for level in (50, 90, 99, 99.9):
            np.testing.assert_allclose(np.percentile(rescaled, level, axis=0), -np.log(1 - level / 100), rtol=0.03)
        # The error of a variable keeps its order.
        order = np.argsort(flat[:, 2])
        self.assertTrue((np.diff(rescaled[order, 2]) >= 0).all())
        # No variable takes the highest scores, unlike with the mean as scale.
        np.testing.assert_allclose(list(manifest["share_of_the_new_score"].values()), 1 / 3, atol=0.01)
        self.assertLess(max(manifest[TOP].values()), 0.5)
        self.assertLess(manifest[TOP]["total_precipitation"], mean[TOP]["total_precipitation"])
        self.assertIsNone(manifest["feature_means"])
        self.assertEqual(by_day(score).idxmax(), UNUSUAL)

    def test_the_most_unusual_variable_instead_of_the_mean(self):
        # One variable alone is far from usual in one cell at one time.
        errors = self.errors.copy()
        errors[1000, 5, 1] = 1.0
        self.write_run("alone", errors)
        mean = script.rebalance(self.root / "alone", self.root / "alone_mean", scale="rarity", scale_stride=1)
        highest = script.rebalance(self.root / "alone", self.root / "alone_max", scale="rarity", scale_stride=1,
                                   combine="max")
        self.assertEqual((mean["combine"], highest["combine"]), ("mean", "max"))
        flat = errors.reshape(-1, 3)
        rescaled = script.transform(flat, "rarity", script.fit(flat, "rarity"))
        scores = {name: np.load(self.root / f"alone_{name}" / "scores" / "anomaly_mean.npy") for name in ("mean", "max")}
        np.testing.assert_allclose(scores["mean"].ravel(), rescaled.mean(axis=1), rtol=1e-5)
        np.testing.assert_allclose(scores["max"].ravel(), rescaled.max(axis=1), rtol=1e-5)
        # No score is higher than the one of that cell, whatever its other two variables are.
        self.assertEqual(scores["max"][1000, 5], scores["max"].max())
        self.assertGreater(scores["max"][1000, 5], scores["mean"][1000, 5])
        with self.assertRaisesRegex(ValueError, "combine must be one of"):
            script.rebalance(self.root / "alone", self.root / "other", scale="rarity", combine="median")
        output = io.StringIO()
        with redirect_stdout(output):
            script.main(["--run-dir", str(self.root / "alone"), "--output-dir", str(self.root / "cli"),
                         "--scale", "rarity", "--combine", "max", "--scale-stride", "1"])
        self.assertIn("scale=rarity combine=max", output.getvalue())
        np.testing.assert_array_equal(np.load(self.root / "cli" / "scores" / "anomaly_mean.npy"), scores["max"])

    def test_the_mean_of_the_most_unusual_variables(self):
        flat = self.errors.reshape(-1, 3)
        rescaled = script.transform(flat, "rarity", script.fit(flat, "rarity"))
        scores = {}
        for k in (1, 2, 3):
            manifest = script.rebalance(self.root / "run", self.root / f"top{k}", scale="rarity", scale_stride=1,
                                        combine="top", top_k=k)
            self.assertEqual((manifest["combine"], manifest["top_k"]), ("top", k))
            scores[k] = np.load(self.root / f"top{k}" / "scores" / "anomaly_mean.npy").ravel()
            np.testing.assert_allclose(scores[k], np.sort(rescaled, axis=1)[:, -k:].mean(axis=1), rtol=1e-5)
        # One variable is the highest of them, all of them is their mean, and the score falls as more are taken.
        np.testing.assert_allclose(scores[1], rescaled.max(axis=1), rtol=1e-5)
        np.testing.assert_allclose(scores[3], rescaled.mean(axis=1), rtol=1e-5)
        self.assertTrue((scores[1] >= scores[2] - 1e-6).all() and (scores[2] >= scores[3] - 1e-6).all())
        self.assertIsNone(script.rebalance(self.root / "run", self.root / "plain_mean", scale="rarity")["top_k"])
        for wrong in (0, 4):
            with self.assertRaisesRegex(ValueError, "top_k must be between"):
                script.rebalance(self.root / "run", self.root / f"wrong{wrong}", combine="top", top_k=wrong)
            self.assertFalse((self.root / f"wrong{wrong}").exists())
        output = io.StringIO()
        with redirect_stdout(output):
            script.main(["--run-dir", str(self.root / "run"), "--output-dir", str(self.root / "cli_top"), "--scale", "rarity",
                         "--combine", "top", "--top-k", "2", "--scale-stride", "1"])
        np.testing.assert_array_equal(np.load(self.root / "cli_top" / "scores" / "anomaly_mean.npy").ravel(), scores[2])

    def test_rarity_of_equal_errors_and_of_errors_beyond_the_sample(self):
        """Half of the errors of a variable are zero: they are not rare. A new maximum has the highest rarity."""
        sample = np.concatenate((np.zeros(5000), np.linspace(1, 2, 5000)))[:, None].astype(np.float32)
        fitted = script.fit(sample, "rarity")
        rarity = script.transform(np.array([[0.0], [1.5], [2.0], [50.0]], dtype=np.float32), "rarity", fitted).ravel()
        self.assertAlmostEqual(rarity[0], -np.log(0.5), delta=0.01)
        self.assertAlmostEqual(rarity[1], -np.log(0.25), delta=0.01)
        self.assertAlmostEqual(rarity[3], -np.log(script.RAREST), delta=1e-3)
        self.assertEqual(rarity[2], rarity[3])
        self.assertTrue(np.isfinite(rarity).all())

    def test_the_weights_of_the_run_say_what_its_error_is_made_of(self):
        # The run weighed its variables so that the first, with errors a hundred times larger, counts as the second.
        weights = dict(zip(NAMES, (0.02, 2.0, 0.98)))
        self.write_run("weighted", self.errors, weights=weights)
        output = io.StringIO()
        with redirect_stdout(output), patch.object(script, "DIAGNOSTIC_STRIDE", 1):
            script.main(["--run-dir", str(self.root / "weighted"), "--output-dir", str(self.root / "as_run"),
                         "--scale", "weights", "--scale-stride", "1"])
        manifest = json.loads((self.root / "as_run" / "metadata.json").read_text(encoding="utf-8"))
        score = np.load(self.root / "as_run" / "scores" / "anomaly_mean.npy")
        np.testing.assert_allclose(score, (self.errors * np.array(list(weights.values()))).mean(axis=2), rtol=1e-4)
        self.assertEqual(manifest["feature_weights"], weights)
        self.assertGreater(manifest["share_of_the_plain_error"]["total_cloud_cover"], 0.95)
        weighted = (self.errors.reshape(-1, 3) * np.array(list(weights.values()))).sum(axis=0, dtype=np.float64)
        np.testing.assert_allclose(list(manifest["share_of_the_new_score"].values()), weighted / weighted.sum(), rtol=1e-4)
        self.assertLess(manifest["share_of_the_new_score"]["total_cloud_cover"], 0.6)
        # The rare large errors of the third variable make most of the highest weighted scores.
        self.assertGreater(manifest[TOP]["total_precipitation"], 0.5)
        self.assertIn("scale=weights", output.getvalue())
        kept = script.rebalance(self.root / "weighted", self.root / "two", scale="weights", features=NAMES[1:])
        self.assertEqual(kept["feature_weights"], {name: weights[name] for name in NAMES[1:]})
        with self.assertRaisesRegex(ValueError, "gave no weights"):
            script.rebalance(self.root / "run", self.root / "none", scale="weights")
        self.assertFalse((self.root / "none").exists())

    def test_features_keep_only_those_variables(self):
        manifest = script.rebalance(self.root / "run", self.root / "one", features=["2m_temperature"], scale_stride=1)
        score = np.load(self.root / "one" / "scores" / "anomaly_mean.npy")
        self.assertEqual(manifest["features"], ["2m_temperature"])
        np.testing.assert_allclose(score, self.errors[:, :, 1] / self.errors[:, :, 1].mean(dtype=np.float64), rtol=1e-4)

    def test_refuses_what_it_cannot_rebalance(self):
        self.write_run("running", self.errors, status="running")
        self.write_run("without", None)
        self.write_run("other", self.errors, features=NAMES[:2])
        for name, message, options in (("running", "must be complete", {}), ("without", "no errors by variable", {}),
                                       ("other", "do not match", {}), ("run", "Unknown variables", {"features": ["wind"]}),
                                       ("run", "scale must be", {"scale": "median"})):
            with self.assertRaisesRegex(ValueError, message):
                script.rebalance(self.root / name, self.root / ("out_" + name + message[:4]), **options)
            self.assertFalse((self.root / ("out_" + name + message[:4])).exists())
        script.rebalance(self.root / "run", self.root / "done")
        with self.assertRaisesRegex(ValueError, "new/empty"):
            script.rebalance(self.root / "run", self.root / "done")


if __name__ == "__main__":
    unittest.main()
