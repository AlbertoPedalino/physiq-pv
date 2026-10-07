"""Min-max factors of the STGAN score: where they come from, what they allow to recover, what is logged.

The same file serves the ConvGRU and the GAT branch.
"""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch

from physiq_pv.anomaly_detection.stgan import fit_and_score_stgan
from physiq_pv.anomaly_detection.stgan.scoring import component_statistics
from physiq_pv.experiments.stgan_wandb import RunLogger

FACTORS = {"reconstruction_min": "r_min", "reconstruction_max": "r_max",
           "discriminator_min": "d_min", "discriminator_max": "d_max"}


class ScoreStatisticsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_statistics_are_exact_for_any_size_and_shape(self):
        rng = np.random.default_rng(4)
        heavy = rng.lognormal(sigma=3, size=(7, 911)).astype(np.float32)  # Most values far below the maximum.
        ties = np.repeat(np.float32([.25, .5, .5, .5, 3.]), 400).reshape(4, 500)
        memmap = np.lib.format.open_memmap(self.root / "values.npy", mode="w+", dtype=np.float32, shape=(40, 300))
        memmap[:] = rng.normal(size=memmap.shape)
        cases = {"heavy": heavy, "ties": ties, "constant": np.full((3, 5), 2.5, np.float32),
                 "single": np.float32([[1.5]]), "pair": np.float32([[1., 4.]]), "memmap": memmap,
                 "signed": rng.normal(size=(5, 400)).astype(np.float64)}
        for name, values in cases.items():
            # Small limits force several narrowing passes; the defaults take the direct path.
            for options in ({}, dict(chunk_size=97, bins=8, exact_size=16)):
                with self.subTest(case=name, options=options):
                    result = component_statistics(values, quantiles=(.5, .95, 0., 1., .999), **options)
                    exact = np.asarray(values, dtype=np.float64)
                    self.assertEqual((result["min"], result["max"], result["count"]),
                                     (exact.min(), exact.max(), exact.size))
                    np.testing.assert_allclose((result["mean"], result["std"]), (exact.mean(), exact.std()),
                                               rtol=1e-12, atol=1e-12)
                    for key, quantile in (("median", .5), ("p95", .95), ("p0", 0.), ("p100", 1.), ("p99.9", .999)):
                        self.assertEqual(result[key], float(np.quantile(exact, quantile)), (key, name))
        memmap._mmap.close()
        with self.assertRaises(ValueError):
            component_statistics(np.float32([[1., np.nan]]))
        with self.assertRaises(ValueError):
            component_statistics(np.empty((0, 3), np.float32))

    def fit(self, **overrides):
        values = np.random.default_rng(3).normal(size=(16, 9, 2)).astype(np.float32)
        times = pd.date_range("2004-12-30", periods=16, freq="3h")
        rows, cols = np.indices((3, 3)).reshape(2, -1)
        options = dict(train_timestamps=times[:8], test_timestamps=times[8:],
            location_names=tuple(map(str, range(9))), feature_names=("a", "b"),
            latitudes=45 - rows * .5, longitudes=7 + cols * .5,
            epochs=1, batch_size=4, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, train_samples_per_epoch=8, device="cpu", grid_crs="EPSG:4326",
            timestep_hours=3, angular_grid_spacing=.5, grid_audit_knn=False, score_mode="paper",
            cache_normalized=False, mc_samples=6, checkpoint_path=self.root / "run/model.pt")
        options.update(overrides)
        return fit_and_score_stgan(values[:8], values[8:], **options)

    def test_factors_come_from_this_runs_test_components_and_rebuild_the_score(self):
        with redirect_stdout(io.StringIO()), self.fit() as result:
            g = np.array(result.test_generator_scores, dtype=np.float64)       # reconstruction_raw
            d = np.array(result.test_discriminator_scores, dtype=np.float64)   # discriminator_raw
            score, std = np.array(result.test_scores), np.array(result.anomaly_std)
            metadata = result.metadata
            files = {path.name for path in (self.root / "run/scores").glob("*.npy")}
        normalization = metadata["score_normalization"]
        # Fitted on the MC-mean maps of the whole test period of this run: no calibration, no validation.
        self.assertEqual((normalization["r_min"], normalization["r_max"]), (g.min(), g.max()))
        self.assertEqual((normalization["d_min"], normalization["d_max"]), (d.min(), d.max()))
        self.assertEqual((normalization["fit_period"], normalization["transductive"]),
                         ("complete_test_mc_mean_components", True))
        self.assertEqual(metadata["splits"].keys(), {"train", "test"})
        g_scale = normalization["r_max"] - normalization["r_min"]
        d_scale = normalization["d_max"] - normalization["d_min"]
        # The published score is exactly the sum of the two min-max components...
        g_normalized = (g - normalization["r_min"]) / g_scale
        d_normalized = (d - normalization["d_min"]) / d_scale
        np.testing.assert_allclose(score, g_normalized + d_normalized, rtol=1e-6, atol=1e-6)
        self.assertEqual((g_normalized.min(), g_normalized.max(), d_normalized.min(), d_normalized.max()),
                         (0., 1., 0., 1.))
        # ...each component returns to its own scale with its two factors...
        np.testing.assert_allclose(g_normalized * g_scale + normalization["r_min"], g, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(d_normalized * d_scale + normalization["d_min"], d, rtol=1e-12, atol=1e-12)
        # ...and its MC spread follows from the saved component moments.
        if files:
            moments = {name: np.load(self.root / f"run/scores/{name}.npy").astype(np.float64)
                       for name in ("generator_std", "discriminator_std", "component_covariance")}
            variance = (moments["generator_std"] ** 2 / g_scale ** 2 + moments["discriminator_std"] ** 2 / d_scale ** 2
                        + 2 * moments["component_covariance"] / (g_scale * d_scale))
            np.testing.assert_allclose(std, np.sqrt(np.maximum(variance, 0)), rtol=1e-4, atol=1e-6)
        # The sum alone does not: different raw pairs give the same score.
        shift = .1
        other_g = (g_normalized + shift) * g_scale + normalization["r_min"]
        other_d = (d_normalized - shift) * d_scale + normalization["d_min"]
        np.testing.assert_allclose((other_g - normalization["r_min"]) / g_scale
                                   + (other_d - normalization["d_min"]) / d_scale, score, rtol=1e-6, atol=1e-6)
        self.assertGreater(np.abs(other_g - g).min(), 0)
        # The raw statistics describe the saved maps.
        statistics = metadata["score_statistics"]
        self.assertEqual(set(statistics), {"reconstruction_raw", "discriminator_raw", "anomaly"})
        for name, values in (("reconstruction_raw", g), ("discriminator_raw", d),
                             ("anomaly", score.astype(np.float64))):
            self.assertEqual((statistics[name]["min"], statistics[name]["max"]), (values.min(), values.max()))
            self.assertEqual((statistics[name]["median"], statistics[name]["p95"]),
                             (float(np.quantile(values, .5)), float(np.quantile(values, .95))))
            np.testing.assert_allclose((statistics[name]["mean"], statistics[name]["std"]),
                                       (values.mean(), values.std()), rtol=1e-10)
        payload = torch.load(self.root / "run/model.pt", weights_only=False)
        self.assertEqual(payload["score_statistics"], statistics)
        self.assertEqual(payload["score_normalization"]["r_max"], normalization["r_max"])

    def test_ranges_change_with_the_scored_period_and_reweight_the_components(self):
        scores = {}
        for name, steps in (("short", 12), ("long", 16)):
            values = np.random.default_rng(3).normal(size=(16, 9, 2)).astype(np.float32)
            values[14:, 4] += 9.  # A late extreme, only inside the longer test period.
            times = pd.date_range("2004-12-30", periods=16, freq="3h")
            rows, cols = np.indices((3, 3)).reshape(2, -1)
            with redirect_stdout(io.StringIO()), fit_and_score_stgan(values[:8], values[8:steps],
                    train_timestamps=times[:8], test_timestamps=times[8:steps],
                    location_names=tuple(map(str, range(9))), feature_names=("a", "b"),
                    latitudes=45 - rows * .5, longitudes=7 + cols * .5, epochs=1, batch_size=4, hidden_size=4,
                    n_layers=1, cnn_channels=2, cnn_layers=1, trend_steps=2, train_samples_per_epoch=8,
                    device="cpu", grid_crs="EPSG:4326", timestep_hours=3, angular_grid_spacing=.5,
                    grid_audit_knn=False, score_mode="paper", cache_normalized=False,
                    dropout_enabled=False, mc_dropout_enabled=False,
                    checkpoint_path=self.root / name / "model.pt") as result:
                scores[name] = (np.array(result.test_scores), np.array(result.test_generator_scores),
                                result.metadata["score_normalization"])
        (short, short_g, short_n), (long, long_g, long_n) = scores["short"], scores["long"]
        # Same model and same first four timestamps: identical raw components...
        np.testing.assert_array_equal(short_g, long_g[:4])
        # ...but the extreme widens the reconstruction range, so the same cells get another score.
        self.assertGreater(long_n["r_max"], 2 * short_n["r_max"])
        self.assertFalse(np.allclose(short, long[:4], rtol=1e-3))

    def test_wandb_logs_the_factors_and_raw_statistics(self):
        class Run:
            def __init__(self):
                self.logs, self.summary = [], {}

            def define_metric(self, *args, **kwargs):
                pass

            def log(self, values):
                self.logs.append(dict(values))

        with redirect_stdout(io.StringIO()), self.fit() as result:
            metadata = dict(result.metadata)
        metadata.setdefault("precision", "fp32")
        run = Run()
        RunLogger(run).result(metadata)
        logged = {key: value for values in run.logs for key, value in values.items() if key.startswith("score/")}
        for name, key in FACTORS.items():
            self.assertEqual(logged[f"score/{name}"], metadata["score_normalization"][key])
        for component in ("reconstruction_raw", "discriminator_raw", "anomaly"):
            for statistic in ("median", "p95", "mean", "std"):
                self.assertEqual(logged[f"score/{component}_{statistic}"],
                                 metadata["score_statistics"][component][statistic])
        self.assertEqual(len(logged), 4 + 3 * 4)
        # A run without these blocks (older metadata) logs no score keys and does not fail.
        run = Run()
        RunLogger(run).result({"precision": "fp32", "performance": {"training_seconds": 2.}})
        self.assertEqual(run.logs, [{"performance/training_seconds": 2.}])


if __name__ == "__main__":
    unittest.main()
