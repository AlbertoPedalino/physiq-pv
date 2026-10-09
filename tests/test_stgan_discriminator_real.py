"""discriminator_real: D's score of the observation, saved next to the score components.

discriminator_mean is D(observation) minus the mean D(reconstruction). Saving the first term
tells which of the two moves the component. The same file serves the ConvGRU and the GAT branch.
"""
from contextlib import redirect_stdout
import inspect
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

from unittest.mock import patch

from physiq_pv.anomaly_detection.stgan import STGAN, fit_and_score_stgan, load_stgan_checkpoint
from test_stgan_mmd import NAMES, TEST, TIMES, TRAIN, VALIDATION, build, build_pca, fields

PARAMETERS = inspect.signature(fit_and_score_stgan).parameters


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.values = fields()
        build_pca(self.root / "pca", self.values)
        build(self.root / "reference", self.values, self.root / "pca")

    def variants(self):
        yield "patch", dict(batch_size=4)
        if "cnn_training_mode" in PARAMETERS:
            yield "full_grid", dict(cnn_training_mode="full_grid", batch_size=1, score_batch_size=1)
        if "spatial_encoder" in PARAMETERS:
            common = dict(spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2, discriminator_chunk_size=4,
                          batch_size=1, score_batch_size=1)
            yield "gat", common
            if "gat_recurrence" in PARAMETERS:
                yield "gat_gru", dict(common, gat_recurrence="gated")

    def fit(self, name, **overrides):
        options = dict(train_timestamps=TIMES[TRAIN], test_timestamps=TIMES[TEST],
            validation=self.values[VALIDATION], validation_timestamps=TIMES[VALIDATION], **NAMES,
            epochs=1, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1, trend_steps=2,
            device="cpu", grid_crs="EPSG:4326", timestep_hours=3, angular_grid_spacing=.5,
            grid_audit_knn=False, score_mode="paper", cache_normalized=False, mc_samples=3,
            monitoring_timestamps=4, monitoring_feature_mmd_samples=16, score_storage="memmap",
            checkpoint_path=self.root / name / "model.pt", score_dir=self.root / name / "scores",
            pca_reference=self.root / "pca", mmd_reference=self.root / "reference")
        options.update(overrides)
        with redirect_stdout(io.StringIO()), fit_and_score_stgan(
                self.values[TRAIN], self.values[TEST], **options) as result:
            self.assertTrue(np.isfinite(np.array(result.test_scores)).all())
        return {path.stem: np.load(path) for path in (self.root / name / "scores").glob("*.npy")}

    def test_the_observation_score_is_saved_and_splits_the_component(self):
        for name, extra in self.variants():
            with self.subTest(variant=name):
                saved = self.fit(name, **extra)
                self.assertIn("discriminator_real", saved)
                real, difference = saved["discriminator_real"], saved["discriminator_mean"]
                self.assertEqual(real.shape, difference.shape)
                self.assertTrue(np.isfinite(real).all())
                # D ends in a sigmoid: its two scores are probabilities, and so is their mean.
                self.assertTrue(((real >= 0) & (real <= 1)).all())
                generated = real - difference
                self.assertTrue(((generated >= -1e-5) & (generated <= 1 + 1e-5)).all())
                self.assertGreater(real.std(), 0)

    def test_without_monte_carlo_the_single_draw_splits_the_same_way(self):
        for name, extra in self.variants():
            with self.subTest(variant=name):
                saved = self.fit(name, mc_dropout_enabled=False, **extra)
                network, _ = load_stgan_checkpoint(self.root / name / "model.pt")
                self.assertTrue(hasattr(network, "observation_scores"))
                # One draw: the component is D(observation) - D(reconstruction).
                real, difference = saved["discriminator_real"], saved["discriminator_mean"]
                generated = real - difference
                self.assertTrue(((generated >= -1e-5) & (generated <= 1 + 1e-5)).all())
                self.assertFalse(np.allclose(real, difference))

    def test_scoring_stops_on_a_non_finite_observation_score(self):
        def broken(self, recent, mask, observed):
            return torch.full((recent.shape[0],), float("nan"))

        with patch.object(STGAN, "observation_scores", broken), self.assertRaises(FloatingPointError):
            self.fit("broken", batch_size=4)


class ModelTests(unittest.TestCase):
    def test_patch_model_returns_the_real_score_of_components(self):
        torch.manual_seed(20)
        network = STGAN(n_features=15, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
                        patch_size=3, time_feature_size=31).eval()
        recent, trend = torch.randn(5, 2, 15, 3, 3), torch.randn(5, 4, 15)
        mask, calendar, observed = torch.ones(5, 1, 3, 3), torch.randn(5, 31), torch.randn(5, 15, 3, 3)
        with torch.no_grad():
            expected = network.components(recent, trend, mask, calendar, observed)[1]
            scores = network.observation_scores(recent, mask, observed)
            draws = network.score_draws(recent, trend, mask, calendar, observed, 1)
            generated = network.components(recent, trend, mask, calendar, observed)[2]
        self.assertEqual(scores.shape, (5,))
        torch.testing.assert_close(scores, expected.squeeze(1))
        torch.testing.assert_close(draws[0, :, 1], scores - generated.squeeze(1), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
