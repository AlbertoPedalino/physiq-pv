"""Effective G/D learning rates, CLI wiring and checkpoint resume contracts."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch

from physiq_pv.anomaly_detection.stgan import STGANCNNConfig, fit_and_score_stgan, masked_cell_mean
from physiq_pv.anomaly_detection.stgan.training import gan_train_step
from test_stgan_mc_dropout import small_model
from test_stgan_performance import fixture


class GANBalanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_rates_are_positive_finite_including_derived_rate(self):
        self.assertEqual(STGANCNNConfig().discriminator_lr_ratio, 1.)
        self.assertEqual(STGANCNNConfig().effective_discriminator_learning_rate, .001)
        for ratio, expected in ((1., .0002), (2., .0004), (.5, .0001)):
            config = STGANCNNConfig(learning_rate=.0002, discriminator_lr_ratio=ratio)
            self.assertEqual((config.learning_rate, config.effective_discriminator_learning_rate), (.0002, expected))
        for ratio in (0., -1., float("nan"), float("inf")):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                STGANCNNConfig(discriminator_lr_ratio=ratio)
        for lr, ratio in ((1e308, 2.), (1e-300, 1e-300)):
            with self.subTest(lr=lr, ratio=ratio), self.assertRaisesRegex(ValueError, "Effective discriminator"):
                STGANCNNConfig(learning_rate=lr, discriminator_lr_ratio=ratio)

    def test_cli_passes_gan_controls_to_both_runners(self):
        from scripts import run_era5_stgan, run_pvgis_stgan
        required = ["train", "--prepared-dir", "unused", "--output-dir", "unused"]
        defaults = run_era5_stgan.parser().parse_args(required)
        self.assertEqual((defaults.lr, defaults.discriminator_lr_ratio,
                          defaults.generator_reconstruction_weight), (.001, 1., 500.))
        self.assertEqual(run_era5_stgan.parser().parse_args(required + ["--lr", "0.0002"]).lr, .0002)
        controls = ["--learning-rate", "0.0002", "--discriminator-lr-ratio", "2",
                    "--generator-reconstruction-weight", "100"]
        with patch.object(run_era5_stgan, "run_training", return_value={}) as train, redirect_stdout(io.StringIO()):
            run_era5_stgan.main([*required, *controls])
        era5 = train.call_args.kwargs["config"]
        with patch.object(run_pvgis_stgan, "run_stgan") as train:
            run_pvgis_stgan.main(["--manifest", "unused", "--out-dir", "unused", *controls])
        for config in (era5, train.call_args.kwargs["config"]):
            self.assertEqual((config.learning_rate, config.effective_discriminator_learning_rate,
                              config.generator_reconstruction_weight), (.0002, .0004, 100.))

    def test_reconstruction_weight_scales_only_the_generator_reconstruction_term(self):
        torch.manual_seed(23)
        reference = small_model(dropout_enabled=False).train()
        batch = fixture().fetch_batch(list(range(7)))[:5]
        recent, trend, mask, calendar, observed = batch
        with torch.no_grad():
            errors = torch.where(mask.bool(), reference.generator(recent, trend, mask, calendar) - observed, 0.).square()
            reconstruction = masked_cell_mean(errors, mask).mean().item()
        self.assertGreater(reconstruction, 0.)
        outcome = {}
        for weight in (50., 500.):
            model = copy.deepcopy(reference)
            generator_loss, discriminator_loss = gan_train_step(
                model, batch, torch.optim.Adam(model.generator.parameters()),
                torch.optim.Adam(model.discriminator.parameters()), reconstruction_weight=weight)
            outcome[weight] = (generator_loss.item(), discriminator_loss.item(), model)
        (g_low, d_low, low), (g_high, d_high, high) = outcome[50.], outcome[500.]
        # D never sees the weight: same loss and the same updated parameters.
        self.assertEqual(d_low, d_high)
        for name, value in low.discriminator.state_dict().items():
            self.assertTrue(torch.equal(value, high.discriminator.state_dict()[name]), name)
        # G loss = weight * reconstruction + adversarial term, the latter unweighted.
        np.testing.assert_allclose(g_high - g_low, 450. * reconstruction, rtol=1e-4)
        np.testing.assert_allclose(g_low - 50. * reconstruction, g_high - 500. * reconstruction, atol=1e-3)
        self.assertGreater(g_low - 50. * reconstruction, 0.)
        self.assertFalse(all(torch.equal(value, high.generator.state_dict()[name])
                             for name, value in low.generator.state_dict().items()))

    def fit(self, name, **overrides):
        values = np.random.default_rng(3).normal(size=(12, 4, 2)).astype(np.float32)
        times = pd.date_range("2004-12-30", periods=12, freq="3h")
        options = dict(train_timestamps=times[:8], test_timestamps=times[8:],
            location_names=("0", "1", "2", "3"), feature_names=("a", "b"),
            latitudes=np.array([45., 45., 44.5, 44.5]), longitudes=np.array([7., 7.5, 7., 7.5]),
            epochs=1, batch_size=1, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, train_samples_per_epoch=2, device="cpu", grid_crs="EPSG:4326",
            timestep_hours=3, angular_grid_spacing=.5, grid_audit_knn=False, score_mode="paper",
            cache_normalized=False, dropout_enabled=False, mc_dropout_enabled=False, lr=.0002,
            checkpoint_path=self.root/name/"model.pt")
        options.update(overrides)
        with redirect_stdout(io.StringIO()), fit_and_score_stgan(values[:8], values[8:], **options) as result:
            return result.metadata

    def test_training_uses_rates_and_exports_them(self):
        for ratio in (1., 2., .5):
            with self.subTest(ratio=ratio):
                rates, weights = [], []
                def observe(model, batch, g_optimizer, d_optimizer, **kwargs):
                    rates.append((g_optimizer.param_groups[0]["lr"], d_optimizer.param_groups[0]["lr"]))
                    weights.append(kwargs["reconstruction_weight"])
                    return gan_train_step(model, batch, g_optimizer, d_optimizer, **kwargs)
                name = f"ratio_{ratio}"
                with patch("physiq_pv.anomaly_detection.stgan.pipeline.gan_train_step", side_effect=observe):
                    metadata = self.fit(name, discriminator_lr_ratio=ratio, generator_reconstruction_weight=100.)
                self.assertTrue(rates)
                self.assertTrue(all(pair == (.0002, .0002 * ratio) for pair in rates))
                self.assertEqual(set(weights), {100.})
                self.assertEqual((metadata["generator_learning_rate"], metadata["discriminator_learning_rate"],
                                  metadata["discriminator_lr_ratio"], metadata["generator_reconstruction_weight"]),
                                 (.0002, .0002 * ratio, ratio, 100.))
                epoch = torch.load(self.root/name/"model_epoch_1.pt", weights_only=False)
                final = torch.load(self.root/name/"model.pt", weights_only=False)
                self.assertEqual(epoch["learning_rates"]["discriminator_learning_rate"], .0002 * ratio)
                self.assertEqual(epoch["generator_optimizer_state_dict"]["param_groups"][0]["lr"], .0002)
                self.assertEqual(epoch["discriminator_optimizer_state_dict"]["param_groups"][0]["lr"], .0002 * ratio)
                self.assertTrue(epoch["discriminator_optimizer_state_dict"]["state"])
                self.assertEqual(final["training"]["discriminator_lr_ratio"], ratio)
                self.assertEqual(final["training"]["discriminator_learning_rate"], .0002 * ratio)

    def test_resume_keeps_distinct_rates_and_rejects_silent_optimizer_override(self):
        self.fit("resume", discriminator_lr_ratio=2.)
        checkpoint = self.root/"resume/model_epoch_1.pt"
        metadata = self.fit("resume", epochs=2, discriminator_lr_ratio=2., resume_from=checkpoint)
        self.assertTrue(metadata["resume"]["optimizer_state_restored"])
        resumed = torch.load(self.root/"resume/model_epoch_2.pt", weights_only=False)
        self.assertEqual(resumed["generator_optimizer_state_dict"]["param_groups"][0]["lr"], .0002)
        self.assertEqual(resumed["discriminator_optimizer_state_dict"]["param_groups"][0]["lr"], .0004)
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            self.fit("resume", epochs=2, discriminator_lr_ratio=1., resume_from=checkpoint)
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            self.fit("resume", epochs=2, lr=.0004, discriminator_lr_ratio=1., resume_from=checkpoint)
        with self.assertRaisesRegex(ValueError, "generator_reconstruction_weight"):
            self.fit("resume", epochs=2, discriminator_lr_ratio=2., generator_reconstruction_weight=100.,
                     resume_from=checkpoint)
        # Legacy epoch files have optimizer rates, but no explicit rate metadata.
        payload = torch.load(checkpoint, weights_only=False)
        payload.pop("learning_rates")
        torch.save(payload, checkpoint)
        with self.assertRaisesRegex(ValueError, "discriminator_learning_rate"):
            self.fit("resume", epochs=2, discriminator_lr_ratio=1., resume_from=checkpoint)
        with patch("physiq_pv.anomaly_detection.stgan.pipeline.gan_train_step") as step:
            metadata = self.fit("resume", discriminator_lr_ratio=2., resume_from=checkpoint)
        step.assert_not_called()  # A completed training run can restart scoring only.
        self.assertEqual(metadata["discriminator_learning_rate"], .0004)


if __name__ == "__main__":
    unittest.main()
