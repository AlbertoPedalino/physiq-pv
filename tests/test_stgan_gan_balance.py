"""Effective G/D learning rates, CLI wiring and checkpoint resume contracts."""
from contextlib import redirect_stdout
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

from physiq_pv.anomaly_detection.stgan import STGANCNNConfig, fit_and_score_stgan
from physiq_pv.anomaly_detection.stgan.training import gan_train_step


class GANBalanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_rates_are_positive_finite_including_derived_rate(self):
        self.assertEqual(STGANCNNConfig().effective_discriminator_learning_rate, .001)
        for ratio in (0., -1., float("nan"), float("inf")):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                STGANCNNConfig(discriminator_lr_ratio=ratio)
        for lr, ratio in ((1e308, 2.), (1e-300, 1e-300)):
            with self.subTest(lr=lr, ratio=ratio), self.assertRaisesRegex(ValueError, "Effective discriminator"):
                STGANCNNConfig(learning_rate=lr, discriminator_lr_ratio=ratio)

    def test_cli_passes_gan_controls_to_both_runners(self):
        from scripts import run_era5_stgan, run_pvgis_stgan
        controls = ["--learning-rate", "0.0002", "--discriminator-lr-ratio", "2",
                    "--generator-reconstruction-weight", "100"]
        with patch.object(run_era5_stgan, "run_training", return_value={}) as train, redirect_stdout(io.StringIO()):
            run_era5_stgan.main(["train", "--prepared-dir", "unused", "--output-dir", "unused", *controls])
        era5 = train.call_args.kwargs["config"]
        with patch.object(run_pvgis_stgan, "run_stgan") as train:
            run_pvgis_stgan.main(["--manifest", "unused", "--out-dir", "unused", *controls])
        for config in (era5, train.call_args.kwargs["config"]):
            self.assertEqual((config.learning_rate, config.effective_discriminator_learning_rate,
                              config.generator_reconstruction_weight), (.0002, .0004, 100.))

    def fit(self, name, **overrides):
        values = np.random.default_rng(3).normal(size=(12, 4, 2)).astype(np.float32)
        times = pd.date_range("2004-12-30", periods=12, freq="3h")
        options = dict(train_timestamps=times[:8], test_timestamps=times[8:],
            location_names=("0", "1", "2", "3"), feature_names=("a", "b"),
            latitudes=np.array([45., 45., 44.5, 44.5]), longitudes=np.array([7., 7.5, 7., 7.5]),
            epochs=1, batch_size=1, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, train_samples_per_epoch=2, device="cpu", grid_crs="EPSG:4326",
            timestep_hours=3, angular_grid_spacing=.5, grid_audit_knn=False, score_mode="paper",
            cache_normalized=False, dropout_enabled=False, mc_dropout_enabled=False,
            gat_hidden_dim=2, gat_heads=2, lr=.0002,
            checkpoint_path=self.root/name/"model.pt")
        options.update(overrides)
        with redirect_stdout(io.StringIO()), fit_and_score_stgan(values[:8], values[8:], **options) as result:
            return result.metadata

    def test_training_uses_rates_and_exports_them_for_gat_and_convgru(self):
        for encoder, ratio in (("convgru", 1.), ("convgru", 2.), ("gat", .5)):
            with self.subTest(encoder=encoder, ratio=ratio):
                rates = []
                def observe(model, batch, g_optimizer, d_optimizer, **kwargs):
                    rates.append((g_optimizer.param_groups[0]["lr"], d_optimizer.param_groups[0]["lr"]))
                    return gan_train_step(model, batch, g_optimizer, d_optimizer, **kwargs)
                name = f"{encoder}_{ratio}"
                with patch("physiq_pv.anomaly_detection.stgan.pipeline.gan_train_step", side_effect=observe):
                    metadata = self.fit(name, spatial_encoder=encoder, discriminator_lr_ratio=ratio)
                self.assertTrue(rates)
                self.assertTrue(all(pair == (.0002, .0002 * ratio) for pair in rates))
                self.assertEqual(metadata["discriminator_learning_rate"], .0002 * ratio)
                epoch = torch.load(self.root/name/"model_epoch_1.pt", weights_only=False)
                final = torch.load(self.root/name/"model.pt", weights_only=False)
                self.assertTrue(epoch["discriminator_optimizer_state_dict"]["state"])
                self.assertEqual(final["training"]["discriminator_lr_ratio"], ratio)

    def test_resume_keeps_distinct_rates_and_rejects_silent_optimizer_override(self):
        self.fit("resume", discriminator_lr_ratio=2.)
        checkpoint = self.root/"resume/model_epoch_1.pt"
        metadata = self.fit("resume", epochs=2, discriminator_lr_ratio=2., resume_from=checkpoint)
        self.assertTrue(metadata["resume"]["optimizer_state_restored"])
        resumed = torch.load(self.root/"resume/model_epoch_2.pt", weights_only=False)
        self.assertEqual(resumed["discriminator_optimizer_state_dict"]["param_groups"][0]["lr"], .0004)
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            self.fit("resume", epochs=2, discriminator_lr_ratio=1., resume_from=checkpoint)
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
