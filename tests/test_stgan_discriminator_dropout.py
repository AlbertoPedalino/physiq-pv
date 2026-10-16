"""Dropout in D after its encoders, and score ranges fitted on the validation.

D's dropout is off by default (the paper's D). When on, it acts in training and in the Monte Carlo
draws of the score, and one draw judges the observation and its reconstruction with the same masks.
With score_mode=validation the two score ranges come from the validation draws, not from the test.
"""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch
from torch import nn

from physiq_pv.anomaly_detection.stgan import (
    STGAN, FullGridSTGAN, STGANCNNConfig, fit_and_score_stgan, load_stgan_checkpoint, masked_cell_mean)
from physiq_pv.anomaly_detection.stgan.loading import make_loader
from physiq_pv.anomaly_detection.stgan.objective import validation_objective
from physiq_pv.anomaly_detection.stgan.scoring import (
    ScoreStore, normalize_paper_mc_scores, scoring_mode, summarize_raw_mc_components)
from scripts import run_era5_stgan
from test_stgan_full_grid import FEATURES, case
from test_stgan_mc_dropout import small_model
from test_stgan_performance import fixture


def batch_of(dataset, count=8):
    return dataset.fetch_batch(list(range(count)))[:5]


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.dataset = fixture()

    def test_off_by_default_and_the_same_state_dict(self):
        torch.manual_seed(1)
        plain = small_model()
        torch.manual_seed(1)
        dropped = small_model(discriminator_dropout_p=.3)
        self.assertFalse(plain.discriminator.has_dropout)
        self.assertFalse(any(isinstance(layer, nn.Dropout) for layer in plain.discriminator.modules()))
        self.assertTrue(dropped.discriminator.has_dropout)
        # No new parameter: a checkpoint of a D without dropout loads into a D with it.
        self.assertEqual(list(plain.state_dict()), list(dropped.state_dict()))
        dropped.load_state_dict(plain.state_dict())
        recent, trend, mask, calendar, observed = batch_of(self.dataset)
        plain.eval(), dropped.eval()
        with torch.no_grad():
            for first, second in zip(plain.discriminator.score_pair(recent, observed, observed * .5, mask),
                                     dropped.discriminator.score_pair(recent, observed, observed * .5, mask)):
                self.assertTrue(torch.equal(first, second))  # In eval mode D's dropout does nothing.
        with self.assertRaises(ValueError):
            small_model(discriminator_dropout_p=1.0)

    def test_one_draw_judges_the_observation_and_the_reconstruction_with_the_same_masks(self):
        model = small_model(discriminator_dropout_p=.3)
        model.train()
        recent, trend, mask, calendar, observed = batch_of(self.dataset)
        d = model.discriminator
        with torch.no_grad():
            real, fake = d.score_pair(recent, observed, observed, mask)
            self.assertTrue(torch.equal(real, fake))  # Same input, same masks: the same output.
            history = d.encode_history(recent, mask)
            first, second = (d.score_current(history, observed, mask) for _ in range(2))
            self.assertFalse(torch.equal(first, second))  # A new draw, new masks.
            # The penultimate route gives the score too, in eval mode.
            model.eval()
            self.assertTrue(torch.allclose(d.score_from_penultimate(d.penultimate(history, observed, mask)),
                                           d.score_current(history, observed, mask), atol=1e-7))

    def test_scoring_mode_turns_on_the_dropout_of_both_networks(self):
        model = small_model(discriminator_dropout_p=.3)
        model.train()
        layers = [layer for layer in model.discriminator.modules() if isinstance(layer, nn.Dropout)]
        self.assertEqual(len(layers), 3)
        with scoring_mode(model, True):
            self.assertTrue(all(layer.training for layer in layers))
            self.assertFalse(model.discriminator.sequence_encoder.training)
        with scoring_mode(model, False):
            self.assertFalse(any(layer.training for layer in layers))
        self.assertTrue(all(layer.training for layer in layers))  # Modes come back.

    def test_draws_vary_in_d_and_equal_repeated_full_draws(self):
        recent, trend, mask, calendar, observed = batch = batch_of(self.dataset, 11)
        centre = self.dataset.grid.patch_size // 2
        for input_dropout in (True, False):
            with self.subTest(discriminator_input_dropout=input_dropout):
                torch.manual_seed(3)
                model = small_model(discriminator_dropout_p=.3, discriminator_input_dropout=input_dropout)
                with scoring_mode(model, True), torch.no_grad():
                    for share_history in (True, False):
                        draws = model.score_draws(*batch, 5, share_history=share_history)
                        self.assertEqual(draws.shape, (5, 11, 5))
                        # D's component changes from draw to draw, even when D sees the reconstruction
                        # without G's dropout: D's own masks change.
                        self.assertGreater(float(draws[:, :, 1].std(dim=0).max()), 0)
                    if input_dropout:
                        # The per-draw loop of full components calls, with the same random stream.
                        torch.manual_seed(11)
                        repeated = []
                        for _ in range(5):
                            _, real, fake, errors = model.components(*batch)
                            repeated.append(torch.cat((masked_cell_mean(errors, mask)[:, None], real - fake,
                                                       errors[:, :, centre, centre]), dim=1))
                        torch.manual_seed(11)
                        self.assertTrue(torch.equal(model.score_draws(*batch, 5), torch.stack(repeated)))

    def test_d_component_is_constant_over_draws_without_d_dropout_and_clean_input(self):
        model = small_model(discriminator_input_dropout=False)
        batch = batch_of(self.dataset)
        with scoring_mode(model, True), torch.no_grad():
            draws = model.score_draws(*batch, 4)
        self.assertEqual(float(draws[:, :, 1].std(dim=0).max()), 0.0)
        self.assertGreater(float(draws[:, :, 0].std(dim=0).max()), 0)

    def test_observation_score_is_the_mean_of_its_draws(self):
        model = small_model(discriminator_dropout_p=.3)
        recent, trend, mask, calendar, observed = batch_of(self.dataset)
        with scoring_mode(model, True), torch.no_grad():
            torch.manual_seed(5)
            found = model.observation_scores(recent, mask, observed, 4)
            torch.manual_seed(5)
            history = model.discriminator.encode_history(recent, mask)
            by_hand = torch.stack([model.discriminator.score_current(history, observed, mask).squeeze(1)
                                   for _ in range(4)]).mean(dim=0)
        self.assertTrue(torch.allclose(found, by_hand, atol=1e-7))
        with scoring_mode(model, False), torch.no_grad():  # Dropout off: one value, whatever `samples`.
            history = model.discriminator.encode_history(recent, mask)
            self.assertTrue(torch.equal(model.observation_scores(recent, mask, observed, 4),
                                        model.discriminator.score_current(history, observed, mask).squeeze(1)))

    def test_full_grid_discriminator_is_the_patch_one_on_its_patches_with_dropout(self):
        c = case(1)
        config = dict(n_features=FEATURES, hidden_size=8, n_layers=1, cnn_channels=4, cnn_layers=2,
                      time_feature_size=c.patches.time_feature_size, dropout_enabled=False, discriminator_dropout_p=.3)
        patch_model = STGAN(**config)
        model = FullGridSTGAN(grid=c.grid, **config)
        model.load_state_dict(patch_model.state_dict())
        patch_model.train(), model.train()
        recent, trend, mask, calendar, observed = c.field[:5]
        other = observed * .5
        with torch.no_grad():
            torch.manual_seed(7)
            found = model.discriminator.score_pair(recent, observed, other, mask)
            torch.manual_seed(7)
            expected = patch_model.discriminator.score_pair(c.patch[0], c.patch[4], model.layout.patches(other), c.patch[2])
            for first, second in zip(found, expected):
                self.assertTrue(torch.allclose(first, second, rtol=0, atol=1e-6))
            draws = model.score_draws(recent, trend, mask, calendar, observed, 3)
        self.assertGreater(float(draws[:, :, 1].std(dim=0).max()), 0)

    def test_a_training_step_with_d_dropout_updates_both_networks(self):
        from physiq_pv.anomaly_detection.stgan.training import gan_train_step
        c = case(16, "cyclic")
        config = dict(n_features=FEATURES, hidden_size=8, n_layers=1, cnn_channels=4, cnn_layers=2,
                      time_feature_size=c.patches.time_feature_size, discriminator_dropout_p=.3,
                      discriminator_input_dropout=False)
        model = FullGridSTGAN(grid=c.grid, **config)
        model.train()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        optimizers = [torch.optim.Adam(part.parameters(), lr=1e-3) for part in (model.generator, model.discriminator)]
        losses = gan_train_step(model, c.field[:5], *optimizers, discriminator_steps=2)
        self.assertTrue(all(torch.isfinite(loss) for loss in losses))
        for part in ("generator", "discriminator"):
            self.assertTrue([name for name, value in model.state_dict().items()
                             if name.startswith(part) and not torch.equal(value, before[name])], part)


class ConfigTests(unittest.TestCase):
    def test_values_and_dependencies(self):
        self.assertEqual((STGANCNNConfig().discriminator_dropout_p, STGANCNNConfig().score_ranges), (0.0, "test"))
        for invalid in (dict(discriminator_dropout_p=1.0), dict(discriminator_dropout_p=-.1),
                        dict(discriminator_dropout_p=True), dict(discriminator_dropout_p=float("nan")),
                        dict(score_ranges="calibration"), dict(score_ranges="validation")):
            with self.subTest(**invalid), self.assertRaises(ValueError):
                STGANCNNConfig(**invalid)
        STGANCNNConfig(score_ranges="validation", validation_holdout=True, discriminator_dropout_p=.2)

    def test_command_line_reaches_the_config(self):
        base = ["train", "--prepared-dir", "unused", "--output-dir", "unused", "--validation-holdout"]
        for arguments, expected in (([], (0.0, "test")),
                                    (["--discriminator-dropout-p", "0.2", "--score-ranges", "validation"], (0.2, "validation"))):
            with patch.object(run_era5_stgan, "run_training", return_value={}) as train, redirect_stdout(io.StringIO()):
                run_era5_stgan.main(base + arguments)
            config = train.call_args.kwargs["config"]
            self.assertEqual((config.discriminator_dropout_p, config.score_ranges), expected)


class RangeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_validation_objective_gives_the_ranges_of_its_own_draws_without_flagged_points(self):
        dataset = fixture()
        model = small_model(discriminator_dropout_p=.3)
        shape = (len(dataset.targets), dataset.n_locations)
        excluded = np.random.default_rng(2).random(shape) < .3
        torch.manual_seed(9)
        with redirect_stdout(io.StringIO()):
            result = validation_objective(model, dataset, batch_size=5, device="cpu", mc_dropout_enabled=True,
                                          mc_samples=3, excluded=excluded, fit_score_ranges=True)
        torch.manual_seed(9)
        draws = np.empty((3, *shape, 2))
        with scoring_mode(model, True), torch.no_grad():
            for batch in make_loader(dataset, batch_size=5, shuffle=False, device="cpu"):
                time, location = batch[-2].numpy().reshape(-1), batch[-1].numpy().reshape(-1)
                draws[:, time, location] = model.score_draws(*batch[:5], 3)[:, :, :2].numpy()
        means = draws.mean(axis=0)
        kept = ~excluded
        ranges = result["score_ranges"]
        self.assertEqual(ranges["points"], int(kept.sum()))
        for key, values, reduce in (("r_min", means[..., 0], np.min), ("r_max", means[..., 0], np.max),
                                    ("d_min", means[..., 1], np.min), ("d_max", means[..., 1], np.max)):
            self.assertAlmostEqual(ranges[key], float(reduce(values[kept])), places=6, msg=key)
        # Without the option the result has no ranges and the statistics do not change.
        torch.manual_seed(9)
        with redirect_stdout(io.StringIO()):
            plain = validation_objective(model, dataset, batch_size=5, device="cpu", mc_dropout_enabled=True,
                                         mc_samples=3, excluded=excluded)
        self.assertNotIn("score_ranges", plain)
        for key in ("mean", "std", "median", "p95"):
            self.assertEqual(plain[key], result[key])

    def test_paper_sum_with_frozen_ranges_fits_nothing_on_the_test(self):
        rng = np.random.default_rng(4)
        raw_g, raw_d = rng.random((3, 6, 5)), rng.random((3, 6, 5)) - .5
        store = ScoreStore(backend="memory")
        g, gs, d, ds, cov = summarize_raw_mc_components(raw_g, raw_d, store)
        ranges = dict(r_min=.1, r_max=.6, d_min=-.2, d_max=.3, points=17)
        mean, std, normalization = normalize_paper_mc_scores(g, gs, d, ds, cov, store, raw_generator=raw_g,
                                                             raw_discriminator=raw_d, ranges=ranges)
        draws = (raw_g - .1) / .5 + (raw_d + .2) / .5
        np.testing.assert_allclose(mean, draws.mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(std, draws.std(axis=0), rtol=1e-5, atol=1e-7)
        self.assertEqual((normalization["r_min"], normalization["d_max"], normalization["fit_points"]), (.1, .3, 17))
        self.assertFalse(normalization["transductive"])
        self.assertGreater(float(mean.max()), 1)  # Not clipped: the test may leave the validation ranges.
        store.close()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.times = pd.date_range("2002-12-29", periods=24, freq="3h")

    def fit(self, name, **overrides):
        values = np.random.default_rng(3).normal(size=(24, 9, 2)).astype(np.float32)
        rows, cols = np.indices((3, 3)).reshape(2, -1)
        options = dict(train_timestamps=self.times[:10], test_timestamps=self.times[18:],
            validation=values[10:18], validation_timestamps=self.times[10:18],
            location_names=tuple(map(str, range(9))), feature_names=("a", "b"),
            latitudes=45 - rows * .5, longitudes=7 + cols * .5,
            epochs=1, batch_size=4, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, train_samples_per_epoch=6, device="cpu", grid_crs="EPSG:4326",
            timestep_hours=3, angular_grid_spacing=.5, grid_audit_knn=False, score_mode="validation",
            cache_normalized=False, mc_samples=3, monitoring_timestamps=4, monitoring_feature_mmd_samples=16,
            checkpoint_path=self.root / name / "model.pt")
        options.update(overrides)
        with redirect_stdout(io.StringIO()), fit_and_score_stgan(values[:10], values[18:], **options) as result:
            return SimpleNamespace(metadata=result.metadata, scores=np.array(result.test_scores),
                                   std=np.array(result.anomaly_std), generator=np.array(result.test_generator_scores),
                                   discriminator=np.array(result.test_discriminator_scores))

    def test_validation_ranges_and_d_dropout_through_a_run(self):
        run = self.fit("validation", discriminator_dropout_p=.3)
        ranges = run.metadata["validation_objective"]["score_ranges"]
        normalization = run.metadata["score_normalization"]
        for key in ("r_min", "r_max", "d_min", "d_max"):
            self.assertEqual(normalization[key], ranges[key])
        self.assertEqual(normalization["method"], "validation_component_minmax_frozen_then_sum_lambda_1")
        expected = ((run.generator - ranges["r_min"]) / (ranges["r_max"] - ranges["r_min"])
                    + (run.discriminator - ranges["d_min"]) / (ranges["d_max"] - ranges["d_min"]))
        np.testing.assert_allclose(run.scores, expected, rtol=1e-5, atol=1e-6)
        self.assertGreater(float(run.std.max()), 0)
        self.assertEqual(run.metadata["discriminator_dropout_p"], .3)
        model, payload = load_stgan_checkpoint(self.root / "validation" / "model.pt")
        self.assertEqual(payload["model_config"]["discriminator_dropout_p"], .3)
        self.assertTrue(model.discriminator.has_dropout)
        # Without D's dropout the configuration saved is that of every earlier run.
        plain = self.fit("plain", score_mode="paper")
        _, payload = load_stgan_checkpoint(self.root / "plain" / "model.pt")
        self.assertNotIn("discriminator_dropout_p", payload["model_config"])
        self.assertTrue(plain.metadata["score_normalization"]["transductive"])
        self.assertNotIn("score_ranges", plain.metadata["validation_objective"])

    def test_validation_mode_needs_a_validation(self):
        with self.assertRaises(ValueError):
            self.fit("none", validation=None, validation_timestamps=None)


if __name__ == "__main__":
    unittest.main()
