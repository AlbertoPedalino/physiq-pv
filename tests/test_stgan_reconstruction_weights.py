"""`reconstruction_weights="daily_change"`: the error of each variable is weighted by the inverse of its
daily change in the training data, in the loss and in the reconstruction component of the score."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch

from physiq_pv.anomaly_detection.stgan import STGANCNNConfig, fit_and_score_stgan, load_stgan_checkpoint
from physiq_pv.anomaly_detection.stgan.model import masked_cell_mean
from physiq_pv.anomaly_detection.stgan.pipeline import _daily_change_weights, _feature_minmax
from physiq_pv.anomaly_detection.stgan.scoring import scoring_mode
from physiq_pv.anomaly_detection.stgan.training import gan_train_step
from scripts import run_era5_stgan
from test_stgan_full_grid import FEATURES, case
from test_stgan_mmd import NAMES, TEST, TIMES, TRAIN, VALIDATION, build, build_pca, fields

WEIGHTS = [0.5, 1.0, 4.0]  # One per variable of the case.


def models(*, dropout=False):
    """The patch model and the full-grid model of one case, each with its batch."""
    c = case(recent_steps=2, dropout=dropout)
    return (("patch", c.patch_model, c.patch[:5]), ("full_grid", c.model, c.field[:5]))


def step(model, batch):
    """One training step; returns the losses, the terms and what the step showed."""
    seen = {}
    optimizers = [torch.optim.Adam(part.parameters(), lr=1e-3) for part in (model.generator, model.discriminator)]
    model.train()
    result = gan_train_step(model, batch, *optimizers, return_terms=True,
                            observe=lambda name, values: seen.setdefault(name, dict(values)))
    return result, seen


class DailyChangeTests(unittest.TestCase):
    def test_weights_are_the_inverse_mean_square_change_and_keep_the_scale(self):
        rng = np.random.default_rng(0)
        # Three variables whose values change with standard deviations 1, 3 and 0.1 from one day to the next.
        data = (rng.normal(size=(400, 6, 3)) * np.array([1.0, 3.0, 0.1])).astype(np.float32)
        scale = np.array([2.0, 5.0, 0.5])
        weights = _daily_change_weights(data, scale, steps_per_day=4, max_pairs=10 ** 6)
        change = (data[4:].astype(np.float64) - data[:-4]) * (2.0 / scale)
        mean_square = np.square(change).mean(axis=(0, 1))
        np.testing.assert_allclose(weights, mean_square.mean() / mean_square, rtol=1e-9)
        # Errors that follow the daily change give the loss they give without weights.
        self.assertAlmostEqual(float((weights * mean_square).mean()), float(mean_square.mean()))
        self.assertGreater(weights[2], weights[0])
        self.assertGreater(weights[0], weights[1])
        # A sample of the timestamps gives nearly the same weights.
        np.testing.assert_allclose(_daily_change_weights(data, scale, steps_per_day=4, max_pairs=100), weights, rtol=0.25)

    def test_data_without_a_daily_change_are_refused(self):
        data = np.random.default_rng(1).normal(size=(40, 4, 2)).astype(np.float32)
        with self.assertRaisesRegex(ValueError, "more than one day"):
            _daily_change_weights(data[:8], np.ones(2), steps_per_day=8)
        data[:, :, 1] = 3.0
        with self.assertRaisesRegex(ValueError, "does not change over 24 hours"):
            _daily_change_weights(data, np.ones(2), steps_per_day=8)


class ModelTests(unittest.TestCase):
    def test_without_weights_the_errors_are_untouched(self):
        for name, model, batch in models():
            with self.subTest(model=name):
                errors = torch.rand_like(batch[4])
                self.assertIsNone(model.reconstruction_feature_weights)
                self.assertIs(model.weighted_errors(errors), errors)

    def test_weights_multiply_the_errors_of_their_variable_and_are_not_learned_state(self):
        for name, model, batch in models():
            with self.subTest(model=name):
                state = set(model.state_dict())
                model.set_reconstruction_feature_weights(WEIGHTS, FEATURES)
                errors = torch.rand_like(batch[4])
                weighted = model.weighted_errors(errors)
                for index, weight in enumerate(WEIGHTS):
                    torch.testing.assert_close(weighted[:, index], errors[:, index] * weight)
                self.assertEqual(set(model.state_dict()), state)
                self.assertNotIn("reconstruction_feature_weights", dict(model.named_parameters()))

    def test_invalid_weights_are_refused(self):
        _, model, _ = models()[0]
        for weights in ([1.0, 2.0], [1.0, 0.0, 2.0], [1.0, float("nan"), 2.0], [1.0, -1.0, 2.0]):
            with self.assertRaisesRegex(ValueError, "one finite positive value per variable"):
                model.set_reconstruction_feature_weights(weights, FEATURES)


class TrainingStepTests(unittest.TestCase):
    def test_the_reconstruction_loss_is_the_weighted_error(self):
        for name, model, batch in models():
            with self.subTest(model=name):
                state = {key: value.clone() for key, value in model.state_dict().items()}
                mask, observed = batch[2], batch[4]
                torch.manual_seed(0)
                (_, _, plain), _ = step(model, batch)
                model.load_state_dict(state)
                model.set_reconstruction_feature_weights(WEIGHTS, FEATURES)
                torch.manual_seed(0)
                (generator_loss, _, terms), seen = step(model, batch)
                errors = torch.where(mask.bool(), seen["G_forward"]["generated"] - observed, 0.0).square()
                scale = torch.tensor(WEIGHTS).view(1, -1, 1, 1)
                torch.testing.assert_close(terms["generator_reconstruction_loss"],
                                           masked_cell_mean(errors * scale, mask).mean().double(), check_dtype=False)
                torch.testing.assert_close(
                    generator_loss.double(),
                    terms["generator_reconstruction_weighted_loss"] + terms["generator_adversarial_loss"],
                    check_dtype=False)
                self.assertFalse(torch.allclose(terms["generator_reconstruction_loss"], plain["generator_reconstruction_loss"]))

    def test_weights_of_one_change_nothing(self):
        for name, model, batch in models():
            with self.subTest(model=name):
                state = {key: value.clone() for key, value in model.state_dict().items()}
                results = []
                for weights in (None, [1.0] * FEATURES):
                    model.load_state_dict(state)
                    model.set_reconstruction_feature_weights(weights, FEATURES)
                    torch.manual_seed(0)
                    (generator_loss, discriminator_loss, _), _ = step(model, batch)
                    results.append((generator_loss, discriminator_loss, *model.state_dict().values()))
                for a, b in zip(*results):
                    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


class ScoreTests(unittest.TestCase):
    def test_the_reconstruction_component_is_weighted_and_the_errors_by_variable_are_not(self):
        for share_history in (True, False):
            for name, model, batch in models(dropout=True):
                with self.subTest(model=name, share_history=share_history):
                    recent, trend, mask, calendar, observed = batch
                    draws = {}
                    for label, weights in (("plain", None), ("double", [2.0] * FEATURES), ("mixed", WEIGHTS)):
                        model.set_reconstruction_feature_weights(weights, FEATURES)
                        with scoring_mode(model, True), torch.no_grad():
                            torch.manual_seed(0)
                            draws[label] = model.score_draws(recent, trend, mask, calendar, observed, 3,
                                                             share_history=share_history)
                            if label == "mixed":
                                # One pass without draws: the monitoring uses the same weighted error.
                                model.eval()
                                monitored = model.monitoring_outputs(recent, trend, mask, calendar, observed)[0]
                                predicted = model.generator(recent, trend, mask, calendar).float()
                                errors = torch.where(mask.bool(), predicted - observed, 0.0).square()
                                torch.testing.assert_close(monitored, model.reconstruction_score(
                                    errors * torch.tensor(WEIGHTS).view(1, -1, 1, 1), mask))
                    # Twice the weight of every variable is twice the reconstruction component.
                    torch.testing.assert_close(draws["double"][:, :, 0], 2 * draws["plain"][:, :, 0])
                    self.assertFalse(torch.allclose(draws["mixed"][:, :, 0], draws["plain"][:, :, 0]))
                    # D's component and the errors by variable do not depend on the weights.
                    for label in ("double", "mixed"):
                        torch.testing.assert_close(draws[label][:, :, 1:], draws["plain"][:, :, 1:])


class RunTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.values = fields()
        build_pca(self.root / "pca", self.values)
        build(self.root / "reference", self.values, self.root / "pca")

    def fit(self, name, **overrides):
        options = dict(train_timestamps=TIMES[TRAIN], test_timestamps=TIMES[TEST],
            validation=self.values[VALIDATION], validation_timestamps=TIMES[VALIDATION], **NAMES,
            epochs=1, batch_size=1, score_batch_size=1, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, device="cpu", grid_crs="EPSG:4326", cnn_training_mode="full_grid", timestep_hours=3,
            angular_grid_spacing=.5, grid_audit_knn=False, score_mode="paper", cache_normalized=False, mc_samples=3,
            monitoring_timestamps=4, monitoring_feature_mmd_samples=16, score_storage="memmap",
            checkpoint_path=self.root / name / "model.pt", score_dir=self.root / name / "scores",
            pca_reference=self.root / "pca", mmd_reference=self.root / "reference")
        options.update(overrides)
        printed = io.StringIO()
        with redirect_stdout(printed), fit_and_score_stgan(self.values[TRAIN], self.values[TEST], **options) as result:
            return result.metadata, printed.getvalue()

    def test_the_run_computes_records_and_reloads_the_weights(self):
        metadata, printed = self.fit("weighted", reconstruction_weights="daily_change")
        train = self.values[TRAIN]
        expected = _daily_change_weights(train, _feature_minmax(train)[1], steps_per_day=8)
        names = list(NAMES["feature_names"])
        self.assertEqual(metadata["reconstruction_weights"], "daily_change")
        self.assertEqual(list(metadata["reconstruction_feature_weights"]), names)
        np.testing.assert_allclose(list(metadata["reconstruction_feature_weights"].values()), expected, rtol=1e-6)
        self.assertIn("[stgan] reconstruction weights of the variables", printed)
        self.assertIn("reconstruction_error_of_each_variable_weighted_by_its_inverse_daily_change", str(metadata))
        model, payload = load_stgan_checkpoint(self.root / "weighted" / "model.pt")
        np.testing.assert_allclose(payload["model_config"]["reconstruction_feature_weights"], expected, rtol=1e-6)
        np.testing.assert_allclose(model.reconstruction_feature_weights.numpy(), expected, rtol=1e-6)
        # The errors by variable are saved without the weights: they are the squared errors of the cells.
        self.assertTrue((self.root / "weighted" / "scores" / "feature_scores.npy").is_file())
        # The default run has no weights, in its metadata and in its checkpoint.
        metadata, printed = self.fit("default")
        self.assertEqual(metadata["reconstruction_weights"], "uniform")
        self.assertIsNone(metadata["reconstruction_feature_weights"])
        self.assertNotIn("reconstruction weights", printed)
        model, payload = load_stgan_checkpoint(self.root / "default" / "model.pt")
        self.assertNotIn("reconstruction_feature_weights", payload["model_config"])
        self.assertIsNone(model.reconstruction_feature_weights)

    def test_the_weights_work_with_the_patch_model_and_the_seasonal_normalization(self):
        metadata, _ = self.fit("patch", reconstruction_weights="daily_change", cnn_training_mode="patch", batch_size=4,
                               score_batch_size=4)
        self.assertEqual(len(metadata["reconstruction_feature_weights"]), len(NAMES["feature_names"]))
        # A year of data with an annual and a daily cycle: with the seasonal normalization the weights
        # are read on the standardised training data, so they are not the ones of the raw data.
        times = pd.date_range("2003-01-01", periods=8 * 365 + 20, freq="3h").as_unit("ns")
        rows, cols = np.indices((3, 3)).reshape(2, -1)
        phase = 2 * np.pi * (times.dayofyear.to_numpy() / 365 + times.hour.to_numpy() / 24)
        data = (np.sin(phase)[:, None, None] * (1 + rows)[None, :, None]
                + np.random.default_rng(9).normal(size=(len(times), 9, 4)) * np.array([.1, .2, .4, .8])).astype(np.float32)
        split = 8 * 365
        options = dict(train_timestamps=times[:split], test_timestamps=times[split:],
            location_names=tuple(map(str, range(9))), feature_names=("a", "b", "c", "d"),
            latitudes=45 - rows * .5, longitudes=7 + cols * .5, hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=1,
            trend_steps=4, grid_crs="EPSG:4326", angular_grid_spacing=.5, grid_audit_knn=False, timestep_hours=3,
            device="cpu", mc_samples=2, score_mode="paper", train_samples_per_epoch=6, time_encoding="cyclic",
            epochs=1, reconstruction_weights="daily_change")
        weights = {}
        for normalization in ("minmax", "seasonal"):
            with redirect_stdout(io.StringIO()), fit_and_score_stgan(
                    data[:split], data[split:], normalization=normalization,
                    checkpoint_path=self.root / normalization / "model.pt", **options) as result:
                weights[normalization] = np.array(list(result.metadata["reconstruction_feature_weights"].values()))
            self.assertTrue(np.isfinite(weights[normalization]).all() and (weights[normalization] > 0).all())
        np.testing.assert_allclose(weights["minmax"], _daily_change_weights(
            data[:split], _feature_minmax(data[:split])[1], steps_per_day=8), rtol=1e-6)
        self.assertFalse(np.allclose(weights["seasonal"], weights["minmax"], rtol=1e-2))

    def test_a_resume_with_the_other_setting_is_refused(self):
        self.fit("weighted", reconstruction_weights="daily_change")
        with self.assertRaisesRegex(ValueError, "reconstruction_weights"):
            self.fit("resumed", epochs=2, resume_from=self.root / "weighted" / "model_epoch_1.pt")
        self.fit("default")
        with self.assertRaisesRegex(ValueError, "reconstruction_weights"):
            self.fit("resumed_default", epochs=2, reconstruction_weights="daily_change",
                     resume_from=self.root / "default" / "model_epoch_1.pt")
        # The same setting resumes.
        metadata, printed = self.fit("continued", epochs=2, reconstruction_weights="daily_change",
                                     resume_from=self.root / "weighted" / "model_epoch_1.pt")
        self.assertIn("completed_epochs=1/2", printed)
        self.assertEqual(metadata["reconstruction_weights"], "daily_change")

    def test_a_timestep_that_does_not_divide_the_day_is_refused(self):
        # The check is in the pipeline, before any training.
        times = pd.date_range("2003-12-28 12:00", periods=28, freq="5h").as_unit("ns")
        with self.assertRaisesRegex(ValueError, "whole number of timesteps in 24 hours"):
            self.fit("odd", reconstruction_weights="daily_change", timestep_hours=5, train_timestamps=times[:20],
                     test_timestamps=times[20:], validation=None, validation_timestamps=None, pca_reference=None,
                     mmd_reference=None)


class OptionTests(unittest.TestCase):
    def test_default_and_validation(self):
        self.assertEqual(STGANCNNConfig().reconstruction_weights, "uniform")
        self.assertEqual(STGANCNNConfig(reconstruction_weights="daily_change").reconstruction_weights, "daily_change")
        with self.assertRaisesRegex(ValueError, "reconstruction_weights must be uniform or daily_change"):
            STGANCNNConfig(reconstruction_weights="equal")

    def test_command_line(self):
        base = ["train", "--prepared-dir", "prepared", "--output-dir", "run"]
        self.assertEqual(run_era5_stgan.parser().parse_args(base).reconstruction_weights, "uniform")
        self.assertEqual(run_era5_stgan.parser().parse_args(
            base + ["--reconstruction-weights", "daily_change"]).reconstruction_weights, "daily_change")


if __name__ == "__main__":
    unittest.main()
