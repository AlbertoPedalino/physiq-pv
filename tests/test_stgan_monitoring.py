"""GAN monitoring: separate loss terms, D outputs, validation monitor (stability, feature discrepancy, MMD),
independent G/D learning rates, resume and W&B names.

The same file serves the ConvGRU and the GAT branch; graph cases run where the GAT exists.
"""
from contextlib import redirect_stdout
import copy
from dataclasses import replace
import io
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from torch.nn.functional import binary_cross_entropy

from physiq_pv.anomaly_detection import stgan
from physiq_pv.anomaly_detection.stgan import STGANCNNConfig, fit_and_score_stgan, masked_cell_mean
from physiq_pv.anomaly_detection.stgan.monitoring import (
    ValidationMonitor, evenly_spaced, feature_discrepancy, rbf_mmd2, score_stability, spearman)
from physiq_pv.anomaly_detection.stgan.training import TERM_NAMES, DeviceLossTotals, gan_train_step
from physiq_pv.experiments.stgan_wandb import RunLogger, default_config, resolve_config, run_tracked, validate_sweep
from test_stgan_mc_dropout import small_model
from test_stgan_performance import fixture

HAS_GRAPH = hasattr(stgan, "STGANGAT")
VALIDATION = ("validation_discriminator_feature_discrepancy", "validation_score_delta_mean",
              "validation_score_delta_median", "validation_score_spearman", "validation_discriminator_feature_mmd")


def training_paths(dropout=False):
    """(name, model, batch, dataset) per architecture of this branch."""
    dataset = fixture()
    yield "convgru", small_model(dropout_enabled=dropout).train(), dataset.fetch_batch(list(range(7)))[:5], dataset
    if HAS_GRAPH:
        from test_stgan_gat import fixture as graph_fixture
        dataset, model = graph_fixture()  # Several discriminator chunks per global batch.
        if not dropout:
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.
        yield "gat", model.train(), dataset.fetch_batch([0, 3])[:5], dataset


def discriminator_outputs(model, batch):
    """D(real) and D(generated) of the D step, recomputed outside the training step (dropout off)."""
    recent, trend, mask, calendar, observed = batch
    with torch.no_grad():
        generated = model.generator(recent, trend, mask, calendar)
        if not getattr(model, "global_graph", False):
            return model.discriminator.score_pair(recent, observed, generated, mask)
        pairs = [model.discriminator.score_pair(history, real, fake, valid)
                 for _, history, real, fake, valid in model.patch_batches(recent, observed, generated)]
        return tuple(torch.cat(values) for values in zip(*pairs))


def reconstruction_error(model, batch):
    """Masked MSE of G on the batch, recomputed outside the training step (dropout off): no weight in it."""
    recent, trend, mask, calendar, observed = batch
    with torch.no_grad():
        generated = model.generator(recent, trend, mask, calendar)
        if not getattr(model, "global_graph", False):
            return masked_cell_mean(torch.where(mask.bool(), generated - observed, 0.).square(), mask).mean()
        cells = [masked_cell_mean((fake - real).square(), valid)
                 for _, _, real, fake, valid in model.patch_batches(recent, observed, generated)]
        return torch.cat(cells).mean()


def step(model, batch, **options):
    model = copy.deepcopy(model)
    g_optimizer = torch.optim.Adam(model.generator.parameters(), lr=options.pop("g_lr", 1e-3))
    d_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=options.pop("d_lr", 1e-3))
    return model, gan_train_step(model, batch, g_optimizer, d_optimizer, **options)


class LossTermTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)

    def test_terms_rebuild_both_losses_for_every_update_ratio(self):
        for name, model, batch, _ in training_paths():
            for (d_steps, g_steps), weight in itertools.product(((1, 1), (2, 1), (1, 2)), (500., 7.)):
                with self.subTest(path=name, ratio=(d_steps, g_steps), weight=weight):
                    options = dict(discriminator_steps=d_steps, generator_steps=g_steps, reconstruction_weight=weight)
                    _, plain = step(model, batch, **options)
                    _, (g_loss, d_loss, terms) = step(model, batch, return_terms=True, **options)
                    # Asking for the terms does not change the losses.
                    self.assertTrue(torch.equal(plain[0], g_loss) and torch.equal(plain[1], d_loss))
                    self.assertEqual(tuple(terms), TERM_NAMES)
                    raw, weighted = terms["generator_reconstruction_loss"], terms["generator_reconstruction_weighted_loss"]
                    # generator_loss = weight * reconstruction + adversarial; the logged error has no weight.
                    torch.testing.assert_close(weighted, weight * raw, rtol=1e-6, atol=0.)
                    torch.testing.assert_close(weighted + terms["generator_adversarial_loss"], g_loss, rtol=1e-5, atol=1e-6)
                    self.assertGreater(float(weighted), float(raw))
                    self.assertGreater(float(raw), 0)
                    torch.testing.assert_close(terms["discriminator_real_loss"] + terms["discriminator_fake_loss"],
                                               d_loss, rtol=1e-5, atol=1e-6)
                    self.assertTrue(all(torch.isfinite(value) and not value.requires_grad for value in terms.values()))

    def test_weight_changes_the_weighted_term_and_not_the_reconstruction_error(self):
        raw, weighted, adversarial = TERM_NAMES[:3]
        for name, model, batch, _ in training_paths():
            with self.subTest(path=name):
                error = reconstruction_error(model, batch)  # The same definition for every architecture.
                runs = {weight: step(model, batch, reconstruction_weight=weight, return_terms=True)[1]
                        for weight in (1., 50., 500.)}
                for weight, (g_loss, _, terms) in runs.items():
                    torch.testing.assert_close(terms[raw], error, rtol=1e-5, atol=1e-8)
                    torch.testing.assert_close(terms[weighted], weight * error, rtol=1e-5, atol=1e-8)
                    torch.testing.assert_close(terms[weighted] + terms[adversarial], g_loss, rtol=1e-5, atol=1e-6)
                (_, _, unit), (_, _, low), (_, _, high) = runs.values()
                self.assertTrue(torch.equal(unit[raw], unit[weighted]))  # Weight 1: the two coincide.
                # The weight moves only the weighted term (the first G step sees the same G and D).
                for key in (raw, adversarial, *TERM_NAMES[3:]):
                    self.assertTrue(torch.equal(high[key], low[key]), key)
                torch.testing.assert_close(high[weighted], 10 * low[weighted], rtol=1e-6, atol=0.)
                self.assertNotEqual(float(high[weighted]), float(low[weighted]))

    def test_discriminator_outputs_follow_real_zero_fake_one(self):
        for name, model, batch, _ in training_paths():
            with self.subTest(path=name):
                real, fake = discriminator_outputs(model, batch)
                _, (_, _, terms) = step(model, batch, return_terms=True)
                torch.testing.assert_close(terms["discriminator_real_mean"], real.mean(), rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(terms["discriminator_fake_mean"], fake.mean(), rtol=1e-5, atol=1e-6)
                # Halves of the BCE with target 0 for real and 1 for generated.
                torch.testing.assert_close(terms["discriminator_real_loss"],
                                           .5 * binary_cross_entropy(real, torch.zeros_like(real)), rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(terms["discriminator_fake_loss"],
                                           .5 * binary_cross_entropy(fake, torch.ones_like(fake)), rtol=1e-5, atol=1e-6)
                # Training D alone (G frozen by a zero rate) drives D(real) to 0 and D(fake) to 1.
                trained = copy.deepcopy(model)
                g_optimizer = torch.optim.Adam(trained.generator.parameters(), lr=0.)
                d_optimizer = torch.optim.Adam(trained.discriminator.parameters(), lr=1e-2)
                history = [gan_train_step(trained, batch, g_optimizer, d_optimizer, return_terms=True)[2] for _ in range(100)]
                self.assertLess(history[-1]["discriminator_real_mean"], history[0]["discriminator_real_mean"])
                self.assertGreater(history[-1]["discriminator_fake_mean"], history[0]["discriminator_fake_mean"])
                self.assertLess(history[-1]["discriminator_real_mean"], history[-1]["discriminator_fake_mean"])

    def test_bf16_terms_are_probabilities(self):
        for module in ("precision", "scoring"):  # CPU autocast is a test surrogate of native CUDA BF16.
            patcher = patch(f"physiq_pv.anomaly_detection.stgan.{module}.validate_precision")
            patcher.start()
            self.addCleanup(patcher.stop)
        for name, model, batch, _ in training_paths():
            with self.subTest(path=name):
                _, (g_loss, d_loss, terms) = step(model, batch, precision="bf16", return_terms=True)
                for key in ("discriminator_real_mean", "discriminator_fake_mean"):
                    self.assertTrue(0 < float(terms[key]) < 1, key)
                torch.testing.assert_close(terms["discriminator_real_loss"] + terms["discriminator_fake_loss"], d_loss,
                                           rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(terms["generator_reconstruction_weighted_loss"]
                                           + terms["generator_adversarial_loss"], g_loss, rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(terms["generator_reconstruction_weighted_loss"],
                                           500. * terms["generator_reconstruction_loss"], rtol=1e-6, atol=0.)

    def test_epoch_totals_weight_terms_by_samples(self):
        totals = DeviceLossTotals(torch.device("cpu"))
        self.assertEqual(totals.term_means(), {})
        batches = [(3, {name: torch.tensor(float(index)) for index, name in enumerate(TERM_NAMES)}),
                   (1, {name: torch.tensor(float(index) + 4) for index, name in enumerate(TERM_NAMES)})]
        for count, terms in batches:
            totals.update(torch.tensor(1.), torch.tensor(2.), count, terms)
        self.assertEqual(totals.means(), [1., 2.])
        self.assertEqual(totals.term_means(), {name: index + 1. for index, name in enumerate(TERM_NAMES)})


class MetricTests(unittest.TestCase):
    def test_spearman_and_score_stability(self):
        rng = np.random.default_rng(0)
        scores = rng.normal(size=400)
        self.assertEqual(spearman(scores, scores), 1.)
        self.assertEqual(spearman(scores, np.exp(scores) * 3 + 1), 1.)  # Same ranking, other values.
        self.assertAlmostEqual(spearman(scores, -scores), -1., places=12)
        other = scores + rng.normal(size=400)
        tied_a, tied_b = np.round(scores, 1), np.round(other, 1)  # Many ties: average ranks.
        for a, b in ((scores, other), (tied_a, tied_b)):
            self.assertAlmostEqual(spearman(a, b), spearmanr(a, b).statistic, places=12)
        self.assertTrue(math.isnan(spearman(np.ones(5), scores[:5])))
        same = score_stability(scores, scores)
        self.assertEqual(same, {"score_delta_mean": 0., "score_delta_median": 0., "score_spearman": 1.})
        moved = score_stability(scores, other)
        self.assertEqual(moved["score_delta_mean"], np.abs(other - scores).mean())
        self.assertEqual(moved["score_delta_median"], np.median(np.abs(other - scores)))
        # The comparison is element by element: the same values in another order are another result.
        shuffled = score_stability(scores, rng.permutation(scores))
        self.assertGreater(shuffled["score_delta_mean"], 0)
        self.assertLess(shuffled["score_spearman"], .5)
        with self.assertRaises(ValueError):
            score_stability(scores, scores[:-1])

    def test_feature_discrepancy_is_a_mean_absolute_difference(self):
        real = torch.tensor([[1., 2.], [3., 5.]])
        fake = torch.tensor([[1.5, 2.], [2., 9.]])
        self.assertEqual(feature_discrepancy(real, real), 0.)
        self.assertAlmostEqual(feature_discrepancy(real, fake), (.5 + 0 + 1 + 4) / 4)
        self.assertEqual(feature_discrepancy(real, fake), feature_discrepancy(fake, real))

    def test_mmd_separates_distributions_and_matches_the_definition(self):
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(300, 6, generator=generator)
        same_distribution = torch.randn(300, 6, generator=generator)
        shifted = torch.randn(300, 6, generator=generator) + 1.5
        identical, sigma2 = rbf_mmd2(x, x)
        self.assertGreater(sigma2, 0)
        self.assertLess(abs(identical), 1e-2)
        self.assertLess(abs(rbf_mmd2(x, same_distribution)[0]), 2e-2)
        different = rbf_mmd2(x, shifted)[0]
        self.assertGreater(different, .2)
        self.assertGreater(rbf_mmd2(x, x + .5)[0], abs(identical) * 5)
        # Direct evaluation of the unbiased estimator with the same median bandwidth.
        pooled = torch.cat((x, shifted)).double()
        distances = torch.cdist(pooled, pooled).square()
        sigma2 = distances[torch.triu(torch.ones_like(distances, dtype=torch.bool), 1)].median()
        kernel = torch.exp(-distances / (2 * sigma2))
        n = len(x)
        kxx, kyy, kxy = kernel[:n, :n], kernel[n:, n:], kernel[:n, n:]
        expected = ((kxx.sum() - n) / (n * (n - 1)) + (kyy.sum() - n) / (n * (n - 1)) - 2 * kxy.mean()).item()
        value, used = rbf_mmd2(x, shifted)
        self.assertAlmostEqual(value, expected, places=10)
        self.assertAlmostEqual(used, sigma2.item(), places=12)
        # Chunking is an implementation detail; a bandwidth subsample still gives a comparable value.
        self.assertAlmostEqual(rbf_mmd2(x, shifted, chunk_size=7)[0], expected, places=10)
        self.assertAlmostEqual(rbf_mmd2(x, shifted, bandwidth_points=120)[0], expected, delta=.05)
        self.assertEqual(rbf_mmd2(torch.ones(5, 3), torch.ones(4, 3)), (0., 0.))
        with self.assertRaises(ValueError):
            rbf_mmd2(x[:1], shifted)

    def test_subset_positions_are_fixed_and_spread(self):
        self.assertEqual(evenly_spaced(100, 5).tolist(), [0, 25, 50, 74, 99])
        self.assertEqual(evenly_spaced(4, 32).tolist(), [0, 1, 2, 3])
        np.testing.assert_array_equal(evenly_spaced(5840, 32), evenly_spaced(5840, 32))


class MonitorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def monitor(self, dataset, **options):
        options = {"timestamps": 3, "batch_size": 5, "device": "cpu", **options}
        return ValidationMonitor(dataset, **options)

    def test_outputs_match_the_score_components_and_penultimate_features(self):
        for name, model, batch, _ in training_paths():
            with self.subTest(path=name), torch.no_grad():
                model.eval()
                g, d, real, fake = model.monitoring_outputs(*batch)
                # Same components as one scoring draw of the model (dropout off).
                draw = model.score_draws(*batch, 1)[0]
                torch.testing.assert_close(g, draw[:, 0], rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(d, draw[:, 1], rtol=1e-6, atol=1e-7)
                self.assertEqual(real.shape, fake.shape)
                self.assertEqual(real.shape, (len(g), model.discriminator.output[0].out_features))
                self.assertTrue(bool((real >= 0).all()))  # After the ReLU that precedes the last layer.
                # The penultimate activations are the ones the score itself is computed from.
                real_score, fake_score = discriminator_outputs(model, batch)
                torch.testing.assert_close(model.discriminator.score_from_penultimate(real), real_score, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(model.discriminator.score_from_penultimate(fake), fake_score, rtol=1e-6, atol=1e-7)
                self.assertGreater(feature_discrepancy(real, fake), 0)

    def test_monitoring_leaves_model_rng_and_gradients_untouched(self):
        for name, model, batch, dataset in training_paths(dropout=True):
            with self.subTest(path=name):
                g_optimizer = torch.optim.Adam(model.generator.parameters())
                d_optimizer = torch.optim.Adam(model.discriminator.parameters())
                gan_train_step(model, batch, g_optimizer, d_optimizer)  # Leaves real gradients behind.
                weights = copy.deepcopy(model.state_dict())
                gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
                modes = [module.training for module in model.modules()]
                np.random.seed(3)
                rng = torch.get_rng_state(), np.random.get_state()[1].copy()
                metrics = self.monitor(dataset).evaluate(model, 1)
                self.assertTrue(math.isfinite(metrics["validation_discriminator_feature_discrepancy"]))
                self.assertGreater(metrics["validation_discriminator_feature_discrepancy"], 0)
                self.assertTrue(torch.equal(rng[0], torch.get_rng_state()))
                np.testing.assert_array_equal(rng[1], np.random.get_state()[1])
                self.assertEqual(modes, [module.training for module in model.modules()])
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, weights[key]), key)
                for parameter, before in zip(model.parameters(), gradients):
                    self.assertTrue(parameter.requires_grad)
                    self.assertTrue(parameter.grad is None if before is None else torch.equal(parameter.grad, before))

    def test_same_cells_same_order_every_epoch_and_fixed_factors(self):
        for name, model, batch, dataset in training_paths():
            with self.subTest(path=name):
                monitor = self.monitor(dataset)
                self.assertEqual(monitor.n_cells, len(monitor.positions) * dataset.n_locations)
                np.testing.assert_array_equal(monitor.positions, self.monitor(dataset).positions)
                first = monitor.evaluate(model, 1)
                self.assertEqual(tuple(first), VALIDATION)
                self.assertTrue(all(first[key] is None for key in VALIDATION[1:4]))  # Nothing to compare with yet.
                factors, scores = dict(monitor.normalization), monitor.previous_scores.copy()
                self.assertEqual((factors["fitted_epoch"], len(scores)), (1, monitor.n_cells))
                self.assertEqual((scores.min() >= 0, round(scores.max(), 6) <= 2), (True, True))
                # Cells are the fixed validation timestamps, each with every location, in that order.
                with torch.no_grad():
                    model.eval()
                    monitored = monitor._components(model)[0]
                    everything = model.monitoring_outputs(*dataset.fetch_batch(np.arange(len(dataset)))[:5])[0]
                    model.train()
                expected = everything.reshape(len(dataset.targets), dataset.n_locations)[monitor.positions]
                self.assertGreater(len(np.unique(expected.numpy().round(6))), monitor.n_cells // 2)
                np.testing.assert_allclose(monitored, expected.reshape(-1).numpy(), rtol=1e-5, atol=1e-7)
                # The same weights again: identical scores, in the same order.
                again = monitor.evaluate(model, 2)
                np.testing.assert_array_equal(monitor.previous_scores, scores)
                self.assertEqual((again["validation_score_delta_mean"], again["validation_score_delta_median"],
                                  again["validation_score_spearman"]), (0., 0., 1.))
                # After real training the scores move, but the factors of epoch 1 stay.
                g_optimizer = torch.optim.Adam(model.generator.parameters(), lr=1e-2)
                d_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=1e-2)
                for _ in range(5):
                    gan_train_step(model, batch, g_optimizer, d_optimizer)
                later = monitor.evaluate(model, 3)
                self.assertGreater(later["validation_score_delta_mean"], 0)
                self.assertGreaterEqual(later["validation_score_delta_mean"], 0)
                self.assertLess(later["validation_score_spearman"], 1)
                self.assertEqual(monitor.normalization, factors)
                # A monitor rebuilt from the checkpoint state continues the same comparison.
                resumed = self.monitor(dataset, state=monitor.state())
                self.assertEqual(resumed.normalization, factors)
                np.testing.assert_array_equal(resumed.previous_scores, monitor.previous_scores)
                repeated = resumed.evaluate(model, 4)
                self.assertEqual((repeated["validation_score_delta_mean"], repeated["validation_score_spearman"]), (0., 1.))
                with self.assertRaisesRegex(ValueError, "subset"):
                    self.monitor(dataset, timestamps=2, state=monitor.state())
                description = monitor.metadata()
                self.assertEqual((description["labels_used"], description["subset"]["cells"]), (False, monitor.n_cells))
                self.assertIn("monitoring_only", description["score_normalization"]["purpose"])

    def test_mmd_follows_its_period(self):
        _, model, _, dataset = next(training_paths())
        for period, expected in ((1, [True, True, True, True]), (2, [False, True, False, True]),
                                 (3, [False, False, True, False]), (0, [False] * 4)):
            monitor = self.monitor(dataset, feature_mmd_every_n_epochs=period, feature_mmd_samples=16)
            values = [monitor.evaluate(model, epoch)["validation_discriminator_feature_mmd"] for epoch in range(1, 5)]
            self.assertEqual([value is not None for value in values], expected, period)
            self.assertTrue(all(math.isfinite(value) for value in values if value is not None))
            self.assertEqual(len(monitor.feature_mmd_rows), 16)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def encoders(self):
        yield "convgru", {}
        if HAS_GRAPH:
            yield "gat", dict(spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2, discriminator_chunk_size=4)

    def fit(self, name, *, holdout=True, **overrides):
        values = np.random.default_rng(3).normal(size=(24, 9, 2)).astype(np.float32)
        times = pd.date_range("2002-12-29", periods=24, freq="3h")
        rows, cols = np.indices((3, 3)).reshape(2, -1)
        splits = (dict(validation=values[10:18], validation_timestamps=times[10:18]) if holdout else {})
        train = slice(0, 10) if holdout else slice(0, 18)
        options = dict(train_timestamps=times[train], test_timestamps=times[18:],
            location_names=tuple(map(str, range(9))), feature_names=("a", "b"),
            latitudes=45 - rows * .5, longitudes=7 + cols * .5,
            epochs=2, batch_size=4, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, train_samples_per_epoch=6, device="cpu", grid_crs="EPSG:4326",
            timestep_hours=3, angular_grid_spacing=.5, grid_audit_knn=False, score_mode="paper",
            cache_normalized=False, mc_samples=3, monitoring_timestamps=4, monitoring_feature_mmd_samples=16,
            checkpoint_path=self.root / name / "model.pt", **splits)
        options.update(overrides)
        with redirect_stdout(io.StringIO()), fit_and_score_stgan(values[train], values[18:], **options) as result:
            return SimpleNamespace(metadata=result.metadata, scores=np.array(result.test_scores),
                history=pd.read_csv(self.root / name / "training_history.csv"),
                final=torch.load(self.root / name / "model.pt", weights_only=False),
                epoch=lambda n: torch.load(self.root / name / f"model_epoch_{n}.pt", weights_only=False))

    def test_history_columns_validation_split_and_metadata(self):
        for encoder, extra in self.encoders():
            with self.subTest(encoder=encoder):
                run = self.fit(encoder, monitoring_feature_mmd_every_n_epochs=2, **extra)
                history = run.history
                for name in TERM_NAMES + ("generator_loss", "discriminator_loss") + VALIDATION:
                    self.assertIn(name, history.columns)
                np.testing.assert_allclose(history.generator_reconstruction_weighted_loss
                                           + history.generator_adversarial_loss, history.generator_loss, rtol=1e-5)
                np.testing.assert_allclose(history.generator_reconstruction_weighted_loss,
                                           500. * history.generator_reconstruction_loss, rtol=1e-5)
                reweighted = self.fit(encoder + "_weight", generator_reconstruction_weight=7., **extra).history
                np.testing.assert_allclose(reweighted.generator_reconstruction_weighted_loss,
                                           7. * reweighted.generator_reconstruction_loss, rtol=1e-5)
                np.testing.assert_allclose(reweighted.generator_reconstruction_weighted_loss
                                           + reweighted.generator_adversarial_loss, reweighted.generator_loss, rtol=1e-5)
                np.testing.assert_allclose(history.discriminator_real_loss + history.discriminator_fake_loss,
                                           history.discriminator_loss, rtol=1e-5)
                self.assertTrue(history[["discriminator_real_mean", "discriminator_fake_mean"]].stack().between(0, 1).all())
                # Epoch 1 has nothing to compare with; the MMD runs every second epoch.
                self.assertTrue(history.loc[0, list(VALIDATION[1:4])].isna().all())
                self.assertTrue(history.loc[1, list(VALIDATION[1:4])].notna().all())
                self.assertEqual(history.validation_discriminator_feature_mmd.notna().tolist(), [False, True])
                self.assertTrue(history.validation_discriminator_feature_discrepancy.gt(0).all())
                metadata = run.metadata
                self.assertEqual(list(metadata["splits"]), ["train", "validation", "test"])
                self.assertEqual(metadata["splits"]["train"]["timestamps"], 10)
                self.assertEqual(metadata["test_context"]["source"], "preceding_validation_history")
                monitoring = metadata["monitoring"]
                self.assertEqual((monitoring["subset"]["n_timestamps"], monitoring["subset"]["cells"]), (4, 36))
                self.assertEqual(monitoring["score_normalization"]["fitted_epoch"], 1)
                self.assertEqual(monitoring["discriminator_feature_mmd"]["every_n_epochs"], 2)
                # The monitoring factors are not the factors of the official score.
                self.assertNotEqual(monitoring["score_normalization"]["r_max"], metadata["score_normalization"]["r_max"])
                self.assertEqual(metadata["score_normalization"]["fit_period"], "complete_test_mc_mean_components")
                self.assertIsNotNone(run.epoch(2)["monitoring"])
                # Without a holdout nothing is monitored and nothing changes name.
                merged = self.fit(encoder + "_merged", holdout=False, **extra)
                self.assertEqual((list(merged.metadata["splits"]), merged.metadata["monitoring"]), (["train", "test"], None))
                self.assertFalse(any(name in merged.history.columns for name in VALIDATION))
                self.assertIn("generator_reconstruction_loss", merged.history.columns)

    def test_every_architecture_monitors_the_same_validation_cells(self):
        values = np.random.default_rng(3).normal(size=(24, 9, 2)).astype(np.float32)  # The data of fit().
        times = pd.date_range("2002-12-29", periods=24, freq="3h")
        chosen = [0, 2, 5, 7]  # 4 of the 8 validation timestamps: evenly spaced, both ends included.
        monitored = {}
        for encoder, extra in self.encoders():
            with self.subTest(encoder=encoder):
                monitors, build = [], ValidationMonitor.__init__
                def capture(monitor, *args, **kwargs):
                    build(monitor, *args, **kwargs)
                    monitors.append(monitor)
                with patch.object(ValidationMonitor, "__init__", capture):
                    run = self.fit("cells_" + encoder, epochs=1, **extra)
                monitor, = monitors
                self.assertEqual(run.metadata["splits"]["validation"],
                                 {"start": str(times[10]), "end": str(times[17]), "timestamps": 8})
                self.assertEqual(run.metadata["monitoring"]["subset"], {
                    "selection": "evenly_spaced_validation_targets_times_all_locations",
                    "timestamps": [str(time) for time in times[10:18][chosen]],
                    "n_timestamps": 4, "n_locations": 9, "cells": 36})
                # One monitored value per (timestamp, location): the centre of the patch for the ConvGRU,
                # the node for the GAT. Batching differs, the observed cells do not.
                batch = monitor.dataset.fetch_batch(monitor.sample_indices)
                mask, observed = batch[2].numpy(), batch[4].numpy()
                if observed.ndim == 4:  # [cell, feature, patch row, patch column]
                    centre = observed.shape[-1] // 2
                    observed, mask = observed[:, :, centre, centre], mask[..., centre, centre]
                observed, mask = observed.reshape(-1, values.shape[-1]), mask.reshape(-1)
                self.assertEqual((len(observed), len(mask), bool((mask > 0).all())), (36, 36, True))
                expected = monitor.dataset._normalise(values[10:18][chosen]).reshape(-1, values.shape[-1])
                np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-7)
                monitored[encoder] = observed
        for observed in monitored.values():
            np.testing.assert_array_equal(observed, monitored["convgru"])

    def test_monitoring_does_not_change_the_training_trajectory(self):
        for encoder, extra in self.encoders():
            with self.subTest(encoder=encoder):
                runs = [self.fit(f"{encoder}_{count}", monitoring_timestamps=count, **extra) for count in (0, 4)]
                self.assertIsNone(runs[0].metadata["monitoring"])
                self.assertIsNotNone(runs[1].metadata["monitoring"])
                for checkpoint in (lambda run: run.final, lambda run: run.epoch(1), lambda run: run.epoch(2)):
                    off, on = (checkpoint(run)["model_state_dict"] for run in runs)
                    for key, value in off.items():
                        self.assertTrue(torch.equal(value, on[key]), key)
                np.testing.assert_array_equal(runs[0].scores, runs[1].scores)  # MC draws included.
                for column in ("generator_loss", "discriminator_loss", *TERM_NAMES):
                    np.testing.assert_array_equal(runs[0].history[column], runs[1].history[column])

    def test_independent_rates_reach_their_optimizers_and_legacy_rates_still_work(self):
        default = STGANCNNConfig()
        self.assertEqual((default.generator_learning_rate, default.discriminator_learning_rate,
                          default.effective_generator_learning_rate, default.effective_discriminator_learning_rate),
                         (None, None, .001, .001))
        legacy = STGANCNNConfig(learning_rate=.0002, discriminator_lr_ratio=2.)
        self.assertEqual((legacy.effective_generator_learning_rate, legacy.effective_discriminator_learning_rate),
                         (.0002, .0004))
        independent = STGANCNNConfig(generator_learning_rate=.0002, discriminator_learning_rate=.0007)
        self.assertEqual((independent.effective_generator_learning_rate, independent.effective_discriminator_learning_rate),
                         (.0002, .0007))
        self.assertEqual(STGANCNNConfig(learning_rate=.0003, discriminator_learning_rate=.0001)
                         .effective_discriminator_learning_rate, .0001)
        for bad in (dict(generator_learning_rate=.0002, learning_rate=.0005),
                    dict(discriminator_learning_rate=.0002, discriminator_lr_ratio=2.),
                    dict(generator_learning_rate=0.), dict(discriminator_learning_rate=float("nan")),
                    dict(discriminator_learning_rate=True)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                STGANCNNConfig(**bad)
        for encoder, extra in self.encoders():
            for label, rates, expected in (
                    ("independent", dict(generator_learning_rate=.0002, discriminator_learning_rate=.0007), (.0002, .0007)),
                    ("legacy", dict(lr=.0002, discriminator_lr_ratio=2.), (.0002, .0004)),
                    ("default", {}, (.001, .001))):
                with self.subTest(encoder=encoder, rates=label):
                    seen = []
                    def observe(model, batch, g_optimizer, d_optimizer, **kwargs):
                        seen.append((g_optimizer.param_groups[0]["lr"], d_optimizer.param_groups[0]["lr"]))
                        return gan_train_step(model, batch, g_optimizer, d_optimizer, **kwargs)
                    with patch("physiq_pv.anomaly_detection.stgan.pipeline.gan_train_step", side_effect=observe):
                        run = self.fit(f"{encoder}_{label}", epochs=1, **rates, **extra)
                    self.assertEqual(set(seen), {expected})
                    self.assertEqual((run.metadata["generator_learning_rate"], run.metadata["discriminator_learning_rate"]),
                                     expected)
                    self.assertAlmostEqual(run.metadata["discriminator_lr_ratio"], expected[1] / expected[0])
                    saved = run.epoch(1)
                    self.assertEqual(saved["learning_rates"]["discriminator_learning_rate"], expected[1])
                    self.assertEqual(saved["discriminator_optimizer_state_dict"]["param_groups"][0]["lr"], expected[1])
                    self.assertEqual(run.final["training"]["generator_learning_rate"], expected[0])

    def test_resume_keeps_rates_split_and_monitoring_state(self):
        rates = dict(generator_learning_rate=.0002, discriminator_learning_rate=.0007)
        for encoder, extra in self.encoders():
            with self.subTest(encoder=encoder):
                # Complete epochs: their order depends on the epoch only, so a resumed run repeats it exactly.
                options = dict(dropout_enabled=False, mc_dropout_enabled=False, train_samples_per_epoch=None,
                               **rates, **extra)
                full = self.fit(f"{encoder}_full", **options)
                self.fit(f"{encoder}_cut", epochs=1, **options)
                checkpoint = self.root / f"{encoder}_cut" / "model_epoch_1.pt"
                resumed = self.fit(f"{encoder}_cut", resume_from=checkpoint, **options)
                self.assertEqual(resumed.metadata["resume"]["completed_epochs"], 1)
                for key, value in full.final["model_state_dict"].items():
                    self.assertTrue(torch.equal(value, resumed.final["model_state_dict"][key]), key)
                np.testing.assert_array_equal(full.scores, resumed.scores)
                # The second epoch is compared with the first one of the interrupted run, with its factors.
                self.assertEqual(resumed.metadata["monitoring"]["score_normalization"],
                                 full.metadata["monitoring"]["score_normalization"])
                np.testing.assert_array_equal(resumed.history[list(VALIDATION)].iloc[1],
                                              full.history[list(VALIDATION)].iloc[1])
                self.assertEqual(resumed.epoch(2)["learning_rates"], full.epoch(2)["learning_rates"])
                for change, message in ((dict(discriminator_learning_rate=.0005), "learning_rate"),
                                        (dict(generator_learning_rate=.0003), "learning_rate"),
                                        (dict(monitoring_timestamps=3), "subset")):
                    with self.assertRaisesRegex(ValueError, message):
                        self.fit(f"{encoder}_cut", resume_from=checkpoint, **{**options, **change})
                with self.assertRaisesRegex(ValueError, "splits"):  # A holdout run cannot continue as a merged one.
                    self.fit(f"{encoder}_cut", resume_from=checkpoint, holdout=False, **options)
                # A checkpoint from before monitoring (no state) is still accepted: the comparison restarts.
                payload = torch.load(checkpoint, weights_only=False)
                del payload["monitoring"]
                torch.save(payload, checkpoint)
                restarted = self.fit(f"{encoder}_cut", resume_from=checkpoint, **options)
                self.assertEqual(restarted.metadata["monitoring"]["score_normalization"]["fitted_epoch"], 2)
                self.assertTrue(pd.isna(restarted.history.validation_score_spearman.iloc[1]))


class InterfaceTests(unittest.TestCase):
    def test_wandb_names_and_skipped_values(self):
        class Run:
            def __init__(self):
                self.logs, self.definitions, self.summary = [], [], {}

            def define_metric(self, name, **kwargs):
                self.definitions.append(name)

            def log(self, values):
                self.logs.append(dict(values))

        run = Run()
        logger = RunLogger(run)
        self.assertIn("validation/*", run.definitions)
        record = {"epoch": 1, "generator_loss": 3., "discriminator_loss": .6, "samples": 4, "seconds": 2.,
                  "discriminator_updates": 8, "generator_updates": 4,
                  **{name: float(index) for index, name in enumerate(TERM_NAMES)},
                  "validation_discriminator_feature_discrepancy": .25, "validation_score_delta_mean": None,
                  "validation_score_delta_median": float("nan"), "validation_score_spearman": None,
                  "validation_discriminator_feature_mmd": .125, "validation_seconds": 1.5}
        logger.epoch(record)
        self.assertEqual(set(run.logs[0]), {
            "epoch", "train/epoch_seconds", "train/samples", "train/discriminator_updates", "train/generator_updates",
            "train/generator_loss", "train/generator_reconstruction_loss",
            "train/generator_reconstruction_weighted_loss", "train/generator_adversarial_loss",
            "train/discriminator_loss", "train/discriminator_real_loss", "train/discriminator_fake_loss",
            "train/discriminator_real_mean", "train/discriminator_fake_mean",
            "validation/discriminator_feature_discrepancy", "validation/discriminator_feature_mmd", "validation/seconds"})
        logger.epoch({**record, "epoch": 2, "validation_score_delta_mean": .1, "validation_score_delta_median": .05,
                      "validation_score_spearman": .9, "validation_discriminator_feature_mmd": None})
        self.assertEqual((run.logs[1]["validation/score_delta_mean"], run.logs[1]["validation/score_delta_median"],
                          run.logs[1]["validation/score_spearman"]), (.1, .05, .9))
        self.assertNotIn("validation/discriminator_feature_mmd", run.logs[1])

    def test_sweep_draft_and_wandb_configuration(self):
        import yaml
        path = Path(__file__).resolve().parents[1] / "sweeps/stgan_bayes.draft.yaml"
        draft = yaml.safe_load(path.read_text())
        parameters = draft["parameters"]
        self.assertEqual(draft["metric"], {"name": "validation/pca_mmd_rolling_mean", "goal": "minimize"})
        self.assertEqual({key for key, spec in parameters.items() if "value" not in spec},
                         {"generator_learning_rate", "discriminator_lr_ratio", "generator_reconstruction_weight",
                          "discriminator_generator_update_ratio"})
        self.assertEqual(parameters["generator_learning_rate"],
                         {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3})
        # D's rate is a multiple of G's, searched symmetrically around 1 on a log scale.
        self.assertEqual(parameters["discriminator_lr_ratio"],
                         {"distribution": "log_uniform_values", "min": .25, "max": 4.})
        self.assertNotIn("discriminator_learning_rate", parameters)
        reference = STGANCNNConfig()
        self.assertEqual((reference.effective_generator_learning_rate, reference.effective_discriminator_learning_rate),
                         (parameters["generator_learning_rate"]["max"],) * 2)
        self.assertEqual(parameters["generator_reconstruction_weight"],
                         {"distribution": "log_uniform_values", "min": 50., "max": 2000.})
        self.assertEqual(parameters["discriminator_generator_update_ratio"], {"values": ["1:1", "2:1", "1:2"]})
        for name in ("learning_rate", "discriminator_learning_rate"):  # No redundant way to set a rate.
            self.assertNotIn(name, parameters)
        fixed = {key: spec["value"] for key, spec in parameters.items() if "value" in spec}
        self.assertEqual((fixed["validation_holdout"], fixed["monitoring_timestamps"],
                          fixed["monitoring_feature_mmd_every_n_epochs"]), (True, 32, 1))
        base = default_config("era5")
        self.assertTrue(base.validation_holdout)
        for g_rate, d_rate in ((1e-5, 1e-3), (1e-3, 1e-5)):
            config, _ = resolve_config(base, {**fixed, "generator_learning_rate": g_rate,
                                              "discriminator_learning_rate": d_rate,
                                              "generator_reconstruction_weight": 50.,
                                              "discriminator_generator_update_ratio": "2:1"}, 20)
            self.assertEqual((config.effective_generator_learning_rate, config.effective_discriminator_learning_rate),
                             (g_rate, d_rate))
        # Earlier configurations keep working through the legacy pair of fields.
        config, _ = resolve_config(base, {"learning_rate": .0002, "discriminator_lr_ratio": .5}, 20)
        self.assertEqual(config.effective_discriminator_learning_rate, .0001)
        with self.assertRaises(ValueError):
            resolve_config(base, {"discriminator_learning_rate": .0002, "discriminator_lr_ratio": .5}, 20)
        # Sampled independent rates never land on a contradictory legacy value of the base configuration.
        sampled = {"generator_learning_rate": 1e-4, "discriminator_learning_rate": 1e-4}
        for legacy in ({"learning_rate": .0005}, {"discriminator_lr_ratio": 2.}):
            with self.subTest(legacy=legacy), self.assertRaisesRegex(ValueError, "once"):
                resolve_config(replace(base, **legacy), sampled, 20)
        # A search space names each rate in one form, whatever the values.
        complete = {**draft, "metric": {"name": "future/metric", "goal": "minimize"}}
        self.assertIs(validate_sweep(complete), complete)
        for other in ("learning_rate", "discriminator_learning_rate"):  # The second name of a searched rate.
            ambiguous = {**complete, "parameters": {**parameters, other: {"value": 1e-4}}}
            with self.subTest(other=other), self.assertRaisesRegex(ValueError, "not both"):
                validate_sweep(ambiguous)

    def test_model_config_names_each_rate_once(self):
        from scripts.run_stgan_wandb import parse_args
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.csv").touch()
            args = parse_args(["--backend", "pvgis", "--manifest", str(root / "manifest.csv"),
                               "--output-root", str(root / "runs"), "--device", "cpu", "--wandb-mode", "offline"])
            args.model_config, args.dry_run = root / "model.json", True
            # The legacy values below are the defaults: only the names can reveal the duplicate.
            cases = (({"learning_rate": .001, "generator_learning_rate": .0002}, None),
                     ({"discriminator_lr_ratio": 1., "discriminator_learning_rate": .0002}, None),
                     ({"generator_learning_rate": .0002, "discriminator_learning_rate": .0007}, (.0002, .0007)),
                     ({"learning_rate": .0002, "discriminator_lr_ratio": 2.}, (.0002, .0004)),
                     ({"learning_rate": .0002, "discriminator_learning_rate": .0007}, (.0002, .0007)))
            for overrides, expected in cases:
                args.model_config.write_text(json.dumps(overrides))
                with self.subTest(overrides=overrides), patch("wandb.init") as init, \
                        redirect_stdout(io.StringIO()) as stream:
                    if expected is None:
                        with self.assertRaisesRegex(ValueError, "not both"):
                            run_tracked(args)
                    else:
                        run_tracked(args)
                        config = STGANCNNConfig(**json.loads(stream.getvalue())["config"])
                        self.assertEqual((config.effective_generator_learning_rate,
                                          config.effective_discriminator_learning_rate), expected)
                    init.assert_not_called()

    def test_runners_wire_the_new_options(self):
        from scripts import run_era5_stgan, run_pvgis_stgan
        from physiq_pv.era5.data import ERA5Cubes
        from physiq_pv.era5.cube import CubeGrid
        era5 = ["train", "--prepared-dir", "unused", "--output-dir", "unused"]
        options = ["--generator-learning-rate", "0.0002", "--discriminator-learning-rate", "0.0007",
                   "--validation-holdout", "--monitoring-timestamps", "8", "--monitoring-feature-mmd-every-n-epochs", "3",
                   "--monitoring-feature-mmd-samples", "64"]
        with patch.object(run_era5_stgan, "run_training", return_value={}) as train, redirect_stdout(io.StringIO()):
            run_era5_stgan.main(era5)
            defaults = train.call_args.kwargs["config"]
            run_era5_stgan.main(era5 + options)
            config = train.call_args.kwargs["config"]
        self.assertEqual((defaults.validation_holdout, defaults.discriminator_learning_rate), (False, None))
        self.assertEqual((config.effective_generator_learning_rate, config.effective_discriminator_learning_rate,
                          config.validation_holdout, config.monitoring_timestamps, config.monitoring_feature_mmd_every_n_epochs,
                          config.monitoring_feature_mmd_samples), (.0002, .0007, True, 8, 3, 64))
        with patch.object(run_pvgis_stgan, "run_stgan") as train:
            run_pvgis_stgan.main(["--manifest", "unused", "--out-dir", "unused", "--discriminator-learning-rate", "0.0007"])
        self.assertEqual(train.call_args.kwargs["config"].effective_discriminator_learning_rate, .0007)
        # The two forms of D's rate cannot be given together, even with the default ratio.
        both = ["--discriminator-lr-ratio", "1", "--discriminator-learning-rate", "0.0007"]
        for runner, entry, required in ((run_era5_stgan, "run_training", era5),
                                        (run_pvgis_stgan, "run_stgan", ["--manifest", "unused", "--out-dir", "unused"])):
            with self.subTest(runner=runner.__name__), patch.object(runner, entry) as train, \
                    patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                runner.main(required + both)
            train.assert_not_called()
        with self.assertRaisesRegex(ValueError, "ERA5"):
            run_pvgis_stgan.run_stgan(manifest_path="unused", out_dir="unused", paper_top_k_percent=1.,
                                      config=STGANCNNConfig(validation_holdout=True))
        # The ERA5 entrypoint: 1980-2003 trains, 2004 is the validation, the test is untouched.
        values = np.zeros((7, 1, 1), np.float32)
        times = pd.date_range("2002-12-31 18:00", periods=7, freq="3h")
        lat, lon = np.array([45.]), np.array([7.])
        result = SimpleNamespace(test_timestamps=times[5:], metadata={},
                                 __enter__=None, __exit__=None)
        for holdout, train_length in ((True, 2), (False, 5)):
            seen = {}
            class Fit:
                def __init__(self, train, test, **kwargs):
                    seen.update(train=len(train), test=len(test), **kwargs)
                def __enter__(self):
                    return result
                def __exit__(self, *args):
                    return False
            cubes = ERA5Cubes(train=values[:2], validation=values[2:5], test=values[5:],
                train_timestamps=times[:2], validation_timestamps=times[2:5], test_timestamps=times[5:],
                location_names=("0",), feature_names=("a",), latitudes=lat, longitudes=lon)
            with tempfile.TemporaryDirectory() as tmp, \
                    patch.object(run_era5_stgan, "load_prepared", return_value=(cubes, CubeGrid.from_locations(lat, lon),
                        {"train_end_year": 2003, "validation_end_year": 2004, "test_start_year": 2005})), \
                    patch.object(run_era5_stgan, "fit_and_score_stgan", Fit), redirect_stdout(io.StringIO()):
                metadata = run_era5_stgan.run_training(prepared_dir="unused", output_dir=Path(tmp) / "run",
                    config=STGANCNNConfig(validation_holdout=holdout), device="cpu")
            self.assertEqual((seen["train"], seen["test"], "validation" in seen), (train_length, 2, holdout))
            self.assertNotIn("validation_holdout", seen)
            self.assertEqual((seen["score_mode"], metadata["effective_train_end_year"], metadata["validation_years"]),
                             ("paper", 2003 if holdout else 2004, [2004] if holdout else None))
            if holdout:
                self.assertEqual(len(seen["validation"]), 3)
                self.assertTrue(seen["validation_timestamps"].equals(times[2:5]))


if __name__ == "__main__":
    unittest.main()
