"""GAT training step: the faster patch handling gives the step it replaced.

`legacy_patch_batches` and `legacy_graph_step` are the previous implementation, kept here verbatim
as the reference: index structures rebuilt for every chunk, history and observed patches gathered
again in the G phase, one host check of the loss per chunk.
"""
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
from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits

from physiq_pv.anomaly_detection.stgan import (
    STGANCNNConfig, STGANGAT, STGANGraphDataset, build_spatial_grid, fit_and_score_stgan, grid_edge_index)
from physiq_pv.anomaly_detection.stgan import training as training_module
from physiq_pv.anomaly_detection.stgan.config import STGANGATConfig
from physiq_pv.anomaly_detection.stgan.model import masked_cell_mean
from physiq_pv.anomaly_detection.stgan.precision import autocast_context
from physiq_pv.anomaly_detection.stgan.training import TERM_NAMES, gan_train_step
from physiq_pv.experiments.stgan_wandb import default_config
from scripts import run_era5_stgan

BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
ERA5_NODES = 81 * 131  # 10611: with this chunk size the whole ERA5 grid of a timestamp is one chunk.


def legacy_patch_batches(model, recent, observed, predicted):
    batch, steps, nodes, features = recent.shape
    size = model.node_indices.shape[-1]
    for start in range(0, batch * nodes, model.discriminator_chunk_size):
        ids = torch.arange(start, min(start + model.discriminator_chunk_size, batch * nodes),
                           device=recent.device)
        times, centers = ids // nodes, ids % nodes
        indices = model.node_indices[centers]
        valid = (indices >= 0)[:, None]
        safe = indices.clamp_min(0).flatten(1)
        history = recent[times[:, None, None], torch.arange(steps, device=recent.device)[None, :, None],
                         safe[:, None]].reshape(-1, steps, size, size, features).permute(0, 1, 4, 2, 3)
        def gather(values):
            patch = values[times[:, None], safe].reshape(-1, size, size, features).permute(0, 3, 1, 2)
            return torch.where(valid, patch, 0.0)
        yield ids, torch.where(valid[:, None], history, 0.0), gather(observed), gather(predicted), valid


def legacy_graph_step(model, batch, generator_optimizer, discriminator_optimizer, *, reconstruction_weight=500.,
                      precision="fp32", discriminator_steps=1, generator_steps=1, return_terms=False):
    recent, trend, mask, calendar, observed = batch
    def amp():
        return autocast_context(precision, recent.device)
    logits_options = {"return_logits": True} if precision == "bf16" else {}
    adversarial_loss = binary_cross_entropy_with_logits if precision == "bf16" else binary_cross_entropy
    probability = (lambda output: torch.sigmoid(output.detach().float())) if precision == "bf16" else (
        lambda output: output.detach().float())
    count = recent.shape[0] * recent.shape[2]
    generator_optimizer.zero_grad()
    discriminator_losses, generator_losses = [], []
    discriminator_terms, generator_terms = [], []
    for _ in range(discriminator_steps):
        discriminator_optimizer.zero_grad()
        with amp(), torch.no_grad():
            generated = model.generator(recent, trend, mask, calendar).float()
        discriminator_loss, step_terms = recent.new_zeros(()), recent.new_zeros(4)
        for ids, history, real_patch, fake_patch, valid in legacy_patch_batches(model, recent, observed, generated):
            with amp():
                real, fake = model.discriminator.score_pair(history, real_patch, fake_patch, valid, **logits_options)
                real_loss = adversarial_loss(real, torch.zeros_like(real))
                fake_loss = adversarial_loss(fake, torch.ones_like(fake))
                loss = .5 * (real_loss + fake_loss) * (len(ids) / count)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite discriminator loss.")
            loss.backward()
            discriminator_loss += loss.detach()
            if return_terms:
                weight = len(ids) / count
                step_terms += torch.stack((.5 * real_loss.detach() * weight, .5 * fake_loss.detach() * weight,
                                           probability(real).sum() / count, probability(fake).sum() / count))
        discriminator_optimizer.step()
        discriminator_losses.append(discriminator_loss)
        if return_terms:
            discriminator_terms.append(step_terms)
    for parameter in model.discriminator.parameters():
        parameter.requires_grad_(False)
    try:
        for _ in range(generator_steps):
            generator_optimizer.zero_grad()
            with amp():
                generated = model.generator(recent, trend, mask, calendar).float()
            prediction_leaf = generated.detach().requires_grad_(True)
            generator_loss, step_terms = recent.new_zeros(()), recent.new_zeros(3)
            for ids, history, real_patch, fake_patch, valid in legacy_patch_batches(model, recent, observed, prediction_leaf):
                with amp():
                    fake = model.discriminator.score_current(model.discriminator.encode_history(history, valid),
                                                             fake_patch, valid, **logits_options)
                    errors = (fake_patch - real_patch).square()
                    reconstruction_error = masked_cell_mean(errors, valid).mean()
                    reconstruction_loss = reconstruction_weight * reconstruction_error
                    adversarial = adversarial_loss(fake, torch.zeros_like(fake))
                    loss = (reconstruction_loss + adversarial) * (len(ids) / count)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite generator loss.")
                loss.backward()
                generator_loss += loss.detach()
                if return_terms:
                    step_terms += torch.stack((reconstruction_error.detach(), reconstruction_loss.detach(),
                                               adversarial.detach())) * (len(ids) / count)
            generated.backward(prediction_leaf.grad)
            generator_optimizer.step()
            generator_losses.append(generator_loss)
            if return_terms:
                generator_terms.append(step_terms)
    finally:
        for parameter in model.discriminator.parameters():
            parameter.requires_grad_(True)
    mean = lambda losses: losses[0] if len(losses) == 1 else torch.stack(losses).mean()
    values = torch.cat((torch.stack(generator_terms).mean(dim=0), torch.stack(discriminator_terms).mean(dim=0)))
    return mean(generator_losses), mean(discriminator_losses), dict(zip(TERM_NAMES, values))


def old_step(model, batch, generator_optimizer, discriminator_optimizer, **options):
    return legacy_graph_step(model, batch, generator_optimizer, discriminator_optimizer, return_terms=True, **options)


def new_step(model, batch, generator_optimizer, discriminator_optimizer, **options):
    return gan_train_step(model, batch, generator_optimizer, discriminator_optimizer, return_terms=True, **options)


def fixture(height=36, width=40, *, recent=1, holes=37, chunk=256):
    """A grid of more than 1024 nodes, with holes, so that chunks of 256, 1024 and 2048 differ."""
    rows, cols = np.indices((height, width)).reshape(2, -1)
    keep = np.sort(np.random.default_rng(3).permutation(len(rows))[holes:])
    rows, cols = rows[keep], cols[keep]
    grid = build_spatial_grid(45 - rows * .5, 7 + cols * .5, grid_crs="EPSG:4326", angular_spacing=.5, audit_knn=False)
    steps = 10
    data = np.random.default_rng(7).normal(size=(steps, len(rows), 15)).astype(np.float32)
    times = pd.date_range("2004-12-30", periods=steps, freq="3h").as_unit("ns")
    dataset = STGANGraphDataset(data, times, grid, feature_minimum=np.zeros(15), feature_scale=np.ones(15),
                                recent_steps=recent, trend_steps=4, stride=1)
    torch.manual_seed(20)
    model = STGANGAT(n_features=15, hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=2,
                     edge_index=grid_edge_index(grid.row_indices, grid.column_indices), node_indices=grid.node_indices,
                     recent_steps=recent, gat_hidden_dim=3, gat_heads=2, discriminator_chunk_size=chunk)
    return dataset, model


def run(step, model, batch, *, chunk, device="cpu", seed=11, **options):
    """One step on a copy of `model`; everything a comparison needs, on the CPU."""
    model = copy.deepcopy(model).to(device).train()
    model.discriminator_chunk_size = chunk
    batch = tuple(values.to(device) for values in batch)
    g_optimizer = torch.optim.Adam(model.generator.parameters(), lr=1e-3)
    d_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=1e-3)
    outputs = {"pair": [], "current": [], "inside_pair": False}
    score_pair, score_current = model.discriminator.score_pair, model.discriminator.score_current

    def record_pair(*args, **kwargs):  # D phase: outputs on real and on generated patches.
        outputs["inside_pair"] = True
        real, fake = score_pair(*args, **kwargs)
        outputs["inside_pair"] = False
        outputs["pair"].append((real.detach().float().cpu(), fake.detach().float().cpu()))
        return real, fake

    def record_current(*args, **kwargs):  # G phase: D's output on the generated patches, with updated D.
        output = score_current(*args, **kwargs)
        if not outputs["inside_pair"]:
            outputs["current"].append(output.detach().float().cpu())
        return output
    torch.manual_seed(seed)
    with patch.object(model.discriminator, "score_pair", record_pair), \
            patch.object(model.discriminator, "score_current", record_current), \
            patch.object(model.generator, "forward", wraps=model.generator.forward) as generator_forward:
        generator_loss, discriminator_loss, terms = step(model, batch, g_optimizer, d_optimizer, **options)
    adam_steps = lambda optimizer: max(int(state["step"]) for state in optimizer.state.values())
    return {"generator_loss": generator_loss.detach().cpu(), "discriminator_loss": discriminator_loss.detach().cpu(),
            "terms": {name: value.detach().cpu() for name, value in terms.items()},
            "real": torch.cat([real for real, _ in outputs["pair"]]),
            "fake": torch.cat([fake for _, fake in outputs["pair"]]),
            "fake_for_generator": torch.cat(outputs["current"]),
            "patches_per_d_pass": sum(len(real) for real, _ in outputs["pair"]),
            "generator_forwards": generator_forward.call_count,
            "generator_updates": adam_steps(g_optimizer), "discriminator_updates": adam_steps(d_optimizer),
            "weights": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}}


def assert_same(test, actual, expected, *, exact, rtol=1e-5, atol=1e-7, weight_rtol=2e-5, weight_atol=2e-6):
    """Losses, their terms, D outputs, counts and updated weights of two steps."""
    def close(a, b, name, r=rtol, t=atol):
        if exact:
            test.assertTrue(torch.equal(a, b), name)
        else:
            torch.testing.assert_close(a, b, rtol=r, atol=t, msg=lambda text: f"{name}: {text}")
    for name in ("generator_loss", "discriminator_loss", "real", "fake", "fake_for_generator"):
        close(actual[name], expected[name], name)
    test.assertEqual(set(actual["terms"]), set(TERM_NAMES))
    for name in TERM_NAMES:  # Reconstruction, adversarial, D real/fake losses and mean outputs.
        close(actual["terms"][name], expected["terms"][name], name)
    for name in ("patches_per_d_pass", "generator_forwards", "generator_updates", "discriminator_updates"):
        test.assertEqual(actual[name], expected[name], name)
    for name, value in expected["weights"].items():
        close(actual["weights"][name], value, name, weight_rtol, weight_atol)


class PatchPlanTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_patches_order_masks_and_borders_are_those_of_the_previous_implementation(self):
        for recent in (1, 2):
            dataset, model = fixture(recent=recent)
            nodes = dataset.n_locations
            self.assertGreater(nodes, 1024)
            self.assertTrue(bool((model.node_indices < 0).any()))  # Borders and holes are in the patches.
            batch = dataset.fetch_batch([0, 3])
            recent_values, observed = batch[0], batch[4]
            predicted = torch.randn_like(observed)
            for chunk in (5, 256, 1024, 2048, ERA5_NODES, 100000):
                with self.subTest(recent=recent, chunk=chunk):
                    model.discriminator_chunk_size = chunk
                    new = list(model.patch_batches(recent_values, observed, predicted))
                    old = list(legacy_patch_batches(model, recent_values, observed, predicted))
                    self.assertEqual(len(new), -(-2 * nodes // chunk))
                    for ours, theirs in zip(new, old):
                        for index, (a, b) in enumerate(zip(ours, theirs)):
                            self.assertTrue(torch.equal(a, b), index)
                    # Every center of every timestamp, once, in (batch, node) order.
                    self.assertTrue(torch.equal(torch.cat([chunk_[0] for chunk_ in new]), torch.arange(2 * nodes)))
                    # The inputs gathered once per batch are the same patches.
                    for (ids, _, _, valid, history, real), reference in zip(model.patch_inputs(recent_values, observed), old):
                        for a, b in ((ids, reference[0]), (history, reference[1]), (real, reference[2]), (valid, reference[4])):
                            self.assertTrue(torch.equal(a, b))

    def test_static_structures_are_built_once_per_shape_chunk_and_device(self):
        dataset, model = fixture()
        nodes = dataset.n_locations
        keys = set(model.state_dict())
        recent, _, _, _, observed, *_ = dataset.fetch_batch([0])
        self.assertIsNone(model._patch_plan)
        with patch.object(torch, "arange", wraps=torch.arange) as arange:
            list(model.patch_batches(recent, observed, observed))
            built = arange.call_count
            plan = model._patch_plan
            for _ in range(3):  # Later batches of the same shape: nothing is rebuilt.
                list(model.patch_batches(recent, torch.randn_like(observed), observed))
                list(model.patch_inputs(torch.randn_like(recent), observed))
            self.assertEqual(arange.call_count, built)
        self.assertIs(model._patch_plan, plan)
        self.assertEqual(built, 1 + -(-nodes // 256))  # One per chunk and the recent-step index.
        self.assertEqual(set(model.state_dict()), keys)  # Not a parameter, not a buffer, not in checkpoints.
        # The plan holds indices only: ids, timestamp, patch nodes and their validity.
        self.assertEqual({item.dtype for chunk in plan[2] for item in chunk}, {torch.int64, torch.bool})
        # Another chunk size or another batch size: another plan, with the right chunks.
        model.discriminator_chunk_size = 1024
        self.assertEqual([len(chunk[0]) for chunk in model.patch_plan(1, 1, nodes, "cpu")[1]], [1024, nodes - 1024])
        self.assertIsNot(model._patch_plan, plan)
        two = dataset.fetch_batch([0, 1])
        self.assertEqual(sum(len(chunk[0]) for chunk in model.patch_batches(two[0], two[4], two[4])), 2 * nodes)
        clone = copy.deepcopy(model)
        self.assertEqual(sum(len(chunk[0]) for chunk in clone.patch_batches(recent, observed, observed)), nodes)


class StepEquivalenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.dataset, self.model = fixture()
        self.nodes = self.dataset.n_locations
        self.batch = self.dataset.fetch_batch([2])[:5]  # batch_size = 1: one timestamp, every node.

    def test_same_chunk_size_reproduces_the_previous_step_exactly(self):
        for d_steps, g_steps in ((1, 1), (2, 1), (1, 2)):
            for chunk in (256, 1024, ERA5_NODES):
                with self.subTest(updates=f"{d_steps}:{g_steps}", chunk=chunk):
                    options = dict(chunk=chunk, discriminator_steps=d_steps, generator_steps=g_steps)
                    new = run(new_step, self.model, self.batch, **options)
                    old = run(old_step, self.model, self.batch, **options)
                    assert_same(self, new, old, exact=True)
                    # Counts: every patch in each D pass, one G forward per optimizer step, the same updates.
                    self.assertEqual(new["patches_per_d_pass"], d_steps * self.nodes)
                    self.assertEqual(len(new["fake_for_generator"]), g_steps * self.nodes)
                    self.assertEqual((new["generator_forwards"], new["discriminator_updates"], new["generator_updates"]),
                                     (d_steps + g_steps, d_steps, g_steps))
                    self.assertFalse(any(torch.equal(value, self.model.state_dict()[name])
                                         for name, value in new["weights"].items() if "weight" in name
                                         and "reset" not in name))  # The step did train.

    def test_chunks_of_256_1024_and_2048_give_the_same_step(self):
        reference = run(old_step, self.model, self.batch, chunk=256)
        for chunk in (7, 256, 1024, 2048, ERA5_NODES):
            with self.subTest(chunk=chunk):
                new = run(new_step, self.model, self.batch, chunk=chunk)
                assert_same(self, new, reference, exact=False)
                self.assertEqual((new["patches_per_d_pass"], new["discriminator_updates"], new["generator_updates"]),
                                 (self.nodes, 1, 1))
        # Chunks only split the same sum: with the weight of the reconstruction changed too.
        first, second = (run(new_step, self.model, self.batch, chunk=chunk, reconstruction_weight=50.)
                         for chunk in (256, ERA5_NODES))
        assert_same(self, second, first, exact=False)
        self.assertFalse(torch.equal(first["generator_loss"], reference["generator_loss"]))

    def test_whole_era5_grid_in_one_chunk_matches_the_previous_step_with_chunks_of_256(self):
        dataset, model = fixture(height=81, width=131, holes=0)
        self.assertEqual(dataset.n_locations, ERA5_NODES)
        batch = dataset.fetch_batch([2])[:5]  # batch_size = 1.
        model.discriminator_chunk_size = ERA5_NODES
        self.assertEqual([len(chunk[0]) for chunk in model.patch_plan(1, 1, ERA5_NODES, "cpu")[1]], [ERA5_NODES])
        model.discriminator_chunk_size = 256
        self.assertEqual(len(model.patch_plan(1, 1, ERA5_NODES, "cpu")[1]), 42)
        for d_steps, g_steps in ((1, 1), (2, 1), (1, 2)):
            with self.subTest(updates=f"{d_steps}:{g_steps}"):
                options = dict(discriminator_steps=d_steps, generator_steps=g_steps)
                reference = run(old_step, model, batch, chunk=256, **options)
                new = run(new_step, model, batch, chunk=ERA5_NODES, **options)
                # Discriminator and generator loss, reconstruction and adversarial terms, D outputs on
                # real and generated patches, weights after the step, patches and optimizer updates.
                assert_same(self, new, reference, exact=False)
                self.assertEqual(new["patches_per_d_pass"], d_steps * ERA5_NODES)
                self.assertEqual(len(new["fake_for_generator"]), g_steps * ERA5_NODES)
                self.assertEqual((new["generator_forwards"], new["discriminator_updates"], new["generator_updates"]),
                                 (d_steps + g_steps, d_steps, g_steps))
        # One chunk in both implementations: the same step, bit for bit.
        assert_same(self, run(new_step, model, batch, chunk=ERA5_NODES), run(old_step, model, batch, chunk=ERA5_NODES),
                    exact=True)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_fp32_matches_the_previous_step_for_256_and_1024(self):
        reference = run(old_step, self.model, self.batch, chunk=256, device="cuda")
        for chunk in (256, 1024, 2048, ERA5_NODES):
            with self.subTest(chunk=chunk):
                new = run(new_step, self.model, self.batch, chunk=chunk, device="cuda")
                assert_same(self, new, reference, exact=False, rtol=1e-4, atol=1e-6, weight_rtol=1e-4, weight_atol=5e-6)

    @unittest.skipUnless(BF16, "needs a CUDA GPU with native BF16")
    def test_bf16_matches_the_previous_step_for_256_and_1024(self):
        reference = run(old_step, self.model, self.batch, chunk=256, device="cuda", precision="bf16")
        for chunk in (256, 1024, 2048, ERA5_NODES):
            with self.subTest(chunk=chunk):
                new = run(new_step, self.model, self.batch, chunk=chunk, device="cuda", precision="bf16")
                # BF16 keeps about three significant digits; the sums over chunks are in FP32.
                assert_same(self, new, reference, exact=False, rtol=2e-2, atol=2e-3, weight_rtol=2e-2, weight_atol=2e-3)

    def test_a_non_finite_chunk_is_raised_before_any_update(self):
        for chunk in (256, 1024, ERA5_NODES):
            chunks = -(-self.nodes // chunk)
            # Discriminator: the loss of one chunk is not finite (not the last one, when there are several).
            model = copy.deepcopy(self.model)
            model.discriminator_chunk_size = chunk
            before = copy.deepcopy(model.state_dict())
            g_optimizer = torch.optim.Adam(model.generator.parameters(), lr=1e-3)
            d_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=1e-3)
            calls = []

            def poisoned_bce(output, target):
                calls.append(1)
                value = binary_cross_entropy(output, target)
                # The real patches of the second chunk, or of the only one.
                return value * float("inf") if len(calls) == min(3, 2 * chunks - 1) else value
            with self.subTest(network="discriminator", chunk=chunk), \
                    patch.object(training_module, "binary_cross_entropy", poisoned_bce), \
                    self.assertRaisesRegex(FloatingPointError, "Non-finite discriminator loss"):
                gan_train_step(model, self.batch, g_optimizer, d_optimizer)
            self.assertEqual(len(calls), 2 * chunks)  # Every chunk ran; the check is one, at the end.
            self.assertFalse(d_optimizer.state or g_optimizer.state)  # No optimizer step at all.
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]), name)
            # Generator: its loss alone is not finite, in one chunk; D has updated, G must not.
            model = copy.deepcopy(self.model)
            model.discriminator_chunk_size = chunk
            g_optimizer = torch.optim.Adam(model.generator.parameters(), lr=1e-3)
            d_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=1e-3)
            calls = []

            def poisoned(errors, valid):
                calls.append(1)
                value = masked_cell_mean(errors, valid)
                return value * float("inf") if len(calls) == min(2, chunks) else value
            with self.subTest(network="generator", chunk=chunk), \
                    patch.object(training_module, "masked_cell_mean", poisoned), \
                    self.assertRaisesRegex(FloatingPointError, "Non-finite generator loss"):
                gan_train_step(model, self.batch, g_optimizer, d_optimizer)
            self.assertEqual(len(calls), chunks)  # Every chunk ran; the check is one, at the end.
            self.assertTrue(d_optimizer.state)
            self.assertFalse(g_optimizer.state)
            for name, value in model.generator.state_dict().items():
                self.assertTrue(torch.equal(value, before["generator." + name]), name)


class DefaultTests(unittest.TestCase):
    def test_era5_gat_default_is_the_whole_grid_and_other_values_stay_available(self):
        self.assertEqual((STGANGATConfig().discriminator_chunk_size, default_config("era5").discriminator_chunk_size),
                         (ERA5_NODES, ERA5_NODES))
        self.assertEqual(ERA5_NODES, 10611)
        self.assertEqual(STGANCNNConfig().discriminator_chunk_size, 256)  # The generic default is untouched.
        self.assertEqual((default_config("era5").batch_size, STGANGATConfig().batch_size), (1, 1))
        era5 = ["train", "--prepared-dir", "unused", "--output-dir", "unused"]
        with patch.object(run_era5_stgan, "run_training", return_value={}) as train, redirect_stdout(io.StringIO()):
            run_era5_stgan.main(era5)
            self.assertEqual((train.call_args.kwargs["config"].discriminator_chunk_size,
                              train.call_args.kwargs["config"].batch_size), (ERA5_NODES, 1))
            for value in (256, 1024, 2048, ERA5_NODES):
                run_era5_stgan.main(era5 + ["--discriminator-chunk-size", str(value)])
                self.assertEqual(train.call_args.kwargs["config"].discriminator_chunk_size, value)
        for bad in (0, -1, 2.5):
            with self.assertRaises(ValueError):
                STGANGATConfig(discriminator_chunk_size=bad)

    def test_a_training_run_does_not_depend_on_the_chunk_size(self):
        torch.set_num_threads(1)
        rows, cols = np.indices((4, 5)).reshape(2, -1)
        times = pd.date_range("2003-12-28 12:00", periods=44, freq="3h").as_unit("ns")
        values = np.random.default_rng(3).normal(size=(44, 20, 2)).astype(np.float32)
        runs = {}
        with tempfile.TemporaryDirectory() as directory:
            for chunk in (3, 256, 1024, ERA5_NODES):
                with redirect_stdout(io.StringIO()), fit_and_score_stgan(
                        values[:28], values[36:], train_timestamps=times[:28], test_timestamps=times[36:],
                        validation=values[28:36], validation_timestamps=times[28:36],
                        feature_names=("a", "b"), location_names=tuple(map(str, range(20))),
                        latitudes=45 - rows * .5, longitudes=7 + cols * .5, epochs=2, batch_size=1,
                        hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1, trend_steps=2, device="cpu",
                        grid_crs="EPSG:4326", timestep_hours=3, angular_grid_spacing=.5, grid_audit_knn=False,
                        score_mode="paper", cache_normalized=False, mc_samples=2, monitoring_timestamps=4,
                        monitoring_feature_mmd_samples=16, spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2,
                        discriminator_chunk_size=chunk,
                        checkpoint_path=Path(directory) / str(chunk) / "model.pt") as result:
                    runs[chunk] = (pd.read_csv(Path(directory) / str(chunk) / "training_history.csv"),
                                   np.array(result.test_scores), result.metadata)
        reference, reference_scores, _ = runs[3]
        for chunk in (256, 1024, ERA5_NODES):
            history, scores, metadata = runs[chunk]
            with self.subTest(chunk=chunk):
                # The same examples and the same number of optimizer steps, epoch by epoch.
                for column in ("epoch", "samples", "discriminator_updates", "generator_updates"):
                    np.testing.assert_array_equal(history[column], reference[column], err_msg=column)
                self.assertEqual(history["samples"].tolist(), [26, 26])
                for column in history.columns:
                    if not column.endswith("seconds"):
                        np.testing.assert_allclose(history[column], reference[column], rtol=2e-4, atol=1e-6, err_msg=column)
                np.testing.assert_allclose(scores, reference_scores, rtol=2e-4, atol=1e-5)
                self.assertEqual(metadata["runtime"]["discriminator_chunk_size"], chunk)


if __name__ == "__main__":
    unittest.main()
