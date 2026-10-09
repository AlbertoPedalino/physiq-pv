"""Gated GAT recurrence: attention inside the GRU gates, as the graph convolution in the paper's GCGRU.

The pointwise recurrence stays the default; these tests cover the new option and what separates the two.
"""
from contextlib import redirect_stdout
import inspect
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

from physiq_pv.anomaly_detection.stgan import (
    GATGRU, GATGRUCell, STGANCNNConfig, STGANGAT, build_spatial_grid, fit_and_score_stgan, grid_edge_index,
    load_stgan_checkpoint,
)
from scripts import run_era5_stgan
from test_stgan_mmd import NAMES, TEST, TIMES, TRAIN, VALIDATION, build, build_pca, fields


def line_edges(nodes):
    """Edges of a 1 x nodes grid: every node receives from itself and its immediate neighbors."""
    return grid_edge_index(np.zeros(nodes, dtype=int), np.arange(nodes))


def model(recurrence, *, recent=3, height=3, width=4):
    rows, cols = np.indices((height, width)).reshape(2, -1)
    grid = build_spatial_grid(45 - rows * .5, 7 + cols * .5, grid_crs="EPSG:4326",
                              angular_spacing=.5, audit_knn=False)
    return STGANGAT(n_features=15, hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=2,
        edge_index=grid_edge_index(grid.row_indices, grid.column_indices), node_indices=grid.node_indices,
        recent_steps=recent, gat_hidden_dim=3, gat_heads=2, discriminator_chunk_size=5,
        gat_recurrence=recurrence)


class CellTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(20)

    def test_cell_is_the_gru_update_when_every_node_attends_only_to_itself(self):
        # With self loops only, every attention coefficient is 1 and a layer is the mean
        # over heads of its projection plus the bias: the gates can be written by hand.
        nodes, features, channels, heads = 5, 3, 4, 2
        cell = GATGRUCell(features, channels, heads)
        for layer in (cell.reset, cell.update, cell.candidate):
            torch.nn.init.normal_(layer.bias)
        edges = torch.arange(nodes).repeat(2, 1)
        values, hidden = torch.randn(2, nodes, features), torch.randn(2, nodes, channels)
        mask = torch.ones(2, nodes, 1)

        def linear(layer, joint):
            return layer.projection(joint).reshape(*joint.shape[:2], heads, channels).mean(2) + layer.bias

        inputs = torch.cat((values, mask), dim=-1)
        joint = torch.cat((inputs, hidden), dim=-1)
        reset = torch.sigmoid(linear(cell.reset, joint))
        update = torch.sigmoid(linear(cell.update, joint))
        candidate = torch.tanh(linear(cell.candidate, torch.cat((inputs, reset * hidden), dim=-1)))
        expected = update * hidden + (1 - update) * candidate
        torch.testing.assert_close(cell(values, hidden, mask, edges), expected, rtol=1e-5, atol=1e-6)

    def test_missing_nodes_keep_a_zero_state_and_their_values_reach_no_neighbor(self):
        nodes = 6
        encoder = GATGRU(3, 4, 2, 2, line_edges(nodes))
        sequence = torch.randn(1, 3, nodes, 3)
        mask = torch.ones(1, nodes, 1)
        mask[0, 2] = 0
        output = encoder(sequence, mask)
        self.assertTrue(torch.equal(output[0, 2], torch.zeros(4)))
        changed = sequence.clone()
        changed[0, :, 2] = torch.nan  # Unavailable cells may hold NaN.
        torch.testing.assert_close(encoder(changed, mask), output)

    def test_shapes_are_validated(self):
        encoder = GATGRU(3, 4, 1, 2, line_edges(5))
        with self.assertRaises(ValueError):
            encoder(torch.zeros(1, 2, 5, 4), torch.ones(1, 5, 1))
        with self.assertRaises(ValueError):
            encoder(torch.zeros(1, 2, 5, 3), torch.ones(1, 5))
        with self.assertRaises(ValueError):
            GATGRU(3, 4, 0, 2, line_edges(5))


class ReachTests(unittest.TestCase):
    """What separates the two recurrences: how far the first recent step is seen."""

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(20)

    @staticmethod
    def reach(encode, sequence):
        """Nodes of the first step whose values move the output of node 0."""
        sequence = sequence.clone().requires_grad_(True)
        encode(sequence)[0, 0].sum().backward()
        return set(torch.nonzero(sequence.grad[0, 0].abs().sum(-1) > 0).flatten().tolist())

    def test_gated_state_spreads_two_hops_per_step(self):
        # The first step starts from a zero state and reaches one hop. Afterwards the
        # reset gate and the candidate attend in sequence: two hops per step. A second
        # layer adds one hop on the first step and follows the same rule later.
        nodes = 12
        mask = torch.ones(1, nodes, 1)
        for steps, layers, farthest in ((1, 1, 1), (3, 1, 5), (5, 1, 9), (1, 2, 2), (3, 2, 7)):
            with self.subTest(steps=steps, layers=layers):
                encoder = GATGRU(3, 4, layers, 2, line_edges(nodes))
                sequence = torch.randn(1, steps, nodes, 3)
                reached = self.reach(lambda x: encoder(x, mask), sequence)
                self.assertEqual(reached, set(range(farthest + 1)))

    def test_pointwise_reach_does_not_grow_with_the_recent_steps(self):
        for recent in (1, 4):
            with self.subTest(recent=recent):
                generator = model("pointwise", recent=recent, height=1, width=12).generator
                mask = torch.ones(1, 12, 1)
                sequence = torch.randn(1, recent, 12, 15)

                def encode(x):
                    batch, steps, nodes, features = x.shape
                    spatial = generator.recent_encoder(x.reshape(batch * steps, nodes, features))
                    spatial = spatial.reshape(batch, steps, nodes, spatial.shape[-1])
                    return generator.recent_temporal(spatial.permute(0, 1, 3, 2)[..., None],
                        mask.transpose(1, 2)[..., None]).squeeze(-1).transpose(1, 2)

                self.assertEqual(self.reach(encode, sequence), {0, 1, 2})

    def test_gated_generator_reaches_farther_than_pointwise_on_the_same_window(self):
        recent = 4
        mask = torch.ones(1, 12, 1)
        sequence = torch.randn(1, recent, 12, 15)
        gated = model("gated", recent=recent, height=1, width=12).generator
        reached = self.reach(lambda x: gated.recent_encoder(x, mask), sequence)
        self.assertEqual(reached, set(range(2 * recent + 2)))  # Two recurrent layers: 2 * recent + 1 hops.


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(20)

    def batch(self, network, recent):
        nodes = network.generator.n_nodes
        return (torch.randn(2, recent, nodes, 15), torch.randn(2, nodes, 4, 15),
                torch.ones(2, nodes, 1), torch.randn(2, 31))

    def test_default_is_pointwise_and_its_parameters_are_unchanged(self):
        self.assertEqual(inspect.signature(STGANGAT).parameters["gat_recurrence"].default, "pointwise")
        self.assertEqual(STGANCNNConfig().gat_recurrence, "pointwise")
        names = set(model("pointwise").generator.state_dict())
        self.assertTrue(any(name.startswith("recent_temporal.") for name in names))
        self.assertTrue(any(name.startswith("recent_encoder.layers.0.projection") for name in names))

    def test_gated_generator_has_attention_gates_and_no_pointwise_gru(self):
        generator = model("gated").generator
        self.assertIsInstance(generator.recent_encoder, GATGRU)
        self.assertFalse(hasattr(generator, "recent_temporal"))
        self.assertEqual(len(generator.recent_encoder.layers), 2)
        names = set(generator.state_dict())
        for gate in ("reset", "update", "candidate"):
            self.assertIn(f"recent_encoder.layers.1.{gate}.attention_source", names)
        self.assertFalse(any(name.startswith("recent_temporal.") for name in names))

    def test_gated_forward_is_finite_and_equals_encode_then_decode(self):
        for recent in (1, 3):
            with self.subTest(recent=recent):
                network = model("gated", recent=recent).eval()
                recent_values, trend, mask, calendar = self.batch(network, recent)
                with torch.no_grad():
                    predicted = network.generator(recent_values, trend, mask, calendar)
                    encoded = network.generator.encode(recent_values, trend, mask, calendar)
                    decoded = network.generator.decode(*encoded, mask)
                self.assertEqual(predicted.shape, (2, network.generator.n_nodes, 15))
                self.assertTrue(torch.isfinite(predicted).all())
                torch.testing.assert_close(decoded, predicted)

    def test_gated_generator_trains_every_gate(self):
        network = model("gated", recent=3)
        recent_values, trend, mask, calendar = self.batch(network, 3)
        network.generator(recent_values, trend, mask, calendar).square().mean().backward()
        for name, parameter in network.generator.recent_encoder.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        gates = [parameter.grad.abs().sum() for name, parameter in network.generator.recent_encoder.named_parameters()
                 if name.endswith("projection.weight")]
        self.assertTrue(all(value > 0 for value in gates))

    def test_gated_generator_runs_under_bf16_autocast_close_to_fp32(self):
        # The run itself needs a GPU with native BF16; CPU autocast exercises the same casts.
        network = model("gated", recent=3).eval()
        batch = self.batch(network, 3)
        with torch.no_grad():
            reference = network.generator(*batch)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                mixed = network.generator(*batch).float()
        self.assertTrue(torch.isfinite(mixed).all())
        torch.testing.assert_close(mixed, reference, rtol=0, atol=5e-2)

    def test_invalid_recurrence_is_rejected(self):
        with self.assertRaises(ValueError):
            model("stacked")
        with self.assertRaises(ValueError):
            STGANCNNConfig(gat_recurrence="stacked")

    def test_command_line_option(self):
        parser = run_era5_stgan.parser()
        base = ["train", "--prepared-dir", "p", "--output-dir", "o"]
        self.assertEqual(parser.parse_args(base).gat_recurrence, "pointwise")
        self.assertEqual(parser.parse_args(base + ["--gat-recurrence", "gated"]).gat_recurrence, "gated")


class PipelineTests(unittest.TestCase):
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
            epochs=2, batch_size=1, hidden_size=4, n_layers=1, cnn_channels=2, cnn_layers=1,
            trend_steps=2, recent_steps=2, device="cpu", grid_crs="EPSG:4326",
            timestep_hours=3, angular_grid_spacing=.5, grid_audit_knn=False, score_mode="paper",
            cache_normalized=False, mc_samples=2, monitoring_timestamps=4, monitoring_feature_mmd_samples=16,
            spatial_encoder="gat", gat_hidden_dim=2, gat_heads=2, discriminator_chunk_size=4,
            checkpoint_path=self.root / name / "model.pt", score_dir=self.root / name / "scores",
            pca_reference=self.root / "pca", mmd_reference=self.root / "reference")
        options.update(overrides)
        stream = io.StringIO()
        with redirect_stdout(stream), fit_and_score_stgan(self.values[TRAIN], self.values[TEST], **options) as result:
            return SimpleNamespace(metadata=result.metadata, printed=stream.getvalue(),
                                   scores=np.array(result.test_scores))

    def test_gated_run_scores_saves_its_recurrence_and_reloads(self):
        run = self.fit("gated", gat_recurrence="gated")
        self.assertTrue(np.isfinite(run.scores).all())
        self.assertIn("'gat_recurrence': 'gated'", run.printed)
        network, payload = load_stgan_checkpoint(self.root / "gated" / "model.pt")
        self.assertEqual(payload["model_config"]["gat_recurrence"], "gated")
        self.assertIsInstance(network.generator.recent_encoder, GATGRU)

    def test_resume_rejects_the_other_recurrence(self):
        self.fit("gated", gat_recurrence="gated", epochs=1)
        checkpoint = self.root / "gated" / "model_epoch_1.pt"
        with self.assertRaisesRegex(ValueError, "gat_recurrence"):
            self.fit("gated", gat_recurrence="pointwise", epochs=2, resume_from=checkpoint)
        resumed = self.fit("gated", gat_recurrence="gated", epochs=2, resume_from=checkpoint)
        self.assertTrue(np.isfinite(resumed.scores).all())


if __name__ == "__main__":
    unittest.main()
