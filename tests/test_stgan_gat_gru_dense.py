"""GAT-GRU speed-ups: attention over a padded neighbor table and one pass for reset and update.

Neither changes the model. The attention over the edges stays in the code as the reference and
every test here compares the fast form with it.
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from physiq_pv.anomaly_detection.stgan import GATGRU, STGANGAT, SparseGATLayer, build_spatial_grid, grid_edge_index
from physiq_pv.anomaly_detection.stgan.graph import dense_attention, neighbor_table


def holed_grid(height=4, width=5, holes=(7, 13)):
    """Grid cells in original order with some cells missing: borders, corners and holes."""
    rows, cols = np.indices((height, width)).reshape(2, -1)
    keep = np.setdiff1d(np.arange(height * width), holes)
    return rows[keep], cols[keep]


def randomized(layer):
    torch.nn.init.normal_(layer.bias)
    return layer


class TableTests(unittest.TestCase):
    def test_every_row_lists_exactly_the_sources_of_its_node(self):
        rows, cols = holed_grid()
        edges = grid_edge_index(rows, cols)
        table, valid = neighbor_table(edges, len(rows))
        self.assertEqual(table.shape[1], int(torch.bincount(edges[1]).max()))
        self.assertLessEqual(table.shape[1], 9)
        full, _ = neighbor_table(grid_edge_index(*np.indices((3, 3)).reshape(2, -1)), 9)
        self.assertEqual(full.shape, (9, 9))  # The center of a complete 3 x 3 has nine sources.
        for node in range(len(rows)):
            expected = sorted(edges[0, edges[1] == node].tolist())
            self.assertEqual(sorted(table[node, valid[node]].tolist()), expected)
            self.assertIn(node, expected)  # The self loop.
            self.assertTrue((table[node, ~valid[node]] == node).all())
        self.assertEqual(int(valid.sum()), edges.shape[1])

    def test_a_node_without_incoming_edges_is_rejected(self):
        with self.assertRaises(ValueError):
            neighbor_table(torch.tensor([[0, 1], [0, 0]]), 2)


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(20)
        rows, cols = holed_grid()
        self.edges = grid_edge_index(rows, cols)
        self.nodes = len(rows)
        self.neighbors = neighbor_table(self.edges, self.nodes)

    def test_one_layer_equals_the_attention_over_the_edges(self):
        layer = randomized(SparseGATLayer(6, 4, 3, concat=False))
        values = torch.randn(2, self.nodes, 6)
        fast = dense_attention((layer,), values, *self.neighbors)[0]
        torch.testing.assert_close(fast, layer(values, self.edges), rtol=1e-5, atol=1e-6)

    def test_layers_sharing_an_input_keep_their_own_parameters(self):
        layers = [randomized(SparseGATLayer(6, 4, 3, concat=False)) for _ in range(3)]
        values = torch.randn(2, self.nodes, 6)
        for fast, layer in zip(dense_attention(layers, values, *self.neighbors), layers):
            torch.testing.assert_close(fast, layer(values, self.edges), rtol=1e-5, atol=1e-6)
        # Changing one layer moves only its own output.
        before = dense_attention(layers, values, *self.neighbors)
        with torch.no_grad():
            layers[1].attention_source.add_(1.0)
        after = dense_attention(layers, values, *self.neighbors)
        torch.testing.assert_close(after[0], before[0])
        torch.testing.assert_close(after[2], before[2])
        self.assertFalse(torch.allclose(after[1], before[1]))

    def test_incompatible_layers_are_rejected(self):
        values = torch.randn(1, self.nodes, 6)
        with self.assertRaises(ValueError):
            dense_attention((SparseGATLayer(6, 4, 3, concat=True),), values, *self.neighbors)
        with self.assertRaises(ValueError):
            dense_attention((SparseGATLayer(6, 4, 3, concat=False), SparseGATLayer(6, 4, 2, concat=False)),
                            values, *self.neighbors)


class EncoderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(20)
        rows, cols = holed_grid()
        self.edges = grid_edge_index(rows, cols)
        self.nodes = len(rows)

    def encoder(self):
        encoder = GATGRU(5, 4, 2, 3, self.edges)
        for cell in encoder.layers:
            for layer in (cell.reset, cell.update, cell.candidate):
                randomized(layer)
        return encoder

    def run_both(self, encoder, sequence, mask):
        results = {}
        for dense in (True, False):
            encoder.dense = dense
            encoder.zero_grad()
            output = encoder(sequence, mask)
            output.square().mean().backward()
            results[dense] = (output.detach().clone(),
                              {name: parameter.grad.clone() for name, parameter in encoder.named_parameters()})
        return results[True], results[False]

    def test_outputs_and_gradients_equal_the_reference_with_missing_nodes(self):
        encoder = self.encoder()
        sequence = torch.randn(2, 4, self.nodes, 5)
        mask = torch.ones(2, self.nodes, 1)
        mask[0, 3], mask[1, 9] = 0, 0
        sequence[0, :, 3] = torch.nan  # Unavailable cells may hold NaN.
        (fast, fast_gradients), (reference, reference_gradients) = self.run_both(encoder, sequence, mask)
        self.assertTrue(encoder.dense is False)  # run_both leaves the last form set.
        torch.testing.assert_close(fast, reference, rtol=1e-5, atol=1e-6)
        self.assertEqual(set(fast_gradients), set(reference_gradients))
        for name in fast_gradients:
            torch.testing.assert_close(fast_gradients[name], reference_gradients[name], rtol=1e-4, atol=1e-6,
                                       msg=name)

    def test_default_is_the_fast_form_and_the_state_dict_is_unchanged(self):
        encoder = self.encoder()
        self.assertIs(encoder.dense, True)
        names = set(encoder.state_dict())
        self.assertFalse(any("neighbor" in name for name in names))
        self.assertIn("edge_index", names)
        # A state dict written before the table existed loads strictly, and gives the same output.
        other = GATGRU(5, 4, 2, 3, self.edges)
        other.load_state_dict(encoder.state_dict(), strict=True)
        sequence, mask = torch.randn(1, 3, self.nodes, 5), torch.ones(1, self.nodes, 1)
        with torch.no_grad():
            torch.testing.assert_close(other(sequence, mask), encoder(sequence, mask))

    def test_table_follows_the_module_to_another_device_and_dtype(self):
        encoder = self.encoder().double()
        self.assertEqual(encoder.neighbor_index.dtype, torch.long)
        self.assertEqual(encoder.neighbor_valid.dtype, torch.bool)
        sequence, mask = torch.randn(1, 2, self.nodes, 5).double(), torch.ones(1, self.nodes, 1).double()
        with torch.no_grad():
            fast = encoder(sequence, mask)
            encoder.dense = False
            reference = encoder(sequence, mask)
        self.assertEqual(fast.dtype, reference.dtype)
        torch.testing.assert_close(fast.float(), reference.float(), rtol=1e-5, atol=1e-6)

    def test_bf16_autocast_stays_close_to_the_reference(self):
        encoder = self.encoder().eval()
        sequence, mask = torch.randn(1, 3, self.nodes, 5), torch.ones(1, self.nodes, 1)
        with torch.no_grad():
            encoder.dense = False
            reference = encoder(sequence, mask)
            encoder.dense = True
            with torch.autocast("cpu", dtype=torch.bfloat16):
                mixed = encoder(sequence, mask).float()
        self.assertTrue(torch.isfinite(mixed).all())
        torch.testing.assert_close(mixed, reference, rtol=0, atol=5e-2)


class GeneratorTests(unittest.TestCase):
    def test_generator_output_equals_the_reference_form(self):
        torch.set_num_threads(1)
        torch.manual_seed(20)
        rows, cols = np.indices((3, 4)).reshape(2, -1)
        grid = build_spatial_grid(45 - rows * .5, 7 + cols * .5, grid_crs="EPSG:4326",
                                  angular_spacing=.5, audit_knn=False)
        network = STGANGAT(n_features=15, hidden_size=4, n_layers=1, cnn_channels=4, cnn_layers=2,
            edge_index=grid_edge_index(grid.row_indices, grid.column_indices), node_indices=grid.node_indices,
            recent_steps=3, gat_hidden_dim=3, gat_heads=2, discriminator_chunk_size=5,
            gat_recurrence="gated").eval()
        nodes = network.generator.n_nodes
        batch = (torch.randn(2, 3, nodes, 15), torch.randn(2, nodes, 4, 15), torch.ones(2, nodes, 1),
                 torch.randn(2, 31))
        with torch.no_grad():
            fast = network.generator(*batch)
            network.generator.recent_encoder.dense = False
            reference = network.generator(*batch)
        torch.testing.assert_close(fast, reference, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
