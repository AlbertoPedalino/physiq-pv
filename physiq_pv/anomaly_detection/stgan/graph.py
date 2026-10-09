"""Fixed directed 8-neighborhood with self loops, in original location order."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def grid_edge_index(rows, columns):
    """O(N+E) construction; holes stay holes, with no wrapping or dense adjacency."""
    rows, columns = np.asarray(rows), np.asarray(columns)
    if (rows.ndim != 1 or columns.shape != rows.shape or not len(rows)
            or rows.dtype.kind not in "iu" or columns.dtype.kind not in "iu"):
        raise ValueError("Require nonempty matching integer grid coordinates.")
    lookup = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, columns))}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate grid cells.")
    source, target = [], []
    for (r, c), destination in lookup.items():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                neighbor = lookup.get((r + dr, c + dc))
                if neighbor is not None:
                    source.append(neighbor)
                    target.append(destination)
    return torch.tensor([source, target], dtype=torch.long)


class SparseGATLayer(nn.Module):
    """Multi-head additive attention and destination softmax over E edges only."""
    def __init__(self, in_features, out_features, heads, *, concat):
        super().__init__()
        self.heads, self.out_features, self.concat = heads, out_features, concat
        self.projection = nn.Linear(in_features, heads * out_features, bias=False)
        self.attention_source = nn.Parameter(torch.empty(heads, out_features))
        self.attention_target = nn.Parameter(torch.empty(heads, out_features))
        self.bias = nn.Parameter(torch.zeros(heads * out_features if concat else out_features))
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_target)

    def forward(self, values, edge_index):
        batch, nodes, _ = values.shape
        source, target = edge_index
        # Keep attention softmax and sparse sums in FP32 under BF16 autocast.
        projected = self.projection(values).float().reshape(batch, nodes, self.heads, self.out_features)
        source_score = (projected * self.attention_source).sum(-1)
        target_score = (projected * self.attention_target).sum(-1)
        logits = F.leaky_relu(source_score[:, source] + target_score[:, target], .2)
        index = target[None, :, None].expand(batch, -1, self.heads)
        maxima = values.new_full((batch, nodes, self.heads), -torch.inf)
        maxima.scatter_reduce_(1, index, logits.detach(), reduce="amax", include_self=True)
        weights = (logits - maxima[:, target]).exp()
        denominator = values.new_zeros((batch, nodes, self.heads))
        denominator.scatter_add_(1, index, weights)
        weights = weights / denominator[:, target]
        messages = projected[:, source] * weights[..., None]
        output = values.new_zeros((batch, nodes, self.heads, self.out_features))
        output.scatter_add_(1, index[..., None].expand_as(messages), messages)
        output = output.flatten(2) if self.concat else output.mean(2)
        return output + self.bias


class TwoLayerGAT(nn.Module):
    def __init__(self, n_features, channels, hidden_dim, heads, edge_index, layers=2):
        super().__init__()
        if type(layers) is not int or layers != 2:
            raise ValueError("This experiment requires exactly 2 GAT layers.")
        if any(type(x) is not int or x < 1 for x in (n_features, channels, hidden_dim, heads)):
            raise ValueError("GAT dimensions and heads must be positive integers.")
        self.register_buffer("edge_index", torch.as_tensor(edge_index, dtype=torch.long).clone())
        self.layers = nn.ModuleList([
            SparseGATLayer(n_features, hidden_dim, heads, concat=True),
            SparseGATLayer(hidden_dim * heads, channels, heads, concat=False),
        ])

    def forward(self, values):
        values = F.elu(self.layers[0](values, self.edge_index))
        return self.layers[1](values, self.edge_index)


def neighbor_table(edge_index, nodes):
    """Sources of every node as a padded table: [nodes, max in-degree] indices and their validity.

    Padding repeats the node itself and is masked out. On a grid graph the table has at most
    nine columns, and attention over it is a dense gather instead of scatters over the edges.
    """
    source, target = torch.as_tensor(edge_index, dtype=torch.long).cpu()
    order = torch.argsort(target, stable=True)
    source, target = source[order], target[order]
    degree = torch.bincount(target, minlength=nodes)
    if not len(target) or int(degree.min()) < 1:
        raise ValueError("Every node needs at least one incoming edge (its self loop).")
    position = torch.arange(len(target)) - torch.repeat_interleave(torch.cumsum(degree, 0) - degree, degree)
    table = torch.arange(nodes)[:, None].repeat(1, int(degree.max()))
    valid = torch.zeros_like(table, dtype=torch.bool)
    table[target, position] = source
    valid[target, position] = True
    return table, valid


def dense_attention(layers, values, table, valid):
    """Outputs of attention layers that share one input, in a single pass over the neighbor table.

    Each layer keeps its own projection, attention vectors and bias; heads never mix, so the
    result is the one of calling every layer on the edges, up to the order of the sums.
    """
    heads, features = layers[0].heads, layers[0].out_features
    if any(layer.concat or (layer.heads, layer.out_features) != (heads, features) for layer in layers):
        raise ValueError("Joint attention needs averaging layers with the same heads and output size.")
    batch, nodes, _ = values.shape
    # Keep attention softmax and sums in FP32 under BF16 autocast, as the edge form does.
    projected = F.linear(values, torch.cat([layer.projection.weight for layer in layers])).float()
    projected = projected.reshape(batch, nodes, len(layers) * heads, features)
    source_score = (projected * torch.cat([layer.attention_source for layer in layers])).sum(-1)
    target_score = (projected * torch.cat([layer.attention_target for layer in layers])).sum(-1)
    logits = F.leaky_relu(source_score[:, table] + target_score[:, :, None], .2)
    weights = torch.softmax(logits.masked_fill(~valid[None, :, :, None], -torch.inf), dim=2)
    output = (projected[:, table] * weights[..., None]).sum(2)
    output = output.reshape(batch, nodes, len(layers), heads, features).mean(3)
    return [output[:, :, index] + layer.bias for index, layer in enumerate(layers)]


class GATGRUCell(nn.Module):
    """Paper GCGRU gates with graph attention as the spatial operator."""

    def __init__(self, n_features, channels, heads):
        super().__init__()
        # Each layer receives the validity mask as an additional input feature.
        joint_features = n_features + 1 + channels
        self.reset = SparseGATLayer(joint_features, channels, heads, concat=False)
        self.update = SparseGATLayer(joint_features, channels, heads, concat=False)
        self.candidate = SparseGATLayer(joint_features, channels, heads, concat=False)

    def forward(self, values, hidden, mask, edge_index, neighbors=None):
        valid = mask.bool()
        values = torch.cat((torch.where(valid, values, 0.0), mask.to(values.dtype)), dim=-1)
        hidden = torch.where(valid, hidden, 0.0)
        joint = torch.cat((values, hidden), dim=-1)
        if neighbors is None:  # Reference form, over the edges; any graph.
            reset = torch.sigmoid(self.reset(joint, edge_index))
            update = torch.sigmoid(self.update(joint, edge_index))
            candidate = torch.tanh(self.candidate(torch.cat((values, reset * hidden), dim=-1), edge_index))
        else:
            # Reset and update read the same input: one pass computes both.
            reset, update = (torch.sigmoid(gate) for gate in
                             dense_attention((self.reset, self.update), joint, *neighbors))
            candidate = torch.tanh(dense_attention(
                (self.candidate,), torch.cat((values, reset * hidden), dim=-1), *neighbors)[0])
        next_hidden = update * hidden + (1.0 - update) * candidate
        # Missing nodes must not acquire state that can reach valid neighbors
        # in a later layer or time step.
        return torch.where(valid, next_hidden, 0.0)


class GATGRU(nn.Module):
    """Encode [batch,time,nodes,features] on the fixed graph; reset state for each window.

    Every gate of every step attends over the neighbors of the input and of the
    state, as the graph convolution does in the paper and the spatial convolution
    in ConvGRU. The reset gate and the candidate attend in sequence, so a step
    moves the state two hops; the first step, from a zero state, reaches one.

    `dense` selects how the same attention is computed: over the padded neighbor
    table (default) or over the edges, the reference the tests compare against.
    """

    def __init__(self, n_features, channels, layers, heads, edge_index):
        super().__init__()
        if any(type(x) is not int or x < 1 for x in (n_features, channels, layers, heads)):
            raise ValueError("GATGRU dimensions, layers and heads must be positive integers.")
        self.n_features, self.channels = n_features, channels
        edges = torch.as_tensor(edge_index, dtype=torch.long).clone()
        self.register_buffer("edge_index", edges)
        # Derived from edge_index: kept out of the state dict, so checkpoints do not change.
        table, valid = neighbor_table(edges, int(edges.max()) + 1)
        self.register_buffer("neighbor_index", table, persistent=False)
        self.register_buffer("neighbor_valid", valid, persistent=False)
        self.dense = True
        self.layers = nn.ModuleList(
            GATGRUCell(n_features if index == 0 else channels, channels, heads)
            for index in range(layers)
        )

    def forward(self, sequence, mask):
        if sequence.ndim != 4 or sequence.shape[1] < 1:
            raise ValueError("GATGRU requires nonempty [batch,time,nodes,features] input.")
        batch, _, nodes, features = sequence.shape
        if features != self.n_features or mask.shape != (batch, nodes, 1):
            raise ValueError("GATGRU input features or validity mask shape do not match.")
        neighbors = (self.neighbor_index, self.neighbor_valid) if self.dense else None
        hidden = [sequence.new_zeros((batch, nodes, self.channels)) for _ in self.layers]
        for time_index in range(sequence.shape[1]):
            output = sequence[:, time_index]
            for index, layer in enumerate(self.layers):
                hidden[index] = layer(output, hidden[index], mask, self.edge_index, neighbors)
                output = hidden[index]
        return output
