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
        projected = self.projection(values).reshape(batch, nodes, self.heads, self.out_features)
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
