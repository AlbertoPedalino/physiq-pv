"""Paper-aligned STGAN generator and discriminator implemented in PyTorch."""

from __future__ import annotations

import torch
from torch import nn


class GraphConvolution(nn.Module):
    def __init__(self, input_size: int, output_size: int, activation: str = "sigmoid"):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        if activation == "tanh":
            self.activation: nn.Module = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or adjacency.ndim != 3:
            raise ValueError("Graph convolution expects [B,N,F] and [B,N,N].")
        return self.activation(self.linear(torch.bmm(adjacency, x)))


class GCGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        joint_size = input_size + hidden_size
        self.reset = GraphConvolution(joint_size, hidden_size)
        self.update = GraphConvolution(joint_size, hidden_size)
        self.candidate = GraphConvolution(joint_size, hidden_size, activation="tanh")

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        joint = torch.cat((x, hidden), dim=-1)
        reset = self.reset(joint, adjacency)
        update = self.update(joint, adjacency)
        candidate = self.candidate(torch.cat((x, reset * hidden), dim=-1), adjacency)
        return update * hidden + (1.0 - update) * candidate


class GCGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least one.")
        self.hidden_size = int(hidden_size)
        self.layers = nn.ModuleList(
            GCGRUCell(input_size if layer == 0 else hidden_size, hidden_size)
            for layer in range(n_layers)
        )

    def forward(
        self, sequence: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim != 4:
            raise ValueError("GCGRU expects [batch,time,nodes,features].")
        batch_size, _, n_nodes, _ = sequence.shape
        hidden = [
            sequence.new_zeros((batch_size, n_nodes, self.hidden_size))
            for _ in self.layers
        ]
        output = sequence[:, 0]
        for time_index in range(sequence.shape[1]):
            output = sequence[:, time_index]
            for layer_index, layer in enumerate(self.layers):
                hidden[layer_index] = layer(output, hidden[layer_index], adjacency)
                output = hidden[layer_index]
        return output, torch.stack(hidden)


class STGANGenerator(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        hidden_size: int,
        n_layers: int,
        time_feature_size: int = 31,
    ):
        super().__init__()
        if hidden_size % 2:
            raise ValueError("hidden_size must be even for the paper STGAN layout.")
        recurrent_size = hidden_size // 2
        self.recent_encoder = GCGRU(n_features, recurrent_size, n_layers)
        self.trend_encoder = nn.LSTM(
            n_features, hidden_size, num_layers=n_layers, batch_first=True
        )
        self.time_projection = nn.Sequential(
            nn.Linear(time_feature_size, hidden_size), nn.ReLU()
        )
        self.output_graph = GraphConvolution(
            recurrent_size + hidden_size * 2, n_features, activation="tanh"
        )

    def forward(
        self,
        recent: torch.Tensor,
        trend: torch.Tensor,
        adjacency: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        recent_state, _ = self.recent_encoder(recent, adjacency)
        trend_state, _ = self.trend_encoder(trend)
        trend_state = trend_state[:, -1]
        time_state = self.time_projection(time_features)
        n_nodes = recent.shape[2]
        combined = torch.cat(
            (
                recent_state,
                trend_state[:, None, :].expand(-1, n_nodes, -1),
                time_state[:, None, :].expand(-1, n_nodes, -1),
            ),
            dim=-1,
        )
        return self.output_graph(combined, adjacency)


class STGANDiscriminator(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        hidden_size: int,
        n_layers: int,
        subgraph_size: int,
    ):
        super().__init__()
        if hidden_size % 2:
            raise ValueError("hidden_size must be even for the paper STGAN layout.")
        recurrent_size = hidden_size // 2
        self.sequence_encoder = GCGRU(n_features, recurrent_size, n_layers)
        self.sequence_projection = nn.Sequential(
            nn.Linear(subgraph_size * recurrent_size, hidden_size), nn.ReLU()
        )
        self.current_graph = GraphConvolution(
            n_features, hidden_size, activation="sigmoid"
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, sequence: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if sequence.shape[1] < 2:
            raise ValueError("Discriminator sequence requires history plus current data.")
        historical, _ = self.sequence_encoder(sequence[:, :-1], adjacency)
        historical = self.sequence_projection(historical.flatten(start_dim=1))
        current = self.current_graph(sequence[:, -1], adjacency).amax(dim=1)
        return self.output(torch.cat((current, historical), dim=-1))


class STGAN(nn.Module):
    """Generator/discriminator pair with the paper's two anomaly components."""

    def __init__(
        self,
        *,
        n_features: int,
        hidden_size: int,
        n_layers: int,
        subgraph_size: int,
        time_feature_size: int = 31,
    ):
        super().__init__()
        self.generator = STGANGenerator(
            n_features=n_features,
            hidden_size=hidden_size,
            n_layers=n_layers,
            time_feature_size=time_feature_size,
        )
        self.discriminator = STGANDiscriminator(
            n_features=n_features,
            hidden_size=hidden_size,
            n_layers=n_layers,
            subgraph_size=subgraph_size,
        )

    def components(
        self,
        recent: torch.Tensor,
        trend: torch.Tensor,
        adjacency: torch.Tensor,
        time_features: torch.Tensor,
        observed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted = self.generator(recent, trend, adjacency, time_features)
        real_sequence = torch.cat((recent, observed[:, None]), dim=1)
        fake_sequence = torch.cat((recent, predicted[:, None]), dim=1)
        real_score = self.discriminator(real_sequence, adjacency)
        fake_score = self.discriminator(fake_sequence, adjacency)
        return predicted, real_score, fake_score, (predicted - observed).square()

