"""MTGFlow base architecture.

This module implements Sections 4.3--4.7 of Zhou et al., "Label-Free
Multivariate Time Series Anomaly Detection".  It is deliberately independent
from the authors' repository: that checkout is an audit reference, not a
runtime dependency.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


class DynamicGraphAttention(nn.Module):
    """Equation 6--7: row-normalised dynamic adjacency matrix."""

    def __init__(self, window_size: int, input_size: int = 1, dropout: float = 0.2):
        super().__init__()
        projection_size = window_size * input_size
        self.query = nn.Linear(projection_size, projection_size, bias=False)
        self.key = nn.Linear(projection_size, projection_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(projection_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [batch, entity, time, feature], got {tuple(x.shape)}")
        flattened = x.flatten(start_dim=2)
        scores = torch.matmul(self.query(flattened), self.key(flattened).transpose(1, 2))
        adjacency = torch.softmax(scores / self.scale, dim=-1)
        return self.dropout(adjacency)


class SpatioTemporalConditioner(nn.Module):
    """Equation 8: ReLU(A H W1 + H_previous W2) W3."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.graph_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.history_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 4 or adjacency.ndim != 3:
            raise ValueError("Invalid hidden or adjacency rank for graph conditioning.")
        # A_ij weights information flowing from entity j into entity i.
        neighbours = torch.einsum("bij,bjth->bith", adjacency, hidden)
        condition = self.graph_projection(neighbours)
        history = torch.zeros_like(condition)
        history[:, :, 1:] = self.history_projection(hidden[:, :, :-1])
        return self.output_projection(F.relu(condition + history))


class ConditionalMAFBlock(nn.Module):
    """One conditional one-dimensional MAF block shared by all entities."""

    def __init__(
        self,
        input_size: int,
        condition_size: int,
        hidden_size: int,
        n_hidden: int,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(condition_size, hidden_size)]
        for _ in range(n_hidden):
            layers.extend((nn.Tanh(), nn.Linear(hidden_size, hidden_size)))
        layers.extend((nn.Tanh(), nn.Linear(hidden_size, 2 * input_size)))
        self.parameter_net = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale = self.parameter_net(condition).chunk(2, dim=-1)
        z = (x - shift) * torch.exp(-log_scale)
        return z, -log_scale


class EntityAwareMAF(nn.Module):
    """Pointwise conditional MAF with entity-aware Gaussian targets.

    The paper does not specify the internal MAF tensorisation. This follows the
    authors' base implementation detail ``input_size=1``: parameters are shared
    over all entity/time points and conditioned by the graph-LSTM state.
    """

    def __init__(
        self,
        *,
        n_blocks: int,
        n_entities: int,
        input_size: int,
        condition_size: int,
        hidden_size: int,
        n_hidden: int,
    ):
        super().__init__()
        if n_blocks < 1:
            raise ValueError("n_blocks must be at least one.")
        self.blocks = nn.ModuleList(
            ConditionalMAFBlock(input_size, condition_size, hidden_size, n_hidden)
            for _ in range(n_blocks)
        )
        # Equation 9: one scalar draw per entity, repeated along the window.
        self.register_buffer("entity_means", torch.randn(n_entities, input_size))

    def point_log_prob(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        entity_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = x
        log_abs_det = torch.zeros_like(x)
        for block in self.blocks:
            z, block_log_abs_det = block(z, condition)
            log_abs_det = log_abs_det + block_log_abs_det
        mean = self.entity_means.index_select(0, entity_index)
        base_log_prob = -0.5 * (z - mean).square() - _LOG_SQRT_2PI
        return (base_log_prob + log_abs_det).sum(dim=-1), z


class MTGFlow(nn.Module):
    """MTGFlow base with exact window- and entity-level likelihood aggregation."""

    def __init__(
        self,
        *,
        n_blocks: int,
        input_size: int,
        hidden_size: int,
        n_hidden: int,
        window_size: int,
        n_entities: int,
        attention_dropout: float = 0.2,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.n_entities = int(n_entities)
        self.input_size = int(input_size)
        self.attention = DynamicGraphAttention(window_size, input_size, attention_dropout)
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=1)
        self.graph_condition = SpatioTemporalConditioner(hidden_size)
        self.flow = EntityAwareMAF(
            n_blocks=n_blocks,
            n_entities=n_entities,
            input_size=input_size,
            condition_size=hidden_size,
            hidden_size=hidden_size,
            n_hidden=n_hidden,
        )

    def likelihood_components(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (self.n_entities, self.window_size, self.input_size)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected [batch, {expected}], got {tuple(x.shape)}")
        batch_size = x.shape[0]
        adjacency = self.attention(x)
        flat_x = x.reshape(batch_size * self.n_entities, self.window_size, self.input_size)
        hidden, _ = self.rnn(flat_x)
        hidden = hidden.reshape(batch_size, self.n_entities, self.window_size, -1)
        condition = self.graph_condition(hidden, adjacency)

        point_x = x.reshape(-1, self.input_size)
        point_condition = condition.reshape(point_x.shape[0], -1)
        entity_index = (
            torch.arange(self.n_entities, device=x.device)
            .view(1, self.n_entities, 1)
            .expand(batch_size, self.n_entities, self.window_size)
            .reshape(-1)
        )
        point_log_prob, z = self.flow.point_log_prob(
            point_x, point_condition, entity_index
        )
        point_log_prob = point_log_prob.reshape(
            batch_size, self.n_entities, self.window_size
        )
        entity_log_prob = point_log_prob.sum(dim=-1)
        return entity_log_prob, z.reshape_as(x), adjacency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Equation 4.6 objective: mean entity window log likelihood."""
        entity_log_prob, _, _ = self.likelihood_components(x)
        return entity_log_prob.mean()

    def test(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-window log likelihood from Equation 12."""
        entity_log_prob, _, _ = self.likelihood_components(x)
        return entity_log_prob.mean(dim=1)

    def locate(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-window, per-entity log likelihoods for Equations 14--15."""
        entity_log_prob, _, _ = self.likelihood_components(x)
        return entity_log_prob
