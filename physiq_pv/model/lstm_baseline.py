"""
Simple LSTM baseline for the PVGIS-only forecasting pipeline (no graph).

Purpose: isolate the ST-GNN architecture as a factor in the low-coverage /
under-dispersion diagnosis. The LSTM is a purely temporal model shared across
locations: every node is encoded independently from its own causal window
(t-seq_len ... t-1); there is NO adjacency, NO edge_index, NO spatial message
passing.

The forward signature mirrors `STGNN.forward(x, edge_index, edge_weight,
ghi_cs)` and returns the same `(pred_ghi, pred_pv)` tuple (pred_ghi is None),
so the existing train/predict/MC-Dropout helpers in pvgis_stgnn_dataset.py work
unchanged. The graph arguments are accepted and ignored.

MC Dropout: the stochasticity comes from explicit nn.Dropout modules (after the
input projection-free LSTM output and before the head), which is what
`enable_dropout_only()` reactivates at inference. The nn.LSTM internal
inter-layer dropout is intentionally NOT used because it is functional (not an
nn.Dropout module) and would stay off in eval mode.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """Per-node temporal LSTM -> dropout -> linear head -> scalar pv(t)."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,  # see module docstring: MC Dropout needs nn.Dropout modules
        )
        # Dropout between LSTM layers' output and the head: the MC-Dropout source.
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,                      # (B, N, seq_len, n_features)
        edge_index: torch.Tensor | None = None,   # ignored (no graph)
        edge_weight: torch.Tensor | None = None,  # ignored (no graph)
        ghi_cs: torch.Tensor | None = None,        # ignored
    ) -> tuple[None, torch.Tensor]:
        """Return (None, pred_pv (B, N)) — same contract as STGNN.forward."""
        B, N, T, F = x.shape
        flat = x.reshape(B * N, T, F)             # nodes are independent samples
        out, _ = self.lstm(flat)                  # (B*N, T, hidden)
        last = out[:, -1, :]                      # causal: last input step t-1
        pred = self.head(self.dropout(last)).squeeze(-1)  # (B*N,)
        return None, pred.reshape(B, N)
