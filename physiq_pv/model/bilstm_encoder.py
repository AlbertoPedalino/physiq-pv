import torch
import torch.nn as nn


class BiLSTMEncoder(nn.Module):
    """
    Per-node temporal encoder using a bidirectional LSTM with optional
    Linear input projection and attention pooling over timesteps.

    Input:  (batch, seq_len, n_features)
    Output: (batch, out_dim)  where out_dim = hidden_dim * 2 (bidirectional)
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
        pooling: str = "attn",
        bidirectional: bool = True,
        input_proj_dim: int | None = 64,
    ):
        super().__init__()
        if pooling not in ("last", "attn"):
            raise ValueError(f"pooling must be 'last' or 'attn', got {pooling!r}")
        self.pooling = pooling
        self.bidirectional = bidirectional

        if input_proj_dim is not None:
            self.input_proj = nn.Sequential(
                nn.Linear(n_features, input_proj_dim),
                nn.LayerNorm(input_proj_dim),
                nn.GELU(),
            )
            lstm_in = input_proj_dim
        else:
            self.input_proj = nn.Identity()
            lstm_in = n_features

        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.out_dim = hidden_dim * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(self.out_dim)
        if pooling == "attn":
            self.pool_attn = nn.Linear(self.out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) -> (B, out_dim)"""
        x = self.input_proj(x)
        h_seq, _ = self.lstm(x)
        if self.pooling == "attn":
            scores = self.pool_attn(h_seq)
            weights = torch.softmax(scores, dim=1)
            h = self.norm((h_seq * weights).sum(dim=1))
        else:
            if self.bidirectional:
                H = h_seq.size(-1) // 2
                h_fwd_last = h_seq[:, -1, :H]
                h_bwd_last = h_seq[:,  0, H:]
                h = self.norm(torch.cat([h_fwd_last, h_bwd_last], dim=-1))
            else:
                h = self.norm(h_seq[:, -1, :])
        return h
