import math
import torch
import torch.nn as nn


class PatchTSTEncoder(nn.Module):
    """
    Channel-independent PatchTST encoder.
    Reference: yuqinie98/PatchTST

    Input:  (batch, seq_len, n_features)
    Output: (batch, n_features * d_model)  — mean-pooled over patches, channels concat
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_features = n_features
        self.d_model = d_model
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.out_dim = n_features * d_model

        self.patch_embed = nn.Linear(patch_len, d_model)

        # Sinusoidal positional encoding
        pe = torch.zeros(self.n_patches, d_model)
        pos = torch.arange(self.n_patches).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, C) → (B, C * d_model)"""
        B, L, C = x.shape

        # Extract patches: (B, n_patches, C, patch_len)
        patches = x.unfold(1, self.patch_len, self.stride)
        # patches: (B, n_patches, C, patch_len) → (B, C, n_patches, patch_len)
        patches = patches.permute(0, 2, 1, 3)

        _, _, P, PL = patches.shape
        # Flatten channels into batch for channel-independent processing
        patches = patches.reshape(B * C, P, PL)       # (B*C, P, PL)

        tok = self.patch_embed(patches) + self.pe      # (B*C, P, d_model)
        out = self.transformer(tok)                    # (B*C, P, d_model)
        out = self.norm(out).mean(dim=1)               # (B*C, d_model)

        return out.reshape(B, C * self.d_model)        # (B, C*d_model)
