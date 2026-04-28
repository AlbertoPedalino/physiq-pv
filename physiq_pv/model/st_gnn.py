import torch
import torch.nn as nn
import torch.nn.functional as F

from physiq_pv.model.patchtst_encoder import PatchTSTEncoder


class GATLayer(nn.Module):
    """
    Batched multi-head Graph Attention layer.
    Processes (B, N, d_in) → (B, N, d_out) with shared edge topology.
    QS is already baked into node features before this layer.
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.out_dim = out_dim

        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * self.head_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        self.norm = nn.LayerNorm(out_dim)
        self.res = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,            # (B, N, in_dim)
        edge_index: torch.Tensor,   # (2, E)
        edge_weight: torch.Tensor,  # (E,)
    ) -> torch.Tensor:              # (B, N, out_dim)
        B, N, _ = x.shape
        H, D = self.n_heads, self.head_dim
        src, dst = edge_index[0], edge_index[1]  # (E,)

        h = self.lin(x).reshape(B, N, H, D)  # (B, N, H, D)

        # Attention coefficients per head
        h_src = h[:, src, :, :]   # (B, E, H, D)
        h_dst = h[:, dst, :, :]   # (B, E, H, D)
        h_cat = torch.cat([h_src, h_dst], dim=-1)  # (B, E, H, 2D)
        e = self.leaky(self.attn(h_cat)).squeeze(-1)  # (B, E, H)

        # Scale by log(1 + edge_weight)
        e = e * edge_weight.log1p().unsqueeze(0).unsqueeze(-1)  # broadcast over B, H

        # Scatter softmax per destination node: build (B, H, N, N) attention matrix
        attn_mat = torch.full((B, H, N, N), float("-inf"), device=x.device)
        attn_mat[:, :, dst, src] = e.permute(0, 2, 1)  # (B, H, E)
        attn_mat = F.softmax(attn_mat, dim=-1)           # (B, H, N, N)
        attn_mat = self.dropout(attn_mat)

        # Aggregate: (B, H, N, N) @ (B, H, N, D) → (B, H, N, D)
        h_perm = h.permute(0, 2, 1, 3)  # (B, H, N, D)
        out = torch.matmul(attn_mat, h_perm)  # (B, H, N, D)
        out = F.elu(out).permute(0, 2, 1, 3).reshape(B, N, self.out_dim)  # (B, N, out_dim)

        return self.norm(out + self.res(x))


class STGNN(nn.Module):
    """
    Spatial-Temporal GNN for PV forecasting.

    Architecture per forward pass:
        1. PatchTST encoder (channel-independent) → per-node temporal embedding
        2. Linear projection → GAT input dim
        3. K × GATLayer (geographic graph, edge_weight = 1/dist_km)
        4. Dual head → pred_ghi (W/m²), pred_pv (kWh)

    QS is included as the last input feature and propagates through GAT.
    """

    def __init__(
        self,
        n_nodes: int,
        n_features: int,          # must include QS as last feature
        seq_len: int = 120,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        gat_dim: int = 256,
        gat_heads: int = 4,
        gat_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_nodes = n_nodes

        self.encoder = PatchTSTEncoder(
            n_features=n_features,
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            dropout=dropout,
        )
        self.proj = nn.Sequential(
            nn.Linear(self.encoder.out_dim, gat_dim),
            nn.GELU(),
            nn.LayerNorm(gat_dim),
        )
        self.gat = nn.ModuleList(
            [GATLayer(gat_dim, gat_dim, n_heads=gat_heads, dropout=dropout) for _ in range(gat_layers)]
        )

        def _head(out: int = 1):
            return nn.Sequential(
                nn.Linear(gat_dim, gat_dim // 2), nn.GELU(),
                nn.Linear(gat_dim // 2, out),
            )

        self.head_ghi = _head()
        self.head_pv = _head()

    def forward(
        self,
        x: torch.Tensor,            # (B, N, seq_len, n_features)
        edge_index: torch.Tensor,   # (2, E)
        edge_weight: torch.Tensor,  # (E,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns pred_ghi (B, N), pred_pv (B, N)."""
        B, N, L, C = x.shape

        # Encode per-node time series (channel-independent)
        enc = self.encoder(x.reshape(B * N, L, C))  # (B*N, enc_dim)
        enc = self.proj(enc).reshape(B, N, -1)       # (B, N, gat_dim)

        # Graph attention
        h = enc
        for gat_layer in self.gat:
            h = gat_layer(h, edge_index, edge_weight)  # (B, N, gat_dim)

        pred_ghi = self.head_ghi(h).squeeze(-1)  # (B, N)
        pred_pv = self.head_pv(h).squeeze(-1)    # (B, N)
        return pred_ghi, pred_pv
