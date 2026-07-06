import torch
import torch.nn as nn
import torch.nn.functional as F

from physiq_pv.model.bilstm_encoder import BiLSTMEncoder


class GATLayer(nn.Module):
    """
    Batched multi-head Graph Attention layer.
    Processes (B, N, d_in) -> (B, N, d_out) with shared edge topology.
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
        E = src.numel()

        h = self.lin(x).reshape(B, N, H, D)  # (B, N, H, D)

        # Attention coefficients per head
        h_src = h[:, src, :, :]   # (B, E, H, D)
        h_dst = h[:, dst, :, :]   # (B, E, H, D)
        h_cat = torch.cat([h_src, h_dst], dim=-1)  # (B, E, H, 2D)
        e = self.leaky(self.attn(h_cat)).squeeze(-1)  # (B, E, H)

        # Scale by log(1 + edge_weight)
        e = e * edge_weight.log1p().unsqueeze(0).unsqueeze(-1)

        # Sparse edge-wise softmax over incoming edges per destination node.
        # This avoids building a dense (B, H, N, N) attention matrix.
        e_bhe = e.permute(0, 2, 1)  # (B, H, E)
        dst_idx = dst.view(1, 1, E).expand(B, H, E)  # (B, H, E)

        max_per_dst = torch.full((B, H, N), float("-inf"), device=x.device, dtype=e_bhe.dtype)
        max_per_dst.scatter_reduce_(2, dst_idx, e_bhe, reduce="amax", include_self=True)

        e_shift = e_bhe - max_per_dst.gather(2, dst_idx)
        exp_e = torch.exp(e_shift)

        sum_per_dst = torch.zeros((B, H, N), device=x.device, dtype=e_bhe.dtype)
        sum_per_dst.scatter_add_(2, dst_idx, exp_e)
        alpha = exp_e / (sum_per_dst.gather(2, dst_idx) + 1e-12)  # (B, H, E)
        alpha = self.dropout(alpha)

        # Aggregate edge messages directly into destination nodes.
        msg = h_src.permute(0, 2, 1, 3) * alpha.unsqueeze(-1)  # (B, H, E, D)
        out = torch.zeros((B, H, N, D), device=x.device, dtype=msg.dtype)
        out.scatter_add_(2, dst_idx.unsqueeze(-1).expand(B, H, E, D), msg)

        out = F.elu(out).permute(0, 2, 1, 3).reshape(B, N, self.out_dim)  # (B, N, out_dim)
        return self.norm(out + self.res(x))


class STGNN(nn.Module):
    """
    Spatial-Temporal GNN for PV forecasting.

    Architecture per forward pass:
        1. BiLSTM encoder (per-node, channel-mixed, attn pooling) -> temporal embedding
        2. Linear projection -> GAT input dim
        3. K x GATLayer (geographic graph, edge_weight = 1/dist_km)
        4. Dual head -> pred_kt (clear-sky index in [0, kt_max]) and pred_pv (normalized PV).
           pred_ghi = pred_kt * ghi_cs (physical residual constraint).

    QS and m1_past are included in node features and propagate through GAT.
    """

    KT_MAX: float = 1.2  # physical upper bound for clear-sky index (snow albedo edge)

    def __init__(
        self,
        n_nodes: int,
        n_features: int,          # includes QS and optional diagnostic features
        seq_len: int = 120,
        d_model: int = 128,
        gat_dim: int = 256,
        gat_heads: int = 4,
        gat_layers: int = 2,
        dropout: float = 0.1,
        use_bilstm: bool = True,
        use_gat: bool = True,
        bilstm_pooling: str = "attn",
        enhanced_dropout: bool = False,
        use_irradiance_head: bool = True,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.use_bilstm = use_bilstm
        self.use_gat = use_gat
        # Irradiance-head ablation: when False, head_ghi is not created and
        # forward returns (None, pred_pv).
        self.use_irradiance_head = use_irradiance_head
        # Enhanced MC Dropout ablation: explicit nn.Dropout modules (findable by
        # enable_dropout_only) on the temporal embedding and the projected hidden
        # representation. nn.Identity when disabled, so the default STGNN forward
        # is unchanged (no parameters, no behaviour change).
        self.enhanced_dropout = enhanced_dropout
        self.temporal_dropout = nn.Dropout(dropout) if enhanced_dropout else nn.Identity()
        self.representation_dropout = nn.Dropout(dropout) if enhanced_dropout else nn.Identity()

        if use_bilstm:
            self.encoder = BiLSTMEncoder(
                n_features=n_features,
                seq_len=seq_len,
                hidden_dim=d_model,
                n_layers=2,
                dropout=max(dropout, 0.2),
                pooling=bilstm_pooling,
                bidirectional=True,
                input_proj_dim=None,
            )
            enc_out_dim = self.encoder.out_dim
        else:
            # Ablation: bypass temporal encoder. Flatten window through Linear.
            self.encoder = None
            enc_out_dim = seq_len * n_features

        self.proj = nn.Sequential(
            nn.Linear(enc_out_dim, gat_dim),
            nn.GELU(),
            nn.LayerNorm(gat_dim),
        )
        if use_gat:
            self.gat = nn.ModuleList(
                [GATLayer(gat_dim, gat_dim, n_heads=gat_heads, dropout=dropout) for _ in range(gat_layers)]
            )
        else:
            # Ablation: no spatial message passing. Per-node predictions only.
            self.gat = nn.ModuleList()

        def _head(out: int = 1, head_dropout: float | None = None):
            layers: list[nn.Module] = [nn.Linear(gat_dim, gat_dim // 2), nn.GELU()]
            if head_dropout is not None:
                # Enhanced MC Dropout: nn.Dropout before the final Linear so the
                # head itself contributes to the MC predictive distribution.
                layers.append(nn.Dropout(head_dropout))
            layers.append(nn.Linear(gat_dim // 2, out))
            return nn.Sequential(*layers)

        # head_ghi is created first (when enabled) so the parameter-init RNG
        # stream of the default configuration is unchanged.
        self.head_ghi = _head() if use_irradiance_head else None
        self.head_pv = _head(head_dropout=dropout if enhanced_dropout else None)

    def forward(
        self,
        x: torch.Tensor,                      # (B, N, seq_len, n_features)
        edge_index: torch.Tensor,             # (2, E)
        edge_weight: torch.Tensor,            # (E,)
        ghi_cs: torch.Tensor | None = None,   # (B, N) clear-sky GHI in kW/m^2
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        Returns pred_ghi (B, N), pred_pv (B, N).

        When ghi_cs is provided, pred_ghi = pred_kt * ghi_cs with
        pred_kt = sigmoid(head_ghi) * KT_MAX. This enforces a hard physical bound:
        the prediction can never exceed KT_MAX * clear_sky and is forced to ~0 at
        night (ghi_cs ~ 0).

        When ghi_cs is None,
        pred_ghi falls back to pred_kt directly (diagnostic only; do not consume).

        When use_irradiance_head=False, pred_ghi is None (production-only model).
        """
        B, N, L, C = x.shape

        if self.use_bilstm:
            enc = self.encoder(x.reshape(B * N, L, C))   # (B*N, enc_dim) - BiLSTM
        else:
            enc = x.reshape(B * N, L * C)                # flatten ablation
        enc = self.temporal_dropout(enc)                 # Identity unless enhanced_dropout
        enc = self.proj(enc).reshape(B, N, -1)           # (B, N, gat_dim)
        enc = self.representation_dropout(enc)           # Identity unless enhanced_dropout

        h = enc
        for gat_layer in self.gat:
            h = gat_layer(h, edge_index, edge_weight)

        if self.head_ghi is None:
            pred_ghi = None
        else:
            pred_kt = torch.sigmoid(self.head_ghi(h).squeeze(-1)) * self.KT_MAX  # (B, N)
            pred_ghi = pred_kt * ghi_cs if ghi_cs is not None else pred_kt
        pred_pv = F.softplus(self.head_pv(h).squeeze(-1))
        return pred_ghi, pred_pv
