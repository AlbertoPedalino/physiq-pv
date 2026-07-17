import math

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
        diffusion_term: torch.Tensor | None = None,
        noise_scale: float = 0.0,
        stochastic: bool = False,
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
        out = out + self.res(x)
        if stochastic:
            if diffusion_term is None or diffusion_term.shape != out.shape:
                got = None if diffusion_term is None else tuple(diffusion_term.shape)
                raise ValueError(
                    f"GAT diffusion term must have shape {tuple(out.shape)}, got {got}."
                )
            out = out + noise_scale * diffusion_term * torch.randn_like(out)
        return self.norm(out)


class MonacoDiffusionEncoder(nn.Module):
    """Diffusion encoder aligned with the BiLSTM+GAT drift stages.

    Monaco et al.'s SDE U-Net uses a second, sequential encoder to produce one
    bounded diffusion term for every drift encoder block.  This is the graph-
    temporal counterpart: a dedicated BiLSTM produces the temporal diffusion
    state and dedicated GAT layers evolve it in parallel with the drift GATs.
    Every returned sigmoid gate has shape ``(B, N, gat_dim)``.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int,
        gat_dim: int,
        gat_heads: int,
        gat_layers: int,
        bilstm_pooling: str,
        use_temporal_encoder: bool,
    ) -> None:
        super().__init__()
        self.use_temporal_encoder = use_temporal_encoder
        if use_temporal_encoder:
            self.temporal = BiLSTMEncoder(
                n_features=n_features,
                seq_len=seq_len,
                hidden_dim=d_model,
                n_layers=2,
                dropout=0.0,
                pooling=bilstm_pooling,
                bidirectional=True,
                input_proj_dim=None,
            )
            temporal_out_dim = self.temporal.out_dim
        else:
            self.temporal = None
            temporal_out_dim = seq_len * n_features

        # Monaco's diffusion blocks use ReLU and expose a sigmoid gate.  The raw
        # ReLU state, not the sigmoid probability, feeds the next diffusion stage.
        self.proj = nn.Sequential(
            nn.Linear(temporal_out_dim, gat_dim),
            nn.ReLU(),
        )
        self.gat = nn.ModuleList(
            [GATLayer(gat_dim, gat_dim, n_heads=gat_heads, dropout=0.0)
             for _ in range(gat_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        B, N, L, C = x.shape
        if self.temporal is not None:
            enc = self.temporal(x.reshape(B * N, L, C))
        else:
            enc = x.reshape(B * N, L * C)
        state = self.proj(enc).reshape(B, N, -1)
        terms = [torch.sigmoid(state)]
        for gat_layer in self.gat:
            state = F.relu(gat_layer(state, edge_index, edge_weight))
            terms.append(torch.sigmoid(state))
        return tuple(terms)


class STGNN(nn.Module):
    """
    Spatial-Temporal GNN with Monaco-style stage-aligned SDE uncertainty.

    Architecture per forward pass:
        1. Drift and diffusion BiLSTMs encode the same per-node input window.
        2. The temporal diffusion gate perturbs the drift temporal state.
        3. Each drift GAT is paired with a diffusion GAT and a fresh Brownian kick.
        4. Dual head -> pred_kt (clear-sky index in [0, KT_MAX]) and pred_pv (normalized PV).
           pred_ghi = pred_kt * ghi_cs (physical residual constraint).

    This maps Monaco et al.'s parallel drift/diffusion U-Net encoders to a
    BiLSTM+GAT backbone.  There is one bounded per-node/per-feature diffusion
    term per encoder stage; the head consumes the final stochastic state.
    """

    KT_MAX: float = 1.2  # physical upper bound for clear-sky index (snow albedo edge)

    def __init__(
        self,
        n_nodes: int,
        n_features: int,          # includes QS and optional diagnostic features
        seq_len: int = 120,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        gat_dim: int = 256,
        gat_heads: int = 4,
        gat_layers: int = 3,
        dropout: float = 0.1,
        use_patchtst: bool = True,
        use_gat: bool = True,
        bilstm_pooling: str = "attn",
        n_sde_steps: int = 4,
        sigma_max: float = 0.5,
        use_irradiance_head: bool = True,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.use_patchtst = use_patchtst
        self.use_gat = use_gat
        if n_sde_steps < 1:
            raise ValueError(f"n_sde_steps must be >= 1, got {n_sde_steps}.")
        if not math.isfinite(sigma_max) or sigma_max <= 0.0:
            raise ValueError(f"sigma_max must be finite and > 0, got {sigma_max}.")
        # Irradiance-head ablation: when False, head_ghi is not created and
        # forward returns (None, pred_pv).
        self.use_irradiance_head = use_irradiance_head

        if use_patchtst:
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

        self.n_sde_stages = 1 + len(self.gat)  # temporal stage + spatial stages
        if n_sde_steps != self.n_sde_stages:
            raise ValueError(
                "Monaco alignment requires n_sde_steps == 1 + active GAT layers; "
                f"got n_sde_steps={n_sde_steps}, active_gat_layers={len(self.gat)}."
            )
        self.n_sde_steps = n_sde_steps
        self.sigma_max = float(sigma_max)
        # Official SDE U-Net: T=4 and dt=4/layer_depth, where layer_depth also
        # counts the input level (one more than the stochastic encoder stages).
        self.time_horizon = 4.0
        self.deltat = self.time_horizon / (self.n_sde_stages + 1)
        self.noise_scale = self.sigma_max * (self.deltat ** 0.5)
        self.diffusion_encoder = MonacoDiffusionEncoder(
            n_features=n_features,
            seq_len=seq_len,
            d_model=d_model,
            gat_dim=gat_dim,
            gat_heads=gat_heads,
            gat_layers=len(self.gat),
            bilstm_pooling=bilstm_pooling,
            use_temporal_encoder=use_patchtst,
        )

        # The temporal block mirrors Monaco's residual refinement around the
        # Brownian injection: refine(clean_state) + clean_state + noisy kick.
        self.temporal_refine = nn.Sequential(
            nn.Linear(gat_dim, gat_dim),
            nn.GELU(),
        )
        self.temporal_norm = nn.LayerNorm(gat_dim)

        def _head(out: int = 1):
            return nn.Sequential(
                nn.Linear(gat_dim, gat_dim // 2), nn.GELU(),
                nn.Linear(gat_dim // 2, out),
            )

        self.head_ghi = _head() if use_irradiance_head else None
        self.head_pv = _head()

    def encode(
        self,
        x: torch.Tensor,            # (B, N, seq_len, n_features)
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        *,
        diffusion_terms: tuple[torch.Tensor, ...] | None = None,
        stochastic: bool = False,
    ) -> torch.Tensor:              # (B, N, gat_dim) — final drift/SDE state
        B, N, L, C = x.shape
        if self.use_patchtst:
            enc = self.encoder(x.reshape(B * N, L, C))   # (B*N, enc_dim) — BiLSTM
        else:
            enc = x.reshape(B * N, L * C)                # flatten ablation
        clean = self.proj(enc).reshape(B, N, -1)          # (B, N, gat_dim)

        if stochastic:
            if diffusion_terms is None or len(diffusion_terms) != self.n_sde_stages:
                got = None if diffusion_terms is None else len(diffusion_terms)
                raise ValueError(
                    f"Expected {self.n_sde_stages} Monaco diffusion terms, got {got}."
                )
            noisy = clean + self.noise_scale * diffusion_terms[0] * torch.randn_like(clean)
        else:
            noisy = clean
        h = self.temporal_norm(self.temporal_refine(clean) + noisy)

        for i, gat_layer in enumerate(self.gat, start=1):
            h = gat_layer(
                h,
                edge_index,
                edge_weight,
                diffusion_term=None if diffusion_terms is None else diffusion_terms[i],
                noise_scale=self.noise_scale,
                stochastic=stochastic,
            )
        return h

    def diffusion(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Return Monaco's bounded diffusion gate at every encoder stage."""
        return self.diffusion_encoder(x, edge_index, edge_weight)

    def diffusion_parameters(self):
        """Parameters optimized only by the ID/pseudo-OOD diffusion objective."""
        return self.diffusion_encoder.parameters()

    def forward(
        self,
        x: torch.Tensor,                      # (B, N, seq_len, n_features)
        edge_index: torch.Tensor,             # (2, E)
        edge_weight: torch.Tensor,            # (E,)
        ghi_cs: torch.Tensor | None = None,   # (B, N) clear-sky GHI in kW/m^2
        stochastic: bool = True,
        return_diffusion: bool = False,
    ):
        """
        Returns (pred_ghi, pred_pv) by default. With return_diffusion the tuple
        of per-stage raw sigmoid gates is appended.

        stochastic=True samples one Brownian path (training / SDE inference);
        stochastic=False evaluates the drift encoder only.

        When ghi_cs is provided, pred_ghi = pred_kt * ghi_cs with
        pred_kt = sigmoid(head_ghi) * KT_MAX (hard physical bound, ~0 at night).
        When use_irradiance_head=False, pred_ghi is None.

        pred_pv is the point prediction (softplus, >= 0).
        """
        diffusion_terms = (
            self.diffusion(x, edge_index, edge_weight)
            if stochastic or return_diffusion
            else None
        )
        h = self.encode(
            x,
            edge_index,
            edge_weight,
            diffusion_terms=diffusion_terms,
            stochastic=stochastic,
        )

        if self.head_ghi is None:
            pred_ghi = None
        else:
            pred_kt = torch.sigmoid(self.head_ghi(h).squeeze(-1)) * self.KT_MAX  # (B, N)
            pred_ghi = pred_kt * ghi_cs if ghi_cs is not None else pred_kt

        pv_out = self.head_pv(h)                 # (B, N, 1)
        pred_pv = F.softplus(pv_out[..., 0])     # (B, N) point prediction, >= 0

        out = [pred_ghi, pred_pv]
        if return_diffusion:
            out.append(diffusion_terms)
        return tuple(out)
