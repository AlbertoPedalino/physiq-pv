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


class SDEBlock(nn.Module):
    """
    Neural SDE block (Kong et al. 2020, "SDE-Net").

    Evolves the node hidden state x0 over [0, 1] by Euler-Maruyama:
        x_{k+1} = x_k + f(x_k, t)·dt + g(x0)·sqrt(dt)·Z_k,   Z_k ~ N(0, I)

    * drift  f(x, t): governs the deterministic dynamics (the prediction);
    * diffusion g(x0): scales the Brownian motion and encodes epistemic
      uncertainty — trained low in-distribution, high out-of-distribution.

    g depends only on the initial state x0 (per the paper: simpler, stable) and is
    bounded to [0, sigma_max] via sigmoid, which prevents an explosive solution.
    Tanh activations keep f and g Lipschitz (existence/uniqueness, Theorem 1).
    """

    def __init__(self, dim: int, n_steps: int = 4, sigma_max: float = 0.5):
        super().__init__()
        self.n_steps = n_steps
        self.sigma_max = sigma_max
        self.drift = nn.Sequential(
            nn.Linear(dim + 1, dim), nn.Tanh(),   # +1: time t appended to the state
            nn.Linear(dim, dim), nn.Tanh(),
        )
        self.diffusion_net = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )

    def diffusion(self, x0: torch.Tensor) -> torch.Tensor:
        """g(x0) in [0, sigma_max], shape (B, N, 1). One scalar per node."""
        return torch.sigmoid(self.diffusion_net(x0)) * self.sigma_max

    def forward(
        self, x0: torch.Tensor, stochastic: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_T, g) where x_T is the terminal state and g is g(x0) (B, N)."""
        dt = 1.0 / self.n_steps
        g = self.diffusion(x0)                 # (B, N, 1)
        x = x0
        for k in range(self.n_steps):
            t = x.new_full((*x.shape[:-1], 1), k * dt)
            x = x + self.drift(torch.cat([x, t], dim=-1)) * dt
            if stochastic:
                x = x + g * (dt ** 0.5) * torch.randn_like(x)
        return x, g.squeeze(-1)


class STGNN(nn.Module):
    """
    Spatial-Temporal GNN for PV forecasting with a neural-SDE uncertainty block.

    Architecture per forward pass:
        1. BiLSTM encoder (per-node) -> temporal embedding
        2. Linear projection -> GAT input dim
        3. K x GATLayer (geographic graph, edge_weight = 1/dist_km)   => x0
        4. SDEBlock: Euler-Maruyama x0 -> x_T (Brownian motion = uncertainty source)
        5. Dual head -> pred_kt (clear-sky index in [0, KT_MAX]) and pred_pv (normalized PV).
           pred_ghi = pred_kt * ghi_cs (physical residual constraint).

    Uncertainty comes from the SDE diffusion term (g·dW), not from dropout. Steps
    1-3 are the "downsampling" h1 in SDE-Net terms; the heads are h2.
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
        gat_layers: int = 2,
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

        self.sde = SDEBlock(gat_dim, n_steps=n_sde_steps, sigma_max=sigma_max)

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
    ) -> torch.Tensor:              # (B, N, gat_dim) — the SDE initial state x0
        B, N, L, C = x.shape
        if self.use_patchtst:
            enc = self.encoder(x.reshape(B * N, L, C))   # (B*N, enc_dim) — BiLSTM
        else:
            enc = x.reshape(B * N, L * C)                # flatten ablation
        h = self.proj(enc).reshape(B, N, -1)             # (B, N, gat_dim)
        for gat_layer in self.gat:
            h = gat_layer(h, edge_index, edge_weight)
        return h

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
        Returns (pred_ghi, pred_pv), or (pred_ghi, pred_pv, g) when return_diffusion.

        stochastic=True samples one Brownian path (training / MC inference);
        stochastic=False integrates the drift only (deterministic SDE mean).

        When ghi_cs is provided, pred_ghi = pred_kt * ghi_cs with
        pred_kt = sigmoid(head_ghi) * KT_MAX (hard physical bound, ~0 at night).
        When use_irradiance_head=False, pred_ghi is None.
        """
        x0 = self.encode(x, edge_index, edge_weight)
        h, g = self.sde(x0, stochastic=stochastic)

        if self.head_ghi is None:
            pred_ghi = None
        else:
            pred_kt = torch.sigmoid(self.head_ghi(h).squeeze(-1)) * self.KT_MAX  # (B, N)
            pred_ghi = pred_kt * ghi_cs if ghi_cs is not None else pred_kt
        pred_pv = F.softplus(self.head_pv(h).squeeze(-1))

        if return_diffusion:
            return pred_ghi, pred_pv, g
        return pred_ghi, pred_pv
