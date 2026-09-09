import torch
import torch.nn as nn
import torch.nn.functional as F

from physiq_pv.model.bilstm_encoder import BiLSTMEncoder
from physiq_pv.model.sde_net import PaperSDEBlock


class SDEBlock(PaperSDEBlock):
    """Backward-compatible name for the paper-faithful SDE block.

    ``sigma_max`` was the name used by the previous PV adaptation.  In the
    paper it is the global multiplier ``sigma`` of the sigmoid diffusion.
    """

    def __init__(self, dim: int, n_steps: int = 4, sigma_max: float = 0.5):
        super().__init__(dim=dim, n_steps=n_steps, sigma=sigma_max)
        self.sigma_max = self.sigma


class STGNN(nn.Module):
    """
    Per-node temporal model for PV forecasting with a neural-SDE uncertainty block.

    GAT ablation: the spatial message-passing stage is removed, so nodes never
    exchange information and every location is forecast independently.  The
    geographic graph is still built and threaded through the call chain, but no
    layer consumes it.

    Architecture per forward pass:
        1. BiLSTM encoder (per-node) -> temporal embedding
        2. Linear projection -> gat_dim                               => x0
        3. PaperSDEBlock: Euler-Maruyama x0 -> x_T (Brownian motion = uncertainty source)
        4. Dual head -> pred_kt_poa and a Gaussian PV head
           (pred_pv_mean, pred_pv_sigma). pred_poa = pred_kt_poa * poa_cs.

    Uncertainty has the two sources of Kong et al. (2020): the SDE diffusion
    term (g·dW) gives epistemic uncertainty (spread of the predictive mean over
    Brownian paths), and the PV head's softplus sigma is the aleatoric
    uncertainty (heteroscedastic Gaussian output, trained with NLL).  This is
    the paper's regression design (supplementary S.4.2: ``mean = x[:,0]``,
    ``sigma = softplus(x[:,1]) + 1e-3``); the only PV-domain change is a softplus
    on the mean so night-time predictions stay non-negative.

    Steps 1-2 are SDE-Net's downsampling h1 and the heads are h2.  The
    diffusion is a scalar per graph example, broadcast over nodes and channels,
    as in the authors' image/regression implementations.
    """

    KT_POA_MAX: float = 1.6

    def __init__(
        self,
        n_nodes: int,
        n_features: int,          # includes QS and optional diagnostic features
        seq_len: int = 120,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        gat_dim: int = 256,       # kept as the model width name for run metadata
        dropout: float = 0.1,
        use_patchtst: bool = True,
        bilstm_pooling: str = "attn",
        n_sde_steps: int = 4,
        sigma_max: float = 0.5,
        use_irradiance_head: bool = True,
        kt_poa_max: float = 1.6,
        forecast_horizons: tuple[int, ...] = (1,),
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.forecast_horizons = tuple(int(value) for value in forecast_horizons)
        if not self.forecast_horizons or any(
            value < 1 for value in self.forecast_horizons
        ):
            raise ValueError(
                "forecast_horizons must contain positive integer horizons."
            )
        self.n_horizons = len(self.forecast_horizons)
        self.use_patchtst = use_patchtst
        if kt_poa_max <= 0:
            raise ValueError("kt_poa_max must be positive.")
        self.kt_poa_max = float(kt_poa_max)
        # Irradiance-head ablation: when False, head_poa is not created and
        # forward returns (None, pred_pv).
        self.use_irradiance_head = use_irradiance_head

        if use_patchtst:
            self.encoder = BiLSTMEncoder(
                n_features=n_features,
                seq_len=seq_len,
                hidden_dim=d_model,
                n_layers=2,
                dropout=dropout,
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
        self.sde = PaperSDEBlock(gat_dim, n_steps=n_sde_steps, sigma=sigma_max)

        def _head(out: int = 1):
            return nn.Sequential(
                nn.Linear(gat_dim, gat_dim // 2), nn.GELU(),
                nn.Linear(gat_dim // 2, out),
            )

        self.head_poa = _head(self.n_horizons) if use_irradiance_head else None
        # One heteroscedastic Gaussian pair per direct forecast horizon.  H=1
        # retains the original scalar interface; H>1 returns (B, N, H).
        self.head_pv = _head(out=2 * self.n_horizons)

    def encode(
        self,
        x: torch.Tensor,            # (B, N, seq_len, n_features)
        edge_index: torch.Tensor,   # unused: GAT ablation, kept for call compatibility
        edge_weight: torch.Tensor,  # unused: GAT ablation, kept for call compatibility
    ) -> torch.Tensor:              # (B, N, gat_dim) — the SDE initial state x0
        B, N, L, C = x.shape
        if self.use_patchtst:
            enc = self.encoder(x.reshape(B * N, L, C))   # (B*N, enc_dim) — BiLSTM
        else:
            enc = x.reshape(B * N, L * C)                # flatten ablation
        return self.proj(enc).reshape(B, N, -1)          # (B, N, gat_dim)

    def forward(
        self,
        x: torch.Tensor,                      # (B, N, seq_len, n_features)
        edge_index: torch.Tensor,             # (2, E)
        edge_weight: torch.Tensor,            # (E,)
        poa_cs: torch.Tensor | None = None,   # (B, N) clear-sky POA in kW/m²
        stochastic: bool = True,
        return_diffusion: bool = False,
    ):
        """
        Returns (pred_poa, pred_pv_mean, pred_pv_sigma) by default. With
        return_diffusion the per-example Brownian scale sigma*g is appended ->
        (pred_poa, pred_pv_mean, pred_pv_sigma, g).

        stochastic=True samples one Brownian path (training / SDE inference);
        stochastic=False integrates the drift only (deterministic SDE mean).

        When poa_cs is provided, pred_poa = pred_kt_poa * poa_cs. During
        auxiliary supervision poa_cs is omitted and the first output is
        pred_kt_poa. When use_irradiance_head=False, pred_poa is None.

        pred_pv_mean is the Gaussian mean (softplus, >= 0); pred_pv_sigma is the
        aleatoric std (softplus + 1e-3 > 0), the heteroscedastic noise of the PV
        likelihood used by the NLL training objective.
        """
        x0 = self.encode(x, edge_index, edge_weight)
        h, g = self.sde(x0, stochastic=stochastic)

        if self.head_poa is None:
            pred_poa = None
        else:
            pred_kt_poa = torch.sigmoid(self.head_poa(h)) * self.kt_poa_max
            if self.n_horizons == 1:
                pred_kt_poa = pred_kt_poa.squeeze(-1)
            if poa_cs is not None and pred_kt_poa.ndim == 3 and poa_cs.ndim == 2:
                poa_cs = poa_cs.unsqueeze(-1)
            pred_poa = pred_kt_poa * poa_cs if poa_cs is not None else pred_kt_poa

        pv_out = self.head_pv(h).reshape(
            *h.shape[:-1], self.n_horizons, 2
        )
        pred_pv_mean = F.softplus(pv_out[..., 0])
        pred_pv_sigma = F.softplus(pv_out[..., 1]) + 1e-3
        if self.n_horizons == 1:
            pred_pv_mean = pred_pv_mean.squeeze(-1)
            pred_pv_sigma = pred_pv_sigma.squeeze(-1)

        out = [pred_poa, pred_pv_mean, pred_pv_sigma]
        if return_diffusion:
            out.append(g)
        return tuple(out)
