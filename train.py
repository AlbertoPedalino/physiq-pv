import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, DataLoader

from physiq_pv.data.synthetic_generator import generate_synthetic_dataset
from physiq_pv.data.quality_score import compute_qs
from physiq_pv.model.st_gnn import STGNN
from physiq_pv.model.graph_builder import build_graph
from physiq_pv.model.physics_loss import physics_loss_full
from physiq_pv.continual.replay_buffer import ReplayBuffer
from physiq_pv.continual.quality_gated_update import QualityGatedUpdater

SEQ_LEN = 120
N_FEATURES = 5   # temperature_2m, solar_irradiance_poa, wind_speed_10m, pvgis_ref, QS
BATCH_SIZE = 16
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PVDataset(Dataset):
    def __init__(self, ds: xr.Dataset, qs: xr.DataArray, seq_len: int = SEQ_LEN):
        self.seq_len = seq_len
        T = ds.sizes["time"]

        def _norm(arr: np.ndarray) -> np.ndarray:
            mu = np.nanmean(arr)
            s = np.nanstd(arr) + 1e-6
            return (np.nan_to_num(arr, nan=mu) - mu) / s

        temp = ds["temperature_2m"].values.T          # (T, N)
        solar = ds["solar_irradiance_poa"].values.T
        wind = ds["wind_speed_10m"].values.T
        ref = ds["pvgis_ref"].values.T
        qs_v = np.nan_to_num(qs.values.T, nan=0.5)   # (T, N)

        # (T, N, 5)
        self.feats = np.stack(
            [_norm(temp), _norm(solar), _norm(wind), _norm(ref), qs_v], axis=-1
        ).astype(np.float32)

        self.target_pv = np.nan_to_num(ds["ENERGIA"].values.T, nan=0.0).astype(np.float32)
        self.target_ghi = (_norm(solar) * np.nanstd(solar) + np.nanmean(solar)).astype(np.float32) / 1000.0
        self.eta_base = ds["eta_base"].values.astype(np.float32)  # (N,)
        self.qs_v = qs_v.astype(np.float32)
        self.valid_starts = np.arange(seq_len, T - 1)

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int):
        t = self.valid_starts[idx]
        # x: (N, seq_len, 5)
        x = torch.from_numpy(self.feats[t - self.seq_len : t].transpose(1, 0, 2))
        y_pv = torch.from_numpy(self.target_pv[t])    # (N,)
        y_ghi = torch.from_numpy(self.target_ghi[t])  # (N,) — scalar broadcast ok
        qs = torch.from_numpy(self.qs_v[t])            # (N,)
        eta = torch.from_numpy(self.eta_base)          # (N,)
        return x, y_ghi, y_pv, qs, eta


def _train_epoch(
    model: STGNN,
    loader: DataLoader,
    updater: QualityGatedUpdater,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    lam: float,
    device: str,
    max_steps: int | None = None,
) -> float:
    model.train()
    losses: list[float] = []

    for step, (x, y_ghi, y_pv, qs, eta) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x = x.to(device)
        y_ghi = y_ghi.to(device)
        y_pv = y_pv.to(device)
        qs = qs.to(device)
        eta = eta.to(device)
        ei = edge_index.to(device)
        ew = edge_weight.to(device)

        pred_ghi, pred_pv = model(x, ei, ew)

        # eta_T broadcast to (B, N)
        eta_T = eta  # (B, N) — already batched by DataLoader

        loss, _ld = physics_loss_full(pred_ghi, pred_pv, y_ghi, y_pv, eta_T, qs, lam=lam)
        qs_mean = float(qs.mean().item())

        updated = updater.step(
            x=x,
            y_pv=y_pv,
            pred_pv=pred_pv.detach(),
            loss=loss,
            qs_mean=qs_mean,
        )
        if updated:
            losses.append(loss.item())

    return float(np.mean(losses)) if losses else float("nan")


def train(
    ds: xr.Dataset | None = None,
    n_epochs: int = 5,
    lam: float = 0.1,
    max_steps_per_epoch: int | None = None,
) -> tuple:
    """Train ST-GNN. ds=None → generate synthetic dataset."""
    if ds is None:
        print("  Generating synthetic dataset...")
        ds = generate_synthetic_dataset()

    qs = compute_qs(ds)
    n_plants = ds.sizes["plant"]
    lats = ds["lat"].values
    lons = ds["lon"].values

    edge_index, edge_weight = build_graph(lats, lons, max_dist_km=50.0)
    print(f"  Graph: {n_plants} nodes, {edge_index.shape[1]} edges")

    dataset = PVDataset(ds, qs)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    model = STGNN(
        n_nodes=n_plants,
        n_features=N_FEATURES,
        seq_len=SEQ_LEN,
        patch_len=16,
        stride=8,
        d_model=128,
        gat_dim=256,
        gat_heads=4,
        gat_layers=2,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    buffer = ReplayBuffer(capacity=1000)
    updater = QualityGatedUpdater(
        model=model,
        optimizer=optimizer,
        buffer=buffer,
        edge_index=edge_index,
        edge_weight=edge_weight,
        qs_threshold=0.5,
        alpha_der=0.2,
        beta_der=1.0,
    )

    loss_history: list[float] = []
    for epoch in range(1, n_epochs + 1):
        avg_loss = _train_epoch(model, loader, updater, edge_index, edge_weight, lam, DEVICE, max_steps=max_steps_per_epoch)
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch}/{n_epochs}  loss={avg_loss:.4f}  buffer={len(buffer)}")

    return model, loss_history, updater, edge_index, edge_weight


if __name__ == "__main__":
    model, history = train(n_epochs=3)
    print("Loss curve:", " → ".join(f"{l:.4f}" for l in history))
