import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

from physiq_pv.data.dataset import PVDataset, SEQ_LEN, N_FEATURES
from physiq_pv.data.synthetic_generator import generate_synthetic_dataset
from physiq_pv.data.quality_score import compute_qs
from physiq_pv.model.st_gnn import STGNN
from physiq_pv.model.graph_builder import build_graph
from physiq_pv.model.physics_loss import physics_loss_full
from physiq_pv.continual.replay_buffer import ReplayBuffer
from physiq_pv.continual.quality_gated_update import QualityGatedUpdater

BATCH_SIZE = 8
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# dropout=0.0 in the model removes the seed/offset requirement in flash SDP,
# allowing it to handle batch sizes > 65,535 (B*N*C = 32*1116*5 = 178,560).
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


def _train_epoch(
    model: STGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    buffer: "ReplayBuffer",
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    lam: float,
    device: str,
    scaler: GradScaler,
    max_steps: int | None = None,
) -> float:
    model.train()
    losses: list[float] = []
    fallback_count = 0

    for step, (x, y_ghi, y_pv, qs, eta) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x     = x.to(device)
        y_ghi = y_ghi.to(device)
        y_pv  = y_pv.to(device)
        qs    = qs.to(device)
        eta   = eta.to(device)
        ei    = edge_index.to(device)
        ew    = edge_weight.to(device)

        # Perturb weather features (channels 0-2: temp, solar_poa, wind) ±5%.
        # Geometry (3,4) and QS (5) are deterministic — not perturbed.
        noise = 1.0 + 0.05 * torch.randn(x.shape[0], x.shape[1], x.shape[2], 3, device=device)
        x = torch.cat([x[..., :3] * noise, x[..., 3:]], dim=-1)

        optimizer.zero_grad()
        try:
            with autocast(device_type='cuda', dtype=torch.float16):
                pred_ghi, pred_pv = model(x, ei, ew)
                loss, _ = physics_loss_full(pred_ghi, pred_pv, y_ghi, y_pv, eta, qs, lam=lam)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        except RuntimeError as e:
            # Fallback: if autocast fails, run in float32
            fallback_count += 1
            pred_ghi, pred_pv = model(x, ei, ew)
            loss, _ = physics_loss_full(pred_ghi, pred_pv, y_ghi, y_pv, eta, qs, lam=lam)
            loss.backward()
            optimizer.step()
        
        losses.append(loss.item())

        buffer.add_batch(x.cpu(), y_pv.cpu(), pred_pv.detach().cpu())

    if fallback_count > 0:
        print(f"  ⚠️  AMP fallback triggered {fallback_count} times (model not compatible with float16)")
    
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def _val_epoch(
    model: STGNN,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    lam: float,
    device: str,
) -> float:
    model.eval()
    losses: list[float] = []
    ei = edge_index.to(device)
    ew = edge_weight.to(device)
    for x, y_ghi, y_pv, qs, eta in loader:
        pred_ghi, pred_pv = model(x.to(device), ei, ew)
        loss, _ = physics_loss_full(pred_ghi, pred_pv, y_ghi.to(device), y_pv.to(device), eta.to(device), qs.to(device), lam=lam)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def train(
    ds: xr.Dataset | None = None,
    n_epochs: int = 5,
    lam: float = 0.1,
    max_steps_per_epoch: int | None = None,
    kwp: "np.ndarray | None" = None,
) -> tuple:
    """Train ST-GNN. ds=None → generate synthetic dataset."""
    if ds is None:
        print("  Generating synthetic dataset...")
        ds = generate_synthetic_dataset()

    qs = compute_qs(ds)
    n_plants = ds.sizes["plant"]
    lats = ds["lat"].values
    lons = ds["lon"].values

    edge_index, edge_weight = build_graph(lats, lons, max_dist_km=20.0)
    print(f"  Graph: {n_plants} nodes, {edge_index.shape[1]} edges")

    # Stratified monthly split: 80% of each month → train, 20% → val.
    # Ensures all seasons represented in both sets — avoids winter-only val distribution shift.
    dataset_full = PVDataset(ds, qs, kwp=kwp)
    times = pd.DatetimeIndex(ds.coords["time"].values)
    valid_starts = dataset_full.valid_starts  # (n_windows,) — time indices of prediction steps

    train_indices: list[int] = []
    val_indices:   list[int] = []
    for month in range(1, 13):
        month_mask = np.where(times[valid_starts].month == month)[0]
        if len(month_mask) == 0:
            continue
        split = int(len(month_mask) * 0.8)
        train_indices.extend(month_mask[:split].tolist())
        val_indices.extend(month_mask[split:].tolist())

    dataset_train = Subset(dataset_full, sorted(train_indices))
    dataset_val   = Subset(dataset_full, sorted(val_indices))
    print(f"  Split: {len(dataset_train)} train windows, {len(dataset_val)} val windows (stratified monthly)")

    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    loader_val   = DataLoader(dataset_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    model = STGNN(
        n_nodes=n_plants,
        n_features=N_FEATURES,
        seq_len=SEQ_LEN,
        patch_len=4,
        stride=2,
        d_model=128,
        gat_dim=256,
        gat_heads=4,
        gat_layers=2,
        dropout=0.0,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = GradScaler(device='cuda')
    buffer = ReplayBuffer(capacity=1000)
    updater = QualityGatedUpdater(
        model=model,
        optimizer=optimizer,
        buffer=buffer,
        edge_index=edge_index,
        edge_weight=edge_weight,
        qs_threshold=0.0,
        alpha_der=0.2,
        beta_der=1.0,
    )

    loss_history: list[float] = []
    val_loss_history: list[float] = []
    best_val_loss = float("inf")
    best_state: dict = {}

    for epoch in range(1, n_epochs + 1):
        avg_loss = _train_epoch(model, loader_train, optimizer, buffer, edge_index, edge_weight, lam, DEVICE, scaler, max_steps=max_steps_per_epoch)
        val_loss = _val_epoch(model, loader_val, edge_index, edge_weight, lam, DEVICE)
        loss_history.append(avg_loss)
        val_loss_history.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  Epoch {epoch}/{n_epochs}  train={avg_loss:.4f}  val={val_loss:.4f}  buffer={len(buffer)}")

    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    return model, loss_history, val_loss_history, updater, edge_index, edge_weight


if __name__ == "__main__":
    model, history, *_ = train(n_epochs=3)
    print("Loss curve:", " -> ".join(f"{l:.4f}" for l in history))
