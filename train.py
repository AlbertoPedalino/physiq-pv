import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader, Subset

from physiq_pv.data.dataset import PVDataset, SEQ_LEN, N_FEATURES
from physiq_pv.data.synthetic_generator import generate_synthetic_dataset
from physiq_pv.data.quality_score import compute_qs
from physiq_pv.model.st_gnn import STGNN
from physiq_pv.model.graph_builder import build_graph
from physiq_pv.model.physics_loss import physics_loss_full, quality_weight
from physiq_pv.model.postprocessing import apply_pv_calibration_np
from physiq_pv.continual.replay_buffer import ReplayBuffer
from physiq_pv.continual.quality_gated_update import QualityGatedUpdater

BATCH_SIZE = 8
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# dropout=0.0 in the model removes the seed/offset requirement in flash SDP,
# allowing it to handle batch sizes > 65,535 (B*N*C = 32*1116*5 = 178,560).
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


def _peak_weight(y_true: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    y_pos = torch.clamp(y_true, min=0.0)
    return 1.0 + alpha * y_pos.pow(gamma)


def _asymmetric_peak_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    alpha: float,
    gamma: float,
    under_penalty: float = 2.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted asymmetric MAE: under-predictions penalized `under_penalty`x harder than over-predictions."""
    w = _peak_weight(true, alpha, gamma)
    if sample_weight is not None:
        w = w * sample_weight
    err = pred - true
    asym = torch.where(err < 0, under_penalty * err.abs(), err.abs())
    return (w * asym).mean()


def _train_epoch(
    model: STGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    buffer: "ReplayBuffer",
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    lam: float,
    device: str,
    peak_alpha: float,
    peak_gamma: float,
    peak_loss_weight: float,
    qs_weight_exponent: float,
    qs_weight_floor: float,
    max_steps: int | None = None,
) -> float:
    model.train()
    losses: list[float] = []
    ei = edge_index.to(device)
    ew = edge_weight.to(device)

    for step, (x, y_ghi, y_pv, qs, eta) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x = x.to(device, non_blocking=True)
        y_ghi = y_ghi.to(device, non_blocking=True)
        y_pv = y_pv.to(device, non_blocking=True)
        qs = qs.to(device, non_blocking=True)
        eta = eta.to(device, non_blocking=True)

        # Perturb weather features (channels 0-2: temp, solar_poa, wind) by +/-5%.
        # Geometry (3,4) and QS (5) are deterministic; do not perturb them.
        noise = 1.0 + 0.05 * torch.randn(x.shape[0], x.shape[1], x.shape[2], 3, device=device)
        x = torch.cat([x[..., :3] * noise, x[..., 3:]], dim=-1)

        pred_ghi, pred_pv = model(x, ei, ew)
        loss_base, _ = physics_loss_full(
            pred_ghi,
            pred_pv,
            y_ghi,
            y_pv,
            eta,
            qs,
            lam=lam,
            qs_weight_exponent=qs_weight_exponent,
            qs_weight_floor=qs_weight_floor,
        )
        q_weight = quality_weight(qs, qs_weight_exponent, qs_weight_floor)
        loss_peak = _asymmetric_peak_loss(
            pred_pv,
            y_pv,
            peak_alpha,
            peak_gamma,
            sample_weight=q_weight,
        )
        loss = loss_base + peak_loss_weight * loss_peak

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        buffer.add_batch(x.cpu(), y_pv.cpu(), pred_pv.detach().cpu())

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def _val_epoch(
    model: STGNN,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    lam: float,
    device: str,
    peak_alpha: float,
    peak_gamma: float,
    peak_loss_weight: float,
    qs_weight_exponent: float,
    qs_weight_floor: float,
) -> float:
    model.eval()
    losses: list[float] = []
    ei = edge_index.to(device)
    ew = edge_weight.to(device)
    for x, y_ghi, y_pv, qs, eta in loader:
        pred_ghi, pred_pv = model(x.to(device), ei, ew)
        y_ghi_d = y_ghi.to(device)
        y_pv_d = y_pv.to(device)
        eta_d = eta.to(device)
        qs_d = qs.to(device)
        loss_base, _ = physics_loss_full(
            pred_ghi,
            pred_pv,
            y_ghi_d,
            y_pv_d,
            eta_d,
            qs_d,
            lam=lam,
            qs_weight_exponent=qs_weight_exponent,
            qs_weight_floor=qs_weight_floor,
        )
        q_weight = quality_weight(qs_d, qs_weight_exponent, qs_weight_floor)
        loss_peak = _asymmetric_peak_loss(
            pred_pv,
            y_pv_d,
            peak_alpha,
            peak_gamma,
            sample_weight=q_weight,
        )
        loss = loss_base + peak_loss_weight * loss_peak
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def _fit_pv_linear_calibration(
    model: STGNN,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: str,
    daytime_ghi_threshold: float = 0.01,
    calibration_kpi: str = "none",
) -> dict:
    if calibration_kpi not in {"rmse", "mae", "both", "none"}:
        raise ValueError("calibration_kpi must be one of: 'rmse', 'mae', 'both', 'none'")

    model.eval()
    ei = edge_index.to(device)
    ew = edge_weight.to(device)
    pred_day: list[np.ndarray] = []
    true_day: list[np.ndarray] = []

    for x, y_ghi, y_pv, _qs, _eta in loader:
        x_d = x.to(device)
        y_ghi_d = y_ghi.to(device)
        y_pv_d = y_pv.to(device)
        _pg, pred_pv = model(x_d, ei, ew)
        mask = y_ghi_d > daytime_ghi_threshold
        if mask.any():
            pred_day.append(pred_pv[mask].detach().cpu().numpy().astype(np.float64))
            true_day.append(y_pv_d[mask].detach().cpu().numpy().astype(np.float64))

    if not pred_day:
        return {
            "enabled": False,
            "reason": "no_daytime_samples",
            "calibration_kpi": calibration_kpi,
            "slope": 1.0,
            "intercept": 0.0,
            "daytime_ghi_threshold": daytime_ghi_threshold,
            "prediction_floor": 0.0,
            "n_samples": 0,
        }

    pred = np.concatenate(pred_day)
    true = np.concatenate(true_day)
    if pred.size < 100:
        return {
            "enabled": False,
            "reason": "too_few_samples",
            "calibration_kpi": calibration_kpi,
            "slope": 1.0,
            "intercept": 0.0,
            "daytime_ghi_threshold": daytime_ghi_threshold,
            "prediction_floor": 0.0,
            "n_samples": int(pred.size),
        }

    x = np.column_stack([pred, np.ones_like(pred)])
    slope, intercept = np.linalg.lstsq(x, true, rcond=None)[0]
    if not np.isfinite(slope) or not np.isfinite(intercept):
        slope, intercept = 1.0, 0.0

    candidate_calibration = {"enabled": True, "slope": slope, "intercept": intercept}
    pred_cal = apply_pv_calibration_np(pred, candidate_calibration)
    mae_before = float(np.mean(np.abs(pred - true)))
    mae_after = float(np.mean(np.abs(pred_cal - true)))
    rmse_before = float(np.sqrt(np.mean((pred - true) ** 2)))
    rmse_after = float(np.sqrt(np.mean((pred_cal - true) ** 2)))
    negative_before = int(np.sum((slope * pred + intercept) < 0.0))
    negative_after = int(np.sum(pred_cal < 0.0))

    if calibration_kpi == "none":
        enabled = False
        selection_reason = "disabled_by_config"
    elif calibration_kpi == "mae":
        enabled = bool(mae_after < mae_before)
        selection_reason = "mae_improved" if enabled else "mae_not_improved"
    elif calibration_kpi == "both":
        enabled = bool(mae_after < mae_before and rmse_after < rmse_before)
        selection_reason = "mae_and_rmse_improved" if enabled else "mae_or_rmse_not_improved"
    else:  # "rmse" (default)
        enabled = bool(rmse_after < rmse_before)
        selection_reason = "rmse_improved" if enabled else "rmse_not_improved"

    return {
        "enabled": enabled,
        "selection_reason": selection_reason,
        "calibration_kpi": calibration_kpi,
        "slope": float(slope),
        "intercept": float(intercept),
        "daytime_ghi_threshold": float(daytime_ghi_threshold),
        "prediction_floor": 0.0,
        "n_samples": int(pred.size),
        "mae_before": mae_before,
        "mae_after": mae_after,
        "rmse_before": rmse_before,
        "rmse_after": rmse_after,
        "negative_before_floor": negative_before,
        "negative_after_floor": negative_after,
    }


def train(
    ds: xr.Dataset | None = None,
    n_epochs: int = 5,
    lam: float = 0.1,
    max_steps_per_epoch: int | None = None,
    kwp: "np.ndarray | None" = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    peak_alpha: float = 2.0,
    peak_gamma: float = 2.0,
    peak_loss_weight: float = 0.5,
    calibration_kpi: str = "none",
    qs_weight_exponent: float = 0.2,
    qs_weight_floor: float = 0.2,
    eta_max: float = 0.98,
) -> tuple:
    """Train ST-GNN. ds=None generates a synthetic dataset."""
    if ds is None:
        print("  Generating synthetic dataset...")
        ds = generate_synthetic_dataset()

    qs = compute_qs(ds)
    n_plants = ds.sizes["plant"]
    lats = ds["lat"].values
    lons = ds["lon"].values

    edge_index, edge_weight = build_graph(lats, lons, max_dist_km=10.0)
    print(f"  Graph: {n_plants} nodes, {edge_index.shape[1]} edges")

    # Stratified monthly split: 80% of each month to train, 20% to validation.
    # This keeps all available seasons represented in both sets.
    dataset_full = PVDataset(ds, qs, kwp=kwp, eta_max=eta_max)
    times = pd.DatetimeIndex(ds.coords["time"].values)
    valid_starts = dataset_full.valid_starts  # (n_windows,) time indices of prediction steps

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
        d_model=64,
        gat_dim=96,
        gat_heads=4,
        gat_layers=1,
        dropout=0.0,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    buffer = ReplayBuffer(capacity=1000)
    updater = QualityGatedUpdater(
        model=model,
        optimizer=optimizer,
        buffer=buffer,
        edge_index=edge_index,
        edge_weight=edge_weight,
        qs_threshold=None,
        alpha_der=0.2,
        beta_der=1.0,
    )

    loss_history: list[float] = []
    val_loss_history: list[float] = []
    best_val_loss = float("inf")
    best_val_epoch: int = 0
    best_state: dict = {}
    no_improve_count = 0

    for epoch in range(1, n_epochs + 1):
        avg_loss = _train_epoch(
            model,
            loader_train,
            optimizer,
            buffer,
            edge_index,
            edge_weight,
            lam,
            DEVICE,
            peak_alpha,
            peak_gamma,
            peak_loss_weight,
            qs_weight_exponent,
            qs_weight_floor,
            max_steps=max_steps_per_epoch,
        )
        val_loss = _val_epoch(
            model,
            loader_val,
            edge_index,
            edge_weight,
            lam,
            DEVICE,
            peak_alpha,
            peak_gamma,
            peak_loss_weight,
            qs_weight_exponent,
            qs_weight_floor,
        )
        loss_history.append(avg_loss)
        val_loss_history.append(val_loss)
        if val_loss < (best_val_loss - early_stopping_min_delta):
            best_val_loss = val_loss
            best_val_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1
        print(f"  Epoch {epoch}/{n_epochs}  train={avg_loss:.4f}  val={val_loss:.4f}  buffer={len(buffer)}")

        if early_stopping_patience is not None and no_improve_count >= early_stopping_patience:
            print(
                f"  Early stopping: no val improvement for {early_stopping_patience} epochs "
                f"(best={best_val_loss:.4f})"
            )
            break

    if best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    pv_calibration = _fit_pv_linear_calibration(
        model, loader_val, edge_index, edge_weight, DEVICE,
        calibration_kpi=calibration_kpi,
    )
    pv_calibration["best_val_epoch"] = best_val_epoch
    return model, loss_history, val_loss_history, updater, edge_index, edge_weight, pv_calibration


if __name__ == "__main__":
    model, history, *_ = train(n_epochs=3)
    print("Loss curve:", " -> ".join(f"{l:.4f}" for l in history))
