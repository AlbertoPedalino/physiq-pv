from __future__ import annotations

import json
import os
import random

import numpy as np
import pandas as pd
import torch
import wandb
import xarray as xr
from torch.utils.data import DataLoader, Subset

from physiq_pv.data.dataset import (
    FEATURE_NAMES,
    N_FEATURES,
    PVDataset,
    SEQ_LEN,
    validate_hourly_grid,
)
from physiq_pv.data.quality_score import compute_qs
from physiq_pv.data.synthetic_generator import generate_synthetic_dataset
from physiq_pv.model.graph_builder import build_graph
from physiq_pv.model.physics_loss import physics_loss_full
from physiq_pv.model.st_gnn import STGNN

BATCH_SIZE = 8
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PV_LAG_INDEX = FEATURE_NAMES.index("pv_lag")
DAY_POA_CS_THRESHOLD = 0.05


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


def _chronological_split(
    n_steps: int,
    seq_len: int,
    validation_fraction: float,
) -> tuple[np.ndarray, list[int], list[int], np.ndarray]:
    """Return target timestamps, subset indices and the training fit mask."""
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    valid_starts = np.arange(seq_len, n_steps)
    split = int(len(valid_starts) * (1.0 - validation_fraction))
    if split < 1 or split >= len(valid_starts):
        raise ValueError("Dataset is too short for the requested split")

    train_indices = list(range(split))
    val_indices = list(range(split, len(valid_starts)))
    fit_end = int(valid_starts[split - 1])
    fit_time_mask = np.arange(n_steps) <= fit_end
    return valid_starts, train_indices, val_indices, fit_time_mask


def _peak_weight(y_true: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    return 1.0 + alpha * torch.clamp(y_true, min=0.0).pow(gamma)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def _asymmetric_peak_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    alpha: float,
    gamma: float,
    under_penalty: float,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    peak_weight = _peak_weight(true, alpha, gamma)
    error = pred - true
    asymmetric = torch.where(
        error < 0,
        under_penalty * error.abs(),
        error.abs(),
    )
    return _weighted_mean(peak_weight * asymmetric, sample_weight)


def _day_weight(poa_cs: torch.Tensor, night_loss_weight: float) -> torch.Tensor:
    return torch.where(
        poa_cs > DAY_POA_CS_THRESHOLD,
        torch.ones_like(poa_cs),
        torch.full_like(poa_cs, night_loss_weight),
    )


def _train_epoch(
    model: STGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    lam: float,
    device: str,
    peak_alpha: float,
    peak_gamma: float,
    peak_loss_weight: float,
    under_penalty: float,
    night_loss_weight: float,
    max_steps: int | None,
) -> float:
    model.train()
    losses: list[float] = []
    edge_index_device = edge_index.to(device)
    edge_weight_device = edge_weight.to(device)

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        (
            x,
            y_poa,
            y_pv,
            pr_proxy,
            poa_cs,
            poa_scale,
            pv_target_valid,
            _pv_lag_valid,
        ) = batch
        x = x.to(device, non_blocking=True)
        y_poa = y_poa.to(device, non_blocking=True)
        y_pv = y_pv.to(device, non_blocking=True)
        pr_proxy = pr_proxy.to(device, non_blocking=True)
        poa_cs = poa_cs.to(device, non_blocking=True)
        poa_scale = poa_scale.to(device, non_blocking=True)
        pv_target_valid = pv_target_valid.to(device, non_blocking=True)

        # Additive noise is appropriate for z-scored temperature and wind.
        x = x.clone()
        x[..., :2] += 0.05 * torch.randn_like(x[..., :2])

        pred_poa, pred_pv = model(
            x,
            edge_index_device,
            edge_weight_device,
            poa_cs,
        )
        sample_weight = _day_weight(poa_cs, night_loss_weight)
        loss_base, _ = physics_loss_full(
            pred_poa,
            pred_pv,
            y_poa,
            y_pv,
            pr_proxy,
            poa_scale,
            lam=lam,
            sample_weight=sample_weight,
            pv_valid=pv_target_valid,
        )
        pv_weight = sample_weight * pv_target_valid
        loss_peak = _asymmetric_peak_loss(
            pred_pv,
            y_pv,
            peak_alpha,
            peak_gamma,
            under_penalty,
            pv_weight,
        )
        loss = loss_base + peak_loss_weight * loss_peak

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else float("nan")


def _error_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    if pred.size == 0:
        return {}
    error = pred - true
    return {
        f"mae_{prefix}": float(np.mean(np.abs(error))),
        f"rmse_{prefix}": float(np.sqrt(np.mean(error**2))),
        f"bias_{prefix}": float(np.mean(error)),
    }


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
    under_penalty: float,
    night_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    edge_index_device = edge_index.to(device)
    edge_weight_device = edge_weight.to(device)

    pv_pred_all: list[np.ndarray] = []
    pv_true_all: list[np.ndarray] = []
    poa_pred_all: list[np.ndarray] = []
    poa_true_all: list[np.ndarray] = []
    poa_cs_all: list[np.ndarray] = []
    persistence_all: list[np.ndarray] = []
    pv_target_valid_all: list[np.ndarray] = []
    pv_lag_valid_all: list[np.ndarray] = []

    for batch in loader:
        (
            x,
            y_poa,
            y_pv,
            pr_proxy,
            poa_cs,
            poa_scale,
            pv_target_valid,
            pv_lag_valid,
        ) = batch
        x_device = x.to(device, non_blocking=True)
        y_poa_device = y_poa.to(device)
        y_pv_device = y_pv.to(device)
        pr_proxy_device = pr_proxy.to(device)
        poa_scale_device = poa_scale.to(device)
        pv_target_valid_device = pv_target_valid.to(device)
        poa_cs_device = poa_cs.to(device, non_blocking=True)
        pred_poa, pred_pv = model(
            x_device,
            edge_index_device,
            edge_weight_device,
            poa_cs_device,
        )
        sample_weight = _day_weight(poa_cs_device, night_loss_weight)
        loss_base, _ = physics_loss_full(
            pred_poa,
            pred_pv,
            y_poa_device,
            y_pv_device,
            pr_proxy_device,
            poa_scale_device,
            lam=lam,
            sample_weight=sample_weight,
            pv_valid=pv_target_valid_device,
        )
        pv_weight = sample_weight * pv_target_valid_device
        loss_peak = _asymmetric_peak_loss(
            pred_pv,
            y_pv_device,
            peak_alpha,
            peak_gamma,
            under_penalty,
            pv_weight,
        )
        losses.append(float((loss_base + peak_loss_weight * loss_peak).item()))

        pv_pred_all.append(pred_pv.cpu().numpy().ravel())
        pv_true_all.append(y_pv.numpy().ravel())
        poa_pred_all.append(pred_poa.cpu().numpy().ravel())
        poa_true_all.append(y_poa.numpy().ravel())
        poa_cs_all.append(poa_cs.numpy().ravel())
        persistence_all.append(x[:, :, -1, PV_LAG_INDEX].numpy().ravel())
        pv_target_valid_all.append(pv_target_valid.numpy().ravel())
        pv_lag_valid_all.append(pv_lag_valid.numpy().ravel())

    metrics: dict[str, float] = {
        "val_loss": float(np.mean(losses)) if losses else float("nan")
    }
    if not pv_pred_all:
        return metrics

    pv_pred = np.concatenate(pv_pred_all)
    pv_true = np.concatenate(pv_true_all)
    poa_pred = np.concatenate(poa_pred_all)
    poa_true = np.concatenate(poa_true_all)
    poa_cs_values = np.concatenate(poa_cs_all)
    persistence = np.concatenate(persistence_all)
    pv_target_valid = np.concatenate(pv_target_valid_all).astype(bool)
    pv_lag_valid = np.concatenate(pv_lag_valid_all).astype(bool)
    poa_day = poa_cs_values > DAY_POA_CS_THRESHOLD
    poa_night = ~poa_day
    pv_day = poa_day & pv_target_valid
    pv_night = poa_night & pv_target_valid
    persistence_day = pv_day & pv_lag_valid

    metrics.update(
        _error_metrics(
            pv_pred[pv_target_valid],
            pv_true[pv_target_valid],
            "pv",
        )
    )
    metrics.update(_error_metrics(pv_pred[pv_day], pv_true[pv_day], "pv_day"))
    metrics.update(
        _error_metrics(pv_pred[pv_night], pv_true[pv_night], "pv_night")
    )
    metrics.update(
        _error_metrics(poa_pred[poa_day], poa_true[poa_day], "poa_day")
    )
    metrics.update(
        _error_metrics(
            persistence[persistence_day],
            pv_true[persistence_day],
            "persistence_day",
        )
    )
    metrics["n_poa_day"] = int(poa_day.sum())
    metrics["n_pv_day"] = int(pv_day.sum())
    metrics["n_pv_night"] = int(pv_night.sum())
    metrics["n_pv_missing"] = int((~pv_target_valid).sum())
    metrics["n_persistence_day"] = int(persistence_day.sum())

    bins = [
        ("0_20", 0.0, 0.2),
        ("20_40", 0.2, 0.4),
        ("40_60", 0.4, 0.6),
        ("60_80", 0.6, 0.8),
        ("80_100", 0.8, 1.0),
        ("over_100", 1.0, np.inf),
    ]
    error = pv_pred - pv_true
    for name, lower, upper in bins:
        mask = (
            pv_target_valid
            & poa_day
            & (pv_true >= lower)
            & (pv_true < upper)
        )
        metrics[f"n_{name}"] = int(mask.sum())
        if mask.any():
            values = error[mask]
            metrics[f"mae_pv_{name}"] = float(np.mean(np.abs(values)))
            metrics[f"rmse_pv_{name}"] = float(
                np.sqrt(np.mean(values**2))
            )
            metrics[f"bias_pv_{name}"] = float(np.mean(values))

    return metrics


def train(
    ds: xr.Dataset | None = None,
    n_epochs: int = 5,
    lam: float = 0.1,
    max_steps_per_epoch: int | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    peak_alpha: float = 2.0,
    peak_gamma: float = 2.0,
    peak_loss_weight: float = 0.5,
    under_penalty: float = 2.0,
    pr_max: float = 1.5,
    night_loss_weight: float = 0.2,
    validation_fraction: float = 0.2,
    selection_metric: str = "rmse_pv_day",
    include_poa_inputs: bool = True,
    poa_kt_max: float = 1.6,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    graph_max_dist_km: float = 20.0,
    graph_distance_scale_km: float | None = None,
    edge_prior_strength: float = 1.0,
    d_model: int = 128,
    gat_dim: int = 96,
    gat_heads: int = 4,
    gat_layers: int = 1,
    dropout: float = 0.2,
    use_wandb: bool = True,
    wandb_project: str = "physiq-pv",
    wandb_entity: str | None = "albertopedalino-politecnico-di-torino",
    wandb_run_name: str | None = None,
    wandb_tags: list[str] | None = None,
    seq_len: int = SEQ_LEN,
    checkpoint_dir: str = "checkpoints",
    use_bilstm: bool = True,
    use_gat: bool = True,
    bilstm_pooling: str = "attn",
    seed: int = 42,
) -> tuple:
    """Train the BiLSTM+GAT using train-only preprocessing statistics."""
    _set_global_seed(seed)
    if not 0.0 <= night_loss_weight <= 1.0:
        raise ValueError("night_loss_weight must be in [0, 1]")

    if ds is None:
        print("  Generating synthetic dataset...")
        ds = generate_synthetic_dataset()

    times = pd.DatetimeIndex(ds.coords["time"].values)
    validate_hourly_grid(times)
    (
        valid_starts,
        train_indices,
        val_indices,
        fit_time_mask,
    ) = _chronological_split(
        ds.sizes["time"],
        seq_len,
        validation_fraction,
    )

    _quality_score, m_components = compute_qs(
        ds,
        debug=True,
        fit_time_mask=fit_time_mask,
    )
    dataset_full = PVDataset(
        ds,
        m_components,
        seq_len=seq_len,
        pr_max=pr_max,
        fit_time_mask=fit_time_mask,
        include_poa_inputs=include_poa_inputs,
    )
    if not np.array_equal(valid_starts, dataset_full.valid_starts):
        raise RuntimeError("Split and dataset target indices are inconsistent")

    dataset_train = Subset(dataset_full, train_indices)
    dataset_val = Subset(dataset_full, val_indices)
    train_end = times[valid_starts[train_indices[-1]]]
    val_start = times[valid_starts[val_indices[0]]]
    print(
        f"  Split: {len(dataset_train)} train windows through {train_end}; "
        f"{len(dataset_val)} validation windows from {val_start}"
    )

    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    lats = ds["lat"].values
    lons = ds["lon"].values
    edge_index, edge_weight = build_graph(
        lats,
        lons,
        max_dist_km=graph_max_dist_km,
        distance_scale_km=graph_distance_scale_km,
    )
    degree = torch.bincount(edge_index[1], minlength=ds.sizes["plant"])
    if (degree == 0).any():
        raise RuntimeError("Graph contains isolated nodes")
    print(
        f"  Graph: {ds.sizes['plant']} nodes, {edge_index.shape[1]} edges, "
        f"min degree={int(degree.min())}"
    )

    model = STGNN(
        n_nodes=ds.sizes["plant"],
        n_features=N_FEATURES,
        seq_len=seq_len,
        d_model=d_model,
        gat_dim=gat_dim,
        gat_heads=gat_heads,
        gat_layers=gat_layers,
        dropout=dropout,
        use_bilstm=use_bilstm,
        use_gat=use_gat,
        bilstm_pooling=bilstm_pooling,
        poa_kt_max=poa_kt_max,
        edge_prior_strength=edge_prior_strength,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    config = {
        "n_epochs": n_epochs,
        "lam": lam,
        "peak_alpha": peak_alpha,
        "peak_gamma": peak_gamma,
        "peak_loss_weight": peak_loss_weight,
        "under_penalty": under_penalty,
        "pr_max": pr_max,
        "night_loss_weight": night_loss_weight,
        "validation_fraction": validation_fraction,
        "selection_metric": selection_metric,
        "include_poa_inputs": include_poa_inputs,
        "poa_kt_max": poa_kt_max,
        "batch_size": batch_size,
        "seed": seed,
        "lr": lr,
        "weight_decay": weight_decay,
        "num_workers": num_workers,
        "graph_max_dist_km": graph_max_dist_km,
        "graph_distance_scale_km": graph_distance_scale_km,
        "edge_prior_strength": edge_prior_strength,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "max_steps_per_epoch": max_steps_per_epoch,
        "device": DEVICE,
        "n_features": N_FEATURES,
        "features": list(FEATURE_NAMES),
        "seq_len": seq_len,
        "checkpoint_dir": checkpoint_dir,
        "use_bilstm": use_bilstm,
        "use_gat": use_gat,
        "bilstm_pooling": bilstm_pooling,
        "d_model": d_model,
        "gat_dim": gat_dim,
        "gat_heads": gat_heads,
        "gat_layers": gat_layers,
        "dropout": dropout,
    }

    run = None
    run_owned_here = False
    if use_wandb:
        if wandb.run is not None:
            run = wandb.run
            run.config.update(config, allow_val_change=True)
        else:
            run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                tags=wandb_tags,
                config=config,
            )
            run_owned_here = True

    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(
        os.path.join(checkpoint_dir, "preprocessing_state.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(dataset_full.preprocessing_state, file, indent=2)

    loss_history: list[float] = []
    val_loss_history: list[float] = []
    best_score = float("inf")
    best_val_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    best_val_metrics: dict[str, float] = {}
    no_improve_count = 0

    for epoch in range(1, n_epochs + 1):
        train_loss = _train_epoch(
            model,
            loader_train,
            optimizer,
            edge_index,
            edge_weight,
            lam,
            DEVICE,
            peak_alpha,
            peak_gamma,
            peak_loss_weight,
            under_penalty,
            night_loss_weight,
            max_steps_per_epoch,
        )
        val_metrics = _val_epoch(
            model,
            loader_val,
            edge_index,
            edge_weight,
            lam,
            DEVICE,
            peak_alpha,
            peak_gamma,
            peak_loss_weight,
            under_penalty,
            night_loss_weight,
        )
        if selection_metric not in val_metrics:
            raise KeyError(
                f"selection_metric {selection_metric!r} not found in validation metrics"
            )
        score = float(val_metrics[selection_metric])
        if not np.isfinite(score):
            raise ValueError(
                f"selection metric {selection_metric!r} is not finite"
            )
        loss_history.append(train_loss)
        val_loss_history.append(float(val_metrics["val_loss"]))

        if score < best_score - early_stopping_min_delta:
            best_score = score
            best_val_epoch = epoch
            best_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_val_metrics = dict(val_metrics)
            no_improve_count = 0
        else:
            no_improve_count += 1

        persistence_score = float(
            val_metrics.get("rmse_persistence_day", float("nan"))
        )
        print(
            f"  Epoch {epoch}/{n_epochs} train={train_loss:.4f} "
            f"val={val_metrics['val_loss']:.4f} "
            f"rmse_pv_day={val_metrics['rmse_pv_day']:.4f} "
            f"persistence={persistence_score:.4f}"
        )
        if run is not None:
            payload = {
                "epoch": epoch,
                "train_loss": train_loss,
                "best_selection_score": best_score,
                "no_improve_count": no_improve_count,
            }
            payload.update(val_metrics)
            run.log(payload)

        if (
            early_stopping_patience is not None
            and no_improve_count >= early_stopping_patience
        ):
            print(
                f"  Early stopping on {selection_metric}: "
                f"best={best_score:.4f} at epoch {best_val_epoch}"
            )
            break

    if best_state:
        model.load_state_dict(
            {key: value.to(DEVICE) for key, value in best_state.items()}
        )

    model.training_summary = {
        "selection_metric": selection_metric,
        "best_selection_score": best_score,
        "best_val_epoch": best_val_epoch,
        "best_val_metrics": best_val_metrics,
        "preprocessing_state": dataset_full.preprocessing_state,
    }

    if run is not None:
        run.summary["selection_metric"] = selection_metric
        run.summary["best_selection_score"] = best_score
        run.summary["best_val_epoch"] = best_val_epoch
        for key, value in best_val_metrics.items():
            run.summary[f"best_{key}"] = value
        if run_owned_here:
            run.finish()

    return (
        model,
        loss_history,
        val_loss_history,
        edge_index,
        edge_weight,
        best_val_epoch,
    )


if __name__ == "__main__":
    trained_model, history, *_ = train(
        n_epochs=3,
        use_wandb=False,
        num_workers=0,
    )
    print("Loss curve:", " -> ".join(f"{value:.4f}" for value in history))
