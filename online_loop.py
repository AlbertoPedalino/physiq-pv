"""
Full online ATSF agentic loop.

Bridges PhysiQAgent (xarray) and the backward-compatible QualityGatedUpdater
class, used here as a quality-weighted updater.

One iteration per stride:
  1. agent.step()        - perception + planning + action decision
  2. _eval_loss()        - loss_before  (only if retraining triggered)
  3. _retrain_window()   - n_batches DER++ updates
  4. _eval_loss()        - loss_after
  5. agent.reflect()     - verdict: improved / stable / degraded
"""
import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

from physiq_pv.data.dataset import PVDataset
from physiq_pv.data.quality_score import compute_qs
from physiq_pv.model.physics_loss import physics_loss_full

_SEQ_LEN   = 120
_BATCH     = 16
_N_BATCHES = 10   # max batches per retrain window (mini-epoch)
_LAM       = 0.1


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _build_loader(
    ds_window: xr.Dataset,
    qs_window,
    seq_len: int = _SEQ_LEN,
    batch_size: int = _BATCH,
) -> DataLoader | None:
    dataset = PVDataset(ds_window, qs_window, seq_len=seq_len)
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
    )


def _eval_loss(
    model,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    lam: float,
    device: str,
    max_batches: int = 20,
) -> float:
    """Forward pass only - no gradient. Returns mean physics loss."""
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for i, (x, y_ghi, y_pv, qs, eta) in enumerate(loader):
            if i >= max_batches:
                break
            x     = x.to(device)
            y_ghi = y_ghi.to(device)
            y_pv  = y_pv.to(device)
            qs    = qs.to(device)
            eta   = eta.to(device)
            ei    = edge_index.to(device)
            ew    = edge_weight.to(device)
            pred_ghi, pred_pv = model(x, ei, ew)
            loss, _ = physics_loss_full(pred_ghi, pred_pv, y_ghi, y_pv, eta, qs, lam=lam)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def _retrain_window(
    model,
    loader: DataLoader,
    updater,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    lam: float,
    device: str,
    n_batches: int = _N_BATCHES,
) -> int:
    """Run up to n_batches DER++ updates. Returns number of batches that updated weights."""
    model.train()
    n_updated = 0
    for i, (x, y_ghi, y_pv, qs, eta) in enumerate(loader):
        if i >= n_batches:
            break
        x     = x.to(device)
        y_ghi = y_ghi.to(device)
        y_pv  = y_pv.to(device)
        qs    = qs.to(device)
        eta   = eta.to(device)
        ei    = edge_index.to(device)
        ew    = edge_weight.to(device)
        pred_ghi, pred_pv = model(x, ei, ew)
        loss, _ = physics_loss_full(pred_ghi, pred_pv, y_ghi, y_pv, eta, qs, lam=lam)
        valid_qs = qs[qs > 0]
        qs_mean = float(valid_qs.mean().item()) if valid_qs.numel() > 0 else 0.0
        if updater.step(x, y_pv, pred_pv.detach(), loss, qs_mean):
            n_updated += 1
    return n_updated


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def run_online(
    ds: xr.Dataset,
    model,
    updater,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    agent,
    window_size: int = 720,
    stride: int = 168,
    seq_len: int = _SEQ_LEN,
    batch_size: int = _BATCH,
    n_retrain_batches: int = _N_BATCHES,
    lam: float = _LAM,
    device: str = "cpu",
    verbose: bool = True,
) -> list[dict]:
    """
    Slide a window of size window_size by stride timesteps over ds.

    At each step:
      perception -> planning -> action decision -> (optional) retrain -> reflection

    Returns list of per-step report dicts.
    """
    T = ds.sizes["time"]
    history: list[dict] = []

    # auto-detect device from model if not specified
    if device == "cpu":
        import torch.nn as nn
        detected = str(next(model.parameters()).device)
        if detected != "cpu":
            device = detected

    for t in range(window_size, T, stride):
        ds_window = ds.isel(time=slice(t - window_size, t))
        qs_window = compute_qs(ds_window)

        # Perception + Planning + Action decision (pass precomputed QS to avoid recomputation)
        report = agent.step(ds_window, updater=updater, qs=qs_window)
        report["t"] = t

        if report["action"] == "retrain_triggered":
            loader = _build_loader(ds_window, qs_window, seq_len=seq_len, batch_size=batch_size)

            if loader is not None and len(loader) > 0:
                loss_before = _eval_loss(model, loader, edge_index, edge_weight, lam, device)
                n_upd = _retrain_window(
                    model, loader, updater, edge_index, edge_weight,
                    lam, device, n_batches=n_retrain_batches,
                )
                loss_after = _eval_loss(model, loader, edge_index, edge_weight, lam, device)
                report["n_batches_updated"] = n_upd
                agent.reflect(loss_before, loss_after, report)

        history.append(report)

        if verbose:
            n_anom = sum(
                1 for d in report["plant_diagnoses"].values()
                if d["cause"] != "normal"
            )
            ref_str = ""
            if "reflection" in report:
                r = report["reflection"]
                ref_str = f"  reflect={r['verdict']}({r['improvement_pct']:+.1f}%)"
            print(
                f"  t={t:5d}  fleet_qs={report['fleet_mean_qs']:.3f}"
                f"  drifting={report['n_drifting']:2d}"
                f"  anomalous={n_anom:2d}"
                f"  action={report['action']}{ref_str}"
            )

    return history
