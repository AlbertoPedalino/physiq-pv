"""
Runtime helpers that bridge MondriaNCP and the streaming protocol.

The calibration set produced by `WalkForwardSplit.calibration_ds` is
forwarded through the (offline-trained) model to collect residuals, then
fed to MondriaNCP for QS-stratified conformal calibration. At stream time,
`conformal_predict_window` produces (pred, lower, upper, ci_width) for each
sample in the current window without retraining CP.
"""
from __future__ import annotations

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

from physiq_pv.data.dataset import PVDataset
from physiq_pv.data.quality_score import compute_qs
from physiq_pv.uncertainty.mondrian_cp import MondriaNCP


def _qs_per_sample(x: torch.Tensor) -> torch.Tensor:
    """Reconstruct aggregate QS = (prod m_i)**0.2 at the last input timestep."""
    m_last = x[..., -1, 5:10].clamp(0.0, 1.0)
    return m_last.prod(dim=-1).pow(0.2)


def _gather_predictions(
    model,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model over loader. Returns flattened (y_true, y_pred, qs) arrays
    across (batch, node).
    """
    model.eval()
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []
    qs_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x, _, y_pv, _ = batch[:4]
            ghi_cs = batch[4] if len(batch) > 4 else None
            x = x.to(device); y_pv = y_pv.to(device)
            ei = edge_index.to(device); ew = edge_weight.to(device)
            ghi_cs_t = ghi_cs.to(device) if ghi_cs is not None else None
            _, pred_pv = model(x, ei, ew, ghi_cs=ghi_cs_t)
            qs = _qs_per_sample(x.cpu()).numpy().ravel()
            y_true_chunks.append(y_pv.detach().cpu().numpy().ravel())
            y_pred_chunks.append(pred_pv.detach().cpu().numpy().ravel())
            qs_chunks.append(qs)
    model.train()
    if not y_true_chunks:
        return np.array([]), np.array([]), np.array([])
    return (
        np.concatenate(y_true_chunks),
        np.concatenate(y_pred_chunks),
        np.concatenate(qs_chunks),
    )


def calibrate_conformal(
    ds_calib: xr.Dataset,
    model,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    seq_len: int = 120,
    batch_size: int = 16,
    n_bins: int = 3,
    confidence_level: float = 0.9,
    device: str = "cpu",
    max_batches: int = 200,
) -> MondriaNCP:
    """
    Fit a QS-stratified MondriaNCP on the calibration period.

    Returns a calibrated MondriaNCP ready to be passed to run_online.
    """
    qs_calib, m_components = compute_qs(ds_calib, debug=True)
    dataset = PVDataset(ds_calib, m_components, seq_len=seq_len)
    if len(dataset) == 0:
        raise RuntimeError("calibration window produced no PVDataset samples")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []
    qs_chunks: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x, _, y_pv, _ = batch[:4]
            ghi_cs = batch[4] if len(batch) > 4 else None
            x = x.to(device); y_pv = y_pv.to(device)
            ei = edge_index.to(device); ew = edge_weight.to(device)
            ghi_cs_t = ghi_cs.to(device) if ghi_cs is not None else None
            _, pred_pv = model(x, ei, ew, ghi_cs=ghi_cs_t)
            qs = _qs_per_sample(x.cpu()).numpy().ravel()
            y_true_chunks.append(y_pv.detach().cpu().numpy().ravel())
            y_pred_chunks.append(pred_pv.detach().cpu().numpy().ravel())
            qs_chunks.append(qs)
    model.train()

    if not y_true_chunks:
        raise RuntimeError("no batches processed during conformal calibration")

    y_true = np.concatenate(y_true_chunks)
    y_pred = np.concatenate(y_pred_chunks)
    qs_cal = np.concatenate(qs_chunks)

    cp = MondriaNCP(n_bins=n_bins, confidence_level=confidence_level)
    cp.calibrate(y_true, y_pred, qs_cal)
    return cp


def conformal_predict_window(
    model,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    cp: MondriaNCP,
    device: str = "cpu",
    max_batches: int = 200,
) -> dict:
    """
    Run model+CP on a window's loader. Returns aggregated stats:
      mean_ci_width, coverage_estimate, n_samples.
    """
    y_true, y_pred, qs = _gather_predictions(
        model, loader, edge_index, edge_weight, device,
    )
    if y_pred.size == 0:
        return {"mean_ci_width": float("nan"), "coverage": float("nan"), "n": 0}

    lo, hi = cp.predict_interval(y_pred, qs)
    width = np.mean(hi - lo)
    covered = ((y_true >= lo) & (y_true <= hi)).mean()
    return {
        "mean_ci_width": float(width),
        "coverage": float(covered),
        "n": int(y_pred.size),
        "lower": lo,
        "upper": hi,
        "pred": y_pred,
        "qs": qs,
    }
