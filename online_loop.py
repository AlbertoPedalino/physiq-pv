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

def _qs_from_features(x: torch.Tensor) -> torch.Tensor:
    """Recompute QS scalar per (batch, node) from m_components in features.

    Features layout: x[..., 5:10] = m1, m2, m3, m4, m5 (clipped in [0, 1]).
    QS = (prod m_i) ** 0.2 evaluated at the last timestep of the input window.
    """
    m_last = x[..., -1, 5:10].clamp(0.0, 1.0)
    return m_last.prod(dim=-1).pow(0.2)


def _build_loader(
    ds_window: xr.Dataset,
    m_components: dict,
    seq_len: int = _SEQ_LEN,
    batch_size: int = _BATCH,
) -> DataLoader | None:
    dataset = PVDataset(ds_window, m_components, seq_len=seq_len)
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
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x, y_ghi, y_pv, eta = batch[:4]
            ghi_cs = batch[4] if len(batch) > 4 else None
            x     = x.to(device)
            y_ghi = y_ghi.to(device)
            y_pv  = y_pv.to(device)
            eta   = eta.to(device)
            ei    = edge_index.to(device)
            ew    = edge_weight.to(device)
            ghi_cs_t = ghi_cs.to(device) if ghi_cs is not None else None
            pred_ghi, pred_pv = model(x, ei, ew, ghi_cs=ghi_cs_t)
            loss, _ = physics_loss_full(pred_ghi, pred_pv, y_ghi, y_pv, eta, lam=lam)
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
    suspicion_mean: float | None = None,
) -> int:
    """Run up to n_batches DER++ updates. Returns number of batches that updated weights."""
    model.train()
    n_updated = 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        x, y_ghi, y_pv, eta = batch[:4]
        ghi_cs = batch[4] if len(batch) > 4 else None
        x     = x.to(device)
        y_ghi = y_ghi.to(device)
        y_pv  = y_pv.to(device)
        eta   = eta.to(device)
        ei    = edge_index.to(device)
        ew    = edge_weight.to(device)
        ghi_cs_t = ghi_cs.to(device) if ghi_cs is not None else None
        pred_ghi, pred_pv = model(x, ei, ew, ghi_cs=ghi_cs_t)
        loss, _ = physics_loss_full(pred_ghi, pred_pv, y_ghi, y_pv, eta, lam=lam)
        qs_per_node = _qs_from_features(x)               # (B, N)
        valid_qs = qs_per_node[qs_per_node > 0]
        qs_mean = float(valid_qs.mean().item()) if valid_qs.numel() > 0 else 0.0
        if updater.step(
            x, y_pv, pred_pv.detach(), loss, qs_mean,
            suspicion_mean=suspicion_mean,
            qs_per_sample=qs_per_node.detach(),
        ):
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
    cp=None,
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
        qs_window, m_components = compute_qs(ds_window, debug=True)

        # Conformal Prediction snapshot first so action policy can read CI width.
        ci_width = None
        if cp is not None:
            try:
                from physiq_pv.uncertainty.conformal_runtime import conformal_predict_window
                loader_cp = _build_loader(
                    ds_window, m_components, seq_len=seq_len, batch_size=batch_size,
                )
                if loader_cp is not None and len(loader_cp) > 0:
                    cp_stats = conformal_predict_window(
                        model, loader_cp, edge_index, edge_weight, cp,
                        device=device, max_batches=10,
                    )
                    ci_width = cp_stats["mean_ci_width"]
            except Exception as exc:  # CP failure should not kill the loop
                ci_width = float("nan")

        # Perception + Planning + Action decision (pass precomputed QS + m_components + CI)
        report = agent.step(
            ds_window, updater=updater, qs=qs_window, m_components=m_components,
            ci_width=ci_width,
        )
        report["t"] = t
        if ci_width is not None:
            report["ci_width"] = ci_width

        if report["action"] == "retrain_triggered":
            loader = _build_loader(ds_window, m_components, seq_len=seq_len, batch_size=batch_size)

            if loader is not None and len(loader) > 0:
                # snapshot weights before update for rollback-on-degraded
                checkpoint_before = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
                susp = report.get("forensic_summary", {}).get("mean_suspicion")
                if susp is not None and (isinstance(susp, float) and susp != susp):
                    susp = None  # drop NaN

                loss_before = _eval_loss(model, loader, edge_index, edge_weight, lam, device)
                n_upd = _retrain_window(
                    model, loader, updater, edge_index, edge_weight,
                    lam, device, n_batches=n_retrain_batches,
                    suspicion_mean=susp,
                )
                loss_after = _eval_loss(model, loader, edge_index, edge_weight, lam, device)
                report["n_batches_updated"] = n_upd
                report["suspicion_mean"] = susp
                agent.reflect(loss_before, loss_after, report)

                if report.get("reflection", {}).get("verdict") == "degraded":
                    model.load_state_dict(checkpoint_before)
                    report["rolled_back"] = True
                else:
                    report["rolled_back"] = False

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
