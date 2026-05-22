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
import json
import numpy as np
import torch
import xarray as xr
from pathlib import Path
from torch.utils.data import DataLoader

from physiq_pv.data.dataset import PVDataset
from physiq_pv.data.quality_score import compute_qs
from physiq_pv.model.physics_loss import physics_loss_full


def _jsonl_safe(obj):
    """Recursively coerce numpy / torch / pandas scalars to JSON-serialisable types."""
    import pandas as _pd
    if isinstance(obj, dict):
        return {str(k): _jsonl_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonl_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (_pd.Timestamp, _pd.Timedelta)):
        return str(obj)
    return obj

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
    """
    Run up to n_batches DER++ updates. Returns number of batches that
    actually updated weights (i.e. passed the updater's quality gate).

    Two distinct quality-related signals flow into the updater. They are
    deliberately kept separate because they serve different purposes:

    1) Memory weighting signal (per-sample Quality Score):
       ``qs_per_node`` is the per-(batch, node) aggregate QS reconstructed
       from feature channels 5..9. It is passed to ``ReplayBuffer`` as
       ``qs_per_sample`` so that cleaner samples are sampled more often
       during replay. This influences MEMORY, not the gate.

    2) Gate signal (fleet-level suspicion from forensics):
       ``suspicion_mean`` is the fleet-averaged QS-forensics suspicion
       in [0, 1]; it drives the Bernoulli gate inside
       ``QualityGatedUpdater`` (probability of skipping the update equals
       suspicion). This influences ACTION (whether to update), not memory.

    The scalar ``qs_mean`` passed positionally is only used by the legacy
    hard threshold ``updater.qs_threshold`` (kept for backward compatibility,
    set to None by default).
    """
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

        # Memory weighting signal: per-(batch, node) Quality Score recovered
        # from the input features. Feeds the QS-weighted replay sampling.
        replay_qs_per_sample = _qs_from_features(x)
        valid_qs = replay_qs_per_sample[replay_qs_per_sample > 0]
        qs_mean_legacy = float(valid_qs.mean().item()) if valid_qs.numel() > 0 else 0.0

        if updater.step(
            x, y_pv, pred_pv.detach(), loss,
            qs_mean=qs_mean_legacy,                     # legacy hard-threshold input
            suspicion_mean=suspicion_mean,              # gate signal (action layer)
            qs_per_sample=replay_qs_per_sample.detach(),  # memory weighting
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
    jsonl_path: "str | Path | None" = None,
) -> list[dict]:
    """
    Slide a window of size window_size by stride timesteps over ds.

    At each step:
      perception -> planning -> action decision -> (optional) retrain -> reflection

    Returns list of per-step report dicts.
    """
    T = ds.sizes["time"]
    history: list[dict] = []

    jsonl_fp = None
    if jsonl_path is not None:
        jsonl_path = Path(jsonl_path)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_fp = open(jsonl_path, "a", encoding="utf-8")

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

        if jsonl_fp is not None:
            # Disambiguate the two quality signals in the log:
            #   - update_gate_signal = suspicion_mean (drives Bernoulli skip)
            #   - replay_weight_signal = "QS per-sample" (drives buffer sampling)
            event = {
                "t": int(t),
                "fleet_mean_qs": report.get("fleet_mean_qs"),
                "n_drifting": report.get("n_drifting"),
                "action": report.get("action"),
                "policy_action": report.get("policy_action"),
                "policy_distribution": report.get("policy_distribution"),
                "ci_width": report.get("ci_width"),
                "suspicion_mean": report.get("suspicion_mean"),
                "update_gate_signal": report.get("suspicion_mean"),  # alias for clarity
                "replay_weight_signal": "qs_per_sample",
                "n_batches_updated": report.get("n_batches_updated"),
                "rolled_back": report.get("rolled_back"),
                "reflection": report.get("reflection"),
                "buffer_size": len(updater.buffer) if updater is not None else None,
            }
            jsonl_fp.write(json.dumps(_jsonl_safe(event)) + "\n")
            jsonl_fp.flush()

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

    if jsonl_fp is not None:
        jsonl_fp.close()

    return history
