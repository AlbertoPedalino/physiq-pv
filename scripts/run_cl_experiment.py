"""
End-to-end ATSF-inspired continual-learning experiment.

Pipeline:
    1. Load PV dataset (synthetic by default; --real swaps in Piedmont loader).
    2. WalkForwardSplit: train | calibration | holdout | stream tasks.
    3. Offline training on train_ds (via train.train).
    4. Mondrian-CP calibration on calibration_ds.
    5. Streaming loop: for each task, run PhysiQAgent.step + optional DER++
       update + reflection + rollback. Record R[i, j] in CLMetricsTracker.
    6. Holdout retention evaluated after each task.
    7. Dump report.json / online_events.jsonl / cl_metrics.json /
       learning_curve.json / holdout_retention.json.

This script is the canonical entry point for ATSF-inspired CL evaluation.
It reuses existing modules — does not re-implement training, conformal
calibration, agentic diagnosis or the DER++ updater.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader

# Make repo root importable when launched as `python scripts/run_cl_experiment.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physiq_pv.agent.action_policy import UtilityActionPolicy
from physiq_pv.agent.cycle import PhysiQAgent
from physiq_pv.continual.quality_gated_update import QualityGatedUpdater
from physiq_pv.continual.replay_buffer import ReplayBuffer
from physiq_pv.data.dataset import PVDataset, N_FEATURES, SEQ_LEN
from physiq_pv.data.quality_score import compute_qs
from physiq_pv.data.synthetic_generator import generate_synthetic_dataset
from physiq_pv.eval.cl_metrics import CLMetricsTracker
from physiq_pv.eval.streaming_protocol import WalkForwardSplit
from physiq_pv.model.physics_loss import physics_loss_full
from physiq_pv.uncertainty.conformal_runtime import (
    calibrate_conformal,
    conformal_predict_window,
    maybe_recalibrate_cp,
)

from online_loop import (
    _build_loader,
    _eval_loss,
    _qs_from_features,
    _retrain_window,
    _jsonl_safe,
)
from train import train as offline_train


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _eval_mae_window(
    model,
    ds_window: xr.Dataset,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: str,
    seq_len: int,
    batch_size: int,
    max_batches: int = 40,
) -> dict:
    """MAE / RMSE on a window, daytime samples only (PV head)."""
    qs, m_components = compute_qs(ds_window, debug=True)
    dataset = PVDataset(ds_window, m_components, seq_len=seq_len)
    if len(dataset) == 0:
        return {"mae_pv": float("nan"), "rmse_pv": float("nan"), "n": 0}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model.eval()
    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    ei = edge_index.to(device)
    ew = edge_weight.to(device)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x, _y_ghi, y_pv, _eta = batch[:4]
            ghi_cs = batch[4] if len(batch) > 4 else None
            x = x.to(device)
            y_pv = y_pv.to(device)
            ghi_cs_t = ghi_cs.to(device) if ghi_cs is not None else None
            _pg, pred_pv = model(x, ei, ew, ghi_cs=ghi_cs_t)
            preds.append(pred_pv.detach().cpu().numpy().ravel())
            truths.append(y_pv.detach().cpu().numpy().ravel())
    model.train()

    if not preds:
        return {"mae_pv": float("nan"), "rmse_pv": float("nan"), "n": 0}
    p = np.concatenate(preds)
    t = np.concatenate(truths)
    err = p - t
    return {
        "mae_pv": float(np.mean(np.abs(err))),
        "rmse_pv": float(np.sqrt(np.mean(err ** 2))),
        "n": int(p.size),
    }


def _snapshot_state(model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _restore_state(model, snapshot: dict) -> None:
    model.load_state_dict(snapshot)


def _safe_compute_qs(ds_window: xr.Dataset):
    return compute_qs(ds_window, debug=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="ATSF-inspired CL experiment")
    parser.add_argument("--run_name", type=str, default=f"cl_run_{int(time.time())}")
    parser.add_argument("--output_dir", type=str, default="outputs/cl_experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3,
                        help="Offline training epochs (small for dry-runs)")
    parser.add_argument("--data", type=str, choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--real_train_end", type=str, default="2019-08-31",
                        help="Train/calibration boundary (real data only)")
    parser.add_argument("--calib_months", type=float, default=0.5)
    parser.add_argument("--holdout_months", type=float, default=0.5)
    parser.add_argument("--stream_window", type=int, default=720,
                        help="Hours per streaming task window (default 30 days)")
    parser.add_argument("--stream_stride", type=int, default=168,
                        help="Hours between consecutive task starts (default 1 week)")
    parser.add_argument("--max_tasks", type=int, default=6,
                        help="Cap the number of streaming tasks (keeps R-matrix tractable)")
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--n_retrain_batches", type=int, default=10)
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B logging during offline training")
    parser.add_argument("--cp_n_bins", type=int, default=3)
    parser.add_argument("--cp_confidence", type=float, default=0.9)
    parser.add_argument(
        "--cp_recalibrate_every", type=int, default=0,
        help="Recalibrate CP every N stream tasks (0 = off, current default). "
             "Currently a no-op hook; adaptive CP is future work (G10).",
    )
    parser.add_argument(
        "--force_update", action="store_true",
        help="Override policy and force `trigger_update` on every task. "
             "Useful for smoke-testing the DER++ + rollback path on synthetic data "
             "where high QS would otherwise prevent any update from firing.",
    )
    args = parser.parse_args()

    _set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "online_events.jsonl"
    events_fp = open(events_path, "w", encoding="utf-8")

    def log_event(event: dict) -> None:
        events_fp.write(json.dumps(_jsonl_safe(event)) + "\n")
        events_fp.flush()

    print(f"[init] run={args.run_name}  device={device}  out={out_dir}")

    # ------------------------------------------------------------------ #
    # 1. Data loading (Perception)
    # ------------------------------------------------------------------ #
    if args.data == "synthetic":
        print("[data] generating synthetic dataset (20 plants x 8760 h)")
        ds = generate_synthetic_dataset(seed=args.seed)
        # Synthetic generator does not emit eta_base; backfill a constant for
        # PVDataset's normaliser. Use 0.18 as a reasonable c-Si mid value.
        if "eta_base" not in ds.data_vars and "eta_base" not in ds.coords:
            ds = ds.assign_coords(
                eta_base=("plant", np.full(ds.sizes["plant"], 0.18, dtype=np.float64))
            )
        # WalkForwardSplit needs a train_end that lies inside the dataset.
        # For synthetic we slice at 70% / 80% / 90% of the time axis.
        times = pd.DatetimeIndex(ds.coords["time"].values)
        train_end_ts = times[int(len(times) * 0.70)]
        train_end_str = str(train_end_ts.date())
        calib_months = 0.5
        holdout_months = 0.5
    else:
        # Real Piedmont data path. Reuses main.py's loaders.
        from physiq_pv.data.sentinel_hourly_loader import (
            load_sentinel_hourly,
            merge_with_weather,
        )
        from main import _normalize_dataset
        print("[data] loading real Piedmont 2019 dataset")
        ds = load_sentinel_hourly(
            sentinel_dir="/data/SentinelPV/energy_data/piemonte_energy_data/single_ups",
            year=2019,
            plant_mapping_path="data/plant_mapping.csv",
            energy_coords_path="data/energy_with_coordinates.csv",
        )
        ds = merge_with_weather(ds, pvgis_path="data/piedmont_pvgis_2019.nc")
        ds = _normalize_dataset(ds)
        train_end_str = args.real_train_end
        calib_months = args.calib_months
        holdout_months = args.holdout_months

    print(f"[data] n_plants={ds.sizes['plant']}  T={ds.sizes['time']}h")

    # ------------------------------------------------------------------ #
    # 2. Walk-forward split
    # ------------------------------------------------------------------ #
    split = WalkForwardSplit(
        ds,
        train_end=train_end_str,
        calibration_months=calib_months,
        holdout_months=holdout_months,
        stream_stride_hours=args.stream_stride,
        stream_window_hours=args.stream_window,
    )
    split_summary = split.summary()
    print(
        f"[split] train={split_summary['train_hours']}h  "
        f"calib={split_summary['calib_hours']}h  "
        f"holdout={split_summary['holdout_hours']}h  "
        f"stream={split_summary['stream_hours']}h  "
        f"n_tasks={split_summary['n_stream_tasks']}"
    )

    # ------------------------------------------------------------------ #
    # 3. Offline training (Action — initial)
    # ------------------------------------------------------------------ #
    print(f"[train] offline training on train_ds for {args.epochs} epochs")
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model, train_loss, val_loss, _updater_unused, edge_index, edge_weight, pv_calibration = offline_train(
        ds=split.train_ds,
        n_epochs=args.epochs,
        lam=args.lam,
        max_steps_per_epoch=None,
        kwp=None,
        early_stopping_patience=None,
        peak_alpha=2.0,
        peak_gamma=2.0,
        peak_loss_weight=0.25,
        under_penalty=2.0,
        calibration_kpi="none",
        seq_len=args.seq_len,
        use_wandb=args.wandb,
        wandb_run_name=f"{args.run_name}_offline",
        wandb_tags=["cl_experiment", "offline"],
        checkpoint_dir=str(ckpt_dir),
        seed=args.seed,
    )
    torch.save(model.state_dict(), ckpt_dir / "initial.pt")
    print(f"[train] final train_loss={train_loss[-1]:.4f}  val_loss={val_loss[-1]:.4f}")

    # Rebuild updater with deterministic-rng buffer so online updates are
    # reproducible. We discard the buffer from train.train (legacy collection).
    rng_buffer = np.random.default_rng(args.seed)
    buffer = ReplayBuffer(capacity=2000, rng=rng_buffer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
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

    # ------------------------------------------------------------------ #
    # 4. Conformal calibration (Reflection — uncertainty)
    # ------------------------------------------------------------------ #
    cp = None
    cp_calibration_report: dict = {}
    try:
        print("[cp] calibrating Mondrian CP on calibration_ds")
        cp = calibrate_conformal(
            split.calibration_ds,
            model,
            edge_index,
            edge_weight,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            n_bins=args.cp_n_bins,
            confidence_level=args.cp_confidence,
            device=device,
        )
        cp_calibration_report = cp.coverage_report()
        print(f"[cp] bands={cp_calibration_report}")
    except Exception as exc:
        print(f"[cp] WARNING calibration failed: {exc} -> continuing without CP")
        cp = None

    # ------------------------------------------------------------------ #
    # 5. Agent + tracker
    # ------------------------------------------------------------------ #
    agent = PhysiQAgent(
        n_clusters=min(4, max(2, ds.sizes["plant"] // 5)),
        drift_window=min(args.stream_window, 720),
        action_policy=UtilityActionPolicy(),
        seed=args.seed,
    )

    # Train causal classifier from synthetic labels — only meaningful when
    # data == synthetic. Real data has no oracle labels; classifier falls back
    # to its rule-based heuristic.
    if args.data == "synthetic":
        try:
            stats = agent.train_classifier(split.train_ds)
            print(f"[agent] causal classifier trained: {stats}")
        except Exception as exc:
            print(f"[agent] classifier training skipped: {exc}")

    # Cap n_tasks to keep R-matrix tractable; tracker needs >= 2 tasks.
    tasks = list(split.stream_tasks())
    if args.max_tasks is not None and args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    n_tasks = len(tasks)
    if n_tasks < 2:
        print(f"[stream] not enough tasks (n={n_tasks}); aborting")
        events_fp.close()
        return
    print(f"[stream] running {n_tasks} task windows")

    tracker = CLMetricsTracker(n_tasks=n_tasks, higher_is_better=False)

    # ------------------------------------------------------------------ #
    # 6. Pre-stream holdout eval (reference R[-1, holdout])
    # ------------------------------------------------------------------ #
    print("[holdout] pre-stream eval on holdout_ds")
    holdout_pre = _eval_mae_window(
        model, split.holdout_ds, edge_index, edge_weight, device,
        args.seq_len, args.batch_size,
    )
    print(f"[holdout] mae_pv={holdout_pre['mae_pv']:.4f}  n={holdout_pre['n']}")
    holdout_retention: list[dict] = [{"task_id": -1, **holdout_pre}]

    # Pre-stream task eval to bootstrap R diagonal upper bound (optional).

    # ------------------------------------------------------------------ #
    # 7. Stream loop (Planning + Action + Reflection)
    # ------------------------------------------------------------------ #
    n_triggers = 0
    n_rollbacks = 0
    for i, (slice_meta, ds_window) in enumerate(tasks):
        ts_start = pd.Timestamp(slice_meta.timestamp_start)
        ts_end = pd.Timestamp(slice_meta.timestamp_end)

        # Perception
        try:
            qs_window, m_components = _safe_compute_qs(ds_window)
        except Exception as exc:
            print(f"[task {i}] perception failed: {exc}")
            log_event({"task_id": i, "error": f"perception:{exc}"})
            continue

        # Optional CP snapshot to feed action policy with ci_width
        ci_width = None
        if cp is not None:
            try:
                loader_cp = _build_loader(
                    ds_window, m_components,
                    seq_len=args.seq_len, batch_size=args.batch_size,
                )
                if loader_cp is not None and len(loader_cp) > 0:
                    cp_stats = conformal_predict_window(
                        model, loader_cp, edge_index, edge_weight, cp,
                        device=device, max_batches=10,
                    )
                    ci_width = cp_stats.get("mean_ci_width")
            except Exception as exc:
                print(f"[task {i}] CP snapshot failed: {exc}")
                ci_width = None

        # Planning + Action decision
        report = agent.step(
            ds_window, updater=updater,
            qs=qs_window, m_components=m_components, ci_width=ci_width,
        )
        report["task_id"] = i
        report["t_start"] = str(ts_start)
        report["t_end"] = str(ts_end)
        if ci_width is not None:
            report["ci_width"] = ci_width

        # Optional override: --force_update flips the action to
        # "retrain_triggered" regardless of policy choice. Smoke-tests the
        # DER++/rollback path on synthetic data where the policy otherwise
        # never fires an update. The original policy action is preserved
        # under "policy_action" / "policy_distribution" for the log.
        if args.force_update:
            report["action"] = "retrain_triggered"
            report["forced_update"] = True

        # Optional adaptive-CP hook (currently a documented no-op, G10).
        cp = maybe_recalibrate_cp(cp, every=args.cp_recalibrate_every, task_id=i)

        # Action: optional retrain
        loss_before = loss_after = None
        rolled_back = False
        n_updated = 0
        if report.get("action") == "retrain_triggered":
            n_triggers += 1
            loader = _build_loader(
                ds_window, m_components,
                seq_len=args.seq_len, batch_size=args.batch_size,
            )
            if loader is not None and len(loader) > 0:
                ckpt_before = _snapshot_state(model)
                susp = report.get("forensic_summary", {}).get("mean_suspicion")
                if susp is not None and (isinstance(susp, float) and susp != susp):
                    susp = None
                loss_before = _eval_loss(
                    model, loader, edge_index, edge_weight,
                    args.lam, device,
                )
                n_updated = _retrain_window(
                    model, loader, updater, edge_index, edge_weight,
                    args.lam, device,
                    n_batches=args.n_retrain_batches,
                    suspicion_mean=susp,
                )
                loss_after = _eval_loss(
                    model, loader, edge_index, edge_weight,
                    args.lam, device,
                )
                agent.reflect(loss_before, loss_after, report)
                if report.get("reflection", {}).get("verdict") == "degraded":
                    _restore_state(model, ckpt_before)
                    rolled_back = True
                    n_rollbacks += 1

        # Reflection: eval current model on every task -> fills R[i, j]
        for j, (_meta_j, ds_j) in enumerate(tasks):
            m_j = _eval_mae_window(
                model, ds_j, edge_index, edge_weight, device,
                args.seq_len, args.batch_size,
            )
            tracker.record(i, j, m_j["mae_pv"])

        # Holdout retention after this task
        holdout_now = _eval_mae_window(
            model, split.holdout_ds, edge_index, edge_weight, device,
            args.seq_len, args.batch_size,
        )
        holdout_retention.append({"task_id": i, **holdout_now})

        event = {
            "task_id": i,
            "t_start": str(ts_start),
            "t_end": str(ts_end),
            "fleet_mean_qs": report.get("fleet_mean_qs"),
            "n_drifting": report.get("n_drifting"),
            "forensic_summary": report.get("forensic_summary"),
            "action": report.get("action"),
            "policy_action": report.get("policy_action"),
            "policy_distribution": report.get("policy_distribution"),
            "policy_selection": report.get("policy_selection"),
            "ci_width": ci_width,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "reflection": report.get("reflection"),
            "rolled_back": rolled_back,
            "n_batches_updated": n_updated,
            "buffer_size": len(buffer),
            "task_mae_pv": tracker.matrix[i, i],
            "holdout_mae_pv": holdout_now["mae_pv"],
        }
        log_event(event)
        print(
            f"  task {i:>2}/{n_tasks-1}  qs={event['fleet_mean_qs']:.3f}  "
            f"drift={event['n_drifting']:>3}  act={event['action']:<18} "
            f"task_mae={event['task_mae_pv']:.4f}  hold_mae={event['holdout_mae_pv']:.4f}"
            f"  buf={event['buffer_size']}{'  ROLLBACK' if rolled_back else ''}"
        )

    events_fp.close()

    # ------------------------------------------------------------------ #
    # 8. Final report
    # ------------------------------------------------------------------ #
    cl_report = tracker.compute(strict=False)

    learning_curve = cl_report.learning_curve
    pd.DataFrame({
        "task_id": list(range(n_tasks)),
        "mae_pv_diag": learning_curve,
    }).to_csv(out_dir / "learning_curve.csv", index=False)
    pd.DataFrame(holdout_retention).to_csv(out_dir / "holdout_retention.csv", index=False)

    with open(out_dir / "cl_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_jsonl_safe({
            "n_tasks": cl_report.n_tasks,
            "higher_is_better": cl_report.higher_is_better,
            "backward_transfer": cl_report.bwt,
            "forward_transfer": cl_report.fwt,
            "average_forgetting": cl_report.avg_forgetting,
            "learning_curve": cl_report.learning_curve,
            "R_matrix": cl_report.matrix.tolist(),
        }), f, indent=2)

    torch.save(model.state_dict(), ckpt_dir / "final.pt")

    report = {
        "run_name": args.run_name,
        "seed": args.seed,
        "device": device,
        "config": vars(args),
        "data": {
            "source": args.data,
            "n_plants": int(ds.sizes["plant"]),
            "n_timesteps": int(ds.sizes["time"]),
        },
        "split": _jsonl_safe(split_summary),
        "offline_training": {
            "epochs": args.epochs,
            "final_train_loss": float(train_loss[-1]) if train_loss else None,
            "final_val_loss": float(val_loss[-1]) if val_loss else None,
            "best_val_epoch": pv_calibration.get("best_val_epoch", None),
            "checkpoint_initial": str(ckpt_dir / "initial.pt"),
            "checkpoint_final": str(ckpt_dir / "final.pt"),
        },
        "conformal": {
            "calibrated": cp is not None,
            "n_bins": args.cp_n_bins,
            "confidence_level": args.cp_confidence,
            "coverage_report": _jsonl_safe(cp_calibration_report),
        },
        "stream": {
            "n_tasks": n_tasks,
            "n_triggers": n_triggers,
            "n_rollbacks": n_rollbacks,
            "buffer_capacity": buffer.capacity,
            "final_buffer_size": len(buffer),
        },
        "cl_metrics": {
            "backward_transfer": cl_report.bwt,
            "forward_transfer": cl_report.fwt,
            "average_forgetting": cl_report.avg_forgetting,
            "learning_curve": cl_report.learning_curve,
        },
        "holdout": {
            "pre_stream_mae_pv": holdout_pre["mae_pv"],
            "post_stream_mae_pv": holdout_retention[-1]["mae_pv"] if holdout_retention else None,
            "retention_delta": (
                holdout_retention[-1]["mae_pv"] - holdout_pre["mae_pv"]
                if holdout_retention and holdout_retention[-1].get("mae_pv") is not None
                else None
            ),
        },
        "limitations": [
            "ATSF-inspired workflow, not a full autonomous agent and not LLM-driven.",
            "No reinforcement learning: UtilityActionPolicy weights are hand-tuned.",
            "Conformal predictor is calibrated once on calibration_ds and not "
            "re-calibrated online (G10 — adaptive CP is future work).",
            "DER++ replay protects only the PV head; the GHI head is not "
            "distilled (G6 — dual-head replay is future work).",
            "Causal classifier supervised only on synthetic fault labels; on "
            "real data it falls back to rule-based heuristics (G2).",
            "Drift detector is a KS two-sample approximation of ADWIN, not "
            "the true incremental algorithm (G7).",
            "Six of the nine policy actions are 'logged-only' and have no "
            "consumer downstream (alert_*, preserve_replay).",
            "Per-plant drift state (KSDriftMonitor._buf) is not serialised; "
            "restarting the orchestrator resets drift memory.",
            "R-matrix evaluation is O(n_tasks^2); bounded via --max_tasks.",
            "Inference is embedded in _eval_loss / _retrain_window; no "
            "dedicated inference service module yet.",
        ],
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(_jsonl_safe(report), f, indent=2)

    print("\n[done]")
    print(f"  BWT  = {cl_report.bwt:.4f}")
    print(f"  FWT  = {cl_report.fwt:.4f}")
    print(f"  AF   = {cl_report.avg_forgetting:.4f}")
    print(f"  triggers={n_triggers}  rollbacks={n_rollbacks}")
    print(f"  outputs -> {out_dir}/")


if __name__ == "__main__":
    main()
