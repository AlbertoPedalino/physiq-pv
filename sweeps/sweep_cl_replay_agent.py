"""W&B sweep agent for replay-based continual adaptation."""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import wandb


def sweep_main() -> None:
    with wandb.init() as run:
        cfg = dict(run.config)

        window_days = cfg.get("window_days")

        argv = [
            "--data-mode", "real",
            "--initial-train-start", "2019-03-01",
            "--initial-train-end", "2019-05-31",
            "--replay-buffer-size", str(cfg.get("replay_buffer_size", 5000)),
            "--replay-batch-size", str(cfg.get("replay_batch_size", 8)),
            "--replay-loss-weight", str(cfg.get("replay_loss_weight", 1.0)),
            "--replay-peak-fraction", str(cfg.get("replay_peak_fraction", 0.0)),
            "--replay-over-100-fraction", str(cfg.get("replay_over_100_fraction", 0.0)),
            "--replay-peak-threshold", str(cfg.get("replay_peak_threshold", 0.6)),
            "--replay-over-100-threshold", str(cfg.get("replay_over_100_threshold", 1.0)),
            "--initial-epochs", str(cfg.get("initial_epochs", 10)),
            "--update-epochs", str(cfg.get("update_epochs", 1)),
            "--lr", str(cfg.get("lr", 0.001)),
            "--peak-alpha", str(cfg.get("peak_alpha", 2.5)),
            "--peak-gamma", str(cfg.get("peak_gamma", 2.0)),
            "--peak-loss-weight", str(cfg.get("peak_loss_weight", 0.25)),
            "--under-penalty", str(cfg.get("under_penalty", 3.0)),
            "--seed", str(cfg.get("seed", 42)),
            "--run-name", run.name or run.id,
        ]

        if window_days is not None:
            argv += ["--window-days", str(int(window_days))]
        else:
            argv += ["--window-months", str(cfg.get("window_months", 1))]

        sys.argv = ["sweep_cl_replay_agent"] + argv

        from physiq_pv.continual.train_replay_continual import main as cl_main
        cl_main()

        out_dir = os.path.join(
            "outputs", "continual_replay", argv[argv.index("--run-name") + 1],
        )
        summary_path = os.path.join(out_dir, "final_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            run.summary["final_mae"] = summary.get("final_mae")
            run.summary["final_rmse"] = summary.get("final_rmse")
            run.summary["initial_mae"] = summary.get("initial_mae")
            run.summary["n_windows"] = summary.get("n_windows")
            run.summary["replay_buffer_final_size"] = summary.get("replay_buffer_final_size")
            run.summary["peak_alpha"] = summary.get("peak_alpha")
            run.summary["peak_gamma"] = summary.get("peak_gamma")
            run.summary["peak_loss_weight"] = summary.get("peak_loss_weight")
            run.summary["under_penalty"] = summary.get("under_penalty")
            run.summary["replay_peak_fraction"] = summary.get("replay_peak_fraction")
            run.summary["replay_over_100_fraction"] = summary.get("replay_over_100_fraction")
            run.summary["replay_peak_threshold"] = summary.get("replay_peak_threshold")
            run.summary["replay_over_100_threshold"] = summary.get("replay_over_100_threshold")

            bins = summary.get("final_window_bin_metrics", {})
            for label, m in bins.items():
                run.summary[f"bin_{label}_mae"] = m.get("mae")
                run.summary[f"bin_{label}_rmse"] = m.get("rmse")

            bin_summary = summary.get("bin_metrics_across_windows", {})
            for label, m in bin_summary.items():
                for metric in (
                    "mean_mae",
                    "weighted_mean_mae",
                    "worst_mae",
                    "final_mae",
                    "mean_rmse",
                    "weighted_mean_rmse",
                    "worst_rmse",
                    "final_rmse",
                    "total_count",
                    "mean_count",
                    "n_nonempty_windows",
                ):
                    run.summary[f"bin_{label}_{metric}"] = m.get(metric)

        metrics_path = os.path.join(out_dir, "metrics_per_window.csv")
        if os.path.exists(metrics_path):
            import pandas as pd
            df = pd.read_csv(metrics_path)
            for _, row in df.iterrows():
                payload = {
                    "window_id": row["window_id"],
                    "window_mae": row["mae"],
                    "window_rmse": row["rmse"],
                    "window_loss": row["loss"],
                    "replay_buffer_size": row["replay_buffer_size"],
                }
                for col in (
                    "update_recent_peak_loss",
                    "update_replay_peak_loss",
                    "num_replay_peak_samples",
                    "num_replay_over_100_samples",
                    "num_replay_low_samples",
                ):
                    if col in row and pd.notna(row[col]):
                        payload[col] = row[col]
                run.log(payload)


if __name__ == "__main__":
    sweep_main()
