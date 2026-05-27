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
            "--initial-epochs", str(cfg.get("initial_epochs", 10)),
            "--update-epochs", str(cfg.get("update_epochs", 1)),
            "--lr", str(cfg.get("lr", 0.001)),
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

            bins = summary.get("final_window_bin_metrics", {})
            for label, m in bins.items():
                run.summary[f"bin_{label}_mae"] = m.get("mae")
                run.summary[f"bin_{label}_rmse"] = m.get("rmse")

        metrics_path = os.path.join(out_dir, "metrics_per_window.csv")
        if os.path.exists(metrics_path):
            import pandas as pd
            df = pd.read_csv(metrics_path)
            for _, row in df.iterrows():
                run.log({
                    "window_id": row["window_id"],
                    "window_mae": row["mae"],
                    "window_rmse": row["rmse"],
                    "window_loss": row["loss"],
                    "replay_buffer_size": row["replay_buffer_size"],
                })


if __name__ == "__main__":
    sweep_main()
