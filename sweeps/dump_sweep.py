"""Dump finished W&B sweep runs to CSV for offline analysis."""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import wandb

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_results.csv")


METRIC_KEYS = [
    "mae_pv", "rmse_pv", "bias_pv", "mae_ghi", "val_loss", "best_val_loss",
    "mae_pv_0_20", "mae_pv_20_40", "mae_pv_40_60",
    "mae_pv_60_80", "mae_pv_80_100", "mae_pv_60_100", "mae_pv_over_100",
    "rmse_pv_60_80", "rmse_pv_80_100", "rmse_pv_60_100",
    "bias_pv_60_80", "bias_pv_80_100", "bias_pv_60_100",
    "checkpoint_dir", "n_epochs_run",
    "initial_mae", "final_mae", "final_rmse", "n_windows",
    "replay_buffer_final_size",
    "bin_60_80_mae", "bin_80_100_mae", "bin_over_100_mae",
    "bin_60_80_mean_mae", "bin_80_100_mean_mae",
    "bin_60_80_weighted_mean_mae", "bin_80_100_weighted_mean_mae",
    "bin_60_80_worst_mae", "bin_80_100_worst_mae",
    "bin_60_80_final_mae", "bin_80_100_final_mae",
    "bin_60_80_total_count", "bin_80_100_total_count",
    "peak_alpha", "peak_gamma", "peak_loss_weight", "under_penalty",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", default=os.environ.get("WANDB_SWEEP_PATH"))
    p.add_argument("--out", default=OUT_CSV)
    p.add_argument("--include-running", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.sweep:
        print("Missing --sweep entity/project/sweep_id", file=sys.stderr)
        sys.exit(2)

    api = wandb.Api()
    runs = api.sweep(args.sweep).runs

    rows = []
    for r in runs:
        if r.state != "finished" and not args.include_running:
            continue
        s = r.summary
        row = {**dict(r.config), "name": r.name, "state": r.state, "run_id": r.id}
        for k in METRIC_KEYS:
            row[k] = s.get(k)
            row[f"best_{k}"] = s.get(f"best_{k}")
        rows.append(row)

    if not rows:
        print("No finished runs found.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)

    peak_cols = [c for c in ("best_mae_pv_60_80", "best_mae_pv_80_100", "best_mae_pv_over_100") if c in df]
    if peak_cols:
        df["best_mae_pv_peak"] = df[peak_cols].mean(axis=1, skipna=True)

    sort_col = None
    for candidate in ("best_val_loss", "final_mae", "bin_60_80_weighted_mean_mae", "best_mae_pv_peak"):
        if candidate in df and pd.to_numeric(df[candidate], errors="coerce").notna().any():
            sort_col = candidate
            break
    if sort_col in df:
        df = df.sort_values(sort_col, na_position="last")
    df.to_csv(args.out, index=False)
    print(f"{len(rows)} runs saved to {args.out}")

    show_cols = [
        "name", "state", "seed", "bilstm_pooling",
        "best_mae_pv_60_80", "best_mae_pv_80_100",
        "best_val_loss", "best_mae_pv", "best_mae_pv_peak",
        "best_mae_pv_0_20", "best_mae_pv_20_40", "best_mae_pv_40_60",
        "best_mae_pv_over_100",
        "bias_pv_60_80", "bias_pv_80_100",
        "window_days", "final_mae", "final_rmse", "initial_mae", "n_windows",
        "bin_60_80_mean_mae", "bin_80_100_mean_mae",
        "bin_60_80_weighted_mean_mae", "bin_80_100_weighted_mean_mae",
        "bin_60_80_worst_mae", "bin_80_100_worst_mae",
        "bin_60_80_final_mae", "bin_80_100_final_mae",
        "seq_len", "lr", "peak_loss_weight", "under_penalty",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    display_sort_col = sort_col or "input order"
    print(f"\nAll runs by {display_sort_col} (lower = better):")
    print(df[show_cols].to_string(index=False))

    print("\nSeed-variance summary (overall):")
    for metric in ("best_val_loss", "best_mae_pv", "best_mae_pv_60_80", "best_mae_pv_80_100", "best_mae_pv_over_100"):
        if metric in df.columns:
            s = pd.to_numeric(df[metric], errors="coerce").dropna()
            if len(s) >= 2:
                cv = s.std() / s.mean() if s.mean() else float("nan")
                print(f"  {metric:<28} n={len(s)}  mean={s.mean():.4f}  std={s.std():.4f}  min={s.min():.4f}  max={s.max():.4f}  cv={cv:.3f}")

    if "bilstm_pooling" in df.columns:
        print("\nPer-pooling summary:")
        for metric in ("best_val_loss", "best_mae_pv", "best_mae_pv_60_80", "best_mae_pv_80_100"):
            if metric in df.columns:
                print(f"  {metric}:")
                grp = df.groupby("bilstm_pooling")[metric].agg(["count", "mean", "std", "min", "max"])
                print(grp.to_string())


if __name__ == "__main__":
    main()
