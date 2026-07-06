#!/usr/bin/env python
"""
Post-hoc interval-miss diagnostics for a saved PVGIS-only ST-GNN run.

Reads the predictions.csv written by `--mode pvgis_stgnn` (MC-Dropout run) and
answers, per stratum: how many normal/rare cases fall inside/outside the
predictive interval, by how many WATT the misses fall outside, and whether
coverage could be fixed by widening the band (required_multiplier quantiles +
PICP-vs-k curve) or the predictive center is biased.

Eval-only: no training, no model, no loss change. Anomaly labels stratify only.

Usage:
  PYTHONPATH=$PWD python scripts/analyze_pvgis_interval_miss_distance.py \
      --predictions outputs/wandb_pvgis_stgnn/<run_id>/predictions.csv \
      --out-dir outputs/interval_miss/<run_id>

Outputs in --out-dir:
  interval_miss_distance.csv   one row per stratum (counts, %, watt distances,
                               required_multiplier quantiles)
  picp_curve.csv               PICP of mean +/- k*std per stratum and k
  interval_miss_report.md      both tables + widen-vs-bias verdicts
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from physiq_pv.data.pvgis_stgnn_dataset import (
    DAYTIME_IRRADIANCE_THRESHOLD_WM2,
    attach_anomaly_labels,
    load_anomaly_labels,
)
from scripts.interval_miss_utils import (
    DEFAULT_PICP_MULTIPLIERS,
    INTERVAL_COLUMNS,
    compute_interval_miss_table,
    compute_picp_curve_table,
    render_interval_miss_report,
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-hoc interval-miss diagnostics on saved PVGIS predictions."
    )
    p.add_argument("--predictions", required=True,
                   help="predictions.csv from a pvgis_stgnn MC-Dropout run.")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for the diagnostic CSVs + report.")
    p.add_argument("--interval", default="pi", choices=sorted(INTERVAL_COLUMNS),
                   help="Which saved band to diagnose: pi (primary, MC empirical "
                        "quantiles), gaussian (mean +/- 1.96*std diagnostic).")
    p.add_argument("--eps", type=float, default=1e-6,
                   help="Epsilon in required_multiplier = |err| / (std + eps).")
    p.add_argument("--daytime-threshold", type=float,
                   default=DAYTIME_IRRADIANCE_THRESHOLD_WM2,
                   help="POA W/m2 above which the target timestep is daytime "
                        "(same default as the runner's eval).")
    p.add_argument("--multipliers", default=",".join(str(k) for k in DEFAULT_PICP_MULTIPLIERS),
                   help="Comma-separated k grid for the PICP curve.")
    p.add_argument("--anomaly-scores", default=None,
                   help="Optional anomaly-scores CSV: attach labels if the "
                        "predictions file lacks anomaly_group/anomaly_label.")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        sys.exit(f"Predictions file not found: {pred_path}")
    predictions = pd.read_csv(pred_path, parse_dates=["timestamp"])
    print(f"[load] {pred_path}  rows={len(predictions)}  cols={len(predictions.columns)}")

    if "anomaly_group" not in predictions.columns:
        if not args.anomaly_scores:
            sys.exit(
                "predictions.csv lacks anomaly_group/anomaly_label; pass "
                "--anomaly-scores to attach them (eval-only)."
            )
        predictions = attach_anomaly_labels(
            predictions, load_anomaly_labels(args.anomaly_scores)
        )
        print(f"[labels] attached from {args.anomaly_scores}")

    multipliers = [float(k) for k in args.multipliers.split(",") if k.strip()]

    miss_df = compute_interval_miss_table(
        predictions, interval=args.interval, eps=args.eps,
        threshold_wm2=args.daytime_threshold,
    )
    curve_df = compute_picp_curve_table(
        predictions, multipliers=multipliers,
        threshold_wm2=args.daytime_threshold,
    )
    report = render_interval_miss_report(
        miss_df, curve_df,
        meta={
            "predictions": str(pred_path),
            "interval": args.interval,
            "eps": args.eps,
            "daytime_threshold_wm2": args.daytime_threshold,
            "multipliers": multipliers,
            "n_rows": len(predictions),
        },
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "interval_miss_distance": out_dir / "interval_miss_distance.csv",
        "picp_curve": out_dir / "picp_curve.csv",
        "report": out_dir / "interval_miss_report.md",
    }
    miss_df.to_csv(paths["interval_miss_distance"], index=False)
    curve_df.to_csv(paths["picp_curve"], index=False)
    paths["report"].write_text(report, encoding="utf-8")
    for name, path in paths.items():
        print(f"[write] {name}: {path}")

    # Console summary on the strata that drive the widen-vs-bias question.
    focus = [
        "global", "daytime", "normal_daytime", "rare_extreme_daytime",
        "label:unusually_low_solar_potential",
        "label:unusually_high_solar_potential", "daytime_gt_100",
    ]
    cols = [
        "group", "n", "inside_interval_pct", "above_interval_pct",
        "below_interval_pct", "mean_outside_distance", "p90_outside_distance",
        "p95_required_multiplier",
    ]
    summary = miss_df[miss_df["group"].isin(focus)][cols]
    print("\nFocus strata:")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return paths


if __name__ == "__main__":
    main()
