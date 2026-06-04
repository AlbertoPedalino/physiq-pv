"""
Run the PVGIS-only forecasting baseline.

Trains/evaluates a simple forecasting baseline using ONLY PVGIS data, then
evaluates it stratified by the rare/extreme labels from the climatology anomaly
pipeline. Separate from train.py / the ST-GNN; no real plant production is used.

Example (server):

    PYTHONPATH=$PWD python scripts/run_pvgis_forecasting_baseline.py \\
      --pvgis-dir /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance \\
      --train-years 2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018 \\
      --test-year 2019 \\
      --anomaly-scores outputs/pvgis_anomaly_2019_2005_2023_w15_q099/pvgis_climatology_scores.csv \\
      --out-dir outputs/pvgis_forecasting_2019 \\
      --seq-len 24 \\
      --horizon 1 \\
      --target-variable pv_power_output \\
      --baseline persistence
"""

from __future__ import annotations

import argparse

from physiq_pv.data.pvgis_forecasting_baseline import (
    DEFAULT_INPUT_VARIABLES,
    DEFAULT_TARGET_VARIABLE,
    attach_anomaly_labels,
    build_meta,
    build_supervised_windows,
    compute_metrics,
    load_anomaly_labels,
    load_pvgis_year,
    load_pvgis_years,
    run_mlp_baseline,
    run_persistence_baseline,
    write_outputs,
)


def _parse_years(text: str) -> list[int]:
    return [int(y) for y in text.split(",") if y.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PVGIS-only forecasting baseline with stratified eval.")
    p.add_argument("--pvgis-dir", required=True, help="Directory with annual PVGIS NetCDF files.")
    p.add_argument("--test-year", type=int, required=True, help="Year to evaluate on.")
    p.add_argument(
        "--train-years",
        default="",
        help="Comma-separated train years (used by --baseline mlp; ignored by persistence).",
    )
    p.add_argument(
        "--anomaly-scores",
        default=None,
        help="pvgis_climatology_scores.csv for stratified evaluation (optional).",
    )
    p.add_argument("--out-dir", default="outputs/pvgis_forecasting")
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--target-variable", default=DEFAULT_TARGET_VARIABLE)
    p.add_argument(
        "--input-variables",
        default=",".join(DEFAULT_INPUT_VARIABLES),
        help="Comma-separated PVGIS input variables (missing ones are ignored).",
    )
    p.add_argument("--baseline", choices=["persistence", "mlp"], default="persistence")
    p.add_argument("--file-template", default="piedmont_pvgis_{year}.nc")
    # MLP-only knobs
    p.add_argument("--mlp-epochs", type=int, default=5)
    p.add_argument("--mlp-hidden", type=int, default=64)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-batch-size", type=int, default=1024)
    p.add_argument("--mlp-max-train-samples", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_vars = [v.strip() for v in args.input_variables.split(",") if v.strip()]

    print(f"[1/5] Loading test year {args.test_year}")
    test_path = f"{args.pvgis_dir}/{args.file_template.format(year=args.test_year)}"
    test_ds = load_pvgis_year(test_path)
    test_windows = build_supervised_windows(
        test_ds, args.seq_len, args.horizon, input_vars, args.target_variable
    )
    used_vars = test_windows["input_variables"]
    print(f"      windows: {len(test_windows['y'])}  |  variables: {', '.join(used_vars)}")

    print(f"[2/5] Running baseline: {args.baseline}")
    if args.baseline == "persistence":
        predictions = run_persistence_baseline(test_windows)
    else:
        train_years = _parse_years(args.train_years)
        if not train_years:
            raise ValueError("--baseline mlp requires --train-years.")
        print(f"      loading {len(train_years)} train years for MLP")
        train_ds = load_pvgis_years(args.pvgis_dir, train_years, file_template=args.file_template)
        # build + concatenate train windows one year at a time
        import numpy as np

        parts = [
            build_supervised_windows(ds, args.seq_len, args.horizon, input_vars, args.target_variable)
            for ds in train_ds.values()
        ]
        train_windows = {
            "X": np.concatenate([w["X"] for w in parts]),
            "y": np.concatenate([w["y"] for w in parts]),
        }
        predictions = run_mlp_baseline(
            train_windows,
            test_windows,
            epochs=args.mlp_epochs,
            hidden=args.mlp_hidden,
            lr=args.mlp_lr,
            batch_size=args.mlp_batch_size,
            max_train_samples=args.mlp_max_train_samples,
            seed=args.seed,
        )
        for ds in train_ds.values():
            ds.close()

    print(f"[3/5] Attaching anomaly labels: {args.anomaly_scores or '(none)'}")
    anomaly_scores = load_anomaly_labels(args.anomaly_scores)
    predictions = attach_anomaly_labels(predictions, anomaly_scores)

    print("[4/5] Computing metrics")
    global_df, by_label_df = compute_metrics(predictions)

    print(f"[5/5] Writing outputs to {args.out_dir}")
    meta = build_meta(
        {
            "baseline": args.baseline,
            "target_variable": args.target_variable,
            "seq_len": args.seq_len,
            "horizon": args.horizon,
            "train_years": args.train_years or None,
            "test_year": args.test_year,
            "anomaly_scores": args.anomaly_scores,
        },
        n_predictions=len(predictions),
        input_variables=used_vars,
    )
    paths = write_outputs(predictions, global_df, by_label_df, args.out_dir, meta)

    print("\nDone.")
    print(global_df.to_string(index=False))
    if not by_label_df.empty:
        print(by_label_df.to_string(index=False))
    for key in ("predictions", "metrics_global", "metrics_by_anomaly_label", "report"):
        print(f"  {key:24s} -> {paths[key]}")

    test_ds.close()


if __name__ == "__main__":
    main()
