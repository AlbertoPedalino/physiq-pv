"""
Run the PVGIS-only ST-GNN forecasting pipeline.

Reuses the existing STGNN architecture on a real PVGIS-only dataset (11 features,
no ENERGIA / no QS / no plant data), trains on the train years, evaluates on the
test year, and reports metrics stratified by the climatology anomaly labels.

Example (server):

    PYTHONPATH=$PWD python scripts/run_pvgis_stgnn_forecasting.py \\
      --pvgis-dir /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance \\
      --train-years 2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018 \\
      --test-year 2019 \\
      --anomaly-scores outputs/pvgis_anomaly_2019_2005_2023_w15_q099/pvgis_climatology_scores.csv \\
      --out-dir outputs/pvgis_stgnn_forecasting_2019 \\
      --seq-len 24 \\
      --horizon 1 \\
      --target-variable pv_power_output \\
      --epochs 10 \\
      --batch-size 8
"""

from __future__ import annotations

import argparse

import torch

from physiq_pv.data.pvgis_stgnn_dataset import (
    DEFAULT_TARGET_VARIABLE,
    attach_anomaly_labels,
    build_datasets,
    build_meta,
    compute_metrics,
    load_anomaly_labels,
    load_pvgis_year,
    load_pvgis_years,
    make_model,
    predict,
    train_model,
    write_outputs,
)
from physiq_pv.model.graph_builder import build_graph


def _parse_years(text: str) -> list[int]:
    return [int(y) for y in text.split(",") if y.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PVGIS-only ST-GNN forecasting with stratified eval.")
    p.add_argument("--pvgis-dir", required=True)
    p.add_argument("--train-years", required=True, help="Comma-separated train years.")
    p.add_argument("--test-year", type=int, required=True)
    p.add_argument("--anomaly-scores", default=None, help="pvgis_climatology_scores.csv (optional).")
    p.add_argument("--out-dir", default="outputs/pvgis_stgnn_forecasting")
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--target-variable", default=DEFAULT_TARGET_VARIABLE)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-dist-km", type=float, default=20.0)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-test-samples", type=int, default=None)
    p.add_argument("--file-template", default="piedmont_pvgis_{year}.nc")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    train_years = _parse_years(args.train_years)

    print(f"[1/6] Loading PVGIS years (train={train_years}, test={args.test_year})")
    train_map = load_pvgis_years(args.pvgis_dir, train_years, file_template=args.file_template)
    test_path = f"{args.pvgis_dir}/{args.file_template.format(year=args.test_year)}"
    test_ds = load_pvgis_year(test_path)

    print("[2/6] Building PVGIS-only datasets (features, normalisation, windows)")
    built = build_datasets(train_map, test_ds, args.seq_len, args.horizon, args.target_variable)
    built["train"].subsample(args.max_train_samples, seed=args.seed)
    built["test"].subsample(args.max_test_samples, seed=args.seed)
    print(
        f"      nodes={len(built['loc_ids'])}  n_features={built['n_features']}  "
        f"train_windows={len(built['train'])}  test_windows={len(built['test'])}"
    )

    print(f"[3/6] Building graph (max_dist_km={args.max_dist_km})")
    edge_index, edge_weight = build_graph(built["lats"], built["lons"], max_dist_km=args.max_dist_km)
    print(f"      edges={edge_index.shape[1]}")

    print(f"[4/6] Training STGNN (n_features={built['n_features']}, epochs={args.epochs}, device={args.device})")
    model = make_model(len(built["loc_ids"]), args.seq_len, built["n_features"])
    model = train_model(
        model, built["train"], edge_index, edge_weight,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device,
    )

    print("[5/6] Predicting on test year + attaching anomaly labels")
    predictions = predict(model, built["test"], edge_index, edge_weight, args.device, args.batch_size)
    anomaly_scores = load_anomaly_labels(args.anomaly_scores)
    predictions = attach_anomaly_labels(predictions, anomaly_scores)
    global_df, by_df = compute_metrics(predictions)

    print(f"[6/6] Writing outputs to {args.out_dir}")
    meta = build_meta(
        {
            "target_variable": args.target_variable,
            "seq_len": args.seq_len,
            "horizon": args.horizon,
            "train_years": args.train_years,
            "test_year": args.test_year,
            "epochs": args.epochs,
            "anomaly_scores": args.anomaly_scores,
            "device": args.device,
        },
        n_predictions=len(predictions),
        n_nodes=len(built["loc_ids"]),
    )
    paths = write_outputs(predictions, global_df, by_df, args.out_dir, meta)

    print("\nDone.")
    print(global_df.to_string(index=False))
    if not by_df.empty:
        print(by_df.to_string(index=False))
    for key in ("predictions", "metrics_global", "metrics_by_anomaly_label", "report"):
        print(f"  {key:24s} -> {paths[key]}")

    test_ds.close()
    for ds in train_map.values():
        ds.close()


if __name__ == "__main__":
    main()
