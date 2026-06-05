"""
PVGIS-only ST-GNN experiment runner.

Single source of truth for both entrypoints:
  * `python main.py --mode pvgis_stgnn ...`
  * `python scripts/run_pvgis_stgnn_forecasting.py ...` (thin wrapper)

Built for sweep / ablation / future uncertainty work:
  * `--feature-set`   selects a subset of the 11 PVGIS-only features; the model
                      is instantiated with STGNN(n_features=len(selected)).
  * `--model-type`    stgnn (implemented); persistence / mlp are scaffolded but
                      raise a clean "not implemented yet" — never run silently.
  * `--mc-dropout`    Monte Carlo Dropout: model.eval() + reactivate only the
                      nn.Dropout layers + `--mc-samples` forward passes ->
                      y_pred_mean/std and a ~95% band. Adds uncertainty metrics
                      (mean/median/p90 std, coverage_95) per anomaly stratum.
  * `--wandb`         optional, lazily imported; logs namespaced params + metrics
                      (mae/*, rmse/*, ratio/*, uncertainty/*, coverage_95/*).

Hard constraints (PVGIS-only): NO ENERGIA, NO real plant production,
NO Sentinel/SCADA, NO kWp/UPN/load_kwp, NO compute_qs / real QS, NO anomaly
labels as input or target. Anomaly labels are used ONLY for stratified eval.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np
import torch

from physiq_pv.data.pvgis_stgnn_dataset import (
    DEFAULT_TARGET_VARIABLE,
    FEATURE_SETS,
    attach_anomaly_labels,
    build_datasets,
    build_meta,
    build_wandb_metrics,
    compute_metrics,
    load_anomaly_labels,
    load_pvgis_year,
    load_pvgis_years,
    make_model,
    predict,
    predict_mc,
    resolve_feature_set,
    train_model,
    write_outputs,
)
from physiq_pv.model.graph_builder import build_graph

# Model-type registry. Only "stgnn" is implemented; the rest are scaffolded so
# the dispatch is ready, but they fail cleanly instead of running silently.
SUPPORTED_MODEL_TYPES = ("stgnn", "persistence", "mlp")
IMPLEMENTED_MODEL_TYPES = ("stgnn",)


def _parse_years(text: str) -> List[int]:
    return [int(y) for y in text.split(",") if y.strip()]


def add_pvgis_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Register PVGIS-only experiment arguments on `parser`.

    Nothing is marked required at the argparse level (so `python main.py` in
    default mode never trips over them); required args are validated in
    `run_from_args` only when this mode actually runs.
    """
    # Each hyphenated flag also gets an underscore alias (e.g. --train_years) so
    # W&B sweep `${args}` (which emits --param_name=value) works without a wrapper.
    g = parser.add_argument_group("pvgis_stgnn mode")
    g.add_argument("--pvgis-dir", "--pvgis_dir", default=None,
                   help="Dir with piedmont_pvgis_{year}.nc files.")
    g.add_argument("--train-years", "--train_years", default=None,
                   help="Comma-separated train years.")
    g.add_argument("--test-year", "--test_year", type=int, default=None)
    g.add_argument("--anomaly-scores", "--anomaly_scores", default=None,
                   help="pvgis_climatology_scores.csv (stratified eval only; never model input).")
    g.add_argument("--out-dir", "--out_dir", default="outputs/pvgis_stgnn_forecasting")
    g.add_argument("--seq-len", "--seq_len", type=int, default=24)
    g.add_argument("--horizon", type=int, default=1)
    g.add_argument("--target-variable", "--target_variable", default=DEFAULT_TARGET_VARIABLE)
    g.add_argument("--epochs", type=int, default=10)
    g.add_argument("--batch-size", "--batch_size", type=int, default=8)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--max-dist-km", "--max_dist_km", type=float, default=20.0)
    g.add_argument("--max-train-samples", "--max_train_samples", type=int, default=None)
    g.add_argument("--max-test-samples", "--max_test_samples", type=int, default=None)
    g.add_argument("--file-template", "--file_template", default="piedmont_pvgis_{year}.nc")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--seed", type=int, default=42)
    # Model / ablation
    g.add_argument("--model-type", "--model_type", default="stgnn", choices=SUPPORTED_MODEL_TYPES,
                   help="stgnn (implemented); persistence/mlp scaffolded (not implemented yet).")
    g.add_argument("--feature-set", "--feature_set", default="full", choices=sorted(FEATURE_SETS),
                   help="Feature ablation; n_features = len(selected features).")
    g.add_argument("--dropout", type=float, default=0.2,
                   help="STGNN dropout (also the basis for MC Dropout sampling).")
    # MC Dropout — eval() + reactivate only nn.Dropout + N passes -> mean/std.
    g.add_argument("--mc-dropout", "--mc_dropout", action="store_true",
                   help="Enable Monte Carlo Dropout uncertainty (needs --dropout > 0).")
    g.add_argument("--mc-samples", "--mc_samples", type=int, default=30,
                   help="Number of MC Dropout forward passes per batch (>= 2).")
    # Optional W&B
    g.add_argument("--wandb", action="store_true", help="Enable optional W&B logging.")
    g.add_argument("--wandb-project", "--wandb_project", default="PhysiQ-PV")
    g.add_argument("--wandb-run-name", "--wandb_run_name", default=None)
    return parser


def build_arg_parser() -> argparse.ArgumentParser:
    """Standalone parser for the script wrapper."""
    p = argparse.ArgumentParser(
        description="PVGIS-only ST-GNN forecasting with stratified eval, feature "
                    "ablation, optional W&B, and MC-Dropout predisposition."
    )
    return add_pvgis_arguments(p)


def _fail(parser: Optional[argparse.ArgumentParser], msg: str) -> None:
    if parser is not None:
        parser.error(msg)
    raise SystemExit(f"error: {msg}")


def _validate(args: argparse.Namespace, parser: Optional[argparse.ArgumentParser]) -> None:
    missing = [
        name for name, val in (
            ("--pvgis-dir", args.pvgis_dir),
            ("--train-years", args.train_years),
            ("--test-year", args.test_year),
        ) if val in (None, "")
    ]
    if missing:
        _fail(parser, f"--mode pvgis_stgnn requires: {', '.join(missing)}.")
    if args.model_type not in IMPLEMENTED_MODEL_TYPES:
        _fail(
            parser,
            f"model_type='{args.model_type}' is not implemented yet. "
            f"Only {list(IMPLEMENTED_MODEL_TYPES)} is available; "
            "persistence/mlp are scaffolded for future work.",
        )
    if args.mc_dropout:
        if args.mc_samples < 2:
            _fail(parser, f"--mc-samples must be >= 2 for MC Dropout, got {args.mc_samples}.")
        if args.dropout <= 0.0:
            _fail(
                parser,
                f"--mc-dropout needs --dropout > 0 for stochasticity (got {args.dropout}).",
            )


def run_from_args(
    args: argparse.Namespace, parser: Optional[argparse.ArgumentParser] = None
) -> dict:
    """Run the PVGIS-only ST-GNN experiment. Returns the written-output paths."""
    _validate(args, parser)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_years = _parse_years(args.train_years)
    features = resolve_feature_set(args.feature_set)

    # Optional W&B (lazy import; never required).
    wandb_run = None
    if args.wandb:
        import wandb  # noqa: PLC0415 — optional dependency, imported only when enabled
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "mode": "pvgis_stgnn",
                "model_type": args.model_type,
                "feature_set": args.feature_set,
                "selected_features": features,
                "n_features": len(features),
                "target_variable": args.target_variable,
                "train_years": args.train_years,
                "test_year": args.test_year,
                "seq_len": args.seq_len,
                "horizon": args.horizon,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "dropout": args.dropout,
                "device": args.device,
                "max_train_samples": args.max_train_samples,
                "max_test_samples": args.max_test_samples,
                "mc_dropout": args.mc_dropout,
                "mc_samples": args.mc_samples,
                "anomaly_scores": args.anomaly_scores,
            },
        )

    print(
        f"[1/6] Loading PVGIS years (train={train_years}, test={args.test_year}) "
        f"| model={args.model_type} feature_set={args.feature_set} "
        f"n_features={len(features)}"
    )
    train_map = load_pvgis_years(args.pvgis_dir, train_years, file_template=args.file_template)
    test_path = f"{args.pvgis_dir}/{args.file_template.format(year=args.test_year)}"
    test_ds = load_pvgis_year(test_path)

    try:
        print(f"[2/6] Building datasets (features={features})")
        built = build_datasets(
            train_map, test_ds, args.seq_len, args.horizon, args.target_variable,
            feature_names=features,
        )
        built["train"].subsample(args.max_train_samples, seed=args.seed)
        built["test"].subsample(args.max_test_samples, seed=args.seed)
        print(
            f"      nodes={len(built['loc_ids'])}  n_features={built['n_features']}  "
            f"train_windows={len(built['train'])}  test_windows={len(built['test'])}"
        )

        print(f"[3/6] Building graph (max_dist_km={args.max_dist_km})")
        edge_index, edge_weight = build_graph(
            built["lats"], built["lons"], max_dist_km=args.max_dist_km
        )
        print(f"      edges={edge_index.shape[1]}")

        print(
            f"[4/6] Training STGNN (n_features={built['n_features']}, "
            f"dropout={args.dropout}, epochs={args.epochs}, device={args.device})"
        )
        model = make_model(
            len(built["loc_ids"]), args.seq_len, built["n_features"], dropout=args.dropout
        )
        model = train_model(
            model, built["train"], edge_index, edge_weight,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device,
        )

        print("[5/6] Predicting on test year + attaching anomaly labels")
        if args.mc_dropout:
            print(
                f"      MC Dropout: eval() + reactivate only nn.Dropout, "
                f"{args.mc_samples} forward passes/batch (no model.train())"
            )
            predictions = predict_mc(
                model, built["test"], edge_index, edge_weight,
                args.device, args.batch_size, mc_samples=args.mc_samples,
            )
        else:
            predictions = predict(
                model, built["test"], edge_index, edge_weight, args.device, args.batch_size
            )
        anomaly_scores = load_anomaly_labels(args.anomaly_scores)
        predictions = attach_anomaly_labels(predictions, anomaly_scores)
        global_df, by_df = compute_metrics(predictions)

        print(f"[6/6] Writing outputs to {args.out_dir}")
        meta = build_meta(
            {
                "mode": "pvgis_stgnn",
                "model_type": args.model_type,
                "feature_set": args.feature_set,
                "target_variable": args.target_variable,
                "seq_len": args.seq_len,
                "horizon": args.horizon,
                "train_years": args.train_years,
                "test_year": args.test_year,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "anomaly_scores": args.anomaly_scores,
                "device": args.device,
                "wandb_enabled": bool(args.wandb),
                "mc_dropout": bool(args.mc_dropout),
                "mc_samples": args.mc_samples,
            },
            n_predictions=len(predictions),
            n_nodes=len(built["loc_ids"]),
            features=features,
        )
        paths = write_outputs(predictions, global_df, by_df, args.out_dir, meta)

        summary = build_wandb_metrics(global_df, by_df, mc_dropout=bool(args.mc_dropout))
        if wandb_run is not None:
            wandb_run.log(summary)
            wandb_run.summary.update(summary)

        print("\nDone.")
        print(global_df.to_string(index=False))
        if not by_df.empty:
            print(by_df.to_string(index=False))
        print("\nKey metrics:")
        for k in sorted(summary):
            print(f"  {k:32s} {summary[k]:.4f}")
        for key in ("predictions", "metrics_global", "metrics_by_anomaly_label", "report"):
            print(f"  {key:24s} -> {paths[key]}")
        return paths
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        test_ds.close()
        for ds in train_map.values():
            ds.close()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_from_args(args, parser=parser)


if __name__ == "__main__":
    main()
