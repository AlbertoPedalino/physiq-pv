"""
PVGIS-only ST-GNN experiment runner.

Single source of truth for both entrypoints:
  * `python main.py --mode pvgis_stgnn ...`
  * `python scripts/run_pvgis_stgnn_forecasting.py ...` (thin wrapper)

Deterministic PVGIS-only forecasting:
  * `--feature-set`   selects a subset of the 11 PVGIS-only features; the model
                      is instantiated with STGNN(n_features=len(selected)).
  * `--model-type`    stgnn (implemented); persistence / mlp are scaffolded but
                      raise a clean "not implemented yet" — never run silently.
  * `--wandb`         optional, lazily imported; logs namespaced params + metrics
                      (mae/global, rmse/global).

Hard constraints (PVGIS-only): NO ENERGIA, NO real plant production,
NO Sentinel/SCADA, NO kWp/UPN/load_kwp, NO compute_qs / real QS.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from physiq_pv.data.pvgis_stgnn_dataset import (
    DEFAULT_TARGET_VARIABLE,
    FEATURE_SETS,
    build_datasets,
    build_meta,
    build_wandb_metrics,
    compute_metrics,
    load_pvgis_year,
    load_pvgis_years,
    make_model,
    predict,
    resolve_feature_set,
    train_model,
    write_outputs,
)
from physiq_pv.model.graph_builder import build_graph

# Model-type registry. Only "stgnn" is implemented; the rest are scaffolded so
# the dispatch is ready, but they fail cleanly instead of running silently.
SUPPORTED_MODEL_TYPES = ("stgnn", "persistence", "mlp")
IMPLEMENTED_MODEL_TYPES = ("stgnn",)

# Default --out-dir. Under --wandb (and when left at this default), each run is
# redirected to outputs/wandb_pvgis_stgnn/<run_id>/ so sweep runs never collide.
DEFAULT_OUT_DIR = "outputs/pvgis_stgnn_forecasting"
WANDB_OUT_ROOT = "outputs/wandb_pvgis_stgnn"


def _parse_years(text: str) -> List[int]:
    return [int(y) for y in text.split(",") if y.strip()]


def _slug(text: str) -> str:
    """Filesystem-safe slug for a W&B run name placeholder."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "run"


def _resolve_out_dir(out_dir: str, run_id: str, run_name: Optional[str]) -> str:
    """
    Resolve a per-run output directory under W&B.

    * `{wandb_run_id}` / `{wandb_run_name}` placeholders are substituted (the
      name is slugified for the filesystem);
    * otherwise, if `out_dir` is still the bare default, it is redirected to
      `outputs/wandb_pvgis_stgnn/<run_id>/` so concurrent sweep runs never
      overwrite each other.
    """
    name_slug = _slug(run_name) if run_name else run_id
    if "{wandb_run_id}" in out_dir or "{wandb_run_name}" in out_dir:
        return (
            out_dir.replace("{wandb_run_id}", run_id)
            .replace("{wandb_run_name}", name_slug)
        )
    if out_dir == DEFAULT_OUT_DIR:
        return f"{WANDB_OUT_ROOT}/{run_id}"
    return out_dir


def _log_wandb_artifact(wandb, wandb_run, paths: dict, log_predictions: bool) -> None:
    """
    Attach the run's local outputs to W&B as a versioned artifact.

    Always includes report.md + metrics_global.csv; predictions.csv is added
    only when `log_predictions` is set (it can be very large).
    """
    artifact = wandb.Artifact(f"pvgis_stgnn_{wandb_run.id}", type="pvgis_stgnn_outputs")
    keys = ["report", "metrics_global"]
    if log_predictions:
        keys.append("predictions")
    for key in keys:
        p = paths.get(key)
        if p is not None and Path(p).exists():
            artifact.add_file(str(p))
    wandb_run.log_artifact(artifact)
    print(
        f"[wandb] logged artifact {artifact.name} "
        f"({'with' if log_predictions else 'without'} predictions.csv)"
    )


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
    g.add_argument("--out-dir", "--out_dir", default=DEFAULT_OUT_DIR,
                   help="Output dir. Supports {wandb_run_id}/{wandb_run_name} "
                        "placeholders; under --wandb the default is redirected to "
                        f"{WANDB_OUT_ROOT}/<run_id>/ so sweep runs stay unique.")
    g.add_argument("--seq-len", "--seq_len", type=int, default=24)
    g.add_argument("--horizon", type=int, default=1)
    g.add_argument("--target-variable", "--target_variable", default=DEFAULT_TARGET_VARIABLE)
    g.add_argument("--epochs", type=int, default=10)
    g.add_argument("--batch-size", "--batch_size", type=int, default=8)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--max-dist-km", "--max_dist_km", type=float, default=20.0)
    g.add_argument("--max-train-samples", "--max_train_samples", type=int, default=None)
    g.add_argument("--max-test-samples", "--max_test_samples", type=int, default=None)
    g.add_argument("--skip-predictions-csv", "--skip_predictions_csv",
                   action="store_true",
                   help="Do not write the (large) predictions.csv; metrics + "
                        "report.md are still produced.")
    g.add_argument("--file-template", "--file_template", default="piedmont_pvgis_{year}.nc")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--seed", type=int, default=42)
    # Model / ablation
    g.add_argument("--model-type", "--model_type", default="stgnn", choices=SUPPORTED_MODEL_TYPES,
                   help="stgnn (implemented); persistence/mlp scaffolded (not implemented yet).")
    g.add_argument("--feature-set", "--feature_set", default="full", choices=sorted(FEATURE_SETS),
                   help="Feature ablation; n_features = len(selected features).")
    g.add_argument("--dropout", type=float, default=0.2, help="STGNN dropout.")
    # Optional W&B
    g.add_argument("--wandb", action="store_true", help="Enable optional W&B logging.")
    g.add_argument("--wandb-project", "--wandb_project", default="PhysiQ-PV")
    g.add_argument("--wandb-entity", "--wandb_entity", default=None,
                   help="W&B entity (team/user); e.g. albertopedalino-politecnico-di-torino.")
    g.add_argument("--wandb-run-name", "--wandb_run_name", default=None)
    g.add_argument("--wandb-log-predictions", "--wandb_log_predictions",
                   action="store_true",
                   help="Also log predictions.csv to the W&B artifact (off by "
                        "default — it can be very large).")
    return parser


def build_arg_parser() -> argparse.ArgumentParser:
    """Standalone parser for the script wrapper."""
    p = argparse.ArgumentParser(
        description="PVGIS-only ST-GNN forecasting with feature ablation and "
                    "optional W&B."
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
            entity=args.wandb_entity,
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
                "skip_predictions_csv": args.skip_predictions_csv,
                "wandb_log_predictions": args.wandb_log_predictions,
            },
        )

    # Per-run output dir: unique under --wandb so sweep runs never collide.
    out_dir = args.out_dir
    if wandb_run is not None:
        out_dir = _resolve_out_dir(out_dir, wandb_run.id, wandb_run.name)
        print(f"[wandb] run id={wandb_run.id} name={wandb_run.name} -> out_dir={out_dir}")

    # Effective config banner — confirms which values the sweep actually injected.
    print(
        "[config] " + "  ".join(
            f"{k}={v}" for k, v in (
                ("seed", args.seed),
                ("batch_size", args.batch_size),
                ("epochs", args.epochs),
                ("skip_predictions_csv", args.skip_predictions_csv),
                ("train_years", args.train_years),
                ("test_year", args.test_year),
                ("out_dir", out_dir),
            )
        )
    )
    t_run_start = time.perf_counter()

    print(
        f"[1/5] Loading PVGIS years (train={train_years}, test={args.test_year}) "
        f"| model={args.model_type} feature_set={args.feature_set} "
        f"n_features={len(features)}"
    )
    train_map = load_pvgis_years(args.pvgis_dir, train_years, file_template=args.file_template)
    test_path = f"{args.pvgis_dir}/{args.file_template.format(year=args.test_year)}"
    test_ds = load_pvgis_year(test_path)

    try:
        print(f"[2/5] Building datasets (features={features})")
        t0 = time.perf_counter()
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
        print(f"      [time] building datasets: {time.perf_counter() - t0:.1f}s")

        print(f"[3/5] Building graph (max_dist_km={args.max_dist_km})")
        edge_index, edge_weight = build_graph(
            built["lats"], built["lons"], max_dist_km=args.max_dist_km
        )
        print(f"      edges={edge_index.shape[1]}")

        print(
            f"[4/5] Training STGNN (n_features={built['n_features']}, "
            f"dropout={args.dropout}, epochs={args.epochs}, device={args.device})"
        )
        model = make_model(
            len(built["loc_ids"]), args.seq_len, built["n_features"], dropout=args.dropout
        )
        t_train = time.perf_counter()
        model = train_model(
            model, built["train"], edge_index, edge_weight,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device,
        )
        print(f"      [time] training total: {time.perf_counter() - t_train:.1f}s")

        print("[5/5] Predicting on test year")
        t_test = time.perf_counter()
        predictions = predict(
            model, built["test"], edge_index, edge_weight, args.device, args.batch_size
        )
        print(f"      [time] test inference: {time.perf_counter() - t_test:.1f}s")
        global_df = compute_metrics(predictions)

        print(f"      Writing outputs to {out_dir}")
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
                "device": args.device,
                "wandb_enabled": bool(args.wandb),
            },
            n_predictions=len(predictions),
            n_nodes=len(built["loc_ids"]),
            features=features,
        )
        t_write = time.perf_counter()
        paths = write_outputs(
            predictions, global_df, out_dir, meta,
            skip_predictions=args.skip_predictions_csv,
        )
        # metrics.json: machine-readable global metrics.
        metrics_payload = {"global": global_df.iloc[0].to_dict()}
        metrics_json_path = Path(out_dir) / "metrics.json"
        metrics_json_path.write_text(
            json.dumps(metrics_payload, indent=2, default=str), encoding="utf-8"
        )
        paths["metrics_json"] = metrics_json_path
        print(f"      metrics.json -> {metrics_json_path}")
        print(f"      [time] writing outputs: {time.perf_counter() - t_write:.1f}s")

        summary = build_wandb_metrics(global_df)
        if wandb_run is not None:
            wandb_run.log(summary)
            wandb_run.summary.update(summary)
            _log_wandb_artifact(wandb, wandb_run, paths, args.wandb_log_predictions)

        print(f"\nDone. [time] total run: {time.perf_counter() - t_run_start:.1f}s")
        print(global_df.to_string(index=False))
        print("\nKey metrics:")
        for k in sorted(summary):
            print(f"  {k:32s} {summary[k]:.4f}")
        for key in ("predictions", "metrics_global", "report"):
            print(f"  {key:24s} -> {paths[key] if paths[key] is not None else '(skipped)'}")
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
