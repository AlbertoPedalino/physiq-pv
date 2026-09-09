"""Train the dedicated CNN + LSTM STGAN ablation; export compatible scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from physiq_pv.anomaly_detection.common import runtime_environment
from physiq_pv.anomaly_detection.stgan import (
    ALIGNMENT_POLICY,
    REFERENCE_CONFIG,
    REFERENCE_SEED,
    STGANCNNConfig,
    fit_and_score_stgan,
    load_aligned_manifest_cubes,
    build_spatial_grid,
)


CANONICAL_SCORE_COLUMNS = [
    "location",
    "timestamp",
    "method",
    "anomaly_score",
    "global_rank",
    "global_percentile",
    "threshold",
    "is_anomaly",
]


def _seed_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers.") from exc
    if not result:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=REFERENCE_CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=REFERENCE_CONFIG.batch_size)
    parser.add_argument("--lr", type=float, default=REFERENCE_CONFIG.learning_rate)
    parser.add_argument("--hidden-size", type=int, default=REFERENCE_CONFIG.hidden_size)
    parser.add_argument("--n-layers", type=int, default=REFERENCE_CONFIG.n_layers)
    parser.add_argument("--cnn-channels", type=int, default=REFERENCE_CONFIG.cnn_channels)
    parser.add_argument("--cnn-layers", type=int, default=REFERENCE_CONFIG.cnn_layers)
    parser.add_argument("--grid-crs", default=REFERENCE_CONFIG.grid_crs)
    parser.add_argument("--grid-spacing", type=float, default=REFERENCE_CONFIG.grid_spacing)
    parser.add_argument("--grid-tolerance", type=float, default=REFERENCE_CONFIG.grid_tolerance)
    parser.add_argument("--audit-only", action="store_true", help="Validate coordinates and export grid audit without loading time series or training.")
    parser.add_argument(
        "--generator-reconstruction-weight",
        type=float,
        default=REFERENCE_CONFIG.generator_reconstruction_weight,
    )
    parser.add_argument(
        "--patch-size", type=int, default=REFERENCE_CONFIG.patch_size
    )
    parser.add_argument("--recent-steps", type=int, default=REFERENCE_CONFIG.recent_steps)
    parser.add_argument("--trend-steps", type=int, default=REFERENCE_CONFIG.trend_steps)
    parser.add_argument("--score-stride", type=int, default=REFERENCE_CONFIG.score_stride)
    parser.add_argument(
        "--train-samples-per-epoch",
        type=int,
        default=REFERENCE_CONFIG.train_samples_per_epoch,
        help=(
            "Use 0 (paper default) to visit every time-location pair per epoch; "
            "a positive value enables replacement sampling."
        ),
    )
    parser.add_argument(
        "--paper-top-k-percent",
        type=float,
        default=1.0,
        help="Percentage of global test scores flagged, as in the paper.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", type=_seed_list, default=(REFERENCE_SEED,))
    parser.add_argument("--max-locations", type=int)
    parser.add_argument(
        "--export-all-feature-scores",
        action="store_true",
        help="By default feature rows are exported only for globally flagged test points.",
    )
    return parser.parse_args(argv)


def _append_csv(frame: pd.DataFrame, path: Path, *, first: bool) -> None:
    frame.to_csv(path, mode="w" if first else "a", header=first, index=False)


def _canonical_location_frame(
    *,
    location: str,
    timestamps: pd.DatetimeIndex,
    scores: np.ndarray,
    ranks: np.ndarray,
    percentiles: np.ndarray,
    flags: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "location": location,
            "timestamp": timestamps,
            "method": "stgan_cnn",
            "anomaly_score": scores,
            "global_rank": ranks,
            "global_percentile": percentiles,
            "threshold": np.nan,
            "is_anomaly": np.asarray(flags, dtype=bool),
        }
    )[CANONICAL_SCORE_COLUMNS]


def paper_top_k_ranking(
    scores: np.ndarray, percentage: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact top-K flags plus a stable global ranking."""
    if not np.isfinite(percentage) or not 0.0 < percentage <= 100.0:
        raise ValueError("paper_top_k percentage must be in (0, 100].")
    values = np.asarray(scores, dtype=np.float64)
    finite_flat = np.flatnonzero(np.isfinite(values.reshape(-1)))
    if finite_flat.size == 0:
        raise ValueError("Cannot rank STGAN test scores without finite values.")
    count = min(
        finite_flat.size,
        max(1, int(np.ceil(finite_flat.size * percentage / 100.0))),
    )
    flat_values = values.reshape(-1)
    # Stable ordering makes tie resolution reproducible while retaining exactly
    # the requested anomaly budget.
    order = np.argsort(-flat_values[finite_flat], kind="stable")
    ranked = finite_flat[order]
    ranks = np.full(flat_values.shape, np.nan, dtype=np.float64)
    ranks[ranked] = np.arange(1, finite_flat.size + 1, dtype=np.float64)
    percentiles = np.full(flat_values.shape, np.nan, dtype=np.float64)
    percentiles[ranked] = (
        100.0 * (finite_flat.size - ranks[ranked] + 1.0) / finite_flat.size
    )
    flags = np.zeros(flat_values.shape, dtype=bool)
    flags[ranked[:count]] = True
    return (
        flags.reshape(values.shape),
        ranks.reshape(values.shape),
        percentiles.reshape(values.shape),
    )


def _feature_frame(
    *,
    location: str,
    timestamps: pd.DatetimeIndex,
    scores: np.ndarray,
    feature_names: tuple[str, ...],
    keep: np.ndarray,
) -> pd.DataFrame:
    selected = scores[keep]
    selected_times = timestamps[keep]
    total = np.maximum(selected.sum(axis=1, keepdims=True), 1e-12)
    n_times, n_features = selected.shape
    return pd.DataFrame(
        {
            "location": location,
            "timestamp": np.repeat(selected_times, n_features),
            "method": "stgan_cnn",
            "entity": np.tile(np.asarray(feature_names), n_times),
            "anomaly_score": selected.reshape(-1),
            "contribution_fraction": (selected / total).reshape(-1),
        }
    )


def run_stgan(
    *,
    manifest_path: str | Path,
    out_dir: str | Path,
    paper_top_k_percent: float,
    config: STGANCNNConfig = REFERENCE_CONFIG,
    device: str = "cuda",
    seeds: tuple[int, ...] = (REFERENCE_SEED,),
    max_locations: int | None = None,
    export_all_feature_scores: bool = False,
    audit_only: bool = False,
) -> Path:
    """Run the CNN spatial ablation with the original GAN loss and score protocol."""
    if not np.isfinite(paper_top_k_percent) or not 0.0 < paper_top_k_percent <= 100.0:
        raise ValueError("paper_top_k_percent must be in (0, 100].")
    resolved_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not resolved_seeds:
        raise ValueError("At least one seed is required.")

    manifest_path = Path(manifest_path).resolve()
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = pd.read_csv(manifest_path, dtype={"location": str, "site_key": str})
    if max_locations is not None:
        manifest = manifest.head(max_locations)
    if len(manifest) < 2:
        raise ValueError("STGAN requires at least two locations.")

    out_root = Path(out_dir).resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    grid = build_spatial_grid(manifest["latitude"], manifest["longitude"],
        patch_size=config.patch_size, grid_crs=config.grid_crs,
        grid_spacing=config.grid_spacing, grid_tolerance=config.grid_tolerance)
    grid_locations = grid.location_frame(manifest["location"].astype(str).tolist())
    grid_locations.to_csv(out_root / "grid_locations.csv", index=False)
    np.savez_compressed(out_root / "grid_layout.npz", node_indices=grid.node_indices,
                        valid_mask=grid.valid_mask)
    (out_root / "grid_audit.json").write_text(json.dumps(grid.metadata, indent=2), encoding="utf-8")
    print(json.dumps(grid.metadata, indent=2), flush=True)
    location_table = manifest.loc[
        :, ["location", "site_key", "latitude", "longitude"]
    ].reset_index(drop=True)
    location_table.to_csv(out_root / "locations.csv", index=False)
    if audit_only:
        return out_root
    cubes = load_aligned_manifest_cubes(manifest, cache_dir=out_root / "cube_cache")
    summaries: list[dict] = []

    try:
        for seed in resolved_seeds:
            print(
                f"[stgan] seed={seed} locations={len(cubes.location_names)} "
                f"features={list(cubes.feature_names)}"
            )
            seed_root = out_root / f"seed_{seed}"
            seed_root.mkdir(parents=True, exist_ok=True)
            result = fit_and_score_stgan(
                cubes.train,
                cubes.test,
                train_timestamps=cubes.train_timestamps,
                test_timestamps=cubes.test_timestamps,
                location_names=cubes.location_names,
                feature_names=cubes.feature_names,
                latitudes=cubes.latitudes,
                longitudes=cubes.longitudes,
                epochs=config.epochs,
                batch_size=config.batch_size,
                lr=config.learning_rate,
                generator_reconstruction_weight=config.generator_reconstruction_weight,
                hidden_size=config.hidden_size,
                n_layers=config.n_layers,
                cnn_channels=config.cnn_channels,
                cnn_layers=config.cnn_layers,
                patch_size=config.patch_size,
                grid_crs=config.grid_crs,
                grid_spacing=config.grid_spacing,
                grid_tolerance=config.grid_tolerance,
                recent_steps=config.recent_steps,
                trend_steps=config.trend_steps,
                score_stride=config.score_stride,
                train_samples_per_epoch=(
                    None
                    if config.train_samples_per_epoch <= 0
                    else config.train_samples_per_epoch
                ),
                device=device,
                seed=seed,
                checkpoint_path=seed_root / "checkpoint.pt",
            )

            test_path = seed_root / "anomaly_scores.csv"
            feature_path = seed_root / "entity_anomaly_scores.csv"
            global_flags, global_ranks, global_percentiles = paper_top_k_ranking(
                result.test_scores, paper_top_k_percent
            )

            for location_index, location in enumerate(result.location_names):
                test_frame = _canonical_location_frame(
                    location=location,
                    timestamps=result.test_timestamps,
                    scores=result.test_scores[:, location_index],
                    ranks=global_ranks[:, location_index],
                    percentiles=global_percentiles[:, location_index],
                    flags=global_flags[:, location_index],
                )
                flags = test_frame["is_anomaly"].to_numpy(dtype=bool)
                feature_keep = (
                    np.ones(len(flags), dtype=bool)
                    if export_all_feature_scores
                    else flags
                )
                feature_frame = _feature_frame(
                    location=location,
                    timestamps=result.test_timestamps,
                    scores=result.test_feature_scores[:, location_index, :],
                    feature_names=result.feature_names,
                    keep=feature_keep,
                )
                first = location_index == 0
                _append_csv(test_frame, test_path, first=first)
                _append_csv(feature_frame, feature_path, first=first)

                top_feature_index = np.argmax(
                    result.test_feature_scores[:, location_index, :], axis=1
                )
                details = test_frame.copy()
                details.insert(1, "latitude", cubes.latitudes[location_index])
                details.insert(2, "longitude", cubes.longitudes[location_index])
                details["generator_score_raw"] = result.test_generator_scores[:, location_index]
                details["discriminator_score_raw"] = result.test_discriminator_scores[
                    :, location_index
                ]
                details["top_feature"] = np.asarray(result.feature_names)[top_feature_index]
                site_key = str(manifest.iloc[location_index]["site_key"])
                site_root = seed_root / "locations" / site_key
                site_root.mkdir(parents=True, exist_ok=True)
                details.to_csv(site_root / "test_scores.csv", index=False)

                summaries.append(
                    {
                        "location": location,
                        "site_key": site_key,
                        "latitude": float(cubes.latitudes[location_index]),
                        "longitude": float(cubes.longitudes[location_index]),
                        "method": "stgan_cnn",
                        "seed": seed,
                        "threshold": np.nan,
                        "n_scored": len(test_frame),
                        "n_anomaly": int(flags.sum()),
                        "anomaly_rate": float(flags.mean()),
                        "n_valid_cells": int(grid.valid_mask[location_index].sum()),
                        "complete_patch": bool(grid.valid_mask[location_index].all()),
                    }
                )

            pd.DataFrame(row for row in summaries if row["seed"] == seed).to_csv(
                seed_root / "summary.csv", index=False
            )
            boundary_rows = []
            for name, select in (("interior", grid.valid_mask.all(axis=(1, 2))),
                                 ("boundary", ~grid.valid_mask.all(axis=(1, 2)))):
                values = result.test_scores[:, select]
                if values.size:
                    boundary_rows.append({"group": name, "n_locations": int(select.sum()),
                        "n_scored": int(values.size), "score_mean": float(values.mean()),
                        "score_median": float(np.median(values)),
                        "score_q95": float(np.quantile(values, .95)),
                        "n_anomaly": int(global_flags[:, select].sum()),
                        "anomaly_share_pct": float(global_flags[:, select].mean() * 100)})
            pd.DataFrame(boundary_rows).to_csv(seed_root / "boundary_summary.csv", index=False)
            (seed_root / "metadata.json").write_text(
                json.dumps(
                    {
                        "method": "stgan_cnn",
                        "seed": seed,
                        "backend": result.metadata,
                        "features": list(result.feature_names),
                        "locations": len(result.location_names),
                        "feature_export": (
                            "all_test_points"
                            if export_all_feature_scores
                            else "globally_flagged_test_points_only"
                        ),
                        "paper_top_k_percent": paper_top_k_percent,
                        "reference_labels_loaded": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    finally:
        cubes.close()

    pd.DataFrame(summaries).to_csv(out_root / "summary_by_seed.csv", index=False)
    run_metadata = {
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": manifest_digest,
        "method": "stgan_cnn",
        "paper": "Graph Convolutional Adversarial Networks for Spatiotemporal Anomaly Detection",
        "doi": "10.1109/TNNLS.2021.3136171",
        "alignment_policy": ALIGNMENT_POLICY,
        "seeds": list(resolved_seeds),
        "test_labels_used": False,
        "training_period_role": "unlabelled_model_fit_only",
        "spatial_semantics": "geographic_grid_patch_with_three_meteo_channels_and_validity_mask",
        "adjacency_used": False,
        "grid": grid.metadata,
        "feature_semantics": "meteorological_channels_plus_non_meteorological_mask",
        "decision_rule": "global_test_top_k",
        "decision_rule_semantics": "paper_global_test_score_ranking",
        "paper_top_k_percent": paper_top_k_percent,
        "configuration": {
            "model": config.to_dict(),
            "device": device,
            "max_locations": max_locations,
            "export_all_feature_scores": export_all_feature_scores,
        },
        "environment": runtime_environment(_ROOT),
    }
    (out_root / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, default=str), encoding="utf-8"
    )
    return out_root


def main(argv=None) -> None:
    args = parse_args(argv)
    config = STGANCNNConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        generator_reconstruction_weight=args.generator_reconstruction_weight,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        cnn_channels=args.cnn_channels,
        cnn_layers=args.cnn_layers,
        patch_size=args.patch_size,
        grid_crs=args.grid_crs,
        grid_spacing=args.grid_spacing,
        grid_tolerance=args.grid_tolerance,
        recent_steps=args.recent_steps,
        trend_steps=args.trend_steps,
        score_stride=args.score_stride,
        train_samples_per_epoch=args.train_samples_per_epoch,
    )
    seeds = (args.seed,) if args.seed is not None else tuple(args.seeds)
    run_stgan(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        paper_top_k_percent=args.paper_top_k_percent,
        config=config,
        device=args.device,
        seeds=seeds,
        max_locations=args.max_locations,
        export_all_feature_scores=args.export_all_feature_scores,
        audit_only=args.audit_only,
    )


if __name__ == "__main__":
    main()
