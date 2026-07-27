"""Train and score the MTGFlow base model on prepared PVGIS shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from physiq_pv.anomaly_detection.common import runtime_environment
from physiq_pv.anomaly_detection.mtgflow import (
    REFERENCE_CONFIG,
    REFERENCE_SEEDS,
    fit_and_score_mtgflow,
    reference_protocol_deviations,
)
from physiq_pv.anomaly_detection.thresholds import (
    apply_entity_thresholds,
    apply_threshold,
    fit_entity_iqr_thresholds,
    fit_threshold,
)


REQUIRED_MANIFEST_COLUMNS = {
    "location",
    "site_key",
    "train_csv",
    "test_csv",
}
CANONICAL_SCORE_COLUMNS = [
    "location",
    "timestamp",
    "method",
    "anomaly_score",
    "threshold",
    "is_anomaly",
]
DEFAULT_EXPORT_TRAIN_YEARS = (2016, 2017, 2018)


def _seed_list(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers.") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return seeds


def _year_list(value: str) -> tuple[int, ...]:
    years: list[int] = []
    try:
        for part in value.split(","):
            bounds = part.strip().split("-")
            if len(bounds) == 1:
                years.append(int(bounds[0]))
            elif len(bounds) == 2:
                start, end = map(int, bounds)
                if start > end:
                    raise ValueError
                years.extend(range(start, end + 1))
            else:
                raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Years must be comma-separated values or inclusive ranges."
        ) from exc
    unique = tuple(dict.fromkeys(years))
    if not unique:
        raise argparse.ArgumentTypeError("At least one export train year is required.")
    return unique


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=REFERENCE_CONFIG.epochs)
    parser.add_argument("--window-size", type=int, default=REFERENCE_CONFIG.window_size)
    parser.add_argument("--batch-size", type=int, default=REFERENCE_CONFIG.batch_size)
    parser.add_argument("--n-blocks", type=int, default=REFERENCE_CONFIG.n_blocks)
    parser.add_argument("--train-stride", type=int, default=REFERENCE_CONFIG.train_stride)
    parser.add_argument(
        "--score-stride",
        type=int,
        default=REFERENCE_CONFIG.score_stride,
        help="10 follows the reference protocol; 1 enables the labelled hourly adaptation.",
    )
    parser.add_argument(
        "--export-train-years",
        type=_year_list,
        default=DEFAULT_EXPORT_TRAIN_YEARS,
        help="Years written to train_anomaly_scores.csv (default: 2016-2018).",
    )
    parser.add_argument("--device", default="cuda")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed", type=int, help="Run one seed instead of the five-seed reference suite."
    )
    seed_group.add_argument(
        "--seeds",
        type=_seed_list,
        default=REFERENCE_SEEDS,
        help="Comma-separated seeds (default: 15,16,17,18,19).",
    )
    parser.add_argument("--iqr-k", type=float, default=REFERENCE_CONFIG.iqr_k)
    parser.add_argument(
        "--entity-threshold-scale",
        type=float,
        default=REFERENCE_CONFIG.entity_threshold_scale,
    )
    parser.add_argument("--max-locations", type=int)
    parser.add_argument(
        "--allow-unverified-preparation",
        action="store_true",
        help="Allow manifests without preparation metadata (not recommended for reference runs).",
    )
    return parser.parse_args(argv)


def _read(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def _read_optional_validation(row) -> pd.DataFrame | None:
    path = getattr(row, "validation_csv", None)
    if path is None or pd.isna(path) or not str(path).strip():
        return None
    return _read(str(path))


def _verify_preparation_metadata(row, *, allow_unverified: bool) -> bool:
    seasonal = getattr(row, "seasonal_normalization", None)
    if seasonal is not None and not pd.isna(seasonal):
        seasonal_enabled = (
            seasonal.strip().lower() in {"1", "true", "yes"}
            if isinstance(seasonal, str)
            else bool(seasonal)
        )
        if seasonal_enabled:
            raise ValueError(
                f"{row.site_key}: seasonal normalization is incompatible with the reference protocol."
            )
        return True

    metadata_path = Path(row.train_csv).resolve().parent / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("seasonal_normalization") is True:
            raise ValueError(
                f"{row.site_key}: prepared data use seasonal normalization; regenerate them."
            )
        if metadata.get("seasonal_normalization") is False:
            return True

    if not allow_unverified:
        raise ValueError(
            f"{row.site_key}: cannot verify preparation mode. Regenerate the manifest or "
            "pass --allow-unverified-preparation for non-reference smoke tests."
        )
    return False


def _entity_output(
    *,
    location: str,
    seed: int,
    window_starts: pd.DatetimeIndex,
    timestamps: pd.DatetimeIndex,
    scores: np.ndarray,
    names: tuple[str, ...],
    thresholds: np.ndarray,
) -> pd.DataFrame:
    flags = apply_entity_thresholds(scores, thresholds)
    n_windows, n_entities = scores.shape
    return pd.DataFrame(
        {
            "location": location,
            "seed": seed,
            "window_start": np.repeat(window_starts, n_entities),
            "window_end": np.repeat(timestamps, n_entities),
            "timestamp": np.repeat(timestamps, n_entities),
            "method": "mtgflow",
            "entity": np.tile(np.asarray(names), n_windows),
            "anomaly_score": scores.reshape(-1),
            "threshold": np.tile(thresholds, n_windows),
            "is_anomaly": flags.reshape(-1),
        }
    )


def _global_output(
    *,
    location: str,
    timestamps: pd.DatetimeIndex,
    scores: np.ndarray,
    threshold,
) -> pd.DataFrame:
    """Return the canonical detector-score contract used downstream."""
    return pd.DataFrame(
        {
            "location": location,
            "timestamp": timestamps,
            "method": "mtgflow",
            "anomaly_score": scores,
            "threshold": threshold.value,
            "is_anomaly": apply_threshold(scores, threshold),
        }
    )[CANONICAL_SCORE_COLUMNS]


def _score_details(
    canonical: pd.DataFrame,
    *,
    seed: int,
    window_starts: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Add diagnostic window metadata without changing canonical exports."""
    details = canonical.copy()
    details.insert(1, "seed", seed)
    details.insert(2, "window_start", window_starts)
    details.insert(3, "window_end", canonical["timestamp"])
    return details


def _select_export_years(
    scores: pd.DataFrame, years: tuple[int, ...]
) -> pd.DataFrame:
    timestamps = pd.to_datetime(scores["timestamp"])
    available = set(timestamps.dt.year.unique().tolist())
    missing = sorted(set(years) - available)
    if missing:
        raise ValueError(
            f"Training scores do not cover requested export years: {missing}."
        )
    selected = scores.loc[timestamps.dt.year.isin(years), CANONICAL_SCORE_COLUMNS]
    if selected.empty:
        raise ValueError("No training scores remain for the requested export years.")
    return selected.reset_index(drop=True)


def _population_std(series: pd.Series) -> float:
    return float(np.std(series.to_numpy(dtype=float), ddof=0))


def main(argv=None):
    args = parse_args(argv)
    seeds = (args.seed,) if args.seed is not None else args.seeds
    manifest = pd.read_csv(args.manifest)
    missing_columns = sorted(REQUIRED_MANIFEST_COLUMNS - set(manifest.columns))
    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {missing_columns}")
    if args.max_locations is not None:
        manifest = manifest.head(args.max_locations)
    if manifest.empty:
        raise ValueError("Manifest contains no locations.")

    out_root = Path(args.out_dir)
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {out_root}. Use a new directory so "
            "artifacts from different runs cannot be mixed."
        )
    out_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    all_preparation_verified = True

    for seed in seeds:
        seed_root = out_root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        all_train: list[pd.DataFrame] = []
        all_test: list[pd.DataFrame] = []
        all_entity_test: list[pd.DataFrame] = []

        for row in manifest.itertuples(index=False):
            location = str(row.location)
            verified = _verify_preparation_metadata(
                row, allow_unverified=args.allow_unverified_preparation
            )
            all_preparation_verified &= verified
            print(f"[mtgflow] seed={seed} location={location} key={row.site_key}")

            site_dir = seed_root / str(row.site_key)
            site_dir.mkdir(parents=True, exist_ok=True)
            result = fit_and_score_mtgflow(
                _read(row.train_csv),
                _read(row.test_csv),
                validation=_read_optional_validation(row),
                epochs=args.epochs,
                window_size=args.window_size,
                train_stride=args.train_stride,
                score_stride=args.score_stride,
                batch_size=args.batch_size,
                n_blocks=args.n_blocks,
                device=args.device,
                seed=seed,
                checkpoint_path=site_dir / "checkpoint.pt",
            )
            if result.train_entity_scores is None or result.test_entity_scores is None:
                raise RuntimeError("MTGFlow must return entity-level scores.")
            if result.train_window_starts is None or result.test_window_starts is None:
                raise RuntimeError("MTGFlow must return complete window boundaries.")

            global_threshold = fit_threshold(
                result.train_scores, method="iqr", iqr_k=args.iqr_k
            )
            global_flags = apply_threshold(result.test_scores, global_threshold)
            entity_thresholds = fit_entity_iqr_thresholds(
                result.train_entity_scores,
                iqr_k=args.iqr_k,
                scale=args.entity_threshold_scale,
            )

            test_frame = _read(row.test_csv)
            test_day = test_frame.set_index("timestamp")["is_daytime"]
            test_out = _global_output(
                location=location,
                timestamps=result.test_timestamps,
                scores=result.test_scores,
                threshold=global_threshold,
            )
            test_details = _score_details(
                test_out,
                seed=seed,
                window_starts=result.test_window_starts,
            )
            test_details["is_daytime"] = (
                test_day.reindex(result.test_timestamps).fillna(False).to_numpy(bool)
            )
            train_out = _global_output(
                location=location,
                timestamps=result.train_timestamps,
                scores=result.train_scores,
                threshold=global_threshold,
            )
            train_details = _score_details(
                train_out,
                seed=seed,
                window_starts=result.train_window_starts,
            )
            test_entity_out = _entity_output(
                location=location,
                seed=seed,
                window_starts=result.test_window_starts,
                timestamps=result.test_timestamps,
                scores=result.test_entity_scores,
                names=result.entity_names,
                thresholds=entity_thresholds,
            )
            train_entity_out = _entity_output(
                location=location,
                seed=seed,
                window_starts=result.train_window_starts,
                timestamps=result.train_timestamps,
                scores=result.train_entity_scores,
                names=result.entity_names,
                thresholds=entity_thresholds,
            )

            test_details.to_csv(site_dir / "test_scores.csv", index=False)
            train_details.to_csv(site_dir / "train_scores.csv", index=False)
            test_entity_out.to_csv(site_dir / "test_entity_scores.csv", index=False)
            train_entity_out.to_csv(site_dir / "train_entity_scores.csv", index=False)
            metadata = {
                "location": location,
                "site_key": str(row.site_key),
                "method": "mtgflow",
                "seed": seed,
                "threshold": global_threshold.to_dict(),
                "entity_thresholds": dict(
                    zip(result.entity_names, entity_thresholds.tolist())
                ),
                "entity_threshold_scale": args.entity_threshold_scale,
                "backend": result.metadata,
                "checkpoint": str((site_dir / "checkpoint.pt").resolve()),
                "preparation_metadata_verified": verified,
                "reference_labels_loaded": False,
            }
            (site_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )

            all_train.append(train_out)
            all_test.append(test_out)
            all_entity_test.append(test_entity_out)
            summaries.append(
                {
                    "location": location,
                    "site_key": str(row.site_key),
                    "method": "mtgflow",
                    "seed": seed,
                    "threshold": global_threshold.value,
                    "n_scored": len(test_out),
                    "n_anomaly": int(global_flags.sum()),
                    "anomaly_rate": float(np.mean(global_flags)),
                }
            )

        combined = pd.concat(all_test, ignore_index=True)
        combined_train = pd.concat(all_train, ignore_index=True)
        combined[CANONICAL_SCORE_COLUMNS].to_csv(
            seed_root / "anomaly_scores.csv", index=False
        )
        export_train = _select_export_years(
            combined_train, args.export_train_years
        )
        export_train.to_csv(
            seed_root / "train_anomaly_scores.csv", index=False
        )
        pd.concat(all_entity_test, ignore_index=True).to_csv(
            seed_root / "entity_anomaly_scores.csv", index=False
        )
        pd.DataFrame([row for row in summaries if row["seed"] == seed]).to_csv(
            seed_root / "summary.csv", index=False
        )
        print(
            f"Wrote seed-{seed} scores to {seed_root}: "
            f"train={len(export_train):,}, test={len(combined):,}"
        )

    summary = pd.DataFrame(summaries)
    summary.to_csv(out_root / "summary_by_seed.csv", index=False)
    aggregate = (
        summary.groupby(["location", "site_key", "method"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            n_scored=("n_scored", "first"),
            threshold_mean=("threshold", "mean"),
            threshold_std=("threshold", _population_std),
            anomaly_rate_mean=("anomaly_rate", "mean"),
            anomaly_rate_std=("anomaly_rate", _population_std),
        )
    )
    aggregate.to_csv(out_root / "summary_aggregate.csv", index=False)
    environment = runtime_environment(_ROOT)
    protocol_deviations = reference_protocol_deviations(
        vars(args), tuple(seeds), all_preparation_verified
    )
    run_metadata = {
        "method": "mtgflow",
        "seeds": list(seeds),
        "reference_seed_suite_matched": tuple(seeds) == REFERENCE_SEEDS,
        "epochs": args.epochs,
        "window_size": args.window_size,
        "batch_size": args.batch_size,
        "n_blocks": args.n_blocks,
        "train_stride": args.train_stride,
        "score_stride": args.score_stride,
        "export_train_years": list(args.export_train_years),
        "iqr_k": args.iqr_k,
        "entity_threshold_scale": args.entity_threshold_scale,
        "scoring_profile": (
            "reference_window_sampling"
            if args.window_size == REFERENCE_CONFIG.window_size
            and args.train_stride == REFERENCE_CONFIG.train_stride
            and args.score_stride == REFERENCE_CONFIG.score_stride
            else (
                "dense_hourly_window_adaptation"
                if args.window_size == REFERENCE_CONFIG.window_size
                and args.train_stride == REFERENCE_CONFIG.train_stride
                and args.score_stride == 1
                else "custom_window_sampling"
            )
        ),
        "reference_protocol_matched": not protocol_deviations,
        "reference_protocol_deviations": protocol_deviations,
        "original_benchmark_reproduction": False,
        "original_benchmark_reproduction_note": (
            "PVGIS climate entities and chronological years replace the paper benchmark datasets."
        ),
        "window_score_semantics": "whole_window_assigned_to_window_end",
        "preparation_metadata_verified": all_preparation_verified,
        "scores_aggregated_across_seeds": False,
        "environment": environment,
    }
    (out_root / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    (out_root / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
