"""Post-hoc forecast-error sensitivity to MTGFlow and STGAN decisions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from physiq_pv.reporting.posthoc_outputs import PERCENT_PRODUCTION_BINS
from physiq_pv.reporting.daytime_bin_anomaly_report import _boxplot_stats


def _normalise_excluded_timestamps(
    values: Sequence[str | pd.Timestamp] | pd.DatetimeIndex | None,
) -> pd.DatetimeIndex:
    if values is None:
        return pd.DatetimeIndex([])
    parsed = pd.to_datetime(list(values), errors="raise", utc=True)
    return pd.DatetimeIndex(parsed).tz_convert(None).drop_duplicates()


def load_prediction_errors(
    path: str | Path,
    *,
    horizons: Sequence[int] = (1, 6),
    daytime_threshold_wm2: float | None = 10.0,
    reference_peak_w: float | None = None,
    include_issue_timestamp: bool = False,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load finite target-time errors for the requested direct outputs.

    When ``reference_peak_w`` is provided, rows also receive the same fixed
    percentage-production bin used by the standard SDE-Net reports.
    """
    source = Path(path)
    selected = tuple(dict.fromkeys(int(value) for value in horizons))
    if not selected or any(value < 1 for value in selected):
        raise ValueError("Forecast horizons must be positive integers.")
    header = set(pd.read_csv(source, nrows=0).columns)
    prediction_column = "y_pred_mean" if "y_pred_mean" in header else "y_pred"
    required = {
        "location", "timestamp", "horizon_hours", "y_true", prediction_column
    }
    if daytime_threshold_wm2 is not None:
        required.add("solar_irradiance_poa_target")
    if include_issue_timestamp:
        required.add("issue_timestamp")
    missing = required - header
    if missing:
        raise ValueError(f"{source} is missing {sorted(missing)}.")
    if reference_peak_w is not None:
        reference_peak_w = float(reference_peak_w)
        if not np.isfinite(reference_peak_w) or reference_peak_w <= 0:
            raise ValueError("reference_peak_w must be finite and positive.")

    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(source, usecols=sorted(required), chunksize=chunksize):
        horizon = pd.to_numeric(chunk["horizon_hours"], errors="coerce")
        chunk = chunk.loc[horizon.isin(selected)].copy()
        if chunk.empty:
            continue
        chunk["horizon_hours"] = pd.to_numeric(
            chunk["horizon_hours"], errors="raise"
        ).astype(int)
        chunk["location"] = chunk["location"].astype(str)
        chunk["timestamp"] = pd.to_datetime(
            chunk["timestamp"], errors="raise", utc=True
        ).dt.tz_convert(None)
        numeric = ["y_true", prediction_column]
        if daytime_threshold_wm2 is not None:
            numeric.append("solar_irradiance_poa_target")
        for column in numeric:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        keep = np.isfinite(chunk[numeric].to_numpy(dtype=float)).all(axis=1)
        if daytime_threshold_wm2 is not None:
            keep &= chunk["solar_irradiance_poa_target"].gt(
                float(daytime_threshold_wm2)
            )
        chunk = chunk.loc[keep].copy()
        error = chunk[prediction_column] - chunk["y_true"]
        chunk["abs_error"] = error.abs()
        chunk["squared_error"] = error**2
        output_columns = [
            "location", "timestamp", "horizon_hours", "abs_error",
            "squared_error",
        ]
        if include_issue_timestamp:
            chunk["issue_timestamp"] = pd.to_datetime(
                chunk["issue_timestamp"], errors="raise", utc=True
            ).dt.tz_convert(None)
            expected_issue = chunk["timestamp"] - pd.to_timedelta(
                chunk["horizon_hours"], unit="h"
            )
            if not chunk["issue_timestamp"].eq(expected_issue).all():
                raise ValueError("Issue timestamps do not match target minus horizon.")
            output_columns.append("issue_timestamp")
        if reference_peak_w is not None:
            production_pct = np.clip(
                100.0 * chunk["y_true"].to_numpy(float) / reference_peak_w,
                0.0,
                100.0,
            )
            production_bin = np.full(len(chunk), None, dtype=object)
            for name, lower, upper in PERCENT_PRODUCTION_BINS:
                selected_bin = production_pct >= float(lower)
                if upper is not None:
                    selected_bin &= production_pct < float(upper)
                production_bin[selected_bin] = name
            if pd.isna(production_bin).any():
                raise ValueError("At least one prediction has no production bin.")
            chunk["production_bin"] = production_bin
            output_columns.append("production_bin")
        parts.append(
            chunk[output_columns]
        )
    if not parts:
        raise ValueError("No usable prediction rows were found.")
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["location", "timestamp", "horizon_hours"]).any():
        raise ValueError("Predictions contain duplicate location/target/horizon rows.")
    available = set(result["horizon_hours"].unique())
    missing_horizons = sorted(set(selected) - available)
    if missing_horizons:
        raise ValueError(f"Missing forecast horizons {missing_horizons}.")
    return result


def sensitivity_sweep_by_bin(
    joined: pd.DataFrame,
    thresholds: Sequence[float],
    *,
    detector: str,
    rare_when: str,
    bin_column: str = "production_bin",
) -> pd.DataFrame:
    """Run the exact threshold sweep independently in every production bin."""
    if bin_column not in joined:
        raise ValueError(f"Sensitivity input is missing {bin_column!r}.")
    if joined[bin_column].isna().any():
        raise ValueError(f"{bin_column} contains missing values.")
    frames: list[pd.DataFrame] = []
    for bin_name, frame in joined.groupby(bin_column, observed=True, sort=False):
        result = sensitivity_sweep(
            frame,
            thresholds,
            detector=detector,
            rare_when=rare_when,
        )
        result.insert(2, bin_column, str(bin_name))
        frames.append(result)
    if not frames:
        raise ValueError("No production bins are available for sensitivity analysis.")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["detector", bin_column, "horizon_hours", "decision_threshold"]
    ).reset_index(drop=True)


def load_mtgflow_coordinates(
    score_path: str | Path,
    threshold_table: pd.DataFrame,
    *,
    reference_k: float = 1.5,
    excluded_timestamps: (
        Sequence[str | pd.Timestamp] | pd.DatetimeIndex | None
    ) = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Convert MTGFlow score to the paper's per-site IQR coordinate k."""
    source = Path(score_path)
    required = {"location", "timestamp", "anomaly_score", "threshold"}
    header = set(pd.read_csv(source, nrows=0).columns)
    missing = required - header
    if missing:
        raise ValueError(f"{source} is missing {sorted(missing)}.")
    table = threshold_table.copy()
    table["location"] = table["location"].astype(str)
    if table["location"].duplicated().any() or not {
        "q3", "iqr", "saved_threshold"
    } <= set(table.columns):
        raise ValueError("MTGFlow threshold table requires q3, iqr and saved_threshold.")
    reconstructed = (
        pd.to_numeric(table["q3"], errors="coerce")
        + float(reference_k) * pd.to_numeric(table["iqr"], errors="coerce")
    )
    if not np.allclose(
        reconstructed,
        pd.to_numeric(table["saved_threshold"], errors="coerce"),
        rtol=1e-6,
        atol=1e-5,
    ):
        raise ValueError(
            "Training quantiles do not reconstruct the saved MTGFlow threshold "
            f"at k={float(reference_k):g}; exact per-site statistics are required."
        )
    table = table.set_index("location")

    excluded = _normalise_excluded_timestamps(excluded_timestamps)
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(source, usecols=sorted(required), chunksize=chunksize):
        chunk["location"] = chunk["location"].astype(str)
        chunk["timestamp"] = pd.to_datetime(
            chunk["timestamp"], errors="raise", utc=True
        ).dt.tz_convert(None)
        if len(excluded):
            chunk = chunk.loc[~chunk["timestamp"].isin(excluded)].copy()
            if chunk.empty:
                continue
        score = pd.to_numeric(chunk["anomaly_score"], errors="coerce")
        saved = chunk["location"].map(table["saved_threshold"])
        q3 = chunk["location"].map(table["q3"])
        iqr = chunk["location"].map(table["iqr"])
        csv_threshold = pd.to_numeric(chunk["threshold"], errors="coerce")
        numeric = np.column_stack([score, saved, q3, iqr, csv_threshold])
        if not np.isfinite(numeric).all() or bool((iqr <= 0).any()):
            raise ValueError("Invalid MTGFlow score or training quantile.")
        if not np.allclose(csv_threshold, saved, rtol=1e-6, atol=1e-5):
            raise ValueError("Saved MTGFlow thresholds do not match the score CSV.")
        chunk["decision_coordinate"] = (score - q3) / iqr
        parts.append(chunk[["location", "timestamp", "decision_coordinate"]])
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["location", "timestamp"]).any():
        raise ValueError("MTGFlow contains duplicate location/timestamp rows.")
    result.attrs["excluded_timestamps"] = int(len(excluded))
    return result


def load_stgan_coordinates(
    score_path: str | Path,
    *,
    excluded_timestamps: (
        Sequence[str | pd.Timestamp] | pd.DatetimeIndex | None
    ) = None,
    recompute_global_ranking: bool = False,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Convert STGAN scores to the global top-K coordinate used by the paper.

    With ``recompute_global_ranking=True``, excluded data-quality timestamps are
    removed first and the global ordering is compacted without fitting a new
    score threshold.  ``global_rank`` is preferred when exported because it
    preserves the detector's exact tie order; otherwise scores are ordered
    descending with a stable sort.
    """
    source = Path(score_path)
    header = set(pd.read_csv(source, nrows=0).columns)
    percentile_column = next(
        (
            name
            for name in ("global_percentile", "score_percentile")
            if name in header
        ),
        None,
    )
    required = {"location", "timestamp"}
    if recompute_global_ranking:
        required.add("anomaly_score")
        if "global_rank" in header:
            required.add("global_rank")
    elif percentile_column is not None:
        required.add(percentile_column)
    missing = required - header
    if missing or (percentile_column is None and not recompute_global_ranking):
        if percentile_column is None:
            missing.add("global_percentile (or legacy score_percentile)")
        raise ValueError(f"{source} is missing {sorted(missing)}.")
    excluded = _normalise_excluded_timestamps(excluded_timestamps)
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(source, usecols=sorted(required), chunksize=chunksize):
        chunk["location"] = chunk["location"].astype(str)
        chunk["timestamp"] = pd.to_datetime(
            chunk["timestamp"], errors="raise", utc=True
        ).dt.tz_convert(None)
        if len(excluded):
            chunk = chunk.loc[~chunk["timestamp"].isin(excluded)].copy()
            if chunk.empty:
                continue
        if recompute_global_ranking:
            score = pd.to_numeric(chunk["anomaly_score"], errors="coerce")
            if not np.isfinite(score).all():
                raise ValueError("STGAN anomaly_score must be finite.")
            chunk["anomaly_score"] = score
            if "global_rank" in chunk:
                rank = pd.to_numeric(chunk["global_rank"], errors="coerce")
                if not np.isfinite(rank).all():
                    raise ValueError("STGAN global_rank must be finite.")
                chunk["global_rank"] = rank
            parts.append(chunk[list(required)])
        else:
            percentile = pd.to_numeric(chunk[percentile_column], errors="coerce")
            if not np.isfinite(percentile).all() or bool(
                ((percentile < 0) | (percentile > 100)).any()
            ):
                raise ValueError(
                    f"STGAN {percentile_column} must be finite and in [0, 100]."
                )
            chunk["decision_coordinate"] = 100.0 - percentile
            parts.append(chunk[["location", "timestamp", "decision_coordinate"]])
    if not parts:
        raise ValueError("No eligible STGAN coordinates were found.")
    result = pd.concat(parts, ignore_index=True)
    ranking_source = percentile_column
    if recompute_global_ranking:
        if "global_rank" in result:
            order = np.argsort(result["global_rank"].to_numpy(float), kind="stable")
            ranking_source = "filtered_global_rank"
        else:
            order = np.argsort(
                -result["anomaly_score"].to_numpy(float), kind="stable"
            )
            ranking_source = "filtered_anomaly_score"
        ranks = np.empty(len(result), dtype=np.int64)
        ranks[order] = np.arange(1, len(result) + 1, dtype=np.int64)
        # A point of rank r enters as soon as top-K contains ceil(N*K/100)
        # points. nextafter keeps exact integer boundaries on the selective side.
        boundary = 100.0 * (ranks - 1.0) / len(result)
        result["decision_coordinate"] = np.nextafter(boundary, np.inf)
        result = result[["location", "timestamp", "decision_coordinate"]]
    if result.duplicated(["location", "timestamp"]).any():
        raise ValueError("STGAN contains duplicate location/timestamp rows.")
    result.attrs["percentile_column"] = ranking_source
    result.attrs["recomputed_global_ranking"] = bool(recompute_global_ranking)
    result.attrs["excluded_timestamps"] = int(len(excluded))
    result.attrs["eligible_coordinates"] = int(len(result))
    return result


def join_detector_errors(
    errors: pd.DataFrame,
    coordinates: pd.DataFrame,
    *,
    min_match_fraction: float = 0.95,
) -> pd.DataFrame:
    """Join one detector decision coordinate to exact forecast target rows."""
    if not 0 < float(min_match_fraction) <= 1:
        raise ValueError("min_match_fraction must be in (0, 1].")
    result = errors.merge(
        coordinates,
        on=["location", "timestamp"],
        how="inner",
        validate="many_to_one",
    )
    coverage = len(result) / max(len(errors), 1)
    if coverage < float(min_match_fraction):
        raise ValueError(
            f"Detector/prediction overlap {coverage:.2%} is below "
            f"{float(min_match_fraction):.2%}."
        )
    result.attrs["match_fraction"] = float(coverage)
    result.attrs["prediction_rows"] = int(len(errors))
    result.attrs["matched_rows"] = int(len(result))
    return result


def _metric_row(
    *,
    detector: str,
    horizon: int,
    threshold: float,
    normal_abs: float,
    normal_square: float,
    n_normal: int,
    rare_abs: float,
    rare_square: float,
    n_rare: int,
    total_n: int,
) -> dict[str, float | int | str]:
    mae_normal = normal_abs / n_normal if n_normal else np.nan
    mae_rare = rare_abs / n_rare if n_rare else np.nan
    rmse_normal = np.sqrt(normal_square / n_normal) if n_normal else np.nan
    rmse_rare = np.sqrt(rare_square / n_rare) if n_rare else np.nan
    return {
        "detector": detector,
        "horizon_hours": int(horizon),
        "decision_threshold": float(threshold),
        "n_normal": int(n_normal),
        "n_rare": int(n_rare),
        "rare_fraction": n_rare / total_n if total_n else np.nan,
        "mae_normal": mae_normal,
        "mae_rare": mae_rare,
        "rmse_normal": rmse_normal,
        "rmse_rare": rmse_rare,
        "mae_gap": mae_rare - mae_normal,
        "rmse_gap": rmse_rare - rmse_normal,
        "mae_ratio": mae_rare / mae_normal if mae_normal else np.nan,
        "rmse_ratio": rmse_rare / rmse_normal if rmse_normal else np.nan,
    }


def sensitivity_sweep(
    joined: pd.DataFrame,
    thresholds: Sequence[float],
    *,
    detector: str,
    rare_when: str,
) -> pd.DataFrame:
    """Compute metrics and absolute-error Tukey statistics at each cutoff.

    The quartiles/whiskers describe individual absolute errors, not confidence
    intervals of MAE or predictive intervals. RMSE remains a pooled metric.

    ``rare_when='coordinate_ge_threshold'`` implements MTGFlow ``score >=
    Q3+k*IQR``. ``rare_when='coordinate_le_threshold'`` implements STGAN top-K,
    where the coordinate is ``100 - global_percentile`` (or the legacy
    ``score_percentile`` alias).
    """
    if rare_when not in {"coordinate_ge_threshold", "coordinate_le_threshold"}:
        raise ValueError("Unsupported rare_when rule.")
    sweep_values = np.asarray(tuple(thresholds), dtype=float)
    if sweep_values.size == 0 or not np.isfinite(sweep_values).all():
        raise ValueError("Sensitivity thresholds must be finite and non-empty.")
    rows: list[dict[str, float | int | str]] = []
    for horizon, frame in joined.groupby("horizon_hours", observed=True):
        coordinate = frame["decision_coordinate"].to_numpy(dtype=float)
        absolute = frame["abs_error"].to_numpy(dtype=float)
        square = frame["squared_error"].to_numpy(dtype=float)
        if not np.isfinite(np.column_stack([coordinate, absolute, square])).all():
            raise ValueError("Sensitivity inputs must be finite.")
        order = np.argsort(coordinate, kind="stable")
        coordinate = coordinate[order]
        absolute = absolute[order]
        square = square[order]
        prefix_abs = np.concatenate(([0.0], np.cumsum(absolute, dtype=np.float64)))
        prefix_square = np.concatenate(([0.0], np.cumsum(square, dtype=np.float64)))
        total_n = len(coordinate)
        for threshold in sweep_values:
            if rare_when == "coordinate_ge_threshold":
                split = int(np.searchsorted(coordinate, threshold, side="left"))
                n_normal, n_rare = split, total_n - split
                normal_abs, normal_square = prefix_abs[split], prefix_square[split]
                rare_abs = prefix_abs[-1] - normal_abs
                rare_square = prefix_square[-1] - normal_square
            else:
                split = int(np.searchsorted(coordinate, threshold, side="right"))
                n_rare, n_normal = split, total_n - split
                rare_abs, rare_square = prefix_abs[split], prefix_square[split]
                normal_abs = prefix_abs[-1] - rare_abs
                normal_square = prefix_square[-1] - rare_square
            rows.append(
                _metric_row(
                    detector=detector,
                    horizon=int(horizon),
                    threshold=float(threshold),
                    normal_abs=float(normal_abs),
                    normal_square=float(normal_square),
                    n_normal=n_normal,
                    rare_abs=float(rare_abs),
                    rare_square=float(rare_square),
                    n_rare=n_rare,
                    total_n=total_n,
                )
            )
            normal_values, rare_values = (
                (absolute[:split], absolute[split:])
                if rare_when == "coordinate_ge_threshold"
                else (absolute[split:], absolute[:split])
            )
            for group, values in (("normal", normal_values), ("rare", rare_values)):
                rows[-1].update(_boxplot_stats(values, f"abs_error_{group}"))
    return pd.DataFrame(rows).sort_values(
        ["detector", "horizon_hours", "decision_threshold"]
    ).reset_index(drop=True)


def plot_mae_dispersion(axis, frame: pd.DataFrame, *, x_column="decision_threshold"):
    """Plot pooled MAE with darker Q1-Q3 and lighter Tukey-whisker bands.

    Outliers contribute to MAE but are not drawn individually. Empty groups
    remain NaN gaps. Colours match the normal/rare curves in existing reports.
    """
    data = frame.sort_values(x_column)
    x = data[x_column].to_numpy(float)
    for group, label, color in (
        ("normal", "Normali", "tab:blue"),
        ("rare", "Rari/anomali", "tab:orange"),
    ):
        prefix = f"abs_error_{group}"
        axis.fill_between(
            x, data[f"{prefix}_whisker_low"].to_numpy(float),
            data[f"{prefix}_whisker_high"].to_numpy(float),
            color=color, alpha=0.10, linewidth=0,
            label=f"{label}: baffi Tukey (1.5 IQR)",
        )
        axis.fill_between(
            x, data[f"{prefix}_q1"].to_numpy(float),
            data[f"{prefix}_q3"].to_numpy(float),
            color=color, alpha=0.25, linewidth=0,
            label=f"{label}: 25-75% errori assoluti",
        )
        axis.plot(x, data[f"mae_{group}"], marker="o", color=color,
                  label=f"{label}: MAE", zorder=3)
