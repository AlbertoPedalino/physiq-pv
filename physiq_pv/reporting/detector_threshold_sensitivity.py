"""Post-hoc forecast-error sensitivity to MTGFlow and STGAN decisions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def load_prediction_errors(
    path: str | Path,
    *,
    horizons: Sequence[int] = (1, 6),
    daytime_threshold_wm2: float | None = 10.0,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load finite target-time errors for the requested direct outputs."""
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
    missing = required - header
    if missing:
        raise ValueError(f"{source} is missing {sorted(missing)}.")

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
        parts.append(
            chunk[[
                "location", "timestamp", "horizon_hours", "abs_error",
                "squared_error",
            ]]
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


def load_mtgflow_coordinates(
    score_path: str | Path,
    threshold_table: pd.DataFrame,
    *,
    reference_k: float = 1.5,
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

    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(source, usecols=sorted(required), chunksize=chunksize):
        chunk["location"] = chunk["location"].astype(str)
        chunk["timestamp"] = pd.to_datetime(
            chunk["timestamp"], errors="raise", utc=True
        ).dt.tz_convert(None)
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
    return result


def load_stgan_coordinates(
    score_path: str | Path,
    *,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Convert STGAN percentile to the top-K percentage that includes a row."""
    source = Path(score_path)
    required = {"location", "timestamp", "score_percentile"}
    header = set(pd.read_csv(source, nrows=0).columns)
    missing = required - header
    if missing:
        raise ValueError(f"{source} is missing {sorted(missing)}.")
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(source, usecols=sorted(required), chunksize=chunksize):
        chunk["location"] = chunk["location"].astype(str)
        chunk["timestamp"] = pd.to_datetime(
            chunk["timestamp"], errors="raise", utc=True
        ).dt.tz_convert(None)
        percentile = pd.to_numeric(chunk["score_percentile"], errors="coerce")
        if not np.isfinite(percentile).all() or bool(
            ((percentile < 0) | (percentile > 100)).any()
        ):
            raise ValueError("STGAN score_percentile must be finite and in [0, 100].")
        chunk["decision_coordinate"] = 100.0 - percentile
        parts.append(chunk[["location", "timestamp", "decision_coordinate"]])
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["location", "timestamp"]).any():
        raise ValueError("STGAN contains duplicate location/timestamp rows.")
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
    """Compute exact normal/rare MAE and RMSE for every horizon and cutoff.

    ``rare_when='coordinate_ge_threshold'`` implements MTGFlow ``score >=
    Q3+k*IQR``. ``rare_when='coordinate_le_threshold'`` implements STGAN top-K,
    where the coordinate is ``100 - score_percentile``.
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
    return pd.DataFrame(rows).sort_values(
        ["detector", "horizon_hours", "decision_threshold"]
    ).reset_index(drop=True)
