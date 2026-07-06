from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from physiq_pv.data.pvgis_stgnn_dataset import (
    DAYTIME_IRRADIANCE_THRESHOLD_WM2,
    GROUP_NORMAL,
    GROUP_RARE,
    SPECIFIC_ANOMALY_LABELS,
)

DAYTIME_PRODUCTION_BINS = (
    ("daytime_0_20", 0.0, 20.0),
    ("daytime_20_40", 20.0, 40.0),
    ("daytime_40_60", 40.0, 60.0),
    ("daytime_60_80", 60.0, 80.0),
    ("daytime_80_100", 80.0, 100.0),
    ("daytime_gt_100", 100.0, None),
)

DEFAULT_PICP_MULTIPLIERS = (1.0, 1.96, 2.5, 3.0, 4.0, 5.0, 8.0, 10.0)
INTERVAL_COLUMNS = {
    "pi": ("lower_pi", "upper_pi"),
    "gaussian": ("lower_gaussian", "upper_gaussian"),
    "calibrated": ("lower_calibrated", "upper_calibrated"),
}
_REQUIRED_BASE = {
    "y_true", "anomaly_group", "anomaly_label", "solar_irradiance_poa_target",
}


def _pred_col(predictions: pd.DataFrame) -> str:
    if "y_pred_mean" in predictions.columns:
        return "y_pred_mean"
    if "y_pred" in predictions.columns:
        return "y_pred"
    raise ValueError("Predictions need y_pred_mean or y_pred.")


def _std_col(predictions: pd.DataFrame) -> str:
    if "y_pred_std_raw" in predictions.columns:
        return "y_pred_std_raw"
    if "y_pred_std" in predictions.columns:
        return "y_pred_std"
    raise ValueError("Predictions need y_pred_std_raw or y_pred_std.")


def validate_predictions(predictions: pd.DataFrame, interval: str) -> None:
    if interval not in INTERVAL_COLUMNS:
        raise ValueError(
            f"Unknown interval '{interval}'. Available: {sorted(INTERVAL_COLUMNS)}."
        )
    missing = (_REQUIRED_BASE | set(INTERVAL_COLUMNS[interval])) - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")
    _pred_col(predictions)
    _std_col(predictions)
    solar = predictions["solar_irradiance_poa_target"].to_numpy(dtype=float)
    if not np.all(np.isfinite(solar)):
        raise ValueError("solar_irradiance_poa_target contains non-finite values.")


def build_strata_masks(
    predictions: pd.DataFrame,
    threshold_wm2: float = DAYTIME_IRRADIANCE_THRESHOLD_WM2,
) -> dict[str, np.ndarray]:
    solar = predictions["solar_irradiance_poa_target"].to_numpy(dtype=float)
    y_true = predictions["y_true"].to_numpy(dtype=float)
    groups = predictions["anomaly_group"].to_numpy()
    daytime = solar > threshold_wm2
    nighttime = ~daytime
    normal = groups == GROUP_NORMAL
    rare = groups == GROUP_RARE

    masks: dict[str, np.ndarray] = {
        "global": np.ones(len(predictions), dtype=bool),
        "daytime": daytime,
        "nighttime": nighttime,
        "normal": normal,
        "rare_extreme": rare,
        "normal_daytime": normal & daytime,
        "rare_extreme_daytime": rare & daytime,
        "normal_nighttime": normal & nighttime,
        "rare_extreme_nighttime": rare & nighttime,
    }
    label_sets = predictions["anomaly_label"].fillna("").astype(str).apply(
        lambda value: {part.strip() for part in value.split(",") if part.strip()}
    )
    for label in SPECIFIC_ANOMALY_LABELS:
        masks[f"label:{label}"] = label_sets.apply(
            lambda values: label in values
        ).to_numpy()
    for name, lower, upper in DAYTIME_PRODUCTION_BINS:
        mask = daytime & (y_true >= lower)
        if upper is not None:
            mask &= y_true < upper
        masks[name] = mask
    return masks


def _quantile_block(values: np.ndarray, prefix: str, qs=(50, 90, 95)) -> dict[str, float]:
    out = {}
    for q in qs:
        out[f"p{q}_{prefix}"] = (
            float(np.percentile(values, q)) if values.size else float("nan")
        )
    return out


def interval_miss_row(
    sub: pd.DataFrame,
    group: str,
    interval: str = "pi",
    eps: float = 1e-6,
) -> dict[str, float]:
    lower_col, upper_col = INTERVAL_COLUMNS[interval]
    n = len(sub)
    row: dict[str, float] = {"group": group, "n": int(n)}
    nan_keys = (
        [
            "inside_interval_count", "outside_interval_count",
            "inside_interval_pct", "outside_interval_pct",
            "above_interval_pct", "below_interval_pct",
            "mean_residual", "median_residual",
        ]
        + [
            f"{s}_{p}_distance"
            for p in ("outside", "above", "below")
            for s in ("mean", "median", "p90", "p95")
        ]
        + [
            "p50_required_multiplier",
            "p90_required_multiplier",
            "p95_required_multiplier",
        ]
    )
    if n == 0:
        row.update({k: float("nan") for k in nan_keys})
        row["inside_interval_count"] = 0
        row["outside_interval_count"] = 0
        return row

    y_true = sub["y_true"].to_numpy(dtype=float)
    y_pred = sub[_pred_col(sub)].to_numpy(dtype=float)
    y_std = sub[_std_col(sub)].to_numpy(dtype=float)
    lower = sub[lower_col].to_numpy(dtype=float)
    upper = sub[upper_col].to_numpy(dtype=float)

    inside = (y_true >= lower) & (y_true <= upper)
    below_distance = np.maximum(lower - y_true, 0.0)
    above_distance = np.maximum(y_true - upper, 0.0)
    outside_distance = below_distance + above_distance
    above_miss = above_distance > 0.0
    below_miss = below_distance > 0.0
    outside = ~inside
    residual = y_pred - y_true

    row.update(
        {
            "inside_interval_count": int(inside.sum()),
            "outside_interval_count": int(outside.sum()),
            "inside_interval_pct": float(inside.mean()),
            "outside_interval_pct": float(outside.mean()),
            "above_interval_pct": float(above_miss.mean()),
            "below_interval_pct": float(below_miss.mean()),
            "mean_residual": float(residual.mean()),
            "median_residual": float(np.median(residual)),
        }
    )
    for prefix, dist, mask in (
        ("outside", outside_distance, outside),
        ("above", above_distance, above_miss),
        ("below", below_distance, below_miss),
    ):
        vals = dist[mask]
        row[f"mean_{prefix}_distance"] = float(vals.mean()) if vals.size else float("nan")
        row[f"median_{prefix}_distance"] = (
            float(np.median(vals)) if vals.size else float("nan")
        )
        row.update(_quantile_block(vals, f"{prefix}_distance", qs=(90, 95)))

    required = np.abs(y_true - y_pred) / (y_std + eps)
    row.update(_quantile_block(required, "required_multiplier", qs=(50, 90, 95)))
    return row


def compute_interval_miss_table(
    predictions: pd.DataFrame,
    interval: str = "pi",
    eps: float = 1e-6,
    threshold_wm2: float = DAYTIME_IRRADIANCE_THRESHOLD_WM2,
) -> pd.DataFrame:
    validate_predictions(predictions, interval)
    masks = build_strata_masks(predictions, threshold_wm2)
    rows = [
        interval_miss_row(predictions.loc[mask], group, interval=interval, eps=eps)
        for group, mask in masks.items()
    ]
    return pd.DataFrame(rows)


def compute_picp_curve_table(
    predictions: pd.DataFrame,
    multipliers: Sequence[float] = DEFAULT_PICP_MULTIPLIERS,
    threshold_wm2: float = DAYTIME_IRRADIANCE_THRESHOLD_WM2,
) -> pd.DataFrame:
    validate_predictions(predictions, "pi")
    masks = build_strata_masks(predictions, threshold_wm2)
    y_true = predictions["y_true"].to_numpy(dtype=float)
    y_pred = predictions[_pred_col(predictions)].to_numpy(dtype=float)
    y_std = predictions[_std_col(predictions)].to_numpy(dtype=float)

    rows = []
    for group, mask in masks.items():
        row: dict[str, float] = {"group": group, "n": int(mask.sum())}
        for k in multipliers:
            key = f"picp_k_{k:g}"
            if not mask.any():
                row[key] = float("nan")
                continue
            covered = (
                (y_true[mask] >= y_pred[mask] - k * y_std[mask])
                & (y_true[mask] <= y_pred[mask] + k * y_std[mask])
            )
            row[key] = float(covered.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _verdict(p95_multiplier: float) -> str:
    if not np.isfinite(p95_multiplier):
        return "n/a"
    if p95_multiplier <= 3.0:
        return "moderate widening/calibration could suffice"
    if p95_multiplier >= 6.0:
        return "center biased or std collapsed; widening alone will not fix it"
    return "borderline: partial fix from widening, residual center bias likely"


def render_interval_miss_report(
    miss_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    meta: Optional[dict] = None,
) -> str:
    meta = meta or {}
    lines: list[str] = [
        "# PVGIS ST-GNN interval-miss diagnostics (post-hoc, eval-only)\n",
        "Computed from saved predictions only. Distances are in watt and "
        "conditional on missed rows. Anomaly labels are eval-only.\n",
    ]
    if meta:
        lines.append("## Inputs\n")
        for key, value in meta.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")

    def _table(df: pd.DataFrame, float_fmt: str = "{:.4g}") -> list[str]:
        cols = list(df.columns)
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in df.iterrows():
            cells = []
            for col in cols:
                value = r[col]
                if isinstance(value, float):
                    cells.append(float_fmt.format(value) if np.isfinite(value) else "-")
                else:
                    cells.append(str(value))
            out.append("| " + " | ".join(cells) + " |")
        return out

    lines.append("## 1. Interval-miss distances per stratum\n")
    lines.extend(_table(miss_df))
    lines.append("")
    lines.append("## 2. PICP curve - mean +/- k*std\n")
    lines.extend(_table(curve_df))
    lines.append("")
    lines.append("## 3. Widen-vs-bias verdict per stratum\n")
    lines.append("| group | n | p95_required_multiplier | above% | below% | mean_residual | verdict |")
    lines.append("|---|---|---|---|---|---|---|")

    def _fmt(value, fmt="{:.3g}"):
        return fmt.format(value) if isinstance(value, float) and np.isfinite(value) else "-"

    for _, row in miss_df.iterrows():
        p95 = row.get("p95_required_multiplier", float("nan"))
        lines.append(
            f"| {row['group']} | {int(row['n'])} | {_fmt(p95)} | "
            f"{_fmt(row.get('above_interval_pct', float('nan')))} | "
            f"{_fmt(row.get('below_interval_pct', float('nan')))} | "
            f"{_fmt(row.get('mean_residual', float('nan')))} | {_verdict(p95)} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"
