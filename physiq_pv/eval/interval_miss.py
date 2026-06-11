"""
Post-hoc interval-miss diagnostics for PVGIS-only ST-GNN predictions.

Eval-only: operates on an already-written predictions.csv (physical watt
space), never touches the model, training, loss or MC Dropout. Anomaly labels
are used ONLY to stratify. Answers two questions per stratum:

  1. When y_true falls outside the predictive interval, by HOW MANY WATT?
     (`*_outside/above/below_distance`, conditional on the missed rows)
  2. Would widening the band fix coverage, or is the predictive CENTER biased?
     (`required_multiplier` = |y_true - mean| / (std + eps) quantiles, plus a
     PICP-vs-k curve for mean +/- k*std)

Strata names match the runner's residual-bias diagnostics
(global/daytime/.../daytime_gt_100/label:*) so tables join cleanly.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from physiq_pv.data.pvgis_stgnn_dataset import (
    DAYTIME_IRRADIANCE_THRESHOLD_WM2,
    GROUP_NORMAL,
    GROUP_RARE,
    SPECIFIC_ANOMALY_LABELS,
)

# Same fixed daytime production bins as the runner's residual diagnostics.
DAYTIME_PRODUCTION_BINS = (
    ("daytime_0_20", 0.0, 20.0),
    ("daytime_20_40", 20.0, 40.0),
    ("daytime_40_60", 40.0, 60.0),
    ("daytime_60_80", 60.0, 80.0),
    ("daytime_80_100", 80.0, 100.0),
    ("daytime_gt_100", 100.0, None),
)

# mean +/- k*std PICP curve grid (k=1.96 ~ the Gaussian diagnostic band).
DEFAULT_PICP_MULTIPLIERS = (1.0, 1.96, 2.5, 3.0, 4.0, 5.0, 8.0, 10.0)

INTERVAL_COLUMNS = {
    "pi": ("lower_pi", "upper_pi"),
    "gaussian": ("lower_gaussian", "upper_gaussian"),
    "calibrated": ("lower_calibrated", "upper_calibrated"),
}

_REQUIRED_BASE = {"y_true", "anomaly_group", "anomaly_label",
                  "solar_irradiance_poa_target"}


def _pred_col(predictions: pd.DataFrame) -> str:
    if "y_pred_mean" in predictions.columns:
        return "y_pred_mean"
    if "y_pred" in predictions.columns:
        return "y_pred"
    raise ValueError("Predictions need y_pred_mean or y_pred.")


def _std_col(predictions: pd.DataFrame) -> str:
    # y_pred_std_raw is the diagnostic MC spread; y_pred_std is its legacy alias.
    if "y_pred_std_raw" in predictions.columns:
        return "y_pred_std_raw"
    if "y_pred_std" in predictions.columns:
        return "y_pred_std"
    raise ValueError("Predictions need y_pred_std_raw or y_pred_std (MC Dropout run).")


def validate_predictions(predictions: pd.DataFrame, interval: str) -> None:
    """Fail fast with the list of missing columns for the requested interval."""
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
) -> Dict[str, np.ndarray]:
    """Boolean mask per stratum; names match the runner's residual diagnostics."""
    solar = predictions["solar_irradiance_poa_target"].to_numpy(dtype=float)
    y_true = predictions["y_true"].to_numpy(dtype=float)
    groups = predictions["anomaly_group"].to_numpy()
    daytime = solar > threshold_wm2
    nighttime = ~daytime
    normal = groups == GROUP_NORMAL
    rare = groups == GROUP_RARE

    masks: Dict[str, np.ndarray] = {
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


def _quantile_block(values: np.ndarray, prefix: str, qs=(50, 90, 95)) -> Dict[str, float]:
    out = {}
    for q in qs:
        key = f"p{q}_{prefix}"
        out[key] = float(np.percentile(values, q)) if values.size else float("nan")
    return out


def interval_miss_row(
    sub: pd.DataFrame,
    group: str,
    interval: str = "pi",
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Interval-miss diagnostics for one stratum (watt space).

    Distances are CONDITIONAL on the missed rows: `*_outside_distance` is
    computed over outside rows only, `*_above/below_distance` over the rows
    missed on that side ("when it falls outside, by how many watt"). A stratum
    with no misses reports NaN distances. `required_multiplier` is computed on
    ALL rows of the stratum: |y_true - y_pred_mean| / (std + eps).
    """
    lower_col, upper_col = INTERVAL_COLUMNS[interval]
    n = len(sub)
    row: Dict[str, float] = {"group": group, "n": int(n)}
    nan_keys = (
        ["inside_interval_count", "outside_interval_count",
         "inside_interval_pct", "outside_interval_pct",
         "above_interval_pct", "below_interval_pct",
         "mean_residual", "median_residual"]
        + [f"{s}_{p}_distance" for p in ("outside", "above", "below")
           for s in ("mean", "median", "p90", "p95")]
        + ["p50_required_multiplier", "p90_required_multiplier",
           "p95_required_multiplier"]
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
    outside = ~inside
    below_distance = np.maximum(lower - y_true, 0.0)   # watt
    above_distance = np.maximum(y_true - upper, 0.0)   # watt
    outside_distance = below_distance + above_distance
    above_miss = above_distance > 0.0
    below_miss = below_distance > 0.0
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
    """One interval-miss row per stratum (watt space, eval-only)."""
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
    """
    PICP of the band mean +/- k*std per stratum and k.

    Diagnostic only: shows whether a (global) std rescale could reach the
    coverage target, independently of the primary empirical-quantile PI.
    """
    validate_predictions(predictions, "pi")  # base columns; band built from std
    masks = build_strata_masks(predictions, threshold_wm2)
    y_true = predictions["y_true"].to_numpy(dtype=float)
    y_pred = predictions[_pred_col(predictions)].to_numpy(dtype=float)
    y_std = predictions[_std_col(predictions)].to_numpy(dtype=float)

    rows = []
    for group, mask in masks.items():
        row: Dict[str, float] = {"group": group, "n": int(mask.sum())}
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
        return "center biased or std collapsed — widening alone will not fix it"
    return "borderline — partial fix from widening, residual center bias likely"


def render_interval_miss_report(
    miss_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    meta: Optional[dict] = None,
) -> str:
    """Markdown report: miss-distance table, PICP curve, per-stratum verdicts."""
    meta = meta or {}
    lines: List[str] = []
    lines.append("# PVGIS ST-GNN interval-miss diagnostics (post-hoc, eval-only)\n")
    lines.append(
        "Computed from saved predictions only — no training, no model, no loss "
        "change. Distances are in **watt** and **conditional on the missed rows** "
        "(`mean_outside_distance` = mean watt outside the band, over outside rows "
        "only). `required_multiplier = |y_true - y_pred_mean| / (std + eps)` is "
        "computed on all rows of the stratum. Anomaly labels are eval-only.\n"
    )
    if meta:
        lines.append("## Inputs\n")
        for key, value in meta.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")

    def _table(df: pd.DataFrame, float_fmt: str = "{:.4g}") -> List[str]:
        cols = list(df.columns)
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in df.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    cells.append(float_fmt.format(v) if np.isfinite(v) else "—")
                else:
                    cells.append(str(v))
            out.append("| " + " | ".join(cells) + " |")
        return out

    lines.append("## 1. Interval-miss distances per stratum\n")
    lines.extend(_table(miss_df))
    lines.append("")

    lines.append("## 2. PICP curve — mean ± k·std\n")
    lines.append(
        "Diagnostic band from the MC std; shows the coverage a pure std rescale "
        "would buy (k=1.96 = the Gaussian diagnostic band).\n"
    )
    lines.extend(_table(curve_df))
    lines.append("")

    lines.append("## 3. Widen-vs-bias verdict per stratum\n")
    lines.append(
        "Heuristic on `p95_required_multiplier`: <= 3 -> widening/calibration "
        "plausible; >= 6 -> predictive center too biased (or std too small) — "
        "fix the point forecast. Above/below asymmetry + mean_residual tell the "
        "direction of the bias.\n"
    )
    def _f(v, fmt="{:.3g}"):
        return fmt.format(v) if isinstance(v, float) and np.isfinite(v) else "—"

    lines.append("| group | n | p95_required_multiplier | above% | below% | mean_residual | verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in miss_df.iterrows():
        p95 = r.get("p95_required_multiplier", float("nan"))
        lines.append(
            f"| {r['group']} | {int(r['n'])} | {_f(p95)} | "
            f"{_f(r.get('above_interval_pct', float('nan')))} | "
            f"{_f(r.get('below_interval_pct', float('nan')))} | "
            f"{_f(r.get('mean_residual', float('nan')))} | {_verdict(p95)} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"
