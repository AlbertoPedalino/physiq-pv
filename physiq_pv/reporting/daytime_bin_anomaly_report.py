#!/usr/bin/env python
"""Post-hoc daytime production-bin x anomaly report for one PVGIS run.

Reads predictions.csv only. Labels stratify saved predictions; training and
model state are untouched. Default bins use one global PVGIS reference peak:
``100 * y_true / q99_daytime(y_true)``.

Usage:
  python scripts/analyze_pvgis_daytime_report.py \
      --predictions outputs/<run>/predictions.csv \
      --out-dir outputs/<run>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from physiq_pv.reporting.posthoc_outputs import PRODUCTION_BINS as RAW_PRODUCTION_BINS

# Kept local so this analysis script stays standalone.
DAYTIME_IRRADIANCE_THRESHOLD_WM2 = 10.0
GROUP_NORMAL = "normal"
GROUP_RARE = "rare_or_extreme"
SPECIFIC_ANOMALY_LABELS = [
    "unusually_low_solar_potential",
    "unusually_high_solar_potential",
    "extreme_temperature_condition",
    "extreme_wind_condition",
]

CATEGORY_ORDER = [
    "normal",
    "rare_extreme",
    "unusually_low_solar_potential",
    "unusually_high_solar_potential",
    "extreme_temperature_condition",
    "extreme_wind_condition",
]
UNCERTAINTY_CATEGORIES = CATEGORY_ORDER[1:]

PERCENT_PRODUCTION_BINS = [
    ("daytime_0_20_pct", 0.0, 20.0),
    ("daytime_20_40_pct", 20.0, 40.0),
    ("daytime_40_60_pct", 40.0, 60.0),
    ("daytime_60_80_pct", 60.0, 80.0),
    ("daytime_80_100_pct", 80.0, None),
]
REFERENCE_PRODUCTION_PEAKS_FILE = "reference_production_peaks.csv"


# --------------------------------------------------------------------------- #
# Column resolution (robust to the runner's alias columns)
# --------------------------------------------------------------------------- #
def _pick(available: set[str], candidates: list[str], what: str) -> str:
    for c in candidates:
        if c in available:
            return c
    raise SystemExit(
        f"predictions.csv has no column for {what}; tried {candidates}. "
        f"Present columns: {sorted(available)}"
    )


def resolve_columns(path: str) -> dict[str, str | None]:
    header = pd.read_csv(path, nrows=0)
    cols = set(header.columns)
    resolved = {
        "y_true": _pick(cols, ["y_true"], "y_true"),
        "y_pred": _pick(cols, ["y_pred_mean", "y_pred"], "mean prediction"),
        "y_std": _pick(cols, ["y_pred_std_raw", "y_pred_std"], "predictive std"),
        "lower_pi": _pick(cols, ["lower_pi"], "lower interval"),
        "upper_pi": _pick(cols, ["upper_pi"], "upper interval"),
        "epistemic_std": next(
            (c for c in ("epistemic_std", "y_pred_epistemic_std") if c in cols),
            None,
        ),
        "aleatoric_std": next(
            (c for c in ("aleatoric_std", "y_pred_aleatoric_std") if c in cols),
            None,
        ),
        "solar": _pick(
            cols,
            ["solar_irradiance_poa_target", "solar_irradiance_poa", "ghi_target"],
            "target-time irradiance (daytime filter)",
        ),
        "group": _pick(
            cols,
            ["event_group", "anomaly_group"],
            "regional event/anomaly group",
        ),
        "label": _pick(cols, ["anomaly_label"], "anomaly_label"),
    }
    resolved["timestamp"] = next(
        (c for c in ("timestamp", "target_timestamp", "time") if c in cols), None
    )
    resolved["location"] = next(
        (c for c in ("location", "location_id", "node_id") if c in cols), None
    )
    return resolved


# --------------------------------------------------------------------------- #
# Chunked load
# --------------------------------------------------------------------------- #
_METRIC_COLS = ["y_true", "y_pred", "y_std", "lower_pi", "upper_pi"]


def _prediction_csv_reader(path: str, col: dict[str, str | None], chunksize: int):
    """Open the large prediction CSV without fragmented dtype inference."""
    usecols = [c for c in dict.fromkeys(col.values()) if c is not None]
    text_dtypes = {
        col[role]: "string"
        for role in ("group", "label", "timestamp", "location")
        if col[role] is not None
    }
    return pd.read_csv(
        path,
        usecols=usecols,
        dtype=text_dtypes,
        chunksize=chunksize,
        low_memory=False,
    )


def load_daytime(path: str, col: dict[str, str | None], threshold: float,
                 chunksize: int) -> tuple[pd.DataFrame, dict]:
    """Read predictions in chunks and keep finite daytime rows."""
    keep = []
    n_total = 0
    n_candidate = 0
    n_invalid = 0
    y_true_min = float("inf")
    y_true_max = float("-inf")
    reader = _prediction_csv_reader(path, col, chunksize)
    for chunk in reader:
        n_total += len(chunk)
        solar = pd.to_numeric(chunk[col["solar"]], errors="coerce").to_numpy(float)
        day_mask = solar > threshold
        if not day_mask.any():
            continue
        sub = chunk.loc[day_mask]
        n_candidate += len(sub)
        group = sub[col["group"]].to_numpy().astype(str)
        out = pd.DataFrame({
            c: pd.to_numeric(sub[col[c]], errors="coerce").to_numpy(np.float64)
            for c in _METRIC_COLS
        })
        for component in ("epistemic_std", "aleatoric_std"):
            if col[component] is not None:
                out[component] = pd.to_numeric(
                    sub[col[component]], errors="coerce"
                ).to_numpy(np.float64)
        if col["timestamp"] is not None:
            out["timestamp"] = sub[col["timestamp"]].to_numpy()
        if col["location"] is not None:
            out["location"] = sub[col["location"]].to_numpy()
        padded = "," + sub[col["label"]].fillna("").astype(str) + ","
        specific_rare = np.zeros(len(out), dtype=bool)
        for lab in SPECIFIC_ANOMALY_LABELS:
            lab_mask = padded.str.contains(
                "," + lab + ",", regex=False
            ).to_numpy()
            out[f"has_{lab}"] = lab_mask
            specific_rare |= lab_mask
        out["is_rare"] = (group == GROUP_RARE) | specific_rare
        out["is_normal"] = (group == GROUP_NORMAL) & ~out["is_rare"]
        finite = np.isfinite(out[_METRIC_COLS].to_numpy(float)).all(axis=1)
        n_invalid += int((~finite).sum())
        if finite.any():
            kept = out.loc[finite].reset_index(drop=True)
            keep.append(kept)
            yt = kept["y_true"].to_numpy(float)
            y_true_min = min(y_true_min, float(yt.min()))
            y_true_max = max(y_true_max, float(yt.max()))
    if not keep:
        raise SystemExit(
            f"No valid daytime rows (solar > {threshold}) found in {path}. "
            f"Scanned {n_total} rows ({n_candidate} candidates, {n_invalid} invalid)."
        )
    day = pd.concat(keep, ignore_index=True)
    stats = {
        "total_samples": n_total,
        "daytime_candidates": n_candidate,
        "daytime_valid": len(day),
        "daytime_invalid": n_invalid,
        "nighttime_samples": n_total - n_candidate,
        "y_true_min": y_true_min if np.isfinite(y_true_min) else float("nan"),
        "y_true_max": y_true_max if np.isfinite(y_true_max) else float("nan"),
    }
    print(f"[load] scanned {n_total:,} rows; {n_candidate:,} daytime candidates "
          f"(solar > {threshold} W/m^2); kept {len(day):,} valid, "
          f"skipped {n_invalid:,} invalid; {stats['nighttime_samples']:,} nighttime")
    return day, stats


def add_reference_peak_production_pct(day: pd.DataFrame, quantile: float) -> dict:
    """Add production percentage using one global PVGIS reference peak."""
    q = float(quantile)
    if not (0.0 < q <= 1.0):
        raise SystemExit("--reference-peak-quantile must be in (0, 1].")

    y_true = day["y_true"].to_numpy(float)
    valid = y_true[np.isfinite(y_true) & (y_true > 0.0)]
    if valid.size == 0:
        raise SystemExit(
            "Reference-peak percentage production bins found no positive "
            "daytime PVGIS production values."
        )
    peak = float(np.quantile(valid, q))
    if not np.isfinite(peak) or peak <= 0.0:
        raise SystemExit("Reference peak is non-positive or non-finite.")
    day["production_pct"] = np.clip(
        100.0 * y_true / peak, 0.0, 100.0
    )
    day["reference_peak_w"] = peak
    day["production_reference_w"] = peak
    return {
        "production_bin_basis": "reference_peak_pct",
        "reference_peak_quantile": q,
        "reference_peak_scope": "global_daytime",
        "reference_peak_w": peak,
        "reference_peak_sample_count": int(valid.size),
    }


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #
def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def subset_metrics(sub: pd.DataFrame) -> dict:
    """Point and interval metrics for one subset."""
    n = len(sub)
    if n == 0:
        return {"count": 0, "count_inside_pi": 0, "PICP": float("nan"),
                "MAE": float("nan"), "RMSE": float("nan"), "mean_std": float("nan"),
                "mpiw": float("nan"), "production_peak_nmpil": float("nan"),
                "count_width": 0}
    y_true = sub["y_true"].to_numpy(float)
    y_pred = sub["y_pred"].to_numpy(float)
    y_std = sub["y_std"].to_numpy(float)
    lower = sub["lower_pi"].to_numpy(float)
    upper = sub["upper_pi"].to_numpy(float)
    residual = y_pred - y_true
    inside = (y_true >= lower) & (y_true <= upper)
    width = upper - lower
    width_ok = np.isfinite(width) & (width >= 0.0)
    mpiw = float(np.mean(width[width_ok])) if width_ok.any() else float("nan")
    production_peak_nmpil = float("nan")
    if "production_reference_w" in sub:
        peak = sub["production_reference_w"].to_numpy(float)
        peak_ok = width_ok & np.isfinite(peak) & (peak > 0.0)
        if peak_ok.any():
            production_peak_nmpil = float(np.mean(width[peak_ok] / peak[peak_ok]))
    return {
        "count": int(n),
        "count_inside_pi": int(inside.sum()),
        "PICP": float(np.mean(inside)),
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual ** 2))),
        "mean_std": float(np.mean(y_std)),
        "mpiw": mpiw,
        "production_peak_nmpil": production_peak_nmpil,
        "count_width": int(width_ok.sum()),
    }


def _nmpil(mpiw: float, target_range: float) -> float:
    """NMPIL = MPIW / target_range; NaN when range is missing/non-positive."""
    if (target_range is None or not np.isfinite(target_range)
            or target_range <= 0.0 or not np.isfinite(mpiw)):
        return float("nan")
    return float(mpiw / target_range)


def _clc(nmpil: float, picp: float, gamma: float, eta: float) -> float:
    if not all(np.isfinite(v) for v in (nmpil, picp, gamma, eta)):
        return float("nan")
    return float(nmpil * (1.0 + np.exp(-eta * (picp - gamma))))


def _bin_mask(day: pd.DataFrame, lower: float, upper,
              production_col: str) -> np.ndarray:
    y = day[production_col].to_numpy(float)
    mask = y >= lower
    if upper is not None:
        mask = mask & (y < upper)
    return mask


def _boxplot_stats(values: np.ndarray, prefix: str) -> dict:
    """Exact Tukey boxplot statistics computed from every finite row."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_q1": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_q3": float("nan"),
            f"{prefix}_whisker_low": float("nan"),
            f"{prefix}_whisker_high": float("nan"),
        }
    q1, median, q3 = np.quantile(finite, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    low_candidates = finite[finite >= q1 - 1.5 * iqr]
    high_candidates = finite[finite <= q3 + 1.5 * iqr]
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_q1": float(q1),
        f"{prefix}_median": float(median),
        f"{prefix}_q3": float(q3),
        f"{prefix}_whisker_low": float(np.min(low_candidates)),
        f"{prefix}_whisker_high": float(np.max(high_candidates)),
    }


def category_mask(day: pd.DataFrame, category: str) -> np.ndarray:
    if category == "normal":
        return day["is_normal"].to_numpy(bool)
    if category == "rare_extreme":
        return day["is_rare"].to_numpy(bool)
    return day[f"has_{category}"].to_numpy(bool)


# --------------------------------------------------------------------------- #
# Table builders
# --------------------------------------------------------------------------- #
def build_overview(day: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Single wide row with the exact field names required by the report spec."""
    n_day = len(day)
    n_normal = int(day["is_normal"].sum())
    n_rare = int(day["is_rare"].sum())
    row = {
        "total_samples_all": int(stats["total_samples"]),
        "total_daytime_samples": int(n_day),
        "normal_daytime_samples": int(n_normal),
        "rare_extreme_daytime_samples": int(n_rare),
        "rare_extreme_daytime_pct": _safe(n_rare / n_day) if n_day else float("nan"),
    }
    for lab in SPECIFIC_ANOMALY_LABELS:
        cnt = int(day[f"has_{lab}"].sum())
        row[f"{lab}_count"] = cnt
        row[f"{lab}_pct_of_daytime"] = _safe(cnt / n_day) if n_day else float("nan")
    return pd.DataFrame([row])


def build_bin_summary(day: pd.DataFrame, target_range: float, production_bins,
                      production_col: str) -> pd.DataFrame:
    rows = []
    for bin_name, lo, hi in production_bins:
        m = subset_metrics(day.loc[_bin_mask(day, lo, hi, production_col)])
        rows.append({"bin": bin_name, "count": m["count"], "mae": m["MAE"],
                     "rmse": m["RMSE"], "picp": m["PICP"], "mean_std": m["mean_std"],
                     "mpiw": m["mpiw"], "nmpil": _nmpil(m["mpiw"], target_range),
                     "production_peak_nmpil": m["production_peak_nmpil"]})
    return pd.DataFrame(rows)


def build_bin_category(day: pd.DataFrame, target_range: float, production_bins,
                       production_col: str) -> pd.DataFrame:
    """Metrics for each production-bin x category cell."""
    rows = []
    n_day = len(day)
    category_counts = {
        category: int(category_mask(day, category).sum())
        for category in CATEGORY_ORDER
    }
    for bin_name, lo, hi in production_bins:
        bmask = _bin_mask(day, lo, hi, production_col)
        for cat in CATEGORY_ORDER:
            sub = day.loc[bmask & category_mask(day, cat)]
            m = subset_metrics(sub)
            category_count = category_counts[cat]
            row = {
                "bin": bin_name, "category": cat, "count": m["count"],
                "category_daytime_count": category_count,
                "frequency_within_category": _safe(m["count"] / category_count)
                if category_count else float("nan"),
                "frequency_of_daytime": _safe(m["count"] / n_day)
                if n_day else float("nan"),
                "count_inside_pi": m["count_inside_pi"], "picp": m["PICP"],
                "mae": m["MAE"], "rmse": m["RMSE"],
                "mpiw": m["mpiw"], "nmpil": _nmpil(m["mpiw"], target_range),
                "production_peak_nmpil": m["production_peak_nmpil"],
            }
            abs_error = np.abs(
                sub["y_pred"].to_numpy(float) - sub["y_true"].to_numpy(float)
            )
            width = (
                sub["upper_pi"].to_numpy(float)
                - sub["lower_pi"].to_numpy(float)
            )
            row.update(_boxplot_stats(abs_error, "abs_error"))
            row.update(_boxplot_stats(width / target_range, "row_nmpil"))
            rows.append(row)
    return pd.DataFrame(rows)


def build_frequency_weighted_bin_summary(
    bin_category: pd.DataFrame, total_daytime_count: int, coverage_target: float
) -> pd.DataFrame:
    """Summarise bin calibration weighted by observed category frequency."""
    rows = []
    for category in CATEGORY_ORDER:
        sub = bin_category.loc[bin_category["category"] == category].copy()
        count = int(sub["count"].sum())
        valid = sub.loc[sub["count"] > 0].copy()
        if count == 0 or valid.empty:
            rows.append({
                "category": category,
                "count": count,
                "frequency_of_daytime": float("nan"),
                "frequency_weighted_picp": float("nan"),
                "frequency_weighted_mae": float("nan"),
                "frequency_weighted_rmse": float("nan"),
                "frequency_weighted_abs_picp_gap": float("nan"),
                "frequency_weighted_undercoverage_gap": float("nan"),
                "frequency_weighted_overcoverage_gap": float("nan"),
                "dominant_bin": "",
                "dominant_bin_frequency": float("nan"),
                "max_gap_bin": "",
                "max_gap_bin_frequency": float("nan"),
                "max_abs_picp_gap": float("nan"),
            })
            continue

        weights = valid["count"].to_numpy(float) / count
        picp = valid["picp"].to_numpy(float)
        gaps = picp - coverage_target
        dominant = valid.loc[valid["count"].idxmax()]
        worst = valid.iloc[int(np.argmax(np.abs(gaps)))]
        rows.append({
            "category": category,
            "count": count,
            "frequency_of_daytime": _safe(count / total_daytime_count)
            if total_daytime_count else float("nan"),
            "frequency_weighted_picp": float(np.dot(weights, picp)),
            "frequency_weighted_mae": float(np.dot(weights, valid["mae"].to_numpy(float))),
            "frequency_weighted_rmse": float(np.sqrt(np.dot(
                weights, valid["rmse"].to_numpy(float) ** 2
            ))),
            "frequency_weighted_abs_picp_gap": float(np.dot(weights, np.abs(gaps))),
            "frequency_weighted_undercoverage_gap": float(np.dot(
                weights, np.maximum(-gaps, 0.0)
            )),
            "frequency_weighted_overcoverage_gap": float(np.dot(
                weights, np.maximum(gaps, 0.0)
            )),
            "dominant_bin": str(dominant["bin"]),
            "dominant_bin_frequency": _safe(dominant["count"] / count),
            "max_gap_bin": str(worst["bin"]),
            "max_gap_bin_frequency": _safe(worst["count"] / count),
            "max_abs_picp_gap": float(abs(worst["picp"] - coverage_target)),
        })
    return pd.DataFrame(rows)


def build_uncertainty_response(day: pd.DataFrame,
                               target_range: float) -> tuple[pd.DataFrame, dict]:
    """Each anomalous category vs ALL normal daytime samples.

    underdispersion_flag = std_ratio_vs_normal < mae_ratio_vs_normal
    (uncertainty grows slower than error -> the model is over-confident).
    """
    normal = subset_metrics(day.loc[category_mask(day, "normal")])
    normal_nmpil = _nmpil(normal["mpiw"], target_range)
    rows = []
    for cat in UNCERTAINTY_CATEGORIES:
        m = subset_metrics(day.loc[category_mask(day, cat)])
        mae_ratio = _safe(m["MAE"] / normal["MAE"]) if normal["MAE"] else float("nan")
        rmse_ratio = _safe(m["RMSE"] / normal["RMSE"]) if normal["RMSE"] else float("nan")
        std_ratio = _safe(m["mean_std"] / normal["mean_std"]) if normal["mean_std"] else float("nan")
        picp_delta = _safe(m["PICP"] - normal["PICP"])
        mpiw_ratio = _safe(m["mpiw"] / normal["mpiw"]) if normal["mpiw"] else float("nan")
        cat_nmpil = _nmpil(m["mpiw"], target_range)
        nmpil_ratio = _safe(cat_nmpil / normal_nmpil) if normal_nmpil else float("nan")
        component_ratios = {}
        normal_rows = day.loc[category_mask(day, "normal")]
        category_rows = day.loc[category_mask(day, cat)]
        for component in ("epistemic_std", "aleatoric_std"):
            if component not in day:
                continue
            normal_component = pd.to_numeric(
                normal_rows[component], errors="coerce"
            ).to_numpy(float)
            category_component = pd.to_numeric(
                category_rows[component], errors="coerce"
            ).to_numpy(float)
            normal_component = normal_component[np.isfinite(normal_component)]
            category_component = category_component[
                np.isfinite(category_component)
            ]
            normal_mean = (
                float(np.mean(normal_component))
                if normal_component.size else float("nan")
            )
            category_mean = (
                float(np.mean(category_component))
                if category_component.size else float("nan")
            )
            component_ratios[f"{component}_ratio_vs_normal"] = (
                _safe(category_mean / normal_mean)
                if np.isfinite(normal_mean) and normal_mean > 0.0
                else float("nan")
            )
        flag = bool(std_ratio < mae_ratio) if (
            np.isfinite(std_ratio) and np.isfinite(mae_ratio)
        ) else False
        rows.append({
            "category": cat,
            "mae_ratio_vs_normal": mae_ratio,
            "rmse_ratio_vs_normal": rmse_ratio,
            "std_ratio_vs_normal": std_ratio,
            "picp_delta_vs_normal": picp_delta,
            "mpiw_ratio_vs_normal": mpiw_ratio,
            "nmpil_ratio_vs_normal": nmpil_ratio,
            "underdispersion_flag": flag,
            **component_ratios,
        })
    return pd.DataFrame(rows), normal


def build_uncertainty_components(day: pd.DataFrame) -> pd.DataFrame:
    """Summarise SDE epistemic and Gaussian-head aleatoric uncertainty."""
    required = {"epistemic_std", "aleatoric_std"}
    if not required <= set(day.columns):
        return pd.DataFrame()

    rows = []
    scopes = (
        ("overall_daytime", np.ones(len(day), dtype=bool)),
        ("normal", category_mask(day, "normal")),
        ("rare_extreme", category_mask(day, "rare_extreme")),
    )
    for scope, mask in scopes:
        sub = day.loc[mask]
        epi = pd.to_numeric(sub["epistemic_std"], errors="coerce").to_numpy(float)
        ale = pd.to_numeric(sub["aleatoric_std"], errors="coerce").to_numpy(float)
        total = pd.to_numeric(sub["y_std"], errors="coerce").to_numpy(float)
        valid = np.isfinite(epi) & np.isfinite(ale) & np.isfinite(total)
        epi, ale, total = epi[valid], ale[valid], total[valid]
        if not valid.any():
            rows.append({"scope": scope, "count": 0})
            continue
        total_variance = epi ** 2 + ale ** 2
        rows.append({
            "scope": scope,
            "count": int(valid.sum()),
            "mean_epistemic_std": float(np.mean(epi)),
            "median_epistemic_std": float(np.median(epi)),
            "p90_epistemic_std": float(np.quantile(epi, 0.90)),
            "mean_aleatoric_std": float(np.mean(ale)),
            "median_aleatoric_std": float(np.median(ale)),
            "p90_aleatoric_std": float(np.quantile(ale, 0.90)),
            "mean_total_std": float(np.mean(total)),
            "mean_epistemic_variance": float(np.mean(epi ** 2)),
            "mean_aleatoric_variance": float(np.mean(ale ** 2)),
            "epistemic_fraction_of_component_variance": float(
                np.mean(np.divide(
                    epi ** 2,
                    total_variance,
                    out=np.zeros_like(total_variance),
                    where=total_variance > 0.0,
                ))
            ),
        })
    return pd.DataFrame(rows)


def build_sharpness_overview(day: pd.DataFrame, target_range: float,
                             gamma: float, eta: float) -> pd.DataFrame:
    """Per-scope interval sharpness, including daily-peak-normalized width."""
    scopes = [
        ("overall_daytime", day),
        ("normal", day.loc[category_mask(day, "normal")]),
        ("rare_extreme", day.loc[category_mask(day, "rare_extreme")]),
    ]
    for lab in SPECIFIC_ANOMALY_LABELS:
        scopes.append((lab, day.loc[category_mask(day, lab)]))
    rows = []
    for scope, sub in scopes:
        m = subset_metrics(sub)
        nmpil = _nmpil(m["mpiw"], target_range)
        rows.append({
            "scope": scope,
            "count": m["count"],
            "picp": m["PICP"],
            "mae": m["MAE"],
            "rmse": m["RMSE"],
            "mean_std": m["mean_std"],
            "mpiw": m["mpiw"],
            "nmpil": nmpil,
            "production_peak_nmpil": m["production_peak_nmpil"],
            "clc": _clc(nmpil, m["PICP"], gamma, eta),
            "target_range": target_range,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
def _fmt(v, nd: int = 4) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{float(v):.{nd}f}"


def _table(df: pd.DataFrame, ndigits: dict | None = None) -> list[str]:
    ndigits = ndigits or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [_fmt(r[c], ndigits.get(c, 4)) for c in cols]
        out.append("| " + " | ".join(cells) + " |")
    return out


def render_report(args, col, stats, day, overview, bin_summary,
                  bin_category, frequency_weighted, uncertainty, normal_metrics,
                  sharpness, uncertainty_components, target_range) -> str:
    n_day = len(day)
    L: list[str] = []
    L.append("# PVGIS-only ST-GNN — daytime production-bin x anomaly report\n")
    L.append("Post-hoc analysis only: it never retrains or modifies the saved model.\n")
    if args.train_normal_only:
        L.append("Training provenance: anomaly labels physically dropped training windows "
                 "with rare target/input-history cells before fitting; they were never "
                 "model inputs or targets.\n")
    else:
        L.append("Training provenance: anomaly labels are used here only to stratify saved predictions.\n")

    # 1. Setup
    L.append("## 1. Setup\n")
    L.append(f"- Predictions: `{args.predictions}`")
    L.append("- PV loss: **heteroscedastic NLL** on the (mean, sigma) head "
             "(Gaussian or Student-t per the training run); band = mean ± z·total_std "
             "(distribution quantile on the SDE total predictive std)")
    L.append(f"- Daytime definition: target-time `solar_irradiance_poa` > "
             f"**{args.daytime_threshold} W/m²**")
    L.append(f"- Coverage target (gamma): **{args.coverage_target:.3f}**")
    L.append(f"- CLC eta: **{args.clc_eta:.2f}**")
    if stats["production_bin_basis"] == "reference_peak_pct":
        L.append("- Production-bin basis: **percentage of one fixed global "
                 "PVGIS reference peak** "
                 f"(q={stats['reference_peak_quantile']:.3f}, "
                 f"peak {_fmt(stats['reference_peak_w'])} W, "
                 f"{stats['reference_peak_sample_count']:,} positive daytime rows)")
    else:
        L.append("- Production-bin basis: **raw `y_true` watts** (legacy compatibility mode)")
    L.append(f"- CSV chunksize: **{args.chunksize:,}** rows")
    L.append(f"- Total input rows: **{stats['total_samples']:,}**")
    L.append(f"- Valid daytime rows analysed: **{n_day:,}**")
    L.append(f"- Invalid daytime rows skipped: **{stats['daytime_invalid']:,}** "
             f"(daytime candidates with non-finite y_true/y_pred/y_std/lower_pi/upper_pi)")
    resolved = {k: v for k, v in col.items() if v is not None}
    L.append(f"- Resolved columns: {resolved}")
    L.append(
        f"- Run config (reported, not read from the CSV): model_type="
        f"stgnn (SDE-Net), feature_set=full, kt-aux ON (w=0.1), "
        f"epochs={args.epochs}, dropout={args.dropout}, mc_samples={args.mc_samples}, "
        f"seed=1.\n"
    )

    # 2. Daytime overview
    L.append("## 2. Daytime overview\n")
    o = overview.iloc[0]
    L.append(f"- total_samples_all: **{int(o['total_samples_all']):,}**")
    L.append(f"- total_daytime_samples: **{int(o['total_daytime_samples']):,}**")
    L.append(f"- normal_daytime_samples: **{int(o['normal_daytime_samples']):,}**")
    L.append(f"- rare_extreme_daytime_samples: **{int(o['rare_extreme_daytime_samples']):,}**")
    L.append(f"- rare_extreme_daytime_pct: **{_fmt(o['rare_extreme_daytime_pct'])}**")
    for lab in SPECIFIC_ANOMALY_LABELS:
        L.append(f"- {lab}_count: **{int(o[f'{lab}_count']):,}**")
        L.append(f"- {lab}_pct_of_daytime: **{_fmt(o[f'{lab}_pct_of_daytime'])}**")
    L.append("")

    # 3. Production-bin summary
    L.append("## 3. Production-bin summary\n")
    if stats["production_bin_basis"] == "reference_peak_pct":
        L.append("Bins use `100 × y_true / reference_peak_w`; final bin is "
                 "80–100%. MAE/RMSE/MPIW remain in watts.\n")
    else:
        L.append("Bins use physical `y_true` in watts, daytime rows only. "
                 "`[lower, upper)`; final bin `y_true >= 100 W`.\n")
    L += _table(bin_summary, {"mae": 4, "rmse": 4, "picp": 3, "mean_std": 4,
                              "mpiw": 4, "nmpil": 4,
                              "production_peak_nmpil": 4})
    L.append("")

    # 4. Production bin x category
    L.append("## 4. Production bin x category\n")
    L.append("`count_inside_pi` uses the inclusive rule "
             "`lower_pi <= y_true <= upper_pi`. `frequency_within_category` is "
             "the cell's share of that category's daytime rows; "
             "`frequency_of_daytime` is its share of all daytime rows. "
             "Specific labels overlap, so their latter frequencies are not additive.\n")
    L += _table(bin_category, {"picp": 3, "mae": 4, "rmse": 4,
                               "mpiw": 4, "nmpil": 4,
                               "production_peak_nmpil": 4,
                               "frequency_within_category": 4,
                               "frequency_of_daytime": 4})
    L.append("")

    # 5. Frequency-weighted bin calibration
    L.append("## 5. Frequency-weighted bin calibration\n")
    L.append("Each metric weights a production-bin result by its observed share "
             "within the category. `frequency_weighted_abs_picp_gap` is the "
             "average absolute PICP gap for a randomly selected category row; "
             "the under-/over-coverage columns keep its direction. "
             "`max_gap_bin_frequency` states how common the worst local bin is, "
             "so a sparse bin cannot dominate the category-level conclusion.\n")
    L += _table(frequency_weighted, {
        "frequency_of_daytime": 4,
        "frequency_weighted_picp": 3,
        "frequency_weighted_mae": 4,
        "frequency_weighted_rmse": 4,
        "frequency_weighted_abs_picp_gap": 4,
        "frequency_weighted_undercoverage_gap": 4,
        "frequency_weighted_overcoverage_gap": 4,
        "dominant_bin_frequency": 4,
        "max_gap_bin_frequency": 4,
        "max_abs_picp_gap": 4,
    })
    L.append("")

    # 6. Uncertainty response
    L.append("## 6. Uncertainty response (vs normal daytime)\n")
    L.append(f"- Reference = ALL normal daytime samples (count "
             f"{normal_metrics['count']:,}, MAE {_fmt(normal_metrics['MAE'])}, "
             f"mean_std {_fmt(normal_metrics['mean_std'])}, "
             f"PICP {_fmt(normal_metrics['PICP'], 3)}).")
    L.append("- `underdispersion_flag = std_ratio_vs_normal < mae_ratio_vs_normal` "
             "(error grows faster than uncertainty -> over-confident).\n")
    L += _table(uncertainty, {
        "mae_ratio_vs_normal": 3, "rmse_ratio_vs_normal": 3,
        "std_ratio_vs_normal": 3, "picp_delta_vs_normal": 3,
        "mpiw_ratio_vs_normal": 3, "nmpil_ratio_vs_normal": 3,
        "epistemic_std_ratio_vs_normal": 3,
        "aleatoric_std_ratio_vs_normal": 3,
    })
    L.append("")

    L.append("## Epistemic / aleatoric decomposition\n")
    if uncertainty_components.empty:
        L.append(
            "The prediction file does not expose both uncertainty components; "
            "only total predictive uncertainty can be reported.\n"
        )
    else:
        L.append(
            "Epistemic uncertainty is the variability across SDE Brownian-path "
            "predictive means; aleatoric uncertainty is the Gaussian PV-head "
            "standard deviation. Every value below uses all valid daytime rows.\n"
        )
        L += _table(uncertainty_components, {
            "mean_epistemic_std": 4,
            "median_epistemic_std": 4,
            "p90_epistemic_std": 4,
            "mean_aleatoric_std": 4,
            "median_aleatoric_std": 4,
            "p90_aleatoric_std": 4,
            "mean_total_std": 4,
            "mean_epistemic_variance": 4,
            "mean_aleatoric_variance": 4,
            "epistemic_fraction_of_component_variance": 4,
        })
        L.append("")

    # Sharpness summary
    s = sharpness.set_index("scope")
    overall_s = subset_metrics(day)
    L.append("## Sharpness summary\n")
    L.append("MPIW = mean interval width; NMPIL = MPIW / target_range.\n")
    if stats["production_bin_basis"] == "reference_peak_pct":
        L.append("production_peak_nmpil = mean(width / reference_peak_w).\n")
    else:
        L.append("production_peak_nmpil is unavailable in raw-watt bin mode.\n")
    L.append("Lower MPIW/NMPIL means sharper intervals; read them together with PICP.\n")
    L.append(f"- target_range: **{_fmt(target_range)}** "
             f"(y_true_max {_fmt(stats['y_true_max'])} − "
             f"y_true_min {_fmt(stats['y_true_min'])}"
             f"{'; overridden via --target-range' if args.target_range is not None else ''})")
    L.append(f"- overall daytime mpiw: **{_fmt(overall_s['mpiw'])}**, "
             f"nmpil: **{_fmt(s.loc['overall_daytime', 'nmpil'])}** "
             f"(count {int(overall_s['count']):,})")
    L.append(f"- normal daytime mpiw: **{_fmt(s.loc['normal', 'mpiw'])}**, "
             f"nmpil: **{_fmt(s.loc['normal', 'nmpil'])}** "
             f"| production_peak_nmpil: **{_fmt(s.loc['normal', 'production_peak_nmpil'])}** "
             f"(count {int(s.loc['normal', 'count']):,})")
    L.append(f"- rare_extreme daytime mpiw: **{_fmt(s.loc['rare_extreme', 'mpiw'])}**, "
             f"nmpil: **{_fmt(s.loc['rare_extreme', 'nmpil'])}** "
             f"| production_peak_nmpil: **{_fmt(s.loc['rare_extreme', 'production_peak_nmpil'])}** "
             f"(count {int(s.loc['rare_extreme', 'count']):,})")
    L.append("")
    L += _table(sharpness, {
        "mpiw": 4, "nmpil": 4, "production_peak_nmpil": 4, "clc": 4,
        "target_range": 4,
    })
    L.append("")

    L.append("## 7. Automatic interpretation\n")
    L += _interpretation(
        day, n_day, uncertainty, normal_metrics, frequency_weighted, args
    )
    L.append("")
    return "\n".join(L)


def _interpretation(day, n_day, uncertainty, normal_metrics, frequency_weighted,
                    args) -> list[str]:
    out: list[str] = []
    u = uncertainty.set_index("category")
    f = frequency_weighted.set_index("category")

    # rare/extreme degradation + uncertainty response
    rare = subset_metrics(day.loc[category_mask(day, "rare_extreme")])
    mae_ratio = (rare["MAE"] / normal_metrics["MAE"]) if normal_metrics["MAE"] else float("nan")
    std_ratio = (rare["mean_std"] / normal_metrics["mean_std"]) if normal_metrics["mean_std"] else float("nan")
    if np.isfinite(mae_ratio):
        verdict = "degrades" if mae_ratio > 1.0 else "does NOT degrade"
        out.append(f"- Rare/extreme daytime **{verdict}** vs normal daytime "
                   f"(MAE ratio {mae_ratio:.2f}×).")
    if np.isfinite(std_ratio):
        direction = "increases" if std_ratio > 1.0 else "does NOT increase"
        out.append(f"- Uncertainty on rare/extreme **{direction}** "
                   f"(std ratio {std_ratio:.2f}×).")

    # under-dispersion overall
    flags = u["underdispersion_flag"]
    n_flag = int(flags.sum())
    if n_flag:
        flagged = ", ".join(flags.index[flags].tolist())
        out.append(f"- **Under-dispersion persists** in {n_flag}/{len(flags)} "
                   f"anomalous categories: {flagged}.")
    else:
        out.append("- No under-dispersion flag set: uncertainty grows at least "
                   "as fast as error in every anomalous category.")

    # per-label specifics (all four anomaly labels)
    for cat, human in (
        ("unusually_low_solar_potential", "Unusually-low solar potential"),
        ("unusually_high_solar_potential", "Unusually-high solar potential"),
        ("extreme_temperature_condition", "Extreme temperature condition"),
        ("extreme_wind_condition", "Extreme wind condition"),
    ):
        if cat in u.index:
            r = u.loc[cat]
            out.append(
                f"- **{human}:** MAE {r['mae_ratio_vs_normal']:.2f}× normal, "
                f"std {r['std_ratio_vs_normal']:.2f}×, "
                f"PICP delta {r['picp_delta_vs_normal']:+.3f}, "
                f"under-dispersion {'YES' if r['underdispersion_flag'] else 'no'}."
            )
        if cat in f.index:
            w = f.loc[cat]
            out.append(
                f"  Frequency-weighted |PICP gap| "
                f"{w['frequency_weighted_abs_picp_gap']:.3f}; largest local gap "
                f"in `{w['max_gap_bin']}` (category frequency "
                f"{w['max_gap_bin_frequency']:.1%})."
            )

    # overall daytime PICP vs target
    day_picp = subset_metrics(day)["PICP"]
    gap = day_picp - args.coverage_target
    rel = "below" if gap < 0 else "at/above"
    out.append(f"- Overall daytime PICP **{day_picp:.3f}** is {rel} the "
               f"{args.coverage_target:.3f} target (delta {gap:+.3f}).")
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True,
                   help="Path to the run's predictions.csv (physical watt space).")
    p.add_argument("--out-dir", "--out_dir", required=True,
                   help="Output directory for the report + CSVs.")
    p.add_argument("--daytime-threshold", "--daytime_threshold", type=float,
                   default=DAYTIME_IRRADIANCE_THRESHOLD_WM2,
                   help="Daytime irradiance threshold in W/m² (default 10.0).")
    p.add_argument("--coverage-target", "--coverage_target", type=float,
                   default=0.95, help="Coverage target gamma for CLC and interpretation.")
    p.add_argument("--clc-eta", "--clc_eta", type=float, default=9.0,
                   help="CLC sharpness/reliability scaling parameter.")
    p.add_argument("--train-normal-only", "--train_normal_only", action="store_true",
                   help="Record that labels physically dropped training windows "
                        "with rare target/input-history cells.")
    p.add_argument("--chunksize", type=int, default=1_000_000,
                   help="CSV read chunk size (rows).")
    p.add_argument("--target-range", "--target_range", type=float, default=None,
                   help="Override target_range for NMPIL.")
    p.add_argument("--production-bin-basis", "--production_bin_basis",
                   choices=("reference_peak_pct", "raw_watt"),
                   default="reference_peak_pct",
                   help="Production-bin coordinate: percentage of one global "
                        "PVGIS reference peak (default), or legacy raw watts.")
    p.add_argument("--reference-peak-quantile", "--reference_peak_quantile",
                   type=float, default=0.99,
                   help="Global daytime y_true quantile used as the fixed "
                        "PVGIS reference peak for reference_peak_pct bins.")
    # Header-only run config; not read from predictions.csv.
    p.add_argument("--epochs", type=int, default=5,
                   help="Run epochs (header/provenance only; not read from CSV).")
    p.add_argument("--dropout", type=float, default=0.3,
                   help="Run dropout (header/provenance only; not read from CSV).")
    p.add_argument("--mc-samples", "--mc_samples", type=int, default=10,
                   help="Run SDE samples (header/provenance only; not read from CSV).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    col = resolve_columns(args.predictions)
    day, stats = load_daytime(
        args.predictions, col, args.daytime_threshold, args.chunksize
    )
    if args.production_bin_basis == "reference_peak_pct":
        stats.update(add_reference_peak_production_pct(day, args.reference_peak_quantile))
        production_bins = PERCENT_PRODUCTION_BINS
        production_col = "production_pct"
    else:
        stats["production_bin_basis"] = "raw_watt"
        production_bins = RAW_PRODUCTION_BINS
        production_col = "y_true"

    if args.target_range is not None:
        target_range = float(args.target_range)
    else:
        target_range = float(stats["y_true_max"]) - float(stats["y_true_min"])
    print(f"[sharpness] target_range = {target_range:.4f} "
          f"(y_true_min {stats['y_true_min']:.4f}, y_true_max {stats['y_true_max']:.4f}"
          f"{', overridden via --target-range' if args.target_range is not None else ''})")

    overview = build_overview(day, stats)
    bin_summary = build_bin_summary(
        day, target_range, production_bins, production_col
    )
    bin_category = build_bin_category(
        day, target_range, production_bins, production_col
    )
    frequency_weighted = build_frequency_weighted_bin_summary(
        bin_category, len(day), args.coverage_target
    )
    uncertainty, normal_metrics = build_uncertainty_response(day, target_range)
    uncertainty_components = build_uncertainty_components(day)
    sharpness = build_sharpness_overview(
        day, target_range, args.coverage_target, args.clc_eta
    )

    overview.to_csv(out_dir / "daytime_anomaly_overview.csv", index=False)
    bin_summary.to_csv(out_dir / "daytime_bin_summary.csv", index=False)
    bin_category.to_csv(out_dir / "daytime_bin_anomaly_metrics.csv", index=False)
    frequency_weighted.to_csv(out_dir / "frequency_weighted_bin_summary.csv", index=False)
    if args.production_bin_basis == "reference_peak_pct":
        ref_peaks = pd.DataFrame([{
            "reference_peak_w": stats["reference_peak_w"],
            "reference_peak_quantile": stats["reference_peak_quantile"],
            "reference_peak_scope": stats["reference_peak_scope"],
            "reference_peak_sample_count": stats["reference_peak_sample_count"],
        }])
        ref_peaks.to_csv(out_dir / REFERENCE_PRODUCTION_PEAKS_FILE, index=False)
    uncertainty.to_csv(out_dir / "uncertainty_response.csv", index=False)
    if not uncertainty_components.empty:
        uncertainty_components.to_csv(
            out_dir / "uncertainty_components.csv", index=False
        )
    sharpness.to_csv(out_dir / "sharpness_overview.csv", index=False)

    report = render_report(args, col, stats, day, overview, bin_summary,
                           bin_category, frequency_weighted, uncertainty, normal_metrics,
                           sharpness, uncertainty_components, target_range)
    report_path = out_dir / "daytime_bin_anomaly_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"[done] wrote:\n  {report_path}")
    for name in ("daytime_anomaly_overview.csv", "daytime_bin_summary.csv",
                 "daytime_bin_anomaly_metrics.csv", "frequency_weighted_bin_summary.csv",
                 REFERENCE_PRODUCTION_PEAKS_FILE,
                 "uncertainty_response.csv",
                 "uncertainty_components.csv",
                 "sharpness_overview.csv"):
        path = out_dir / name
        if path.exists():
            print(f"  {path}")


if __name__ == "__main__":
    main()
