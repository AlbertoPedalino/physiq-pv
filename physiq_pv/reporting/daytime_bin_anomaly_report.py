#!/usr/bin/env python
"""
Daytime production-bin x anomaly report for ONE PVGIS-only ST-GNN MC-Dropout run
(built for the Huber-loss vs MSE comparison, but works on ANY run's
predictions.csv).

EVAL-ONLY. Reads an already-written predictions.csv (physical watt space) and
nothing else: it never touches training, the model, the loss, the MC Dropout
pass or the sweep. Anomaly labels are used ONLY to stratify the saved rows.
loss_type / huber_delta / clc_eta / coverage_target are NOT in the CSV; they are
passed on the CLI purely so the report header is self-describing.

Sections (daytime = solar_irradiance_poa_target > threshold):
  1. Setup
  2. Daytime overview
  3. Production-bin summary           -> daytime_bin_summary.csv
  4. Production bin x category        -> daytime_bin_anomaly_metrics.csv
  5. Uncertainty response             -> uncertainty_response.csv
  6. Automatic interpretation
Plus a daytime_anomaly_overview.csv for section 2 and the markdown report
daytime_bin_anomaly_report.md.

The CSV can be ~10M rows, so it is read in chunks; only the (small) daytime
subset with the columns we need is kept in memory, then aggregated exactly.

Usage (no PYTHONPATH needed; the script is standalone):
  python scripts/analyze_pvgis_huber_daytime_report.py \
      --predictions outputs/pvgis_stgnn_huber_d01_mc_dropout_seed1/predictions.csv \
      --out-dir outputs/pvgis_stgnn_huber_d01_mc_dropout_seed1 \
      --loss-type huber --huber-delta 0.1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Mirrors physiq_pv.data.pvgis_dataset (kept local so this analysis script
# stays standalone and does not import torch/the training package).
DAYTIME_IRRADIANCE_THRESHOLD_WM2 = 10.0
GROUP_NORMAL = "normal"
GROUP_RARE = "rare_or_extreme"
SPECIFIC_ANOMALY_LABELS = [
    "unusually_low_solar_potential",
    "unusually_high_solar_potential",
    "extreme_temperature_condition",
    "extreme_wind_condition",
]

# Six daytime production bins on physical y_true (watt). Half-open [lower, upper);
# the final bin is y_true >= 100 W (upper = None). Same convention as the runner.
PRODUCTION_BINS = (
    ("daytime_0_20", 0.0, 20.0),
    ("daytime_20_40", 20.0, 40.0),
    ("daytime_40_60", 40.0, 60.0),
    ("daytime_60_80", 60.0, 80.0),
    ("daytime_80_100", 80.0, 100.0),
    ("daytime_gt_100", 100.0, None),
)

# The categories compared in sections 4-5. normal/rare partition daytime; the
# four labels are subsets of the rare_or_extreme group.
CATEGORY_ORDER = [
    "normal",
    "rare_extreme",
    "unusually_low_solar_potential",
    "unusually_high_solar_potential",
    "extreme_temperature_condition",
    "extreme_wind_condition",
]
# Categories compared against normal in section 5 (everything but normal itself).
UNCERTAINTY_CATEGORIES = CATEGORY_ORDER[1:]


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


def resolve_columns(path: str) -> dict[str, str]:
    header = pd.read_csv(path, nrows=0)
    cols = set(header.columns)
    resolved = {
        "y_true": _pick(cols, ["y_true"], "y_true"),
        "y_pred": _pick(cols, ["y_pred_mean", "y_pred"], "mean prediction"),
        "y_std": _pick(cols, ["y_pred_std_raw", "y_pred_std"], "MC std"),
        "lower_pi": _pick(cols, ["lower_pi"], "lower interval"),
        "upper_pi": _pick(cols, ["upper_pi"], "upper interval"),
        "solar": _pick(
            cols,
            ["solar_irradiance_poa_target", "solar_irradiance_poa", "ghi_target"],
            "target-time irradiance (daytime filter)",
        ),
        "group": _pick(cols, ["anomaly_group"], "anomaly_group"),
        "label": _pick(cols, ["anomaly_label"], "anomaly_label"),
    }
    resolved["timestamp"] = "timestamp" if "timestamp" in cols else None
    return resolved


# --------------------------------------------------------------------------- #
# Chunked load: keep only the daytime rows + the columns we need
# --------------------------------------------------------------------------- #
_METRIC_COLS = ["y_true", "y_pred", "y_std", "lower_pi", "upper_pi"]


def load_daytime(path: str, col: dict[str, str], threshold: float,
                 chunksize: int) -> tuple[pd.DataFrame, dict]:
    """Stream the big CSV, keep VALID daytime rows with a compact schema.

    A row is a daytime candidate when target-time solar > threshold. A candidate
    is VALID only when all five metric columns (y_true, y_pred, y_std, lower_pi,
    upper_pi) are finite; candidates with any non-finite metric are SKIPPED and
    counted in stats['daytime_invalid']. Output frame: y_true, y_pred, y_std,
    lower_pi, upper_pi, is_rare, is_normal, one bool per anomaly label
    (has_<label>). Returns (daytime_frame, stats).
    """
    usecols = [c for c in dict.fromkeys(col.values()) if c is not None]
    keep = []
    n_total = 0
    n_candidate = 0   # solar > threshold (daytime candidates)
    n_invalid = 0     # candidates dropped for non-finite metrics
    y_true_min = float("inf")   # over VALID daytime rows only (for target_range)
    y_true_max = float("-inf")
    reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize)
    for chunk in reader:
        n_total += len(chunk)
        solar = pd.to_numeric(chunk[col["solar"]], errors="coerce").to_numpy(float)
        day_mask = solar > threshold     # NaN solar -> False -> treated as non-daytime
        if not day_mask.any():
            continue
        sub = chunk.loc[day_mask]
        n_candidate += len(sub)
        group = sub[col["group"]].to_numpy().astype(str)
        out = pd.DataFrame({
            c: pd.to_numeric(sub[col[c]], errors="coerce").to_numpy(np.float64)
            for c in _METRIC_COLS
        })
        out["is_rare"] = group == GROUP_RARE
        out["is_normal"] = group == GROUP_NORMAL
        # Exact, vectorized membership in the comma-joined label string. Pad with
        # commas so ",label," matches whole tokens only (no substring leakage).
        padded = "," + sub[col["label"]].fillna("").astype(str) + ","
        for lab in SPECIFIC_ANOMALY_LABELS:
            out[f"has_{lab}"] = padded.str.contains(
                "," + lab + ",", regex=False
            ).to_numpy()
        finite = np.isfinite(out[_METRIC_COLS].to_numpy(float)).all(axis=1)
        n_invalid += int((~finite).sum())
        if finite.any():
            kept = out.loc[finite].reset_index(drop=True)
            keep.append(kept)
            yt = kept["y_true"].to_numpy(float)  # finite here -> safe min/max
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


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #
def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def subset_metrics(sub: pd.DataFrame) -> dict:
    """count, count_inside_pi, PICP, MAE, RMSE, mean_std, mpiw for one subset.

    mpiw = mean(upper_pi - lower_pi) over rows whose interval width is finite AND
    non-negative; degenerate widths (non-finite or < 0) are dropped and counted in
    count_width. (lower/upper are already finite from load_daytime, but the guard
    keeps this primitive correct on any subset.)
    """
    n = len(sub)
    if n == 0:
        return {"count": 0, "count_inside_pi": 0, "PICP": float("nan"),
                "MAE": float("nan"), "RMSE": float("nan"), "mean_std": float("nan"),
                "mpiw": float("nan"), "count_width": 0}
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
    return {
        "count": int(n),
        "count_inside_pi": int(inside.sum()),
        "PICP": float(np.mean(inside)),
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual ** 2))),
        "mean_std": float(np.mean(y_std)),
        "mpiw": mpiw,
        "count_width": int(width_ok.sum()),
    }


def _nmpil(mpiw: float, target_range: float) -> float:
    """NMPIL = MPIW / target_range; NaN when range is missing/non-positive."""
    if (target_range is None or not np.isfinite(target_range)
            or target_range <= 0.0 or not np.isfinite(mpiw)):
        return float("nan")
    return float(mpiw / target_range)


def _bin_mask(day: pd.DataFrame, lower: float, upper) -> np.ndarray:
    y = day["y_true"].to_numpy(float)
    mask = y >= lower
    if upper is not None:
        mask = mask & (y < upper)
    return mask


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


def build_bin_summary(day: pd.DataFrame, target_range: float) -> pd.DataFrame:
    rows = []
    for bin_name, lo, hi in PRODUCTION_BINS:
        m = subset_metrics(day.loc[_bin_mask(day, lo, hi)])
        rows.append({"bin": bin_name, "count": m["count"], "mae": m["MAE"],
                     "rmse": m["RMSE"], "picp": m["PICP"], "mean_std": m["mean_std"],
                     "mpiw": m["mpiw"], "nmpil": _nmpil(m["mpiw"], target_range)})
    return pd.DataFrame(rows)


def build_bin_category(day: pd.DataFrame, target_range: float) -> pd.DataFrame:
    rows = []
    for bin_name, lo, hi in PRODUCTION_BINS:
        bmask = _bin_mask(day, lo, hi)
        for cat in CATEGORY_ORDER:
            sub = day.loc[bmask & category_mask(day, cat)]
            m = subset_metrics(sub)
            rows.append({
                "bin": bin_name, "category": cat, "count": m["count"],
                "count_inside_pi": m["count_inside_pi"], "picp": m["PICP"],
                "mae": m["MAE"], "rmse": m["RMSE"],
                "mpiw": m["mpiw"], "nmpil": _nmpil(m["mpiw"], target_range),
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
        })
    return pd.DataFrame(rows), normal


def build_sharpness_overview(day: pd.DataFrame, target_range: float) -> pd.DataFrame:
    """Small per-scope sharpness table: scope, count, mpiw, nmpil, target_range."""
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
        rows.append({
            "scope": scope,
            "count": m["count"],
            "mpiw": m["mpiw"],
            "nmpil": _nmpil(m["mpiw"], target_range),
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
                  bin_category, uncertainty, normal_metrics,
                  sharpness, target_range) -> str:
    n_day = len(day)
    L: list[str] = []
    L.append("# PVGIS-only ST-GNN — daytime production-bin x anomaly report\n")
    L.append("EVAL-ONLY post-hoc analysis of one MC-Dropout run. Anomaly labels "
             "are used only to stratify the saved predictions; they are never "
             "model inputs or targets.\n")

    # 1. Setup
    L.append("## 1. Setup\n")
    L.append(f"- Predictions: `{args.predictions}`")
    L.append(f"- Loss type: **{args.loss_type}**")
    if args.loss_type == "huber":
        L.append(f"- Huber delta: **{args.huber_delta}**")
    L.append(f"- Daytime definition: target-time `solar_irradiance_poa` > "
             f"**{args.daytime_threshold} W/m²**")
    L.append(f"- Coverage target (gamma): **{args.coverage_target:.3f}**")
    L.append(f"- CLC eta: **{args.clc_eta:.2f}**")
    L.append(f"- CSV chunksize: **{args.chunksize:,}** rows")
    L.append(f"- Total input rows: **{stats['total_samples']:,}**")
    L.append(f"- Valid daytime rows analysed: **{n_day:,}**")
    L.append(f"- Invalid daytime rows skipped: **{stats['daytime_invalid']:,}** "
             f"(daytime candidates with non-finite y_true/y_pred/y_std/lower_pi/upper_pi)")
    resolved = {k: v for k, v in col.items() if v is not None}
    L.append(f"- Resolved columns: {resolved}")
    L.append(
        f"- Run config (reported, not read from the CSV): model_type="
        f"stgnn_enhanced_dropout, feature_set=full, kt-aux ON (w=0.1), "
        f"epochs={args.epochs}, dropout={args.dropout}, mc_samples={args.mc_samples}, "
        f"seed=1."
    )
    L.append(
        f"- **Baseline provenance / methodological note:** this Huber run mirrors "
        f"the baseline `{args.baseline_name}` (kt-aux w=0.1) for the GENERAL setup "
        f"(dataset, features, model, target, anomaly eval). `epochs`, `dropout` and "
        f"`mc_samples` shown above were RECONSTRUCTED from sibling configurations "
        f"and were NOT directly confirmed against `{args.baseline_name}` "
        f"(its output folder / metrics.json / wandb config live on the server "
        f"and were not accessible at report-generation time). Verify them on the "
        f"server before drawing strong MSE-vs-Huber conclusions.\n"
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
    L.append("Bins use physical `y_true` in watts, daytime rows only. "
             "`[lower, upper)`; final bin `y_true >= 100 W`.\n")
    L += _table(bin_summary, {"mae": 4, "rmse": 4, "picp": 3, "mean_std": 4,
                              "mpiw": 4, "nmpil": 4})
    L.append("")

    # 4. Production bin x category
    L.append("## 4. Production bin x category\n")
    L.append("`count_inside_pi` uses the inclusive rule "
             "`lower_pi <= y_true <= upper_pi`.\n")
    L += _table(bin_category, {"picp": 3, "mae": 4, "rmse": 4,
                               "mpiw": 4, "nmpil": 4})
    L.append("")

    # 5. Uncertainty response
    L.append("## 5. Uncertainty response (vs normal daytime)\n")
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
    })
    L.append("")

    # Sharpness summary
    s = sharpness.set_index("scope")
    overall_s = subset_metrics(day)
    L.append("## Sharpness summary\n")
    L.append("MPIW measures the average prediction interval width:")
    L.append("MPIW = mean(upper_pi - lower_pi)\n")
    L.append("NMPIL normalizes MPIW by the target range:")
    L.append("NMPIL = MPIW / target_range\n")
    L.append("Lower MPIW/NMPIL means sharper intervals. PICP should therefore be "
             "interpreted together with MPIW/NMPIL: increasing coverage is useful "
             "only if the interval width does not become excessive.\n")
    L.append(f"- target_range: **{_fmt(target_range)}** "
             f"(y_true_max {_fmt(stats['y_true_max'])} − "
             f"y_true_min {_fmt(stats['y_true_min'])}"
             f"{'; overridden via --target-range' if args.target_range is not None else ''})")
    L.append(f"- overall daytime mpiw: **{_fmt(overall_s['mpiw'])}**, "
             f"nmpil: **{_fmt(s.loc['overall_daytime', 'nmpil'])}** "
             f"(count {int(overall_s['count']):,})")
    L.append(f"- normal daytime mpiw: **{_fmt(s.loc['normal', 'mpiw'])}**, "
             f"nmpil: **{_fmt(s.loc['normal', 'nmpil'])}** "
             f"(count {int(s.loc['normal', 'count']):,})")
    L.append(f"- rare_extreme daytime mpiw: **{_fmt(s.loc['rare_extreme', 'mpiw'])}**, "
             f"nmpil: **{_fmt(s.loc['rare_extreme', 'nmpil'])}** "
             f"(count {int(s.loc['rare_extreme', 'count']):,})")
    L.append("")
    L += _table(sharpness, {"mpiw": 4, "nmpil": 4, "target_range": 4})
    L.append("")

    # 6. Automatic interpretation
    L.append("## 6. Automatic interpretation\n")
    L += _interpretation(day, n_day, uncertainty, normal_metrics, args)
    L.append("")
    return "\n".join(L)


def _interpretation(day, n_day, uncertainty, normal_metrics, args) -> list[str]:
    out: list[str] = []
    u = uncertainty.set_index("category")

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
    p.add_argument("--loss-type", "--loss_type", default="huber",
                   choices=("mse", "huber"),
                   help="Loss used by the run (header only; not in the CSV).")
    p.add_argument("--huber-delta", "--huber_delta", type=float, default=0.1,
                   help="Huber delta of the run (header only; not in the CSV).")
    p.add_argument("--daytime-threshold", "--daytime_threshold", type=float,
                   default=DAYTIME_IRRADIANCE_THRESHOLD_WM2,
                   help="Daytime irradiance threshold in W/m² (default 10.0).")
    p.add_argument("--coverage-target", "--coverage_target", type=float,
                   default=0.95, help="Coverage target gamma (header/interpretation).")
    p.add_argument("--clc-eta", "--clc_eta", type=float, default=10.0,
                   help="CLC eta (header only).")
    p.add_argument("--chunksize", type=int, default=1_000_000,
                   help="CSV read chunk size (rows).")
    p.add_argument("--target-range", "--target_range", type=float, default=None,
                   help="Override target_range for NMPIL (= max-min y_true daytime "
                        "valid by default). NMPIL = MPIW / target_range.")
    # Header-only run config (not in the CSV). Defaults mirror the reconstructed
    # kt-aux w=0.1 baseline; override if the server config differs.
    p.add_argument("--epochs", type=int, default=5,
                   help="Run epochs (header/provenance only; not read from CSV).")
    p.add_argument("--dropout", type=float, default=0.3,
                   help="Run dropout (header/provenance only; not read from CSV).")
    p.add_argument("--mc-samples", "--mc_samples", type=int, default=20,
                   help="Run MC samples (header/provenance only; not read from CSV).")
    p.add_argument("--baseline-name", "--baseline_name",
                   default="pvgis_ktaux_w01_mc_wandb",
                   help="Baseline run name mirrored, shown in the provenance note.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    col = resolve_columns(args.predictions)
    day, stats = load_daytime(
        args.predictions, col, args.daytime_threshold, args.chunksize
    )

    # target_range for NMPIL: CLI override wins, else max-min y_true daytime valid.
    if args.target_range is not None:
        target_range = float(args.target_range)
    else:
        target_range = float(stats["y_true_max"]) - float(stats["y_true_min"])
    print(f"[sharpness] target_range = {target_range:.4f} "
          f"(y_true_min {stats['y_true_min']:.4f}, y_true_max {stats['y_true_max']:.4f}"
          f"{', overridden via --target-range' if args.target_range is not None else ''})")

    overview = build_overview(day, stats)
    bin_summary = build_bin_summary(day, target_range)
    bin_category = build_bin_category(day, target_range)
    uncertainty, normal_metrics = build_uncertainty_response(day, target_range)
    sharpness = build_sharpness_overview(day, target_range)

    overview.to_csv(out_dir / "daytime_anomaly_overview.csv", index=False)
    bin_summary.to_csv(out_dir / "daytime_bin_summary.csv", index=False)
    bin_category.to_csv(out_dir / "daytime_bin_anomaly_metrics.csv", index=False)
    uncertainty.to_csv(out_dir / "uncertainty_response.csv", index=False)
    sharpness.to_csv(out_dir / "sharpness_overview.csv", index=False)

    report = render_report(args, col, stats, day, overview, bin_summary,
                           bin_category, uncertainty, normal_metrics,
                           sharpness, target_range)
    report_path = out_dir / "daytime_bin_anomaly_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"[done] wrote:\n  {report_path}")
    for name in ("daytime_anomaly_overview.csv", "daytime_bin_summary.csv",
                 "daytime_bin_anomaly_metrics.csv", "uncertainty_response.csv",
                 "sharpness_overview.csv"):
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
