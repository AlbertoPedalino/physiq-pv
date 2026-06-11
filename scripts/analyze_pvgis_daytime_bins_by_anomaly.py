#!/usr/bin/env python
"""
Post-hoc daytime production-bin x anomaly-label analysis for ONE PVGIS-only
ST-GNN MC-Dropout run (e.g. outputs/wandb_pvgis_stgnn/qub1w3vt/predictions.csv).

EVAL-ONLY. Reads an already-written predictions.csv (physical watt space) and
nothing else: it never touches training, the model, the loss, the MC Dropout
pass or the sweep. Anomaly labels are used ONLY to stratify the saved rows.

It answers, in the DAYTIME regime (solar_irradiance_poa_target > threshold):

  1. How many anomalous samples fall in the 60-80 W and 80-100 W production
     bins, split by anomaly label.                  -> daytime_bin_anomaly_counts.csv
  2. Error vs MC-Dropout uncertainty per (bin x label).
                                                     -> daytime_bin_anomaly_metrics.csv
  3. Does the MC std grow as fast as the error in the anomalous strata, or does
     it grow too little? (std_growth_vs_error_growth < 1 -> std under-grows).
                                                     -> uncertainty_error_growth_by_anomaly.csv
  4. A synthetic markdown report.                    -> daytime_bin_anomaly_report.md

The CSV can be ~10M rows, so it is read in chunks; only the (small) daytime
subset with the columns we need is kept in memory, then aggregated exactly.

Usage (no PYTHONPATH needed; the script is standalone):
  python scripts/analyze_pvgis_daytime_bins_by_anomaly.py \
      --predictions outputs/wandb_pvgis_stgnn/qub1w3vt/predictions.csv \
      --out-dir outputs/wandb_pvgis_stgnn/qub1w3vt/daytime_bin_anomaly_analysis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Mirrors physiq_pv.data.pvgis_stgnn_dataset (kept local so this analysis script
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

# The two production bins of interest (watt, on physical y_true). Half-open
# [lower, upper). Same convention as the runner's daytime production bins.
PRODUCTION_BINS = (
    ("daytime_60_80", 60.0, 80.0),
    ("daytime_80_100", 80.0, 100.0),
)

# Run setup (from the sweep config; the CSV does not carry it). Shown verbatim
# in the report header so the analysis is self-describing.
RUN_SETUP = {
    "model_type": "stgnn_enhanced_dropout",
    "feature_set": "full",
    "dropout": 0.3,
    "MC Dropout": "enabled",
    "MC samples": 20,
    "target": "pv_power_output",
    "pv_target_clip_max": "none",
    "train_years": "2016,2017,2018",
    "test_year": 2019,
}

COVERAGE_TARGET = 0.95  # the PI lower_pi/upper_pi was built for ~95% coverage.


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
    # Optional: enables the unique-timestamp (distinct-hour) counts.
    resolved["timestamp"] = "timestamp" if "timestamp" in cols else None
    return resolved


# --------------------------------------------------------------------------- #
# Chunked load: keep only the daytime rows + the columns we need
# --------------------------------------------------------------------------- #
def load_daytime(path: str, col: dict[str, str], threshold: float,
                 chunksize: int) -> tuple[pd.DataFrame, dict]:
    """Stream the big CSV, keep daytime rows with a compact schema.

    Output frame columns: y_true, y_pred, y_std, lower_pi, upper_pi, is_rare,
    one boolean per anomaly label (has_<label>), plus `ts` (raw timestamp string)
    when the timestamp column is available.

    Returns (daytime_frame, stats). `stats` carries the population counts that
    cannot be recovered from the daytime-only frame: total_samples,
    nighttime_samples, has_timestamp and the set of ALL distinct timestamps
    (day + night), used for the unique-hour overview.
    """
    ts_col = col.get("timestamp")
    usecols = [c for c in dict.fromkeys(col.values()) if c is not None]
    keep = []
    n_total = 0
    n_day = 0
    all_ts: set[str] = set()
    reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize)
    for chunk in reader:
        n_total += len(chunk)
        if ts_col is not None:
            all_ts.update(pd.unique(chunk[ts_col].astype(str)))
        solar = pd.to_numeric(chunk[col["solar"]], errors="coerce").to_numpy(float)
        day_mask = solar > threshold
        if not day_mask.any():
            continue
        sub = chunk.loc[day_mask]
        n_day += len(sub)
        out = pd.DataFrame({
            "y_true": sub[col["y_true"]].to_numpy(np.float64),
            "y_pred": sub[col["y_pred"]].to_numpy(np.float64),
            "y_std": sub[col["y_std"]].to_numpy(np.float64),
            "lower_pi": sub[col["lower_pi"]].to_numpy(np.float64),
            "upper_pi": sub[col["upper_pi"]].to_numpy(np.float64),
        })
        out["is_rare"] = (sub[col["group"]].to_numpy().astype(str) == GROUP_RARE)
        if ts_col is not None:
            out["ts"] = sub[ts_col].astype(str).to_numpy()
        # Exact, vectorized membership in the comma-joined label string. Pad with
        # commas so ",label," matches whole tokens only (no substring leakage).
        padded = "," + sub[col["label"]].fillna("").astype(str) + ","
        for lab in SPECIFIC_ANOMALY_LABELS:
            out[f"has_{lab}"] = padded.str.contains("," + lab + ",", regex=False).to_numpy()
        keep.append(out)
    if not keep:
        raise SystemExit(
            f"No daytime rows (solar > {threshold}) found in {path}. "
            f"Scanned {n_total} rows."
        )
    day = pd.concat(keep, ignore_index=True)
    stats = {
        "total_samples": n_total,
        "daytime_samples": n_day,
        "nighttime_samples": n_total - n_day,
        "has_timestamp": ts_col is not None,
        "unique_timestamps_total": len(all_ts) if ts_col is not None else None,
    }
    print(f"[load] scanned {n_total:,} rows; kept {n_day:,} daytime "
          f"(solar > {threshold} W/m^2); {stats['nighttime_samples']:,} nighttime"
          + (f"; {stats['unique_timestamps_total']:,} distinct timestamps"
             if ts_col is not None else "; no timestamp column"))
    return day, stats


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #
def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def error_uncertainty_metrics(sub: pd.DataFrame) -> dict:
    """Point error, MC-uncertainty and PI metrics for one subset (watt space)."""
    n = len(sub)
    if n == 0:
        keys = ["count", "MAE", "RMSE", "mean_residual", "median_residual",
                "overprediction_pct", "underprediction_pct", "mean_std_mc",
                "median_std_mc", "p90_std_mc", "std_over_mae", "PICP_PI",
                "MPIW_PI", "above_interval_pct", "below_interval_pct"]
        d = {k: float("nan") for k in keys}
        d["count"] = 0
        return d

    y_true = sub["y_true"].to_numpy(float)
    y_pred = sub["y_pred"].to_numpy(float)
    y_std = sub["y_std"].to_numpy(float)
    lower = sub["lower_pi"].to_numpy(float)
    upper = sub["upper_pi"].to_numpy(float)

    residual = y_pred - y_true                       # >0 over-, <0 under-prediction
    mae = float(np.mean(np.abs(residual)))
    inside = (y_true >= lower) & (y_true <= upper)
    mean_std = float(np.mean(y_std))

    return {
        "count": int(n),
        "MAE": mae,
        "RMSE": float(np.sqrt(np.mean(residual ** 2))),
        "mean_residual": float(np.mean(residual)),
        "median_residual": float(np.median(residual)),
        "overprediction_pct": float(np.mean(residual > 0.0)),
        "underprediction_pct": float(np.mean(residual < 0.0)),
        "mean_std_mc": mean_std,
        "median_std_mc": float(np.median(y_std)),
        "p90_std_mc": float(np.percentile(y_std, 90)),
        "std_over_mae": _safe(mean_std / mae) if mae > 0 else float("nan"),
        "PICP_PI": float(np.mean(inside)),
        "MPIW_PI": float(np.mean(upper - lower)),
        "above_interval_pct": float(np.mean(y_true > upper)),
        "below_interval_pct": float(np.mean(y_true < lower)),
    }


def _bin_mask(day: pd.DataFrame, lower: float, upper: float) -> np.ndarray:
    y = day["y_true"].to_numpy(float)
    return (y >= lower) & (y < upper)


def _label_mask(day: pd.DataFrame, label: str) -> np.ndarray:
    return day[f"has_{label}"].to_numpy(bool)


# --------------------------------------------------------------------------- #
# Table builders
# --------------------------------------------------------------------------- #
def table_counts(day: pd.DataFrame) -> pd.DataFrame:
    """Table 1: count of each anomaly label inside each production bin."""
    label_daytime_total = {
        lab: int(_label_mask(day, lab).sum()) for lab in SPECIFIC_ANOMALY_LABELS
    }
    rows = []
    for bin_name, lo, hi in PRODUCTION_BINS:
        bmask = _bin_mask(day, lo, hi)
        bin_total = int(bmask.sum())
        # context row: all daytime rows in the bin (any/none label).
        rows.append({
            "production_bin": bin_name, "anomaly_label": "all_daytime_in_bin",
            "count": bin_total, "bin_daytime_count": bin_total,
            "label_daytime_count": bin_total,
            "fraction_of_bin": 1.0 if bin_total else float("nan"),
            "fraction_of_label_daytime": float("nan"),
        })
        for lab in SPECIFIC_ANOMALY_LABELS:
            cnt = int((bmask & _label_mask(day, lab)).sum())
            lab_tot = label_daytime_total[lab]
            rows.append({
                "production_bin": bin_name,
                "anomaly_label": lab,
                "count": cnt,
                "bin_daytime_count": bin_total,
                "label_daytime_count": lab_tot,
                "fraction_of_bin": _safe(cnt / bin_total) if bin_total else float("nan"),
                "fraction_of_label_daytime": _safe(cnt / lab_tot) if lab_tot else float("nan"),
            })
    return pd.DataFrame(rows)


def table_metrics(day: pd.DataFrame) -> pd.DataFrame:
    """Table 2: error/uncertainty metrics per (production_bin x anomaly_label)."""
    rows = []
    for bin_name, lo, hi in PRODUCTION_BINS:
        bmask = _bin_mask(day, lo, hi)
        # context row: whole bin.
        rows.append({"production_bin": bin_name, "anomaly_label": "all_daytime_in_bin",
                     **error_uncertainty_metrics(day.loc[bmask])})
        for lab in SPECIFIC_ANOMALY_LABELS:
            mask = bmask & _label_mask(day, lab)
            rows.append({"production_bin": bin_name, "anomaly_label": lab,
                         **error_uncertainty_metrics(day.loc[mask])})
    return pd.DataFrame(rows)


def table_growth(day: pd.DataFrame) -> pd.DataFrame:
    """Table 3: error vs uncertainty growth relative to normal_daytime."""
    is_rare = day["is_rare"].to_numpy(bool)
    strata = {
        "normal_daytime": ~is_rare,
        "rare_extreme_daytime": is_rare,
    }
    for lab in SPECIFIC_ANOMALY_LABELS:
        strata[lab] = _label_mask(day, lab)

    base = error_uncertainty_metrics(day.loc[strata["normal_daytime"]])
    base_mae = base["MAE"]
    base_std = base["mean_std_mc"]

    rows = []
    for name, mask in strata.items():
        m = error_uncertainty_metrics(day.loc[mask])
        mae_ratio = _safe(m["MAE"] / base_mae) if base_mae > 0 else float("nan")
        std_ratio = _safe(m["mean_std_mc"] / base_std) if base_std > 0 else float("nan")
        growth = (_safe(std_ratio / mae_ratio)
                  if np.isfinite(mae_ratio) and mae_ratio > 0 else float("nan"))
        rows.append({
            "stratum": name,
            "count": m["count"],
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "mean_std_mc": m["mean_std_mc"],
            "median_std_mc": m["median_std_mc"],
            "p90_std_mc": m["p90_std_mc"],
            "MAE_ratio_vs_normal_daytime": mae_ratio,
            "std_ratio_vs_normal_daytime": std_ratio,
            "std_growth_vs_error_growth": growth,
        })
    return pd.DataFrame(rows)


def table_overview(day: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Overview (metric,value): daytime/night split, rare/extreme share, and the
    per-label counts + percentages. Labels overlap, so the per-label counts do
    NOT sum to rare_extreme_daytime_count.
    """
    n_day = stats["daytime_samples"]
    is_rare = day["is_rare"].to_numpy(bool)
    n_rare = int(is_rare.sum())
    n_normal = int((~is_rare).sum())

    rows: list[tuple[str, object]] = [
        ("total_samples", stats["total_samples"]),
        ("daytime_samples", n_day),
        ("nighttime_samples", stats["nighttime_samples"]),
        ("normal_daytime_count", n_normal),
        ("rare_extreme_daytime_count", n_rare),
        ("rare_extreme_daytime_pct_of_daytime",
         _safe(100.0 * n_rare / n_day) if n_day else float("nan")),
    ]

    for lab in SPECIFIC_ANOMALY_LABELS:
        cnt = int(_label_mask(day, lab).sum())
        rows.append((f"{lab}_count", cnt))
        rows.append((f"{lab}_pct_of_daytime",
                     _safe(100.0 * cnt / n_day) if n_day else float("nan")))
        rows.append((f"{lab}_pct_of_rare_extreme_daytime",
                     _safe(100.0 * cnt / n_rare) if n_rare else float("nan")))

    # Unique-hour (distinct timestamp) counts — only when timestamp is present.
    if stats["has_timestamp"]:
        ts = day["ts"]
        rows.append(("unique_timestamps_total", stats["unique_timestamps_total"]))
        rows.append(("unique_timestamps_daytime", int(ts.nunique())))
        rows.append(("unique_timestamps_with_any_anomaly_daytime",
                     int(ts[is_rare].nunique())))
        for lab in SPECIFIC_ANOMALY_LABELS:
            rows.append((f"unique_timestamps_with_{lab}",
                         int(ts[_label_mask(day, lab)].nunique())))

    # object dtype keeps counts as ints (no "50000.0") next to the pct floats.
    return pd.DataFrame({
        "metric": [m for m, _ in rows],
        "value": pd.array([v for _, v in rows], dtype=object),
    })


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def _f(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and np.isfinite(x) else "—"


def _pct(x, nd=1):
    return f"{100 * x:.{nd}f}%" if isinstance(x, (int, float)) and np.isfinite(x) else "—"


def _md_table(df: pd.DataFrame, pct_cols=()) -> list[str]:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in pct_cols:
                cells.append(_pct(v))
            elif isinstance(v, float):
                cells.append(_f(v, 4) if np.isfinite(v) else "—")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return out


def _fmt_overview_value(metric: str, value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if metric.endswith("_pct_of_daytime") or metric.endswith("_pct_of_rare_extreme_daytime"):
        return f"{float(value):.2f}%"
    return f"{int(value):,}"


def render_report(counts: pd.DataFrame, metrics: pd.DataFrame,
                  growth: pd.DataFrame, overview: pd.DataFrame,
                  threshold: float, n_daytime: int) -> str:
    L: list[str] = []
    A = L.append
    A("# PVGIS ST-GNN MC-Dropout — daytime production-bin x anomaly analysis\n")
    A("Post-hoc, eval-only: computed from `predictions.csv` alone. No training, "
      "model, loss or MC-Dropout change. Anomaly labels stratify only.\n")

    A("## 1. Run setup\n")
    for k, v in RUN_SETUP.items():
        A(f"- {k}: `{v}`")
    A(f"- daytime filter: `solar_irradiance_poa_target > {threshold}` W/m^2 "
      f"({n_daytime:,} daytime samples)")
    A(f"- PI coverage target: `{COVERAGE_TARGET}` (PICP_PI should approach this)")
    A("- residual = y_pred_mean - y_true  (>0 overprediction, <0 underprediction)\n")

    A("## Daytime anomaly overview\n")
    A("sample = location x timestamp. **Anomaly labels are not mutually "
      "exclusive; a sample can be associated with multiple anomaly labels** — "
      "so the per-label counts do NOT sum to `rare_extreme_daytime_count`.\n")
    A("| metric | value |")
    A("|---|---|")
    for _, r in overview.iterrows():
        A(f"| {r['metric']} | {_fmt_overview_value(r['metric'], r['value'])} |")
    A("")

    A("## 2. Counts in the 60-80 W and 80-100 W bins\n")
    A("`fraction_of_bin` = label share within the bin; `fraction_of_label_daytime` "
      "= share of that label's daytime samples landing in the bin.\n")
    L.extend(_md_table(
        counts, pct_cols=("fraction_of_bin", "fraction_of_label_daytime")))
    A("")

    A("## 3. Error vs MC uncertainty per (bin x label)\n")
    show = ["production_bin", "anomaly_label", "count", "MAE", "RMSE",
            "mean_residual", "overprediction_pct", "underprediction_pct",
            "mean_std_mc", "p90_std_mc", "std_over_mae", "PICP_PI", "MPIW_PI",
            "above_interval_pct", "below_interval_pct"]
    L.extend(_md_table(
        metrics[show],
        pct_cols=("overprediction_pct", "underprediction_pct",
                  "PICP_PI", "above_interval_pct", "below_interval_pct")))
    A("")

    A("## 4. Error vs uncertainty growth (baseline = normal_daytime)\n")
    A("`std_growth_vs_error_growth = std_ratio / MAE_ratio`. **< 1 -> MC std "
      "grows LESS than the error** (under-dispersed under that anomaly).\n")
    L.extend(_md_table(growth))
    A("")

    A("## 5. Automatic interpretation\n")
    L.extend(_auto_interpretation(metrics, growth))
    A("")
    return "\n".join(L) + "\n"


def _auto_interpretation(metrics: pd.DataFrame, growth: pd.DataFrame) -> list[str]:
    out: list[str] = []
    g = growth.set_index("stratum")

    def grow(label):
        return g.loc[label, "std_growth_vs_error_growth"] if label in g.index else float("nan")

    def mae_ratio(label):
        return g.loc[label, "MAE_ratio_vs_normal_daytime"] if label in g.index else float("nan")

    # unusually_low_solar_potential: high error but std under-grows?
    lab = "unusually_low_solar_potential"
    if lab in g.index:
        gr, mr = grow(lab), mae_ratio(lab)
        verdict = ("std grows LESS than error (under-dispersed)" if np.isfinite(gr) and gr < 1
                   else "std keeps up with error" if np.isfinite(gr) else "n/a")
        out.append(f"- **{lab}**: MAE x{_f(mr,2)} vs normal_daytime, "
                   f"std_growth_vs_error_growth = {_f(gr,2)} -> {verdict}.")

    # unusually_high_solar_potential: underprediction-heavy?
    lab = "unusually_high_solar_potential"
    sub = metrics[(metrics["anomaly_label"] == lab)]
    if not sub.empty:
        under = float(np.nanmean(sub["underprediction_pct"]))
        gr = grow(lab)
        flag = "HIGH underprediction" if np.isfinite(under) and under > 0.6 else "balanced"
        out.append(f"- **{lab}**: mean underprediction {_pct(under)} across the "
                   f"two bins ({flag}); std_growth_vs_error_growth = {_f(gr,2)}.")

    # temperature / wind: smaller impact?
    for lab in ("extreme_temperature_condition", "extreme_wind_condition"):
        if lab in g.index:
            mr, gr = mae_ratio(lab), grow(lab)
            impact = ("LARGE" if np.isfinite(mr) and mr >= 1.5
                      else "moderate" if np.isfinite(mr) and mr >= 1.1
                      else "small/none")
            out.append(f"- **{lab}**: error impact {impact} (MAE x{_f(mr,2)}), "
                       f"std_growth_vs_error_growth = {_f(gr,2)}.")

    # PICP in the two bins (whole-bin rows).
    for bin_name in ("daytime_60_80", "daytime_80_100"):
        row = metrics[(metrics["production_bin"] == bin_name) &
                      (metrics["anomaly_label"] == "all_daytime_in_bin")]
        if not row.empty:
            picp = float(row["PICP_PI"].iloc[0])
            acc = ("acceptable" if np.isfinite(picp) and picp >= COVERAGE_TARGET - 0.05
                   else "LOW (under-covered)" if np.isfinite(picp) else "n/a")
            out.append(f"- **{bin_name}**: PICP_PI = {_pct(picp)} vs target "
                       f"{_pct(COVERAGE_TARGET)} -> {acc}.")

    # Overall: do anomalous strata under-disperse?
    anomalous = g.loc[g.index.isin(SPECIFIC_ANOMALY_LABELS + ["rare_extreme_daytime"])]
    under_disp = anomalous[anomalous["std_growth_vs_error_growth"] < 1.0]
    if not under_disp.empty:
        names = ", ".join(under_disp.index)
        out.append(f"- **Overall**: MC std grows slower than error "
                   f"(std_growth_vs_error_growth < 1) for: {names}. "
                   "Consistent with the known under-dispersion of this single "
                   "MC-Dropout run.")
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True,
                   help="Path to the run's predictions.csv (physical watt space).")
    p.add_argument("--out-dir", required=True,
                   help="Directory for the output CSVs + report.")
    p.add_argument("--daytime-threshold", type=float,
                   default=DAYTIME_IRRADIANCE_THRESHOLD_WM2,
                   help="solar_irradiance_poa_target > threshold defines daytime "
                        f"(default {DAYTIME_IRRADIANCE_THRESHOLD_WM2}).")
    p.add_argument("--chunksize", type=int, default=1_000_000,
                   help="Rows per CSV chunk (default 1,000,000).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = args.predictions
    if not Path(pred_path).is_file():
        raise SystemExit(f"predictions file not found: {pred_path}")

    col = resolve_columns(pred_path)
    print(f"[cols] using: {col}")
    day, stats = load_daytime(pred_path, col, args.daytime_threshold, args.chunksize)

    overview = table_overview(day, stats)
    counts = table_counts(day)
    metrics = table_metrics(day)
    growth = table_growth(day)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overview.to_csv(out_dir / "daytime_anomaly_overview.csv", index=False)
    counts.to_csv(out_dir / "daytime_bin_anomaly_counts.csv", index=False)
    metrics.to_csv(out_dir / "daytime_bin_anomaly_metrics.csv", index=False)
    growth.to_csv(out_dir / "uncertainty_error_growth_by_anomaly.csv", index=False)
    report = render_report(counts, metrics, growth, overview,
                           args.daytime_threshold, len(day))
    (out_dir / "daytime_bin_anomaly_report.md").write_text(report, encoding="utf-8")

    print(f"[done] wrote 5 files to {out_dir}:")
    for f in ("daytime_anomaly_overview.csv", "daytime_bin_anomaly_counts.csv",
              "daytime_bin_anomaly_metrics.csv",
              "uncertainty_error_growth_by_anomaly.csv",
              "daytime_bin_anomaly_report.md"):
        print(f"  - {out_dir / f}")

    print("\n[summary] daytime anomaly overview:")
    print(overview.to_string(index=False))

    # Console summary of the two bins.
    print("\n[summary] counts in the two bins:")
    print(counts[counts["anomaly_label"] != "all_daytime_in_bin"]
          .pivot(index="anomaly_label", columns="production_bin", values="count")
          .to_string())
    print("\n[summary] growth vs normal_daytime "
          "(std_growth_vs_error_growth < 1 => std under-grows error):")
    print(growth[["stratum", "count", "MAE", "mean_std_mc",
                  "MAE_ratio_vs_normal_daytime", "std_ratio_vs_normal_daytime",
                  "std_growth_vs_error_growth"]].to_string(index=False))


if __name__ == "__main__":
    main()
