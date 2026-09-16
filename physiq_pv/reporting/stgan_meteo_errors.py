"""Meteorological subsets of quality-filtered STGAN forecast errors."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .daytime_bin_anomaly_report import _boxplot_stats
from .pointwise_detector_posthoc import _boolean_flags, _normalise_timestamp


CATEGORIES = {
    "normal": "Normali STGAN",
    "temperature": "Temperatura estrema",
    "wind": "Vento estremo",
    "irradiance_high": "Irradianza estrema alta",
    "irradiance_low": "Irradianza estrema bassa",
}
METEO_LABELS = {
    "temperature": ("temperature_2m", "extreme_temperature_condition"),
    "wind": ("wind_speed_10m", "extreme_wind_condition"),
    "irradiance_high": ("solar_irradiance_poa", "unusually_high_solar_potential"),
    "irradiance_low": ("solar_irradiance_poa", "unusually_low_solar_potential"),
}
QUALITY_POLICY = "isolated_regional_solar_dropout_plus_immediate_recovery"


def build_stgan_meteo_errors(
    evaluation_dir: str | Path,
    climatology_scores: str | Path,
    *,
    detail_horizons: tuple[int, ...] = (1, 6),
) -> dict[str, Path]:
    """Plot exact absolute-error boxes and between-location RMSE IQR bands.

    Normals retain the clean STGAN decision. Each extreme subset intersects
    STGAN anomalies with existing per-variable climatology flags at the exact
    location and target timestamp. Subsets may overlap; untyped anomalies are
    counted in the audit, never reassigned to normal. All plots are daytime.
    RMSE bands are descriptive spatial quartiles, not confidence intervals.
    """
    root = Path(evaluation_dir)
    provenance = json.loads((root / "evaluation_source.json").read_text(encoding="utf-8"))
    if (provenance.get("quality_filter_policy") != QUALITY_POLICY
            or not provenance.get("pvgis_quality_source")
            or provenance.get("clean_top_k_percent") is None):
        raise ValueError("A quality-filtered evaluation with clean STGAN top-K is required.")

    required = {
        "location", "timestamp", "horizon_hours", "y_true",
        "solar_irradiance_poa_target", "detector_is_anomaly", "data_quality_issue",
    }
    header = set(pd.read_csv(root / "predictions.csv", nrows=0).columns)
    prediction = "y_pred_mean" if "y_pred_mean" in header else "y_pred"
    missing = (required | {prediction}) - header
    if missing:
        raise ValueError(f"Quality-filtered predictions missing columns: {sorted(missing)}")
    work = pd.read_csv(root / "predictions.csv", usecols=sorted(required | {prediction}),
                       dtype={"location": str})
    work["timestamp"] = _normalise_timestamp(work["timestamp"])
    if work.duplicated(["location", "timestamp", "horizon_hours"]).any():
        raise ValueError("Duplicate location/target timestamp/horizon predictions.")
    issues = pd.read_csv(root / "pvgis_data_quality_issues.csv")
    invalid_times = _normalise_timestamp(issues["timestamp"])
    bad_quality = (_boolean_flags(work["data_quality_issue"])
                   | work["timestamp"].isin(invalid_times))
    audit = {"input_rows": len(work), "excluded_quality_rows_on_reload": int(bad_quality.sum()),
             "original_excluded_quality_rows": provenance.get("excluded_data_quality_rows"),
             "quality_filter_policy": QUALITY_POLICY,
             "evaluation_source": str((root / "evaluation_source.json").resolve()),
             "climatology_scores": str(Path(climatology_scores).resolve()),
             "category_policy": "clean_STGAN_anomaly_intersect_climatology; overlapping subsets",
             "rmse_band": "25th-75th percentiles of location RMSE; line = median location RMSE"}
    work = work.loc[~bad_quality].copy()
    numeric = ["y_true", prediction, "solar_irradiance_poa_target", "horizon_hours"]
    work[numeric] = work[numeric].apply(pd.to_numeric, errors="raise")
    horizons = sorted(work["horizon_hours"].unique())
    if not horizons or not np.isfinite(horizons).all() or any(h < 1 or h % 1 for h in horizons):
        raise ValueError("Expected positive integer forecast horizons.")
    if not set(detail_horizons) <= set(horizons):
        raise ValueError(f"Requested detail horizons {detail_horizons}; available: {horizons}")
    valid = np.isfinite(work[numeric]).all(axis=1)
    audit["excluded_nonfinite_rows"] = int((~valid).sum())
    day = work["solar_irradiance_poa_target"].gt(10.0)
    audit["excluded_nighttime_rows"] = int((valid & ~day).sum())
    work = work.loc[valid & day].copy()
    if work.empty:
        raise ValueError("No finite quality-filtered daytime predictions.")
    work["detector_is_anomaly"] = _boolean_flags(work["detector_is_anomaly"])

    # The solar semantic label also exists for PV power: require the irradiance
    # variable explicitly so production extremes cannot masquerade as irradiance.
    scores = pd.read_csv(climatology_scores, usecols=["location", "timestamp", "variable", "label"],
                         dtype={"location": str})
    scores["timestamp"] = _normalise_timestamp(scores["timestamp"])
    keys = pd.MultiIndex.from_frame(work[["location", "timestamp"]])
    for category, (variable, label) in METEO_LABELS.items():
        selected = scores.loc[scores["variable"].eq(variable) & scores["label"].eq(label)]
        flagged = pd.MultiIndex.from_frame(selected[["location", "timestamp"]])
        work[category] = work["detector_is_anomaly"] & keys.isin(flagged)
    work["normal"] = ~work["detector_is_anomaly"]
    n_conditions = work[list(METEO_LABELS)].sum(axis=1)
    work["untyped_anomaly"] = work["detector_is_anomaly"] & n_conditions.eq(0)
    work["multiple_conditions"] = n_conditions.gt(1)
    residual = work[prediction] - work["y_true"]
    work["abs_error"] = residual.abs()
    work["squared_error"] = residual ** 2

    # One shared reference across horizons, each physical target counted once.
    targets = work.drop_duplicates(["location", "timestamp"])
    positive = targets.loc[targets["y_true"].gt(0), "y_true"]
    if positive.empty:
        raise ValueError("No positive daytime production for the reference peak.")
    peak = float(positive.quantile(0.99))
    work["production_pct"] = (100 * work["y_true"] / peak).clip(0, 100)
    bins = {"all_daytime": "Tutte le produzioni diurne"}
    bins.update({f"{lo}_{lo + 20}_pct": f"Produzione {lo}-{lo + 20}%" for lo in range(0, 100, 20)})
    metrics, sites, counts = [], [], []
    for horizon in horizons:
        horizon_rows = work.loc[work["horizon_hours"].eq(horizon)]
        counts.append({"horizon_hours": horizon, "daytime_rows": len(horizon_rows),
                       **{c: int(horizon_rows[c].sum()) for c in CATEGORIES},
                       "untyped_anomaly": int(horizon_rows["untyped_anomaly"].sum()),
                       "multiple_conditions": int(horizon_rows["multiple_conditions"].sum())})
        for bin_name in bins:
            selected = horizon_rows
            if bin_name != "all_daytime":
                lo = int(bin_name.split("_")[0])
                pct = selected["production_pct"]
                selected = selected.loc[pct.ge(lo) & (pct.le(100) if lo == 80 else pct.lt(lo + 20))]
            for category in CATEGORIES:
                rows = selected.loc[selected[category]]
                site = rows.groupby("location").agg(count=("squared_error", "size"),
                                                      mse=("squared_error", "mean")).reset_index()
                site["rmse"] = np.sqrt(site.pop("mse"))
                site = site.assign(horizon_hours=horizon, bin=bin_name, category=category)
                sites.append(site)
                q = site["rmse"].quantile([0.25, 0.5, 0.75])
                metrics.append({"horizon_hours": horizon, "bin": bin_name, "category": category,
                                "count": len(rows), "n_locations": len(site),
                                "mae": rows["abs_error"].mean(),
                                "rmse": np.sqrt(rows["squared_error"].mean()),
                                "rmse_site_q1": q.loc[0.25], "rmse_site_median": q.loc[0.5],
                                "rmse_site_q3": q.loc[0.75],
                                **_boxplot_stats(rows["abs_error"].to_numpy(), "abs_error")})

    out = root / "meteo_errors"
    out.mkdir(parents=True, exist_ok=True)
    paths = {name: out / filename for name, filename in {
        "metrics": "meteo_error_metrics.csv", "site_metrics": "meteo_location_rmse.csv",
        "counts": "meteo_category_counts.csv", "audit": "meteo_error_audit.json",
    }.items()}
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(paths["metrics"], index=False)
    pd.concat(sites, ignore_index=True).to_csv(paths["site_metrics"], index=False)
    pd.DataFrame(counts).to_csv(paths["counts"], index=False)
    audit.update(reference_peak_w=peak, reference_peak_quantile=0.99,
                 reference_peak_scope="unique quality-filtered daytime location-targets",
                 daytime_threshold_wm2=10.0, daytime_rows=len(work))
    paths["audit"].write_text(json.dumps(audit, indent=2), encoding="utf-8")
    paths.update(_plot_metrics(metrics, out, bins, detail_horizons))
    return paths


def _plot_metrics(metrics, out, bins, detail_horizons):
    import matplotlib.pyplot as plt

    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]
    paths = {}
    for horizon in detail_horizons:
        fig, axes = plt.subplots(2, 3, figsize=(19, 10))
        for axis, (bin_name, title) in zip(axes.flat, bins.items()):
            rows = metrics.loc[metrics["horizon_hours"].eq(horizon) & metrics["bin"].eq(bin_name)]
            labels = []
            for position, ((category, label), color) in enumerate(zip(CATEGORIES.items(), colors), 1):
                row = rows.loc[rows["category"].eq(category)].iloc[0]
                labels.append(f"{label}\n(n={int(row['count']):,})")
                if row["count"]:
                    stats = {key: row[f"abs_error_{column}"] for key, column in {
                        "mean": "mean", "med": "median", "q1": "q1", "q3": "q3",
                        "whislo": "whisker_low", "whishi": "whisker_high",
                    }.items()}
                    axis.bxp([stats], positions=[position], showfliers=False, showmeans=True,
                             patch_artist=True, boxprops={"facecolor": color, "alpha": 0.5},
                             meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black"})
            axis.set_xticks(range(1, 6), labels, rotation=25, ha="right", fontsize=8)
            axis.set(title=title, ylabel="Errore assoluto [W]", xlim=(0.5, 5.5))
            if not rows["count"].sum():
                axis.text(0.5, 0.5, "Nessun campione", transform=axis.transAxes, ha="center")
            axis.set_ylim(bottom=0)
            axis.grid(axis="y", alpha=0.2)
        fig.suptitle(f"STGAN quality filtered - t+{horizon}h | rombo = MAE; box = quartili; outlier nascosti")
        fig.tight_layout()
        key = f"mae_boxplots_t{horizon}"
        paths[key] = out / f"{key}.png"
        fig.savefig(paths[key], dpi=140, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for axis, (bin_name, title) in zip(axes.flat, bins.items()):
        for (category, label), color in zip(CATEGORIES.items(), colors):
            rows = metrics.loc[metrics["bin"].eq(bin_name) & metrics["category"].eq(category)]
            axis.plot(rows["horizon_hours"], rows["rmse_site_median"], "o-", color=color, label=label)
            # A single site has no spatial spread to estimate.
            enough = rows["n_locations"].ge(2)
            axis.fill_between(rows["horizon_hours"], rows["rmse_site_q1"].where(enough),
                              rows["rmse_site_q3"].where(enough), color=color, alpha=0.15)
        axis.set(title=title, xlabel="Orizzonte di previsione [h]", ylabel="RMSE per località [W]")
        if not metrics.loc[metrics["bin"].eq(bin_name), "count"].sum():
            axis.text(0.5, 0.5, "Nessun campione", transform=axis.transAxes, ha="center")
        axis.set_ylim(bottom=0)
        axis.set_xticks(sorted(metrics["horizon_hours"].unique()))
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("STGAN quality filtered | linea = mediana fra località; banda = 25°-75° percentile (non IC)")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    paths["rmse_bands"] = out / "rmse_bands.png"
    fig.savefig(paths["rmse_bands"], dpi=140, bbox_inches="tight")
    plt.close(fig)
    return paths
