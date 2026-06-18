"""Run report + output writers: report.md, metrics CSVs, predictions.csv, meta."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from physiq_pv.data.pvgis_dataset import (
    DAYTIME_IRRADIANCE_THRESHOLD_WM2,
    GROUP_NORMAL,
    GROUP_RARE,
    PVGIS_STGNN_FEATURES,
    SPECIFIC_ANOMALY_LABELS,
)


def _render_report(global_df: pd.DataFrame, by_df: pd.DataFrame, meta: dict) -> str:
    lines: List[str] = []
    lines.append("# PVGIS-only ST-GNN forecasting report\n")
    lines.append(
        "Reuses the existing STGNN architecture on a **PVGIS-only** input. No real "
        "plant production, no ENERGIA, no quality score, no kWp/UPN. Anomaly labels "
        "are used **only** for stratified evaluation.\n"
    )
    lines.append("## Experiment\n")
    lines.append(f"- Mode: **{meta.get('mode', 'pvgis_stgnn')}**")
    lines.append(f"- Model type: **{meta.get('model_type', 'stgnn')}**")
    lines.append(f"- Feature set: **{meta.get('feature_set', 'full')}**")
    lines.append(f"- W&B enabled: **{bool(meta.get('wandb_enabled', False))}**")
    mc = meta.get("mc_dropout", False)
    lines.append(
        f"- MC Dropout: **{'enabled' if mc else 'disabled'}** "
        f"({'not implemented yet — reserved flag' if not mc else 'experimental'})\n"
    )

    lines.append("## Parameters\n")
    lines.append(f"- Target variable: **{meta['target_variable']}**")
    clip_max = meta.get("pv_target_clip_max", 1.5)
    clip_label = "none" if clip_max is None else clip_max
    lines.append(f"- PV normalized target upper clip: **{clip_label}**")
    if clip_max is None:
        lines.append("- PV normalized target lower clip: **0.0**")
    lines.append(f"- Selected features ({meta['n_features']}): {', '.join(meta['features'])}")
    loss_type = meta.get("loss_type", "mse")
    if loss_type == "huber":
        lines.append(f"- Training loss: **huber** (delta={meta.get('huber_delta', 1.0)})")
    else:
        lines.append("- Training loss: **mse**")
    if meta.get("train_mc_uncertainty_penalty", False):
        mode = meta.get("uncertainty_penalty_mode", "underdispersion")
        if mode == "sde_proxy":
            lines.append(
                "- Train-time MC uncertainty penalty: **enabled (mode=sde_proxy)** "
                f"(train_mc_samples={meta.get('train_mc_samples', 1)}, "
                f"sde_in_weight={meta.get('sde_proxy_in_weight', 0.001)}, "
                f"sde_out_weight={meta.get('sde_proxy_out_weight', 0.1)}, "
                f"std_min_ood={meta.get('sde_proxy_std_min_ood', 0.05)} [normalized "
                "target scale])"
            )
            lines.append(
                "- SDE-proxy penalty (SDE-Net style): minimise MC std on normal "
                "(in-distribution) cells, keep std above std_min_ood on "
                "rare_or_extreme (OOD) cells. Uses the anomaly mask -> requires "
                "anomaly-aware training (NOT eval-only-labels). Applied to PV only."
            )
        else:
            lines.append(
                "- Train-time MC uncertainty penalty: **enabled (mode=underdispersion)** "
                f"(train_mc_samples={meta.get('train_mc_samples', 1)}, "
                f"penalty_weight={meta.get('uncertainty_penalty_weight', 0.0)}, "
                f"k={meta.get('uncertainty_penalty_k', 1.0)}, "
                f"std_reg_weight={meta.get('uncertainty_std_reg_weight', 0.0)})"
            )
    else:
        lines.append("- Train-time MC uncertainty penalty: **disabled**")
    noise_std = meta.get("train_noise_std", 0.0)
    noise_prob = meta.get("train_noise_prob", 0.0)
    noise_mode = meta.get("train_noise_mode", "random")
    anom_std = meta.get("anomaly_noise_std", 0.0)
    anom_prob = meta.get("anomaly_noise_prob", 0.0)
    anomaly_noise_on = noise_mode == "anomaly" and anom_std > 0.0 and anom_prob > 0.0
    random_noise_on = noise_std > 0.0 and noise_prob > 0.0
    if anomaly_noise_on:
        lines.append(
            f"- Train input noise: **enabled (anomaly-aware)** "
            f"(anomaly std={anom_std}, anomaly prob={anom_prob}; "
            f"normal std={noise_std}, normal prob={noise_prob}; "
            f"sin_elev/cos_elev excluded; train only)"
        )
        lines.append(
            "- ⚠️ **Anomaly-aware training**: anomaly labels are used during "
            "TRAINING to target input noise on rare_or_extreme samples. This run "
            "is therefore NOT an eval-only-labels configuration — the usual "
            "'anomaly labels used only for evaluation' guarantee does NOT hold."
        )
    elif random_noise_on:
        lines.append(
            f"- Train input noise: **enabled (random)** (std={noise_std}, "
            f"prob={noise_prob}; sin_elev/cos_elev excluded; train only)"
        )
    else:
        lines.append("- Train input noise: **disabled**")
    lines.append(f"- Use irradiance head: **{bool(meta.get('use_irradiance_head', True))}**")
    lines.append(f"- Use irradiance loss: **{bool(meta.get('use_irradiance_loss', False))}**")
    lines.append(f"- Irradiance loss weight (kt aux): **{meta.get('irradiance_loss_weight', 1.0)}**")
    if meta.get("use_irradiance_loss", False):
        lines.append(
            "- KT target definition: target-time clear-sky index proxy "
            "`kt = solar_irradiance_poa / clearsky_GHI` (pvlib Ineichen, fallback "
            "simplified-Solis; eps=1e-6; 0 when clearsky_GHI <= 0.1 kW/m2), "
            "clipped to [0, 1.5] in the dataset, re-clipped to [0, KT_MAX=1.2] in "
            "the loss to match `pred_kt = sigmoid(head_ghi) * 1.2`. PVGIS values "
            "at the TARGET timestamp; supervision target only, never a model "
            "input. Not normalised (kt is already dimensionless)."
        )
    if mc:
        lines.append(f"- MC samples: **{meta.get('mc_samples')}**")
    lines.append(f"- seq_len: **{meta['seq_len']}**  |  horizon: **{meta['horizon']}**")
    lines.append(f"- Train years: {meta['train_years']}")
    lines.append(f"- Test year: **{meta['test_year']}**")
    lines.append(f"- Nodes (locations): **{meta['n_nodes']}**  |  epochs: **{meta['epochs']}**")
    if meta.get("batch_size") is not None or meta.get("lr") is not None:
        lines.append(f"- batch_size: {meta.get('batch_size')}  |  lr: {meta.get('lr')}")
    lines.append(f"- Anomaly scores: {meta['anomaly_scores'] or '(none — all normal)'}")
    lines.append(f"- Predictions: **{meta['n_predictions']}**")
    lines.append(f"- Device: {meta['device']}  |  Generated (UTC): {meta['generated_utc']}")
    lines.append(f"- W&B artifacts uploaded: **{bool(meta.get('wandb_artifacts_uploaded', False))}**")
    lines.append(f"- Post-hoc analysis executed: **{bool(meta.get('posthoc_executed', False))}**")
    lines.append(f"- Post-hoc artifact uploaded: **{bool(meta.get('posthoc_uploaded', False))}**\n")

    def _fmt(v, nd=4):
        return f"{float(v):.{nd}f}" if v is not None and pd.notna(v) else "—"

    def _by_value(by_index: pd.DataFrame, stratum: str, col: str):
        if not by_index.empty and stratum in by_index.index and col in by_index.columns:
            return by_index.loc[stratum, col]
        return float("nan")

    g = global_df.iloc[0]
    lines.append("## Global metrics\n")
    if mc:
        lines.append("| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw |")
        lines.append("|---|---|---|---|---|---|---|---|")
        lines.append(
            f"| {g['stratum']} | {int(g['count'])} | {_fmt(g['MAE'])} | {_fmt(g['RMSE'])} | "
            f"{_fmt(g['mean_pred_std'])} | {_fmt(g['median_pred_std'])} | {_fmt(g['p90_pred_std'])} | "
            f"{_fmt(g['coverage_95_raw'], 3)} |\n"
        )
    else:
        lines.append("| stratum | count | MAE | RMSE |")
        lines.append("|---|---|---|---|")
        lines.append(f"| {g['stratum']} | {int(g['count'])} | {g['MAE']:.4f} | {g['RMSE']:.4f} |\n")

    lines.append("## Metrics by anomaly stratum\n")
    if by_df.empty:
        lines.append("_No strata available._\n")
    elif mc:
        lines.append("| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in by_df.iterrows():
            lines.append(
                f"| {r['stratum']} | {int(r['count'])} | {_fmt(r['MAE'])} | {_fmt(r['RMSE'])} | "
                f"{_fmt(r['mean_pred_std'])} | {_fmt(r['median_pred_std'])} | {_fmt(r['p90_pred_std'])} | "
                f"{_fmt(r['coverage_95_raw'], 3)} |"
            )
        lines.append("")
    else:
        lines.append("| stratum | count | MAE | RMSE |")
        lines.append("|---|---|---|---|")
        for _, r in by_df.iterrows():
            lines.append(f"| {r['stratum']} | {int(r['count'])} | {r['MAE']:.4f} | {r['RMSE']:.4f} |")
        lines.append("")

    lines.append("## Does ST-GNN degrade on rare/extreme PVGIS conditions?\n")
    lines.append(
        "The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of "
        "the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly "
        "labels are used only for evaluation/stratification and are never used as "
        "model inputs or targets.\n"
    )
    by = by_df.set_index("stratum") if not by_df.empty else pd.DataFrame()
    if "group:normal" in by.index and "group:rare_or_extreme" in by.index:
        mae_n = by.loc["group:normal", "MAE"]
        mae_r = by.loc["group:rare_or_extreme", "MAE"]
        ratio = mae_r / mae_n if mae_n else float("nan")
        if np.isnan(ratio):
            verdict = "inconclusive (normal MAE is zero)"
        elif ratio > 1.1:
            verdict = "**yes** — ST-GNN is worse on rare/extreme conditions"
        elif ratio < 0.9:
            verdict = "no — ST-GNN is actually better on rare/extreme conditions"
        else:
            verdict = "comparable — no clear degradation"
        lines.append(f"- MAE normal: {mae_n:.4f}  |  MAE rare/extreme: {mae_r:.4f}  |  ratio: **{ratio:.2f}×**")
        lines.append(f"- Verdict: {verdict}.\n")
    else:
        lines.append("_Not enough strata to compare (no rare/extreme points in the test year)._\n")

    if mc:
        by = by_df.set_index("stratum") if not by_df.empty else pd.DataFrame()
        lines.append("## Uncertainty by anomaly stratum\n")
        have = (
            not by.empty
            and "group:normal" in by.index
            and "group:rare_or_extreme" in by.index
            and "mean_pred_std" in by.columns
        )
        if have:
            mae_n = by.loc["group:normal", "MAE"]
            mae_r = by.loc["group:rare_or_extreme", "MAE"]
            unc_n = by.loc["group:normal", "mean_pred_std"]
            unc_r = by.loc["group:rare_or_extreme", "mean_pred_std"]
            cov_n = by.loc["group:normal", "coverage_95_raw"]
            cov_r = by.loc["group:rare_or_extreme", "coverage_95_raw"]
            mae_ratio = mae_r / mae_n if mae_n else float("nan")
            unc_ratio = unc_r / unc_n if unc_n else float("nan")

            def _verdict(r):
                if np.isnan(r):
                    return "inconclusive"
                return "**yes**" if r > 1.1 else ("no" if r < 0.9 else "comparable")

            lines.append(f"- MC samples: **{meta.get('mc_samples')}**")
            lines.append(f"- MAE normal: {mae_n:.4f}  |  MAE rare/extreme: {mae_r:.4f}  |  rare/normal MAE ratio: **{mae_ratio:.2f}×**")
            lines.append(
                f"- Mean uncertainty (std) normal: {unc_n:.4f}  |  rare/extreme: {unc_r:.4f}  "
                f"|  rare/normal uncertainty ratio: **{unc_ratio:.2f}×**"
            )
            lines.append(
                f"- Gaussian coverage@95 (diagnostic) normal: {_fmt(cov_n, 3)}  |  "
                f"rare/extreme: {_fmt(cov_r, 3)}"
            )
            lines.append(
                "- Primary paper-style PI coverage (PICP) is reported in "
                "*Interval reliability & sharpness*.\n"
            )
            lines.append(f"1. Does the model err more on rare/extreme? {_verdict(mae_ratio)} (MAE ratio {mae_ratio:.2f}×).")
            lines.append(f"2. Is the model also more uncertain on rare/extreme? {_verdict(unc_ratio)} (uncertainty ratio {unc_ratio:.2f}×).\n")
        else:
            lines.append("_Not enough strata for an uncertainty comparison._\n")

    iv = meta.get("interval_metrics")
    if iv:
        lines.append("## Interval reliability & sharpness (PICP / NMPIL / CLC)\n")
        lines.append(
            "Paper-style evaluation (uncertainty-aware rainfall prediction). The "
            "**primary predictive intervals (`pi`) are built directly from the MC "
            "Dropout sample distribution** (empirical quantiles q(alpha/2), "
            "q(1-alpha/2)). The Gaussian band (`gaussian`, mean ± 1.96·std_raw) is "
            "a secondary diagnostic only.\n"
        )
        lines.append(
            "- **PICP** measures empirical coverage (fraction of y_true inside the "
            "interval). It is **evaluated, not forced** to 0.95 — no factor is fit "
            "to hit the target in the main protocol."
        )
        lines.append("- **NMPIL** measures normalized interval width (MPIW / target_range).")
        lines.append(
            "- **CLC** measures the sharpness/reliability trade-off: "
            "`CLC = NMPIL·(1 + exp(-eta·(PICP - gamma)))` (lower is better once PICP >= gamma)."
        )
        lines.append(
            "- A very low PICP for `pi` means raw MC Dropout is sharp but **not "
            "reliable** in this PVGIS-only setting."
        )
        lines.append(
            f"- gamma (coverage target): **{_fmt(meta.get('clc_gamma'), 3)}**  |  "
            f"eta (clc_eta): **{_fmt(meta.get('clc_eta'), 2)}**  |  "
            f"target_range: **{_fmt(meta.get('target_range'), 4)}**\n"
        )

        # Ordered: pi (primary) first, then diagnostics that are present.
        kinds = [("pi", "PI (primary, MC quantiles)")]
        if "gaussian" in iv:
            kinds.append(("gaussian", "Gaussian (diagnostic)"))
        for kind, label in kinds:
            lines.append(f"### {label}\n")
            lines.append("| stratum | PICP | MPIW | NMPIL | CLC |")
            lines.append("|---|---|---|---|---|")
            for gname in ("global", "normal", "rare_extreme"):
                m = iv.get(kind, {}).get(gname, {})
                if not m:
                    continue
                lines.append(
                    f"| {gname} | {_fmt(m.get('picp'), 3)} | {_fmt(m.get('mpiw'), 4)} | "
                    f"{_fmt(m.get('nmpil'), 4)} | {_fmt(m.get('clc'), 4)} |"
                )
            lines.append("")

    daytime_metrics = meta.get("daytime_metrics")
    if daytime_metrics:
        threshold = meta.get(
            "daytime_threshold_wm2", DAYTIME_IRRADIANCE_THRESHOLD_WM2
        )
        lines.append("## Daytime-only interval reliability\n")
        lines.append(
            "Eval-only split based on PVGIS `solar_irradiance_poa` at the target "
            f"timestamp: daytime > **{_fmt(threshold, 1)} W/m²**, nighttime <= "
            f"**{_fmt(threshold, 1)} W/m²**. The irradiance is diagnostic metadata "
            "and is not added to the model inputs or targets.\n"
        )
        selected_strata = (
            "daytime",
            "nighttime",
            "normal_daytime",
            "rare_extreme_daytime",
            "high_daytime",
            "peak_daytime",
            "extreme_peak_daytime",
        )
        lines.append(
            "| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | "
            "PICP PI | MPIW PI | NMPIL PI | CLC PI | PICP Gaussian | "
            "MPIW Gaussian | NMPIL Gaussian | CLC Gaussian |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|"
        )
        for stratum in selected_strata:
            metrics = daytime_metrics.get(stratum, {})
            if not metrics:
                continue
            lines.append(
                f"| {stratum} | {int(metrics.get('count', 0))} | "
                f"{_fmt(metrics.get('mae'))} | {_fmt(metrics.get('rmse'))} | "
                f"{_fmt(metrics.get('mean_std'))} | "
                f"{_fmt(metrics.get('median_std'))} | "
                f"{_fmt(metrics.get('p90_std'))} | "
                f"{_fmt(metrics.get('picp_pi'), 3)} | "
                f"{_fmt(metrics.get('mpiw_pi'))} | "
                f"{_fmt(metrics.get('nmpil_pi'))} | "
                f"{_fmt(metrics.get('clc_pi'))} | "
                f"{_fmt(metrics.get('picp_gaussian'), 3)} | "
                f"{_fmt(metrics.get('mpiw_gaussian'))} | "
                f"{_fmt(metrics.get('nmpil_gaussian'))} | "
                f"{_fmt(metrics.get('clc_gaussian'))} |"
            )
        lines.append("")

        peak_strata = (
            "high_daytime",
            "peak_daytime",
            "extreme_peak_daytime",
        )
        lines.append("### Daytime production-tail diagnostics\n")
        lines.append(
            "| stratum | count | MAE | RMSE | mean residual | median residual | "
            "fraction underprediction | fraction above PI | PICP PI | "
            "PICP Gaussian |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for stratum in peak_strata:
            metrics = daytime_metrics.get(stratum, {})
            if not metrics:
                continue
            lines.append(
                f"| {stratum} | {int(metrics.get('count', 0))} | "
                f"{_fmt(metrics.get('mae'))} | {_fmt(metrics.get('rmse'))} | "
                f"{_fmt(metrics.get('mean_residual'))} | "
                f"{_fmt(metrics.get('median_residual'))} | "
                f"{_fmt(metrics.get('fraction_underprediction'), 3)} | "
                f"{_fmt(metrics.get('fraction_above_interval'), 3)} | "
                f"{_fmt(metrics.get('picp_pi'), 3)} | "
                f"{_fmt(metrics.get('picp_gaussian'), 3)} |"
            )
        lines.append("")

        lines.append(
            "| stratum | fraction y_true=0 | fraction lower PI <= 0 | "
            "fraction lower Gaussian <= 0 | PI coverage y=0 | "
            "Gaussian coverage y=0 | PI coverage y>0 | "
            "Gaussian coverage y>0 |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for stratum in selected_strata:
            metrics = daytime_metrics.get(stratum, {})
            if not metrics:
                continue
            lines.append(
                f"| {stratum} | "
                f"{_fmt(metrics.get('fraction_y_true_zero'), 3)} | "
                f"{_fmt(metrics.get('fraction_lower_pi_leq_zero'), 3)} | "
                f"{_fmt(metrics.get('fraction_lower_gaussian_leq_zero'), 3)} | "
                f"{_fmt(metrics.get('coverage_pi_y_true_zero'), 3)} | "
                f"{_fmt(metrics.get('coverage_gaussian_y_true_zero'), 3)} | "
                f"{_fmt(metrics.get('coverage_pi_y_true_positive'), 3)} | "
                f"{_fmt(metrics.get('coverage_gaussian_y_true_positive'), 3)} |"
            )
        lines.append("")

        global_picp = daytime_metrics.get("global", {}).get("picp_pi")
        daytime_picp = daytime_metrics.get("daytime", {}).get("picp_pi")
        if (
            global_picp is not None
            and daytime_picp is not None
            and np.isfinite(global_picp)
            and np.isfinite(daytime_picp)
        ):
            delta = daytime_picp - global_picp
            if delta >= 0.10:
                lines.append(
                    f"**PICP PI daytime is materially higher than global** "
                    f"({_fmt(daytime_picp, 3)} vs {_fmt(global_picp, 3)}, "
                    f"delta {_fmt(delta, 3)})."
                )
                if daytime_picp < 0.90:
                    lines.append(
                        "It nevertheless remains low relative to the 0.95 "
                        "coverage target.\n"
                    )
                else:
                    lines.append("")
            elif daytime_picp < 0.90:
                lines.append(
                    f"**PICP PI remains low also during daytime** "
                    f"({_fmt(daytime_picp, 3)} vs global "
                    f"{_fmt(global_picp, 3)}, delta {_fmt(delta, 3)}).\n"
                )
            else:
                lines.append(
                    f"**PICP PI daytime is close to the target but not materially "
                    f"higher than global** ({_fmt(daytime_picp, 3)} vs "
                    f"{_fmt(global_picp, 3)}, delta {_fmt(delta, 3)}).\n"
                )

    residual_rows = meta.get("residual_bias_metrics") or []
    if residual_rows:
        residual_by = {row["stratum"]: row for row in residual_rows}

        def _pct(value):
            return (
                f"{100.0 * float(value):.1f}%"
                if value is not None and pd.notna(value)
                else "—"
            )

        lines.append("## Residual bias diagnostics by stratum\n")
        lines.append(
            "`residual = y_pred_mean - y_true`: positive means overprediction, "
            "negative means underprediction.\n"
        )
        lines.append(
            "| stratum | count | MAE | RMSE | mean_residual | median_residual | "
            "overprediction% | underprediction% | PICP PI | above_interval% | "
            "below_interval% |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        residual_strata = (
            "global",
            "daytime",
            "nighttime",
            "normal",
            "rare_extreme",
            "normal_daytime",
            "rare_extreme_daytime",
            "normal_nighttime",
            "rare_extreme_nighttime",
            "label:unusually_low_solar_potential",
            "label:unusually_high_solar_potential",
            "label:extreme_temperature_condition",
            "label:extreme_wind_condition",
        )
        for stratum in residual_strata:
            row = residual_by.get(stratum)
            if row is None:
                continue
            lines.append(
                f"| {stratum} | {int(row.get('count', 0))} | "
                f"{_fmt(row.get('mae'))} | {_fmt(row.get('rmse'))} | "
                f"{_fmt(row.get('mean_residual'))} | "
                f"{_fmt(row.get('median_residual'))} | "
                f"{_pct(row.get('fraction_overprediction'))} | "
                f"{_pct(row.get('fraction_underprediction'))} | "
                f"{_fmt(row.get('picp_pi'), 3)} | "
                f"{_pct(row.get('fraction_above_interval'))} | "
                f"{_pct(row.get('fraction_below_interval'))} |"
            )
        lines.append("")

        lines.append("## Daytime production-bin diagnostics\n")
        lines.append(
            "Bins use physical `y_true` in watts and only samples with target-time "
            "`solar_irradiance_poa > 10 W/m²`. Intervals are `[lower, upper)`, "
            "with the final bin `y_true >= 100 W`.\n"
        )
        lines.append(
            "| bin | count | MAE | RMSE | mean_residual | median_residual | "
            "underprediction% | overprediction% | PICP PI | MPIW PI | "
            "above_interval% | below_interval% |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        production_bins = (
            "daytime_0_20",
            "daytime_20_40",
            "daytime_40_60",
            "daytime_60_80",
            "daytime_80_100",
            "daytime_gt_100",
        )
        for stratum in production_bins:
            row = residual_by.get(stratum)
            if row is None:
                continue
            lines.append(
                f"| {stratum} | {int(row.get('count', 0))} | "
                f"{_fmt(row.get('mae'))} | {_fmt(row.get('rmse'))} | "
                f"{_fmt(row.get('mean_residual'))} | "
                f"{_fmt(row.get('median_residual'))} | "
                f"{_pct(row.get('fraction_underprediction'))} | "
                f"{_pct(row.get('fraction_overprediction'))} | "
                f"{_fmt(row.get('picp_pi'), 3)} | "
                f"{_fmt(row.get('mpiw_pi'))} | "
                f"{_pct(row.get('fraction_above_interval'))} | "
                f"{_pct(row.get('fraction_below_interval'))} |"
            )
        lines.append("")

        lines.append("## Automatic interpretation of residual asymmetry\n")

        def _bias_line(label, stratum):
            row = residual_by.get(stratum, {})
            if not row or not row.get("count"):
                return f"- **{label}:** no samples available."
            mean_residual = row.get("mean_residual")
            if mean_residual < 0:
                direction = "underprediction"
            elif mean_residual > 0:
                direction = "overprediction"
            else:
                direction = "no mean bias"
            return (
                f"- **{label}:** {direction}; mean residual "
                f"{_fmt(mean_residual)} W, over {_pct(row.get('fraction_overprediction'))}, "
                f"under {_pct(row.get('fraction_underprediction'))}."
            )

        lines.append(_bias_line("Global", "global"))
        lines.append(_bias_line("Daytime", "daytime"))
        lines.append(_bias_line("Nighttime", "nighttime"))
        lines.append(
            _bias_line(
                "Unusually low solar potential",
                "label:unusually_low_solar_potential",
            )
        )
        lines.append(
            _bias_line(
                "Unusually high solar potential",
                "label:unusually_high_solar_potential",
            )
        )
        lines.append(_bias_line("Rare/extreme daytime", "rare_extreme_daytime"))
        lines.append(_bias_line("Production >= 100 W", "daytime_gt_100"))

        night = residual_by.get("nighttime", {})
        night_day_metrics = (daytime_metrics or {}).get("nighttime", {})
        if (
            night.get("count", 0)
            and night.get("fraction_below_interval", 0.0)
            > night.get("fraction_above_interval", 0.0)
            and night_day_metrics.get("fraction_y_true_zero", 0.0) >= 0.5
            and night_day_metrics.get("fraction_lower_pi_leq_zero", 1.0) < 0.5
        ):
            lines.append(
                "- **Nighttime softplus signature:** misses are predominantly below "
                "the PI while most targets are zero and most empirical lower bounds "
                "remain positive. This is consistent with `softplus` plus `y_true=0`."
            )

        low = residual_by.get("label:unusually_low_solar_potential", {})
        if (
            low.get("count", 0)
            and low.get("mean_residual", 0.0) > 0.0
            and low.get("fraction_overprediction", 0.0) > 0.5
        ):
            lines.append(
                "- The model tends to **overpredict unusually low solar potential**."
            )
        high = residual_by.get("label:unusually_high_solar_potential", {})
        if (
            high.get("count", 0)
            and high.get("mean_residual", 0.0) < 0.0
            and high.get("fraction_underprediction", 0.0) > 0.5
        ):
            lines.append(
                "- The model tends to **underpredict unusually high solar potential**."
            )

        peak = residual_by.get("daytime_gt_100", {})
        if peak.get("count", 0) and peak.get("mean_residual", 0.0) < 0.0:
            lines.append(
                "- The model **underpredicts the >=100 W production bin**. "
                f"Targets exceed `upper_pi` in "
                f"{_pct(peak.get('fraction_above_interval'))} of these samples."
            )
            if meta.get("pv_target_clip_max", 1.5) is not None:
                lines.append(
                    "- The active normalized-target upper clip is consistent with a "
                    "peak-smoothing hypothesis, but this diagnostic is observational "
                    "and does not establish causality."
                )

        low_bin = residual_by.get("daytime_0_20", {})
        if low_bin.get("count", 0) and low_bin.get("mean_residual", 0.0) > 0.0:
            lines.append("- The model overpredicts the low daytime production bin.")

        global_row = residual_by.get("global", {})
        below = global_row.get("fraction_below_interval")
        above = global_row.get("fraction_above_interval")
        if below is not None and above is not None:
            if below > 1.25 * above:
                miss_diagnosis = "intervals/centres are predominantly too high"
            elif above > 1.25 * below:
                miss_diagnosis = "intervals/centres are predominantly too low"
            else:
                miss_diagnosis = (
                    "misses occur on both sides, consistent with intervals that are "
                    "too narrow and/or condition-dependent centre bias"
                )
            lines.append(
                f"- **Global PI miss direction:** below {_pct(below)}, above "
                f"{_pct(above)}; {miss_diagnosis}."
            )
            global_picp = global_row.get("picp_pi")
            if global_picp is not None and global_picp < meta.get("clc_gamma", 0.95):
                night_below = night.get("fraction_below_interval", 0.0)
                peak_above = peak.get("fraction_above_interval", 0.0)
                causes = []
                if (
                    night_below > 0.5
                    and night_day_metrics.get("fraction_y_true_zero", 0.0) >= 0.5
                ):
                    causes.append("softplus/zero-target nighttime misses")
                if peak.get("count", 0) and peak_above > 0.25:
                    causes.append("high-production targets above the PI")
                if 0.8 <= (below / above if above else float("inf")) <= 1.25:
                    causes.append("intervals that are too narrow on both sides")
                cause_text = (
                    ", ".join(causes)
                    if causes
                    else "the dominant miss direction reported above"
                )
                lines.append(
                    f"- **Likely cause of low PICP:** {cause_text}. Centre bias and "
                    "interval width should be interpreted together."
                )
        lines.append("")

    return "\n".join(lines) + "\n"


def write_outputs(
    predictions: pd.DataFrame,
    global_df: pd.DataFrame,
    by_df: pd.DataFrame,
    out_dir: str,
    meta: dict,
    skip_predictions: bool = False,
) -> Dict[str, Path]:
    """Write metrics + report (+ predictions.csv unless `skip_predictions`).

    When `skip_predictions` is set, predictions.csv is not written and
    paths["predictions"] is None. The legacy metrics CSVs and report.md are
    always produced; metrics_daytime.csv is produced when daytime diagnostics
    are available.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": None if skip_predictions else out / "predictions.csv",
        "metrics_global": out / "metrics_global.csv",
        "metrics_by_anomaly_label": out / "metrics_by_anomaly_label.csv",
        "metrics_daytime": (
            out / "metrics_daytime.csv"
            if meta.get("daytime_metrics")
            else None
        ),
        "residual_bias_metrics": (
            out / "residual_bias_and_bin_metrics.csv"
            if meta.get("residual_bias_metrics")
            else None
        ),
        "report": out / "report.md",
    }
    if not skip_predictions:
        predictions.to_csv(paths["predictions"], index=False)
    global_df.to_csv(paths["metrics_global"], index=False)
    by_df.to_csv(paths["metrics_by_anomaly_label"], index=False)
    if paths["metrics_daytime"] is not None:
        rows = []
        for stratum, metrics in meta["daytime_metrics"].items():
            row = {
                "stratum": "all" if stratum == "global" else stratum,
                **metrics,
            }
            rows.append(row)
        pd.DataFrame(rows).to_csv(paths["metrics_daytime"], index=False)
    if paths["residual_bias_metrics"] is not None:
        pd.DataFrame(meta["residual_bias_metrics"]).to_csv(
            paths["residual_bias_metrics"], index=False
        )
    write_report(paths["report"], global_df, by_df, meta)
    return paths


def write_report(
    report_path: Path,
    global_df: pd.DataFrame,
    by_df: pd.DataFrame,
    meta: dict,
) -> None:
    """Write report.md, including final post-processing and upload outcomes."""
    Path(report_path).write_text(
        _render_report(global_df, by_df, meta),
        encoding="utf-8",
    )


def build_meta(
    args_like: dict,
    n_predictions: int,
    n_nodes: int,
    features: Optional[List[str]] = None,
) -> dict:
    feats = list(features) if features is not None else list(PVGIS_STGNN_FEATURES)
    return {
        "mode": args_like.get("mode", "pvgis_stgnn"),
        "model_type": args_like.get("model_type", "stgnn"),
        "feature_set": args_like.get("feature_set", "full"),
        "target_variable": args_like["target_variable"],
        "pv_target_clip_max": args_like.get("pv_target_clip_max", 1.5),
        "features": feats,
        "n_features": len(feats),
        "use_irradiance_head": args_like.get("use_irradiance_head", True),
        "use_irradiance_loss": args_like.get("use_irradiance_loss", False),
        "irradiance_loss_weight": args_like.get("irradiance_loss_weight", 1.0),
        "loss_type": args_like.get("loss_type", "mse"),
        "huber_delta": args_like.get("huber_delta", 1.0),
        "train_mc_uncertainty_penalty": args_like.get(
            "train_mc_uncertainty_penalty", False
        ),
        "train_mc_samples": args_like.get("train_mc_samples", 1),
        "uncertainty_penalty_mode": args_like.get(
            "uncertainty_penalty_mode", "underdispersion"
        ),
        "uncertainty_penalty_weight": args_like.get("uncertainty_penalty_weight", 0.0),
        "uncertainty_penalty_k": args_like.get("uncertainty_penalty_k", 1.0),
        "uncertainty_std_reg_weight": args_like.get("uncertainty_std_reg_weight", 0.0),
        "sde_proxy_in_weight": args_like.get("sde_proxy_in_weight", 0.001),
        "sde_proxy_out_weight": args_like.get("sde_proxy_out_weight", 0.1),
        "sde_proxy_std_min_ood": args_like.get("sde_proxy_std_min_ood", 0.05),
        "train_noise_std": args_like.get("train_noise_std", 0.0),
        "train_noise_prob": args_like.get("train_noise_prob", 0.0),
        "train_noise_mode": args_like.get("train_noise_mode", "random"),
        "anomaly_noise_std": args_like.get("anomaly_noise_std", 0.0),
        "anomaly_noise_prob": args_like.get("anomaly_noise_prob", 0.0),
        "seq_len": args_like["seq_len"],
        "horizon": args_like["horizon"],
        "train_years": args_like["train_years"],
        "test_year": args_like["test_year"],
        "n_nodes": n_nodes,
        "epochs": args_like["epochs"],
        "batch_size": args_like.get("batch_size"),
        "lr": args_like.get("lr"),
        "anomaly_scores": args_like.get("anomaly_scores"),
        "device": args_like.get("device", "cpu"),
        "wandb_enabled": args_like.get("wandb_enabled", False),
        "mc_dropout": args_like.get("mc_dropout", False),
        "mc_samples": args_like.get("mc_samples"),
        "coverage_target": args_like.get("coverage_target"),
        "n_predictions": n_predictions,
        "wandb_artifacts_uploaded": args_like.get("wandb_artifacts_uploaded", False),
        "posthoc_executed": args_like.get("posthoc_executed", False),
        "posthoc_uploaded": args_like.get("posthoc_uploaded", False),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
