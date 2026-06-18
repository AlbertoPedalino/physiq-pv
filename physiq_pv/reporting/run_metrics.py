"""Run metrics: per-stratum MAE/RMSE/PICP/MPIW + the W&B metric payload."""
from __future__ import annotations

import numpy as np
import pandas as pd

from physiq_pv.data.pvgis_dataset import (
    GROUP_NORMAL,
    GROUP_RARE,
    SPECIFIC_ANOMALY_LABELS,
)


def _metric_row(stratum: str, df: pd.DataFrame) -> dict:
    n = len(df)
    row = {
        "stratum": stratum,
        "count": int(n),
        "MAE": float(df["abs_error"].mean()) if n else float("nan"),
        "RMSE": float(np.sqrt(df["squared_error"].mean())) if n else float("nan"),
    }
    # Uncertainty columns: populated only when MC-Dropout produced y_pred_std.
    # Always present (NaN in the deterministic path) so the CSV schema is stable.
    if n and "y_pred_std" in df.columns:
        std = df["y_pred_std"].to_numpy(dtype=float)
        row["mean_pred_std"] = float(np.mean(std))
        row["median_pred_std"] = float(np.median(std))
        row["p90_pred_std"] = float(np.percentile(std, 90))
        inside = (df["y_true"] >= df["y_pred_lower"]) & (df["y_true"] <= df["y_pred_upper"])
        row["coverage_95"] = float(inside.mean())
        row["coverage_95_raw"] = row["coverage_95"]
    else:
        row["mean_pred_std"] = float("nan")
        row["median_pred_std"] = float("nan")
        row["p90_pred_std"] = float("nan")
        row["coverage_95"] = float("nan")
        row["coverage_95_raw"] = float("nan")
    return row


def compute_metrics(predictions: pd.DataFrame) -> tuple:
    """Global + per-stratum (anomaly group / specific label) metrics."""
    def _row(stratum: str, df: pd.DataFrame) -> dict:
        return _metric_row(stratum, df)

    global_df = pd.DataFrame([_row("all", predictions)])
    rows = []
    for group in (GROUP_NORMAL, GROUP_RARE):
        sub = predictions[predictions["anomaly_group"] == group]
        if len(sub):
            rows.append(_row(f"group:{group}", sub))
    for label in SPECIFIC_ANOMALY_LABELS:
        mask = predictions["anomaly_label"].apply(
            lambda d: label in d.split(",") if d else False
        )
        sub = predictions[mask]
        if len(sub):
            rows.append(_row(f"label:{label}", sub))
    return global_df, pd.DataFrame(rows)


def build_wandb_metrics(
    global_df: pd.DataFrame,
    by_df: pd.DataFrame,
    mc_dropout: bool = False,
) -> dict:
    """
    Flatten global + by-stratum metrics into namespaced W&B scalars.

    Keys: mae/global, rmse/global, mae|rmse/{normal,rare_extreme},
    ratio/{mae,rmse}_rare_normal, and (when mc_dropout) uncertainty/* (mean +
    p90 std). Only finite (numeric) values are emitted.
    """
    g = global_df.iloc[0]
    by = by_df.set_index("stratum") if not by_df.empty else pd.DataFrame()

    def _get(stratum: str, col: str):
        if not by.empty and stratum in by.index and col in by.columns:
            v = by.loc[stratum, col]
            return float(v) if pd.notna(v) else None
        return None

    out: dict = {"mae/global": float(g["MAE"]), "rmse/global": float(g["RMSE"])}

    mae_n, mae_r = _get("group:normal", "MAE"), _get("group:rare_or_extreme", "MAE")
    rmse_n, rmse_r = _get("group:normal", "RMSE"), _get("group:rare_or_extreme", "RMSE")
    if mae_n is not None:
        out["mae/normal"] = mae_n
    if mae_r is not None:
        out["mae/rare_extreme"] = mae_r
    if rmse_n is not None:
        out["rmse/normal"] = rmse_n
    if rmse_r is not None:
        out["rmse/rare_extreme"] = rmse_r
    if mae_n and mae_r is not None:
        out["ratio/mae_rare_normal"] = mae_r / mae_n
    if rmse_n and rmse_r is not None:
        out["ratio/rmse_rare_normal"] = rmse_r / rmse_n

    if mc_dropout:
        std_g = float(g.get("mean_pred_std", float("nan")))
        if pd.notna(std_g):
            out["uncertainty/mean_std_global"] = std_g
        std_n = _get("group:normal", "mean_pred_std")
        std_r = _get("group:rare_or_extreme", "mean_pred_std")
        if std_n is not None:
            out["uncertainty/mean_std_normal"] = std_n
        if std_r is not None:
            out["uncertainty/mean_std_rare_extreme"] = std_r
        if std_n and std_r is not None:
            out["uncertainty/ratio_rare_normal"] = std_r / std_n
        p90_g = float(g.get("p90_pred_std", float("nan")))
        if pd.notna(p90_g):
            out["uncertainty/p90_std_global"] = p90_g
        p90_n = _get("group:normal", "p90_pred_std")
        p90_r = _get("group:rare_or_extreme", "p90_pred_std")
        if p90_n is not None:
            out["uncertainty/p90_std_normal"] = p90_n
        if p90_r is not None:
            out["uncertainty/p90_std_rare_extreme"] = p90_r
    return out
