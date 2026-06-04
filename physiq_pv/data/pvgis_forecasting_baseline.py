"""
PVGIS-only forecasting baseline (separate from the main ST-GNN training).

Task:
    input : last `seq_len` hours of PVGIS variables
    target: one PVGIS variable, `horizon` hours ahead (default pv_power_output)

Question it answers:
    Does a PVGIS-only forecast do *worse* on the rare / extreme PVGIS conditions
    found by the climatology anomaly pipeline than on normal conditions?

Scope (intentionally narrow):
  - PVGIS only. No real plant production, no Sentinel/SCADA energy.
  - Anomaly labels (from pvgis_climatology_scores.csv) are used ONLY for
    stratified evaluation, never as a supervised target.
  - Not linked to train.py / the ST-GNN. No new NetCDF written, no API download,
    no new dependencies.

Baselines:
  - `persistence`: y_hat(t+horizon) = y(t) (last observed target). No training.
  - `mlp`: small torch MLP on flattened windows (optional). Trained on the
    train years, evaluated on the test year.

Public API (small functions, easy to compose):
    load_pvgis_year(path)
    load_pvgis_years(pvgis_dir, years, ...)
    build_supervised_windows(ds, seq_len, horizon, input_variables, target_variable)
    run_persistence_baseline(test_windows)
    run_mlp_baseline(train_windows, test_windows, ...)
    load_anomaly_labels(path)
    attach_anomaly_labels(predictions, anomaly_scores)
    compute_metrics(predictions)
    write_outputs(...)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view

DEFAULT_INPUT_VARIABLES: List[str] = [
    "solar_irradiance_poa",
    "pv_power_output",
    "temperature_2m",
    "wind_speed_10m",
    "sun_height",
]
DEFAULT_TARGET_VARIABLE = "pv_power_output"

# Specific anomaly labels reported separately (climatology-level only).
SPECIFIC_ANOMALY_LABELS = [
    "unusually_low_solar_potential",
    "unusually_high_solar_potential",
    "extreme_temperature_condition",
    "extreme_wind_condition",
]
LABEL_NORMAL = "normal"
LABEL_RARE = "rare_or_extreme"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_pvgis_year(path: str) -> xr.Dataset:
    """Load a single annual PVGIS NetCDF file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PVGIS file not found: {p}")
    return xr.open_dataset(p)


def load_pvgis_years(
    pvgis_dir: str,
    years: List[int],
    file_template: str = "piedmont_pvgis_{year}.nc",
) -> Dict[int, xr.Dataset]:
    """Load several annual PVGIS files; skip missing ones with a clear message."""
    d = Path(pvgis_dir)
    if not d.is_dir():
        raise NotADirectoryError(f"PVGIS directory not found: {d}")
    datasets: Dict[int, xr.Dataset] = {}
    for year in years:
        fp = d / file_template.format(year=year)
        if not fp.exists():
            print(f"  [skip] missing PVGIS file for {year}: {fp}")
            continue
        datasets[year] = xr.open_dataset(fp)
        print(f"  [ok]   loaded PVGIS year {year}: {fp.name}")
    if not datasets:
        raise FileNotFoundError(f"No PVGIS files found in {d} for years {years}.")
    return datasets


# --------------------------------------------------------------------------- #
# Supervised windows
# --------------------------------------------------------------------------- #
def build_supervised_windows(
    ds: xr.Dataset,
    seq_len: int,
    horizon: int,
    input_variables: List[str],
    target_variable: str,
    loc_dim: str = "location",
) -> dict:
    """
    Build (N, seq_len, n_vars) input windows and the target `horizon` hours ahead.

    Returns a dict with X, y (target), last_obs (target at the last input hour,
    for persistence), location, target_time, plus the used variable lists.
    Windows containing any NaN (in X, y or last_obs) are dropped.
    """
    if target_variable not in ds:
        raise ValueError(f"Target variable '{target_variable}' not in dataset.")
    used_vars = [v for v in input_variables if v in ds]
    if not used_vars:
        raise ValueError(f"None of the input variables {input_variables} are present.")

    times = pd.DatetimeIndex(ds["time"].values)
    loc_ids = np.asarray(ds[loc_dim].values)
    n_loc, n_time = ds.sizes[loc_dim], ds.sizes["time"]
    if n_time < seq_len + horizon:
        raise ValueError(
            f"Not enough timesteps ({n_time}) for seq_len={seq_len} + horizon={horizon}."
        )

    # (n_vars, L, T) inputs and (L, T) target
    V = np.stack(
        [np.asarray(ds[v].transpose(loc_dim, "time").values, dtype=np.float32) for v in used_vars]
    )
    target = np.asarray(ds[target_variable].transpose(loc_dim, "time").values, dtype=np.float32)

    n_windows = n_time - seq_len - horizon + 1
    # sliding windows over time: (n_vars, L, n_starts, seq_len) -> keep first n_windows starts
    sw = sliding_window_view(V, window_shape=seq_len, axis=2)[:, :, :n_windows, :]
    X = sw.transpose(1, 2, 3, 0).reshape(n_loc * n_windows, seq_len, len(used_vars))

    tgt_idx0 = seq_len + horizon - 1
    y = target[:, tgt_idx0 : tgt_idx0 + n_windows].reshape(-1)
    last_obs = target[:, seq_len - 1 : seq_len - 1 + n_windows].reshape(-1)
    target_time = np.tile(times[tgt_idx0 : tgt_idx0 + n_windows].values, n_loc)
    location = np.repeat(loc_ids, n_windows)

    valid = np.isfinite(y) & np.isfinite(last_obs) & np.isfinite(X).all(axis=(1, 2))
    return {
        "X": X[valid],
        "y": y[valid],
        "last_obs": last_obs[valid],
        "location": location[valid],
        "target_time": pd.DatetimeIndex(target_time[valid]),
        "input_variables": used_vars,
        "target_variable": target_variable,
    }


def _predictions_frame(w: dict, y_pred: np.ndarray) -> pd.DataFrame:
    """Assemble the per-point predictions DataFrame (errors included)."""
    y_true = w["y"].astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    error = y_pred - y_true
    return pd.DataFrame(
        {
            "location": w["location"],
            "timestamp": w["target_time"],
            "target_variable": w["target_variable"],
            "y_true": y_true,
            "y_pred": y_pred,
            "error": error,
            "abs_error": np.abs(error),
            "squared_error": error ** 2,
        }
    )


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def run_persistence_baseline(test_windows: dict) -> pd.DataFrame:
    """y_hat(t+horizon) = y(t): predict the last observed target value."""
    return _predictions_frame(test_windows, test_windows["last_obs"])


def run_mlp_baseline(
    train_windows: dict,
    test_windows: dict,
    epochs: int = 5,
    hidden: int = 64,
    lr: float = 1e-3,
    batch_size: int = 1024,
    max_train_samples: int = 200_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Small torch MLP on flattened windows. Optional baseline; torch already a project dep."""
    import torch

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    Xtr = train_windows["X"].reshape(len(train_windows["X"]), -1).astype(np.float32)
    ytr = train_windows["y"].astype(np.float32)
    if len(Xtr) > max_train_samples:  # subsample to keep training tractable
        idx = rng.choice(len(Xtr), size=max_train_samples, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]

    # standardise inputs and target with train statistics
    x_mean, x_std = Xtr.mean(0), Xtr.std(0) + 1e-8
    y_mean, y_std = float(ytr.mean()), float(ytr.std()) + 1e-8
    Xtr_n = (Xtr - x_mean) / x_std
    ytr_n = (ytr - y_mean) / y_std

    Xte = test_windows["X"].reshape(len(test_windows["X"]), -1).astype(np.float32)
    Xte_n = (Xte - x_mean) / x_std

    model = torch.nn.Sequential(
        torch.nn.Linear(Xtr_n.shape[1], hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    Xt = torch.from_numpy(Xtr_n)
    yt = torch.from_numpy(ytr_n).unsqueeze(1)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.from_numpy(rng.permutation(n))
        running = 0.0
        for s in range(0, n, batch_size):
            b = perm[s : s + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            running += loss.item() * len(b)
        print(f"  [mlp] epoch {ep + 1}/{epochs}  train_mse(norm)={running / n:.4f}")

    model.eval()
    with torch.no_grad():
        pred_n = model(torch.from_numpy(Xte_n)).squeeze(1).numpy()
    y_pred = pred_n * y_std + y_mean
    return _predictions_frame(test_windows, y_pred)


# --------------------------------------------------------------------------- #
# Anomaly labels (stratified evaluation only)
# --------------------------------------------------------------------------- #
def load_anomaly_labels(path: Optional[str]) -> Optional[pd.DataFrame]:
    """Load (location, timestamp, label) from a pvgis_climatology_scores.csv file."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Anomaly scores file not found: {p}")
    df = pd.read_csv(p)
    missing = {"location", "timestamp", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Anomaly scores file missing columns: {sorted(missing)}")
    df = df[["location", "timestamp", "label"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def attach_anomaly_labels(
    predictions: pd.DataFrame, anomaly_scores: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Tag each prediction as `normal` or `rare_or_extreme` by matching
    (location, timestamp) against the anomaly scores. The specific labels are
    kept (comma-joined) in `anomaly_labels_detail` for per-label metrics.
    """
    out = predictions.copy()
    if anomaly_scores is None or anomaly_scores.empty:
        out["anomaly_label"] = LABEL_NORMAL
        out["anomaly_labels_detail"] = ""
        return out

    agg = (
        anomaly_scores.groupby(["location", "timestamp"])["label"]
        .agg(lambda s: ",".join(sorted(set(s))))
        .reset_index()
        .rename(columns={"label": "anomaly_labels_detail"})
    )
    agg["location"] = agg["location"].astype(out["location"].dtype)
    out = out.merge(agg, on=["location", "timestamp"], how="left")
    out["anomaly_label"] = np.where(
        out["anomaly_labels_detail"].notna(), LABEL_RARE, LABEL_NORMAL
    )
    out["anomaly_labels_detail"] = out["anomaly_labels_detail"].fillna("")
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _metric_row(stratum: str, df: pd.DataFrame) -> dict:
    return {
        "stratum": stratum,
        "count": int(len(df)),
        "MAE": float(df["abs_error"].mean()) if len(df) else float("nan"),
        "RMSE": float(np.sqrt(df["squared_error"].mean())) if len(df) else float("nan"),
    }


def compute_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (global metrics, stratified metrics) DataFrames."""
    global_df = pd.DataFrame([_metric_row("all", predictions)])

    rows = []
    for group in (LABEL_NORMAL, LABEL_RARE):
        sub = predictions[predictions["anomaly_label"] == group]
        if len(sub):
            rows.append(_metric_row(f"group:{group}", sub))
    if "anomaly_labels_detail" in predictions:
        for label in SPECIFIC_ANOMALY_LABELS:
            mask = predictions["anomaly_labels_detail"].apply(
                lambda d: label in d.split(",") if d else False
            )
            sub = predictions[mask]
            if len(sub):
                rows.append(_metric_row(f"label:{label}", sub))
    by_label_df = pd.DataFrame(rows)
    return global_df, by_label_df


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _render_report(
    global_df: pd.DataFrame, by_label_df: pd.DataFrame, meta: dict
) -> str:
    lines: List[str] = []
    lines.append("# PVGIS forecasting baseline report\n")
    lines.append(
        "PVGIS-only forecasting baseline, separate from the main ST-GNN training. "
        "It uses **only** PVGIS variables; real plant production is never used. "
        "Anomaly labels are used **only** for stratified evaluation, never as a "
        "supervised target.\n"
    )

    lines.append("## Parameters\n")
    lines.append(f"- Baseline: **{meta['baseline']}**")
    lines.append(f"- Target variable: **{meta['target_variable']}**")
    lines.append(f"- Input variables: {', '.join(meta['input_variables'])}")
    lines.append(f"- seq_len: **{meta['seq_len']}**  |  horizon: **{meta['horizon']}**")
    lines.append(f"- Train years: {meta['train_years'] or '(none — persistence needs no training)'}")
    lines.append(f"- Test year: **{meta['test_year']}**")
    lines.append(f"- Anomaly scores: {meta['anomaly_scores'] or '(none — all points treated as normal)'}")
    lines.append(f"- Predictions: **{meta['n_predictions']}**")
    lines.append(f"- Generated (UTC): {meta['generated_utc']}\n")

    lines.append("## Global metrics\n")
    g = global_df.iloc[0]
    lines.append("| stratum | count | MAE | RMSE |")
    lines.append("|---|---|---|---|")
    lines.append(f"| {g['stratum']} | {int(g['count'])} | {g['MAE']:.4f} | {g['RMSE']:.4f} |\n")

    lines.append("## Metrics by anomaly stratum\n")
    if by_label_df.empty:
        lines.append("_No strata available._\n")
    else:
        lines.append("| stratum | count | MAE | RMSE |")
        lines.append("|---|---|---|---|")
        for _, r in by_label_df.iterrows():
            lines.append(f"| {r['stratum']} | {int(r['count'])} | {r['MAE']:.4f} | {r['RMSE']:.4f} |")
        lines.append("")

    # interpretation: does the forecast degrade on rare/extreme conditions?
    lines.append("## Does the forecast degrade on rare/extreme conditions?\n")
    by = by_label_df.set_index("stratum") if not by_label_df.empty else pd.DataFrame()
    if "group:normal" in by.index and "group:rare_or_extreme" in by.index:
        mae_n = by.loc["group:normal", "MAE"]
        mae_r = by.loc["group:rare_or_extreme", "MAE"]
        ratio = mae_r / mae_n if mae_n else float("nan")
        if np.isnan(ratio):
            verdict = "inconclusive (normal MAE is zero)"
        elif ratio > 1.1:
            verdict = "**yes** — the forecast is worse on rare/extreme conditions"
        elif ratio < 0.9:
            verdict = "no — the forecast is actually better on rare/extreme conditions"
        else:
            verdict = "comparable — no clear degradation"
        lines.append(
            f"- MAE normal: {mae_n:.4f}  |  MAE rare/extreme: {mae_r:.4f}  "
            f"|  ratio: **{ratio:.2f}×**"
        )
        lines.append(f"- Verdict: {verdict}.\n")
    else:
        lines.append("_Not enough strata to compare (no rare/extreme points in the test year)._\n")
    return "\n".join(lines) + "\n"


def write_outputs(
    predictions: pd.DataFrame,
    global_df: pd.DataFrame,
    by_label_df: pd.DataFrame,
    out_dir: str,
    meta: dict,
) -> Dict[str, Path]:
    """Write predictions / metrics CSVs and a markdown report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": out / "predictions.csv",
        "metrics_global": out / "metrics_global.csv",
        "metrics_by_anomaly_label": out / "metrics_by_anomaly_label.csv",
        "report": out / "report.md",
    }
    predictions.to_csv(paths["predictions"], index=False)
    global_df.to_csv(paths["metrics_global"], index=False)
    by_label_df.to_csv(paths["metrics_by_anomaly_label"], index=False)
    paths["report"].write_text(_render_report(global_df, by_label_df, meta), encoding="utf-8")
    return paths


def build_meta(args_like: dict, n_predictions: int, input_variables: List[str]) -> dict:
    """Small helper to assemble report metadata."""
    return {
        "baseline": args_like["baseline"],
        "target_variable": args_like["target_variable"],
        "input_variables": input_variables,
        "seq_len": args_like["seq_len"],
        "horizon": args_like["horizon"],
        "train_years": args_like.get("train_years"),
        "test_year": args_like["test_year"],
        "anomaly_scores": args_like.get("anomaly_scores"),
        "n_predictions": n_predictions,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
