"""MC-Dropout inference, predictive intervals and post-hoc MC calibration."""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from physiq_pv.data.pvgis_dataset import (
    GROUP_NORMAL,
    GROUP_RARE,
    SPECIFIC_ANOMALY_LABELS,
    PVGISWindowDataset,
)
from physiq_pv.data.pvgis_labels import attach_anomaly_labels
from physiq_pv.model.st_gnn import STGNN


@torch.no_grad()
def predict(
    model: STGNN,
    dataset: PVGISWindowDataset,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    """Predict on `dataset`; return per-(location, timestamp) predictions in physical units."""
    if dataset.solar_irradiance_poa_target_all is None:
        raise ValueError(
            "Prediction dataset is missing target-time solar irradiance diagnostics."
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ei, ew = edge_index.to(device), edge_weight.to(device)
    model = model.to(device).eval()
    pv_scale = dataset.pv_scale[None, :]  # (1, N)
    loc_ids = dataset.loc_ids

    locs, times, ytrue, solar_targets, ypred = [], [], [], [], []
    for x, _y, k in loader:
        pred_norm = model(x.to(device), ei, ew, None)[1].cpu().numpy()  # (B, N)
        k = k.numpy()
        y_true = dataset.y_true_all[k]  # (B, N) physical
        solar_target = dataset.solar_irradiance_poa_target_all[k]  # (B, N) W/m2
        pred_phys = pred_norm * pv_scale  # (B, N) physical
        ts = dataset.target_time_all[k].values  # (B,)
        B, N = pred_phys.shape
        locs.append(np.tile(loc_ids, B))
        times.append(np.repeat(ts, N))
        ytrue.append(y_true.reshape(-1))
        solar_targets.append(solar_target.reshape(-1))
        ypred.append(pred_phys.reshape(-1))

    y_true = np.concatenate(ytrue).astype(np.float64)
    solar_target = np.concatenate(solar_targets).astype(np.float64)
    y_pred = np.concatenate(ypred).astype(np.float64)
    error = y_pred - y_true
    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(np.concatenate(times)),
            "location": np.concatenate(locs),
            "y_true": y_true,
            "solar_irradiance_poa_target": solar_target,
            "y_pred": y_pred,
            "error": error,
            "abs_error": np.abs(error),
            "squared_error": error ** 2,
        }
    )


# --------------------------------------------------------------------------- #
# Monte Carlo Dropout (uncertainty estimation)
# --------------------------------------------------------------------------- #
def enable_dropout_only(model: torch.nn.Module) -> int:
    """
    Put the model in eval() and reactivate *only* the dropout layers.

    This is the MC-Dropout trick: BatchNorm/LSTM/LayerNorm stay in eval mode
    (deterministic), but every nn.Dropout (and Dropout2d/Dropout3d) is switched
    back to train() so it keeps sampling masks at inference. We never call
    model.train() on the whole model. Returns the number of dropout layers
    reactivated (0 means dropout=0.0 -> no stochasticity).
    """
    n_active = 0
    for module in model.modules():
        if isinstance(
            module,
            (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d),
        ):
            module.train()
            n_active += 1
    return n_active


def active_dropout_names(model: torch.nn.Module) -> List[str]:
    """Qualified names of the nn.Dropout modules currently in train mode.

    Diagnostic companion of enable_dropout_only(): lets MC inference log WHICH
    dropout modules are stochastic (e.g. verify the stgnn_enhanced_dropout
    ablation reactivates more than gat.0.dropout)."""
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d))
        and module.training
    ]


@torch.no_grad()
def predict_mc(
    model: STGNN,
    dataset: PVGISWindowDataset,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: str,
    batch_size: int,
    mc_samples: int,
    coverage_target: float = 0.95,
    z: float = 1.96,
) -> pd.DataFrame:
    """
    Monte Carlo Dropout prediction: eval() + dropout-on + `mc_samples` passes.

    Paper-style: the predictive interval is built **directly from the MC sample
    distribution** (empirical quantiles), with no post-hoc calibration. For every
    batch we run `mc_samples` stochastic forward passes (dropout active) and
    aggregate per-(location, timestamp):
        y_pred_mean              mean over passes (physical units; used for MAE/RMSE)
        y_pred_std_raw           std over passes — diagnostic spread
                                 (`y_pred_std` kept as a back-compat alias)
        lower_pi / upper_pi      PRIMARY interval: empirical quantiles of the MC
                                 samples at alpha/2 and 1-alpha/2, alpha =
                                 1 - coverage_target (0.95 -> q0.025 / q0.975)
        lower_gaussian/upper_gaussian = mean -/+ z*std_raw (z=1.96): a DIAGNOSTIC
                                 Gaussian band only (secondary comparison).
                                 (`lower_raw`/`upper_raw`, `y_pred_lower`/
                                 `y_pred_upper` are back-compat aliases of the
                                 Gaussian band.)

    For back-compat `y_pred = y_pred_mean`. The whole model stays in eval(); only
    nn.Dropout layers are reactivated via enable_dropout_only().
    """
    if mc_samples < 2:
        raise ValueError(f"mc_samples must be >= 2 for MC Dropout, got {mc_samples}.")
    if not 0.0 < coverage_target < 1.0:
        raise ValueError(f"coverage_target must be in (0, 1), got {coverage_target}.")
    if dataset.solar_irradiance_poa_target_all is None:
        raise ValueError(
            "Prediction dataset is missing target-time solar irradiance diagnostics."
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ei, ew = edge_index.to(device), edge_weight.to(device)
    model = model.to(device).eval()
    n_active = enable_dropout_only(model)
    if n_active == 0:
        raise RuntimeError(
            "MC Dropout requested but no nn.Dropout layers are present/active "
            "(dropout=0.0?). Re-run with --dropout > 0 so there is stochasticity."
        )
    print(
        f"  [mc] eval() + dropout-only: {n_active} Dropout layer(s) reactivated, "
        f"model.training={model.training} (False = only dropout in train mode), "
        f"mc_samples={mc_samples}"
    )
    print("  [mc] dropout modules reactivated:")
    for name in active_dropout_names(model):
        print(f"    - {name}")

    pv_scale = dataset.pv_scale[None, :]  # (1, N)
    loc_ids = dataset.loc_ids

    alpha = 1.0 - coverage_target
    q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0
    print(
        f"  [mc] paper-style PI from MC samples: empirical quantiles "
        f"q{q_lo:.3f}/q{q_hi:.3f} (coverage_target={coverage_target})"
    )

    locs, times, ytrue, solar_targets = [], [], [], []
    means, stds, pis_lo, pis_hi = [], [], [], []
    for x, _y, k in loader:
        x = x.to(device)
        k = k.numpy()
        B = len(k)
        samples = np.empty((mc_samples, B, len(loc_ids)), dtype=np.float64)
        for s in range(mc_samples):
            pred_norm = model(x, ei, ew, None)[1].cpu().numpy()  # (B, N) normalised
            samples[s] = pred_norm * pv_scale                    # (B, N) physical
        mean = samples.mean(axis=0)  # (B, N)
        std = samples.std(axis=0)    # (B, N) population std over passes
        # PRIMARY interval: empirical quantiles of the MC sample distribution.
        lo_pi = np.quantile(samples, q_lo, axis=0)  # (B, N)
        hi_pi = np.quantile(samples, q_hi, axis=0)  # (B, N)
        y_true = dataset.y_true_all[k]  # (B, N) physical
        solar_target = dataset.solar_irradiance_poa_target_all[k]  # (B, N) W/m2
        ts = dataset.target_time_all[k].values  # (B,)
        N = mean.shape[1]
        locs.append(np.tile(loc_ids, B))
        times.append(np.repeat(ts, N))
        ytrue.append(y_true.reshape(-1))
        solar_targets.append(solar_target.reshape(-1))
        means.append(mean.reshape(-1))
        stds.append(std.reshape(-1))
        pis_lo.append(lo_pi.reshape(-1))
        pis_hi.append(hi_pi.reshape(-1))

    y_true = np.concatenate(ytrue).astype(np.float64)
    solar_target = np.concatenate(solar_targets).astype(np.float64)
    y_mean = np.concatenate(means).astype(np.float64)
    y_std = np.concatenate(stds).astype(np.float64)
    y_lower_pi = np.concatenate(pis_lo).astype(np.float64)
    y_upper_pi = np.concatenate(pis_hi).astype(np.float64)
    # Gaussian band: DIAGNOSTIC only (secondary comparison), not the main PI.
    y_lower_g = y_mean - z * y_std
    y_upper_g = y_mean + z * y_std
    error = y_mean - y_true  # y_pred == y_pred_mean
    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(np.concatenate(times)),
            "location": np.concatenate(locs),
            "y_true": y_true,
            "solar_irradiance_poa_target": solar_target,
            "y_pred": y_mean,
            "y_pred_mean": y_mean,
            "y_pred_std": y_std,        # back-compat alias of y_pred_std_raw
            "y_pred_std_raw": y_std,    # diagnostic MC spread
            # PRIMARY paper-style predictive interval (empirical MC quantiles).
            "lower_pi": y_lower_pi,
            "upper_pi": y_upper_pi,
            # DIAGNOSTIC Gaussian band (mean ± 1.96·std_raw). Secondary comparison
            # only — NOT the primary PI (use lower_pi/upper_pi). lower_raw/upper_raw
            # and y_pred_lower/upper are LEGACY aliases of lower_gaussian/
            # upper_gaussian, kept only for back-compat.
            "lower_gaussian": y_lower_g,
            "upper_gaussian": y_upper_g,
            "lower_raw": y_lower_g,      # legacy alias of lower_gaussian
            "upper_raw": y_upper_g,      # legacy alias of upper_gaussian
            "y_pred_lower": y_lower_g,   # legacy alias of lower_gaussian
            "y_pred_upper": y_upper_g,   # legacy alias of upper_gaussian
            "error": error,
            "abs_error": np.abs(error),
            "squared_error": error ** 2,
        }
    )


CALIBRATION_STRATEGIES = ("global", "group", "label")


def _ratio_quantile(
    df: pd.DataFrame, coverage_target: float, eps: float
) -> tuple:
    """Return (factor, n_finite) for one (sub)set of calibration predictions.

    factor is the `coverage_target` quantile of
        abs(y_true - y_pred_mean) / max(y_pred_std, eps).
    factor is None when the subset has no finite ratios.
    """
    if df.empty:
        return None, 0
    y_true = df["y_true"].to_numpy(dtype=float)
    y_mean = df["y_pred_mean"].to_numpy(dtype=float)
    y_std = df["y_pred_std"].to_numpy(dtype=float)
    ratio = np.abs(y_true - y_mean) / np.maximum(y_std, eps)
    ratio = ratio[np.isfinite(ratio)]
    if len(ratio) == 0:
        return None, 0
    return float(np.quantile(ratio, coverage_target)), int(len(ratio))


def estimate_mc_calibration_factor(
    predictions: pd.DataFrame,
    coverage_target: float = 0.95,
    eps: float = 1e-6,
) -> float:
    """
    Estimate the post-hoc MC-Dropout std scale factor on a calibration set only.

    k is the requested quantile of
        abs(y_true - y_pred_mean) / max(y_pred_std, eps)
    so intervals mean +/- k * std target the requested marginal coverage on the
    calibration distribution.
    """
    if not 0.0 < coverage_target < 1.0:
        raise ValueError(f"coverage_target must be in (0, 1), got {coverage_target}.")
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}.")
    required = {"y_true", "y_pred_mean", "y_pred_std"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Calibration predictions missing columns: {sorted(missing)}")

    factor, _n = _ratio_quantile(predictions, coverage_target, eps)
    if factor is None:
        raise ValueError("No finite calibration ratios available.")
    return factor


def estimate_mc_calibration_factors(
    predictions: pd.DataFrame,
    strategy: str = "global",
    coverage_target: float = 0.95,
    eps: float = 1e-6,
    min_samples: int = 1000,
) -> dict:
    """
    Estimate stratified MC-Dropout std scale factors on a calibration set only.

    A global factor `k_global` is always computed. Depending on `strategy`:
      * "global": only k_global.
      * "group" : also k for `group:normal` and `group:rare_or_extreme`
                  (needs `anomaly_group` on the calibration predictions).
      * "label" : the group factors plus one per specific anomaly label
                  (needs `anomaly_label`).

    A per-stratum factor is kept only when its subset has >= `min_samples` finite
    ratios; otherwise the stratum is recorded as a fallback (a group falls back to
    global; a specific label falls back to its rare/extreme group factor if that
    exists, else global). Returns a dict::

        {strategy, coverage_target, min_samples, global, factors, counts, fallbacks}

    where `factors` holds only strata that earned their own factor, so lookups can
    `.get(key, fallback)` to implement the fallback chain.
    """
    if strategy not in CALIBRATION_STRATEGIES:
        raise ValueError(
            f"Unknown calibration strategy '{strategy}'. "
            f"Available: {list(CALIBRATION_STRATEGIES)}."
        )
    if not 0.0 < coverage_target < 1.0:
        raise ValueError(f"coverage_target must be in (0, 1), got {coverage_target}.")
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}.")
    if min_samples < 1:
        raise ValueError(f"min_samples must be >= 1, got {min_samples}.")
    required = {"y_true", "y_pred_mean", "y_pred_std"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Calibration predictions missing columns: {sorted(missing)}")

    k_global, n_global = _ratio_quantile(predictions, coverage_target, eps)
    if k_global is None:
        raise ValueError("No finite calibration ratios available.")

    factors: Dict[str, float] = {}
    counts: Dict[str, int] = {"global": n_global}
    fallbacks: Dict[str, str] = {}

    if strategy in ("group", "label"):
        if "anomaly_group" not in predictions.columns:
            raise ValueError(
                "group/label calibration needs `anomaly_group` on the calibration "
                "predictions; pass --calibration-anomaly-scores."
            )
        for group in (GROUP_NORMAL, GROUP_RARE):
            key = f"group:{group}"
            sub = predictions[predictions["anomaly_group"] == group]
            k, n = _ratio_quantile(sub, coverage_target, eps)
            counts[key] = n
            if k is not None and n >= min_samples:
                factors[key] = k
            else:
                fallbacks[key] = "global"

    if strategy == "label":
        if "anomaly_label" not in predictions.columns:
            raise ValueError(
                "label calibration needs `anomaly_label` on the calibration "
                "predictions; pass --calibration-anomaly-scores."
            )
        for label in SPECIFIC_ANOMALY_LABELS:
            key = f"label:{label}"
            mask = predictions["anomaly_label"].apply(
                lambda d: label in d.split(",") if d else False
            )
            sub = predictions[mask]
            k, n = _ratio_quantile(sub, coverage_target, eps)
            counts[key] = n
            if k is not None and n >= min_samples:
                factors[key] = k
            else:
                fallbacks[key] = (
                    "group:rare_or_extreme"
                    if "group:rare_or_extreme" in factors
                    else "global"
                )

    # Raw-std diagnostics on the calibration set — a sanity check that huge
    # factors come from genuinely small std, not from near-zero/degenerate std.
    std_col = "y_pred_std_raw" if "y_pred_std_raw" in predictions.columns else "y_pred_std"

    def _std_stats(df: pd.DataFrame) -> dict:
        s = df[std_col].to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        if len(s) == 0:
            return {}
        return {
            "n": int(len(s)),
            "mean": float(np.mean(s)),
            "median": float(np.median(s)),
            "min": float(np.min(s)),
            "max": float(np.max(s)),
            "pct_below_eps": float(np.mean(s < eps)),
        }

    std_diagnostics = {"global": _std_stats(predictions)}
    if "anomaly_group" in predictions.columns:
        for group in (GROUP_NORMAL, GROUP_RARE):
            std_diagnostics[group] = _std_stats(
                predictions[predictions["anomaly_group"] == group]
            )

    return {
        "strategy": strategy,
        "coverage_target": float(coverage_target),
        "min_samples": int(min_samples),
        "global": float(k_global),
        "factors": factors,
        "counts": counts,
        "fallbacks": fallbacks,
        "std_diagnostics": std_diagnostics,
    }



def apply_mc_uncertainty_calibration(
    predictions: pd.DataFrame,
    calibration_factor: float,
) -> pd.DataFrame:
    """Add calibrated MC-Dropout intervals and raw/calibrated coverage flags."""
    required = {"y_true", "y_pred_mean", "y_pred_std", "y_pred_lower", "y_pred_upper"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Test predictions missing columns: {sorted(missing)}")
    if not np.isfinite(calibration_factor):
        raise ValueError(f"calibration_factor must be finite, got {calibration_factor}.")

    out = predictions.copy()
    y_mean = out["y_pred_mean"].to_numpy(dtype=float)
    std_col = "y_pred_std_raw" if "y_pred_std_raw" in out.columns else "y_pred_std"
    y_std = out[std_col].to_numpy(dtype=float)
    # PRIMARY calibrated interval: mean ± k·std_raw (k already absorbs the
    # quantile — no extra 1.96 factor).
    out["calibration_factor_used"] = float(calibration_factor)
    out["y_pred_std_calibrated"] = calibration_factor * y_std
    out["y_pred_lower_calibrated"] = y_mean - calibration_factor * y_std
    out["y_pred_upper_calibrated"] = y_mean + calibration_factor * y_std
    out["lower_calibrated"] = out["y_pred_lower_calibrated"]
    out["upper_calibrated"] = out["y_pred_upper_calibrated"]
    out["covered_95_raw"] = (
        (out["y_true"] >= out["y_pred_lower"]) & (out["y_true"] <= out["y_pred_upper"])
    )
    out["covered_95_calibrated"] = (
        (out["y_true"] >= out["y_pred_lower_calibrated"])
        & (out["y_true"] <= out["y_pred_upper_calibrated"])
    )
    return out


def apply_mc_uncertainty_calibration_stratified(
    predictions: pd.DataFrame,
    calibration: dict,
) -> pd.DataFrame:
    """
    Apply per-stratum MC-Dropout calibration using a factor map from
    `estimate_mc_calibration_factors`.

    Each test row gets a `calibration_factor_used`:
      * strategy "global": k_global for every row;
      * strategy "group" : the row's `anomaly_group` factor, else k_global;
      * strategy "label" : the highest-priority specific label factor present on
                           the row, else its group factor, else k_global.
    Then adds calibrated bounds and raw/calibrated coverage flags. Needs anomaly
    labels already attached (`attach_anomaly_labels`).
    """
    required = {
        "y_true", "y_pred_mean", "y_pred_std", "y_pred_lower", "y_pred_upper",
        "anomaly_group", "anomaly_label",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Test predictions missing columns: {sorted(missing)}")

    strategy = calibration["strategy"]
    factors = calibration["factors"]
    k_global = calibration["global"]
    if not np.isfinite(k_global):
        raise ValueError(f"Global calibration factor must be finite, got {k_global}.")

    out = predictions.copy()
    factor_used = np.full(len(out), float(k_global), dtype=float)

    if strategy in ("group", "label"):
        group = out["anomaly_group"].to_numpy()
        for key, k in factors.items():
            if key.startswith("group:"):
                factor_used[group == key.split(":", 1)[1]] = k
    if strategy == "label":
        labels = out["anomaly_label"].fillna("").to_numpy()
        # Apply lowest-priority first so the first label in SPECIFIC_ANOMALY_LABELS
        # wins when a row carries several labels.
        for label in reversed(SPECIFIC_ANOMALY_LABELS):
            key = f"label:{label}"
            if key not in factors:
                continue
            mask = np.array(
                [label in (d.split(",") if d else []) for d in labels], dtype=bool
            )
            factor_used[mask] = factors[key]

    y_mean = out["y_pred_mean"].to_numpy(dtype=float)
    std_col = "y_pred_std_raw" if "y_pred_std_raw" in out.columns else "y_pred_std"
    y_std = out[std_col].to_numpy(dtype=float)
    out["calibration_factor_used"] = factor_used
    # PRIMARY calibrated interval: mean ± k·std_raw. k (calibration_factor_used)
    # is already the coverage_target quantile of |y_true-mean|/std_raw, so it
    # absorbs the quantile — DO NOT multiply by 1.96 again.
    out["y_pred_std_calibrated"] = factor_used * y_std
    out["y_pred_lower_calibrated"] = y_mean - factor_used * y_std
    out["y_pred_upper_calibrated"] = y_mean + factor_used * y_std
    out["lower_calibrated"] = out["y_pred_lower_calibrated"]
    out["upper_calibrated"] = out["y_pred_upper_calibrated"]
    out["covered_95_raw"] = (
        (out["y_true"] >= out["y_pred_lower"]) & (out["y_true"] <= out["y_pred_upper"])
    )
    out["covered_95_calibrated"] = (
        (out["y_true"] >= out["y_pred_lower_calibrated"])
        & (out["y_true"] <= out["y_pred_upper_calibrated"])
    )
    return out
