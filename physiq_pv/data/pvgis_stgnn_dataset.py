"""
PVGIS-only ST-GNN forecasting dataset + thin train/eval helpers.

Reuses the existing `STGNN` architecture (instantiated with `n_features=11`),
but with a *real* PVGIS-only input: NO ENERGIA, NO observed plant production,
NO quality score (compute_qs), NO kWp/UPN/load_kwp, NO plant-quality filters.
The m1..m5 QS channels and the real `pv_lag` are removed entirely (not
neutralised). `pv_lag_pvgis` is the lag of PVGIS `pv_power_output`.

Task: PVGIS past -> PVGIS future
    input : last `seq_len` hours of PVGIS-only features (per location node)
    target: PVGIS `pv_power_output`, `horizon` hours ahead

Nodes = PVGIS locations. Anomaly labels (from the climatology pipeline) are used
ONLY for stratified evaluation, never as input nor as a supervised target.

Feature set (11, all derivable from PVGIS + solar geometry):
    temperature_2m, solar_irradiance_poa, wind_speed_10m,   (meteo, z-scored)
    sin_elev, cos_elev,                                      (solar geometry)
    kt, kt_std_3h, dghi_dt, dni_norm, dhi_norm,              (derived irradiance)
    pv_lag_pvgis                                             (lag of PVGIS pv, normalised)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pvlib
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from physiq_pv.model.st_gnn import STGNN

PVGIS_STGNN_FEATURES: List[str] = [
    "temperature_2m",
    "solar_irradiance_poa",
    "wind_speed_10m",
    "sin_elev",
    "cos_elev",
    "kt",
    "kt_std_3h",
    "dghi_dt",
    "dni_norm",
    "dhi_norm",
    "pv_lag_pvgis",
]
N_FEATURES = len(PVGIS_STGNN_FEATURES)
DEFAULT_TARGET_VARIABLE = "pv_power_output"
_REQUIRED_VARS = ["temperature_2m", "solar_irradiance_poa", "wind_speed_10m"]

# --------------------------------------------------------------------------- #
# Feature-set ablation
# --------------------------------------------------------------------------- #
# Each set is a subset of PVGIS_STGNN_FEATURES. resolve_feature_set() always
# returns the selected names in the canonical PVGIS_STGNN_FEATURES order so the
# channel subsetting in build_datasets stays consistent. n_features is then
# len(selected) — never hardcode 11 when a feature set is in play.
FEATURE_SETS: Dict[str, List[str]] = {
    "full": list(PVGIS_STGNN_FEATURES),
    "no_pv_lag": [f for f in PVGIS_STGNN_FEATURES if f != "pv_lag_pvgis"],
    "meteo_only": [
        "temperature_2m", "solar_irradiance_poa", "wind_speed_10m",
        "sin_elev", "cos_elev",
    ],
    "irradiance_only": [
        "solar_irradiance_poa", "sin_elev", "cos_elev",
        "kt", "kt_std_3h", "dghi_dt", "dni_norm", "dhi_norm",
    ],
    "no_derived_irradiance": [
        "temperature_2m", "solar_irradiance_poa", "wind_speed_10m",
        "sin_elev", "cos_elev", "pv_lag_pvgis",
    ],
}


def resolve_feature_set(name: str) -> List[str]:
    """Return the selected feature names in canonical PVGIS_STGNN_FEATURES order."""
    if name not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature_set '{name}'. Available: {sorted(FEATURE_SETS)}."
        )
    selected = set(FEATURE_SETS[name])
    return [f for f in PVGIS_STGNN_FEATURES if f in selected]

SPECIFIC_ANOMALY_LABELS = [
    "unusually_low_solar_potential",
    "unusually_high_solar_potential",
    "extreme_temperature_condition",
    "extreme_wind_condition",
]
GROUP_NORMAL = "normal"
GROUP_RARE = "rare_or_extreme"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_pvgis_year(path: str) -> xr.Dataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PVGIS file not found: {p}")
    return xr.open_dataset(p)


def load_pvgis_years(
    pvgis_dir: str, years: List[int], file_template: str = "piedmont_pvgis_{year}.nc"
) -> Dict[int, xr.Dataset]:
    d = Path(pvgis_dir)
    if not d.is_dir():
        raise NotADirectoryError(f"PVGIS directory not found: {d}")
    out: Dict[int, xr.Dataset] = {}
    for year in years:
        fp = d / file_template.format(year=year)
        if not fp.exists():
            print(f"  [skip] missing PVGIS file for {year}: {fp}")
            continue
        out[year] = xr.open_dataset(fp)
        print(f"  [ok]   loaded PVGIS year {year}: {fp.name}")
    if not out:
        raise FileNotFoundError(f"No PVGIS files found in {d} for years {years}.")
    return out


# --------------------------------------------------------------------------- #
# Solar geometry + raw channels (PVGIS-only)
# --------------------------------------------------------------------------- #
def _solar_geometry(times: pd.DatetimeIndex, lats: np.ndarray, lons: np.ndarray) -> tuple:
    """sin/cos apparent elevation, clear-sky GHI [kW/m^2], apparent zenith [deg]."""
    T, N = len(times), len(lats)
    times_utc = times.tz_localize("UTC") if times.tzinfo is None else times
    sin_elev = np.zeros((T, N), dtype=np.float32)
    cos_elev = np.zeros((T, N), dtype=np.float32)
    ghi_cs = np.zeros((T, N), dtype=np.float32)
    zenith = np.zeros((T, N), dtype=np.float32)
    fleet_lat, fleet_lon = float(np.nanmean(lats)), float(np.nanmean(lons))
    for p in range(N):
        lat_p = float(lats[p]) if np.isfinite(lats[p]) else fleet_lat
        lon_p = float(lons[p]) if np.isfinite(lons[p]) else fleet_lon
        loc = pvlib.location.Location(lat_p, lon_p, tz="UTC")
        sp = loc.get_solarposition(times_utc)
        elev = np.clip(sp["apparent_elevation"].values, 0.0, 90.0).astype(np.float32)
        sin_elev[:, p] = np.sin(np.radians(elev))
        cos_elev[:, p] = np.cos(np.radians(elev))
        zenith[:, p] = np.clip(sp["apparent_zenith"].values, 0.0, 90.0).astype(np.float32)
        try:
            cs = loc.get_clearsky(times_utc, model="ineichen")
        except Exception:
            cs = loc.get_clearsky(times_utc, model="simplified_solis")
        ghi_cs[:, p] = np.clip(np.nan_to_num(cs["ghi"].values, nan=0.0) / 1000.0, 0.0, None)
    return sin_elev, cos_elev, ghi_cs, zenith


def build_year_raw(
    ds: xr.Dataset, target_variable: str, loc_dim: str = "location"
) -> dict:
    """Build raw (un-normalised) per-(time, location) channels for one PVGIS year."""
    for v in _REQUIRED_VARS + [target_variable]:
        if v not in ds:
            raise ValueError(f"Required PVGIS variable '{v}' missing from dataset.")

    times = pd.DatetimeIndex(ds["time"].values)
    lats = np.asarray(ds["lat"].values, dtype=float)
    lons = np.asarray(ds["lon"].values, dtype=float)

    def col(v):  # (T, N)
        return np.asarray(ds[v].transpose(loc_dim, "time").values, dtype=np.float32).T

    temp = col("temperature_2m")
    solar_wm2 = col("solar_irradiance_poa")
    wind = col("wind_speed_10m")
    pv = col(target_variable)
    solar_kwm2 = np.clip(solar_wm2 / 1000.0, 0.0, None)

    sin_elev, cos_elev, ghi_cs, zenith = _solar_geometry(times, lats, lons)
    day = (sin_elev > 0.05) & (solar_kwm2 > 0.03)

    kt = np.where(ghi_cs > 0.1, solar_kwm2 / (ghi_cs + 1e-6), 0.0)
    kt = np.clip(kt, 0.0, 1.5).astype(np.float32)

    kt_std = np.zeros_like(kt)
    for p in range(kt.shape[1]):
        kt_std[:, p] = (
            pd.Series(kt[:, p]).rolling(3, min_periods=1).std().fillna(0.0).to_numpy()
        )

    dghi = np.zeros_like(solar_kwm2)
    dghi[1:, :] = solar_kwm2[1:, :] - solar_kwm2[:-1, :]

    doy = times.dayofyear.to_numpy()
    dni = np.zeros_like(solar_kwm2)
    dhi = np.zeros_like(solar_kwm2)
    for p in range(solar_kwm2.shape[1]):
        erbs = pvlib.irradiance.erbs(
            ghi=solar_kwm2[:, p] * 1000.0, zenith=zenith[:, p], datetime_or_doy=doy
        )
        dni[:, p] = np.nan_to_num(erbs["dni"], nan=0.0) / 1000.0
        dhi[:, p] = np.nan_to_num(erbs["dhi"], nan=0.0) / 1000.0
    dni = np.clip(dni, 0.0, 1.5).astype(np.float32)
    dhi = np.clip(dhi, 0.0, 1.0).astype(np.float32)

    return {
        "times": times,
        "lats": lats,
        "lons": lons,
        "temp": temp,
        "solar_wm2": solar_wm2,
        "wind": wind,
        "sin": sin_elev,
        "cos": cos_elev,
        "kt": kt,
        "kt_std": kt_std,
        "dghi": dghi,
        "dni": dni,
        "dhi": dhi,
        "pv": pv,
        "day": day,
    }


def fit_normalization(train_raws: List[dict]) -> dict:
    """Per-location pv_scale (p99 daytime pv) and global z-score stats from TRAIN only."""
    n_loc = train_raws[0]["pv"].shape[1]
    # per-location p99 of daytime, positive pv (target scale)
    pv_stack = np.concatenate([r["pv"] for r in train_raws], axis=0)  # (sum_T, N)
    day_stack = np.concatenate([r["day"] for r in train_raws], axis=0)
    pv_scale = np.ones(n_loc, dtype=np.float64)
    for p in range(n_loc):
        vals = pv_stack[day_stack[:, p], p]
        vals = vals[vals > 0]
        if len(vals) > 10:
            pv_scale[p] = float(np.percentile(vals, 99)) + 1e-6

    def gstats(key):
        arr = np.concatenate([r[key] for r in train_raws], axis=0)
        return float(np.nanmean(arr)), float(np.nanstd(arr) + 1e-6)

    z = {k: gstats(k) for k in ("temp", "solar_wm2", "wind", "dghi")}
    return {"pv_scale": pv_scale, "z": z}


def assemble_feats(raw: dict, norm: dict) -> tuple:
    """Return (feats (T,N,11), pv_norm (T,N), pv_raw (T,N)) using train normalisation."""
    z = norm["z"]
    pv_scale = norm["pv_scale"][None, :]

    def zc(key):
        mu, sd = z[key]
        return (raw[key] - mu) / sd

    pv_norm = np.clip(raw["pv"] / pv_scale, 0.0, 1.5).astype(np.float32)
    channels = [
        zc("temp"),
        zc("solar_wm2"),
        zc("wind"),
        raw["sin"],
        raw["cos"],
        raw["kt"],
        raw["kt_std"],
        zc("dghi"),
        raw["dni"],
        raw["dhi"],
        pv_norm,  # pv_lag_pvgis (causal lag once sliced inside the window)
    ]
    feats = np.stack(channels, axis=-1).astype(np.float32)  # (T, N, 11)
    return feats, pv_norm, raw["pv"].astype(np.float32)


# --------------------------------------------------------------------------- #
# Window dataset
# --------------------------------------------------------------------------- #
class PVGISWindowDataset(Dataset):
    """
    Sliding windows over one or more PVGIS years (no cross-year windows).

    __getitem__ -> (x, y_norm, k) with
        x      (N, seq_len, n_features)
        y_norm (N,)                       normalised pv target at t+horizon
        k      int                        global sample index (for eval lookups)
    """

    def __init__(
        self,
        feats_by_year: Dict[int, np.ndarray],
        pvnorm_by_year: Dict[int, np.ndarray],
        pvraw_by_year: Dict[int, np.ndarray],
        times_by_year: Dict[int, pd.DatetimeIndex],
        seq_len: int,
        horizon: int,
        pv_scale: np.ndarray,
        loc_ids: np.ndarray,
    ):
        self.feats_by_year = feats_by_year
        self.seq_len = seq_len
        self.horizon = horizon
        self.pv_scale = pv_scale.astype(np.float32)
        self.loc_ids = loc_ids
        self.n_nodes = len(loc_ids)

        samples: List[tuple] = []
        y_norm_rows: List[np.ndarray] = []
        y_true_rows: List[np.ndarray] = []
        times_rows: List[np.datetime64] = []
        for year, feats in feats_by_year.items():
            T = feats.shape[0]
            n_windows = T - seq_len - horizon + 1
            if n_windows <= 0:
                continue
            tgt = seq_len + horizon - 1
            pvn = pvnorm_by_year[year]
            pvr = pvraw_by_year[year]
            ts = times_by_year[year]
            for i in range(n_windows):
                samples.append((year, i))
                y_norm_rows.append(pvn[i + tgt])
                y_true_rows.append(pvr[i + tgt])
                times_rows.append(ts.values[i + tgt])
        if not samples:
            raise ValueError("No supervised windows could be built (year too short?).")

        self.samples = samples
        self.y_norm_all = np.stack(y_norm_rows)  # (n_samples, N)
        self.y_true_all = np.stack(y_true_rows)  # (n_samples, N)
        self.target_time_all = pd.DatetimeIndex(times_rows)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, k: int):
        year, i = self.samples[k]
        win = self.feats_by_year[year][i : i + self.seq_len]  # (seq_len, N, C)
        x = torch.from_numpy(np.ascontiguousarray(win.transpose(1, 0, 2)))  # (N, seq_len, C)
        y = torch.from_numpy(self.y_norm_all[k])  # (N,)
        return x, y, k

    def subsample(self, max_samples: Optional[int], seed: int = 0) -> "PVGISWindowDataset":
        """Randomly keep at most `max_samples` windows (in place); returns self."""
        if max_samples is None or len(self.samples) <= max_samples:
            return self
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(self.samples), size=max_samples, replace=False))
        self.samples = [self.samples[i] for i in keep]
        self.y_norm_all = self.y_norm_all[keep]
        self.y_true_all = self.y_true_all[keep]
        self.target_time_all = self.target_time_all[keep]
        return self


def build_datasets(
    train_ds_map: Dict[int, xr.Dataset],
    test_ds: xr.Dataset,
    seq_len: int,
    horizon: int,
    target_variable: str = DEFAULT_TARGET_VARIABLE,
    feature_names: Optional[List[str]] = None,
    calibration_ds_map: Optional[Dict[int, xr.Dataset]] = None,
) -> dict:
    """
    Build train/test/(optional) calibration window datasets with train-fitted normalisation.

    `feature_names` selects a subset of PVGIS_STGNN_FEATURES (feature-set
    ablation). When None, all 11 features are used. The returned `n_features`
    reflects the selection, so STGNN can be instantiated with the right size.
    """
    selected = list(feature_names) if feature_names is not None else list(PVGIS_STGNN_FEATURES)
    unknown = [f for f in selected if f not in PVGIS_STGNN_FEATURES]
    if unknown:
        raise ValueError(f"Unknown feature(s) {unknown}; valid: {PVGIS_STGNN_FEATURES}.")
    if not selected:
        raise ValueError("feature_names selected an empty feature set.")
    keep_idx = [PVGIS_STGNN_FEATURES.index(f) for f in selected]

    calibration_ds_map = calibration_ds_map or {}

    train_raws = {y: build_year_raw(ds, target_variable) for y, ds in train_ds_map.items()}
    calibration_raws = {
        y: build_year_raw(ds, target_variable) for y, ds in calibration_ds_map.items()
    }
    test_raw = build_year_raw(test_ds, target_variable)

    n_loc = test_raw["pv"].shape[1]
    for y, r in train_raws.items():
        if r["pv"].shape[1] != n_loc:
            raise ValueError(
                f"Location count mismatch: train year {y} has {r['pv'].shape[1]}, "
                f"test has {n_loc}. PVGIS node set must be consistent."
            )
    for y, r in calibration_raws.items():
        if r["pv"].shape[1] != n_loc:
            raise ValueError(
                f"Location count mismatch: calibration year {y} has {r['pv'].shape[1]}, "
                f"test has {n_loc}. PVGIS node set must be consistent."
            )

    norm = fit_normalization(list(train_raws.values()))

    def _select(feats):  # (T, N, 11) -> (T, N, len(selected))
        return np.ascontiguousarray(feats[:, :, keep_idx])

    feats_tr, pvn_tr, pvr_tr, times_tr = {}, {}, {}, {}
    for y, r in train_raws.items():
        f, pn, pr = assemble_feats(r, norm)
        feats_tr[y], pvn_tr[y], pvr_tr[y], times_tr[y] = _select(f), pn, pr, r["times"]

    feats_cal, pvn_cal, pvr_cal, times_cal = {}, {}, {}, {}
    for y, r in calibration_raws.items():
        f, pn, pr = assemble_feats(r, norm)
        feats_cal[y], pvn_cal[y], pvr_cal[y], times_cal[y] = _select(f), pn, pr, r["times"]

    f, pn, pr = assemble_feats(test_raw, norm)
    feats_te = {-1: _select(f)}
    pvn_te = {-1: pn}
    pvr_te = {-1: pr}
    times_te = {-1: test_raw["times"]}

    loc_ids = np.asarray(test_ds["location"].values)
    train_dataset = PVGISWindowDataset(
        feats_tr, pvn_tr, pvr_tr, times_tr, seq_len, horizon, norm["pv_scale"], loc_ids
    )
    test_dataset = PVGISWindowDataset(
        feats_te, pvn_te, pvr_te, times_te, seq_len, horizon, norm["pv_scale"], loc_ids
    )
    calibration_dataset = None
    if calibration_raws:
        calibration_dataset = PVGISWindowDataset(
            feats_cal, pvn_cal, pvr_cal, times_cal,
            seq_len, horizon, norm["pv_scale"], loc_ids,
        )
    return {
        "train": train_dataset,
        "test": test_dataset,
        "calibration": calibration_dataset,
        "loc_ids": loc_ids,
        "lats": test_raw["lats"],
        "lons": test_raw["lons"],
        "n_features": len(selected),
        "features": selected,
        "pv_scale": norm["pv_scale"],
    }


# --------------------------------------------------------------------------- #
# Model reuse + train / predict
# --------------------------------------------------------------------------- #
def make_model(
    n_nodes: int, seq_len: int, n_features: int = N_FEATURES, dropout: float = 0.2
) -> STGNN:
    """Instantiate the existing STGNN with the PVGIS-only feature count.

    `dropout` is exposed so future MC-Dropout experiments can keep dropout layers
    active at inference; it does not change the deterministic eval path here.
    """
    return STGNN(
        n_nodes=n_nodes,
        n_features=n_features,
        seq_len=seq_len,
        patch_len=4 if seq_len > 1 else 1,
        stride=2 if seq_len > 1 else 1,
        d_model=128,
        gat_dim=96,
        gat_heads=4,
        gat_layers=1,
        dropout=dropout,
        use_patchtst=True,
        use_gat=True,
        bilstm_pooling="attn",
    )


def train_model(
    model: STGNN,
    dataset: PVGISWindowDataset,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> STGNN:
    """Train pred_pv against the normalised PVGIS pv target (MSE). Deterministic, no QS."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    ei, ew = edge_index.to(device), edge_weight.to(device)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    for ep in range(epochs):
        model.train()
        losses = []
        for x, y, _k in loader:
            x, y = x.to(device), y.to(device)
            _pred_ghi, pred_pv = model(x, ei, ew, None)  # ghi_cs=None -> pred_pv only
            loss = loss_fn(pred_pv, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"  [stgnn] epoch {ep + 1}/{epochs}  train_mse(norm)={np.mean(losses):.5f}")
    return model


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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ei, ew = edge_index.to(device), edge_weight.to(device)
    model = model.to(device).eval()
    pv_scale = dataset.pv_scale[None, :]  # (1, N)
    loc_ids = dataset.loc_ids

    locs, times, ytrue, ypred = [], [], [], []
    for x, _y, k in loader:
        pred_norm = model(x.to(device), ei, ew, None)[1].cpu().numpy()  # (B, N)
        k = k.numpy()
        y_true = dataset.y_true_all[k]  # (B, N) physical
        pred_phys = pred_norm * pv_scale  # (B, N) physical
        ts = dataset.target_time_all[k].values  # (B,)
        B, N = pred_phys.shape
        locs.append(np.tile(loc_ids, B))
        times.append(np.repeat(ts, N))
        ytrue.append(y_true.reshape(-1))
        ypred.append(pred_phys.reshape(-1))

    y_true = np.concatenate(ytrue).astype(np.float64)
    y_pred = np.concatenate(ypred).astype(np.float64)
    error = y_pred - y_true
    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(np.concatenate(times)),
            "location": np.concatenate(locs),
            "y_true": y_true,
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


@torch.no_grad()
def predict_mc(
    model: STGNN,
    dataset: PVGISWindowDataset,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: str,
    batch_size: int,
    mc_samples: int,
    z: float = 1.96,
) -> pd.DataFrame:
    """
    Monte Carlo Dropout prediction: eval() + dropout-on + `mc_samples` passes.

    For every batch we run `mc_samples` stochastic forward passes (dropout active)
    and aggregate per-(location, timestamp):
        y_pred_mean  mean over passes (physical units; used for MAE/RMSE)
        y_pred_std   std  over passes (uncertainty measure)
        y_pred_lower/upper = mean -/+ z * std   (z=1.96 -> ~95% band)

    For back-compat `y_pred = y_pred_mean`. The whole model stays in eval(); only
    nn.Dropout layers are reactivated via enable_dropout_only().
    """
    if mc_samples < 2:
        raise ValueError(f"mc_samples must be >= 2 for MC Dropout, got {mc_samples}.")
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

    pv_scale = dataset.pv_scale[None, :]  # (1, N)
    loc_ids = dataset.loc_ids

    locs, times, ytrue, means, stds = [], [], [], [], []
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
        y_true = dataset.y_true_all[k]  # (B, N) physical
        ts = dataset.target_time_all[k].values  # (B,)
        N = mean.shape[1]
        locs.append(np.tile(loc_ids, B))
        times.append(np.repeat(ts, N))
        ytrue.append(y_true.reshape(-1))
        means.append(mean.reshape(-1))
        stds.append(std.reshape(-1))

    y_true = np.concatenate(ytrue).astype(np.float64)
    y_mean = np.concatenate(means).astype(np.float64)
    y_std = np.concatenate(stds).astype(np.float64)
    y_lower = y_mean - z * y_std
    y_upper = y_mean + z * y_std
    error = y_mean - y_true  # y_pred == y_pred_mean
    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(np.concatenate(times)),
            "location": np.concatenate(locs),
            "y_true": y_true,
            "y_pred": y_mean,
            "y_pred_mean": y_mean,
            "y_pred_std": y_std,
            "y_pred_lower": y_lower,
            "y_pred_upper": y_upper,
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

    return {
        "strategy": strategy,
        "coverage_target": float(coverage_target),
        "min_samples": int(min_samples),
        "global": float(k_global),
        "factors": factors,
        "counts": counts,
        "fallbacks": fallbacks,
    }


def _factor_for_stratum(stratum: str, calibration: Optional[dict]) -> Optional[float]:
    """Resolve the calibration factor that applies to a metrics stratum row."""
    if calibration is None:
        return None
    factors = calibration["factors"]
    k_global = calibration["global"]
    if stratum == "all":
        return k_global
    if stratum in factors:
        return factors[stratum]
    if stratum.startswith("label:"):
        return factors.get("group:rare_or_extreme", k_global)
    return k_global


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
    y_std = out["y_pred_std"].to_numpy(dtype=float)
    out["y_pred_lower_calibrated"] = y_mean - calibration_factor * y_std
    out["y_pred_upper_calibrated"] = y_mean + calibration_factor * y_std
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
    y_std = out["y_pred_std"].to_numpy(dtype=float)
    out["calibration_factor_used"] = factor_used
    out["y_pred_lower_calibrated"] = y_mean - factor_used * y_std
    out["y_pred_upper_calibrated"] = y_mean + factor_used * y_std
    out["covered_95_raw"] = (
        (out["y_true"] >= out["y_pred_lower"]) & (out["y_true"] <= out["y_pred_upper"])
    )
    out["covered_95_calibrated"] = (
        (out["y_true"] >= out["y_pred_lower_calibrated"])
        & (out["y_true"] <= out["y_pred_upper_calibrated"])
    )
    return out


# --------------------------------------------------------------------------- #
# Anomaly labels (stratified evaluation only)
# --------------------------------------------------------------------------- #
def load_anomaly_labels(path: Optional[str]) -> Optional[pd.DataFrame]:
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
    """Add `anomaly_group` (normal / rare_or_extreme) and `anomaly_label` (specific)."""
    out = predictions.copy()
    if anomaly_scores is None or anomaly_scores.empty:
        out["anomaly_group"] = GROUP_NORMAL
        out["anomaly_label"] = ""
        return out
    agg = (
        anomaly_scores.groupby(["location", "timestamp"])["label"]
        .agg(lambda s: ",".join(sorted(set(s))))
        .reset_index()
        .rename(columns={"label": "anomaly_label"})
    )
    agg["location"] = agg["location"].astype(out["location"].dtype)
    out = out.merge(agg, on=["location", "timestamp"], how="left")
    out["anomaly_group"] = np.where(out["anomaly_label"].notna(), GROUP_RARE, GROUP_NORMAL)
    out["anomaly_label"] = out["anomaly_label"].fillna("")
    return out


# --------------------------------------------------------------------------- #
# Metrics + output
# --------------------------------------------------------------------------- #
def _metric_row(
    stratum: str,
    df: pd.DataFrame,
    calibration_factor: Optional[float] = None,
    calibration_strategy: Optional[str] = None,
) -> dict:
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
        if {"y_pred_lower_calibrated", "y_pred_upper_calibrated"} <= set(df.columns):
            inside_cal = (
                (df["y_true"] >= df["y_pred_lower_calibrated"])
                & (df["y_true"] <= df["y_pred_upper_calibrated"])
            )
            row["coverage_95_calibrated"] = float(inside_cal.mean())
        else:
            row["coverage_95_calibrated"] = float("nan")
    else:
        row["mean_pred_std"] = float("nan")
        row["median_pred_std"] = float("nan")
        row["p90_pred_std"] = float("nan")
        row["coverage_95"] = float("nan")
        row["coverage_95_raw"] = float("nan")
        row["coverage_95_calibrated"] = float("nan")
    row["calibration_factor"] = (
        float(calibration_factor)
        if calibration_factor is not None and np.isfinite(calibration_factor)
        else float("nan")
    )
    row["calibration_strategy"] = calibration_strategy or ""
    return row


def compute_metrics(
    predictions: pd.DataFrame, calibration: Optional[dict] = None
) -> tuple:
    """Global + per-stratum metrics. `calibration` is the factor map from
    `estimate_mc_calibration_factors`; each stratum reports the factor that was
    actually applied to it (with the group/global fallback resolved)."""
    strategy = calibration["strategy"] if calibration else None

    def _row(stratum: str, df: pd.DataFrame) -> dict:
        return _metric_row(stratum, df, _factor_for_stratum(stratum, calibration), strategy)

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
    global_df: pd.DataFrame, by_df: pd.DataFrame, mc_dropout: bool = False
) -> dict:
    """
    Flatten global + by-stratum metrics into namespaced W&B scalars.

    Keys: mae/global, rmse/global, mae|rmse/{normal,rare_extreme},
    ratio/{mae,rmse}_rare_normal, and (when mc_dropout) uncertainty/* and
    coverage_95/*. Only keys with finite values are emitted.
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
        cov_g = float(g.get("coverage_95", float("nan")))
        if pd.notna(cov_g):
            out["coverage_95/global"] = cov_g
        cov_raw_g = float(g.get("coverage_95_raw", cov_g))
        if pd.notna(cov_raw_g):
            out["coverage_95_raw/global"] = cov_raw_g
        cov_cal_g = float(g.get("coverage_95_calibrated", float("nan")))
        if pd.notna(cov_cal_g):
            out["coverage_95_calibrated/global"] = cov_cal_g
        cov_n = _get("group:normal", "coverage_95")
        cov_r = _get("group:rare_or_extreme", "coverage_95")
        if cov_n is not None:
            out["coverage_95/normal"] = cov_n
        if cov_r is not None:
            out["coverage_95/rare_extreme"] = cov_r
        cov_raw_n = _get("group:normal", "coverage_95_raw")
        cov_raw_r = _get("group:rare_or_extreme", "coverage_95_raw")
        if cov_raw_n is not None:
            out["coverage_95_raw/normal"] = cov_raw_n
        if cov_raw_r is not None:
            out["coverage_95_raw/rare_extreme"] = cov_raw_r
        cov_cal_n = _get("group:normal", "coverage_95_calibrated")
        cov_cal_r = _get("group:rare_or_extreme", "coverage_95_calibrated")
        if cov_cal_n is not None:
            out["coverage_95_calibrated/normal"] = cov_cal_n
        if cov_cal_r is not None:
            out["coverage_95_calibrated/rare_extreme"] = cov_cal_r
        cal_factor = float(g.get("calibration_factor", float("nan")))
        if pd.notna(cal_factor):
            out["uncertainty/calibration_factor"] = cal_factor
    return out


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
    lines.append(f"- Selected features ({meta['n_features']}): {', '.join(meta['features'])}")
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
    lines.append(f"- Device: {meta['device']}  |  Generated (UTC): {meta['generated_utc']}\n")

    def _fmt(v, nd=4):
        return f"{float(v):.{nd}f}" if v is not None and pd.notna(v) else "—"

    def _by_value(by_index: pd.DataFrame, stratum: str, col: str):
        if not by_index.empty and stratum in by_index.index and col in by_index.columns:
            return by_index.loc[stratum, col]
        return float("nan")

    g = global_df.iloc[0]
    lines.append("## Global metrics\n")
    if mc:
        lines.append("| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        lines.append(
            f"| {g['stratum']} | {int(g['count'])} | {_fmt(g['MAE'])} | {_fmt(g['RMSE'])} | "
            f"{_fmt(g['mean_pred_std'])} | {_fmt(g['median_pred_std'])} | {_fmt(g['p90_pred_std'])} | "
            f"{_fmt(g['coverage_95_raw'], 3)} | {_fmt(g['coverage_95_calibrated'], 3)} |\n"
        )
    else:
        lines.append("| stratum | count | MAE | RMSE |")
        lines.append("|---|---|---|---|")
        lines.append(f"| {g['stratum']} | {int(g['count'])} | {g['MAE']:.4f} | {g['RMSE']:.4f} |\n")

    lines.append("## Metrics by anomaly stratum\n")
    if by_df.empty:
        lines.append("_No strata available._\n")
    elif mc:
        lines.append("| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in by_df.iterrows():
            lines.append(
                f"| {r['stratum']} | {int(r['count'])} | {_fmt(r['MAE'])} | {_fmt(r['RMSE'])} | "
                f"{_fmt(r['mean_pred_std'])} | {_fmt(r['median_pred_std'])} | {_fmt(r['p90_pred_std'])} | "
                f"{_fmt(r['coverage_95_raw'], 3)} | {_fmt(r['coverage_95_calibrated'], 3)} |"
            )
        lines.append("")
    else:
        lines.append("| stratum | count | MAE | RMSE |")
        lines.append("|---|---|---|---|")
        for _, r in by_df.iterrows():
            lines.append(f"| {r['stratum']} | {int(r['count'])} | {r['MAE']:.4f} | {r['RMSE']:.4f} |")
        lines.append("")

    lines.append("## Does ST-GNN degrade on rare/extreme PVGIS conditions?\n")
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
        lines.append("## Uncertainty calibration\n")
        by = by_df.set_index("stratum") if not by_df.empty else pd.DataFrame()
        cal_factor = meta.get("calibration_factor")
        has_calibration = cal_factor is not None and pd.notna(cal_factor)
        lines.append(
            "MC Dropout std is a relative uncertainty measure. Raw intervals "
            "`mean ± 1.96 std` are not guaranteed to be calibrated. Post-hoc "
            "calibration scales std with a factor estimated on a separate "
            "calibration set; the test year is used only for evaluation.\n"
        )
        lines.append(f"- Calibration years: {meta.get('calibration_years') or '(none)'}")
        lines.append(f"- Coverage target: **{_fmt(meta.get('coverage_target'), 3)}**")
        lines.append(f"- Calibration factor: **{_fmt(cal_factor, 4)}**")
        lines.append(f"- Calibration predictions: **{meta.get('n_calibration_predictions', 0)}**")
        if not has_calibration:
            lines.append("- Calibrated coverage: not available because no calibration set was provided.\n")
        lines.append("")
        lines.append("| stratum | raw coverage | calibrated coverage |")
        lines.append("|---|---|---|")
        lines.append(
            f"| global | {_fmt(g.get('coverage_95_raw'), 3)} | "
            f"{_fmt(g.get('coverage_95_calibrated'), 3)} |"
        )
        lines.append(
            f"| normal | {_fmt(_by_value(by, 'group:normal', 'coverage_95_raw'), 3)} | "
            f"{_fmt(_by_value(by, 'group:normal', 'coverage_95_calibrated'), 3)} |"
        )
        lines.append(
            f"| rare/extreme | {_fmt(_by_value(by, 'group:rare_or_extreme', 'coverage_95_raw'), 3)} | "
            f"{_fmt(_by_value(by, 'group:rare_or_extreme', 'coverage_95_calibrated'), 3)} |"
        )
        lines.append("")

        lines.append("## Stratified uncertainty calibration\n")
        cal = meta.get("calibration")
        lines.append(
            "Global calibration uses a single factor for every test row; "
            "**group**/**label** strategies estimate separate factors on the "
            "calibration year's anomaly strata so rare/extreme bands are not "
            "under-covered. A stratum with fewer than `min_samples` calibration "
            "points falls back (label → rare/extreme group → global).\n"
        )
        if cal is None:
            lines.append("- Calibration strategy: **global** (no calibration set provided).\n")
        else:
            cal_factors = cal.get("factors", {})
            cal_counts = cal.get("counts", {})
            cal_fallbacks = cal.get("fallbacks", {})

            def _factor_line(key: str) -> str:
                n = cal_counts.get(key)
                n_txt = f" (n={int(n)})" if n is not None else ""
                if key in cal_factors:
                    return f"{cal_factors[key]:.4f}{n_txt}"
                fb = cal_fallbacks.get(key, "global")
                return f"fallback → {fb}{n_txt}"

            lines.append(f"- Calibration strategy: **{cal.get('strategy')}**")
            lines.append(
                f"- Calibration anomaly scores: "
                f"{meta.get('calibration_anomaly_scores') or '(none)'}"
            )
            lines.append(
                f"- Min samples per stratum: **{cal.get('min_samples')}**"
            )
            lines.append(f"- Global factor (k_global): **{cal.get('global'):.4f}**")
            lines.append(f"- Factor normal: **{_factor_line('group:normal')}**")
            lines.append(
                f"- Factor rare_or_extreme: **{_factor_line('group:rare_or_extreme')}**"
            )
            if cal.get("strategy") == "label":
                lines.append("- Label-specific factors:")
                for label in SPECIFIC_ANOMALY_LABELS:
                    lines.append(f"  - {label}: {_factor_line(f'label:{label}')}")
            lines.append("")
            lines.append("| stratum | raw coverage | calibrated coverage |")
            lines.append("|---|---|---|")
            lines.append(
                f"| normal | {_fmt(_by_value(by, 'group:normal', 'coverage_95_raw'), 3)} | "
                f"{_fmt(_by_value(by, 'group:normal', 'coverage_95_calibrated'), 3)} |"
            )
            lines.append(
                f"| rare/extreme | {_fmt(_by_value(by, 'group:rare_or_extreme', 'coverage_95_raw'), 3)} | "
                f"{_fmt(_by_value(by, 'group:rare_or_extreme', 'coverage_95_calibrated'), 3)} |"
            )
            lines.append("")

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
            cov_cal_n = by.loc["group:normal", "coverage_95_calibrated"]
            cov_cal_r = by.loc["group:rare_or_extreme", "coverage_95_calibrated"]
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
                f"- Raw coverage@95 normal: {_fmt(cov_n, 3)}  |  rare/extreme: {_fmt(cov_r, 3)}"
            )
            lines.append(
                f"- Calibrated coverage@95 normal: {_fmt(cov_cal_n, 3)}  |  "
                f"rare/extreme: {_fmt(cov_cal_r, 3)}\n"
            )
            lines.append(f"1. Does the model err more on rare/extreme? {_verdict(mae_ratio)} (MAE ratio {mae_ratio:.2f}×).")
            lines.append(f"2. Is the model also more uncertain on rare/extreme? {_verdict(unc_ratio)} (uncertainty ratio {unc_ratio:.2f}×).\n")
        else:
            lines.append("_Not enough strata for an uncertainty comparison._\n")
    return "\n".join(lines) + "\n"


def write_outputs(
    predictions: pd.DataFrame,
    global_df: pd.DataFrame,
    by_df: pd.DataFrame,
    out_dir: str,
    meta: dict,
) -> Dict[str, Path]:
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
    by_df.to_csv(paths["metrics_by_anomaly_label"], index=False)
    paths["report"].write_text(_render_report(global_df, by_df, meta), encoding="utf-8")
    return paths


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
        "features": feats,
        "n_features": len(feats),
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
        "calibration_years": args_like.get("calibration_years"),
        "coverage_target": args_like.get("coverage_target"),
        "calibration_eps": args_like.get("calibration_eps"),
        "calibration_factor": args_like.get("calibration_factor"),
        "calibration_strategy": args_like.get("calibration_strategy"),
        "calibration_anomaly_scores": args_like.get("calibration_anomaly_scores"),
        "min_calibration_samples_per_stratum": args_like.get(
            "min_calibration_samples_per_stratum"
        ),
        "calibration": args_like.get("calibration"),
        "n_calibration_predictions": args_like.get("n_calibration_predictions", 0),
        "n_predictions": n_predictions,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
