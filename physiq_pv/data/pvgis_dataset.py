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

import time
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
DAYTIME_IRRADIANCE_THRESHOLD_WM2 = 10.0
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


def assemble_feats(
    raw: dict,
    norm: dict,
    pv_target_clip_max: Optional[float] = 1.5,
) -> tuple:
    """Return (feats (T,N,11), pv_norm (T,N), pv_raw (T,N)) using train normalisation."""
    z = norm["z"]
    pv_scale = norm["pv_scale"][None, :]

    def zc(key):
        mu, sd = z[key]
        return (raw[key] - mu) / sd

    pv_scaled = raw["pv"] / pv_scale
    if pv_target_clip_max is None:
        pv_norm = np.maximum(pv_scaled, 0.0).astype(np.float32)
    else:
        pv_norm = np.clip(
            pv_scaled, 0.0, pv_target_clip_max
        ).astype(np.float32)
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
        solarraw_by_year: Optional[Dict[int, np.ndarray]],
        times_by_year: Dict[int, pd.DatetimeIndex],
        seq_len: int,
        horizon: int,
        pv_scale: np.ndarray,
        loc_ids: np.ndarray,
        kt_by_year: Optional[Dict[int, np.ndarray]] = None,
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
        solar_target_rows: List[np.ndarray] = []
        kt_target_rows: List[np.ndarray] = []
        times_rows: List[np.datetime64] = []
        for year, feats in feats_by_year.items():
            T = feats.shape[0]
            n_windows = T - seq_len - horizon + 1
            if n_windows <= 0:
                continue
            tgt = seq_len + horizon - 1
            pvn = pvnorm_by_year[year]
            pvr = pvraw_by_year[year]
            solar = solarraw_by_year[year] if solarraw_by_year is not None else None
            kt = kt_by_year[year] if kt_by_year is not None else None
            ts = times_by_year[year]
            for i in range(n_windows):
                samples.append((year, i))
                y_norm_rows.append(pvn[i + tgt])
                y_true_rows.append(pvr[i + tgt])
                if solar is not None:
                    solar_target_rows.append(solar[i + tgt])
                if kt is not None:
                    kt_target_rows.append(kt[i + tgt])
                times_rows.append(ts.values[i + tgt])
        if not samples:
            raise ValueError("No supervised windows could be built (year too short?).")

        self.samples = samples
        self.y_norm_all = np.stack(y_norm_rows)  # (n_samples, N)
        self.y_true_all = np.stack(y_true_rows)  # (n_samples, N)
        self.solar_irradiance_poa_target_all = (
            np.stack(solar_target_rows) if solar_target_rows else None
        )
        # Target-time clear-sky index (kt): supervision target for the optional
        # auxiliary irradiance loss (use_irradiance_loss in train_model).
        self.kt_target_all = np.stack(kt_target_rows) if kt_target_rows else None
        self.target_time_all = pd.DatetimeIndex(times_rows)
        # Per-(sample, node) anomaly mask. None until attach_anomaly_mask() is
        # called. Used ONLY to TARGET anomaly-aware training noise — never a model
        # input or supervision target. Default training keeps this None.
        self.anomaly_mask_all: Optional[np.ndarray] = None

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
        if self.solar_irradiance_poa_target_all is not None:
            self.solar_irradiance_poa_target_all = (
                self.solar_irradiance_poa_target_all[keep]
            )
        if self.kt_target_all is not None:
            self.kt_target_all = self.kt_target_all[keep]
        if self.anomaly_mask_all is not None:
            self.anomaly_mask_all = self.anomaly_mask_all[keep]
        self.target_time_all = self.target_time_all[keep]
        return self

    def attach_anomaly_mask(self, anomaly_scores: Optional[pd.DataFrame]) -> int:
        """Build self.anomaly_mask_all (n_samples, N) bool from anomaly scores.

        A (location, target_time) carrying any anomaly label is marked True
        (rare_or_extreme), matching attach_anomaly_labels. The labels are used
        ONLY to target anomaly-aware training noise, NEVER as a model input or
        supervised target. Returns the number of True (anomalous) cells.
        """
        n_samples = len(self.samples)
        mask = np.zeros((n_samples, self.n_nodes), dtype=bool)
        self.anomaly_mask_all = mask
        if anomaly_scores is None or anomaly_scores.empty:
            return 0
        scores = anomaly_scores[["location", "timestamp"]].copy()
        scores["location"] = scores["location"].astype(str)
        ts_int = pd.to_datetime(scores["timestamp"]).to_numpy("datetime64[ns]").astype("int64")
        scores["ts_int"] = ts_int
        by_loc = scores.groupby("location")["ts_int"].apply(
            lambda s: np.unique(s.to_numpy())
        ).to_dict()
        target_int = self.target_time_all.to_numpy("datetime64[ns]").astype("int64")
        for n, loc in enumerate(self.loc_ids):
            ts_arr = by_loc.get(str(loc))
            if ts_arr is None or ts_arr.size == 0:
                continue
            mask[:, n] = np.isin(target_int, ts_arr)
        return int(mask.sum())


def build_datasets(
    train_ds_map: Dict[int, xr.Dataset],
    test_ds: xr.Dataset,
    seq_len: int,
    horizon: int,
    target_variable: str = DEFAULT_TARGET_VARIABLE,
    feature_names: Optional[List[str]] = None,
    calibration_ds_map: Optional[Dict[int, xr.Dataset]] = None,
    pv_target_clip_max: Optional[float] = 1.5,
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

    feats_tr, pvn_tr, pvr_tr, kt_tr, times_tr = {}, {}, {}, {}, {}
    for y, r in train_raws.items():
        f, pn, pr = assemble_feats(r, norm, pv_target_clip_max)
        feats_tr[y], pvn_tr[y], pvr_tr[y] = _select(f), pn, pr
        kt_tr[y], times_tr[y] = r["kt"], r["times"]

    feats_cal, pvn_cal, pvr_cal, solar_cal, kt_cal, times_cal = {}, {}, {}, {}, {}, {}
    for y, r in calibration_raws.items():
        f, pn, pr = assemble_feats(r, norm, pv_target_clip_max)
        feats_cal[y], pvn_cal[y], pvr_cal[y] = _select(f), pn, pr
        solar_cal[y], kt_cal[y], times_cal[y] = r["solar_wm2"], r["kt"], r["times"]

    f, pn, pr = assemble_feats(test_raw, norm, pv_target_clip_max)
    feats_te = {-1: _select(f)}
    pvn_te = {-1: pn}
    pvr_te = {-1: pr}
    solar_te = {-1: test_raw["solar_wm2"]}
    kt_te = {-1: test_raw["kt"]}
    times_te = {-1: test_raw["times"]}

    loc_ids = np.asarray(test_ds["location"].values)
    train_dataset = PVGISWindowDataset(
        feats_tr, pvn_tr, pvr_tr, None, times_tr,
        seq_len, horizon, norm["pv_scale"], loc_ids,
        kt_by_year=kt_tr,
    )
    test_dataset = PVGISWindowDataset(
        feats_te, pvn_te, pvr_te, solar_te, times_te,
        seq_len, horizon, norm["pv_scale"], loc_ids,
        kt_by_year=kt_te,
    )
    calibration_dataset = None
    if calibration_raws:
        calibration_dataset = PVGISWindowDataset(
            feats_cal, pvn_cal, pvr_cal, solar_cal, times_cal,
            seq_len, horizon, norm["pv_scale"], loc_ids,
            kt_by_year=kt_cal,
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
    n_nodes: int,
    seq_len: int,
    n_features: int = N_FEATURES,
    dropout: float = 0.2,
    enhanced_dropout: bool = False,
    use_irradiance_head: bool = True,
) -> STGNN:
    """Instantiate the existing STGNN with the PVGIS-only feature count.

    `dropout` is exposed so future MC-Dropout experiments can keep dropout layers
    active at inference; it does not change the deterministic eval path here.

    `enhanced_dropout` (model_type=stgnn_enhanced_dropout ablation) adds explicit
    nn.Dropout modules after the BiLSTM temporal embedding, after the projection,
    and inside the pv head, so enable_dropout_only() reactivates more than just
    the GAT attention dropout at MC inference. False -> identical to the default.

    `use_irradiance_head` (irradiance ablation): False removes head_ghi entirely
    (production-only model; forward returns (None, pred_pv)). True -> identical
    to the historical architecture.
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
        enhanced_dropout=enhanced_dropout,
        use_irradiance_head=use_irradiance_head,
    )


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
    global_df: pd.DataFrame,
    by_df: pd.DataFrame,
    mc_dropout: bool = False,
    calibration: Optional[dict] = None,
) -> dict:
    """
    Flatten global + by-stratum metrics into namespaced W&B scalars.

    Keys: mae/global, rmse/global, mae|rmse/{normal,rare_extreme},
    ratio/{mae,rmse}_rare_normal, and (when mc_dropout) uncertainty/* (mean +
    p90 std), and (only under post-hoc calibration) coverage_95_calibrated/*. When `calibration` is
    given, also calibration/{factor_global,factor_normal,factor_rare_extreme,
    coverage_target}. Only finite (numeric) values are emitted; the string
    strategy is logged separately by the runner.
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
        # NOTE: coverage from the Gaussian band is logged by the runner under the
        # explicit `picp_gaussian/*` keys (paper-style interval metrics). The old
        # ambiguous `coverage_95/*` and `coverage_95_raw/*` keys are intentionally
        # NOT emitted here. The primary coverage metric is `picp_pi/*`.
        # Post-hoc calibrated coverage is logged only when a calibration was run
        # (NaN otherwise -> skipped), and is a SECONDARY result.
        cov_cal_g = float(g.get("coverage_95_calibrated", float("nan")))
        if pd.notna(cov_cal_g):
            out["coverage_95_calibrated/global"] = cov_cal_g
        cov_cal_n = _get("group:normal", "coverage_95_calibrated")
        cov_cal_r = _get("group:rare_or_extreme", "coverage_95_calibrated")
        if cov_cal_n is not None:
            out["coverage_95_calibrated/normal"] = cov_cal_n
        if cov_cal_r is not None:
            out["coverage_95_calibrated/rare_extreme"] = cov_cal_r
        cal_factor = float(g.get("calibration_factor", float("nan")))
        if pd.notna(cal_factor):
            out["uncertainty/calibration_factor"] = cal_factor

    if calibration is not None:
        out["calibration/factor_global"] = float(calibration["global"])
        fac_n = _get("group:normal", "calibration_factor")
        fac_r = _get("group:rare_or_extreme", "calibration_factor")
        if fac_n is not None:
            out["calibration/factor_normal"] = fac_n
        if fac_r is not None:
            out["calibration/factor_rare_extreme"] = fac_r
        ct = calibration.get("coverage_target")
        if ct is not None:
            out["calibration/coverage_target"] = float(ct)
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
        # Post-hoc calibration is a SECONDARY, opt-in variant. These sections are
        # rendered only when a calibration was actually estimated; the paper-style
        # primary result lives in "Interval reliability & sharpness".
        if meta.get("calibration") is not None:
            cal_factor = meta.get("calibration_factor")
            lines.append("## Post-hoc calibrated variant (secondary)\n")
            lines.append(
                "**Secondary, opt-in result** (`--enable-posthoc-calibration`). This is "
                "NOT the paper-style primary interval — see *Interval reliability & "
                "sharpness*. Here the band is `mean ± k·std_raw`, where k is the "
                "`coverage_target` quantile of `|y_true − y_pred_mean| / std_raw` "
                "estimated on a separate calibration set (k absorbs the quantile — no "
                "extra 1.96 factor). The test year is used only for evaluation.\n"
            )
            lines.append(
                "- `coverage_95_raw` = coverage of the Gaussian diagnostic band "
                "(`mean ± 1.96·std_raw`)."
            )
            lines.append(
                "- `coverage_95_calibrated` = coverage of this post-hoc calibrated band."
            )
            lines.append(f"- Calibration years: {meta.get('calibration_years') or '(none)'}")
            lines.append(f"- Coverage target: **{_fmt(meta.get('coverage_target'), 3)}**")
            lines.append(f"- Calibration factor: **{_fmt(cal_factor, 4)}**")
            lines.append(f"- Calibration predictions: **{meta.get('n_calibration_predictions', 0)}**")
            lines.append("")
            lines.append("| stratum | Gaussian coverage | calibrated coverage |")
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

            lines.append("## Stratified post-hoc calibrated variant (secondary)\n")
            cal = meta.get("calibration")
            lines.append(
                "Per-stratum post-hoc factors (**group**/**label**): separate factors "
                "on the calibration year's anomaly strata so rare/extreme bands are not "
                "under-covered; a stratum with fewer than `min_samples` calibration "
                "points falls back (label → rare/extreme group → global). Secondary "
                "diagnostic only — not the paper-style primary interval.\n"
            )
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
            lines.append(f"- Min samples per stratum: **{cal.get('min_samples')}**")
            lines.append(f"- Global factor (k_global): **{cal.get('global'):.4f}**")
            lines.append(f"- Factor normal: **{_factor_line('group:normal')}**")
            lines.append(
                f"- Factor rare_or_extreme: **{_factor_line('group:rare_or_extreme')}**"
            )
            if cal.get("strategy") == "label":
                lines.append("- Label-specific factors:")
                for label in SPECIFIC_ANOMALY_LABELS:
                    lines.append(f"  - {label}: {_factor_line(f'label:{label}')}")
            sd = cal.get("std_diagnostics", {})
            if sd:
                lines.append("")
                lines.append(
                    "Raw MC std on the calibration set (sanity check that large k "
                    "is not driven by near-zero std):"
                )
                gd = sd.get("global", {})
                if gd:
                    lines.append(
                        f"- std_raw global: min **{gd['min']:.4g}**, max **{gd['max']:.4g}**, "
                        f"% < eps **{gd['pct_below_eps'] * 100:.2f}%** (n={gd['n']})"
                    )
                nd = sd.get(GROUP_NORMAL, {})
                rd = sd.get(GROUP_RARE, {})
                if nd:
                    lines.append(
                        f"- std_raw normal: mean **{nd['mean']:.4g}**, "
                        f"median **{nd['median']:.4g}** (n={nd['n']})"
                    )
                if rd:
                    lines.append(
                        f"- std_raw rare/extreme: mean **{rd['mean']:.4g}**, "
                        f"median **{rd['median']:.4g}** (n={rd['n']})"
                    )
            lines.append("")
            lines.append("| stratum | Gaussian coverage | calibrated coverage |")
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
                f"- Gaussian coverage@95 (diagnostic) normal: {_fmt(cov_n, 3)}  |  "
                f"rare/extreme: {_fmt(cov_r, 3)}"
            )
            if meta.get("calibration") is not None:
                lines.append(
                    f"- Post-hoc calibrated coverage@95 (secondary) normal: {_fmt(cov_cal_n, 3)}  |  "
                    f"rare/extreme: {_fmt(cov_cal_r, 3)}"
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
        posthoc = "calibrated" in iv
        lines.append("## Interval reliability & sharpness (PICP / NMPIL / CLC)\n")
        lines.append(
            "Paper-style evaluation (uncertainty-aware rainfall prediction). The "
            "**primary predictive intervals (`pi`) are built directly from the MC "
            "Dropout sample distribution** (empirical quantiles q(alpha/2), "
            "q(1-alpha/2)); **no post-hoc calibration is used in the main "
            "protocol**"
            + (" (a post-hoc calibrated band is shown below only because "
               "--enable-posthoc-calibration was set)." if posthoc else ".")
            + " The Gaussian band (`gaussian`, mean ± 1.96·std_raw) is a secondary "
              "diagnostic only.\n"
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
        if posthoc:
            kinds.append(("calibrated", "post-hoc calibrated (diagnostic)"))
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
        "wandb_artifacts_uploaded": args_like.get("wandb_artifacts_uploaded", False),
        "posthoc_executed": args_like.get("posthoc_executed", False),
        "posthoc_uploaded": args_like.get("posthoc_uploaded", False),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
