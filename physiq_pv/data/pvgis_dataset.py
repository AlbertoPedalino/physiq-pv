"""PVGIS-only dataset builders for the ST-GNN forecasting run."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pvlib
import torch
import xarray as xr
from torch.utils.data import Dataset

from physiq_pv.data.pvgis_irradiance import with_effective_poa
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
_REQUIRED_VARS = ["temperature_2m", "wind_speed_10m"]

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
    ds = with_effective_poa(ds)
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

    # Beam/diffuse split: prefer real PVGIS plane-of-array components
    # (direct+diffuse tilted, sum ~= POA); fall back to Erbs decomposition.
    if "direct_irradiance_tilted" in ds and "diffuse_irradiance_tilted" in ds:
        dni = np.clip(col("direct_irradiance_tilted") / 1000.0, 0.0, 1.5).astype(np.float32)
        dhi = np.clip(col("diffuse_irradiance_tilted") / 1000.0, 0.0, 1.0).astype(np.float32)
    else:
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
        self.times_by_year = times_by_year
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
        # Target and input-history anomaly masks for normal-only training.
        self.anomaly_mask_all: Optional[np.ndarray] = None
        self.anomaly_history_mask_all: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, k: int):
        year, i = self.samples[k]
        win = self.feats_by_year[year][i : i + self.seq_len]  # (seq_len, N, C)
        x = torch.from_numpy(np.ascontiguousarray(win.transpose(1, 0, 2)))  # (N, seq_len, C)
        y = torch.from_numpy(self.y_norm_all[k])  # (N,)
        return x, y, k

    def _select_indices(self, keep: np.ndarray) -> None:
        """Keep the selected sample indices in every per-window array."""
        keep = np.asarray(keep, dtype=np.int64)
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
        if self.anomaly_history_mask_all is not None:
            self.anomaly_history_mask_all = self.anomaly_history_mask_all[keep]
        self.target_time_all = self.target_time_all[keep]

    def subsample(self, max_samples: Optional[int], seed: int = 0) -> "PVGISWindowDataset":
        """Randomly keep at most `max_samples` windows (in place); returns self."""
        if max_samples is None or len(self.samples) <= max_samples:
            return self
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(self.samples), size=max_samples, replace=False))
        self._select_indices(keep)
        return self

    def attach_anomaly_mask(self, anomaly_scores: Optional[pd.DataFrame]) -> int:
        """Attach target and input-history anomaly masks; return target count."""
        n_samples = len(self.samples)
        mask = np.zeros((n_samples, self.n_nodes), dtype=bool)
        history_mask = np.zeros((n_samples, self.n_nodes), dtype=bool)
        self.anomaly_mask_all = mask
        self.anomaly_history_mask_all = history_mask
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
        sample_years = np.fromiter((year for year, _ in self.samples), dtype=np.int64,
                                    count=n_samples)
        sample_starts = np.fromiter((start for _, start in self.samples), dtype=np.int64,
                                      count=n_samples)
        sample_idx_by_year = {
            year: np.flatnonzero(sample_years == year)
            for year in self.times_by_year
        }
        time_int_by_year = {
            year: pd.DatetimeIndex(times).to_numpy("datetime64[ns]").astype("int64")
            for year, times in self.times_by_year.items()
        }
        for n, loc in enumerate(self.loc_ids):
            ts_arr = by_loc.get(str(loc))
            if ts_arr is None or ts_arr.size == 0:
                continue
            mask[:, n] = np.isin(target_int, ts_arr)
            # A prefix sum marks all sliding input windows containing at least
            # one labelled timestamp for this node, without iterating windows.
            for year, sample_idx in sample_idx_by_year.items():
                if sample_idx.size == 0:
                    continue
                is_rare_at_time = np.isin(time_int_by_year[year], ts_arr)
                prefix = np.concatenate(([0], np.cumsum(is_rare_at_time, dtype=np.int64)))
                starts = sample_starts[sample_idx]
                history_mask[sample_idx, n] = (
                    prefix[starts + self.seq_len] > prefix[starts]
                )
        return int(mask.sum())

    def filter_normal_only_windows(self) -> tuple[int, int]:
        """Drop windows containing any target/history anomaly in any node.

        This is the paper-literal normal-only protocol adapted to PVGIS windows:
        the training loader sees only fully in-distribution windows, then the
        SDE pseudo-OOD batch is generated by perturbing those remaining windows.
        """
        mask_all = self.anomaly_mask_all
        history_mask_all = self.anomaly_history_mask_all
        if mask_all is None or history_mask_all is None:
            raise ValueError(
                "filter_normal_only_windows requires anomaly masks; call "
                "attach_anomaly_mask(train_scores) first."
            )
        if history_mask_all.shape != mask_all.shape:
            raise ValueError(
                "anomaly_history_mask_all must match anomaly_mask_all shape; "
                f"got {history_mask_all.shape} vs {mask_all.shape}."
            )
        before = len(self.samples)
        keep_window = ~np.any(mask_all | history_mask_all, axis=1)
        kept = int(keep_window.sum())
        if kept == 0:
            raise ValueError(
                "normal-only window filtering removed every training window; "
                "relax the anomaly labels/window definition or disable "
                "--train-normal-only."
            )
        self._select_indices(np.flatnonzero(keep_window))
        return kept, before


def build_datasets(
    train_ds_map: Dict[int, xr.Dataset],
    test_ds: xr.Dataset,
    seq_len: int,
    horizon: int,
    target_variable: str = DEFAULT_TARGET_VARIABLE,
    feature_names: Optional[List[str]] = None,
    pv_target_clip_max: Optional[float] = 1.5,
) -> dict:
    """
    Build train/test window datasets with train-fitted normalisation.

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

    train_raws = {y: build_year_raw(ds, target_variable) for y, ds in train_ds_map.items()}
    test_raw = build_year_raw(test_ds, target_variable)

    n_loc = test_raw["pv"].shape[1]
    for y, r in train_raws.items():
        if r["pv"].shape[1] != n_loc:
            raise ValueError(
                f"Location count mismatch: train year {y} has {r['pv'].shape[1]}, "
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
    return {
        "train": train_dataset,
        "test": test_dataset,
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
    n_sde_steps: int = 4,
    sigma_max: float = 0.5,
    use_irradiance_head: bool = True,
) -> STGNN:
    """Instantiate STGNN with the selected PVGIS feature count."""
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
        n_sde_steps=n_sde_steps,
        sigma_max=sigma_max,
        use_irradiance_head=use_irradiance_head,
    )


