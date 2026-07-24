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

from physiq_pv.data.pvgis_irradiance import (
    DIFFUSE_TILTED_VAR,
    DIRECT_TILTED_VAR,
    with_effective_poa,
)
from physiq_pv.model.st_gnn import STGNN

PVGIS_STGNN_FEATURES: List[str] = [
    "temperature_2m",
    "solar_irradiance_poa",
    "wind_speed_10m",
    "sin_elev",
    "cos_elev",
    "kt_poa",
    "kt_poa_std_3h",
    "dpoa_dt",
    "direct_irradiance_tilted",
    "diffuse_irradiance_tilted",
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
        "kt_poa", "kt_poa_std_3h", "dpoa_dt",
        "direct_irradiance_tilted", "diffuse_irradiance_tilted",
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
def validate_hourly_grid(times: pd.DatetimeIndex, *, label: str) -> None:
    """Require a unique, increasing time axis with exactly one-hour steps."""
    if len(times) < 2:
        raise ValueError(f"{label}: at least two timestamps are required.")
    if times.has_duplicates:
        raise ValueError(f"{label}: duplicate timestamps are not allowed.")
    if not times.is_monotonic_increasing:
        raise ValueError(f"{label}: timestamps must be strictly increasing.")
    deltas = np.diff(times.to_numpy(dtype="datetime64[ns]").astype(np.int64))
    expected = pd.Timedelta(hours=1).value
    bad = np.flatnonzero(deltas != expected)
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"{label}: non-hourly step between {times[i]} and {times[i + 1]} "
            f"({pd.Timedelta(int(deltas[i]), unit='ns')})."
        )


def _solar_geometry_and_clearsky_poa(
    times: pd.DatetimeIndex,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    surface_tilt: float,
    surface_azimuth: float,
) -> tuple:
    """Return solar elevation channels and clear-sky inclined POA [kW/m²]."""
    T, N = len(times), len(lats)
    times_utc = times.tz_localize("UTC") if times.tzinfo is None else times
    sin_elev = np.zeros((T, N), dtype=np.float32)
    cos_elev = np.zeros((T, N), dtype=np.float32)
    poa_cs = np.zeros((T, N), dtype=np.float32)
    fleet_lat, fleet_lon = float(np.nanmean(lats)), float(np.nanmean(lons))
    for p in range(N):
        lat_p = float(lats[p]) if np.isfinite(lats[p]) else fleet_lat
        lon_p = float(lons[p]) if np.isfinite(lons[p]) else fleet_lon
        loc = pvlib.location.Location(lat_p, lon_p, tz="UTC")
        sp = loc.get_solarposition(times_utc)
        elev = np.clip(sp["apparent_elevation"].values, 0.0, 90.0).astype(np.float32)
        sin_elev[:, p] = np.sin(np.radians(elev))
        cos_elev[:, p] = np.cos(np.radians(elev))
        try:
            cs = loc.get_clearsky(times_utc, model="ineichen")
        except Exception:
            cs = loc.get_clearsky(times_utc, model="simplified_solis")
        total = pvlib.irradiance.get_total_irradiance(
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            solar_zenith=sp["apparent_zenith"].to_numpy(),
            solar_azimuth=sp["azimuth"].to_numpy(),
            dni=cs["dni"].to_numpy(),
            ghi=cs["ghi"].to_numpy(),
            dhi=cs["dhi"].to_numpy(),
            albedo=0.0,
        )
        poa_cs[:, p] = np.clip(
            np.nan_to_num(np.asarray(total["poa_global"]), nan=0.0) / 1000.0,
            0.0,
            None,
        )
    return sin_elev, cos_elev, poa_cs


def build_year_raw(
    ds: xr.Dataset,
    target_variable: str,
    loc_dim: str = "location",
    kt_poa_max: float = 1.6,
) -> dict:
    """Build raw (un-normalised) per-(time, location) channels for one PVGIS year."""
    ds = with_effective_poa(ds)
    if not np.isfinite(kt_poa_max) or kt_poa_max <= 0:
        raise ValueError("kt_poa_max must be finite and positive.")
    for v in _REQUIRED_VARS + [target_variable]:
        if v not in ds:
            raise ValueError(f"Required PVGIS variable '{v}' missing from dataset.")

    times = pd.DatetimeIndex(ds["time"].values)
    validate_hourly_grid(times, label="PVGIS year")
    lats = np.asarray(ds["lat"].values, dtype=float)
    lons = np.asarray(ds["lon"].values, dtype=float)
    if not np.isfinite(lats).all() or not np.isfinite(lons).all():
        raise ValueError("PVGIS locations require finite latitude/longitude.")

    def col(v):  # (T, N)
        return np.asarray(ds[v].transpose(loc_dim, "time").values, dtype=np.float32).T

    temp = col("temperature_2m")
    solar_wm2 = col("solar_irradiance_poa")
    direct_wm2 = col(DIRECT_TILTED_VAR)
    diffuse_wm2 = col(DIFFUSE_TILTED_VAR)
    wind = col("wind_speed_10m")
    pv = col(target_variable)
    for name, values in (
        ("temperature_2m", temp),
        ("solar_irradiance_poa", solar_wm2),
        (DIRECT_TILTED_VAR, direct_wm2),
        (DIFFUSE_TILTED_VAR, diffuse_wm2),
        ("wind_speed_10m", wind),
        (target_variable, pv),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"PVGIS variable {name!r} contains non-finite values.")
    solar_kwm2 = np.clip(solar_wm2 / 1000.0, 0.0, None)

    surface_tilt = float(
        ds.attrs.get("tilt_angle", ds.attrs.get("pvgis_tilt_angle", 30.0))
    )
    surface_azimuth = float(
        ds.attrs.get("azimuth_angle", ds.attrs.get("pvgis_azimuth_angle", 180.0))
    )
    sin_elev, cos_elev, poa_cs = _solar_geometry_and_clearsky_poa(
        times,
        lats,
        lons,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
    )
    day = (sin_elev > 0.05) & (solar_kwm2 > 0.03)

    kt_poa = np.where(poa_cs > 0.1, solar_kwm2 / (poa_cs + 1e-6), 0.0)
    kt_poa = np.clip(kt_poa, 0.0, kt_poa_max).astype(np.float32)

    kt_poa_std = np.zeros_like(kt_poa)
    for p in range(kt_poa.shape[1]):
        kt_poa_std[:, p] = (
            pd.Series(kt_poa[:, p])
            .rolling(3, min_periods=1)
            .std()
            .fillna(0.0)
            .to_numpy()
        )

    dpoa = np.zeros_like(solar_kwm2)
    dpoa[1:, :] = solar_kwm2[1:, :] - solar_kwm2[:-1, :]
    direct_tilted = np.clip(direct_wm2 / 1000.0, 0.0, None).astype(np.float32)
    diffuse_tilted = np.clip(diffuse_wm2 / 1000.0, 0.0, None).astype(np.float32)

    return {
        "times": times,
        "lats": lats,
        "lons": lons,
        "temp": temp,
        "solar_wm2": solar_wm2,
        "wind": wind,
        "sin": sin_elev,
        "cos": cos_elev,
        "poa_cs": poa_cs,
        "kt_poa": kt_poa,
        "kt_poa_std": kt_poa_std,
        "dpoa": dpoa,
        "direct_tilted": direct_tilted,
        "diffuse_tilted": diffuse_tilted,
        "pv": pv,
        "day": day,
    }


def fit_normalization(train_raws: List[dict]) -> dict:
    """Per-location pv_scale (p99 daytime pv) and global z-score stats from TRAIN only."""
    n_loc = train_raws[0]["pv"].shape[1]
    # per-location p99 of daytime, positive pv (target scale)
    pv_stack = np.concatenate([r["pv"] for r in train_raws], axis=0)  # (sum_T, N)
    day_stack = np.concatenate([r["day"] for r in train_raws], axis=0)
    pv_scale = np.full(n_loc, np.nan, dtype=np.float64)
    for p in range(n_loc):
        vals = pv_stack[day_stack[:, p], p]
        vals = vals[vals > 0]
        if len(vals) > 10:
            pv_scale[p] = float(np.percentile(vals, 99)) + 1e-6
    fitted = np.isfinite(pv_scale) & (pv_scale > 0)
    if not fitted.any():
        raise ValueError("Training data has no usable positive daytime PV values.")
    fallback_scale = float(np.median(pv_scale[fitted]))
    pv_scale[~fitted] = fallback_scale

    def gstats(key):
        arr = np.concatenate([r[key] for r in train_raws], axis=0)
        return float(np.nanmean(arr)), float(np.nanstd(arr) + 1e-6)

    z = {k: gstats(k) for k in ("temp", "solar_wm2", "wind", "dpoa")}
    return {
        "pv_scale": pv_scale,
        "z": z,
        "pv_scale_fallback": fallback_scale,
        "pv_scale_fallback_count": int((~fitted).sum()),
    }


def assemble_feats(
    raw: dict,
    norm: dict,
    pv_target_clip_max: Optional[float] = None,
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
        raw["kt_poa"],
        raw["kt_poa_std"],
        zc("dpoa"),
        raw["direct_tilted"],
        raw["diffuse_tilted"],
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
        kt_poa_by_year: Optional[Dict[int, np.ndarray]] = None,
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
            kt_poa = (
                kt_poa_by_year[year] if kt_poa_by_year is not None else None
            )
            ts = times_by_year[year]
            for i in range(n_windows):
                samples.append((year, i))
                y_norm_rows.append(pvn[i + tgt])
                y_true_rows.append(pvr[i + tgt])
                if solar is not None:
                    solar_target_rows.append(solar[i + tgt])
                if kt_poa is not None:
                    kt_target_rows.append(kt_poa[i + tgt])
                times_rows.append(ts.values[i + tgt])
        if not samples:
            raise ValueError("No supervised windows could be built (year too short?).")

        self.samples = samples
        self.y_norm_all = np.stack(y_norm_rows)  # (n_samples, N)
        self.y_true_all = np.stack(y_true_rows)  # (n_samples, N)
        self.solar_irradiance_poa_target_all = (
            np.stack(solar_target_rows) if solar_target_rows else None
        )
        # Target-time inclined clear-sky index: auxiliary supervision target.
        self.kt_poa_target_all = (
            np.stack(kt_target_rows) if kt_target_rows else None
        )
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
        if self.kt_poa_target_all is not None:
            self.kt_poa_target_all = self.kt_poa_target_all[keep]
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

    def normal_training_mask(self) -> np.ndarray:
        """Return normal target cells whose own input history is also normal."""
        mask_all = self.anomaly_mask_all
        history_mask_all = self.anomaly_history_mask_all
        if mask_all is None or history_mask_all is None:
            raise ValueError(
                "normal_training_mask requires anomaly masks; call "
                "attach_anomaly_mask(train_scores) first."
            )
        if history_mask_all.shape != mask_all.shape:
            raise ValueError(
                "anomaly_history_mask_all must match anomaly_mask_all shape; "
                f"got {history_mask_all.shape} vs {mask_all.shape}."
            )
        normal = ~(mask_all | history_mask_all)
        if not bool(normal.any()):
            raise ValueError(
                "normal-only masking left no normal target/history cells."
            )
        return normal

    def drop_windows_without_normal_cells(self) -> tuple[int, int]:
        """Drop only windows for which every node is rare.

        Loss masking remains node-specific. This avoids discarding a regional
        window merely because one of many locations is anomalous.
        """
        normal = self.normal_training_mask()
        before = len(self.samples)
        keep_window = np.any(normal, axis=1)
        self._select_indices(np.flatnonzero(keep_window))
        return len(self.samples), before


def _align_locations(
    ds: xr.Dataset,
    canonical_ids: np.ndarray,
    canonical_lats: np.ndarray,
    canonical_lons: np.ndarray,
    *,
    label: str,
    loc_dim: str = "location",
) -> xr.Dataset:
    """Validate the node set and reorder it to the canonical training order."""
    if loc_dim not in ds.dims or loc_dim not in ds.coords:
        raise ValueError(f"{label}: missing {loc_dim!r} dimension/coordinate.")
    for name in ("lat", "lon"):
        if name not in ds:
            raise ValueError(f"{label}: missing location coordinate {name!r}.")

    ids = np.asarray(ds[loc_dim].values)
    ids_text = np.asarray([str(v) for v in ids])
    canonical_text = np.asarray([str(v) for v in canonical_ids])
    if len(np.unique(ids_text)) != len(ids_text):
        raise ValueError(f"{label}: duplicate location IDs are not allowed.")
    if set(ids_text) != set(canonical_text):
        missing = sorted(set(canonical_text) - set(ids_text))
        extra = sorted(set(ids_text) - set(canonical_text))
        raise ValueError(
            f"{label}: location IDs differ from training nodes; "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )

    position = {value: i for i, value in enumerate(ids_text)}
    order = np.asarray([position[value] for value in canonical_text], dtype=np.int64)
    aligned = ds.isel({loc_dim: order})
    lats = np.asarray(aligned["lat"].values, dtype=float)
    lons = np.asarray(aligned["lon"].values, dtype=float)
    if not np.allclose(lats, canonical_lats, rtol=0.0, atol=1e-6):
        raise ValueError(f"{label}: latitude changed for one or more location IDs.")
    if not np.allclose(lons, canonical_lons, rtol=0.0, atol=1e-6):
        raise ValueError(f"{label}: longitude changed for one or more location IDs.")
    return aligned


def build_datasets(
    train_ds_map: Dict[int, xr.Dataset],
    test_ds: xr.Dataset,
    seq_len: int,
    horizon: int,
    target_variable: str = DEFAULT_TARGET_VARIABLE,
    feature_names: Optional[List[str]] = None,
    pv_target_clip_max: Optional[float] = None,
    validation_year: Optional[int] = None,
    kt_poa_max: float = 1.6,
) -> dict:
    """Build disjoint train/validation/test datasets with train-only fitting."""
    if len(train_ds_map) < 2:
        raise ValueError(
            "At least two training years are required: the latest (or "
            "validation_year) is held out for validation."
        )
    if pv_target_clip_max is not None and pv_target_clip_max <= 0:
        raise ValueError("pv_target_clip_max must be positive or None.")
    selected = (
        list(feature_names)
        if feature_names is not None
        else list(PVGIS_STGNN_FEATURES)
    )
    unknown = [f for f in selected if f not in PVGIS_STGNN_FEATURES]
    if unknown:
        raise ValueError(f"Unknown feature(s) {unknown}; valid: {PVGIS_STGNN_FEATURES}.")
    if not selected:
        raise ValueError("feature_names selected an empty feature set.")
    keep_idx = [PVGIS_STGNN_FEATURES.index(f) for f in selected]

    years = sorted(train_ds_map)
    validation_year = max(years) if validation_year is None else validation_year
    if validation_year not in train_ds_map:
        raise ValueError(
            f"validation_year={validation_year} is not among training years {years}."
        )
    fit_years = [year for year in years if year != validation_year]
    if not fit_years:
        raise ValueError("Validation split leaves no year available for training.")

    canonical_ds = train_ds_map[fit_years[0]]
    loc_ids = np.asarray(canonical_ds["location"].values)
    canonical_lats = np.asarray(canonical_ds["lat"].values, dtype=float)
    canonical_lons = np.asarray(canonical_ds["lon"].values, dtype=float)
    if len(np.unique(np.asarray([str(v) for v in loc_ids]))) != len(loc_ids):
        raise ValueError("Canonical training year contains duplicate location IDs.")
    if not np.isfinite(canonical_lats).all() or not np.isfinite(canonical_lons).all():
        raise ValueError("Canonical training locations require finite coordinates.")

    aligned_train = {
        year: _align_locations(
            train_ds_map[year],
            loc_ids,
            canonical_lats,
            canonical_lons,
            label=f"PVGIS {year}",
        )
        for year in years
    }
    aligned_test = _align_locations(
        test_ds,
        loc_ids,
        canonical_lats,
        canonical_lons,
        label="PVGIS test year",
    )
    raws = {
        year: build_year_raw(
            aligned_train[year], target_variable, kt_poa_max=kt_poa_max
        )
        for year in years
    }
    test_raw = build_year_raw(
        aligned_test, target_variable, kt_poa_max=kt_poa_max
    )
    norm = fit_normalization([raws[year] for year in fit_years])

    def _select(feats: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(feats[:, :, keep_idx])

    def _make_dataset(
        raw_map: Dict[int, dict],
        *,
        include_physical_targets: bool,
    ) -> PVGISWindowDataset:
        feats, pvn, pvr, solar, kt_poa, times = {}, {}, {}, {}, {}, {}
        for year, raw in raw_map.items():
            feature_array, pv_norm, pv_raw = assemble_feats(
                raw, norm, pv_target_clip_max
            )
            feats[year] = _select(feature_array)
            pvn[year] = pv_norm
            pvr[year] = pv_raw
            if include_physical_targets:
                solar[year] = raw["solar_wm2"]
            kt_poa[year] = raw["kt_poa"]
            times[year] = raw["times"]
        return PVGISWindowDataset(
            feats,
            pvn,
            pvr,
            solar if include_physical_targets else None,
            times,
            seq_len,
            horizon,
            norm["pv_scale"],
            loc_ids,
            kt_poa_by_year=kt_poa,
        )

    train_dataset = _make_dataset(
        {year: raws[year] for year in fit_years},
        include_physical_targets=False,
    )
    validation_dataset = _make_dataset(
        {validation_year: raws[validation_year]},
        include_physical_targets=True,
    )
    test_dataset = _make_dataset({-1: test_raw}, include_physical_targets=True)
    return {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
        "train_years": fit_years,
        "validation_year": validation_year,
        "loc_ids": loc_ids,
        "lats": canonical_lats,
        "lons": canonical_lons,
        "n_features": len(selected),
        "features": selected,
        "pv_scale": norm["pv_scale"],
        "normalization": norm,
        "pv_target_clip_max": pv_target_clip_max,
    }


# --------------------------------------------------------------------------- #
# Model reuse + train / predict
# --------------------------------------------------------------------------- #
def make_model(
    n_nodes: int,
    seq_len: int,
    n_features: int = N_FEATURES,
    dropout: float = 0.0,
    n_sde_steps: int = 2,
    sigma_max: float = 0.5,
    use_irradiance_head: bool = True,
    kt_poa_max: float = 1.6,
    edge_prior_strength: float = 1.0,
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
        # Preserve the original feat/bilstm-gat backbone: exactly one GAT layer.
        gat_layers=1,
        dropout=dropout,
        use_patchtst=True,
        use_gat=True,
        bilstm_pooling="attn",
        n_sde_steps=n_sde_steps,
        sigma_max=sigma_max,
        use_irradiance_head=use_irradiance_head,
        kt_poa_max=kt_poa_max,
        edge_prior_strength=edge_prior_strength,
    )


