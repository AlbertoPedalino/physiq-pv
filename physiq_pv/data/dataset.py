from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib
import torch
import xarray as xr
from torch.utils.data import Dataset

SEQ_LEN = 24
FEATURE_NAMES = (
    "temp_z",
    "wind_z",
    "sin_elev",
    "cos_elev",
    "pv_lag",
    "pv_observed",
    "poa_z",
    "diffuse_fraction",
    "kt_poa",
    "kt_poa_std_3h",
    "dpoa_z",
    "m1",
    "m2",
    "m3",
    "m4",
    "m5",
    "quality_valid",
)
N_FEATURES = len(FEATURE_NAMES)
POA_INPUT_FEATURES = (
    "poa_z",
    "diffuse_fraction",
    "kt_poa",
    "kt_poa_std_3h",
    "dpoa_z",
    "m1",
    "m2",
    "m4",
    "m5",
    "quality_valid",
)
POA_INPUT_INDICES = tuple(FEATURE_NAMES.index(name) for name in POA_INPUT_FEATURES)
KT_INPUT_MAX = 2.5


def validate_hourly_grid(times: pd.DatetimeIndex) -> None:
    """Require a strictly increasing, duplicate-free hourly time axis."""
    if len(times) < 2:
        raise ValueError("At least two hourly timestamps are required")
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Dataset time coordinate must be unique and increasing")
    deltas = times[1:] - times[:-1]
    hourly = np.asarray(deltas == pd.Timedelta(hours=1))
    if not hourly.all():
        first_bad = int(np.flatnonzero(~hourly)[0])
        raise ValueError(
            "Dataset time coordinate must be a complete hourly grid; "
            f"gap between {times[first_bad]} and {times[first_bad + 1]}"
        )


def _solar_geometry_and_clearsky_poa(
    times: pd.DatetimeIndex,
    lats: np.ndarray,
    lons: np.ndarray,
    surface_tilt: float,
    surface_azimuth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return solar elevation features and clear-sky tilted POA in kW/m²."""
    n_steps, n_plants = len(times), len(lats)
    times_utc = times.tz_localize("UTC") if times.tz is None else times.tz_convert("UTC")
    fleet_lat = float(np.nanmean(lats))
    fleet_lon = float(np.nanmean(lons))

    sin_elev = np.zeros((n_steps, n_plants), dtype=np.float32)
    cos_elev = np.zeros((n_steps, n_plants), dtype=np.float32)
    poa_cs = np.zeros((n_steps, n_plants), dtype=np.float32)

    for plant in range(n_plants):
        lat = float(lats[plant]) if np.isfinite(lats[plant]) else fleet_lat
        lon = float(lons[plant]) if np.isfinite(lons[plant]) else fleet_lon
        loc = pvlib.location.Location(lat, lon, tz="UTC")
        solar_position = loc.get_solarposition(times_utc)
        elevation = np.clip(
            solar_position["apparent_elevation"].to_numpy(), 0.0, 90.0
        ).astype(np.float32)
        sin_elev[:, plant] = np.sin(np.radians(elevation))
        cos_elev[:, plant] = np.cos(np.radians(elevation))

        try:
            clear_sky = loc.get_clearsky(times_utc, model="ineichen")
        except Exception:
            clear_sky = loc.get_clearsky(times_utc, model="simplified_solis")
        total = pvlib.irradiance.get_total_irradiance(
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            solar_zenith=solar_position["apparent_zenith"],
            solar_azimuth=solar_position["azimuth"],
            dni=clear_sky["dni"],
            ghi=clear_sky["ghi"],
            dhi=clear_sky["dhi"],
            # The supplied NetCDF has beam + diffuse tilted components but no
            # separate PVGIS ground-reflected Gr(i) channel.
            albedo=0.0,
        )
        poa = np.nan_to_num(
            total["poa_global"].to_numpy(), nan=0.0
        ).astype(np.float32)
        poa_cs[:, plant] = np.clip(poa / 1000.0, 0.0, None)

    return sin_elev, cos_elev, poa_cs


class PVDataset(Dataset):
    """
    Sliding-window dataset for one-hour-ahead fleet PV forecasting.

    All fitted statistics use only ``fit_time_mask``. POA remains an auxiliary
    target in both variants. ``include_poa_inputs=False`` masks every encoder
    channel that contains observed POA information while preserving the exact
    same tensor shape and model parameterization.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        m_components: dict[str, np.ndarray],
        seq_len: int = SEQ_LEN,
        pr_max: float = 1.5,
        fit_time_mask: np.ndarray | None = None,
        include_poa_inputs: bool = True,
        surface_tilt: float | None = None,
        surface_azimuth: float | None = None,
    ):
        if pr_max <= 0.1:
            raise ValueError("pr_max must be greater than 0.1")
        for key in ("m1", "m2", "m3", "m4", "m5"):
            if key not in m_components:
                raise ValueError(f"m_components missing required key '{key}'")

        self.seq_len = int(seq_len)
        if self.seq_len < 1:
            raise ValueError("seq_len must be positive")
        self.include_poa_inputs = bool(include_poa_inputs)
        self.pr_max = float(pr_max)
        n_steps = ds.sizes["time"]
        n_plants = ds.sizes["plant"]
        if self.include_poa_inputs:
            required_components = (
                "direct_irradiance_tilted",
                "diffuse_irradiance_tilted",
            )
            missing_components = [
                name for name in required_components if name not in ds
            ]
            if missing_components:
                raise ValueError(
                    "POA-enabled inputs require tilted components: "
                    + ", ".join(missing_components)
                )

        if fit_time_mask is None:
            fit_time_mask = np.ones(n_steps, dtype=bool)
        else:
            fit_time_mask = np.asarray(fit_time_mask, dtype=bool)
            if fit_time_mask.shape != (n_steps,):
                raise ValueError(
                    f"fit_time_mask shape {fit_time_mask.shape} != ({n_steps},)"
                )
            if not fit_time_mask.any():
                raise ValueError("fit_time_mask must contain training timesteps")
        self.fit_time_mask = fit_time_mask

        surface_tilt = float(
            ds.attrs.get("pvgis_tilt_angle", 30.0)
            if surface_tilt is None
            else surface_tilt
        )
        surface_azimuth = float(
            ds.attrs.get("pvgis_azimuth_angle", 180.0)
            if surface_azimuth is None
            else surface_azimuth
        )

        zscore_state: dict[str, dict[str, float]] = {}

        def _fit_zscore(name: str, values: np.ndarray) -> np.ndarray:
            fit_values = values[fit_time_mask]
            mean = float(np.nanmean(fit_values))
            std = float(np.nanstd(fit_values))
            if not np.isfinite(mean):
                mean = 0.0
            if not np.isfinite(std) or std < 1e-6:
                std = 1.0
            zscore_state[name] = {"mean": mean, "std": std}
            return ((np.nan_to_num(values, nan=mean) - mean) / std).astype(
                np.float32
            )

        temperature = ds["temperature_2m"].values.T.astype(float)
        wind = ds["wind_speed_10m"].values.T.astype(float)
        poa_kwm2 = np.clip(
            ds["solar_irradiance_poa"].values.T.astype(float) / 1000.0,
            0.0,
            None,
        )
        energy_raw = ds["ENERGIA"].values.T.astype(float)
        pv_observed = np.isfinite(energy_raw)
        energy = np.nan_to_num(energy_raw, nan=0.0)

        lats = ds["lat"].values.astype(float)
        lons = ds["lon"].values.astype(float)
        times = pd.DatetimeIndex(ds.coords["time"].values)
        validate_hourly_grid(times)
        if n_steps <= self.seq_len:
            raise ValueError(
                f"Dataset has {n_steps} steps but seq_len={self.seq_len}"
            )
        sin_elev, cos_elev, poa_cs = _solar_geometry_and_clearsky_poa(
            times,
            lats,
            lons,
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
        )
        self.poa_cs = poa_cs

        geometric_day = sin_elev > 0.05
        irradiance_day = poa_kwm2 > 0.03
        day_mask = geometric_day & irradiance_day
        fit_poa_day_mask = day_mask & fit_time_mask[:, None]
        fit_pv_day_mask = fit_poa_day_mask & pv_observed

        pv_scale = np.full(n_plants, np.nan, dtype=np.float64)
        poa_scale = np.full(n_plants, np.nan, dtype=np.float64)
        for plant in range(n_plants):
            energy_values = energy[fit_pv_day_mask[:, plant], plant]
            energy_values = energy_values[energy_values > 0]
            poa_values = poa_kwm2[fit_poa_day_mask[:, plant], plant]
            poa_values = poa_values[poa_values > 0]
            if len(energy_values) > 10:
                pv_scale[plant] = float(np.percentile(energy_values, 99)) + 1e-6
            if len(poa_values) > 10:
                poa_scale[plant] = float(np.percentile(poa_values, 99)) + 1e-6
        pv_scale_fallback = ~np.isfinite(pv_scale)
        poa_scale_fallback = ~np.isfinite(poa_scale)
        fitted_pv_scale = pv_scale[~pv_scale_fallback]
        fitted_poa_scale = poa_scale[~poa_scale_fallback]
        pv_scale[pv_scale_fallback] = (
            float(np.median(fitted_pv_scale))
            if fitted_pv_scale.size
            else 1.0
        )
        poa_scale[poa_scale_fallback] = (
            float(np.median(fitted_poa_scale))
            if fitted_poa_scale.size
            else 1.0
        )

        target_pv = np.clip(energy / pv_scale[None, :], 0.0, 1.5)
        pv_lag = target_pv.astype(np.float32)

        if "diffuse_irradiance_tilted" in ds:
            diffuse_poa = np.clip(
                ds["diffuse_irradiance_tilted"].values.T.astype(float) / 1000.0,
                0.0,
                None,
            )
            diffuse_fraction = np.divide(
                diffuse_poa,
                poa_kwm2,
                out=np.zeros_like(diffuse_poa),
                where=poa_kwm2 > 1e-6,
            )
        else:
            diffuse_fraction = np.zeros_like(poa_kwm2)
        diffuse_fraction = np.clip(diffuse_fraction, 0.0, 1.0).astype(np.float32)

        kt_poa = np.divide(
            poa_kwm2,
            poa_cs,
            out=np.zeros_like(poa_kwm2),
            where=poa_cs > 0.1,
        )
        kt_poa = np.clip(kt_poa, 0.0, KT_INPUT_MAX).astype(np.float32)
        kt_poa_std = np.zeros_like(kt_poa)
        for plant in range(n_plants):
            kt_poa_std[:, plant] = (
                pd.Series(kt_poa[:, plant])
                .rolling(window=3, min_periods=1)
                .std()
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )

        dpoa = np.zeros_like(poa_kwm2)
        dpoa[1:] = poa_kwm2[1:] - poa_kwm2[:-1]

        quality_arrays: dict[str, np.ndarray] = {}
        quality_finite = np.ones((n_steps, n_plants), dtype=bool)
        for key in ("m1", "m2", "m3", "m4", "m5"):
            raw = np.asarray(m_components[key], dtype=float).T
            if raw.shape != (n_steps, n_plants):
                raise ValueError(
                    f"{key} shape {raw.shape} != ({n_steps}, {n_plants})"
                )
            quality_arrays[key] = np.nan_to_num(raw, nan=0.0).astype(np.float32)
            if key != "m3":
                quality_finite &= np.isfinite(raw)
        quality_valid = quality_finite.astype(np.float32)

        feature_arrays = [
            _fit_zscore("temperature_2m", temperature),
            _fit_zscore("wind_speed_10m", wind),
            sin_elev,
            cos_elev,
            pv_lag,
            pv_observed.astype(np.float32),
            _fit_zscore("solar_irradiance_poa_kwm2", poa_kwm2),
            diffuse_fraction,
            kt_poa,
            kt_poa_std,
            _fit_zscore("dpoa_kwm2", dpoa),
            quality_arrays["m1"],
            quality_arrays["m2"],
            quality_arrays["m3"],
            quality_arrays["m4"],
            quality_arrays["m5"],
            quality_valid,
        ]
        feats = np.stack(feature_arrays, axis=-1).astype(np.float32)
        if not include_poa_inputs:
            feats[..., POA_INPUT_INDICES] = 0.0
        self.feats = feats

        pr_proxy = np.ones(n_plants, dtype=np.float64)
        valid_counts = np.zeros(n_plants, dtype=int)
        for plant in range(n_plants):
            mask = (
                fit_pv_day_mask[:, plant]
                & (poa_kwm2[:, plant] > 0)
                & (target_pv[:, plant] > 0)
            )
            valid_counts[plant] = int(mask.sum())
            if valid_counts[plant] > 10:
                x = poa_kwm2[mask, plant] / poa_scale[plant]
                y = target_pv[mask, plant]
                weights = x
                numerator = float(np.sum(weights * x * y))
                denominator = float(np.sum(weights * x * x)) + 1e-12
                pr_proxy[plant] = numerator / denominator

        insufficient = valid_counts < 50
        if insufficient.any():
            fitted = pr_proxy[~insufficient]
            fleet_median = float(np.median(fitted)) if fitted.size else 1.0
            pr_proxy[insufficient] = fleet_median
        pr_proxy = np.clip(pr_proxy, 0.1, self.pr_max)

        self.pv_scale = pv_scale
        self.poa_scale = poa_scale.astype(np.float32)
        self.pr_proxy = pr_proxy.astype(np.float32)
        self.target_pv = target_pv.astype(np.float32)
        self.target_pv_valid = pv_observed.astype(np.float32)
        self.target_poa = poa_kwm2.astype(np.float32)
        self.valid_starts = np.arange(self.seq_len, n_steps)
        self.preprocessing_state = {
            "feature_names": list(FEATURE_NAMES),
            "include_poa_inputs": self.include_poa_inputs,
            "surface_tilt": surface_tilt,
            "surface_azimuth": surface_azimuth,
            "zscore": zscore_state,
            "pv_scale": pv_scale.tolist(),
            "poa_scale": poa_scale.tolist(),
            "pv_scale_fallback": pv_scale_fallback.tolist(),
            "poa_scale_fallback": poa_scale_fallback.tolist(),
            "pr_proxy": pr_proxy.tolist(),
            "time_grid": "hourly",
            "missing_pv_fraction": float(1.0 - pv_observed.mean()),
            "fit_start": str(times[np.flatnonzero(fit_time_mask)[0]]),
            "fit_end": str(times[np.flatnonzero(fit_time_mask)[-1]]),
        }

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int):
        target_index = self.valid_starts[idx]
        x = torch.from_numpy(
            self.feats[
                target_index - self.seq_len : target_index
            ].transpose(1, 0, 2)
        )
        y_poa = torch.from_numpy(self.target_poa[target_index])
        y_pv = torch.from_numpy(self.target_pv[target_index])
        pr_proxy = torch.from_numpy(self.pr_proxy)
        poa_cs = torch.from_numpy(self.poa_cs[target_index])
        poa_scale = torch.from_numpy(self.poa_scale)
        pv_target_valid = torch.from_numpy(
            self.target_pv_valid[target_index]
        )
        pv_lag_valid = torch.from_numpy(
            self.target_pv_valid[target_index - 1]
        )
        return (
            x,
            y_poa,
            y_pv,
            pr_proxy,
            poa_cs,
            poa_scale,
            pv_target_valid,
            pv_lag_valid,
        )
