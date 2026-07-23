import numpy as np
import pandas as pd
import xarray as xr

_EPS = 1e-6
_NIGHT_KW = 0.05  # irradiance_ref below this (kW/m^2) -> nighttime, QS=NaN
_GAMMA = 0.004    # IEC 61215 temperature coefficient [K^-1]


def compute_qs(
    ds: xr.Dataset,
    window: int = 720,
    eps: float = _EPS,
    debug: bool = False,
    fit_time_mask: np.ndarray | None = None,
):
    """
    5-metric composite QS(plant, time) in [0,1], geometric mean.

    Metrics (each clipped to [0,1], 1=good):
      m1 corr_score  Pearson(real, irradiance_ref) in rolling window
      m2 bias_score  1 - |mean(real-ref)| / mean(ref)
      m3 nan_score   1 - nan_fraction in window
      m4 var_score   asymmetric variance ratio (penalizes stuck sensor)
      m5 eta_score   physical consistency real/ref vs eta_base*(1-gamma*(T-25))

    irradiance_ref is taken from reconstructed tilted-plane
    solar_irradiance_poa (kW/m^2).
    Per-plant capacity scaling is applied using p99 daytime values, so
    ENERGIA and irradiance_ref need not share the same absolute scale.

    Nighttime (irradiance_ref < _NIGHT_KW) -> NaN.
    Rolling NaN (first ~window/4 steps) -> NaN (handled downstream via skipna).

    Capacity and efficiency calibration use only ``fit_time_mask``; rolling
    metrics remain causal and are evaluated on the complete timeline.

    If debug=True, returns (qs_da, metrics_dict) where metrics_dict has keys
    "m1".."m5" and "capacity_scale" arrays for diagnostics.
    """
    real = ds["ENERGIA"].values.astype(float)                              # (N, T)
    ref_raw = (ds["solar_irradiance_poa"].values.astype(float) / 1000.0)  # (N, T) kW/m^2
    temp = ds["temperature_2m"].values.astype(float)
    eta_base = ds["eta_base"].values.astype(float).copy()

    N, T = real.shape
    if fit_time_mask is None:
        fit_time_mask = np.ones(T, dtype=bool)
    else:
        fit_time_mask = np.asarray(fit_time_mask, dtype=bool)
        if fit_time_mask.shape != (T,):
            raise ValueError(
                f"fit_time_mask shape {fit_time_mask.shape} != ({T},)"
            )
        if not fit_time_mask.any():
            raise ValueError("fit_time_mask must contain at least one training step")

    # Per-plant capacity scaling: align reference to actual plant output scale.
    # Scale factor = p99(real_day) / p99(ref_day).
    capacity_scale = np.ones(N)
    daytime_raw = ref_raw > _NIGHT_KW
    for p in range(N):
        mask = (
            fit_time_mask
            & daytime_raw[p]
            & ~np.isnan(real[p])
            & (real[p] > 0)
        )
        if mask.sum() > 10:
            p99r = np.percentile(real[p][mask], 99)
            p99v = np.percentile(ref_raw[p][mask], 99)
            if p99v > eps and p99r > eps:
                capacity_scale[p] = p99r / p99v
    ref = ref_raw * capacity_scale[:, None]

    # After capacity scaling, ref ~ real/PR, so expected ratio real/ref ~ 1.0.
    # Recompute eta_base from data as the median PR after scaling.
    for p in range(N):
        mask = fit_time_mask & daytime_raw[p] & ~np.isnan(real[p])
        if mask.sum() > 10:
            eta_base[p] = float(np.nanmedian(real[p][mask] / (ref[p][mask] + eps)))
    eta_base = np.clip(eta_base, 0.1, 2.0)

    eta_T = eta_base[:, None] * (1.0 - _GAMMA * (temp - 25.0))

    qs_arr = np.full((N, T), np.nan)
    min_p = max(window // 4, 10)

    if debug:
        _m1 = np.full((N, T), np.nan)
        _m2 = np.full((N, T), np.nan)
        _m3 = np.full((N, T), np.nan)
        _m4 = np.full((N, T), np.nan)
        _m5 = np.full((N, T), np.nan)

    for p in range(N):
        day = daytime_raw[p]
        # Mask both series to NaN outside daytime so rolling ops use same valid indices
        r = pd.Series(np.where(day, real[p], np.nan))
        v = pd.Series(np.where(day, ref[p], np.nan))

        # m1: Pearson
        m1 = np.clip(r.rolling(window, min_periods=min_p).corr(v).values, 0.0, 1.0)

        # m2: bias score
        diff_mean = (r - v).rolling(window, min_periods=min_p).mean().values
        v_mean = v.rolling(window, min_periods=min_p).mean().values
        m2 = np.clip(1.0 - np.abs(diff_mean) / (np.abs(v_mean) + eps), 0.0, 1.0)

        # m3: completeness from original real series
        r_orig = pd.Series(real[p])
        m3 = np.clip(
            1.0 - r_orig.isna().rolling(window, min_periods=1).mean().values, 0.0, 1.0
        )

        # m4: variance ratio (asymmetric penalty)
        r_std = r.rolling(window, min_periods=min_p).std().values
        v_std = v.rolling(window, min_periods=min_p).std().values
        var_ratio = r_std / (v_std + eps)
        m4 = np.clip(var_ratio, 0.0, 1.0)

        # m5: eta(T) physical consistency
        pr_obs = np.where(day, real[p] / (ref[p] + eps), np.nan)
        pr_ratio = pr_obs / (eta_T[p] + eps)
        pr_err = pd.Series(np.maximum(0.0, 1.0 - pr_ratio))
        eta_roll = pr_err.rolling(window, min_periods=min_p).mean().values
        m5 = np.clip(1.0 - eta_roll, 0.0, 1.0)

        qs_p = np.clip((m1 * m2 * m3 * m4 * m5) ** 0.2, 0.0, 1.0)

        if debug:
            _m1[p] = np.where(day, m1, np.nan)
            _m2[p] = np.where(day, m2, np.nan)
            _m3[p] = m3
            _m4[p] = np.where(day, m4, np.nan)
            _m5[p] = np.where(day, m5, np.nan)

        # Off-daytime hours split into two cases:
        #   true_dark  (irradiance_ref ~= 0): no sun possible -> spurious check
        #              real ~= 0 -> QS=1.0 (sensor correct)
        #              real > noise -> QS=0.0 (production in darkness = fault)
        #   marginal   (0 < irradiance_ref < _NIGHT_KW): dawn/dusk -> QS=NaN
        day_peak = np.nanpercentile(real[p][day], 99) if day.sum() > 0 else 1.0
        spurious_thresh = 0.01 * (day_peak + eps)
        true_dark = ref_raw[p] < eps
        night_qs = np.where(
            ~true_dark,
            np.nan,
            np.where(
                np.isnan(real[p]),
                np.nan,
                np.where(real[p] <= spurious_thresh, 1.0, 0.0),
            )
        )
        qs_arr[p] = np.where(day, qs_p, night_qs)

    qs_da = xr.DataArray(
        qs_arr,
        dims=["plant", "time"],
        coords={"plant": ds["plant"].values, "time": ds["time"].values},
        name="QS",
    )

    if debug:
        return qs_da, {
            "m1": _m1, "m2": _m2, "m3": _m3, "m4": _m4, "m5": _m5,
            "capacity_scale": capacity_scale,
        }
    return qs_da
