import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

_EPS = 1e-6
_NIGHT_KW = 0.1   # pvgis_ref below this -> nighttime, QS=NaN
_GAMMA = 0.004    # IEC 61215 temperature coefficient [K⁻¹]


def compute_qs(ds: xr.Dataset, window: int = 720, eps: float = _EPS,
               debug: bool = False):
    """
    5-metric composite QS(plant, time) ∈ [0,1], geometric mean.

    Metrics (each clipped to [0,1], 1=good):
      m1 corr_score  Pearson(real, pvgis_ref) in rolling window
      m2 bias_score  1 - |mean(real-ref)| / mean(ref)
      m3 nan_score   1 - nan_fraction in window
      m4 var_score   asymmetric variance ratio (penalizes stuck sensor)
      m5 eta_score   physical consistency real/ref vs eta_base*(1-γ*(T-25))

    pvgis_ref is normalized per 1 kWp reference. Per-plant capacity scaling is
    applied automatically using the 99th percentile of daytime production, so
    ENERGIA and pvgis_ref need not share the same absolute scale.

    Nighttime (pvgis_ref < _NIGHT_KW after scaling) → NaN.
    Rolling NaN (first ~window/4 steps) → NaN (handled downstream via skipna).

    If debug=True, returns (qs_da, metrics_dict) where metrics_dict has keys
    "m1".."m5" and "capacity_scale" arrays for diagnostic use.
    """
    real     = ds["ENERGIA"].values.astype(float)      # (N, T)
    ref_raw  = ds["pvgis_ref"].values.astype(float)    # (N, T) possibly per-1kWp
    temp     = ds["temperature_2m"].values.astype(float)
    eta_base = ds["eta_base"].values.astype(float).copy()

    N, T = real.shape

    # Per-plant capacity scaling: align ref to actual plant output scale.
    # PVGIS outputs power for a 1 kWp reference system; real plants have
    # varying capacity. Scale factor = p99(real_day) / p99(ref_day).
    capacity_scale = np.ones(N)
    daytime_raw = ref_raw > _NIGHT_KW
    for p in range(N):
        mask = daytime_raw[p] & ~np.isnan(real[p]) & (real[p] > 0)
        if mask.sum() > 10:
            p99r = np.percentile(real[p][mask], 99)
            p99v = np.percentile(ref_raw[p][mask], 99)
            if p99v > eps and p99r > eps:
                capacity_scale[p] = p99r / p99v
    ref = ref_raw * capacity_scale[:, None]  # ref now in same scale as real

    # After capacity scaling, ref ≈ real/PR, so expected ratio real/ref ≈ 1.0.
    # Recompute eta_base from data as the median PR after scaling (used by m5).
    # This is robust to whatever value was stored in the dataset.
    for p in range(N):
        mask = daytime_raw[p] & ~np.isnan(real[p])
        if mask.sum() > 10:
            eta_base[p] = float(np.nanmedian(real[p][mask] / (ref[p][mask] + eps)))
    eta_base = np.clip(eta_base, 0.1, 2.0)

    eta_T = eta_base[:, None] * (1.0 - _GAMMA * (temp - 25.0))  # (N, T)

    qs_arr = np.full((N, T), np.nan)
    min_p = max(window // 4, 10)

    if debug:
        _m1 = np.full((N, T), np.nan)
        _m2 = np.full((N, T), np.nan)
        _m3 = np.full((N, T), np.nan)
        _m4 = np.full((N, T), np.nan)
        _m5 = np.full((N, T), np.nan)

    for p in range(N):
        day = daytime_raw[p]  # threshold on unscaled ref (per-1kWp units)
        # Mask both series to NaN outside daytime so rolling ops use same valid indices
        r = pd.Series(np.where(day, real[p], np.nan))
        v = pd.Series(np.where(day, ref[p], np.nan))

        # m1: Pearson
        m1 = np.clip(r.rolling(window, min_periods=min_p).corr(v).values, 0.0, 1.0)

        # m2: bias score
        diff_mean = (r - v).rolling(window, min_periods=min_p).mean().values
        v_mean    = v.rolling(window, min_periods=min_p).mean().values
        m2 = np.clip(1.0 - np.abs(diff_mean) / (np.abs(v_mean) + eps), 0.0, 1.0)

        # m3: completeness — count NaN in original real (daytime mask would inflate count)
        r_orig = pd.Series(real[p])
        m3 = np.clip(
            1.0 - r_orig.isna().rolling(window, min_periods=1).mean().values, 0.0, 1.0
        )

        # m4: variance ratio — asymmetric penalty
        # var_ratio < 1: real less variable than ref → possible stuck sensor → penalize
        # var_ratio > 1: real more variable than ref → cloud cover, natural variability → no penalty
        r_std = r.rolling(window, min_periods=min_p).std().values
        v_std = v.rolling(window, min_periods=min_p).std().values
        var_ratio = r_std / (v_std + eps)
        m4 = np.clip(var_ratio, 0.0, 1.0)

        # m5: η(T) physical consistency — rolling mean PR deviation from expected
        # After capacity scaling, expected PR = eta_base ≈ median(real/ref)
        # Only deviations BELOW expected are penalized (above = clouds, not sensor fault)
        pr_obs   = np.where(day, real[p] / (ref[p] + eps), np.nan)
        pr_ratio = pr_obs / (eta_T[p] + eps)          # should be ≈ 1.0 for healthy plant
        pr_err   = pd.Series(np.maximum(0.0, 1.0 - pr_ratio))  # only downward deviations
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
        #   true_dark  (pvgis_ref ≈ 0): no sun possible → spurious check
        #              real ≈ 0 → QS=1.0 (sensor correct)
        #              real > noise → QS=0.0 (production in darkness = fault)
        #   marginal   (0 < pvgis_ref < _NIGHT_KW): dawn/dusk with legitimate
        #              small production → QS=NaN (cannot assess fairly)
        day_peak = np.nanpercentile(real[p][day], 99) if day.sum() > 0 else 1.0
        spurious_thresh = 0.01 * (day_peak + eps)  # 1% of plant peak = noise floor
        true_dark = ref_raw[p] < eps               # pvgis_ref == 0: genuine darkness
        night_qs = np.where(
            ~true_dark,                            # marginal light → NaN
            np.nan,
            np.where(
                np.isnan(real[p]),                 # no data at dark → unknown
                np.nan,
                np.where(real[p] <= spurious_thresh,
                         1.0,                      # correct zero reading
                         0.0),                     # spurious production in darkness
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


def temporal_qs(qs: xr.DataArray, window: int = 720) -> xr.DataArray:
    """Rolling mean of QS per plant — smoothed signal for drift detection."""
    return qs.rolling(time=window, min_periods=window // 4, center=True).mean()


def temporal_qs_slope(qs: xr.DataArray) -> np.ndarray:
    """Per-plant OLS slope over full time axis. Negative → degradation."""
    n = qs.shape[0]
    t = np.arange(qs.shape[1], dtype=float)
    slopes = np.zeros(n)
    for p in range(n):
        y = qs.values[p]
        mask = ~np.isnan(y)
        if mask.sum() < 10:
            continue
        slopes[p] = stats.linregress(t[mask], y[mask]).slope
    return slopes


def spatial_qs(qs: xr.DataArray) -> xr.DataArray:
    """
    Per-timestep cross-plant z-score (plant, time).
    |z| >> 0 → isolated anomaly. All z near 0 + low fleet → regional event.
    """
    mean = qs.mean(dim="plant")
    std  = qs.std(dim="plant") + _EPS
    return (qs - mean) / std


def fleet_mean_qs(qs: xr.DataArray) -> xr.DataArray:
    """Fleet-level mean QS(t). Sharp drop → regional event."""
    return qs.mean(dim="plant")


def diagnose_scenarios(ds: xr.Dataset) -> dict[str, dict]:
    """
    Run QS diagnostics on the 4 injected synthetic fault scenarios.
    Returns dict keyed by scenario name with detection flag and metrics.
    """
    qs = compute_qs(ds)
    slopes = temporal_qs_slope(qs)
    z = spatial_qs(qs)
    fleet = fleet_mean_qs(qs)

    results: dict[str, dict] = {}

    # Scenario 1: Plant 0 gradual degradation
    results["plant0_degradation"] = {
        "slope": float(slopes[0]),
        "detected": bool(slopes[0] < -1e-7),
        "description": f"slope={slopes[0]:.2e}",
    }

    # Scenario 2: Plant 1 sudden sensor failure at t=4000
    z1_pre  = float(z.isel(plant=1, time=slice(3500, 4000)).mean(skipna=True))
    z1_post = float(z.isel(plant=1, time=slice(4100, 5000)).mean(skipna=True))
    results["plant1_sensor_failure"] = {
        "z_pre": z1_pre,
        "z_post": z1_post,
        "detected": bool(z1_post < -1.0),
        "description": f"z_pre={z1_pre:.2f} z_post={z1_post:.2f}",
    }

    # Scenario 3: Plants 2-5 regional cloud t=[6000,6500)
    fleet_pre  = float(fleet.isel(time=slice(5500, 6000)).mean(skipna=True))
    fleet_evt  = float(fleet.isel(time=slice(6100, 6400)).mean(skipna=True))
    z_spread   = float(z.isel(plant=slice(2, 6), time=slice(6100, 6400)).std(skipna=True))
    results["plants2_5_regional_cloud"] = {
        "fleet_pre": fleet_pre,
        "fleet_evt": fleet_evt,
        "z_spread_affected": z_spread,
        "detected": bool(fleet_evt < fleet_pre * 0.88 and z_spread < 0.3),
        "description": f"fleet {fleet_pre:.3f}->{fleet_evt:.3f}, z_spread={z_spread:.2f}",
    }

    # Scenario 4: Plant 6 cyclic soiling period=720h
    qs6   = qs.isel(plant=6).values
    valid = ~np.isnan(qs6)
    lag   = 720
    corr  = 0.0
    if valid.sum() > lag + 100:
        v = qs6[valid]
        if len(v) > lag:
            corr = float(np.corrcoef(v[:-lag], v[lag:])[0, 1])
    rolling_std = float(pd.Series(qs6[valid]).rolling(720, min_periods=100).std().mean())
    results["plant6_soiling_cycle"] = {
        "autocorr_720h": corr,
        "rolling_std": rolling_std,
        "detected": bool(corr > 0.25 or rolling_std > 0.02),
        "description": f"autocorr_720h={corr:.3f}, rolling_std={rolling_std:.4f}",
    }

    return results
