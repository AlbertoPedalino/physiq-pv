"""
Analyze long-term PV performance decline as a domain-shift signal.

This script is intentionally separate from the forecasting notebook. It does
not evaluate ST-GNN error; it evaluates the plant production process itself:

1. PVGIS-normalized performance:
      PR_pvgis(t) = actual_energy(t) / (kWp * PVGIS_POA(t))

   If PR_pvgis trends downward, the plant/fleet produces less than expected
   under comparable irradiance. This is the closest signal to soiling,
   degradation or persistent domain shift.

2. Intra-plant standardized performance:
      PR_plant_z(t) = (PR_pvgis(t) - mean(PR_pvgis for plant)) /
                      std(PR_pvgis for plant)

   This is a support metric: it expresses the same plant-level trend in
   standard deviations from that plant's own mean. The primary degradation /
   performance-loss percentage remains the monthly PR_pvgis trend.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from physiq_pv.data.load_kwp import load_kwp
from physiq_pv.data.sentinel_hourly_loader import load_sentinel_hourly, merge_with_weather


@dataclass
class TrendResult:
    metric: str
    plant: int | str
    plant_id: int | str
    n_points: int
    start: str
    end: str
    mean_value: float
    first_value: float
    last_value: float
    slope_per_month: float
    slope_per_year: float
    relative_change_pct_per_year: float
    p_value: float
    kendall_tau: float
    kendall_p_value: float
    decreasing: bool


def _normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Minimal schema alignment used by the training/notebook pipeline."""
    renames: dict[str, str] = {}
    if "latitude" in ds and "lat" not in ds:
        renames["latitude"] = "lat"
    if "longitude" in ds and "lon" not in ds:
        renames["longitude"] = "lon"
    if renames:
        ds = ds.rename(renames)
    if "eta_base" not in ds.data_vars and "eta_base" in ds.coords:
        ds = ds.assign({"eta_base": ds["eta_base"]})
    return ds


def _safe_coord(ds: xr.Dataset, name: str, fallback: np.ndarray) -> np.ndarray:
    if name in ds.coords:
        return np.asarray(ds.coords[name].values)
    if name in ds:
        return np.asarray(ds[name].values)
    return fallback


def _upn_key(value: object) -> str:
    """Normalize UPN strings so UPN_0119237_01 and UPN_119237_1 match."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().upper()
    match = re.search(r"UPN[_\s-]*(\d+)[_\s-]*(\d+)", text)
    if match:
        return f"UPN_{int(match.group(1))}_{int(match.group(2))}"
    match = re.search(r"(\d{4,})[_\s-]+(\d+)", text)
    if match:
        return f"UPN_{int(match.group(1))}_{int(match.group(2))}"
    return text


def _capacity_proxy_kwp(energy: np.ndarray, poa_kwm2: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Fallback kWp proxy: p99(energy_day) / p99(POA_day)."""
    n_plants = energy.shape[0]
    out = np.full(n_plants, np.nan, dtype=np.float64)
    for p in range(n_plants):
        m = day[p] & np.isfinite(energy[p]) & np.isfinite(poa_kwm2[p])
        e = energy[p, m]
        g = poa_kwm2[p, m]
        e = e[e > 0]
        g = g[g > 0]
        if len(e) >= 20 and len(g) >= 20:
            den = float(np.nanpercentile(g, 99))
            if den > 1e-6:
                out[p] = float(np.nanpercentile(e, 99)) / den
    return out


def _load_kwp_by_plant_dim(
    plant_mapping: Path,
    energy_coords: Path,
    plant_ids: np.ndarray,
    upns: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return kWp aligned to dataset plant dim.

    The authoritative link is UPN -> Censimp -> kWp. Falling back through the
    numeric plant_id is unsafe when the loader used positional fallback IDs for
    UPNs absent from plant_mapping.csv.
    """
    n_plants = len(plant_ids)
    if not plant_mapping.exists() or not energy_coords.exists():
        return np.full(n_plants, np.nan), np.zeros(n_plants, dtype=bool)

    pm = pd.read_csv(plant_mapping)
    ec = pd.read_csv(energy_coords)

    if upns is not None and "Codice UP" in pm.columns and "Codice Censimp Impianto" in pm.columns:
        if "Codice Censimp Impianto" in ec.columns and "Potenza di picco (kW)" in ec.columns:
            pm_u = pm[["Codice UP", "Codice Censimp Impianto"]].copy()
            pm_u["upn_key"] = pm_u["Codice UP"].map(_upn_key)
            ec_u = ec[["Codice Censimp Impianto", "Potenza di picco (kW)"]].copy()
            merged = pm_u.merge(ec_u, on="Codice Censimp Impianto", how="left")
            merged = merged[np.isfinite(pd.to_numeric(merged["Potenza di picco (kW)"], errors="coerce"))].copy()
            merged["Potenza di picco (kW)"] = pd.to_numeric(merged["Potenza di picco (kW)"], errors="coerce")
            # If duplicates exist, keep the first finite registry value for the UPN.
            by_upn = (
                merged.dropna(subset=["upn_key", "Potenza di picco (kW)"])
                .drop_duplicates("upn_key")
                .set_index("upn_key")["Potenza di picco (kW)"]
            )

            kwp = np.full(n_plants, np.nan, dtype=np.float64)
            for i, upn in enumerate(upns):
                val = by_upn.get(_upn_key(upn), np.nan)
                if pd.notna(val):
                    kwp[i] = float(val)
            return kwp, np.isfinite(kwp) & (kwp > 0)

    # Legacy fallback for old datasets without UPN coordinate.
    max_pid = int(np.nanmax(plant_ids)) if len(plant_ids) else n_plants - 1
    lookup = load_kwp(str(plant_mapping), str(energy_coords), max(max_pid + 1, n_plants))
    kwp = np.full(n_plants, np.nan, dtype=np.float64)
    for i, pid in enumerate(plant_ids):
        try:
            pid_i = int(pid)
        except Exception:
            pid_i = i
        if 0 <= pid_i < len(lookup):
            kwp[i] = lookup[pid_i]
    return kwp, np.isfinite(kwp) & (kwp > 0)


def _fit_trend(
    series: pd.Series,
    metric: str,
    plant: int | str,
    plant_id: int | str,
    min_points: int,
    alpha: float,
) -> TrendResult:
    s = series.dropna().astype(float)
    if len(s) < min_points:
        return TrendResult(
            metric=metric,
            plant=plant,
            plant_id=plant_id,
            n_points=int(len(s)),
            start="",
            end="",
            mean_value=float("nan"),
            first_value=float("nan"),
            last_value=float("nan"),
            slope_per_month=float("nan"),
            slope_per_year=float("nan"),
            relative_change_pct_per_year=float("nan"),
            p_value=float("nan"),
            kendall_tau=float("nan"),
            kendall_p_value=float("nan"),
            decreasing=False,
        )

    idx = pd.DatetimeIndex(s.index)
    x_months = (idx - idx[0]).days.to_numpy(dtype=float) / 30.4375
    y = s.to_numpy(dtype=float)
    lin = stats.linregress(x_months, y)
    tau, tau_p = stats.kendalltau(x_months, y)
    mean_y = float(np.nanmean(y))
    slope_year = float(lin.slope * 12.0)
    rel_year = (
        float("nan")
        if "_z_" in metric or metric.endswith("_z_monthly") or abs(mean_y) <= 1e-12
        else float((slope_year / mean_y) * 100.0)
    )
    return TrendResult(
        metric=metric,
        plant=plant,
        plant_id=plant_id,
        n_points=int(len(s)),
        start=str(idx[0].date()),
        end=str(idx[-1].date()),
        mean_value=mean_y,
        first_value=float(y[0]),
        last_value=float(y[-1]),
        slope_per_month=float(lin.slope),
        slope_per_year=slope_year,
        relative_change_pct_per_year=rel_year,
        p_value=float(lin.pvalue),
        kendall_tau=float(tau),
        kendall_p_value=float(tau_p),
        decreasing=bool(lin.slope < 0 and lin.pvalue < alpha),
    )


def _resample_ratio(
    actual: pd.Series,
    expected: pd.Series,
    valid: pd.Series,
    freq: str,
    min_hours: int,
) -> pd.DataFrame:
    actual_sum = actual.where(valid).resample(freq).sum(min_count=min_hours)
    expected_sum = expected.where(valid).resample(freq).sum(min_count=min_hours)
    valid_hours = valid.astype(int).resample(freq).sum()
    pr = actual_sum / expected_sum.replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "actual_kwh": actual_sum,
            "pvgis_expected_kwh": expected_sum,
            "valid_hours": valid_hours,
            "pr_pvgis": pr,
        }
    )


def _build_performance_tables(
    ds: xr.Dataset,
    kwp: np.ndarray,
    kwp_is_real: np.ndarray,
    kwp_mode: str,
    daytime_poa_threshold: float,
    reference_pr: float,
    min_day_hours: int,
    min_month_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = pd.DatetimeIndex(ds.coords["time"].values)
    plant_ids = _safe_coord(ds, "plant_id", np.arange(ds.sizes["plant"]))
    upns = _safe_coord(ds, "upn", np.array([""] * ds.sizes["plant"], dtype=object))
    energy = np.asarray(ds["ENERGIA"].values, dtype=np.float64)
    poa_wm2 = np.asarray(ds["solar_irradiance_poa"].values, dtype=np.float64)
    poa_kwm2 = np.clip(poa_wm2 / 1000.0, 0.0, None)
    day = poa_wm2 >= daytime_poa_threshold

    proxy = _capacity_proxy_kwp(energy, poa_kwm2, day)
    capacity = kwp.copy()
    missing = ~np.isfinite(capacity) | (capacity <= 0)
    if kwp_mode == "real-or-proxy":
        capacity[missing] = proxy[missing]
    elif kwp_mode != "real-only":
        raise ValueError(f"Unsupported kwp_mode: {kwp_mode}")

    daily_rows: list[pd.DataFrame] = []
    monthly_rows: list[pd.DataFrame] = []
    for p in range(ds.sizes["plant"]):
        if kwp_mode == "real-only" and not kwp_is_real[p]:
            continue
        if not np.isfinite(capacity[p]) or capacity[p] <= 0:
            continue
        actual = pd.Series(energy[p], index=times, dtype="float64")
        expected = pd.Series(poa_kwm2[p] * capacity[p] * reference_pr, index=times, dtype="float64")
        valid = pd.Series(
            day[p]
            & np.isfinite(energy[p])
            & np.isfinite(poa_kwm2[p])
            & (energy[p] >= 0)
            & (expected.to_numpy() > 1e-9),
            index=times,
        )

        daily = _resample_ratio(actual, expected, valid, "1D", min_day_hours)
        monthly = _resample_ratio(actual, expected, valid, "ME", min_month_hours)
        for frame in (daily, monthly):
            frame.insert(0, "plant", p)
            frame.insert(1, "plant_id", plant_ids[p])
            frame.insert(2, "upn", str(upns[p]) if p < len(upns) else "")
            frame.insert(3, "kwp_used", capacity[p])
            frame.insert(4, "kwp_source", "real" if kwp_is_real[p] else "proxy")
            frame.insert(5, "date", frame.index)
            frame.insert(6, "calendar_month", frame.index.month)
        daily_rows.append(daily.reset_index(drop=True))
        monthly_rows.append(monthly.reset_index(drop=True))

    if not daily_rows or not monthly_rows:
        raise RuntimeError("No valid plant performance rows were produced.")

    daily_df = pd.concat(daily_rows, ignore_index=True)
    monthly_df = pd.concat(monthly_rows, ignore_index=True)
    return daily_df, monthly_df


def _filter_plausible_pr(
    daily_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    pr_min: float,
    pr_max: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop plants whose mean monthly PR is outside a physically plausible range."""
    monthly = monthly_df.copy()
    monthly["pr_pvgis"] = monthly["pr_pvgis"].replace([np.inf, -np.inf], np.nan)

    stats_df = (
        monthly.groupby(["plant", "plant_id"], as_index=False)
        .agg(
            mean_pr_pvgis=("pr_pvgis", "mean"),
            median_pr_pvgis=("pr_pvgis", "median"),
            min_pr_pvgis=("pr_pvgis", "min"),
            max_pr_pvgis=("pr_pvgis", "max"),
            n_valid_months=("pr_pvgis", "count"),
            kwp_used=("kwp_used", "first"),
            kwp_source=("kwp_source", "first"),
        )
    )
    stats_df["plausible_pr"] = (
        stats_df["mean_pr_pvgis"].between(pr_min, pr_max, inclusive="both")
        & (stats_df["n_valid_months"] > 0)
    )

    keep_plants = set(stats_df.loc[stats_df["plausible_pr"], "plant"].tolist())
    daily_f = daily_df[daily_df["plant"].isin(keep_plants)].copy()
    monthly_f = monthly_df[monthly_df["plant"].isin(keep_plants)].copy()
    excluded = stats_df[~stats_df["plausible_pr"]].copy().sort_values("mean_pr_pvgis")
    return daily_f, monthly_f, excluded


def _add_plant_standardization(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Add a within-plant z-score for PR_PVGIS as a support diagnostic."""
    monthly = monthly_df.copy()
    monthly["pr_pvgis"] = monthly["pr_pvgis"].replace([np.inf, -np.inf], np.nan)
    plant_stats = (
        monthly.groupby(["plant", "plant_id"], as_index=False)
        .agg(
            plant_pr_mean=("pr_pvgis", "mean"),
            plant_pr_std=("pr_pvgis", "std"),
            plant_pr_n=("pr_pvgis", "count"),
        )
    )
    monthly = monthly.merge(plant_stats, on=["plant", "plant_id"], how="left")
    monthly["pr_plant_z"] = (
        (monthly["pr_pvgis"] - monthly["plant_pr_mean"])
        / monthly["plant_pr_std"].replace(0.0, np.nan)
    )
    return monthly


def _mapping_details(plant_mapping: Path, energy_coords: Path) -> pd.DataFrame:
    if not plant_mapping.exists():
        return pd.DataFrame()

    pm = pd.read_csv(plant_mapping)
    if "plant_id" not in pm.columns:
        return pd.DataFrame()

    keep_pm = [
        c for c in (
            "plant_id",
            "Codice UP",
            "Codice Censimp Impianto",
            "Latitude",
            "Longitude",
            "eta_base",
        )
        if c in pm.columns
    ]
    details = pm[keep_pm].copy()
    if "Codice UP" in details.columns:
        details["upn_key"] = details["Codice UP"].map(_upn_key)

    if energy_coords.exists() and "Codice Censimp Impianto" in details.columns:
        ec = pd.read_csv(energy_coords)
        keep_ec = [
            c for c in (
                "Codice Censimp Impianto",
                "Potenza di picco (kW)",
                "Latitude",
                "Longitude",
            )
            if c in ec.columns
        ]
        if "Codice Censimp Impianto" in keep_ec:
            details = details.merge(
                ec[keep_ec].drop_duplicates("Codice Censimp Impianto"),
                on="Codice Censimp Impianto",
                how="left",
                suffixes=("_mapping", "_registry"),
            )

    if "Codice UP" in details.columns:
        return details.drop_duplicates("upn_key")
    return details.drop_duplicates("plant_id")


def _diagnose_implausible_pr(
    ds: xr.Dataset,
    kwp: np.ndarray,
    kwp_is_real: np.ndarray,
    excluded_pr: pd.DataFrame,
    daytime_poa_threshold: float,
    plausible_pr_min: float,
    plausible_pr_max: float,
    plant_mapping: Path,
    energy_coords: Path,
) -> pd.DataFrame:
    """Explain impossible PR values by exposing numerator/denominator components."""
    diag_columns = [
        "plant",
        "plant_id",
        "upn",
        "upn_key",
        "lat",
        "lon",
        "kwp_used",
        "kwp_source",
        "mean_pr_pvgis",
        "median_pr_pvgis",
        "n_valid_months",
        "n_day_hours",
        "period_start",
        "period_end",
        "actual_sum_kwh",
        "expected_sum_kwh_pr1",
        "actual_over_expected_sum",
        "energy_p50",
        "energy_p95",
        "energy_p99",
        "energy_max",
        "poa_kwm2_p50",
        "poa_kwm2_p95",
        "poa_kwm2_p99",
        "poa_kwm2_max",
        "energy_p99_over_kwp",
        "energy_max_over_kwp",
        "suggested_energy_multiplier",
        "mean_pr_after_suggested_multiplier",
        "diagnostic_flags",
    ]
    if excluded_pr.empty:
        return pd.DataFrame(columns=diag_columns)

    times = pd.DatetimeIndex(ds.coords["time"].values)
    plant_ids = _safe_coord(ds, "plant_id", np.arange(ds.sizes["plant"]))
    upns = _safe_coord(ds, "upn", np.array([""] * ds.sizes["plant"], dtype=object))
    lat = _safe_coord(ds, "lat", np.full(ds.sizes["plant"], np.nan))
    lon = _safe_coord(ds, "lon", np.full(ds.sizes["plant"], np.nan))
    energy = np.asarray(ds["ENERGIA"].values, dtype=np.float64)
    poa_wm2 = np.asarray(ds["solar_irradiance_poa"].values, dtype=np.float64)
    poa_kwm2 = np.clip(poa_wm2 / 1000.0, 0.0, None)
    mapping = _mapping_details(plant_mapping, energy_coords)

    def suggested_scale(mean_pr: float) -> tuple[float, float]:
        if not np.isfinite(mean_pr) or mean_pr <= 0:
            return float("nan"), float("nan")
        target = 0.8
        candidates = np.array([10.0 ** k for k in range(-6, 7)], dtype=float)
        scaled = mean_pr * candidates
        plausible = (scaled >= plausible_pr_min) & (scaled <= plausible_pr_max)
        if plausible.any():
            idx = np.where(plausible)[0][np.argmin(np.abs(scaled[plausible] - target))]
        else:
            mid = (plausible_pr_min + plausible_pr_max) / 2.0
            idx = int(np.argmin(np.abs(scaled - mid)))
        return float(candidates[idx]), float(scaled[idx])

    rows: list[dict] = []
    for _, ex in excluded_pr.iterrows():
        p = int(ex["plant"])
        day = (
            poa_wm2[p] >= daytime_poa_threshold
        ) & np.isfinite(energy[p]) & np.isfinite(poa_kwm2[p]) & (energy[p] >= 0)
        e = energy[p, day]
        g = poa_kwm2[p, day]
        cap = float(kwp[p]) if p < len(kwp) else float("nan")
        expected = g * cap if np.isfinite(cap) and cap > 0 else np.full_like(g, np.nan)

        def pct(arr: np.ndarray, q: float) -> float:
            arr = arr[np.isfinite(arr)]
            return float(np.nanpercentile(arr, q)) if len(arr) else float("nan")

        actual_sum = float(np.nansum(e)) if len(e) else float("nan")
        expected_sum = float(np.nansum(expected)) if len(expected) else float("nan")
        p99_over_kwp = pct(e, 99) / cap if np.isfinite(cap) and cap > 0 else float("nan")
        max_over_kwp = float(np.nanmax(e) / cap) if len(e) and np.isfinite(cap) and cap > 0 else float("nan")
        poa_p99 = pct(g, 99)

        flags: list[str] = []
        if np.isfinite(max_over_kwp) and max_over_kwp > 1.5:
            flags.append("ENERGIA_peak_exceeds_kWp")
        if np.isfinite(p99_over_kwp) and p99_over_kwp > 1.2:
            flags.append("ENERGIA_p99_exceeds_kWp")
        if np.isfinite(poa_p99) and poa_p99 < 0.4:
            flags.append("PVGIS_POA_too_low_or_mismatched")
        if np.isfinite(cap) and cap <= 0:
            flags.append("invalid_kWp")
        if not kwp_is_real[p]:
            flags.append("proxy_kWp")
        if not flags:
            flags.append("inspect_mapping_or_units")
        scale_factor, mean_after_scale = suggested_scale(float(ex["mean_pr_pvgis"]))

        rows.append(
            {
                "plant": p,
                "plant_id": plant_ids[p],
                "upn": str(upns[p]) if p < len(upns) else "",
                "upn_key": _upn_key(upns[p]) if p < len(upns) else "",
                "lat": float(lat[p]) if p < len(lat) and np.isfinite(lat[p]) else np.nan,
                "lon": float(lon[p]) if p < len(lon) and np.isfinite(lon[p]) else np.nan,
                "kwp_used": cap,
                "kwp_source": "real" if kwp_is_real[p] else "proxy",
                "mean_pr_pvgis": float(ex["mean_pr_pvgis"]),
                "median_pr_pvgis": float(ex["median_pr_pvgis"]),
                "n_valid_months": int(ex["n_valid_months"]),
                "n_day_hours": int(day.sum()),
                "period_start": str(times[day][0]) if day.any() else "",
                "period_end": str(times[day][-1]) if day.any() else "",
                "actual_sum_kwh": actual_sum,
                "expected_sum_kwh_pr1": expected_sum,
                "actual_over_expected_sum": actual_sum / expected_sum if expected_sum > 0 else np.nan,
                "energy_p50": pct(e, 50),
                "energy_p95": pct(e, 95),
                "energy_p99": pct(e, 99),
                "energy_max": float(np.nanmax(e)) if len(e) else np.nan,
                "poa_kwm2_p50": pct(g, 50),
                "poa_kwm2_p95": pct(g, 95),
                "poa_kwm2_p99": poa_p99,
                "poa_kwm2_max": float(np.nanmax(g)) if len(g) else np.nan,
                "energy_p99_over_kwp": p99_over_kwp,
                "energy_max_over_kwp": max_over_kwp,
                "suggested_energy_multiplier": scale_factor,
                "mean_pr_after_suggested_multiplier": mean_after_scale,
                "diagnostic_flags": "|".join(flags),
            }
        )

    diag = pd.DataFrame(rows, columns=diag_columns)
    if not mapping.empty:
        if "upn_key" in diag.columns and "upn_key" in mapping.columns:
            diag = diag.merge(mapping, on="upn_key", how="left")
        else:
            diag = diag.merge(mapping, on="plant_id", how="left")
    return diag.sort_values("mean_pr_pvgis", ascending=False)


def _fleet_tables(daily_df: pd.DataFrame, monthly_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = daily_df.copy()
    monthly = monthly_df.copy()

    def _sum_valid(s: pd.Series) -> float:
        return float(s.sum(min_count=1))

    fleet_daily = (
        daily.groupby("date", as_index=False)
        .agg(
            actual_kwh=("actual_kwh", _sum_valid),
            pvgis_expected_kwh=("pvgis_expected_kwh", _sum_valid),
            n_plants=("pr_pvgis", lambda s: int(s.notna().sum())),
            median_pr_pvgis=("pr_pvgis", "median"),
        )
    )
    fleet_daily["weighted_pr_pvgis"] = fleet_daily["actual_kwh"] / fleet_daily["pvgis_expected_kwh"].replace(0.0, np.nan)

    fleet_monthly = (
        monthly.groupby("date", as_index=False)
        .agg(
            actual_kwh=("actual_kwh", _sum_valid),
            pvgis_expected_kwh=("pvgis_expected_kwh", _sum_valid),
            n_plants=("pr_pvgis", lambda s: int(s.notna().sum())),
            median_pr_pvgis=("pr_pvgis", "median"),
        )
    )
    fleet_monthly["weighted_pr_pvgis"] = fleet_monthly["actual_kwh"] / fleet_monthly["pvgis_expected_kwh"].replace(0.0, np.nan)
    return fleet_daily, fleet_monthly


def _trend_tables(
    monthly_df: pd.DataFrame,
    fleet_monthly: pd.DataFrame,
    min_months: int,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    plant_results: list[TrendResult] = []
    for (plant, plant_id), g in monthly_df.groupby(["plant", "plant_id"]):
        g = g.sort_values("date")
        s_pr = pd.Series(g["pr_pvgis"].to_numpy(dtype=float), index=pd.to_datetime(g["date"]))
        plant_results.append(_fit_trend(s_pr, "pr_pvgis_monthly", plant, plant_id, min_months, alpha))
        if "pr_plant_z" in g.columns:
            s_z = pd.Series(g["pr_plant_z"].to_numpy(dtype=float), index=pd.to_datetime(g["date"]))
            plant_results.append(_fit_trend(s_z, "pr_plant_z_monthly", plant, plant_id, min_months, alpha))

    fleet_results = [
        _fit_trend(
            pd.Series(fleet_monthly["weighted_pr_pvgis"].to_numpy(dtype=float), index=pd.to_datetime(fleet_monthly["date"])),
            "fleet_weighted_pr_pvgis_monthly",
            "fleet",
            "fleet",
            min_months,
            alpha,
        ),
        _fit_trend(
            pd.Series(fleet_monthly["median_pr_pvgis"].to_numpy(dtype=float), index=pd.to_datetime(fleet_monthly["date"])),
            "fleet_median_pr_pvgis_monthly",
            "fleet",
            "fleet",
            min_months,
            alpha,
        ),
    ]
    return (
        pd.DataFrame([asdict(r) for r in plant_results]),
        pd.DataFrame([asdict(r) for r in fleet_results]),
    )


def _plant_candidate_summary(
    monthly_df: pd.DataFrame,
    plant_trends: pd.DataFrame,
) -> pd.DataFrame:
    """One row per plant, using PR_PVGIS as primary and plant-z as support."""
    pr = plant_trends[plant_trends["metric"] == "pr_pvgis_monthly"].copy()
    plant_z = plant_trends[plant_trends["metric"] == "pr_plant_z_monthly"].copy()
    pr_cols = {
        "n_points": "pr_n_points",
        "mean_value": "pr_mean",
        "first_value": "pr_first",
        "last_value": "pr_last",
        "slope_per_year": "pr_slope_per_year",
        "relative_change_pct_per_year": "pr_relative_change_pct_per_year",
        "p_value": "pr_p_value",
        "kendall_tau": "pr_kendall_tau",
        "kendall_p_value": "pr_kendall_p_value",
        "decreasing": "pr_decreasing",
    }
    pr = pr[["plant", "plant_id", *pr_cols.keys()]].rename(columns=pr_cols)
    plant_z_cols = {
        "n_points": "plant_z_n_points",
        "mean_value": "plant_z_mean",
        "first_value": "plant_z_first",
        "last_value": "plant_z_last",
        "slope_per_year": "plant_z_slope_per_year",
        "p_value": "plant_z_p_value",
        "kendall_tau": "plant_z_kendall_tau",
        "kendall_p_value": "plant_z_kendall_p_value",
        "decreasing": "plant_z_decreasing",
    }
    plant_z = plant_z[["plant", "plant_id", *plant_z_cols.keys()]].rename(columns=plant_z_cols)

    meta = (
        monthly_df.sort_values("date")
        .groupby(["plant", "plant_id"], as_index=False)
        .agg(
            upn=("upn", "first") if "upn" in monthly_df.columns else ("plant_id", "first"),
            kwp_used=("kwp_used", "first"),
            kwp_source=("kwp_source", "first"),
            valid_months=("pr_pvgis", "count"),
            min_pr=("pr_pvgis", "min"),
            max_pr=("pr_pvgis", "max"),
            median_pr=("pr_pvgis", "median"),
            plant_pr_mean=("plant_pr_mean", "first") if "plant_pr_mean" in monthly_df.columns else ("pr_pvgis", "mean"),
            plant_pr_std=("plant_pr_std", "first") if "plant_pr_std" in monthly_df.columns else ("pr_pvgis", "std"),
            first_month=("date", "first"),
            last_month=("date", "last"),
        )
    )
    out = (
        meta
        .merge(pr, on=["plant", "plant_id"], how="left")
        .merge(plant_z, on=["plant", "plant_id"], how="left")
    )

    out["first_to_last_pct"] = np.where(
        np.abs(out["pr_first"]) > 1e-12,
        (out["pr_last"] - out["pr_first"]) / out["pr_first"] * 100.0,
        np.nan,
    )
    out["peak_to_last_pct"] = np.where(
        np.abs(out["max_pr"]) > 1e-12,
        (out["pr_last"] - out["max_pr"]) / out["max_pr"] * 100.0,
        np.nan,
    )
    out["pr_decreasing"] = out["pr_decreasing"].fillna(False).astype(bool)
    out["plant_z_decreasing"] = out["plant_z_decreasing"].fillna(False).astype(bool)

    conditions = [out["pr_decreasing"]]
    choices = ["performance_decline"]
    out["candidate_class"] = np.select(conditions, choices, default="no_significant_decline")
    out["candidate_rank_score"] = (
        out["pr_decreasing"].astype(int)
        + np.clip(-out["pr_slope_per_year"].fillna(0.0), 0.0, None)
    )
    out["local_decline_flag"] = out["candidate_class"] != "no_significant_decline"
    out["statistical_note"] = np.where(
        out["valid_months"] < 12,
        "exploratory: single incomplete year",
        "multi-month trend",
    )
    return out.sort_values(
        ["candidate_rank_score", "pr_decreasing", "pr_slope_per_year"],
        ascending=[False, False, True],
    )


# ---------------------------------------------------------------------------
# Full distribution / classification of per-plant trends
# ---------------------------------------------------------------------------

DEFAULT_DECLINE_BIN_EDGES: tuple[float, ...] = (-10.0, -5.0, -3.0, -2.0, -1.0, 0.0)
DEFAULT_DECLINE_BIN_LABELS: tuple[str, ...] = (
    "strong_decline_gt_10",
    "relevant_decline_5_10",
    "moderate_decline_3_5",
    "mild_decline_2_3",
    "weak_decline_1_2",
    "very_weak_decline_0_1",
    "stable_or_positive",
)


def _classify_decline_bin(
    rel_pct_per_year: float,
    edges: tuple[float, ...] = DEFAULT_DECLINE_BIN_EDGES,
    labels: tuple[str, ...] = DEFAULT_DECLINE_BIN_LABELS,
) -> str:
    """Map relative_change_pct_per_year to a discrete decline class.

    edges ascending, e.g. (-10, -5, -3, -2, -1, 0). Semantics:
      rel <  edges[0]              -> labels[0]   (e.g. < -10%  -> strong)
      edges[i] <= rel < edges[i+1] -> labels[i+1]
      rel >= edges[-1]             -> labels[-1]  (>= 0%  -> stable_or_positive)
    """
    if rel_pct_per_year is None or not np.isfinite(rel_pct_per_year):
        return "undefined"
    if len(labels) != len(edges) + 1:
        raise ValueError("decline labels must equal edges + 1")
    if rel_pct_per_year < edges[0]:
        return labels[0]
    for i in range(len(edges) - 1):
        if edges[i] <= rel_pct_per_year < edges[i + 1]:
            return labels[i + 1]
    if rel_pct_per_year >= edges[-1]:
        return labels[-1]
    return "undefined"


def _classify_monotonic(
    tau: float,
    strong_threshold: float = -0.85,
    directional_threshold: float = -0.60,
) -> str:
    if tau is None or not np.isfinite(tau):
        return "undefined"
    if tau <= strong_threshold:
        return "strictly_or_nearly_monotonic_decline"
    if tau <= directional_threshold:
        return "directional_decline"
    return "weak_or_no_monotonic_decline"


def _build_all_plant_trend_table(
    candidate_summary: pd.DataFrame,
    decline_edges: tuple[float, ...],
    decline_labels: tuple[str, ...],
    monotonic_strong: float,
    monotonic_directional: float,
    alpha: float,
) -> pd.DataFrame:
    """Full per-plant trend table for every plant that produced a PR fit.

    Built on top of candidate_summary (already one row per plant). Adds
    relative_change_pct_per_year, decline_class, monotonic_class and the
    significant_decline flag. No regression logic is duplicated.
    """
    if candidate_summary.empty:
        return candidate_summary.copy()

    df = candidate_summary.copy()
    pr_mean = df["plant_pr_mean"].to_numpy(dtype=float)
    slope_year = df["pr_slope_per_year"].to_numpy(dtype=float)
    rel = np.where(
        np.abs(pr_mean) > 1e-12,
        slope_year / pr_mean * 100.0,
        np.nan,
    )
    df["relative_change_pct_per_year"] = rel
    df["slope_per_year"] = slope_year
    df["slope_per_month"] = slope_year / 12.0
    df["p_value"] = df["pr_p_value"]
    df["kendall_tau"] = df["pr_kendall_tau"]
    df["kendall_p_value"] = df["pr_kendall_p_value"]
    df["decreasing"] = df["pr_decreasing"]
    df["pr_mean"] = pr_mean

    df["decline_class"] = [
        _classify_decline_bin(v, decline_edges, decline_labels) for v in rel
    ]
    df["monotonic_class"] = [
        _classify_monotonic(t, monotonic_strong, monotonic_directional)
        for t in df["kendall_tau"].to_numpy(dtype=float)
    ]
    df["significant_decline"] = (
        (df["p_value"].to_numpy(dtype=float) < alpha)
        & (df["slope_per_year"].to_numpy(dtype=float) < 0)
    )

    cols_order = [
        "plant",
        "plant_id",
        "upn",
        "kwp_used",
        "kwp_source",
        "valid_months",
        "first_month",
        "last_month",
        "pr_mean",
        "pr_first",
        "pr_last",
        "median_pr",
        "min_pr",
        "max_pr",
        "first_to_last_pct",
        "peak_to_last_pct",
        "slope_per_month",
        "slope_per_year",
        "relative_change_pct_per_year",
        "p_value",
        "kendall_tau",
        "kendall_p_value",
        "decreasing",
        "significant_decline",
        "decline_class",
        "monotonic_class",
        "candidate_class",
        "local_decline_flag",
        "statistical_note",
    ]
    present = [c for c in cols_order if c in df.columns]
    extras = [c for c in df.columns if c not in present]
    return df[present + extras].sort_values(
        "relative_change_pct_per_year", ascending=True, na_position="last"
    )


def _build_decline_distribution_summary(
    all_plants: pd.DataFrame,
    decline_labels: tuple[str, ...],
    monotonic_strong: float,
    monotonic_directional: float,
    alpha: float,
) -> dict:
    n_total = int(len(all_plants))
    decline_counts: dict[str, int] = {lbl: 0 for lbl in decline_labels}
    decline_counts["undefined"] = 0
    for lbl, cnt in all_plants["decline_class"].value_counts(dropna=False).items():
        decline_counts[str(lbl)] = int(cnt)
    decline_pct = {
        k: (float(v) / n_total * 100.0 if n_total else 0.0)
        for k, v in decline_counts.items()
    }

    mono_counts: dict[str, int] = {
        "strictly_or_nearly_monotonic_decline": 0,
        "directional_decline": 0,
        "weak_or_no_monotonic_decline": 0,
        "undefined": 0,
    }
    for lbl, cnt in all_plants["monotonic_class"].value_counts(dropna=False).items():
        mono_counts[str(lbl)] = int(cnt)
    mono_pct = {
        k: (float(v) / n_total * 100.0 if n_total else 0.0)
        for k, v in mono_counts.items()
    }

    n_significant = int(all_plants["significant_decline"].sum())
    n_decreasing_any = int(all_plants["decreasing"].sum())
    rel = all_plants["relative_change_pct_per_year"].to_numpy(dtype=float)
    rel_finite = rel[np.isfinite(rel)]

    quantiles: dict[str, float] = {}
    if rel_finite.size:
        for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
            quantiles[f"p{int(q * 100):02d}"] = float(np.quantile(rel_finite, q))

    return {
        "n_plants_analyzed": n_total,
        "alpha": alpha,
        "monotonic_strong_tau_threshold": monotonic_strong,
        "monotonic_directional_tau_threshold": monotonic_directional,
        "decline_class_counts": decline_counts,
        "decline_class_pct": decline_pct,
        "monotonic_class_counts": mono_counts,
        "monotonic_class_pct": mono_pct,
        "n_significant_p_lt_alpha_and_negative_slope": n_significant,
        "pct_significant_p_lt_alpha_and_negative_slope": (
            n_significant / n_total * 100.0 if n_total else 0.0
        ),
        "n_decreasing_pr_pvgis": n_decreasing_any,
        "relative_change_pct_per_year_quantiles": quantiles,
        "interpretation_caveat": (
            "Weak / mild decline counts (e.g. -1%/yr to -3%/yr) are computed on "
            "a single incomplete year (Mar-Dec 2019) with ~7-8 valid monthly "
            "points per plant. With this little data, a downward slope of "
            "2-3%/yr cannot be cleanly separated from residual seasonality, "
            "soiling, availability losses, curtailment, clipping, or upstream "
            "data issues. Rows in the weak / mild bins should be read as "
            "'apparent weak / mild performance loss', not as physical module "
            "degradation. Large drops (> 5-10%/yr) are likely operational "
            "anomalies or data problems, not physiological degradation."
        ),
    }


def _plot_decline_histogram(
    out_dir: Path,
    all_plants: pd.DataFrame,
    decline_edges: tuple[float, ...],
) -> None:
    rel = all_plants["relative_change_pct_per_year"].to_numpy(dtype=float)
    rel = rel[np.isfinite(rel)]
    if rel.size == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bins = np.linspace(
        min(float(rel.min()), -12.0),
        max(float(rel.max()), 5.0),
        40,
    )
    ax.hist(rel, bins=bins, color="tab:blue", edgecolor="white", alpha=0.85)
    for e in decline_edges:
        ax.axvline(e, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("relative_change_pct_per_year (PR_PVGIS)")
    ax.set_ylabel("plant count")
    ax.set_title(
        f"Distribution of per-plant PR_PVGIS relative trend  (n={rel.size})"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "all_plant_relative_change_histogram.png", dpi=180)
    plt.close(fig)


def _plot_outputs(
    out_dir: Path,
    fleet_monthly: pd.DataFrame,
    plant_trends: pd.DataFrame,
    monthly_df: pd.DataFrame,
    top_k: int,
) -> None:
    plt.style.use("seaborn-v0_8-darkgrid")

    fm = fleet_monthly.sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(fm["date"]), fm["weighted_pr_pvgis"], "o-", label="weighted fleet PR")
    ax.plot(pd.to_datetime(fm["date"]), fm["median_pr_pvgis"], "s--", label="median plant PR")
    ax.set_title("Fleet PVGIS-normalized performance")
    ax.set_xlabel("Month")
    ax.set_ylabel("actual / (kWp * PVGIS POA)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fleet_pvgis_pr_trend.png", dpi=180)
    plt.close(fig)

    pr_trends = plant_trends[plant_trends["metric"] == "pr_pvgis_monthly"].copy()
    pr_trends = pr_trends.dropna(subset=["slope_per_month"]).sort_values("slope_per_month")
    if pr_trends.empty:
        return

    top = pr_trends.head(top_k)
    md = monthly_df.copy()
    md["date"] = pd.to_datetime(md["date"])
    fig, ax = plt.subplots(figsize=(11, 5))
    for _, row in top.iterrows():
        g = md[md["plant"] == row["plant"]].sort_values("date")
        ax.plot(g["date"], g["pr_pvgis"], marker="o", linewidth=1.5, label=f"plant {row['plant']}")
    ax.set_title(f"Top {len(top)} decreasing plants by PVGIS-normalized PR")
    ax.set_xlabel("Month")
    ax.set_ylabel("PR_pvgis")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "top_decreasing_plants_pvgis_pr.png", dpi=180)
    plt.close(fig)


def _plot_individual_candidates(
    out_dir: Path,
    monthly_df: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    top_k: int,
) -> None:
    candidates = candidate_summary[candidate_summary["local_decline_flag"]].copy()
    if candidates.empty:
        candidates = candidate_summary.sort_values("pr_slope_per_year").head(top_k).copy()
    else:
        candidates = candidates.head(top_k).copy()
    if candidates.empty:
        return

    n = len(candidates)
    n_cols = 2
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, max(3.2 * n_rows, 4)), squeeze=False)
    md = monthly_df.copy()
    md["date"] = pd.to_datetime(md["date"])

    for ax, (_, row) in zip(axes.ravel(), candidates.iterrows()):
        g = md[md["plant"] == row["plant"]].sort_values("date")
        ax.plot(g["date"], g["pr_pvgis"], "o-", color="tab:blue", label="PR PVGIS")
        if "pr_plant_z" in g.columns:
            ax2 = ax.twinx()
            ax2.plot(g["date"], g["pr_plant_z"], "s--", color="tab:orange", alpha=0.65, label="plant-z")
            ax2.axhline(0.0, color="tab:orange", linewidth=0.8, alpha=0.4)
            ax2.set_ylabel("PR plant-z")
        ax.set_title(
            f"plant {row['plant']} / id {row['plant_id']}  {row['candidate_class']}\n"
            f"PR slope/y={row['pr_slope_per_year']:+.3f}, p={row['pr_p_value']:.3g}"
        )
        ax.set_ylabel("PR PVGIS")
        ax.grid(True, alpha=0.25)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "individual_candidate_plant_trends.png", dpi=180)
    plt.close(fig)


def _write_summary(
    out_dir: Path,
    args: argparse.Namespace,
    ds: xr.Dataset,
    kwp_is_real: np.ndarray,
    excluded_pr: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    plant_trends: pd.DataFrame,
    fleet_trends: pd.DataFrame,
    fleet_monthly: pd.DataFrame,
    decline_distribution: dict | None = None,
) -> None:
    pr = plant_trends[plant_trends["metric"] == "pr_pvgis_monthly"]
    plant_z = plant_trends[plant_trends["metric"] == "pr_plant_z_monthly"]
    summary = {
        "period_start": str(pd.Timestamp(ds.time.values[0]).date()),
        "period_end": str(pd.Timestamp(ds.time.values[-1]).date()),
        "n_plants": int(ds.sizes["plant"]),
        "n_hours": int(ds.sizes["time"]),
        "n_real_kwp": int(np.sum(kwp_is_real)),
        "n_proxy_kwp": int(len(kwp_is_real) - np.sum(kwp_is_real)),
        "daytime_poa_threshold_wm2": args.daytime_poa_threshold,
        "reference_pr": args.reference_pr,
        "kwp_mode": args.kwp_mode,
        "plausible_pr_min": args.plausible_pr_min,
        "plausible_pr_max": args.plausible_pr_max,
        "n_excluded_implausible_pr": int(len(excluded_pr)),
        "min_months": args.min_months,
        "alpha": args.alpha,
        "fleet_trends": fleet_trends.to_dict(orient="records"),
        "n_decreasing_plants_pvgis_pr": int(pr["decreasing"].sum()),
        "n_decreasing_plants_plant_z": int(plant_z["decreasing"].sum()),
        "n_performance_decline_candidates": int(
            (candidate_summary["candidate_class"] == "performance_decline").sum()
        ) if not candidate_summary.empty else 0,
        "fleet_monthly_rows": int(len(fleet_monthly)),
        "caveat": (
            "PR_PVGIS trends estimate apparent performance loss. PR_plant_z "
            "standardizes each plant by its own mean and standard deviation, "
            "but it is not a true calendar-month seasonal correction with a "
            "single-year window. Trends can include module degradation, soiling, "
            "availability losses, curtailment, clipping, or data issues."
        ),
        "weak_decline_caveat": (
            "Per-plant relative_change_pct_per_year in the -1% to -3% band is "
            "labelled 'apparent weak / mild performance loss', not physical "
            "module degradation: a single incomplete year (Mar-Dec 2019, ~7-8 "
            "monthly points per plant) cannot separate weak degradation from "
            "residual seasonality, soiling, availability, curtailment, "
            "clipping or data quality. Drops > 5-10%/yr are most likely "
            "operational anomalies or data problems, not physiological "
            "degradation."
        ),
        "decline_distribution": decline_distribution,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect decreasing PV performance trends as domain-shift/degradation evidence."
    )
    parser.add_argument("--sentinel-dir", default="/data/SentinelPV/energy_data/piemonte_energy_data/single_ups")
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--plant-mapping", default="data/plant_mapping.csv")
    parser.add_argument("--energy-coords", default="data/energy_with_coordinates.csv")
    parser.add_argument("--pvgis", default="data/piedmont_pvgis_2019.nc")
    parser.add_argument("--out-dir", default="outputs/domain_shift_trend")
    parser.add_argument("--daytime-poa-threshold", type=float, default=50.0)
    parser.add_argument("--reference-pr", type=float, default=1.0)
    parser.add_argument(
        "--kwp-mode",
        choices=("real-only", "real-or-proxy"),
        default="real-only",
        help=(
            "real-only uses only plants with registry kWp. real-or-proxy fills "
            "missing kWp with a p99 energy / p99 PVGIS proxy."
        ),
    )
    parser.add_argument("--plausible-pr-min", type=float, default=0.2)
    parser.add_argument("--plausible-pr-max", type=float, default=2.0)
    parser.add_argument("--min-day-hours", type=int, default=4)
    parser.add_argument("--min-month-hours", type=int, default=80)
    parser.add_argument("--min-months", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--decline-bin-edges",
        default=",".join(str(e) for e in DEFAULT_DECLINE_BIN_EDGES),
        help=(
            "Ascending comma-separated edges (percent per year) for the decline-class bins. "
            "Default: -10,-5,-3,-2,-1,0  -> 7 bins from strong_decline_gt_10 to stable_or_positive."
        ),
    )
    parser.add_argument(
        "--decline-bin-labels",
        default=",".join(DEFAULT_DECLINE_BIN_LABELS),
        help=(
            "Comma-separated labels for the bins, one more than edges. "
            "Order is from most-negative to most-positive."
        ),
    )
    parser.add_argument(
        "--monotonic-strong-tau",
        type=float,
        default=-0.85,
        help="Kendall tau <= this is classified as strictly_or_nearly_monotonic_decline.",
    )
    parser.add_argument(
        "--monotonic-directional-tau",
        type=float,
        default=-0.60,
        help="Kendall tau in (strong, directional] is classified as directional_decline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ds = load_sentinel_hourly(
            sentinel_dir=args.sentinel_dir,
            year=args.year,
            plant_mapping_path=args.plant_mapping,
            energy_coords_path=args.energy_coords,
        )
        ds = merge_with_weather(ds, pvgis_path=args.pvgis)
        ds = _normalize_dataset(ds)

    if "solar_irradiance_poa" not in ds:
        raise ValueError("Dataset must contain solar_irradiance_poa after PVGIS merge.")
    if "ENERGIA" not in ds:
        raise ValueError("Dataset must contain ENERGIA.")

    plant_ids = _safe_coord(ds, "plant_id", np.arange(ds.sizes["plant"]))
    upns = _safe_coord(ds, "upn", np.array([""] * ds.sizes["plant"], dtype=object))
    kwp, kwp_is_real = _load_kwp_by_plant_dim(
        Path(args.plant_mapping),
        Path(args.energy_coords),
        plant_ids,
        upns=upns,
    )

    daily_df, monthly_df = _build_performance_tables(
        ds=ds,
        kwp=kwp,
        kwp_is_real=kwp_is_real,
        kwp_mode=args.kwp_mode,
        daytime_poa_threshold=args.daytime_poa_threshold,
        reference_pr=args.reference_pr,
        min_day_hours=args.min_day_hours,
        min_month_hours=args.min_month_hours,
    )
    daily_df, monthly_df, excluded_pr = _filter_plausible_pr(
        daily_df=daily_df,
        monthly_df=monthly_df,
        pr_min=args.plausible_pr_min,
        pr_max=args.plausible_pr_max,
    )
    monthly_df = _add_plant_standardization(monthly_df)
    excluded_diag = _diagnose_implausible_pr(
        ds=ds,
        kwp=kwp,
        kwp_is_real=kwp_is_real,
        excluded_pr=excluded_pr,
        daytime_poa_threshold=args.daytime_poa_threshold,
        plausible_pr_min=args.plausible_pr_min,
        plausible_pr_max=args.plausible_pr_max,
        plant_mapping=Path(args.plant_mapping),
        energy_coords=Path(args.energy_coords),
    )
    if monthly_df.empty:
        raise RuntimeError(
            "No plants left after plausible-PR filtering. "
            "Relax --plausible-pr-min/--plausible-pr-max or inspect excluded_implausible_pr.csv."
        )
    fleet_daily, fleet_monthly = _fleet_tables(daily_df, monthly_df)
    plant_trends, fleet_trends = _trend_tables(
        monthly_df=monthly_df,
        fleet_monthly=fleet_monthly,
        min_months=args.min_months,
        alpha=args.alpha,
    )
    candidate_summary = _plant_candidate_summary(monthly_df, plant_trends)

    decline_edges = tuple(
        float(x) for x in str(args.decline_bin_edges).split(",") if x.strip() != ""
    )
    decline_labels = tuple(
        s.strip() for s in str(args.decline_bin_labels).split(",") if s.strip() != ""
    )
    if len(decline_labels) != len(decline_edges) + 1:
        raise ValueError(
            f"--decline-bin-labels must have len(edges)+1={len(decline_edges) + 1}, "
            f"got {len(decline_labels)}."
        )
    all_plant_trends = _build_all_plant_trend_table(
        candidate_summary=candidate_summary,
        decline_edges=decline_edges,
        decline_labels=decline_labels,
        monotonic_strong=args.monotonic_strong_tau,
        monotonic_directional=args.monotonic_directional_tau,
        alpha=args.alpha,
    )
    decline_distribution = _build_decline_distribution_summary(
        all_plants=all_plant_trends,
        decline_labels=decline_labels,
        monotonic_strong=args.monotonic_strong_tau,
        monotonic_directional=args.monotonic_directional_tau,
        alpha=args.alpha,
    )

    daily_df.to_csv(out_dir / "plant_daily_performance.csv", index=False)
    monthly_df.to_csv(out_dir / "plant_monthly_performance.csv", index=False)
    fleet_daily.to_csv(out_dir / "fleet_daily_performance.csv", index=False)
    fleet_monthly.to_csv(out_dir / "fleet_monthly_performance.csv", index=False)
    excluded_pr.to_csv(out_dir / "excluded_implausible_pr.csv", index=False)
    excluded_diag.to_csv(out_dir / "excluded_implausible_pr_diagnostics.csv", index=False)
    plant_trends.to_csv(out_dir / "plant_trend_summary.csv", index=False)
    candidate_summary.to_csv(out_dir / "plant_candidate_summary.csv", index=False)
    fleet_trends.to_csv(out_dir / "fleet_trend_summary.csv", index=False)

    all_plant_trends.to_csv(out_dir / "all_plant_pr_trends.csv", index=False)
    classes_cols = [
        c
        for c in (
            "plant",
            "plant_id",
            "upn",
            "valid_months",
            "pr_mean",
            "pr_first",
            "pr_last",
            "first_to_last_pct",
            "slope_per_month",
            "slope_per_year",
            "relative_change_pct_per_year",
            "p_value",
            "kendall_tau",
            "kendall_p_value",
            "decreasing",
            "significant_decline",
            "decline_class",
            "monotonic_class",
            "candidate_class",
        )
        if c in all_plant_trends.columns
    ]
    all_plant_trends[classes_cols].to_csv(
        out_dir / "all_plant_decline_classes.csv", index=False
    )
    with open(out_dir / "decline_distribution_summary.json", "w", encoding="utf-8") as f:
        json.dump(decline_distribution, f, indent=2)
    _plot_decline_histogram(out_dir, all_plant_trends, decline_edges)

    _plot_outputs(out_dir, fleet_monthly, plant_trends, monthly_df, args.top_k)
    _plot_individual_candidates(out_dir, monthly_df, candidate_summary, args.top_k)
    _write_summary(
        out_dir,
        args,
        ds,
        kwp_is_real,
        excluded_pr,
        candidate_summary,
        plant_trends,
        fleet_trends,
        fleet_monthly,
        decline_distribution=decline_distribution,
    )

    pr = plant_trends[plant_trends["metric"] == "pr_pvgis_monthly"]
    plant_z = plant_trends[plant_trends["metric"] == "pr_plant_z_monthly"]
    print("\n=== DOMAIN-SHIFT / DEGRADATION TREND REPORT ===")
    print(f"Output directory: {out_dir}")
    print(f"Plants: {ds.sizes['plant']}  hours: {ds.sizes['time']}")
    print(f"kWp real/proxy: {int(kwp_is_real.sum())}/{int((~kwp_is_real).sum())}")
    print(f"kWp mode: {args.kwp_mode}")
    print(
        f"plausible PR filter: [{args.plausible_pr_min}, {args.plausible_pr_max}]  "
        f"excluded={len(excluded_pr)}"
    )
    print("\nFleet trends:")
    for _, row in fleet_trends.iterrows():
        print(
            f"  {row['metric']}: slope/year={row['slope_per_year']:+.4f}, "
            f"rel/year={row['relative_change_pct_per_year']:+.2f}%, "
            f"p={row['p_value']:.3g}, decreasing={bool(row['decreasing'])}"
        )
    print("\nPlant trends:")
    print(f"  decreasing PR_pvgis plants: {int(pr['decreasing'].sum())}/{pr['plant'].nunique()}")
    print(f"  decreasing plant-z plants:  {int(plant_z['decreasing'].sum())}/{plant_z['plant'].nunique()}")
    print(
        "  performance-decline candidates: "
        f"{int(candidate_summary['local_decline_flag'].sum())}/{candidate_summary['plant'].nunique()}"
    )
    print("\nFull per-plant trend distribution (all plants with valid PR fit):")
    print(f"  n_plants_analyzed: {decline_distribution['n_plants_analyzed']}")
    for lbl, cnt in decline_distribution["decline_class_counts"].items():
        pct = decline_distribution["decline_class_pct"].get(lbl, 0.0)
        print(f"    {lbl:<26s} {cnt:>4d}  ({pct:5.1f}%)")
    print(
        f"  significant (p<{args.alpha} & slope<0): "
        f"{decline_distribution['n_significant_p_lt_alpha_and_negative_slope']}"
    )
    print("  monotonic (Kendall tau):")
    for lbl, cnt in decline_distribution["monotonic_class_counts"].items():
        pct = decline_distribution["monotonic_class_pct"].get(lbl, 0.0)
        print(f"    {lbl:<40s} {cnt:>4d}  ({pct:5.1f}%)")
    print(
        "\nCAVEAT: with a single incomplete year (Mar-Dec 2019, ~7-8 monthly "
        "points/plant), weak/mild decline bins (-1% to -3%/yr) describe "
        "'apparent weak/mild performance loss', NOT confirmed physical "
        "degradation. Drops > 5-10%/yr likely reflect operational anomalies "
        "or data issues, not physiological module degradation."
    )

    print("\nKey files:")
    print("  plant_monthly_performance.csv")
    print("  plant_candidate_summary.csv")
    print("  plant_trend_summary.csv")
    print("  fleet_trend_summary.csv")
    print("  all_plant_pr_trends.csv             [NEW: full per-plant table]")
    print("  all_plant_decline_classes.csv       [NEW: per-plant classification]")
    print("  decline_distribution_summary.json   [NEW: counts + percentages]")
    print("  all_plant_relative_change_histogram.png  [NEW: trend distribution]")
    print("  fleet_pvgis_pr_trend.png")
    print("  individual_candidate_plant_trends.png")


if __name__ == "__main__":
    main()
