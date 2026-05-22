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

from physiq_pv.data.quality_score import compute_qs
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


def _per_plant_trends(
    monthly_df: pd.DataFrame,
    value_col: str,
    min_months: int,
    alpha: float,
    metric_name: str | None = None,
    extra_metrics: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Fit a linear trend per (plant, plant_id) on monthly_df[value_col].

    Returns the same schema as _trend_tables but on an arbitrary value column,
    so the relative-index pipeline can reuse _plant_candidate_summary without
    duplicating fit code.

    extra_metrics: optional {col: metric_label} for additional series to fit
        (e.g. plant-z support metric).
    """
    metric = metric_name or f"{value_col}_monthly"
    results: list[TrendResult] = []
    for (plant, plant_id), g in monthly_df.groupby(["plant", "plant_id"]):
        g = g.sort_values("date")
        s = pd.Series(
            g[value_col].to_numpy(dtype=float),
            index=pd.to_datetime(g["date"]),
        )
        results.append(_fit_trend(s, metric, plant, plant_id, min_months, alpha))
        for col, lbl in (extra_metrics or {}).items():
            if col in g.columns:
                s2 = pd.Series(
                    g[col].to_numpy(dtype=float),
                    index=pd.to_datetime(g["date"]),
                )
                results.append(_fit_trend(s2, lbl, plant, plant_id, min_months, alpha))
    return pd.DataFrame([asdict(r) for r in results])


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


def _add_plant_standardization(
    monthly_df: pd.DataFrame,
    value_col: str = "pr_pvgis",
    output_col: str = "pr_plant_z",
    stats_prefix: str = "plant_pr",
) -> pd.DataFrame:
    """Add a within-plant z-score for an arbitrary monthly metric."""
    monthly = monthly_df.copy()
    monthly[value_col] = monthly[value_col].replace([np.inf, -np.inf], np.nan)
    mean_col = f"{stats_prefix}_mean"
    std_col = f"{stats_prefix}_std"
    n_col = f"{stats_prefix}_n"
    plant_stats = (
        monthly.groupby(["plant", "plant_id"], as_index=False)
        .agg(
            **{
                mean_col: (value_col, "mean"),
                std_col: (value_col, "std"),
                n_col: (value_col, "count"),
            }
        )
    )
    monthly = monthly.merge(plant_stats, on=["plant", "plant_id"], how="left")
    monthly[output_col] = (
        (monthly[value_col] - monthly[mean_col])
        / monthly[std_col].replace(0.0, np.nan)
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
    value_col: str = "pr_pvgis",
    metric_name: str = "pr_pvgis_monthly",
    output_prefix: str = "pr",
    z_metric_name: str | None = "pr_plant_z_monthly",
    z_output_prefix: str = "plant_z",
    stats_prefix: str = "plant_pr",
) -> pd.DataFrame:
    """One row per plant for an arbitrary monthly metric.

    The defaults preserve the real-kWp PR_PVGIS output schema. For the
    non-real-kWp branch callers pass value_col='relative_index' and
    output_prefix='relative_index', so the exported columns never label the
    proxy-free metric as PR.
    """
    primary = plant_trends[plant_trends["metric"] == metric_name].copy()
    primary_cols = {
        "n_points": f"{output_prefix}_n_points",
        "mean_value": f"{output_prefix}_mean",
        "first_value": f"{output_prefix}_first",
        "last_value": f"{output_prefix}_last",
        "slope_per_year": f"{output_prefix}_slope_per_year",
        "relative_change_pct_per_year": f"{output_prefix}_relative_change_pct_per_year",
        "p_value": f"{output_prefix}_p_value",
        "kendall_tau": f"{output_prefix}_kendall_tau",
        "kendall_p_value": f"{output_prefix}_kendall_p_value",
        "decreasing": f"{output_prefix}_decreasing",
    }
    primary = primary[["plant", "plant_id", *primary_cols.keys()]].rename(
        columns=primary_cols
    )

    z = pd.DataFrame(columns=["plant", "plant_id"])
    if z_metric_name is not None:
        z = plant_trends[plant_trends["metric"] == z_metric_name].copy()
        if not z.empty:
            z_cols = {
                "n_points": f"{z_output_prefix}_n_points",
                "mean_value": f"{z_output_prefix}_mean",
                "first_value": f"{z_output_prefix}_first",
                "last_value": f"{z_output_prefix}_last",
                "slope_per_year": f"{z_output_prefix}_slope_per_year",
                "p_value": f"{z_output_prefix}_p_value",
                "kendall_tau": f"{z_output_prefix}_kendall_tau",
                "kendall_p_value": f"{z_output_prefix}_kendall_p_value",
                "decreasing": f"{z_output_prefix}_decreasing",
            }
            z = z[["plant", "plant_id", *z_cols.keys()]].rename(columns=z_cols)

    min_col = "min_pr" if output_prefix == "pr" else f"min_{output_prefix}"
    max_col = "max_pr" if output_prefix == "pr" else f"max_{output_prefix}"
    median_col = (
        "median_pr" if output_prefix == "pr" else f"median_{output_prefix}"
    )
    mean_stat_col = f"{stats_prefix}_mean"
    std_stat_col = f"{stats_prefix}_std"

    agg_spec = {
        "upn": ("upn", "first") if "upn" in monthly_df.columns else ("plant_id", "first"),
        "kwp_used": ("kwp_used", "first"),
        "kwp_source": ("kwp_source", "first"),
        "valid_months": (value_col, "count"),
        min_col: (value_col, "min"),
        max_col: (value_col, "max"),
        median_col: (value_col, "median"),
        mean_stat_col: (
            mean_stat_col,
            "first",
        )
        if mean_stat_col in monthly_df.columns
        else (value_col, "mean"),
        std_stat_col: (
            std_stat_col,
            "first",
        )
        if std_stat_col in monthly_df.columns
        else (value_col, "std"),
        "first_month": ("date", "first"),
        "last_month": ("date", "last"),
    }
    if output_prefix == "relative_index":
        for optional_col in ("baseline_apparent_capacity", "baseline_method", "baseline_n_months"):
            if optional_col in monthly_df.columns:
                agg_spec[optional_col] = (optional_col, "first")

    meta = (
        monthly_df.sort_values("date")
        .groupby(["plant", "plant_id"], as_index=False)
        .agg(**agg_spec)
    )
    out = meta.merge(primary, on=["plant", "plant_id"], how="left")
    if not z.empty:
        out = out.merge(z, on=["plant", "plant_id"], how="left")

    first_col = f"{output_prefix}_first"
    last_col = f"{output_prefix}_last"
    slope_col = f"{output_prefix}_slope_per_year"
    decreasing_col = f"{output_prefix}_decreasing"
    z_decreasing_col = f"{z_output_prefix}_decreasing"

    out["first_to_last_pct"] = np.where(
        np.abs(out[first_col]) > 1e-12,
        (out[last_col] - out[first_col]) / out[first_col] * 100.0,
        np.nan,
    )
    out["peak_to_last_pct"] = np.where(
        np.abs(out[max_col]) > 1e-12,
        (out[last_col] - out[max_col]) / out[max_col] * 100.0,
        np.nan,
    )
    out[decreasing_col] = out[decreasing_col].fillna(False).astype(bool)
    if z_decreasing_col in out.columns:
        out[z_decreasing_col] = out[z_decreasing_col].fillna(False).astype(bool)

    conditions = [out[decreasing_col]]
    choices = ["performance_decline"]
    out["candidate_class"] = np.select(
        conditions, choices, default="no_significant_decline"
    )
    out["candidate_rank_score"] = (
        out[decreasing_col].astype(int)
        + np.clip(-out[slope_col].fillna(0.0), 0.0, None)
    )
    out["local_decline_flag"] = out["candidate_class"] != "no_significant_decline"
    out["statistical_note"] = np.where(
        out["valid_months"] < 12,
        "exploratory: single incomplete year",
        "multi-month trend",
    )
    return out.sort_values(
        ["candidate_rank_score", decreasing_col, slope_col],
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
    output_prefix: str = "pr",
) -> pd.DataFrame:
    """Full per-plant trend table for every plant that produced a PR fit.

    Built on top of candidate_summary (already one row per plant). Adds
    relative_change_pct_per_year, decline_class, monotonic_class and the
    significant_decline flag. No regression logic is duplicated.
    """
    if candidate_summary.empty:
        return candidate_summary.copy()

    df = candidate_summary.copy()
    mean_col = f"{output_prefix}_mean"
    slope_source_col = f"{output_prefix}_slope_per_year"
    p_source_col = f"{output_prefix}_p_value"
    tau_source_col = f"{output_prefix}_kendall_tau"
    tau_p_source_col = f"{output_prefix}_kendall_p_value"
    decreasing_source_col = f"{output_prefix}_decreasing"

    metric_mean = df[mean_col].to_numpy(dtype=float)
    slope_year = df[slope_source_col].to_numpy(dtype=float)
    rel = np.where(
        np.abs(metric_mean) > 1e-12,
        slope_year / metric_mean * 100.0,
        np.nan,
    )
    df["relative_change_pct_per_year"] = rel
    df["slope_per_year"] = slope_year
    df["slope_per_month"] = slope_year / 12.0
    df["p_value"] = df[p_source_col]
    df["kendall_tau"] = df[tau_source_col]
    df["kendall_p_value"] = df[tau_p_source_col]
    df["decreasing"] = df[decreasing_source_col]

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
        mean_col,
        f"{output_prefix}_first",
        f"{output_prefix}_last",
        "median_pr" if output_prefix == "pr" else f"median_{output_prefix}",
        "min_pr" if output_prefix == "pr" else f"min_{output_prefix}",
        "max_pr" if output_prefix == "pr" else f"max_{output_prefix}",
        "baseline_apparent_capacity",
        "baseline_method",
        "baseline_n_months",
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
    metric_label: str = "pr_pvgis",
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
        f"n_decreasing_{metric_label}": n_decreasing_any,
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


# ---------------------------------------------------------------------------
# Level-shift / regime-shift analysis
# ---------------------------------------------------------------------------

DEFAULT_HALF_SHIFT_EDGES: tuple[float, ...] = (-10.0, -5.0, -2.0)
DEFAULT_HALF_SHIFT_LABELS: tuple[str, ...] = (
    "strong_downshift_gt_10",
    "moderate_downshift_5_10",
    "mild_downshift_2_5",
    "stable_or_improved_shift",
)

DEFAULT_BREAK_SHIFT_EDGES: tuple[float, ...] = (-10.0, -5.0, -2.0)
DEFAULT_BREAK_SHIFT_LABELS: tuple[str, ...] = (
    "strong_downshift",
    "moderate_downshift",
    "mild_downshift",
    "no_downshift",
)


def _classify_band(
    value: float,
    edges: tuple[float, ...],
    labels: tuple[str, ...],
) -> str:
    """Ascending edges semantics matching _classify_decline_bin."""
    if value is None or not np.isfinite(value):
        return "undefined"
    if len(labels) != len(edges) + 1:
        raise ValueError("labels must equal edges + 1")
    if value < edges[0]:
        return labels[0]
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return labels[i + 1]
    if value >= edges[-1]:
        return labels[-1]
    return "undefined"


def _half_split_stats(values: np.ndarray, mean_prefix: str = "pr") -> dict:
    """Stats for a fixed first-half / second-half split on a chronological series."""
    first_mean_col = f"first_half_{mean_prefix}_mean"
    second_mean_col = f"second_half_{mean_prefix}_mean"
    out = {
        "first_half_n_months": 0,
        "second_half_n_months": 0,
        first_mean_col: float("nan"),
        second_mean_col: float("nan"),
        "half_delta_abs": float("nan"),
        "half_delta_pct": float("nan"),
    }
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return out
    mid = n // 2  # second half gets the extra point when n is odd
    first, second = v[:mid], v[mid:]
    if first.size == 0 or second.size == 0:
        return out
    m1, m2 = float(first.mean()), float(second.mean())
    out["first_half_n_months"] = int(first.size)
    out["second_half_n_months"] = int(second.size)
    out[first_mean_col] = m1
    out[second_mean_col] = m2
    out["half_delta_abs"] = m2 - m1
    out["half_delta_pct"] = (
        (m2 - m1) / m1 * 100.0 if abs(m1) > 1e-12 else float("nan")
    )
    return out


def _best_break_stats(
    dates: np.ndarray,
    values: np.ndarray,
    min_pre_months: int,
    min_post_months: int,
    cv_penalty: float,
    selection: str,
) -> dict:
    """Scan every valid split point and pick the one with the strongest downshift.

    selection:
      - 'max_drop'  -> pick the most-negative break_delta_pct.
      - 'score'     -> pick max( |break_delta_pct| - cv_penalty * post_cv_pct )
                       only among candidates with break_delta_pct < 0; falls back
                       to max_drop otherwise.

    post_break_cv is computed on raw PR (std / mean), post_cv_pct is in percent.
    """
    out = {
        "best_break_month": "",
        "best_break_index": -1,
        "n_break_candidates_tried": 0,
        "pre_break_n_months": 0,
        "post_break_n_months": 0,
        "pre_break_mean": float("nan"),
        "post_break_mean": float("nan"),
        "pre_break_std": float("nan"),
        "post_break_std": float("nan"),
        "post_break_cv": float("nan"),
        "break_delta_abs": float("nan"),
        "break_delta_pct": float("nan"),
        "best_break_score": float("nan"),
    }
    v = np.asarray(values, dtype=float)
    d = np.asarray(dates)
    m = np.isfinite(v)
    v = v[m]
    d = d[m]
    n = v.size
    if n < (min_pre_months + min_post_months):
        return out

    tried = 0
    best_idx = -1
    best_score = -np.inf
    best_pack: dict | None = None
    for k in range(min_pre_months, n - min_post_months + 1):
        pre = v[:k]
        post = v[k:]
        if pre.size < min_pre_months or post.size < min_post_months:
            continue
        tried += 1
        m1, m2 = float(pre.mean()), float(post.mean())
        s1 = float(pre.std(ddof=1)) if pre.size >= 2 else float("nan")
        s2 = float(post.std(ddof=1)) if post.size >= 2 else float("nan")
        delta_abs = m2 - m1
        delta_pct = (m2 - m1) / m1 * 100.0 if abs(m1) > 1e-12 else float("nan")
        post_cv = s2 / m2 if np.isfinite(s2) and abs(m2) > 1e-12 else float("nan")
        post_cv_pct = post_cv * 100.0 if np.isfinite(post_cv) else float("nan")

        if selection == "max_drop":
            score = -delta_pct if np.isfinite(delta_pct) else -np.inf
        else:
            if not np.isfinite(delta_pct) or delta_pct >= 0:
                score = -np.inf
            else:
                penalty = (
                    cv_penalty * post_cv_pct if np.isfinite(post_cv_pct) else 0.0
                )
                score = abs(delta_pct) - penalty
        if score > best_score:
            best_score = score
            best_idx = k
            best_pack = {
                "pre_n": int(pre.size),
                "post_n": int(post.size),
                "m1": m1,
                "m2": m2,
                "s1": s1,
                "s2": s2,
                "post_cv": post_cv,
                "delta_abs": delta_abs,
                "delta_pct": delta_pct,
                "score": score,
                "break_label": str(pd.Timestamp(d[k]).date()),
            }

    out["n_break_candidates_tried"] = tried
    if best_pack is None or selection == "score" and not np.isfinite(best_score):
        # selection='score' and no negative candidate -> fall back to max_drop
        for k in range(min_pre_months, n - min_post_months + 1):
            pre = v[:k]
            post = v[k:]
            m1, m2 = float(pre.mean()), float(post.mean())
            s1 = float(pre.std(ddof=1)) if pre.size >= 2 else float("nan")
            s2 = float(post.std(ddof=1)) if post.size >= 2 else float("nan")
            delta_pct = (m2 - m1) / m1 * 100.0 if abs(m1) > 1e-12 else float("nan")
            if best_pack is None or (
                np.isfinite(delta_pct) and delta_pct < best_pack["delta_pct"]
            ):
                best_idx = k
                best_pack = {
                    "pre_n": int(pre.size),
                    "post_n": int(post.size),
                    "m1": m1,
                    "m2": m2,
                    "s1": s1,
                    "s2": s2,
                    "post_cv": (
                        s2 / m2 if np.isfinite(s2) and abs(m2) > 1e-12 else float("nan")
                    ),
                    "delta_abs": m2 - m1,
                    "delta_pct": delta_pct,
                    "score": (-delta_pct) if np.isfinite(delta_pct) else float("nan"),
                    "break_label": str(pd.Timestamp(d[k]).date()),
                }

    if best_pack is None:
        return out
    out["best_break_index"] = int(best_idx)
    out["best_break_month"] = best_pack["break_label"]
    out["pre_break_n_months"] = best_pack["pre_n"]
    out["post_break_n_months"] = best_pack["post_n"]
    out["pre_break_mean"] = best_pack["m1"]
    out["post_break_mean"] = best_pack["m2"]
    out["pre_break_std"] = best_pack["s1"]
    out["post_break_std"] = best_pack["s2"]
    out["post_break_cv"] = best_pack["post_cv"]
    out["break_delta_abs"] = best_pack["delta_abs"]
    out["break_delta_pct"] = best_pack["delta_pct"]
    out["best_break_score"] = best_pack["score"]
    return out


def _build_level_shift_table(
    monthly_df: pd.DataFrame,
    all_plant_trends: pd.DataFrame,
    half_shift_edges: tuple[float, ...],
    half_shift_labels: tuple[str, ...],
    break_shift_edges: tuple[float, ...],
    break_shift_labels: tuple[str, ...],
    min_pre_months: int,
    min_post_months: int,
    cv_penalty: float,
    break_selection: str,
    plateau_break_pct_threshold: float,
    plateau_post_cv_max: float,
    value_col: str = "pr_pvgis",
    mean_prefix: str = "pr",
    rename_break_means: bool = False,
) -> pd.DataFrame:
    """One row per plant with half-split + best-break-search level-shift stats.

    Joined on (plant, plant_id) with all_plant_trends. No regression logic is
    duplicated: monthly series come from monthly_df[value_col].
    """
    rows: list[dict] = []
    monthly = monthly_df.copy()
    monthly["date"] = pd.to_datetime(monthly["date"])
    for (plant, plant_id), g in monthly.groupby(["plant", "plant_id"]):
        g = g.sort_values("date").dropna(subset=[value_col])
        if g.empty:
            continue
        dates = g["date"].to_numpy()
        values = g[value_col].to_numpy(dtype=float)

        half = _half_split_stats(values, mean_prefix=mean_prefix)
        brk = _best_break_stats(
            dates=dates,
            values=values,
            min_pre_months=min_pre_months,
            min_post_months=min_post_months,
            cv_penalty=cv_penalty,
            selection=break_selection,
        )

        half_class = _classify_band(
            half["half_delta_pct"], half_shift_edges, half_shift_labels
        )
        break_class = _classify_band(
            brk["break_delta_pct"], break_shift_edges, break_shift_labels
        )

        post_cv = brk["post_break_cv"]
        delta_pct = brk["break_delta_pct"]
        plateau_flag = bool(
            np.isfinite(delta_pct)
            and delta_pct < plateau_break_pct_threshold
            and (
                (np.isfinite(post_cv) and post_cv <= plateau_post_cv_max)
                or not np.isfinite(post_cv)
            )
        )

        pre_m = brk["pre_break_mean"]
        post_m = brk["post_break_mean"]
        if np.isfinite(pre_m) and np.isfinite(post_m):
            exp_bias_abs = pre_m - post_m
            exp_bias_pct = (
                (pre_m - post_m) / post_m * 100.0 if abs(post_m) > 1e-12 else float("nan")
            )
        else:
            exp_bias_abs = float("nan")
            exp_bias_pct = float("nan")

        if rename_break_means:
            brk = brk.copy()
            brk[f"pre_break_{mean_prefix}_mean"] = brk.pop("pre_break_mean")
            brk[f"post_break_{mean_prefix}_mean"] = brk.pop("post_break_mean")

        rows.append(
            {
                "plant": plant,
                "plant_id": plant_id,
                **half,
                "half_shift_class": half_class,
                **brk,
                "best_break_shift_class": break_class,
                "possible_step_change_with_plateau": plateau_flag,
                "expected_bias_if_train_pre_break": exp_bias_abs,
                "expected_bias_pct_if_train_pre_break": exp_bias_pct,
            }
        )

    shift_df = pd.DataFrame(rows)
    if shift_df.empty or all_plant_trends.empty:
        return shift_df

    merged = all_plant_trends.merge(shift_df, on=["plant", "plant_id"], how="left")
    return merged


def _build_level_shift_summary(
    merged: pd.DataFrame,
    half_shift_labels: tuple[str, ...],
    break_shift_labels: tuple[str, ...],
    alpha: float,
    overprediction_bias_pct_threshold: float | None = None,
) -> dict:
    n_total = int(len(merged))
    half_counts = {lbl: 0 for lbl in (*half_shift_labels, "undefined")}
    for lbl, cnt in merged["half_shift_class"].value_counts(dropna=False).items():
        half_counts[str(lbl)] = int(cnt)
    half_pct = {k: (v / n_total * 100.0 if n_total else 0.0) for k, v in half_counts.items()}

    break_counts = {lbl: 0 for lbl in (*break_shift_labels, "undefined")}
    for lbl, cnt in merged["best_break_shift_class"].value_counts(dropna=False).items():
        break_counts[str(lbl)] = int(cnt)
    break_pct = {k: (v / n_total * 100.0 if n_total else 0.0) for k, v in break_counts.items()}

    plateau = merged["possible_step_change_with_plateau"].fillna(False).astype(bool)
    n_plateau = int(plateau.sum())
    half_downshift = (
        merged["half_shift_class"].astype(str).str.contains("downshift", na=False)
        & ~merged["half_shift_class"].astype(str).str.contains("no_downshift", na=False)
    )
    break_downshift = (
        merged["best_break_shift_class"].astype(str).str.contains("downshift", na=False)
        & ~merged["best_break_shift_class"].astype(str).str.contains("no_downshift", na=False)
    )
    downshift = half_downshift | break_downshift

    if "monotonic_class" in merged.columns:
        monotonic_decline = merged["monotonic_class"].isin(
            ["strictly_or_nearly_monotonic_decline", "directional_decline"]
        )
        non_monotonic = ~monotonic_decline
    else:
        non_monotonic = pd.Series([False] * n_total, index=merged.index)
    if "significant_decline" in merged.columns:
        non_significant = ~merged["significant_decline"].fillna(False).astype(bool)
    else:
        non_significant = pd.Series([False] * n_total, index=merged.index)

    n_plateau_non_monotonic = int((plateau & non_monotonic).sum())
    n_plateau_non_significant = int((plateau & non_significant).sum())
    n_downshift_non_monotonic = int((downshift & non_monotonic).sum())
    n_downshift_non_significant = int((downshift & non_significant).sum())

    if (
        overprediction_bias_pct_threshold is not None
        and "expected_bias_pct_if_train_pre_break" in merged.columns
    ):
        bias_pct = merged["expected_bias_pct_if_train_pre_break"].fillna(0.0)
        n_overprediction_risk = int((bias_pct >= overprediction_bias_pct_threshold).sum())
    else:
        n_overprediction_risk = 0

    plateau_by_decline_class: dict[str, int] = {}
    if "decline_class" in merged.columns:
        for lbl, cnt in (
            merged.loc[plateau, "decline_class"].value_counts(dropna=False).items()
        ):
            plateau_by_decline_class[str(lbl)] = int(cnt)

    return {
        "n_plants_analyzed": n_total,
        "alpha": alpha,
        "half_shift_class_counts": half_counts,
        "half_shift_class_pct": half_pct,
        "best_break_shift_class_counts": break_counts,
        "best_break_shift_class_pct": break_pct,
        "n_possible_step_change_with_plateau": n_plateau,
        "pct_possible_step_change_with_plateau": (
            n_plateau / n_total * 100.0 if n_total else 0.0
        ),
        "n_plateau_and_not_monotonic_decline": n_plateau_non_monotonic,
        "n_plateau_and_not_significant_decline": n_plateau_non_significant,
        "n_downshift_but_not_monotonic_decline": n_downshift_non_monotonic,
        "n_downshift_but_not_significant_slope": n_downshift_non_significant,
        "overprediction_bias_pct_threshold": overprediction_bias_pct_threshold,
        "n_overprediction_risk_if_trained_pre_break": n_overprediction_risk,
        "plateau_candidates_by_decline_class": plateau_by_decline_class,
        "interpretation_caveat": (
            "Level-shift / regime-shift candidates are NOT automatically physical "
            "module degradation. A negative half- or break-shift can come from "
            "soiling, inverter or string faults, availability losses, grid "
            "curtailment, inverter clipping, data quality changes, operational "
            "changes (cleaning, replacements, configuration), residual "
            "seasonality on a single incomplete year, or - among other causes - "
            "physical degradation. Use the label 'apparent performance "
            "downshift' / 'apparent performance loss', never 'degradation' as a "
            "conclusion. For forecasting: pre-break-trained models can over- or "
            "under-predict a post-break regime by roughly "
            "expected_bias_pct_if_train_pre_break."
        ),
    }


def _plot_level_shift(
    out_dir: Path,
    merged: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-darkgrid")
    rel = merged["relative_change_pct_per_year"].to_numpy(dtype=float)
    half = merged["half_delta_pct"].to_numpy(dtype=float)
    brk = merged["break_delta_pct"].to_numpy(dtype=float)
    tau = merged["kendall_tau"].to_numpy(dtype=float) if "kendall_tau" in merged.columns else None

    half_finite = half[np.isfinite(half)]
    brk_finite = brk[np.isfinite(brk)]

    if half_finite.size:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(half_finite, bins=30, color="tab:blue", edgecolor="white", alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=1.0)
        for e in (-2.0, -5.0, -10.0):
            ax.axvline(e, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("half_delta_pct (second-half - first-half) / first-half * 100")
        ax.set_ylabel("plant count")
        ax.set_title(f"Half-split level shift distribution (n={half_finite.size})")
        fig.tight_layout()
        fig.savefig(out_dir / "level_shift_half_delta_pct_histogram.png", dpi=180)
        plt.close(fig)

    if brk_finite.size:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(brk_finite, bins=30, color="tab:purple", edgecolor="white", alpha=0.85)
        ax.axvline(0.0, color="black", linewidth=1.0)
        for e in (-2.0, -5.0, -10.0):
            ax.axvline(e, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("best_break_delta_pct (post - pre) / pre * 100")
        ax.set_ylabel("plant count")
        ax.set_title(f"Best-break level shift distribution (n={brk_finite.size})")
        fig.tight_layout()
        fig.savefig(out_dir / "level_shift_break_delta_pct_histogram.png", dpi=180)
        plt.close(fig)

    mask = np.isfinite(rel) & np.isfinite(half)
    if mask.any():
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(rel[mask], half[mask], alpha=0.7, s=22, color="tab:blue")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("relative_change_pct_per_year (slope-based)")
        ax.set_ylabel("half_delta_pct (level-shift based)")
        ax.set_title("Slope-based trend vs half-split level shift")
        fig.tight_layout()
        fig.savefig(out_dir / "level_shift_scatter_slope_vs_half.png", dpi=180)
        plt.close(fig)

    if tau is not None:
        mask = np.isfinite(tau) & np.isfinite(brk)
        if mask.any():
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(tau[mask], brk[mask], alpha=0.7, s=22, color="tab:purple")
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.axvline(0.0, color="black", linewidth=0.8)
            ax.set_xlabel("Kendall tau (monotonicity)")
            ax.set_ylabel("best_break_delta_pct (post - pre) / pre * 100")
            ax.set_title("Monotonicity vs best-break shift")
            fig.tight_layout()
            fig.savefig(out_dir / "level_shift_scatter_tau_vs_break.png", dpi=180)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Relative (proxy-free) intra-plant index for plants WITHOUT real kWp
# ---------------------------------------------------------------------------
#
# Methodology note:
#   For plants without registry kWp the absolute PR_PVGIS cannot be computed,
#   because it requires the nominal installed power as denominator.
#   We instead compute a monthly "apparent capacity proxy":
#
#       apparent_capacity_i,m = sum_h(actual_kwh_h) / sum_h(expected_pvgis_per_kwp_h)
#
#   where expected_pvgis_per_kwp_h is POA_kWh_per_m2 * reference_pr, matching
#   the denominator used by the real-kWp PR_PVGIS pipeline before multiplying
#   by installed kWp. The ratio is proportional to kWp_i * PR_i,m, so the time
#   series WITHIN a plant tracks apparent performance over time. It is NOT a
#   true PR: it conflates the unknown installed kWp with PR losses, soiling,
#   curtailment, clipping, availability and data quality.
#
#   Each plant is then normalized by its own robust baseline:
#
#       relative_index_i,m = apparent_capacity_i,m / baseline_i
#
#   where baseline_i defaults to the median of the first N valid monthly values
#   (N=3). Rationale: the median of the first N months is robust to a single
#   anomalous month and anchors the comparison to the early period, so a real
#   downshift mid-period is NOT absorbed by the denominator. Using the full-
#   period median would mask exactly the regime shift we want to detect.
#
#   The relative_index is meaningful ONLY within the same plant. It must not
#   be compared across plants - the unknown kWp cancels out per plant but
#   does not cancel out between plants.

DEFAULT_BASELINE_STRATEGIES = ("median_first_n", "mean_first_n", "median_all")


def _compute_plant_baseline(
    values: np.ndarray,
    strategy: str,
    n_months: int,
) -> tuple[float, str, int]:
    """Return (baseline, applied_method, n_months_used)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), "no_data", 0
    if strategy == "median_first_n":
        if v.size >= n_months:
            head = v[:n_months]
            return float(np.median(head)), f"median_first_{n_months}", int(head.size)
        return float(np.median(v)), "median_all_valid_fallback", int(v.size)
    if strategy == "mean_first_n":
        if v.size >= n_months:
            head = v[:n_months]
            return float(np.mean(head)), f"mean_first_{n_months}", int(head.size)
        return float(np.median(v)), "median_all_valid_fallback", int(v.size)
    if strategy == "median_all":
        return float(np.median(v)), "median_all", int(v.size)
    raise ValueError(f"Unknown baseline strategy: {strategy!r}")


def _build_relative_monthly(
    ds: xr.Dataset,
    kwp_is_real: np.ndarray,
    daytime_poa_threshold: float,
    reference_pr: float,
    min_day_hours: int,
    min_month_hours: int,
    baseline_strategy: str,
    baseline_n_months: int,
    min_valid_months: int,
) -> tuple[pd.DataFrame, dict]:
    """Per-plant monthly apparent-capacity + intra-plant relative index for
    plants WITHOUT real kWp. No interaction with the real-kWp pipeline.

    Returns (monthly_rel, diagnostics) where monthly_rel has the schema:
      plant, plant_id, upn, date, calendar_month, valid_hours,
      actual_kwh, expected_pvgis_per_kwp, apparent_capacity,
      baseline_apparent_capacity, baseline_method, baseline_n_months,
      relative_index.
    """
    times = pd.DatetimeIndex(ds.coords["time"].values)
    plant_ids = _safe_coord(ds, "plant_id", np.arange(ds.sizes["plant"]))
    upns = _safe_coord(ds, "upn", np.array([""] * ds.sizes["plant"], dtype=object))
    energy = np.asarray(ds["ENERGIA"].values, dtype=np.float64)
    poa_wm2 = np.asarray(ds["solar_irradiance_poa"].values, dtype=np.float64)
    poa_kwm2 = np.clip(poa_wm2 / 1000.0, 0.0, None)
    expected_per_kwp = poa_kwm2 * reference_pr
    day = poa_wm2 >= daytime_poa_threshold

    rows: list[pd.DataFrame] = []
    n_candidates = 0
    n_no_data = 0
    n_insufficient_data = 0
    n_no_baseline = 0

    for p in range(ds.sizes["plant"]):
        if kwp_is_real[p]:
            continue
        n_candidates += 1

        actual = pd.Series(energy[p], index=times, dtype="float64")
        expected = pd.Series(expected_per_kwp[p], index=times, dtype="float64")
        valid = pd.Series(
            day[p]
            & np.isfinite(energy[p])
            & np.isfinite(expected_per_kwp[p])
            & (energy[p] >= 0)
            & (expected_per_kwp[p] > 1e-9),
            index=times,
        )

        monthly = _resample_ratio(actual, expected, valid, "ME", min_month_hours)
        monthly = monthly.rename(
            columns={
                "pvgis_expected_kwh": "expected_pvgis_per_kwp",
                "pr_pvgis": "apparent_capacity",
            }
        )
        monthly["apparent_capacity"] = monthly["apparent_capacity"].replace(
            [np.inf, -np.inf], np.nan
        )
        finite = monthly["apparent_capacity"].dropna()
        if finite.empty:
            n_no_data += 1
            continue
        if finite.size < min_valid_months:
            n_insufficient_data += 1
            continue

        baseline, baseline_method, baseline_n_used = _compute_plant_baseline(
            finite.to_numpy(dtype=float),
            strategy=baseline_strategy,
            n_months=baseline_n_months,
        )
        if not np.isfinite(baseline) or baseline <= 0:
            n_no_baseline += 1
            continue

        monthly["baseline_apparent_capacity"] = baseline
        monthly["baseline_method"] = baseline_method
        monthly["baseline_n_months"] = baseline_n_used
        monthly["relative_index"] = monthly["apparent_capacity"] / baseline
        monthly["kwp_used"] = float("nan")
        monthly["kwp_source"] = "not_real"
        monthly.insert(0, "plant", p)
        monthly.insert(1, "plant_id", plant_ids[p])
        monthly.insert(2, "upn", str(upns[p]) if p < len(upns) else "")
        monthly.insert(3, "date", monthly.index)
        monthly.insert(4, "calendar_month", monthly.index.month)
        rows.append(monthly.reset_index(drop=True))

    diagnostics = {
        "n_non_real_kwp_plants_total": n_candidates,
        "n_excluded_no_data": n_no_data,
        "n_excluded_insufficient_data": n_no_data + n_insufficient_data + n_no_baseline,
        "n_excluded_too_few_months": n_insufficient_data,
        "n_excluded_no_baseline": n_no_baseline,
        "baseline_strategy_requested": baseline_strategy,
        "baseline_n_months_requested": baseline_n_months,
        "min_valid_months_required": min_valid_months,
    }
    if not rows:
        return pd.DataFrame(), diagnostics
    monthly_rel = pd.concat(rows, ignore_index=True)
    return monthly_rel, diagnostics


def _plot_relative_outputs(
    out_dir: Path,
    all_plant_trends_rel: pd.DataFrame,
    level_shift_rel: pd.DataFrame,
    decline_edges: tuple[float, ...],
) -> None:
    """Histograms + scatters for the non-real-kWp relative-index pipeline."""
    plt.style.use("seaborn-v0_8-darkgrid")
    rel = all_plant_trends_rel.get("relative_change_pct_per_year")
    if rel is not None:
        rel_arr = rel.to_numpy(dtype=float)
        rel_arr = rel_arr[np.isfinite(rel_arr)]
        if rel_arr.size:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            bins = np.linspace(
                min(float(rel_arr.min()), -12.0),
                max(float(rel_arr.max()), 5.0),
                40,
            )
            ax.hist(rel_arr, bins=bins, color="tab:cyan", edgecolor="white", alpha=0.85)
            for e in decline_edges:
                ax.axvline(e, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.axvline(0.0, color="black", linewidth=1.0)
            ax.set_xlabel(
                "relative_change_pct_per_year (intra-plant relative index)"
            )
            ax.set_ylabel("plant count")
            ax.set_title(
                "Non-real-kWp plants: relative-index trend distribution "
                f"(n={rel_arr.size})"
            )
            fig.tight_layout()
            fig.savefig(
                out_dir / "histogram_relative_change_pct_per_year_non_real.png",
                dpi=180,
            )
            plt.close(fig)

    if not level_shift_rel.empty:
        brk = level_shift_rel.get("break_delta_pct")
        half = level_shift_rel.get("half_delta_pct")
        tau = level_shift_rel.get("kendall_tau")
        slope = level_shift_rel.get("relative_change_pct_per_year")

        if brk is not None:
            arr = brk.to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.hist(arr, bins=30, color="tab:purple", edgecolor="white", alpha=0.85)
                ax.axvline(0.0, color="black", linewidth=1.0)
                for e in (-2.0, -5.0, -10.0):
                    ax.axvline(e, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.6)
                ax.set_xlabel("best_break_delta_pct (post - pre) / pre * 100")
                ax.set_ylabel("plant count")
                ax.set_title(
                    f"Non-real-kWp plants: best-break shift distribution (n={arr.size})"
                )
                fig.tight_layout()
                fig.savefig(
                    out_dir / "histogram_best_break_delta_pct_non_real.png", dpi=180
                )
                plt.close(fig)

        if tau is not None and brk is not None:
            ta = tau.to_numpy(dtype=float)
            ba = brk.to_numpy(dtype=float)
            m = np.isfinite(ta) & np.isfinite(ba)
            if m.any():
                fig, ax = plt.subplots(figsize=(7, 6))
                ax.scatter(ta[m], ba[m], alpha=0.6, s=18, color="tab:purple")
                ax.axhline(0.0, color="black", linewidth=0.8)
                ax.axvline(0.0, color="black", linewidth=0.8)
                ax.set_xlabel("Kendall tau (monotonicity)")
                ax.set_ylabel("best_break_delta_pct")
                ax.set_title("Non-real-kWp: monotonicity vs best-break shift")
                fig.tight_layout()
                fig.savefig(
                    out_dir / "scatter_kendall_tau_vs_best_break_delta_pct_non_real.png",
                    dpi=180,
                )
                plt.close(fig)

        if slope is not None and half is not None:
            sa = slope.to_numpy(dtype=float)
            ha = half.to_numpy(dtype=float)
            m = np.isfinite(sa) & np.isfinite(ha)
            if m.any():
                fig, ax = plt.subplots(figsize=(7, 6))
                ax.scatter(sa[m], ha[m], alpha=0.6, s=18, color="tab:cyan")
                ax.axhline(0.0, color="black", linewidth=0.8)
                ax.axvline(0.0, color="black", linewidth=0.8)
                ax.set_xlabel("relative_change_pct_per_year (slope-based)")
                ax.set_ylabel("half_delta_pct")
                ax.set_title("Non-real-kWp: slope vs half-split shift")
                fig.tight_layout()
                fig.savefig(
                    out_dir / "scatter_slope_vs_half_delta_pct_non_real.png", dpi=180
                )
                plt.close(fig)

        candidates = level_shift_rel.dropna(subset=["break_delta_pct"]).copy()
        if not candidates.empty:
            top = candidates.sort_values("break_delta_pct").head(20).copy()
            labels = [
                f"{row.plant} / {row.plant_id}"
                for row in top[["plant", "plant_id"]].itertuples(index=False)
            ]
            values = top["break_delta_pct"].to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(10, max(5.0, 0.35 * len(top))))
            y = np.arange(len(top))
            ax.barh(y, values, color="tab:red", alpha=0.78)
            ax.axvline(0.0, color="black", linewidth=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("best_break_delta_pct")
            ax.set_title("Top 20 non-real-kWp level-shift candidates")
            fig.tight_layout()
            fig.savefig(out_dir / "top20_level_shift_candidates_non_real.png", dpi=180)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Quality-score component trend analysis (QS, m1..m5)
# ---------------------------------------------------------------------------

QUALITY_SCORE_METRICS: tuple[str, ...] = ("quality_score", "m1", "m2", "m3", "m4", "m5")
QUALITY_SCORE_COMPONENT_LABELS: dict[str, str] = {
    "quality_score": "QS geometric mean",
    "m1": "corr_score",
    "m2": "bias_score",
    "m3": "nan_score",
    "m4": "var_score",
    "m5": "eta_score",
}


def _build_quality_score_monthly(
    ds: xr.Dataset,
    kwp: np.ndarray,
    kwp_is_real: np.ndarray,
    daytime_poa_threshold: float,
    qs_window: int,
) -> tuple[pd.DataFrame, dict]:
    """Monthly daytime means for QS aggregate and its m1..m5 components."""
    required = ("ENERGIA", "solar_irradiance_poa", "temperature_2m", "eta_base")
    missing = [name for name in required if name not in ds]
    diagnostics = {
        "qs_window_hours": int(qs_window),
        "metrics": list(QUALITY_SCORE_METRICS),
        "component_labels": QUALITY_SCORE_COMPONENT_LABELS,
        "daytime_poa_threshold_wm2": float(daytime_poa_threshold),
        "skipped": bool(missing),
        "missing_required_variables": missing,
    }
    if missing:
        return pd.DataFrame(), diagnostics

    qs_da, components = compute_qs(ds, window=qs_window, debug=True)
    times = pd.DatetimeIndex(ds.coords["time"].values)
    plant_ids = _safe_coord(ds, "plant_id", np.arange(ds.sizes["plant"]))
    upns = _safe_coord(ds, "upn", np.array([""] * ds.sizes["plant"], dtype=object))
    poa_wm2 = np.asarray(ds["solar_irradiance_poa"].values, dtype=np.float64)
    day = poa_wm2 >= daytime_poa_threshold

    metric_arrays = {"quality_score": np.asarray(qs_da.values, dtype=float)}
    for name in ("m1", "m2", "m3", "m4", "m5"):
        metric_arrays[name] = np.asarray(components[name], dtype=float)
    capacity_scale = np.asarray(components.get("capacity_scale", np.nan), dtype=float)

    rows: list[pd.DataFrame] = []
    for p in range(ds.sizes["plant"]):
        data: dict[str, pd.Series] = {}
        for name, arr in metric_arrays.items():
            vals = np.where(day[p], arr[p], np.nan)
            data[name] = pd.Series(vals, index=times, dtype="float64").resample("ME").mean()
        valid_qs = pd.Series(
            np.isfinite(np.where(day[p], metric_arrays["quality_score"][p], np.nan)).astype(int),
            index=times,
        ).resample("ME").sum()

        monthly = pd.DataFrame(data)
        monthly["valid_quality_hours"] = valid_qs
        monthly = monthly[monthly["valid_quality_hours"] > 0].copy()
        if monthly.empty:
            continue
        monthly.insert(0, "plant", p)
        monthly.insert(1, "plant_id", plant_ids[p])
        monthly.insert(2, "upn", str(upns[p]) if p < len(upns) else "")
        monthly.insert(3, "kwp_used", float(kwp[p]) if np.isfinite(kwp[p]) else float("nan"))
        monthly.insert(4, "kwp_source", "real" if kwp_is_real[p] else "not_real")
        monthly.insert(5, "kwp_is_real", bool(kwp_is_real[p]))
        monthly.insert(6, "kwp_group", "real_kwp" if kwp_is_real[p] else "non_real_kwp")
        monthly.insert(
            7,
            "quality_capacity_scale",
            float(capacity_scale[p]) if capacity_scale.ndim == 1 and np.isfinite(capacity_scale[p]) else float("nan"),
        )
        monthly.insert(8, "date", monthly.index)
        monthly.insert(9, "calendar_month", monthly.index.month)
        rows.append(monthly.reset_index(drop=True))

    diagnostics["n_plants_total"] = int(ds.sizes["plant"])
    diagnostics["n_plants_analyzed"] = int(len(rows))
    diagnostics["n_plants_without_quality_months"] = int(ds.sizes["plant"] - len(rows))
    if not rows:
        return pd.DataFrame(), diagnostics
    return pd.concat(rows, ignore_index=True), diagnostics


def _build_quality_component_trends(
    quality_monthly: pd.DataFrame,
    min_months: int,
    alpha: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for metric in QUALITY_SCORE_METRICS:
        tr = _per_plant_trends(
            monthly_df=quality_monthly,
            value_col=metric,
            min_months=min_months,
            alpha=alpha,
            metric_name=f"{metric}_monthly",
        )
        tr.insert(0, "quality_metric", metric)
        tr.insert(1, "quality_metric_label", QUALITY_SCORE_COMPONENT_LABELS[metric])
        frames.append(tr)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _downshift_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    out = pd.Series(False, index=df.index)
    for col in ("best_break_shift_class", "half_shift_class"):
        if col in df.columns:
            vals = df[col].astype(str)
            out = out | (
                vals.str.contains("downshift", na=False)
                & ~vals.str.contains("no_downshift", na=False)
            )
    return out


def _correlation_summary(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return {"n": 0, "pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return {"n": int(mask.sum()), "pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    pearson = stats.pearsonr(x[mask], y[mask])
    spearman = stats.spearmanr(x[mask], y[mask])
    return {
        "n": int(mask.sum()),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def _build_quality_effectiveness_table(
    quality_shift: pd.DataFrame,
    performance_shift: pd.DataFrame | None,
    performance_label: str,
) -> pd.DataFrame:
    if quality_shift.empty or performance_shift is None or performance_shift.empty:
        return pd.DataFrame()

    q_cols = [
        c
        for c in (
            "plant",
            "plant_id",
            "quality_score_mean",
            "relative_change_pct_per_year",
            "kendall_tau",
            "half_delta_pct",
            "best_break_month",
            "break_delta_pct",
            "best_break_shift_class",
            "possible_step_change_with_plateau",
        )
        if c in quality_shift.columns
    ]
    p_cols = [
        c
        for c in (
            "plant",
            "plant_id",
            "relative_change_pct_per_year",
            "kendall_tau",
            "half_delta_pct",
            "best_break_month",
            "break_delta_pct",
            "best_break_shift_class",
            "possible_step_change_with_plateau",
        )
        if c in performance_shift.columns
    ]
    q = quality_shift[q_cols].rename(
        columns={
            "relative_change_pct_per_year": "quality_score_relative_change_pct_per_year",
            "kendall_tau": "quality_score_kendall_tau",
            "half_delta_pct": "quality_score_half_delta_pct",
            "best_break_month": "quality_score_best_break_month",
            "break_delta_pct": "quality_score_break_delta_pct",
            "best_break_shift_class": "quality_score_best_break_shift_class",
            "possible_step_change_with_plateau": "quality_score_possible_step_change_with_plateau",
        }
    )
    p = performance_shift[p_cols].rename(
        columns={
            "relative_change_pct_per_year": "performance_relative_change_pct_per_year",
            "kendall_tau": "performance_kendall_tau",
            "half_delta_pct": "performance_half_delta_pct",
            "best_break_month": "performance_best_break_month",
            "break_delta_pct": "performance_break_delta_pct",
            "best_break_shift_class": "performance_best_break_shift_class",
            "possible_step_change_with_plateau": "performance_possible_step_change_with_plateau",
        }
    )
    merged = q.merge(p, on=["plant", "plant_id"], how="inner")
    if merged.empty:
        return merged
    merged.insert(2, "performance_metric", performance_label)
    q_class = merged["quality_score_best_break_shift_class"].astype(str)
    p_class = merged["performance_best_break_shift_class"].astype(str)
    merged["quality_score_downshift"] = (
        q_class.str.contains("downshift", na=False)
        & ~q_class.str.contains("no_downshift", na=False)
    )
    merged["performance_downshift"] = (
        p_class.str.contains("downshift", na=False)
        & ~p_class.str.contains("no_downshift", na=False)
    )
    merged["both_quality_and_performance_downshift"] = (
        merged["quality_score_downshift"] & merged["performance_downshift"]
    )
    return merged


def _quality_effectiveness_summary(effect: pd.DataFrame) -> dict:
    n = int(len(effect))
    if n == 0:
        return {
            "n_pairs": 0,
            "correlation_quality_break_vs_performance_break": _correlation_summary(
                effect, "quality_score_break_delta_pct", "performance_break_delta_pct"
            ),
        }
    q_down = effect["quality_score_downshift"].fillna(False).astype(bool)
    p_down = effect["performance_downshift"].fillna(False).astype(bool)
    both = q_down & p_down
    return {
        "n_pairs": n,
        "n_quality_score_downshift": int(q_down.sum()),
        "n_performance_downshift": int(p_down.sum()),
        "n_both_quality_and_performance_downshift": int(both.sum()),
        "quality_downshift_precision_for_performance_downshift": (
            float(both.sum() / q_down.sum()) if int(q_down.sum()) else None
        ),
        "quality_downshift_recall_of_performance_downshift": (
            float(both.sum() / p_down.sum()) if int(p_down.sum()) else None
        ),
        "correlation_quality_break_vs_performance_break": _correlation_summary(
            effect, "quality_score_break_delta_pct", "performance_break_delta_pct"
        ),
    }


def _build_quality_score_summary(
    quality_monthly: pd.DataFrame,
    quality_shift: pd.DataFrame,
    component_trends: pd.DataFrame,
    diagnostics: dict,
    real_effect: pd.DataFrame,
    relative_effect: pd.DataFrame,
) -> dict:
    qs_down = _downshift_mask(quality_shift) if not quality_shift.empty else pd.Series(dtype=bool)
    component_summary: dict[str, dict] = {}
    if not component_trends.empty:
        for metric, g in component_trends.groupby("quality_metric"):
            rel = g["relative_change_pct_per_year"].to_numpy(dtype=float)
            rel = rel[np.isfinite(rel)]
            component_summary[str(metric)] = {
                "label": QUALITY_SCORE_COMPONENT_LABELS.get(str(metric), str(metric)),
                "n_plants": int(g["plant"].nunique()),
                "n_decreasing_p_lt_alpha": int(g["decreasing"].fillna(False).sum()),
                "median_relative_change_pct_per_year": float(np.median(rel)) if rel.size else None,
            }

    return {
        **diagnostics,
        "n_quality_score_plants_analyzed": int(len(quality_shift)),
        "n_quality_score_downshift": int(qs_down.sum()) if len(qs_down) else 0,
        "n_quality_score_significant_decline": int(
            quality_shift["significant_decline"].fillna(False).sum()
        ) if "significant_decline" in quality_shift.columns else 0,
        "component_trend_summary": component_summary,
        "real_kwp_effectiveness": _quality_effectiveness_summary(real_effect),
        "non_real_kwp_relative_effectiveness": _quality_effectiveness_summary(relative_effect),
        "interpretation": (
            "QS and m1..m5 are data-quality diagnostics, not PR metrics. A QS "
            "downshift is useful when it co-occurs with PR_PVGIS or relative_index "
            "downshift, because it suggests data/sensor/physics-coherence changes "
            "may explain the forecasting regime change. Lack of co-occurrence does "
            "not invalidate a performance shift; it can indicate real operating "
            "changes not captured by the QS components."
        ),
    }


def _plot_quality_score_outputs(
    out_dir: Path,
    quality_monthly: pd.DataFrame,
    quality_shift: pd.DataFrame,
    component_trends: pd.DataFrame,
    real_effect: pd.DataFrame,
    relative_effect: pd.DataFrame,
) -> None:
    if quality_monthly.empty:
        return
    plt.style.use("seaborn-v0_8-darkgrid")
    qm = quality_monthly.copy()
    qm["date"] = pd.to_datetime(qm["date"])

    fleet = qm.groupby("date", as_index=False)[list(QUALITY_SCORE_METRICS)].mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    for metric in QUALITY_SCORE_METRICS:
        ax.plot(fleet["date"], fleet[metric], marker="o", linewidth=1.3, label=metric)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Month")
    ax.set_ylabel("daytime monthly mean")
    ax.set_title("Fleet quality-score components over time")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "quality_score_fleet_monthly_components.png", dpi=180)
    plt.close(fig)

    if not quality_shift.empty and "relative_change_pct_per_year" in quality_shift.columns:
        rel = quality_shift["relative_change_pct_per_year"].to_numpy(dtype=float)
        rel = rel[np.isfinite(rel)]
        if rel.size:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            bins = np.linspace(min(float(rel.min()), -20.0), max(float(rel.max()), 10.0), 40)
            ax.hist(rel, bins=bins, color="tab:green", edgecolor="white", alpha=0.85)
            ax.axvline(0.0, color="black", linewidth=1.0)
            ax.set_xlabel("quality_score relative_change_pct_per_year")
            ax.set_ylabel("plant count")
            ax.set_title(f"Quality-score trend distribution (n={rel.size})")
            fig.tight_layout()
            fig.savefig(out_dir / "histogram_quality_score_change_pct_per_year.png", dpi=180)
            plt.close(fig)

    if not quality_shift.empty and "break_delta_pct" in quality_shift.columns:
        top = quality_shift.dropna(subset=["break_delta_pct"]).sort_values("break_delta_pct").head(20)
        if not top.empty:
            labels = [f"{row.plant} / {row.plant_id}" for row in top[["plant", "plant_id"]].itertuples(index=False)]
            fig, ax = plt.subplots(figsize=(10, max(5.0, 0.35 * len(top))))
            y = np.arange(len(top))
            ax.barh(y, top["break_delta_pct"].to_numpy(dtype=float), color="tab:green", alpha=0.78)
            ax.axvline(0.0, color="black", linewidth=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("quality_score best_break_delta_pct")
            ax.set_title("Top 20 quality-score downshift candidates")
            fig.tight_layout()
            fig.savefig(out_dir / "top20_quality_score_decline.png", dpi=180)
            plt.close(fig)

    if not component_trends.empty:
        comp = component_trends.groupby("quality_metric", as_index=False)[
            "relative_change_pct_per_year"
        ].median()
        comp = comp[comp["quality_metric"].isin(QUALITY_SCORE_METRICS)]
        if not comp.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(
                comp["quality_metric"],
                comp["relative_change_pct_per_year"],
                color="tab:olive",
                alpha=0.8,
            )
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_ylabel("median relative_change_pct_per_year")
            ax.set_title("Median trend by QS component")
            fig.tight_layout()
            fig.savefig(out_dir / "quality_score_component_trend_bars.png", dpi=180)
            plt.close(fig)

    for effect, filename, title in (
        (
            real_effect,
            "scatter_quality_score_vs_real_pr_break_delta_pct.png",
            "Quality-score shift vs real-kWp PR_PVGIS shift",
        ),
        (
            relative_effect,
            "scatter_quality_score_vs_non_real_relative_break_delta_pct.png",
            "Quality-score shift vs non-real relative-index shift",
        ),
    ):
        if effect.empty:
            continue
        x = effect["quality_score_break_delta_pct"].to_numpy(dtype=float)
        y = effect["performance_break_delta_pct"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            continue
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(x[mask], y[mask], alpha=0.65, s=22, color="tab:green")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("quality_score break_delta_pct")
        ax.set_ylabel("performance break_delta_pct")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)


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
    level_shift_summary: dict | None = None,
    quality_score_summary: dict | None = None,
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
        "level_shift_summary": level_shift_summary,
        "quality_score_summary": quality_score_summary,
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
    parser.add_argument(
        "--half-shift-edges",
        default=",".join(str(e) for e in DEFAULT_HALF_SHIFT_EDGES),
        help=(
            "Ascending edges (% delta) for half-split classification. "
            "Default: -10,-5,-2  -> strong/moderate/mild/stable_or_improved."
        ),
    )
    parser.add_argument(
        "--half-shift-labels",
        default=",".join(DEFAULT_HALF_SHIFT_LABELS),
    )
    parser.add_argument(
        "--break-shift-edges",
        default=",".join(str(e) for e in DEFAULT_BREAK_SHIFT_EDGES),
        help=(
            "Ascending edges (% delta) for best-break classification. "
            "Default: -10,-5,-2  -> strong/moderate/mild/no_downshift."
        ),
    )
    parser.add_argument(
        "--break-shift-labels",
        default=",".join(DEFAULT_BREAK_SHIFT_LABELS),
    )
    parser.add_argument(
        "--min-pre-months",
        type=int,
        default=3,
        help="Minimum monthly points before a candidate break.",
    )
    parser.add_argument(
        "--min-post-months",
        type=int,
        default=3,
        help="Minimum monthly points after a candidate break.",
    )
    parser.add_argument(
        "--break-selection",
        choices=("score", "max_drop"),
        default="score",
        help=(
            "How to pick the best break. 'score' = max(|delta_pct| - "
            "cv_penalty * post_cv_pct) over negative-delta candidates. "
            "'max_drop' = most negative break_delta_pct."
        ),
    )
    parser.add_argument(
        "--break-cv-penalty",
        type=float,
        default=0.5,
        help="Weight applied to post-break CV (in percent) in 'score' selection.",
    )
    parser.add_argument(
        "--plateau-break-pct-threshold",
        type=float,
        default=-5.0,
        help="break_delta_pct below this is required to flag plateau step change.",
    )
    parser.add_argument(
        "--plateau-post-cv-max",
        type=float,
        default=0.10,
        help="Maximum post-break CV (std/mean) to call the post period a plateau.",
    )
    parser.add_argument(
        "--analyze-non-real-kwp",
        action="store_true",
        default=True,
        help=(
            "Also run a proxy-free intra-plant relative-index analysis on "
            "plants without registry kWp. Outputs are kept separate."
        ),
    )
    parser.add_argument(
        "--skip-non-real-kwp",
        dest="analyze_non_real_kwp",
        action="store_false",
        help="Disable the relative-index analysis on plants without real kWp.",
    )
    parser.add_argument(
        "--non-real-baseline-strategy",
        choices=DEFAULT_BASELINE_STRATEGIES,
        default="median_first_n",
        help=(
            "Per-plant baseline for the relative index. "
            "median_first_n (default) anchors the comparison to the early "
            "period and is robust to a single anomalous month; "
            "mean_first_n is similar but less robust to outliers; "
            "median_all would absorb a mid-period downshift into the "
            "denominator and is not recommended for shift detection."
        ),
    )
    parser.add_argument(
        "--non-real-baseline-n-months",
        type=int,
        default=3,
        help="N for median_first_n / mean_first_n baseline strategies.",
    )
    parser.add_argument(
        "--overprediction-bias-pct",
        "--non-real-overprediction-bias-pct",
        dest="overprediction_bias_pct",
        type=float,
        default=5.0,
        help=(
            "expected_bias_pct_if_train_pre_break >= this is flagged as "
            "'overprediction risk if trained on pre-break regime'. Applies "
            "to both real-kWp PR_PVGIS and non-real-kWp relative_index summaries. "
            "The legacy --non-real-overprediction-bias-pct alias is kept for "
            "backward compatibility."
        ),
    )
    parser.add_argument(
        "--analyze-quality-score-trend",
        action="store_true",
        default=True,
        help=(
            "Compute monthly trends and level shifts for QS and m1..m5 "
            "quality-score components, then compare QS downshifts with "
            "performance downshifts."
        ),
    )
    parser.add_argument(
        "--skip-quality-score-trend",
        dest="analyze_quality_score_trend",
        action="store_false",
        help="Disable the QS/m1..m5 trend diagnostics.",
    )
    parser.add_argument(
        "--quality-score-window",
        type=int,
        default=720,
        help="Rolling window in hours passed to compute_qs for m1..m5.",
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

    half_shift_edges = tuple(
        float(x) for x in str(args.half_shift_edges).split(",") if x.strip() != ""
    )
    half_shift_labels = tuple(
        s.strip() for s in str(args.half_shift_labels).split(",") if s.strip() != ""
    )
    if len(half_shift_labels) != len(half_shift_edges) + 1:
        raise ValueError("--half-shift-labels must have len(edges)+1 entries")
    break_shift_edges = tuple(
        float(x) for x in str(args.break_shift_edges).split(",") if x.strip() != ""
    )
    break_shift_labels = tuple(
        s.strip() for s in str(args.break_shift_labels).split(",") if s.strip() != ""
    )
    if len(break_shift_labels) != len(break_shift_edges) + 1:
        raise ValueError("--break-shift-labels must have len(edges)+1 entries")

    level_shift_table = _build_level_shift_table(
        monthly_df=monthly_df,
        all_plant_trends=all_plant_trends,
        half_shift_edges=half_shift_edges,
        half_shift_labels=half_shift_labels,
        break_shift_edges=break_shift_edges,
        break_shift_labels=break_shift_labels,
        min_pre_months=args.min_pre_months,
        min_post_months=args.min_post_months,
        cv_penalty=args.break_cv_penalty,
        break_selection=args.break_selection,
        plateau_break_pct_threshold=args.plateau_break_pct_threshold,
        plateau_post_cv_max=args.plateau_post_cv_max,
    )
    level_shift_summary = _build_level_shift_summary(
        merged=level_shift_table,
        half_shift_labels=half_shift_labels,
        break_shift_labels=break_shift_labels,
        alpha=args.alpha,
        overprediction_bias_pct_threshold=args.overprediction_bias_pct,
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

    # Persist real-kWp result also under the requested explicit name.
    level_shift_table.to_csv(out_dir / "real_kwp_plant_pr_trends.csv", index=False)
    level_shift_table.to_csv(out_dir / "all_plant_pr_trends_with_shift.csv", index=False)
    plateau_mask = level_shift_table["possible_step_change_with_plateau"].fillna(False).astype(bool)
    downshift_mask = (
        level_shift_table["best_break_shift_class"].isin(
            ["strong_downshift", "moderate_downshift", "mild_downshift"]
        )
        | level_shift_table["half_shift_class"].isin(
            ["strong_downshift_gt_10", "moderate_downshift_5_10", "mild_downshift_2_5"]
        )
    )
    candidate_mask = plateau_mask | downshift_mask
    level_shift_table[candidate_mask].sort_values("break_delta_pct").to_csv(
        out_dir / "plant_level_shift_candidates.csv", index=False
    )
    with open(out_dir / "level_shift_summary.json", "w", encoding="utf-8") as f:
        json.dump(level_shift_summary, f, indent=2)
    _plot_level_shift(out_dir, level_shift_table)

    # -----------------------------------------------------------------------
    # Non-real-kWp: proxy-free intra-plant relative-index analysis
    # -----------------------------------------------------------------------
    relative_summary: dict | None = None
    relative_decline_distribution: dict | None = None
    relative_level_shift_summary: dict | None = None
    relative_diagnostics: dict | None = None
    level_shift_rel: pd.DataFrame | None = None
    if args.analyze_non_real_kwp:
        monthly_rel, relative_diagnostics = _build_relative_monthly(
            ds=ds,
            kwp_is_real=kwp_is_real,
            daytime_poa_threshold=args.daytime_poa_threshold,
            reference_pr=args.reference_pr,
            min_day_hours=args.min_day_hours,
            min_month_hours=args.min_month_hours,
            baseline_strategy=args.non_real_baseline_strategy,
            baseline_n_months=args.non_real_baseline_n_months,
            min_valid_months=args.min_months,
        )

        if monthly_rel.empty:
            print(
                "\n[non-real-kWp] no plants without real kWp produced a "
                "usable relative-index monthly series; skipping relative analysis."
            )
        else:
            monthly_rel = _add_plant_standardization(
                monthly_rel,
                value_col="relative_index",
                output_col="relative_index_z",
                stats_prefix="plant_relative_index",
            )
            plant_trends_rel = _per_plant_trends(
                monthly_df=monthly_rel,
                value_col="relative_index",
                min_months=args.min_months,
                alpha=args.alpha,
                metric_name="relative_index_monthly",
                extra_metrics={"relative_index_z": "relative_index_z_monthly"},
            )
            candidate_rel = _plant_candidate_summary(
                monthly_df=monthly_rel,
                plant_trends=plant_trends_rel,
                value_col="relative_index",
                metric_name="relative_index_monthly",
                output_prefix="relative_index",
                z_metric_name="relative_index_z_monthly",
                z_output_prefix="relative_index_z",
                stats_prefix="plant_relative_index",
            )
            all_plant_trends_rel = _build_all_plant_trend_table(
                candidate_summary=candidate_rel,
                decline_edges=decline_edges,
                decline_labels=decline_labels,
                monotonic_strong=args.monotonic_strong_tau,
                monotonic_directional=args.monotonic_directional_tau,
                alpha=args.alpha,
                output_prefix="relative_index",
            )
            level_shift_rel = _build_level_shift_table(
                monthly_df=monthly_rel,
                all_plant_trends=all_plant_trends_rel,
                half_shift_edges=half_shift_edges,
                half_shift_labels=half_shift_labels,
                break_shift_edges=break_shift_edges,
                break_shift_labels=break_shift_labels,
                min_pre_months=args.min_pre_months,
                min_post_months=args.min_post_months,
                cv_penalty=args.break_cv_penalty,
                break_selection=args.break_selection,
                plateau_break_pct_threshold=args.plateau_break_pct_threshold,
                plateau_post_cv_max=args.plateau_post_cv_max,
                value_col="relative_index",
                mean_prefix="index",
                rename_break_means=True,
            )
            relative_decline_distribution = _build_decline_distribution_summary(
                all_plants=all_plant_trends_rel,
                decline_labels=decline_labels,
                monotonic_strong=args.monotonic_strong_tau,
                monotonic_directional=args.monotonic_directional_tau,
                alpha=args.alpha,
                metric_label="relative_index",
            )
            relative_level_shift_summary = _build_level_shift_summary(
                merged=level_shift_rel,
                half_shift_labels=half_shift_labels,
                break_shift_labels=break_shift_labels,
                alpha=args.alpha,
                overprediction_bias_pct_threshold=args.overprediction_bias_pct,
            )

            bias_pct = level_shift_rel.get(
                "expected_bias_pct_if_train_pre_break"
            )
            if bias_pct is not None:
                overpred_mask = bias_pct.fillna(0.0) >= args.overprediction_bias_pct
                n_overpred = int(overpred_mask.sum())
            else:
                n_overpred = 0

            plateau_mask_rel = level_shift_rel["possible_step_change_with_plateau"].fillna(False).astype(bool)
            downshift_mask_rel = (
                level_shift_rel["best_break_shift_class"].isin(
                    ["strong_downshift", "moderate_downshift", "mild_downshift"]
                )
                | level_shift_rel["half_shift_class"].isin(
                    [
                        "strong_downshift_gt_10",
                        "moderate_downshift_5_10",
                        "mild_downshift_2_5",
                    ]
                )
            )
            candidate_mask_rel = plateau_mask_rel | downshift_mask_rel

            # Save CSVs
            monthly_rel.to_csv(
                out_dir / "non_real_kwp_relative_monthly.csv", index=False
            )
            level_shift_rel.to_csv(
                out_dir / "non_real_kwp_relative_trends.csv", index=False
            )
            level_shift_rel[candidate_mask_rel].sort_values("break_delta_pct").to_csv(
                out_dir / "non_real_kwp_level_shift_candidates.csv", index=False
            )

            relative_summary = {
                "n_non_real_kwp_plants_total": relative_diagnostics[
                    "n_non_real_kwp_plants_total"
                ],
                "n_non_real_kwp_plants_analyzed": int(len(level_shift_rel)),
                "n_excluded_insufficient_data": relative_diagnostics[
                    "n_excluded_insufficient_data"
                ],
                "n_excluded_no_data": relative_diagnostics["n_excluded_no_data"],
                "n_excluded_too_few_months": relative_diagnostics[
                    "n_excluded_too_few_months"
                ],
                "n_excluded_no_baseline": relative_diagnostics["n_excluded_no_baseline"],
                "baseline_strategy_requested": args.non_real_baseline_strategy,
                "baseline_n_months_requested": args.non_real_baseline_n_months,
                "baseline_default": (
                    "Median of the first 3 valid monthly apparent-capacity values "
                    "when available; otherwise median of all valid months as a "
                    "documented fallback."
                ),
                "overprediction_bias_pct_threshold": args.overprediction_bias_pct,
                "n_overprediction_risk_if_trained_pre_break": n_overpred,
                "decline_class_counts": relative_decline_distribution[
                    "decline_class_counts"
                ],
                "decline_class_pct": relative_decline_distribution["decline_class_pct"],
                "monotonic_class_counts": relative_decline_distribution[
                    "monotonic_class_counts"
                ],
                "monotonic_class_pct": relative_decline_distribution[
                    "monotonic_class_pct"
                ],
                "half_shift_class_counts": relative_level_shift_summary[
                    "half_shift_class_counts"
                ],
                "half_shift_class_pct": relative_level_shift_summary[
                    "half_shift_class_pct"
                ],
                "best_break_shift_class_counts": relative_level_shift_summary[
                    "best_break_shift_class_counts"
                ],
                "best_break_shift_class_pct": relative_level_shift_summary[
                    "best_break_shift_class_pct"
                ],
                "n_possible_step_change_with_plateau": relative_level_shift_summary[
                    "n_possible_step_change_with_plateau"
                ],
                "n_downshift_but_not_monotonic_decline": relative_level_shift_summary[
                    "n_downshift_but_not_monotonic_decline"
                ],
                "n_downshift_but_not_significant_slope": relative_level_shift_summary[
                    "n_downshift_but_not_significant_slope"
                ],
                "decline_distribution": relative_decline_distribution,
                "level_shift": relative_level_shift_summary,
                "methodology_caveat": (
                    "The 94 plants with registry kWp can be analysed with "
                    "absolute PR_PVGIS. The plants without registry kWp cannot "
                    "be assigned a reliable absolute PR because installed "
                    "nominal power is unknown. For those plants this script "
                    "computes apparent_capacity = actual_kwh / "
                    "expected_pvgis_per_kwp on valid daytime monthly sums, "
                    "then normalizes it by a per-plant baseline to obtain "
                    "relative_index. The default baseline is the median of "
                    "the first 3 valid months; if fewer baseline months are "
                    "available but the plant still passes the minimum-data "
                    "rule, the fallback is the median of all valid months and "
                    "the row is marked through baseline_method and "
                    "baseline_n_months. relative_index is meaningful only "
                    "within the same plant and must not be used for absolute "
                    "comparisons between plants. These results are "
                    "exploratory. Apparent downshifts are not automatically "
                    "physical degradation; possible causes include soiling, "
                    "faults, availability losses, curtailment, clipping, data "
                    "problems, operational changes, residual seasonality or "
                    "physical degradation."
                ),
                "forecasting_note": (
                    "Even without real kWp, the relative_index identifies "
                    "plants undergoing an apparent regime change. A model "
                    "trained on the pre-break regime can over- or under-"
                    "predict the post-break regime by roughly "
                    "expected_bias_pct_if_train_pre_break (positive = "
                    "overprediction of post-break production). This subset "
                    "is therefore a useful stratifier when diagnosing "
                    "forecast degradation between an early-period train and "
                    "a late-period test."
                ),
            }
            with open(
                out_dir / "non_real_kwp_relative_summary.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(relative_summary, f, indent=2)

            _plot_relative_outputs(
                out_dir=out_dir,
                all_plant_trends_rel=all_plant_trends_rel,
                level_shift_rel=level_shift_rel,
                decline_edges=decline_edges,
            )

            # combined descriptive summary (no cross-population comparison)
            combined_summary = {
                "real_kwp": {
                    "n_plants_analyzed": (
                        decline_distribution["n_plants_analyzed"]
                        if decline_distribution else 0
                    ),
                    "decline_class_counts": (
                        decline_distribution["decline_class_counts"]
                        if decline_distribution else {}
                    ),
                    "level_shift": level_shift_summary,
                    "overprediction_bias_pct_threshold": (
                        level_shift_summary["overprediction_bias_pct_threshold"]
                        if level_shift_summary else args.overprediction_bias_pct
                    ),
                    "n_overprediction_risk_if_trained_pre_break": (
                        level_shift_summary[
                            "n_overprediction_risk_if_trained_pre_break"
                        ]
                        if level_shift_summary else 0
                    ),
                },
                "non_real_kwp_relative_index": {
                    "n_plants_analyzed": relative_summary[
                        "n_non_real_kwp_plants_analyzed"
                    ],
                    "decline_class_counts": (
                        relative_decline_distribution["decline_class_counts"]
                        if relative_decline_distribution else {}
                    ),
                    "level_shift": relative_level_shift_summary,
                    "overprediction_bias_pct_threshold": args.overprediction_bias_pct,
                    "n_overprediction_risk_if_trained_pre_break": n_overpred,
                },
                "note": (
                    "The two populations are reported side by side for "
                    "context only. Decline classes are not directly "
                    "comparable: real-kWp uses absolute PR_PVGIS, non-real-"
                    "kWp uses an intra-plant relative index. Use real-kWp "
                    "numbers for absolute statements; use non-real-kWp "
                    "numbers only for intra-plant temporal change."
                ),
            }
            with open(
                out_dir / "combined_real_and_relative_summary.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(combined_summary, f, indent=2)

    # -----------------------------------------------------------------------
    # Quality-score diagnostics: QS + m1..m5 trends and effectiveness check
    # -----------------------------------------------------------------------
    quality_score_summary: dict | None = None
    quality_monthly = pd.DataFrame()
    quality_shift = pd.DataFrame()
    quality_component_trends = pd.DataFrame()
    quality_real_effect = pd.DataFrame()
    quality_relative_effect = pd.DataFrame()
    if args.analyze_quality_score_trend:
        quality_monthly, quality_diag = _build_quality_score_monthly(
            ds=ds,
            kwp=kwp,
            kwp_is_real=kwp_is_real,
            daytime_poa_threshold=args.daytime_poa_threshold,
            qs_window=args.quality_score_window,
        )
        if quality_monthly.empty:
            quality_score_summary = {
                **quality_diag,
                "n_quality_score_plants_analyzed": 0,
                "interpretation": (
                    "Quality-score trend analysis was skipped because no monthly "
                    "QS/m1..m5 series could be produced."
                ),
            }
            with open(out_dir / "quality_score_summary.json", "w", encoding="utf-8") as f:
                json.dump(quality_score_summary, f, indent=2)
            print(
                "\n[quality-score] no usable QS/m1..m5 monthly series; "
                "skipping QS trend diagnostics."
            )
        else:
            quality_component_trends = _build_quality_component_trends(
                quality_monthly=quality_monthly,
                min_months=args.min_months,
                alpha=args.alpha,
            )
            quality_primary_trends = _per_plant_trends(
                monthly_df=quality_monthly,
                value_col="quality_score",
                min_months=args.min_months,
                alpha=args.alpha,
                metric_name="quality_score_monthly",
            )
            quality_candidates = _plant_candidate_summary(
                monthly_df=quality_monthly,
                plant_trends=quality_primary_trends,
                value_col="quality_score",
                metric_name="quality_score_monthly",
                output_prefix="quality_score",
                z_metric_name=None,
                stats_prefix="plant_quality_score",
            )
            quality_all_trends = _build_all_plant_trend_table(
                candidate_summary=quality_candidates,
                decline_edges=decline_edges,
                decline_labels=decline_labels,
                monotonic_strong=args.monotonic_strong_tau,
                monotonic_directional=args.monotonic_directional_tau,
                alpha=args.alpha,
                output_prefix="quality_score",
            )
            quality_shift = _build_level_shift_table(
                monthly_df=quality_monthly,
                all_plant_trends=quality_all_trends,
                half_shift_edges=half_shift_edges,
                half_shift_labels=half_shift_labels,
                break_shift_edges=break_shift_edges,
                break_shift_labels=break_shift_labels,
                min_pre_months=args.min_pre_months,
                min_post_months=args.min_post_months,
                cv_penalty=args.break_cv_penalty,
                break_selection=args.break_selection,
                plateau_break_pct_threshold=args.plateau_break_pct_threshold,
                plateau_post_cv_max=args.plateau_post_cv_max,
                value_col="quality_score",
                mean_prefix="quality_score",
                rename_break_means=True,
            )
            quality_real_effect = _build_quality_effectiveness_table(
                quality_shift=quality_shift,
                performance_shift=level_shift_table,
                performance_label="real_kwp_pr_pvgis",
            )
            quality_relative_effect = _build_quality_effectiveness_table(
                quality_shift=quality_shift,
                performance_shift=level_shift_rel,
                performance_label="non_real_kwp_relative_index",
            )
            quality_score_summary = _build_quality_score_summary(
                quality_monthly=quality_monthly,
                quality_shift=quality_shift,
                component_trends=quality_component_trends,
                diagnostics=quality_diag,
                real_effect=quality_real_effect,
                relative_effect=quality_relative_effect,
            )

            quality_monthly.to_csv(out_dir / "quality_score_monthly.csv", index=False)
            quality_shift.to_csv(out_dir / "quality_score_trends.csv", index=False)
            quality_component_trends.to_csv(
                out_dir / "quality_score_component_trends.csv", index=False
            )
            if not quality_shift.empty and "break_delta_pct" in quality_shift.columns:
                quality_candidates_mask = _downshift_mask(quality_shift)
                if "possible_step_change_with_plateau" in quality_shift.columns:
                    quality_candidates_mask = quality_candidates_mask | quality_shift[
                        "possible_step_change_with_plateau"
                    ].fillna(False).astype(bool)
                quality_candidates = quality_shift[quality_candidates_mask].sort_values(
                    "break_delta_pct"
                )
            else:
                quality_candidates = pd.DataFrame()
            quality_candidates.to_csv(
                out_dir / "quality_score_level_shift_candidates.csv", index=False
            )
            quality_effect = pd.concat(
                [quality_real_effect, quality_relative_effect],
                ignore_index=True,
            )
            quality_effect.to_csv(
                out_dir / "quality_score_vs_performance_shift.csv", index=False
            )
            with open(out_dir / "quality_score_summary.json", "w", encoding="utf-8") as f:
                json.dump(quality_score_summary, f, indent=2)

            _plot_quality_score_outputs(
                out_dir=out_dir,
                quality_monthly=quality_monthly,
                quality_shift=quality_shift,
                component_trends=quality_component_trends,
                real_effect=quality_real_effect,
                relative_effect=quality_relative_effect,
            )

    if relative_summary is not None:
        combined_summary = {
            "real_kwp": {
                "n_plants_analyzed": (
                    decline_distribution["n_plants_analyzed"]
                    if decline_distribution else 0
                ),
                "decline_class_counts": (
                    decline_distribution["decline_class_counts"]
                    if decline_distribution else {}
                ),
                "level_shift": level_shift_summary,
                "overprediction_bias_pct_threshold": (
                    level_shift_summary["overprediction_bias_pct_threshold"]
                    if level_shift_summary else args.overprediction_bias_pct
                ),
                "n_overprediction_risk_if_trained_pre_break": (
                    level_shift_summary[
                        "n_overprediction_risk_if_trained_pre_break"
                    ]
                    if level_shift_summary else 0
                ),
            },
            "non_real_kwp_relative_index": {
                "n_plants_analyzed": relative_summary[
                    "n_non_real_kwp_plants_analyzed"
                ],
                "decline_class_counts": (
                    relative_decline_distribution["decline_class_counts"]
                    if relative_decline_distribution else {}
                ),
                "level_shift": relative_level_shift_summary,
                "overprediction_bias_pct_threshold": args.overprediction_bias_pct,
                "n_overprediction_risk_if_trained_pre_break": relative_summary[
                    "n_overprediction_risk_if_trained_pre_break"
                ],
            },
            "quality_score_diagnostics": quality_score_summary,
            "note": (
                "The real-kWp, non-real-kWp and quality-score diagnostics are "
                "reported side by side for context only. PR_PVGIS is an absolute "
                "performance metric for real-kWp plants; relative_index is an "
                "intra-plant temporal metric for non-real-kWp plants; QS/m1..m5 "
                "are data-quality diagnostics."
            ),
        }
        with open(
            out_dir / "combined_real_and_relative_summary.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(combined_summary, f, indent=2)

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
        level_shift_summary=level_shift_summary,
        quality_score_summary=quality_score_summary,
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

    print("\nLevel-shift / regime-shift distribution:")
    print(f"  n_plants_analyzed: {level_shift_summary['n_plants_analyzed']}")
    print("  half_shift_class:")
    for lbl, cnt in level_shift_summary["half_shift_class_counts"].items():
        pct = level_shift_summary["half_shift_class_pct"].get(lbl, 0.0)
        print(f"    {lbl:<30s} {cnt:>4d}  ({pct:5.1f}%)")
    print("  best_break_shift_class:")
    for lbl, cnt in level_shift_summary["best_break_shift_class_counts"].items():
        pct = level_shift_summary["best_break_shift_class_pct"].get(lbl, 0.0)
        print(f"    {lbl:<30s} {cnt:>4d}  ({pct:5.1f}%)")
    print(
        f"  possible_step_change_with_plateau: "
        f"{level_shift_summary['n_possible_step_change_with_plateau']} "
        f"({level_shift_summary['pct_possible_step_change_with_plateau']:.1f}%)"
    )
    print(
        f"    of which NOT monotonic decline: "
        f"{level_shift_summary['n_plateau_and_not_monotonic_decline']}"
    )
    print(
        f"    of which NOT significant (p>={args.alpha} or slope>=0): "
        f"{level_shift_summary['n_plateau_and_not_significant_decline']}"
    )
    if level_shift_summary["plateau_candidates_by_decline_class"]:
        print("  plateau candidates by existing decline_class:")
        for lbl, cnt in level_shift_summary["plateau_candidates_by_decline_class"].items():
            print(f"    {lbl:<30s} {cnt}")
    print(
        f"  overprediction risk if trained pre-break "
        f"(expected_bias_pct >= {args.overprediction_bias_pct}): "
        f"{level_shift_summary['n_overprediction_risk_if_trained_pre_break']}"
    )

    top_shift_cols = [
        c
        for c in (
            "plant",
            "plant_id",
            "upn",
            "valid_months",
            "relative_change_pct_per_year",
            "kendall_tau",
            "half_delta_pct",
            "half_shift_class",
            "best_break_month",
            "break_delta_pct",
            "post_break_cv",
            "best_break_shift_class",
            "possible_step_change_with_plateau",
            "expected_bias_pct_if_train_pre_break",
        )
        if c in level_shift_table.columns
    ]
    top_shift = level_shift_table.sort_values("break_delta_pct").head(20)
    print("\nTop 20 level-shift candidates (most negative break_delta_pct):")
    with pd.option_context("display.max_columns", 200, "display.width", 200):
        print(top_shift[top_shift_cols].to_string(index=False))

    print(
        "\nCAVEAT (level shift): a level / regime shift is not a proof of "
        "physical module degradation. Possible causes include soiling, "
        "inverter or string faults, availability losses, curtailment, "
        "clipping, data quality changes, operational changes (cleaning, "
        "replacements, configuration), residual seasonality, and - among "
        "other causes - physical degradation. Use the label 'apparent "
        "performance downshift', not 'degradation'."
    )
    print(
        "Forecasting note: if a model is trained on pre-break months and "
        "evaluated on post-break months for the same plant, expected bias "
        "is roughly expected_bias_pct_if_train_pre_break (positive = "
        "over-prediction of post-break production)."
    )

    if args.analyze_non_real_kwp and relative_summary is not None:
        print("\n--- Non-real-kWp relative-index analysis ---")
        print(
            f"  candidates (kwp not real): "
            f"{relative_summary['n_non_real_kwp_plants_total']}"
        )
        print(
            f"  analyzed:                 "
            f"{relative_summary['n_non_real_kwp_plants_analyzed']}"
        )
        print(
            f"  excluded insufficient:    "
            f"{relative_summary['n_excluded_insufficient_data']}"
        )
        print(f"    no data:                {relative_summary['n_excluded_no_data']}")
        print(f"    too few months:         {relative_summary['n_excluded_too_few_months']}")
        print(f"    no baseline:            {relative_summary['n_excluded_no_baseline']}")
        rd = relative_summary["decline_distribution"]
        if rd:
            print("  decline_class (relative slope):")
            for lbl, cnt in rd["decline_class_counts"].items():
                pct = rd["decline_class_pct"].get(lbl, 0.0)
                print(f"    {lbl:<28s} {cnt:>4d}  ({pct:5.1f}%)")
            print("  monotonic_class (relative):")
            for lbl, cnt in rd["monotonic_class_counts"].items():
                pct = rd["monotonic_class_pct"].get(lbl, 0.0)
                print(f"    {lbl:<40s} {cnt:>4d}  ({pct:5.1f}%)")
        rs = relative_summary["level_shift"]
        if rs:
            print("  half_shift_class (relative):")
            for lbl, cnt in rs["half_shift_class_counts"].items():
                pct = rs["half_shift_class_pct"].get(lbl, 0.0)
                print(f"    {lbl:<30s} {cnt:>4d}  ({pct:5.1f}%)")
            print("  best_break_shift_class (relative):")
            for lbl, cnt in rs["best_break_shift_class_counts"].items():
                pct = rs["best_break_shift_class_pct"].get(lbl, 0.0)
                print(f"    {lbl:<30s} {cnt:>4d}  ({pct:5.1f}%)")
            print(
                f"  possible_step_change_with_plateau: "
                f"{relative_summary['n_possible_step_change_with_plateau']} "
                f"({rs['pct_possible_step_change_with_plateau']:.1f}%)"
            )
            print(
                f"  downshift but NOT monotonic decline: "
                f"{relative_summary['n_downshift_but_not_monotonic_decline']}"
            )
            print(
                f"  downshift but NOT significant slope: "
                f"{relative_summary['n_downshift_but_not_significant_slope']}"
            )
        print(
            f"  overprediction risk if trained pre-break "
            f"(expected_bias_pct >= {args.overprediction_bias_pct}): "
            f"{relative_summary['n_overprediction_risk_if_trained_pre_break']}"
        )
        print("\nMETHODOLOGY (non-real-kWp):")
        print(relative_summary["methodology_caveat"])
        print("\nFORECASTING NOTE (non-real-kWp):")
        print(relative_summary["forecasting_note"])

    if args.analyze_quality_score_trend and quality_score_summary is not None:
        print("\n--- Quality-score trend diagnostics (QS, m1..m5) ---")
        print(
            f"  analyzed plants: "
            f"{quality_score_summary.get('n_quality_score_plants_analyzed', 0)}"
        )
        print(
            f"  QS downshift candidates: "
            f"{quality_score_summary.get('n_quality_score_downshift', 0)}"
        )
        print(
            f"  QS significant decline: "
            f"{quality_score_summary.get('n_quality_score_significant_decline', 0)}"
        )
        real_eff = quality_score_summary.get("real_kwp_effectiveness", {})
        if real_eff:
            print(
                "  real-kWp effectiveness: "
                f"pairs={real_eff.get('n_pairs', 0)}, "
                f"both_downshift={real_eff.get('n_both_quality_and_performance_downshift', 0)}, "
                f"recall={real_eff.get('quality_downshift_recall_of_performance_downshift')}"
            )
        rel_eff = quality_score_summary.get("non_real_kwp_relative_effectiveness", {})
        if rel_eff:
            print(
                "  non-real relative effectiveness: "
                f"pairs={rel_eff.get('n_pairs', 0)}, "
                f"both_downshift={rel_eff.get('n_both_quality_and_performance_downshift', 0)}, "
                f"recall={rel_eff.get('quality_downshift_recall_of_performance_downshift')}"
            )
        print("  metrics: QS aggregate plus m1 corr, m2 bias, m3 completeness, m4 variance, m5 eta.")

    print("\nKey files:")
    print("  plant_monthly_performance.csv")
    print("  plant_candidate_summary.csv")
    print("  plant_trend_summary.csv")
    print("  fleet_trend_summary.csv")
    print("  all_plant_pr_trends.csv             [NEW: full per-plant table]")
    print("  all_plant_decline_classes.csv       [NEW: per-plant classification]")
    print("  decline_distribution_summary.json   [NEW: counts + percentages]")
    print("  all_plant_relative_change_histogram.png  [NEW: trend distribution]")
    print("  all_plant_pr_trends_with_shift.csv  [NEW: full table + level-shift]")
    print("  plant_level_shift_candidates.csv    [NEW: shift candidates only]")
    print("  level_shift_summary.json            [NEW: half + break breakdown]")
    print("  level_shift_half_delta_pct_histogram.png   [NEW]")
    print("  level_shift_break_delta_pct_histogram.png  [NEW]")
    print("  level_shift_scatter_slope_vs_half.png      [NEW]")
    print("  level_shift_scatter_tau_vs_break.png       [NEW]")
    print("  real_kwp_plant_pr_trends.csv                [real-kWp 94 plants]")
    print("  non_real_kwp_relative_monthly.csv           [NEW: relative index]")
    print("  non_real_kwp_relative_trends.csv            [NEW: trend + shift]")
    print("  non_real_kwp_level_shift_candidates.csv     [NEW: shift candidates]")
    print("  non_real_kwp_relative_summary.json          [NEW: summary]")
    print("  combined_real_and_relative_summary.json     [NEW: side by side]")
    print("  histogram_relative_change_pct_per_year_non_real.png [NEW]")
    print("  histogram_best_break_delta_pct_non_real.png          [NEW]")
    print("  scatter_kendall_tau_vs_best_break_delta_pct_non_real.png [NEW]")
    print("  scatter_slope_vs_half_delta_pct_non_real.png         [NEW]")
    print("  top20_level_shift_candidates_non_real.png            [NEW]")
    print("  quality_score_monthly.csv                            [NEW: QS+m1..m5 monthly]")
    print("  quality_score_trends.csv                             [NEW: QS trend + shift]")
    print("  quality_score_component_trends.csv                   [NEW: m1..m5 trends]")
    print("  quality_score_level_shift_candidates.csv             [NEW: QS shift candidates]")
    print("  quality_score_vs_performance_shift.csv               [NEW: QS effectiveness]")
    print("  quality_score_summary.json                           [NEW: QS effectiveness summary]")
    print("  quality_score_fleet_monthly_components.png           [NEW]")
    print("  histogram_quality_score_change_pct_per_year.png      [NEW]")
    print("  quality_score_component_trend_bars.png               [NEW]")
    print("  scatter_quality_score_vs_real_pr_break_delta_pct.png [NEW]")
    print("  scatter_quality_score_vs_non_real_relative_break_delta_pct.png [NEW]")
    print("  top20_quality_score_decline.png                      [NEW]")
    print("  fleet_pvgis_pr_trend.png")
    print("  individual_candidate_plant_trends.png")


if __name__ == "__main__":
    main()
