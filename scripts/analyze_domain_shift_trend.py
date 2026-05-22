"""
Analyze long-term PV performance decline as a domain-shift signal.

This script is intentionally separate from the forecasting notebook. It does
not evaluate ST-GNN error; it evaluates the plant production process itself:

1. PVGIS-normalized performance:
      PR_pvgis(t) = actual_energy(t) / (kWp * PVGIS_POA(t))

   If PR_pvgis trends downward, the plant/fleet produces less than expected
   under comparable irradiance. This is the closest signal to soiling,
   degradation or persistent domain shift.

2. Month-standardized performance:
      z(t) = (PR_pvgis(t) - mean(PR_pvgis | calendar_month)) /
             std(PR_pvgis | calendar_month)

   This removes month-level seasonality by comparing each observation with
   the typical value for the same calendar month. With only one year this is
   a weak diagnostic; with multiple years it becomes the preferred
   de-seasonalized trend.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

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
) -> tuple[np.ndarray, np.ndarray]:
    """Return kWp aligned to the dataset plant dimension plus a real-data mask."""
    n_plants = len(plant_ids)
    if not plant_mapping.exists() or not energy_coords.exists():
        return np.full(n_plants, np.nan), np.zeros(n_plants, dtype=bool)

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
    rel_year = float((slope_year / mean_y) * 100.0) if abs(mean_y) > 1e-12 else float("nan")
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
    daytime_poa_threshold: float,
    reference_pr: float,
    min_day_hours: int,
    min_month_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = pd.DatetimeIndex(ds.coords["time"].values)
    plant_ids = _safe_coord(ds, "plant_id", np.arange(ds.sizes["plant"]))
    energy = np.asarray(ds["ENERGIA"].values, dtype=np.float64)
    poa_wm2 = np.asarray(ds["solar_irradiance_poa"].values, dtype=np.float64)
    poa_kwm2 = np.clip(poa_wm2 / 1000.0, 0.0, None)
    day = poa_wm2 >= daytime_poa_threshold

    proxy = _capacity_proxy_kwp(energy, poa_kwm2, day)
    capacity = kwp.copy()
    missing = ~np.isfinite(capacity) | (capacity <= 0)
    capacity[missing] = proxy[missing]

    daily_rows: list[pd.DataFrame] = []
    monthly_rows: list[pd.DataFrame] = []
    for p in range(ds.sizes["plant"]):
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
            frame.insert(2, "kwp_used", capacity[p])
            frame.insert(3, "kwp_source", "real" if kwp_is_real[p] else "proxy")
            frame.insert(4, "date", frame.index)
            frame.insert(5, "calendar_month", frame.index.month)
        daily_rows.append(daily.reset_index(drop=True))
        monthly_rows.append(monthly.reset_index(drop=True))

    if not daily_rows or not monthly_rows:
        raise RuntimeError("No valid plant performance rows were produced.")

    daily_df = pd.concat(daily_rows, ignore_index=True)
    monthly_df = pd.concat(monthly_rows, ignore_index=True)
    return daily_df, monthly_df


def _add_month_standardization(daily_df: pd.DataFrame, monthly_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = daily_df.copy()
    valid = daily["pr_pvgis"].replace([np.inf, -np.inf], np.nan)
    daily["pr_pvgis"] = valid

    month_stats = (
        daily.dropna(subset=["pr_pvgis"])
        .groupby("calendar_month")["pr_pvgis"]
        .agg(month_mean="mean", month_std="std", month_n="count")
        .reset_index()
    )
    daily = daily.merge(month_stats, on="calendar_month", how="left")
    daily["pr_month_z"] = (daily["pr_pvgis"] - daily["month_mean"]) / daily["month_std"].replace(0.0, np.nan)

    # Aggregate standardized daily values to the same plant/month rows used for
    # the PVGIS trend. The extra indirection keeps daily CSVs useful for plots.
    daily["month_period"] = pd.to_datetime(daily["date"]).dt.to_period("M").astype(str)
    z_month = (
        daily.groupby(["plant", "plant_id", "month_period"], as_index=False)
        .agg(pr_month_z=("pr_month_z", "mean"), z_valid_days=("pr_month_z", "count"))
    )
    monthly = monthly_df.copy()
    monthly["date"] = pd.to_datetime(monthly["date"])
    monthly["month_period"] = monthly["date"].dt.to_period("M").astype(str)
    monthly = monthly.merge(z_month, on=["plant", "plant_id", "month_period"], how="left")
    return daily, monthly


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
            mean_pr_month_z=("pr_month_z", "mean"),
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
            mean_pr_month_z=("pr_month_z", "mean"),
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
        s_z = pd.Series(g["pr_month_z"].to_numpy(dtype=float), index=pd.to_datetime(g["date"]))
        plant_results.append(_fit_trend(s_pr, "pr_pvgis_monthly", plant, plant_id, min_months, alpha))
        plant_results.append(_fit_trend(s_z, "pr_month_z_monthly", plant, plant_id, min_months, alpha))

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


def _write_summary(
    out_dir: Path,
    args: argparse.Namespace,
    ds: xr.Dataset,
    kwp_is_real: np.ndarray,
    plant_trends: pd.DataFrame,
    fleet_trends: pd.DataFrame,
    fleet_monthly: pd.DataFrame,
) -> None:
    pr = plant_trends[plant_trends["metric"] == "pr_pvgis_monthly"]
    z = plant_trends[plant_trends["metric"] == "pr_month_z_monthly"]
    summary = {
        "period_start": str(pd.Timestamp(ds.time.values[0]).date()),
        "period_end": str(pd.Timestamp(ds.time.values[-1]).date()),
        "n_plants": int(ds.sizes["plant"]),
        "n_hours": int(ds.sizes["time"]),
        "n_real_kwp": int(np.sum(kwp_is_real)),
        "n_proxy_kwp": int(len(kwp_is_real) - np.sum(kwp_is_real)),
        "daytime_poa_threshold_wm2": args.daytime_poa_threshold,
        "reference_pr": args.reference_pr,
        "min_months": args.min_months,
        "alpha": args.alpha,
        "fleet_trends": fleet_trends.to_dict(orient="records"),
        "n_decreasing_plants_pvgis_pr": int(pr["decreasing"].sum()),
        "n_decreasing_plants_month_z": int(z["decreasing"].sum()),
        "fleet_monthly_rows": int(len(fleet_monthly)),
        "caveat": (
            "Month-standardized z-scores remove calendar-month seasonality. "
            "With a single year they are a relative diagnostic, not strong "
            "evidence of long-term degradation. Multi-year data is recommended."
        ),
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
    parser.add_argument("--min-day-hours", type=int, default=4)
    parser.add_argument("--min-month-hours", type=int, default=80)
    parser.add_argument("--min-months", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=8)
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
    kwp, kwp_is_real = _load_kwp_by_plant_dim(
        Path(args.plant_mapping),
        Path(args.energy_coords),
        plant_ids,
    )

    daily_df, monthly_df = _build_performance_tables(
        ds=ds,
        kwp=kwp,
        kwp_is_real=kwp_is_real,
        daytime_poa_threshold=args.daytime_poa_threshold,
        reference_pr=args.reference_pr,
        min_day_hours=args.min_day_hours,
        min_month_hours=args.min_month_hours,
    )
    daily_df, monthly_df = _add_month_standardization(daily_df, monthly_df)
    fleet_daily, fleet_monthly = _fleet_tables(daily_df, monthly_df)
    plant_trends, fleet_trends = _trend_tables(
        monthly_df=monthly_df,
        fleet_monthly=fleet_monthly,
        min_months=args.min_months,
        alpha=args.alpha,
    )

    daily_df.to_csv(out_dir / "plant_daily_performance.csv", index=False)
    monthly_df.to_csv(out_dir / "plant_monthly_performance.csv", index=False)
    fleet_daily.to_csv(out_dir / "fleet_daily_performance.csv", index=False)
    fleet_monthly.to_csv(out_dir / "fleet_monthly_performance.csv", index=False)
    plant_trends.to_csv(out_dir / "plant_trend_summary.csv", index=False)
    fleet_trends.to_csv(out_dir / "fleet_trend_summary.csv", index=False)
    _plot_outputs(out_dir, fleet_monthly, plant_trends, monthly_df, args.top_k)
    _write_summary(out_dir, args, ds, kwp_is_real, plant_trends, fleet_trends, fleet_monthly)

    pr = plant_trends[plant_trends["metric"] == "pr_pvgis_monthly"]
    z = plant_trends[plant_trends["metric"] == "pr_month_z_monthly"]
    print("\n=== DOMAIN-SHIFT / DEGRADATION TREND REPORT ===")
    print(f"Output directory: {out_dir}")
    print(f"Plants: {ds.sizes['plant']}  hours: {ds.sizes['time']}")
    print(f"kWp real/proxy: {int(kwp_is_real.sum())}/{int((~kwp_is_real).sum())}")
    print("\nFleet trends:")
    for _, row in fleet_trends.iterrows():
        print(
            f"  {row['metric']}: slope/year={row['slope_per_year']:+.4f}, "
            f"rel/year={row['relative_change_pct_per_year']:+.2f}%, "
            f"p={row['p_value']:.3g}, decreasing={bool(row['decreasing'])}"
        )
    print("\nPlant trends:")
    print(f"  decreasing PR_pvgis plants: {int(pr['decreasing'].sum())}/{pr['plant'].nunique()}")
    print(f"  decreasing month-z plants:  {int(z['decreasing'].sum())}/{z['plant'].nunique()}")
    print("\nKey files:")
    print("  plant_monthly_performance.csv")
    print("  plant_trend_summary.csv")
    print("  fleet_trend_summary.csv")
    print("  fleet_pvgis_pr_trend.png")


if __name__ == "__main__":
    main()
