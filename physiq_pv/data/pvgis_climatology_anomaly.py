"""
PVGIS climatology-based anomaly scoring.

Compares a single PVGIS *target year* against a multi-year PVGIS *climatology*
built in memory from separate annual PVGIS NetCDF files.

Scope (intentionally narrow):
  - PVGIS is treated as a consistent physical reference.
  - We flag rare / extreme meteo-solar conditions relative to the PVGIS
    climatology itself.
  - We do NOT analyse real plant production, do NOT use observed plant energy,
    do NOT detect plant faults, do NOT train any model, and do NOT touch the
    training pipeline.
  - No multi-year NetCDF file is written: the climatology lives only in memory.

Climatology design (deliberately simple, no rolling window):
  - Bins are (location, month, day, hour). Binning by (month, day) instead of
    day_of_year keeps the same calendar day aligned across leap and non-leap
    years (otherwise day_of_year drifts after Feb 29).
  - Each climatology year contributes ONE sample per bin.
  - Per bin we store: median and the band [q_low, q_high], with
        q_low  = 1 - quantile
        q_high = quantile
  - A target point is flagged when its value falls outside [q_low, q_high] for
    its bin, provided the bin has at least `min_climatology_years` samples.
    Bins with fewer samples are labelled `insufficient_climatology`.

Public API (small functions, easy to compose):
    load_target_pvgis(path)
    load_climatology_files(dir, start_year, end_year, ...)
    build_climatology(datasets, variables, quantile)
    score_target_against_climatology(target_ds, climatology, ...)
    write_outputs(result, out_dir, ...)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

# Candidate variables. Only those present in BOTH target and climatology are used.
DEFAULT_VARIABLES: List[str] = [
    "solar_irradiance_poa",
    "pv_power_output",
    "temperature_2m",
    "wind_speed_10m",
]

SOLAR_VARIABLES = {"solar_irradiance_poa", "pv_power_output"}

# Climatology-level labels only (no plant-level semantics).
LABEL_NORMAL = "normal"
LABEL_LOW_SOLAR = "unusually_low_solar_potential"
LABEL_HIGH_SOLAR = "unusually_high_solar_potential"
LABEL_EXTREME_TEMP = "extreme_temperature_condition"
LABEL_EXTREME_WIND = "extreme_wind_condition"
LABEL_INSUFFICIENT = "insufficient_climatology"

SCORE_COLUMNS = [
    "location",
    "latitude",
    "longitude",
    "timestamp",
    "year",
    "month",
    "day",
    "day_of_year",
    "hour",
    "variable",
    "value",
    "climatology_median",
    "climatology_q_low",
    "climatology_q_high",
    "deviation_from_median",
    "anomaly_score",
    "label",
]

_EPS = 1e-9
_N_MONTH = 12
_N_DAY = 31
_N_MD = _N_MONTH * _N_DAY  # (month, day) bins; avoids leap-year day_of_year drift
_N_HOUR = 24


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_target_pvgis(path: str) -> xr.Dataset:
    """Load the PVGIS target-year NetCDF file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PVGIS target file not found: {p}")
    return xr.open_dataset(p)


def load_climatology_files(
    climatology_dir: str,
    start_year: int,
    end_year: int,
    file_template: str = "piedmont_pvgis_{year}.nc",
    exclude_year: Optional[int] = None,
) -> Dict[int, xr.Dataset]:
    """
    Load separate annual PVGIS files in [start_year, end_year].

    Missing yearly files are skipped with a clear message. If `exclude_year` is
    given, that year is left out of the climatology (leave-one-out, so the
    target year is not compared against itself).
    """
    d = Path(climatology_dir)
    if not d.is_dir():
        raise NotADirectoryError(f"PVGIS climatology directory not found: {d}")

    datasets: Dict[int, xr.Dataset] = {}
    for year in range(start_year, end_year + 1):
        if exclude_year is not None and year == exclude_year:
            print(f"  [skip] excluding target year {year} from climatology")
            continue
        fp = d / file_template.format(year=year)
        if not fp.exists():
            print(f"  [skip] missing climatology file for {year}: {fp}")
            continue
        datasets[year] = xr.open_dataset(fp)
        print(f"  [ok]   loaded climatology year {year}: {fp.name}")

    if not datasets:
        raise FileNotFoundError(
            f"No climatology files found in {d} for {start_year}-{end_year} "
            f"(template '{file_template}')."
        )
    return datasets


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _location_dim(ds: xr.Dataset) -> str:
    if "location" in ds.dims:
        return "location"
    raise ValueError(f"Expected a 'location' dimension, got dims={tuple(ds.dims)}")


def _latlon(ds: xr.Dataset, n_loc: int) -> tuple[np.ndarray, np.ndarray]:
    for la, lo in (("lat", "lon"), ("latitude", "longitude")):
        if la in ds.coords and lo in ds.coords:
            return (
                np.asarray(ds[la].values, dtype=float),
                np.asarray(ds[lo].values, dtype=float),
            )
    return np.full(n_loc, np.nan), np.full(n_loc, np.nan)


def _md_hour(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return flat (month, day) bin index and hour for each timestamp.

    md = (month - 1) * 31 + (day - 1), so the same calendar day maps to the
    same bin across leap and non-leap years (no day_of_year drift after Feb 29).
    """
    t = pd.DatetimeIndex(times)
    md = (t.month.values - 1) * _N_DAY + (t.day.values - 1)
    return md, t.hour.values


# --------------------------------------------------------------------------- #
# Climatology
# --------------------------------------------------------------------------- #
@dataclass
class VariableClimatology:
    """Per-(location, month, day, hour) climatology statistics for one variable."""

    median: np.ndarray  # (L, 372, 24)  -- 372 = 12 months * 31 days
    q_low: np.ndarray  # (L, 372, 24)
    q_high: np.ndarray  # (L, 372, 24)
    count: np.ndarray  # (L, 372, 24) number of contributing years
    n_years: int


def _scatter_year(slot: np.ndarray, da: xr.DataArray, loc_dim: str) -> None:
    """Place one year's (location, time) values into a (L, 372, 24) slot."""
    da = da.transpose(loc_dim, "time")
    vals = np.asarray(da.values, dtype=np.float32)
    md, hour = _md_hour(da["time"].values)
    slot[:, md, hour] = vals


def build_climatology(
    datasets: Dict[int, xr.Dataset],
    variables: List[str],
    quantile: float,
    loc_dim: str = "location",
) -> Dict[str, VariableClimatology]:
    """
    Build an in-memory climatology per variable.

    For each variable we stack the contributing years into a
    (n_years, L, 372, 24) array (one variable at a time to bound memory) and
    reduce over the year axis to median / q_low / q_high / count.
    """
    q_low_p = 1.0 - quantile
    q_high_p = quantile
    years = sorted(datasets)
    n_loc = datasets[years[0]].sizes[loc_dim]

    climatology: Dict[str, VariableClimatology] = {}
    for var in variables:
        present = [y for y in years if var in datasets[y]]
        if not present:
            print(f"  [skip] variable '{var}' absent from all climatology files")
            continue

        stack = np.full((len(present), n_loc, _N_MD, _N_HOUR), np.nan, dtype=np.float32)
        for i, y in enumerate(present):
            _scatter_year(stack[i], datasets[y][var], loc_dim)

        count = np.sum(~np.isnan(stack), axis=0).astype(np.int16)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN bins
            median = np.nanmedian(stack, axis=0).astype(np.float32)
            q_low = np.nanquantile(stack, q_low_p, axis=0).astype(np.float32)
            q_high = np.nanquantile(stack, q_high_p, axis=0).astype(np.float32)

        climatology[var] = VariableClimatology(
            median=median, q_low=q_low, q_high=q_high, count=count, n_years=len(present)
        )
        print(f"  [ok]   climatology built for '{var}' from {len(present)} years")
        del stack
    return climatology


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@dataclass
class AnomalyResult:
    scores: pd.DataFrame  # one row per flagged anomaly
    labels: pd.DataFrame  # per (location, variable, label) tally
    summary: pd.DataFrame  # per-variable summary
    meta: dict


def _solar_high_low(var: str) -> tuple[str, str]:
    """Return (high_label, low_label) for a variable category."""
    if var in SOLAR_VARIABLES:
        return LABEL_HIGH_SOLAR, LABEL_LOW_SOLAR
    if var == "temperature_2m":
        return LABEL_EXTREME_TEMP, LABEL_EXTREME_TEMP
    if var == "wind_speed_10m":
        return LABEL_EXTREME_WIND, LABEL_EXTREME_WIND
    return "extreme_condition", "extreme_condition"


def score_target_against_climatology(
    target_ds: xr.Dataset,
    climatology: Dict[str, VariableClimatology],
    variables: List[str],
    quantile: float,
    min_years: int,
    loc_dim: str = "location",
) -> AnomalyResult:
    """
    Score every target point against its climatology bin and collect anomalies.

    Returns flagged points (value outside [q_low, q_high]) in `scores`, a
    per-(location, variable, label) tally in `labels`, and a per-variable
    `summary`. Normal points are not enumerated in `scores` (only counted),
    keeping the output bounded to the climatological tail.
    """
    n_loc = target_ds.sizes[loc_dim]
    lat, lon = _latlon(target_ds, n_loc)
    loc_ids = np.asarray(target_ds[loc_dim].values)
    times = target_ds["time"].values
    md, hour = _md_hour(times)
    ts_index = pd.DatetimeIndex(times)

    score_frames: List[pd.DataFrame] = []
    label_rows: List[dict] = []
    summary_rows: List[dict] = []
    used_variables: List[str] = []

    for var in variables:
        if var not in target_ds or var not in climatology:
            continue
        clim = climatology[var]
        if clim.median.shape[0] != n_loc:
            raise ValueError(
                f"Location mismatch for '{var}': target has {n_loc} locations, "
                f"climatology has {clim.median.shape[0]}."
            )
        used_variables.append(var)

        da = target_ds[var].transpose(loc_dim, "time")
        val = np.asarray(da.values, dtype=np.float32)  # (L, T)
        med = clim.median[:, md, hour]
        qlo = clim.q_low[:, md, hour]
        qhi = clim.q_high[:, md, hour]
        cnt = clim.count[:, md, hour]

        valid = np.isfinite(val) & np.isfinite(med)
        enough = cnt >= min_years
        insufficient = valid & ~enough
        above = valid & enough & (val > qhi)
        below = valid & enough & (val < qlo)
        flagged = above | below
        normal = valid & enough & ~flagged

        half_spread = np.abs(qhi - qlo) / 2.0
        score = (val - med) / (half_spread + _EPS)

        # --- flagged anomalies -> long rows ---
        hi_label, lo_label = _solar_high_low(var)
        if flagged.any():
            li, ti = np.where(flagged)
            f_val = val[flagged]
            f_qhi = qhi[flagged]
            ts = ts_index[ti]
            df = pd.DataFrame(
                {
                    "location": loc_ids[li],
                    "latitude": lat[li],
                    "longitude": lon[li],
                    "timestamp": ts,
                    "year": ts.year,
                    "month": ts.month,
                    "day": ts.day,
                    "day_of_year": ts.dayofyear,
                    "hour": hour[ti],
                    "variable": var,
                    "value": f_val,
                    "climatology_median": med[flagged],
                    "climatology_q_low": qlo[flagged],
                    "climatology_q_high": qhi[flagged],
                    "deviation_from_median": f_val - med[flagged],
                    "anomaly_score": score[flagged],
                    "label": np.where(f_val > f_qhi, hi_label, lo_label),
                }
            )
            score_frames.append(df[SCORE_COLUMNS])

        # --- per-location label tally (counts only) ---
        per_loc = {
            LABEL_NORMAL: normal.sum(axis=1),
            LABEL_INSUFFICIENT: insufficient.sum(axis=1),
            hi_label: above.sum(axis=1),
            lo_label: below.sum(axis=1),
        }
        for li in range(n_loc):
            for label, counts in per_loc.items():
                c = int(counts[li])
                if c > 0:
                    label_rows.append(
                        {
                            "location": loc_ids[li],
                            "latitude": lat[li],
                            "longitude": lon[li],
                            "variable": var,
                            "label": label,
                            "count": c,
                        }
                    )

        # --- per-variable summary ---
        n_valid_enough = int((valid & enough).sum())
        n_flagged = int(flagged.sum())
        flagged_scores = np.abs(score[flagged]) if n_flagged else np.array([0.0])
        summary_rows.append(
            {
                "variable": var,
                "n_points": int(val.size),
                "n_valid": int(valid.sum()),
                "n_insufficient": int(insufficient.sum()),
                "n_normal": int(normal.sum()),
                "n_flagged": n_flagged,
                "n_high": int(above.sum()),
                "n_low": int(below.sum()),
                "pct_flagged": (100.0 * n_flagged / n_valid_enough) if n_valid_enough else 0.0,
                "mean_abs_anomaly_score": float(flagged_scores.mean()),
                "max_abs_anomaly_score": float(flagged_scores.max()),
                "climatology_years": clim.n_years,
            }
        )

    scores = (
        pd.concat(score_frames, ignore_index=True)
        if score_frames
        else pd.DataFrame(columns=SCORE_COLUMNS)
    )
    scores = scores.sort_values(
        "anomaly_score", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)
    labels = pd.DataFrame(
        label_rows, columns=["location", "latitude", "longitude", "variable", "label", "count"]
    )
    summary = pd.DataFrame(summary_rows)

    meta = {
        "target_year": int(pd.DatetimeIndex(times).year[0]) if len(times) else None,
        "quantile": quantile,
        "q_low": round(1.0 - quantile, 6),
        "q_high": round(quantile, 6),
        "min_climatology_years": min_years,
        "variables_analyzed": used_variables,
        "n_locations": n_loc,
        "n_target_timesteps": int(len(times)),
        "total_flagged": int(len(scores)),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return AnomalyResult(scores=scores, labels=labels, summary=summary, meta=meta)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _render_report(result: AnomalyResult, top_n: int) -> str:
    m = result.meta
    lines: List[str] = []
    lines.append("# PVGIS climatology anomaly report\n")
    lines.append(
        "Target PVGIS year compared against a multi-year PVGIS climatology. "
        "PVGIS is used as a consistent physical reference. This analysis does "
        "**not** inspect real plants, does **not** use observed plant energy, "
        "and does **not** touch the training pipeline.\n"
    )
    lines.append("## Parameters\n")
    lines.append(f"- Target year: **{m['target_year']}**")
    lines.append(f"- Quantile band: **[{m['q_low']}, {m['q_high']}]** (quantile={m['quantile']})")
    lines.append(f"- Min climatology years per bin: **{m['min_climatology_years']}**")
    lines.append(f"- Locations: **{m['n_locations']}**  |  Target timesteps: **{m['n_target_timesteps']}**")
    lines.append(f"- Variables analyzed: {', '.join(m['variables_analyzed']) or '(none)'}")
    lines.append(f"- Total flagged anomalies: **{m['total_flagged']}**")
    lines.append(f"- Generated (UTC): {m['generated_utc']}\n")

    lines.append("## Per-variable summary\n")
    if result.summary.empty:
        lines.append("_No variables analyzed._\n")
    else:
        lines.append("| variable | valid | normal | insufficient | flagged | high | low | % flagged | max |score| |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in result.summary.iterrows():
            lines.append(
                f"| {r['variable']} | {int(r['n_valid'])} | {int(r['n_normal'])} | "
                f"{int(r['n_insufficient'])} | {int(r['n_flagged'])} | {int(r['n_high'])} | "
                f"{int(r['n_low'])} | {r['pct_flagged']:.3f}% | {r['max_abs_anomaly_score']:.2f} |"
            )
        lines.append("")

    lines.append(f"## Top {top_n} most extreme conditions\n")
    if result.scores.empty:
        lines.append("_No anomalies flagged for the chosen quantile band._\n")
    else:
        top = result.scores.head(top_n)
        lines.append("| timestamp | location | variable | value | median | band | score | label |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in top.iterrows():
            band = f"[{r['climatology_q_low']:.3g}, {r['climatology_q_high']:.3g}]"
            lines.append(
                f"| {pd.Timestamp(r['timestamp'])} | {r['location']} | {r['variable']} | "
                f"{r['value']:.3g} | {r['climatology_median']:.3g} | {band} | "
                f"{r['anomaly_score']:.2f} | {r['label']} |"
            )
        lines.append("")

    lines.append("## Notes\n")
    lines.append("- Climatology bins are (location, month, day, hour); one sample per year, no rolling window.")
    lines.append("- A point is flagged when its value falls outside the [q_low, q_high] band of its bin.")
    lines.append("- `anomaly_score = (value - median) / (|q_high - q_low| / 2 + eps)`; sign encodes direction.")
    lines.append("- Bins with fewer than the minimum number of years are labelled `insufficient_climatology`.")
    return "\n".join(lines) + "\n"


def write_outputs(result: AnomalyResult, out_dir: str, top_n: int = 20) -> Dict[str, Path]:
    """Write scores / labels / summary CSVs and a markdown report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "scores": out / "pvgis_climatology_scores.csv",
        "labels": out / "pvgis_climatology_labels.csv",
        "summary": out / "pvgis_climatology_summary.csv",
        "report": out / "pvgis_climatology_report.md",
    }
    result.scores.to_csv(paths["scores"], index=False)
    result.labels.to_csv(paths["labels"], index=False)
    result.summary.to_csv(paths["summary"], index=False)
    paths["report"].write_text(_render_report(result, top_n), encoding="utf-8")
    return paths
