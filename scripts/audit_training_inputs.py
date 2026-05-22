"""
Audit the exact data path used by training.

This is not a model evaluation. It checks whether the plant/time tensors fed to
PVDataset look coherent after:

  Sentinel CSV loading -> PVGIS merge -> schema normalization -> QS -> PVDataset.

Outputs:
  outputs/training_input_audit/summary.json
  outputs/training_input_audit/plant_audit.csv
  outputs/training_input_audit/monthly_coverage.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import _normalize_dataset  # noqa: E402
from physiq_pv.data.dataset import PVDataset  # noqa: E402
from physiq_pv.data.quality_score import compute_qs  # noqa: E402
from physiq_pv.data.sentinel_hourly_loader import load_sentinel_hourly, merge_with_weather  # noqa: E402


def _upn_key(value: object) -> str:
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


def _mapping_table(plant_mapping: Path, energy_coords: Path) -> pd.DataFrame:
    if not plant_mapping.exists():
        return pd.DataFrame()
    pm = pd.read_csv(plant_mapping)
    cols = [c for c in ["Codice UP", "Codice Censimp Impianto", "plant_id", "Latitude", "Longitude"] if c in pm.columns]
    out = pm[cols].copy()
    if "Codice UP" not in out.columns:
        return pd.DataFrame()
    out["upn_key"] = out["Codice UP"].map(_upn_key)

    if energy_coords.exists() and "Codice Censimp Impianto" in out.columns:
        ec = pd.read_csv(energy_coords)
        ec_cols = [c for c in ["Codice Censimp Impianto", "Potenza di picco (kW)"] if c in ec.columns]
        if len(ec_cols) == 2:
            out = out.merge(
                ec[ec_cols].drop_duplicates("Codice Censimp Impianto"),
                on="Codice Censimp Impianto",
                how="left",
            )
    return out.drop_duplicates("upn_key")


def _describe(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {k: float("nan") for k in ["min", "p1", "p5", "median", "p95", "p99", "max", "mean"]}
    return {
        "min": float(np.min(v)),
        "p1": float(np.percentile(v, 1)),
        "p5": float(np.percentile(v, 5)),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "max": float(np.max(v)),
        "mean": float(np.mean(v)),
    }


def _monthly_coverage(ds) -> pd.DataFrame:
    times = pd.DatetimeIndex(ds["time"].values)
    energy = np.asarray(ds["ENERGIA"].values, dtype=float)
    poa = np.asarray(ds["solar_irradiance_poa"].values, dtype=float)
    rows = []
    for month_end, idx in pd.Series(np.arange(len(times)), index=times).resample("ME"):
        t_idx = idx.to_numpy(dtype=int)
        if len(t_idx) == 0:
            continue
        e_m = energy[:, t_idx]
        poa_m = poa[:, t_idx]
        rows.append(
            {
                "month": str(month_end.date()),
                "hours": int(len(t_idx)),
                "expected_hours": int(month_end.days_in_month * 24),
                "plants_with_any_energy": int(np.sum(np.isfinite(e_m).any(axis=1))),
                "finite_energy_pct": float(np.isfinite(e_m).mean() * 100.0),
                "daylight_hours": int(np.sum(np.nanmean(poa_m, axis=0) > 50.0)),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_sentinel_hourly(
        sentinel_dir=args.sentinel_dir,
        year=args.year,
        plant_mapping_path=args.plant_mapping,
        energy_coords_path=args.energy_coords,
    )
    ds = merge_with_weather(ds, pvgis_path=args.pvgis)
    ds = _normalize_dataset(ds)

    qs, m_components = compute_qs(ds, debug=True)
    dataset = PVDataset(ds, m_components, seq_len=args.seq_len, kwp=None, eta_max=0.98)

    times = pd.DatetimeIndex(ds["time"].values)
    energy = np.asarray(ds["ENERGIA"].values, dtype=float)
    poa = np.asarray(ds["solar_irradiance_poa"].values, dtype=float)
    lat = np.asarray(ds["lat"].values, dtype=float)
    lon = np.asarray(ds["lon"].values, dtype=float)
    plant_ids = np.asarray(ds["plant_id"].values) if "plant_id" in ds.coords else np.arange(ds.sizes["plant"])
    upns = np.asarray(ds["upn"].values) if "upn" in ds.coords else np.array([""] * ds.sizes["plant"], dtype=object)

    mapping = _mapping_table(Path(args.plant_mapping), Path(args.energy_coords))
    mapped_keys = set(mapping["upn_key"].dropna().astype(str)) if not mapping.empty and "upn_key" in mapping else set()
    ds_keys = np.array([_upn_key(u) for u in upns], dtype=object)
    upn_mapped = np.array([k in mapped_keys for k in ds_keys], dtype=bool)

    day = poa > args.daytime_poa_threshold
    plant_rows = []
    for p in range(ds.sizes["plant"]):
        e = energy[p]
        d = day[p] & np.isfinite(e)
        e_day = e[d]
        finite = np.isfinite(e)
        target = dataset.target_pv[:, p]
        feats_p = dataset.feats[:, p, :]
        plant_rows.append(
            {
                "plant": p,
                "plant_id": plant_ids[p],
                "upn": upns[p],
                "upn_key": ds_keys[p],
                "upn_has_mapping": bool(upn_mapped[p]),
                "lat": lat[p],
                "lon": lon[p],
                "has_coords": bool(np.isfinite(lat[p]) and np.isfinite(lon[p])),
                "finite_energy_hours": int(finite.sum()),
                "finite_energy_pct": float(finite.mean() * 100.0),
                "day_energy_hours": int(len(e_day)),
                "energy_p50": float(np.nanpercentile(e_day, 50)) if len(e_day) else np.nan,
                "energy_p99": float(np.nanpercentile(e_day, 99)) if len(e_day) else np.nan,
                "energy_max": float(np.nanmax(e_day)) if len(e_day) else np.nan,
                "poa_p99_wm2": float(np.nanpercentile(poa[p][day[p]], 99)) if day[p].any() else np.nan,
                "pv_scale": float(dataset.pv_scale[p]),
                "target_p99": float(np.nanpercentile(target, 99)),
                "target_max": float(np.nanmax(target)),
                "eta_adjusted": float(dataset.eta_adjusted[p]),
                "feature_nan_count": int(np.isnan(feats_p).sum()),
            }
        )

    plant_audit = pd.DataFrame(plant_rows)
    monthly = _monthly_coverage(ds)

    valid_starts = dataset.valid_starts
    train_n = 0
    val_n = 0
    for month in range(1, 13):
        month_mask = np.where(times[valid_starts].month == month)[0]
        split = int(len(month_mask) * 0.8)
        train_n += split
        val_n += len(month_mask) - split

    summary = {
        "n_plants": int(ds.sizes["plant"]),
        "n_hours": int(ds.sizes["time"]),
        "period_start": str(times[0]),
        "period_end": str(times[-1]),
        "n_unique_upn": int(len(set(map(str, upns)))),
        "n_duplicate_plant_id_values": int(pd.Series(plant_ids).duplicated().sum()),
        "n_upn_with_mapping": int(upn_mapped.sum()),
        "n_upn_without_mapping": int((~upn_mapped).sum()),
        "n_plants_with_coords": int(plant_audit["has_coords"].sum()),
        "n_plants_missing_coords": int((~plant_audit["has_coords"]).sum()),
        "energy_all": _describe(energy),
        "poa_all_wm2": _describe(poa),
        "pv_scale": _describe(dataset.pv_scale),
        "target_pv_norm": _describe(dataset.target_pv),
        "eta_adjusted": _describe(dataset.eta_adjusted),
        "feature_nan_count_total": int(np.isnan(dataset.feats).sum()),
        "target_nan_count_total": int(np.isnan(dataset.target_pv).sum()),
        "train_windows": int(train_n),
        "val_windows": int(val_n),
        "seq_len": int(args.seq_len),
        "warnings": [],
    }
    if summary["n_upn_without_mapping"] > 0:
        summary["warnings"].append("Some Sentinel UPNs do not match plant_mapping.csv after normalization.")
    if summary["n_plants_missing_coords"] > 0:
        summary["warnings"].append("Some plants lack coordinates; geometry/PVGIS matching may fall back or misbehave.")
    if summary["n_duplicate_plant_id_values"] > 0:
        summary["warnings"].append("plant_id coordinate has duplicates; use positional plant index for tensors.")
    if summary["feature_nan_count_total"] > 0 or summary["target_nan_count_total"] > 0:
        summary["warnings"].append("PVDataset contains NaNs after preprocessing.")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    plant_audit.to_csv(out_dir / "plant_audit.csv", index=False)
    monthly.to_csv(out_dir / "monthly_coverage.csv", index=False)

    print("\n=== TRAINING INPUT AUDIT ===")
    print(json.dumps(summary, indent=2))
    print("\nKey files:")
    print(f"  {out_dir / 'summary.json'}")
    print(f"  {out_dir / 'plant_audit.csv'}")
    print(f"  {out_dir / 'monthly_coverage.csv'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit training input tensors.")
    p.add_argument("--sentinel-dir", default="/data/SentinelPV/energy_data/piemonte_energy_data/single_ups")
    p.add_argument("--year", type=int, default=2019)
    p.add_argument("--plant-mapping", default="data/plant_mapping.csv")
    p.add_argument("--energy-coords", default="data/energy_with_coordinates.csv")
    p.add_argument("--pvgis", default="data/piedmont_pvgis_2019.nc")
    p.add_argument("--out-dir", default="outputs/training_input_audit")
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--daytime-poa-threshold", type=float, default=50.0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
