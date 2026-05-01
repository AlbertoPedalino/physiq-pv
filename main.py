"""
PhysiQ-PV end-to-end pipeline entry point.

Runs:
  1. Real dataset loading (Piedmont 2019)
  2. QS computation per (plant, time)
  3. ST-GNN training on real data
  4. Online agentic loop (disabled)
  5. Summary report
"""
import json
import os
import numpy as np
import torch
import xarray as xr

from physiq_pv.data.quality_score import compute_qs
from physiq_pv.data.load_kwp import load_kwp
from physiq_pv.data.sentinel_hourly_loader import load_sentinel_hourly, merge_with_weather
from train import train
# from physiq_pv.agent.cycle import PhysiQAgent
# from online_loop import run_online


def _normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Align real dataset variable/coord names to the expected schema."""
    if "time" in ds.coords and ds["time"].dims != ("time",):
        time_vals = ds["time"].values
        ds = ds.drop_vars("time")
        orphan_dims = [
            d for d in ds.dims
            if d not in ds.data_vars and d not in ds.coords and d not in ("plant", "time", "plant_id")
        ]
        for d in orphan_dims:
            if ds.sizes[d] == len(time_vals):
                ds = ds.drop_dims(d)
                break
        ds = ds.assign_coords(time=("time", time_vals))

    if "plant" in ds.coords and ds["plant"].dims != ("plant",):
        plant_vals = ds["plant"].values
        ds = ds.drop_vars("plant")
        orphan_dims = [
            d for d in ds.dims
            if d not in ds.data_vars and d not in ds.coords and d not in ("plant", "time")
        ]
        for d in orphan_dims:
            if ds.sizes[d] == len(plant_vals):
                ds = ds.drop_dims(d)
                break
        ds = ds.assign_coords(plant=("plant", plant_vals))

    renames: dict[str, str] = {}
    if "latitude" in ds and "lat" not in ds:
        renames["latitude"] = "lat"
    if "longitude" in ds and "lon" not in ds:
        renames["longitude"] = "lon"
    if renames:
        ds = ds.rename(renames)

    if "eta_base" not in ds.data_vars and "eta_base" in ds.coords:
        ds = ds.assign({"eta_base": ds["eta_base"]})

    n_plants, n_steps = ds.sizes["plant"], ds.sizes["time"]

    if "solar_irradiance_poa" not in ds:
        raise ValueError(
            "Missing required variable 'solar_irradiance_poa'. "
            "Provide weather irradiance directly in the dataset."
        )

    if "wind_speed_10m" not in ds:
        ds = ds.assign({
            "wind_speed_10m": xr.DataArray(
                np.full((n_plants, n_steps), 3.0, dtype="float64"),
                dims=["plant", "time"],
            )
        })

    return ds


def main() -> None:
    sep = "=" * 62

    print(sep)
    print("PhysiQ-PV -- End-to-End Pipeline (real Piedmont 2019 data)")
    print(sep)
    print("\n[1] Loading real dataset (Sentinel hourly + weather)...")

    print("    -> Loading Sentinel hourly energy data (94 plants)...")
    ds = load_sentinel_hourly(
        sentinel_dir="/data/SentinelPV/energy_data/piemonte_energy_data/single_ups",
        year=2019,
        plant_mapping_path="data/plant_mapping.csv",
        energy_coords_path="data/energy_with_coordinates.csv",
    )

    print("    -> Merging weather variables...")
    ds = merge_with_weather(ds, pvgis_path="data/piedmont_pvgis_2019.nc")

    ds = _normalize_dataset(ds)
    print(f"    OK {ds.sizes['plant']} plants x {ds.sizes['time']} timesteps (hourly)")
    print("    Period: 2019-01-03 to 2019-12-31")
    print(f"    Variables: {list(ds.data_vars.keys())} [ENERGIA, solar_irradiance_poa, temperature_2m]")

    print("\n[2] Quality Score computation (per-plant per-time):")
    qs = compute_qs(ds)
    qs_valid = qs.values[~np.isnan(qs.values)]
    fleet_qs = float(qs.mean(skipna=True))
    print(f"    QS shape={qs.shape} (plant={ds.sizes['plant']}, time={ds.sizes['time']})")
    print(f"    Fleet QS mean={fleet_qs:.3f}, median={float(qs.median(skipna=True)):.3f}")
    print(f"    Valid data: {len(qs_valid):,} ({len(qs_valid)/qs.size*100:.1f}%)")

    print("\n[3] Training ST-GNN (max 10 epochs, peak-aware + quality-aware loss)...")
    kwp = None
    if os.path.exists("data/plant_mapping.csv") and os.path.exists("data/energy_with_coordinates.csv"):
        kwp = load_kwp("data/plant_mapping.csv", "data/energy_with_coordinates.csv", ds.sizes["plant"])
        n_real = int(np.sum(np.isfinite(kwp)))
        print(
            f"    Real kWp loaded: {n_real}/{ds.sizes['plant']} plants "
            f"(range {np.nanmin(kwp):.0f}-{np.nanmax(kwp):.0f} kW)"
        )

    model, loss_history, val_loss_history, updater, edge_index, edge_weight, pv_calibration = train(
        ds=ds,
        n_epochs=10,
        max_steps_per_epoch=None,
        kwp=kwp,
        early_stopping_patience=3,
        early_stopping_min_delta=1e-4,
        peak_alpha=2.0,
        peak_gamma=2.0,
        peak_loss_weight=0.5,
        calibration_kpi="none",
        qs_weight_exponent=0.2,
        qs_weight_floor=0.2,
        quality_over_loss_weight=0.02,
        eta_max=0.98,
    )

    curve = " -> ".join(f"{l:.4f}" for l in loss_history)
    val_curve = " -> ".join(f"{l:.4f}" for l in val_loss_history)
    print(f"    Train loss: {curve}")
    print(f"    Val   loss: {val_curve}")
    print(f"    Best val:   {min(val_loss_history):.4f} @ epoch {val_loss_history.index(min(val_loss_history)) + 1}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/model.pt")
    with open("checkpoints/loss_history.json", "w") as f:
        json.dump({
            "train": loss_history,
            "val": val_loss_history,
            "best_epoch": pv_calibration.get("best_val_epoch", 0),
        }, f)
    with open("checkpoints/pv_calibration.json", "w") as f:
        json.dump(pv_calibration, f, indent=2)
    with open("checkpoints/model_config.json", "w") as f:
        from physiq_pv.data.dataset import N_FEATURES, SEQ_LEN
        json.dump({
            "n_nodes": ds.sizes["plant"],
            "n_features": N_FEATURES,
            "seq_len": SEQ_LEN,
            "patch_len": 4,
            "stride": 2,
            "d_model": 64,
            "gat_dim": 96,
            "gat_heads": 4,
            "gat_layers": 1,
            "dropout": 0.0,
        }, f)
    with open("checkpoints/training_config.json", "w") as f:
        json.dump({
            "qs_weight_exponent": 0.2,
            "qs_weight_floor": 0.2,
            "quality_over_loss_weight": 0.02,
            "eta_max": 0.98,
            "calibration_kpi": "none",
        }, f)
    print("    Checkpoint saved -> checkpoints/")
    if pv_calibration.get("enabled", False):
        print(
            "    PV calibration (daytime): "
            f"slope={pv_calibration['slope']:.4f}, "
            f"intercept={pv_calibration['intercept']:+.4f}, "
            f"n={pv_calibration['n_samples']:,}, "
            f"RMSE {pv_calibration['rmse_before']:.4f}->{pv_calibration['rmse_after']:.4f}"
        )
    else:
        reason = pv_calibration.get("reason", pv_calibration.get("selection_reason", "unknown"))
        print(f"    PV calibration disabled: {reason}")

    print(f"\n  Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Loss final: {loss_history[-1]:.4f}")


if __name__ == "__main__":
    main()
