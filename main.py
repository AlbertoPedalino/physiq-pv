"""
PhysiQ-PV end-to-end pipeline entry point.

Runs:
  1. Real dataset loading (Piedmont 2019, PVGIS-aligned)
  2. QS computation per (plant, time)
  3. ST-GNN training on real data
  4. Online agentic loop (ATSF: perception->planning->action->reflection)
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
# from physiq_pv.agent.cycle import PhysiQAgent  # re-enable for online loop
# from online_loop import run_online              # re-enable for online loop


def _normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Align real dataset variable/coord names to the expected schema."""
    # Fix time coord: in real dataset 'time' coord lives on 'date' dim, not 'time' dim.
    # After isel(time=...) the data shrinks but the coord stays full-length → crash.
    # Reassign 'time' coord onto the 'time' dimension.
    if "time" in ds.coords and ds["time"].dims != ("time",):
        time_vals = ds["time"].values
        ds = ds.drop_vars("time")
        orphan_dims = [d for d in ds.dims
                       if d not in ds.data_vars and d not in ds.coords
                       and d not in ("plant", "time", "plant_id")]
        for d in orphan_dims:
            if ds.sizes[d] == len(time_vals):
                ds = ds.drop_dims(d)
                break
        ds = ds.assign_coords(time=("time", time_vals))

    # Fix plant coord similarly
    if "plant" in ds.coords and ds["plant"].dims != ("plant",):
        plant_vals = ds["plant"].values
        ds = ds.drop_vars("plant")
        orphan_dims = [d for d in ds.dims
                       if d not in ds.data_vars and d not in ds.coords
                       and d not in ("plant", "time")]
        for d in orphan_dims:
            if ds.sizes[d] == len(plant_vals):
                ds = ds.drop_dims(d)
                break
        ds = ds.assign_coords(plant=("plant", plant_vals))

    renames = {}
    if "latitude" in ds and "lat" not in ds:
        renames["latitude"] = "lat"
    if "longitude" in ds and "lon" not in ds:
        renames["longitude"] = "lon"
    if renames:
        ds = ds.rename(renames)

    if "eta_base" not in ds.data_vars and "eta_base" in ds.coords:
        ds = ds.assign({"eta_base": ds["eta_base"]})

    N, T = ds.sizes["plant"], ds.sizes["time"]

    if "solar_irradiance_poa" not in ds:
        ds = ds.assign({"solar_irradiance_poa": ds["pvgis_ref"] * 1000.0})

    if "wind_speed_10m" not in ds:
        ds = ds.assign({
            "wind_speed_10m": xr.DataArray(
                np.full((N, T), 3.0, dtype="float64"),
                dims=["plant", "time"],
            )
        })

    return ds


def main() -> None:
    sep = "=" * 62

    # ------------------------------------------------------------------ #
    # 1. Real data (Piedmont 2019 with PVGIS)
    # ------------------------------------------------------------------ #
    print(sep)
    print("PhysiQ-PV -- End-to-End Pipeline (real Piedmont 2019 data)")
    print(sep)
    print("\n[1] Loading real dataset (Sentinel hourly + PVGIS 2019)...")
    
    # Load Sentinel hourly data from /data/SentinelPV/energy_data/piemonte_energy_data/single_ups/
    print("    → Loading Sentinel hourly energy data (94 plants)...")
    ds = load_sentinel_hourly(
        sentinel_dir="/data/SentinelPV/energy_data/piemonte_energy_data/single_ups",
        year=2019,
        plant_mapping_path='data/plant_mapping.csv',
        energy_coords_path='data/energy_with_coordinates.csv',
    )
    
    # Merge with PVGIS weather
    print("    → Merging with PVGIS reference + weather...")
    ds = merge_with_weather(ds, pvgis_path='data/piedmont_pvgis_2019.nc')
    
    ds = _normalize_dataset(ds)
    print(f"    ✅ {ds.sizes['plant']} plants × {ds.sizes['time']} timesteps (hourly)")
    print(f"    Period: 2019-01-03 to 2019-12-31")
    print(f"    Variables: {list(ds.data_vars.keys())} [ENERGIA, pvgis_ref, temperature_2m]")

    # ------------------------------------------------------------------ #
    # 2. Quality Score (per-plant per-timestamp)
    # ------------------------------------------------------------------ #
    print("\n[2] Quality Score computation (per-plant per-time):")
    qs = compute_qs(ds)
    qs_valid = qs.values[~np.isnan(qs.values)]
    fleet_qs = float(qs.mean(skipna=True))
    print(f"    QS shape={qs.shape} (plant={ds.sizes['plant']}, time={ds.sizes['time']})")
    print(f"    Fleet QS mean={fleet_qs:.3f}, median={float(qs.median(skipna=True)):.3f}")
    print(f"    Valid data: {len(qs_valid):,} ({len(qs_valid)/qs.size*100:.1f}%)")

    # ------------------------------------------------------------------ #
    # 3. ST-GNN training (on real data)
    # ------------------------------------------------------------------ #
    print("\n[3] Training ST-GNN (20 epochs, full dataset)...")
    kwp = None
    if os.path.exists("data/plant_mapping.csv") and os.path.exists("data/energy_with_coordinates.csv"):
        kwp = load_kwp("data/plant_mapping.csv", "data/energy_with_coordinates.csv", ds.sizes["plant"])
        n_real = int(np.sum(np.isfinite(kwp)))
        print(f"    Real kWp loaded: {n_real}/{ds.sizes['plant']} plants (range {np.nanmin(kwp):.0f}-{np.nanmax(kwp):.0f} kW)")
    model, loss_history, val_loss_history, updater, edge_index, edge_weight = train(
        ds=ds, n_epochs=20, max_steps_per_epoch=None, kwp=kwp
    )
    curve = " -> ".join(f"{l:.4f}" for l in loss_history)
    val_curve = " -> ".join(f"{l:.4f}" for l in val_loss_history)
    print(f"    Train loss: {curve}")
    print(f"    Val   loss: {val_curve}")
    print(f"    Best val:   {min(val_loss_history):.4f} @ epoch {val_loss_history.index(min(val_loss_history))+1}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/model.pt")
    with open("checkpoints/loss_history.json", "w") as f:
        json.dump({"train": loss_history, "val": val_loss_history}, f)
    with open("checkpoints/model_config.json", "w") as f:
        from physiq_pv.data.dataset import SEQ_LEN
        json.dump({
            "n_nodes": ds.sizes["plant"],
            "n_features": 6,
            "seq_len": SEQ_LEN,
            "patch_len": 4,
            "stride": 2,
            "d_model": 64,
            "gat_dim": 128,
            "gat_heads": 4,
            "gat_layers": 2,
            "dropout": 0.0,
        }, f)
    print(f"    Checkpoint saved → checkpoints/")

    print(f"\n  Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Loss final: {loss_history[-1]:.4f}")

    # ------------------------------------------------------------------ #
    # 4. Online agentic loop (ATSF)  — DISABLED for now
    # ------------------------------------------------------------------ #
    # agent = PhysiQAgent(n_clusters=4, drift_window=720)
    # clf_summary = agent.train_classifier(ds)
    # history = run_online(
    #     ds=ds, model=model, updater=updater,
    #     edge_index=edge_index, edge_weight=edge_weight,
    #     agent=agent, window_size=720, stride=168, verbose=True,
    # )
    # n_retrained = sum(1 for r in history if r.get("action") == "retrain_triggered")
    # print(f"    Steps: {len(history)}  retraining triggered: {n_retrained}")

    # ------------------------------------------------------------------ #
    # 5. Summary  — DISABLED for now
    # ------------------------------------------------------------------ #
    # print(f"  Online steps: {len(history)}, retraining: {n_retrained}x")
    # print(f"\n✅ QS applied to EVERY (plant, time) during online loop!\n")


if __name__ == "__main__":
    main()
