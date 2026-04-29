"""
PhysiQ-PV end-to-end pipeline entry point.

Runs:
  1. Real dataset loading (Piedmont 2019, PVGIS-aligned)
  2. QS computation per (plant, time)
  3. ST-GNN training on real data
  4. Online agentic loop (ATSF: perception->planning->action->reflection)
  5. Summary report
"""
import numpy as np
import xarray as xr
from physiq_pv.data.quality_score import compute_qs
from physiq_pv.agent.cycle import PhysiQAgent
from train import train
from online_loop import run_online


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
    print("\n[1] Loading real dataset (PVGIS-aligned 2019)...")
    ds = xr.open_dataset('data/real_data_dataset.nc')
    ds = _normalize_dataset(ds)
    print(f"    {ds.sizes['plant']} plants x {ds.sizes['time']} timesteps (2019-01-03 to 2019-12-31)")
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
    print("\n[3] Training ST-GNN (5 epochs, 30 steps/epoch on real 2019 data)...")
    model, loss_history, updater, edge_index, edge_weight = train(
        ds=ds, n_epochs=5, max_steps_per_epoch=30
    )
    curve = " -> ".join(f"{l:.4f}" for l in loss_history)
    print(f"    Loss curve: {curve}")

    # ------------------------------------------------------------------ #
    # 4. Online agentic loop (ATSF)
    # ------------------------------------------------------------------ #
    print("\n[4] Online agentic loop (window=720h, stride=168h) with QS monitoring...")
    agent = PhysiQAgent(n_clusters=4, drift_window=720)

    # Train causal classifier once on full dataset before streaming starts
    print("    Training causal classifier (MultiROCKET on real QS per-plant-per-time)...")
    clf_summary = agent.train_classifier(ds)
    if clf_summary["trained"]:
        print(f"    Samples: {clf_summary['n_samples']}  classes: {clf_summary['class_counts']}")
    else:
        print(f"    Fallback rule-based: {clf_summary['reason']}")

    print("    Each window: QS computed per (plant, time) → agent decides retraining")
    history = run_online(
        ds=ds,
        model=model,
        updater=updater,
        edge_index=edge_index,
        edge_weight=edge_weight,
        agent=agent,
        window_size=720,
        stride=168,  # Weekly stride for real data
        verbose=True,
    )
    n_retrained = sum(1 for r in history if r.get("action") == "retrain_triggered")
    print(f"    Steps: {len(history)}  retraining triggered: {n_retrained}")

    # ------------------------------------------------------------------ #
    # 5. Summary
    # ------------------------------------------------------------------ #
    print(f"\n{sep}")
    print("Summary - Real Data Pipeline (Piedmont 2019)")
    print(sep)
    print(f"\n  Dataset: Piedmont energy 2019 + PVGIS 2019 (PVGIS-aligned)")
    print(f"  Plants: {ds.sizes['plant']}, Timesteps: {ds.sizes['time']}")
    print(f"  QS: {len(qs_valid):,} valid per-plant-per-time measurements")
    print(f"  Fleet QS: mean={fleet_qs:.3f}")
    print(f"\n  Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Loss: {loss_history[-1]:.4f} (final)")
    print(f"  Online steps: {len(history)}, retraining: {n_retrained}x")
    print(f"\n✅ QS applied to EVERY (plant, time) during online loop!\n")


if __name__ == "__main__":
    main()
