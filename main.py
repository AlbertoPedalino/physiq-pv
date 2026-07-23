"""
PhysiQ-PV end-to-end pipeline entry point.

Runs:
  1. Real dataset loading (Piedmont 2019)
  2. QS computation per (plant, time)
  3. ST-GNN training on real data
  4. Summary report
"""
import json
import os
import torch
import xarray as xr

from physiq_pv.data.sentinel_hourly_loader import load_sentinel_hourly, merge_with_weather
from train import train

SEQ_LEN_MODEL = 24
INCLUDE_POA_INPUTS = True
CHECKPOINT_DIR_BASE = "checkpoints/bilstm_gat_poa_v2_seq_len_24"
FEATURE_SET = "poa_physics_v2"


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

    required = (
        "ENERGIA",
        "temperature_2m",
        "wind_speed_10m",
        "solar_irradiance_poa",
        "eta_base",
        "lat",
        "lon",
    )
    missing = [name for name in required if name not in ds]
    if missing:
        raise ValueError(
            "Dataset is missing required fields: " + ", ".join(missing)
        )

    return ds


def main() -> None:
    sep = "=" * 62

    print(sep)
    print("PhysiQ-PV -- End-to-End Pipeline (real Piedmont 2019 data)")
    print(sep)
    print("\n[1] Loading real dataset (Sentinel hourly + weather)...")

    print("    -> Loading Sentinel hourly energy data...")
    # Multi-year hook: when CSVs for additional years are available under sentinel_dir,
    # call load_sentinel_hourly per year, align time coords, and concat along time.
    # Example:
    #   parts = [load_sentinel_hourly(..., year=y) for y in (2018, 2019, 2020)]
    #   ds = xr.concat(parts, dim="time")
    # Skipped in current run because only 2019 data is present locally.
    ds = load_sentinel_hourly(
        sentinel_dir="/data/SentinelPV/energy_data/piemonte_energy_data/single_ups",
        year=2019,
        plant_mapping_path="data/plant_mapping.csv",
        energy_coords_path="data/energy_with_coordinates.csv",
        source_timezone="Europe/Rome",
    )

    print("    -> Merging weather variables...")
    ds = merge_with_weather(ds, pvgis_path="data/piedmont_pvgis_2019.nc")

    ds = _normalize_dataset(ds)
    print(f"    OK {ds.sizes['plant']} plants x {ds.sizes['time']} timesteps (hourly)")
    print(f"    Period: {ds.time.values[0]} to {ds.time.values[-1]}")
    print(f"    Variables: {list(ds.data_vars.keys())} [ENERGIA, solar_irradiance_poa, temperature_2m]")

    print("\n[2] Training ST-GNN (max 15 epochs, peak-aware + quality-aware loss)...")
    peak_alpha       = 2.5
    peak_gamma       = 2.0
    peak_loss_weight = 0.25
    under_penalty    = 3.0

    from physiq_pv.data.dataset import N_FEATURES as _NF

    # Multi-seed loop. SEEDS env var overrides default list (comma-separated).
    seeds_env = os.environ.get("SEEDS", "42,123,2024")
    SEEDS = [int(s.strip()) for s in seeds_env.split(",") if s.strip()]
    BILSTM_POOLING = os.environ.get("BILSTM_POOLING", "attn")
    if BILSTM_POOLING not in ("attn", "last"):
        raise ValueError(f"BILSTM_POOLING must be 'attn' or 'last', got {BILSTM_POOLING!r}")
    print(f"\n[3a] Multi-seed plan: seeds={SEEDS}, bilstm_pooling={BILSTM_POOLING}")

    seed_summary: list[dict] = []
    for SEED in SEEDS:
        CHECKPOINT_DIR = f"{CHECKPOINT_DIR_BASE}_pool{BILSTM_POOLING}_seed{SEED}"
        print(f"\n{'='*62}\n[Seed {SEED}] training (checkpoint -> {CHECKPOINT_DIR})\n{'='*62}")

        model, loss_history, val_loss_history, edge_index, edge_weight, best_val_epoch = train(
            ds=ds,
            n_epochs=15,
            max_steps_per_epoch=None,
            early_stopping_patience=5,
            early_stopping_min_delta=1e-4,
            peak_alpha=peak_alpha,
            peak_gamma=peak_gamma,
            peak_loss_weight=peak_loss_weight,
            under_penalty=under_penalty,
            pr_max=1.5,
            night_loss_weight=0.2,
            validation_fraction=0.2,
            selection_metric="rmse_pv_day",
            include_poa_inputs=INCLUDE_POA_INPUTS,
            poa_kt_max=1.6,
            seq_len=SEQ_LEN_MODEL,
            checkpoint_dir=CHECKPOINT_DIR,
            use_wandb=True,
            wandb_entity="albertopedalino-politecnico-di-torino",
            wandb_project="PhysiQ-PV",
            wandb_run_name=f"{FEATURE_SET}_f{_NF}_seq{SEQ_LEN_MODEL}_a{peak_alpha}_g{peak_gamma}_w{peak_loss_weight}_pool{BILSTM_POOLING}_seed{SEED}",
            wandb_tags=["bilstm-gat", "poa-clear-sky", "train-only-preprocessing", FEATURE_SET, f"seq_len_{SEQ_LEN_MODEL}", f"seed_{SEED}", f"pool_{BILSTM_POOLING}", "multi_seed"],
            bilstm_pooling=BILSTM_POOLING,
            seed=SEED,
        )

        curve = " -> ".join(f"{l:.4f}" for l in loss_history)
        val_curve = " -> ".join(f"{l:.4f}" for l in val_loss_history)
        print(f"    Train loss: {curve}")
        print(f"    Val   loss: {val_curve}")
        training_summary = model.training_summary
        best_ep = int(training_summary["best_val_epoch"])
        best_metrics = training_summary["best_val_metrics"]
        best_val = float(best_metrics["val_loss"])
        best_score = float(training_summary["best_selection_score"])
        print(
            f"    Best checkpoint: rmse_pv_day={best_score:.4f}, "
            f"val_loss={best_val:.4f} @ epoch {best_ep}"
        )

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/model.pt")
        with open(f"{CHECKPOINT_DIR}/loss_history.json", "w") as f:
            json.dump({
                "train": loss_history,
                "val": val_loss_history,
                "best_epoch": best_val_epoch,
            }, f)
        with open(f"{CHECKPOINT_DIR}/model_config.json", "w") as f:
            from physiq_pv.data.dataset import N_FEATURES
            json.dump({
                "n_nodes": ds.sizes["plant"],
                "n_features": N_FEATURES,
                "seq_len": SEQ_LEN_MODEL,
                "d_model": 128,
                "gat_dim": 96,
                "gat_heads": 4,
                "gat_layers": 1,
                "dropout": 0.2,
                "use_bilstm": True,
                "use_gat": True,
                "bilstm_pooling": BILSTM_POOLING,
                "poa_kt_max": 1.6,
                "edge_prior_strength": 1.0,
                "include_poa_inputs": INCLUDE_POA_INPUTS,
                "seed": SEED,
            }, f)
        with open(f"{CHECKPOINT_DIR}/training_config.json", "w") as f:
            json.dump({
                "pr_max": 1.5,
                "night_loss_weight": 0.2,
                "validation_fraction": 0.2,
                "selection_metric": "rmse_pv_day",
                "include_poa_inputs": INCLUDE_POA_INPUTS,
                "ablation": f"seq_len_{SEQ_LEN_MODEL}",
                "description": (
                    f"BiLSTM+GAT with {SEQ_LEN_MODEL}h context, "
                    "tilted POA clear-sky head and train-only preprocessing"
                ),
                "checkpoint_dir": CHECKPOINT_DIR,
                "seed": SEED,
            }, f)
        print(f"    Checkpoint saved -> {CHECKPOINT_DIR}/")

        seed_summary.append({
            "seed": SEED,
            "best_val_loss": best_val,
            "best_rmse_pv_day": best_score,
            "best_epoch": best_ep,
            "final_train_loss": loss_history[-1],
            "checkpoint_dir": CHECKPOINT_DIR,
        })

    print(f"\n{'='*62}\nMulti-seed summary\n{'='*62}")
    for s in seed_summary:
        print(f"  seed={s['seed']:>5}  rmse_pv_day={s['best_rmse_pv_day']:.4f} @ ep {s['best_epoch']:>2}  "
              f"final_train={s['final_train_loss']:.4f}")
    vals = [s["best_rmse_pv_day"] for s in seed_summary]
    if len(vals) > 1:
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"\n  best_rmse_pv_day: mean={mean:.4f}  std={std:.4f}  n={len(vals)}")

    summary_path = f"{CHECKPOINT_DIR_BASE}_multi_seed_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"seeds": SEEDS, "runs": seed_summary}, f, indent=2)
    print(f"\n  Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
