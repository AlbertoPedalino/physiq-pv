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


def _filter_outlier_plants(
    ds: xr.Dataset,
    kwp: "np.ndarray | None",
    qs_daytime_threshold: float = 0.30,
    min_n_valid_daytime: int = 200,
) -> tuple[xr.Dataset, "np.ndarray | None", np.ndarray]:
    """
    Drop plants whose daytime data is too sparse or whose QS mean is below
    qs_daytime_threshold. Returns (filtered_ds, filtered_kwp, keep_mask).

    A plant is kept iff:
      - it has at least min_n_valid_daytime daytime samples with PV>0, AND
      - daytime mean QS (from compute_qs) >= qs_daytime_threshold

    The keep_mask (over original plant dim) is returned so callers can also
    filter precomputed arrays (kwp, etc.) in lock-step.
    """
    qs = compute_qs(ds)  # (plant, time)
    qs_arr = qs.values
    poa = ds["solar_irradiance_poa"].values  # (plant, time) W/m^2
    energia = ds["ENERGIA"].values  # (plant, time)
    daytime = poa > 50.0

    n_plants = qs_arr.shape[0]
    qs_mean = np.full(n_plants, np.nan, dtype=np.float64)
    n_valid = np.zeros(n_plants, dtype=int)
    for p in range(n_plants):
        day_p = daytime[p]
        if not day_p.any():
            continue
        qs_p = qs_arr[p, day_p]
        qs_p = qs_p[np.isfinite(qs_p)]
        if len(qs_p) > 0:
            qs_mean[p] = float(np.mean(qs_p))
        n_valid[p] = int(np.sum((energia[p, day_p] > 0) & np.isfinite(energia[p, day_p])))

    keep = (np.nan_to_num(qs_mean, nan=0.0) >= qs_daytime_threshold) & (n_valid >= min_n_valid_daytime)
    n_drop = int((~keep).sum())
    print(
        f"    Outlier filter: drop {n_drop}/{n_plants} plants "
        f"(QS<{qs_daytime_threshold} or n_valid<{min_n_valid_daytime})"
    )
    if n_drop == 0:
        return ds, kwp, keep

    plant_idx = np.where(keep)[0]
    ds_f = ds.isel(plant=plant_idx)
    kwp_f = kwp[plant_idx] if kwp is not None else None
    return ds_f, kwp_f, keep


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

    # Outlier filter kept available for ablation but disabled by default:
    # filtering degraded plants contradicts the data-centric / CL narrative
    # (CL must monitor and gate, not discard). Flip APPLY_OUTLIER_FILTER to True
    # only to produce an "apples-to-literature" ablation number.
    APPLY_OUTLIER_FILTER = False
    if APPLY_OUTLIER_FILTER:
        ds, kwp, _keep_mask = _filter_outlier_plants(
            ds, kwp, qs_daytime_threshold=0.30, min_n_valid_daytime=200,
        )

    peak_alpha       = 2.0
    peak_gamma       = 2.0
    peak_loss_weight = 0.25
    under_penalty    = 2.0

    # Baseline L=24: ST-GNN sees 24h of history (production model).
    SEQ_LEN_ABLATION = 24
    PATCH_LEN_ABLATION = 4
    STRIDE_ABLATION = 2
    CHECKPOINT_DIR = "checkpoints/seq_len_24"

    # Feature set: baseline (11) + cloud dynamics (kt, kt_std_3h, dghi_dt) + Erbs DNI/DHI split
    feature_set = "cloud_kt01_erbs"
    from physiq_pv.data.dataset import N_FEATURES as _NF

    model, loss_history, val_loss_history, updater, edge_index, edge_weight, pv_calibration = train(
        ds=ds,
        n_epochs=10,
        max_steps_per_epoch=None,
        kwp=kwp,
        early_stopping_patience=3,
        early_stopping_min_delta=1e-4,
        peak_alpha=peak_alpha,
        peak_gamma=peak_gamma,
        peak_loss_weight=peak_loss_weight,
        under_penalty=under_penalty,
        calibration_kpi="none",
        eta_max=0.98,
        seq_len=SEQ_LEN_ABLATION,
        patch_len=PATCH_LEN_ABLATION,
        stride=STRIDE_ABLATION,
        checkpoint_dir=CHECKPOINT_DIR,
        use_wandb=True,
        wandb_entity="albertopedalino-politecnico-di-torino",
        wandb_project="PhysiQ-PV",
        wandb_run_name=f"{feature_set}_f{_NF}_seq{SEQ_LEN_ABLATION}_a{peak_alpha}_g{peak_gamma}_w{peak_loss_weight}",
        wandb_tags=["erbs-dni-dhi", feature_set, f"seq_len_{SEQ_LEN_ABLATION}"],
    )

    curve = " -> ".join(f"{l:.4f}" for l in loss_history)
    val_curve = " -> ".join(f"{l:.4f}" for l in val_loss_history)
    print(f"    Train loss: {curve}")
    print(f"    Val   loss: {val_curve}")
    print(f"    Best val:   {min(val_loss_history):.4f} @ epoch {val_loss_history.index(min(val_loss_history)) + 1}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/model.pt")
    with open(f"{CHECKPOINT_DIR}/loss_history.json", "w") as f:
        json.dump({
            "train": loss_history,
            "val": val_loss_history,
            "best_epoch": pv_calibration.get("best_val_epoch", 0),
        }, f)
    with open(f"{CHECKPOINT_DIR}/pv_calibration.json", "w") as f:
        json.dump(pv_calibration, f, indent=2)
    with open(f"{CHECKPOINT_DIR}/model_config.json", "w") as f:
        from physiq_pv.data.dataset import N_FEATURES
        json.dump({
            "n_nodes": ds.sizes["plant"],
            "n_features": N_FEATURES,
            "seq_len": SEQ_LEN_ABLATION,
            "patch_len": PATCH_LEN_ABLATION,
            "stride": STRIDE_ABLATION,
            "d_model": 64,
            "gat_dim": 96,
            "gat_heads": 4,
            "gat_layers": 1,
            "dropout": 0.0,
        }, f)
    with open(f"{CHECKPOINT_DIR}/training_config.json", "w") as f:
        json.dump({
            "eta_max": 0.98,
            "calibration_kpi": "none",
            "ablation": f"seq_len_{SEQ_LEN_ABLATION}",
            "description": f"ST-GNN trained with {SEQ_LEN_ABLATION}h temporal context + Erbs DNI/DHI features",
            "checkpoint_dir": CHECKPOINT_DIR,
        }, f)
    print(f"    Checkpoint saved -> {CHECKPOINT_DIR}/")
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
