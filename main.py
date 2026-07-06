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
import numpy as np
import torch
import xarray as xr

from physiq_pv.data.quality_score import compute_qs
from physiq_pv.data.load_kwp import load_kwp
from physiq_pv.data.openmeteo_loader import merge_with_openmeteo
from physiq_pv.data.sentinel_hourly_loader import load_sentinel_hourly
from train import train


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_optional_int(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


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


def _drop_missing_coordinate_plants(
    ds: xr.Dataset,
    kwp: "np.ndarray | None" = None,
) -> tuple[xr.Dataset, "np.ndarray | None", np.ndarray]:
    """
    Drop plants without finite latitude/longitude, keeping optional per-plant
    arrays aligned with the original plant dimension.
    """
    variables = set(ds.variables)
    lat_name = "lat" if "lat" in variables else "latitude" if "latitude" in variables else None
    lon_name = "lon" if "lon" in variables else "longitude" if "longitude" in variables else None
    if lat_name is None or lon_name is None:
        raise ValueError("Dataset must contain lat/lon or latitude/longitude coordinates")

    lats = ds[lat_name].values.astype(float)
    lons = ds[lon_name].values.astype(float)
    keep = np.isfinite(lats) & np.isfinite(lons)

    if kwp is not None and len(kwp) != ds.sizes["plant"]:
        raise ValueError(
            f"kwp length ({len(kwp)}) must match plant dimension ({ds.sizes['plant']})"
        )
    if not keep.any():
        raise ValueError("All plants are missing coordinates; cannot train geographic model")

    n_drop = int((~keep).sum())
    if n_drop == 0:
        print("    Coordinate filter: no plants dropped")
        return ds, kwp, keep

    plant_idx = np.where(keep)[0]
    ds_f = ds.isel(plant=plant_idx)
    kwp_f = kwp[keep] if kwp is not None else None
    print(
        f"    Coordinate filter: drop {n_drop}/{ds.sizes['plant']} plants "
        "with missing lat/lon"
    )
    return ds_f, kwp_f, keep


def _load_kwp_for_dataset(
    ds: xr.Dataset,
    plant_mapping_path: str,
    energy_coords_path: str,
) -> np.ndarray:
    """
    Load kWp by registry plant_id, then align it to the dataset's positional
    plant axis.
    """
    n_plants = ds.sizes["plant"]
    if "plant_id" not in ds.variables:
        return load_kwp(plant_mapping_path, energy_coords_path, n_plants)

    try:
        plant_ids = np.asarray(ds["plant_id"].values).astype(int)
    except (TypeError, ValueError):
        return load_kwp(plant_mapping_path, energy_coords_path, n_plants)

    valid_ids = plant_ids[plant_ids >= 0]
    lookup_size = n_plants
    if len(valid_ids) > 0:
        lookup_size = max(n_plants, int(valid_ids.max()) + 1)

    kwp_by_plant_id = load_kwp(plant_mapping_path, energy_coords_path, lookup_size)
    kwp = np.full(n_plants, np.nan, dtype=np.float64)
    for pos, plant_id in enumerate(plant_ids):
        if 0 <= plant_id < len(kwp_by_plant_id):
            kwp[pos] = kwp_by_plant_id[plant_id]
    return kwp


def main() -> None:
    sep = "=" * 62
    sentinel_dir = os.environ.get(
        "SENTINEL_DIR",
        "/data/SentinelPV/energy_data/piemonte_energy_data/single_ups",
    )
    data_year = int(os.environ.get("DATA_YEAR", "2019"))
    plant_mapping_path = os.environ.get("PLANT_MAPPING_PATH", "data/plant_mapping.csv")
    energy_coords_path = os.environ.get("ENERGY_COORDS_PATH", "data/energy_with_coordinates.csv")

    print(sep)
    print(f"PhysiQ-PV -- End-to-End Pipeline (real Piedmont {data_year} data)")
    print(sep)
    print("\n[1] Loading real dataset (Sentinel hourly + weather)...")

    print("    -> Loading Sentinel hourly energy data...")
    ds = load_sentinel_hourly(
        sentinel_dir=sentinel_dir,
        year=data_year,
        plant_mapping_path=plant_mapping_path,
        energy_coords_path=energy_coords_path,
    )

    kwp = None
    if os.path.exists(plant_mapping_path) and os.path.exists(energy_coords_path):
        kwp = _load_kwp_for_dataset(
            ds,
            plant_mapping_path,
            energy_coords_path,
        )

    ds, kwp, _coord_keep_mask = _drop_missing_coordinate_plants(ds, kwp)

    WEATHER_SOURCE = os.environ.get("WEATHER_SOURCE", "openmeteo_historical_forecast")
    FEATURE_SET = os.environ.get("FEATURE_SET", "openmeteo_operational")
    OPENMETEO_PATH = os.environ.get("OPENMETEO_PATH", "data/openmeteo_piedmont_2019.nc")

    print(f"    -> weather_source={WEATHER_SOURCE}, feature_set={FEATURE_SET}")
    if not WEATHER_SOURCE.startswith("openmeteo"):
        raise ValueError(f"This branch expects Open-Meteo weather_source, got {WEATHER_SOURCE!r}")
    ds = merge_with_openmeteo(ds, openmeteo_path=OPENMETEO_PATH)

    ds = _normalize_dataset(ds)
    time_values = ds["time"].values
    period_start = np.datetime_as_string(time_values[0], unit="D")
    period_end = np.datetime_as_string(time_values[-1], unit="D")
    print(f"    OK {ds.sizes['plant']} plants x {ds.sizes['time']} timesteps (hourly)")
    print(f"    Period: {period_start} to {period_end}")
    print(f"    Variables: {list(ds.data_vars.keys())} [ENERGIA, solar_irradiance_poa, temperature_2m]")

    print("\n[2] Quality Score computation (per-plant per-time):")
    qs = compute_qs(ds)
    qs_valid = qs.values[~np.isnan(qs.values)]
    fleet_qs = float(qs.mean(skipna=True))
    print(f"    QS shape={qs.shape} (plant={ds.sizes['plant']}, time={ds.sizes['time']})")
    print(f"    Fleet QS mean={fleet_qs:.3f}, median={float(qs.median(skipna=True)):.3f}")
    print(f"    Valid data: {len(qs_valid):,} ({len(qs_valid)/qs.size*100:.1f}%)")

    qs_loss_weighting = _env_bool("QS_LOSS_WEIGHTING", False)
    qs_loss_floor = float(os.environ.get("QS_LOSS_FLOOR", "0.2"))
    n_epochs = int(os.environ.get("N_EPOCHS", "15"))
    max_steps_per_epoch = _env_optional_int("MAX_STEPS_PER_EPOCH")
    early_stopping_patience = int(os.environ.get("EARLY_STOPPING_PATIENCE", "5"))
    early_stopping_min_delta = float(os.environ.get("EARLY_STOPPING_MIN_DELTA", "1e-4"))
    peak_alpha = float(os.environ.get("PEAK_ALPHA", "2.5"))
    peak_gamma = float(os.environ.get("PEAK_GAMMA", "2.0"))
    peak_loss_weight = float(os.environ.get("PEAK_LOSS_WEIGHT", "0.25"))
    under_penalty = float(os.environ.get("UNDER_PENALTY", "3.0"))
    eta_max = float(os.environ.get("ETA_MAX", "0.98"))
    calibration_kpi = os.environ.get("CALIBRATION_KPI", "none")
    seq_len = int(os.environ.get("SEQ_LEN", "24"))
    patch_len = int(os.environ.get("PATCH_LEN", "4"))
    stride = int(os.environ.get("STRIDE", "2"))
    checkpoint_dir_base = os.environ.get("CHECKPOINT_DIR_BASE", f"checkpoints/seq_len_{seq_len}")
    apply_outlier_filter = _env_bool("APPLY_OUTLIER_FILTER", False)
    outlier_qs_threshold = float(os.environ.get("OUTLIER_QS_DAYTIME_THRESHOLD", "0.30"))
    outlier_min_valid = int(os.environ.get("OUTLIER_MIN_VALID_DAYTIME", "200"))

    loss_desc = "peak-aware + QS-weighted" if qs_loss_weighting else "peak-aware"
    print(f"\n[3] Training ST-GNN (max {n_epochs} epochs, {loss_desc} loss)...")
    if kwp is not None:
        finite_kwp = np.isfinite(kwp)
        n_real = int(np.sum(finite_kwp))
        if n_real > 0:
            print(
                f"    Real kWp loaded: {n_real}/{ds.sizes['plant']} plants "
                f"(range {np.nanmin(kwp):.0f}-{np.nanmax(kwp):.0f} kW)"
            )
        else:
            print(f"    Real kWp loaded: 0/{ds.sizes['plant']} plants")

    if apply_outlier_filter:
        ds, kwp, _keep_mask = _filter_outlier_plants(
            ds,
            kwp,
            qs_daytime_threshold=outlier_qs_threshold,
            min_n_valid_daytime=outlier_min_valid,
        )

    from physiq_pv.data.dataset import N_FEATURES as _NF

    # Multi-seed loop. SEEDS env var overrides default list (comma-separated).
    seeds_env = os.environ.get("SEEDS", "42,123,2024")
    SEEDS = [int(s.strip()) for s in seeds_env.split(",") if s.strip()]
    BILSTM_POOLING = os.environ.get("BILSTM_POOLING", "attn")
    WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "PhysiQ-PV")
    WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "albertopedalino-politecnico-di-torino")
    WANDB_RUN_NAME = os.environ.get(
        "WANDB_RUN_NAME",
        (
            "{feature_set}_f{n_features}_seq{seq_len}_a{peak_alpha:g}_g{peak_gamma:g}"
            "_w{peak_loss_weight:g}_pool{bilstm_pooling}{quality_suffix}_seed{seed}"
        ),
    )
    tags_env = os.environ.get("WANDB_TAGS", "").strip()
    base_wandb_tags = [tag.strip() for tag in tags_env.split(",") if tag.strip()]
    if not base_wandb_tags:
        base_wandb_tags = [
            "bilstm-gat",
            FEATURE_SET,
            f"seq_len_{seq_len}",
            "multi_seed",
            "qs_weighted_loss" if qs_loss_weighting else "unweighted_loss",
        ]
    if BILSTM_POOLING not in ("attn", "last"):
        raise ValueError(f"BILSTM_POOLING must be 'attn' or 'last', got {BILSTM_POOLING!r}")
    quality_suffix = f"_qs{qs_loss_floor:g}" if qs_loss_weighting else ""
    print(
        f"\n[3a] Multi-seed plan: seeds={SEEDS}, bilstm_pooling={BILSTM_POOLING}, "
        f"qs_loss_weighting={qs_loss_weighting}, qs_loss_floor={qs_loss_floor:g}"
    )

    seed_summary: list[dict] = []
    for SEED in SEEDS:
        CHECKPOINT_DIR = f"{checkpoint_dir_base}_pool{BILSTM_POOLING}{quality_suffix}_seed{SEED}"
        wandb_run_name = WANDB_RUN_NAME.format(
            feature_set=FEATURE_SET,
            weather_source=WEATHER_SOURCE,
            n_features=_NF,
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            peak_alpha=peak_alpha,
            peak_gamma=peak_gamma,
            peak_loss_weight=peak_loss_weight,
            under_penalty=under_penalty,
            bilstm_pooling=BILSTM_POOLING,
            quality_suffix=quality_suffix,
            qs_loss_floor=qs_loss_floor,
            seed=SEED,
        )
        print(f"\n{'='*62}\n[Seed {SEED}] training (checkpoint -> {CHECKPOINT_DIR})\n{'='*62}")

        model, loss_history, val_loss_history, edge_index, edge_weight, pv_calibration = train(
            ds=ds,
            n_epochs=n_epochs,
            max_steps_per_epoch=max_steps_per_epoch,
            kwp=kwp,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            peak_alpha=peak_alpha,
            peak_gamma=peak_gamma,
            peak_loss_weight=peak_loss_weight,
            under_penalty=under_penalty,
            qs_loss_weighting=qs_loss_weighting,
            qs_loss_floor=qs_loss_floor,
            calibration_kpi=calibration_kpi,
            eta_max=eta_max,
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            checkpoint_dir=CHECKPOINT_DIR,
            use_wandb=True,
            wandb_entity=WANDB_ENTITY,
            wandb_project=WANDB_PROJECT,
            wandb_run_name=wandb_run_name,
            wandb_tags=base_wandb_tags + [f"seed_{SEED}", f"pool_{BILSTM_POOLING}"],
            bilstm_pooling=BILSTM_POOLING,
            seed=SEED,
            weather_source=WEATHER_SOURCE,
            feature_set=FEATURE_SET,
        )

        curve = " -> ".join(f"{l:.4f}" for l in loss_history)
        val_curve = " -> ".join(f"{l:.4f}" for l in val_loss_history)
        print(f"    Train loss: {curve}")
        print(f"    Val   loss: {val_curve}")
        best_val = min(val_loss_history)
        best_ep = val_loss_history.index(best_val) + 1
        print(f"    Best val:   {best_val:.4f} @ epoch {best_ep}")

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
                "seq_len": seq_len,
                "patch_len": patch_len,
                "stride": stride,
                "d_model": 128,
                "gat_dim": 96,
                "gat_heads": 4,
                "gat_layers": 1,
                "dropout": 0.0,
                "seed": SEED,
                "qs_loss_weighting": qs_loss_weighting,
                "qs_loss_floor": qs_loss_floor,
            }, f)
        with open(f"{CHECKPOINT_DIR}/training_config.json", "w") as f:
            json.dump({
                "eta_max": eta_max,
                "calibration_kpi": calibration_kpi,
                "qs_loss_weighting": qs_loss_weighting,
                "qs_loss_floor": qs_loss_floor,
                "ablation": f"seq_len_{seq_len}",
                "description": f"ST-GNN trained with {seq_len}h temporal context + Open-Meteo features",
                "checkpoint_dir": CHECKPOINT_DIR,
                "seed": SEED,
            }, f)
        print(f"    Checkpoint saved -> {CHECKPOINT_DIR}/")

        seed_summary.append({
            "seed": SEED,
            "best_val_loss": best_val,
            "best_epoch": best_ep,
            "final_train_loss": loss_history[-1],
            "checkpoint_dir": CHECKPOINT_DIR,
            "qs_loss_weighting": qs_loss_weighting,
            "qs_loss_floor": qs_loss_floor,
        })

    print(f"\n{'='*62}\nMulti-seed summary\n{'='*62}")
    for s in seed_summary:
        print(f"  seed={s['seed']:>5}  best_val={s['best_val_loss']:.4f} @ ep {s['best_epoch']:>2}  "
              f"final_train={s['final_train_loss']:.4f}")
    vals = [s["best_val_loss"] for s in seed_summary]
    if len(vals) > 1:
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"\n  best_val_loss: mean={mean:.4f}  std={std:.4f}  n={len(vals)}")

    summary_path = f"{checkpoint_dir_base}{quality_suffix}_multi_seed_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "seeds": SEEDS,
            "qs_loss_weighting": qs_loss_weighting,
            "qs_loss_floor": qs_loss_floor,
            "runs": seed_summary,
        }, f, indent=2)
    print(f"\n  Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
