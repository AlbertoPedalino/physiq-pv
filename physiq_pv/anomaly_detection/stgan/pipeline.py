"""Train-only feature scaling and paper-protocol scoring for grid ConvGRU."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from ..common import runtime_environment, seed_everything
from .config import ALIGNMENT_POLICY, REFERENCE_CONFIG, REFERENCE_SEED, STGANCNNConfig
from .data import STGANWindowDataset, prepend_training_context_to_test, normalized_memmap
from .grid import build_spatial_grid
from .model import STGAN
from .result import STGANResult
from .loading import make_loader, close_loader
from .training import gan_train_step, DeviceLossTotals
from .sampling import EpochShuffleSampler
from .scoring import score_components, normalize_mc_scores, fit_calibration_ranges


def _feature_minmax(data: np.ndarray, chunk_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    chunk_size = min(chunk_size, max(1, (16 * 1024**2) // (int(np.prod(data.shape[1:])) * 8)))
    minimum = np.full(data.shape[2], np.inf, dtype=np.float64)
    maximum = np.full(data.shape[2], -np.inf, dtype=np.float64)
    for start in range(0, data.shape[0], chunk_size):
        block = np.asarray(data[start : start + chunk_size], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("STGAN training data contain non-finite values.")
        minimum = np.minimum(minimum, np.min(block, axis=(0, 1)))
        maximum = np.maximum(maximum, np.max(block, axis=(0, 1)))
    scale = maximum - minimum
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    if not np.isfinite(minimum).all():
        raise ValueError("STGAN training data contain non-finite values.")
    return minimum.astype(np.float32), scale.astype(np.float32)


def load_stgan_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[STGAN, dict]:
    import torch

    resolved = Path(checkpoint_path).resolve()
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "STGAN checkpoint loading requested CUDA, but no CUDA GPU is visible."
        )
    torch_device = torch.device(device)
    payload = torch.load(resolved, map_location=torch_device, weights_only=False)
    if payload.get("model_class") == "STGAN_CNN":
        raise ValueError("Legacy feed-forward CNN checkpoint is incompatible with ConvGRU; retrain in a new output directory.")
    if payload.get("format_version") != 2 or payload.get("model_class") != "STGAN_CONVGRU":
        raise ValueError(f"Unsupported STGAN checkpoint: {resolved}")
    model = STGAN(**payload["model_config"]).to(torch_device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def fit_and_score_stgan(
    train: np.ndarray,
    test: np.ndarray,
    *,
    calibration: np.ndarray,
    calibration_timestamps: pd.DatetimeIndex,
    train_timestamps: pd.DatetimeIndex,
    test_timestamps: pd.DatetimeIndex,
    location_names: tuple[str, ...],
    feature_names: tuple[str, ...],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    epochs: int = REFERENCE_CONFIG.epochs,
    batch_size: int = REFERENCE_CONFIG.batch_size,
    lr: float = REFERENCE_CONFIG.learning_rate,
    generator_reconstruction_weight: float = REFERENCE_CONFIG.generator_reconstruction_weight,
    hidden_size: int = REFERENCE_CONFIG.hidden_size,
    n_layers: int = REFERENCE_CONFIG.n_layers,
    cnn_channels: int = REFERENCE_CONFIG.cnn_channels,
    cnn_layers: int = REFERENCE_CONFIG.cnn_layers,
    patch_size: int = REFERENCE_CONFIG.patch_size,
    kernel_size: int = REFERENCE_CONFIG.kernel_size,
    grid_crs: str = REFERENCE_CONFIG.grid_crs,
    grid_spacing: float = REFERENCE_CONFIG.grid_spacing,
    grid_tolerance: float = REFERENCE_CONFIG.grid_tolerance,
    recent_steps: int = REFERENCE_CONFIG.recent_steps,
    trend_steps: int = REFERENCE_CONFIG.trend_steps,
    score_stride: int = REFERENCE_CONFIG.score_stride,
    train_samples_per_epoch: int | None = REFERENCE_CONFIG.train_samples_per_epoch,
    device: str = "cuda",
    seed: int = REFERENCE_SEED,
    checkpoint_path: str | Path | None = None,
    num_workers: int = REFERENCE_CONFIG.num_workers,
    train_num_workers: int | None = REFERENCE_CONFIG.train_num_workers,
    score_num_workers: int | None = REFERENCE_CONFIG.score_num_workers,
    score_batch_size: int | None = REFERENCE_CONFIG.score_batch_size,
    log_interval: int | None = REFERENCE_CONFIG.log_interval,
    persistent_workers: bool = REFERENCE_CONFIG.persistent_workers,
    prefetch_factor: int = REFERENCE_CONFIG.prefetch_factor,
    pin_memory: bool = REFERENCE_CONFIG.pin_memory,
    cache_normalized: bool = REFERENCE_CONFIG.cache_normalized,
    normalized_cache_dir: str | Path | None = None,
    shuffle_mode: str = REFERENCE_CONFIG.shuffle_mode,
    shuffle_block_size: int = REFERENCE_CONFIG.shuffle_block_size,
    execution_mode: str = REFERENCE_CONFIG.execution_mode,
    score_storage: str = REFERENCE_CONFIG.score_storage,
    score_memory_limit_mb: int = REFERENCE_CONFIG.score_memory_limit_mb,
    score_chunk_size: int = REFERENCE_CONFIG.score_chunk_size,
    score_dir: str | Path | None = None,
    timestep_hours: float = 1.0,
    dataset_name: str = "pvgis",
    angular_grid_spacing: float | None = None,
    grid_audit_knn: bool = True,
    dropout_enabled: bool = REFERENCE_CONFIG.dropout_enabled,
    dropout_p: float = REFERENCE_CONFIG.dropout_p,
    mc_dropout_enabled: bool = REFERENCE_CONFIG.mc_dropout_enabled,
    mc_samples: int = REFERENCE_CONFIG.mc_samples,
    save_raw_mc: bool = REFERENCE_CONFIG.save_raw_mc,
) -> STGANResult:
    """Train, fit calibration ranges, then score test with frozen normalization.

    Calibration is required and chronologically disjoint from training/test.
    batch_size is training-only; score_batch_size controls both inference phases.
    """
    import torch
    from torch.utils.data import RandomSampler

    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if any(data.ndim != 3 for data in (train, calibration, test)):
        raise ValueError("STGAN train/calibration/test arrays must be [time,location,feature].")
    expected_tail = (len(location_names), len(feature_names))
    if any(tuple(data.shape[1:]) != expected_tail for data in (train, calibration, test)):
        raise ValueError(f"STGAN arrays must end in {expected_tail}.")
    if not np.isfinite(timestep_hours) or timestep_hours <= 0:
        raise ValueError("timestep_hours must be finite and positive.")
    for name, times, data in (("train", train_timestamps, train),
                              ("calibration", calibration_timestamps, calibration),
                              ("test", test_timestamps, test)):
        if len(times) != len(data) or len(times) < 1 or np.any(np.diff(times.as_unit("ns").asi8) != pd.Timedelta(hours=timestep_hours).value):
            raise ValueError(f"STGAN {name} timestamps must match the data and form a contiguous {timestep_hours:g}h series.")
        validation_rows = max(1, (16 * 1024**2) // (int(np.prod(data.shape[1:])) * data.dtype.itemsize))
        for start in range(0, len(data), validation_rows):
            if not np.isfinite(data[start:start + validation_rows]).all():
                raise ValueError(f"Observed {name} values must be finite; the mask represents absent sites only.")
    splits = {name: {"start": str(times[0]), "end": str(times[-1]), "timestamps": len(times)}
              for name, times in (("train", train_timestamps), ("calibration", calibration_timestamps),
                                  ("test", test_timestamps))}
    seed_everything(seed)
    STGANCNNConfig(epochs=epochs, batch_size=batch_size, learning_rate=lr,
        generator_reconstruction_weight=generator_reconstruction_weight,
        hidden_size=hidden_size, n_layers=n_layers, cnn_channels=cnn_channels,
        cnn_layers=cnn_layers, patch_size=patch_size, kernel_size=kernel_size,
        recent_steps=recent_steps,
        trend_steps=trend_steps, score_stride=score_stride,
        train_samples_per_epoch=train_samples_per_epoch or 0,
        grid_crs=grid_crs, grid_spacing=grid_spacing, grid_tolerance=grid_tolerance,
        num_workers=num_workers, persistent_workers=persistent_workers,
        train_num_workers=train_num_workers, score_num_workers=score_num_workers,
        score_batch_size=score_batch_size, log_interval=log_interval,
        prefetch_factor=prefetch_factor, pin_memory=pin_memory,
        shuffle_mode=shuffle_mode, shuffle_block_size=shuffle_block_size,
        execution_mode=execution_mode, score_storage=score_storage,
        dropout_enabled=dropout_enabled, dropout_p=dropout_p,
        mc_dropout_enabled=mc_dropout_enabled, mc_samples=mc_samples,
        save_raw_mc=save_raw_mc,
        score_memory_limit_mb=score_memory_limit_mb, score_chunk_size=score_chunk_size)
    preparation_start = perf_counter()
    grid = build_spatial_grid(latitudes, longitudes, patch_size=patch_size,
        grid_crs=grid_crs, grid_spacing=grid_spacing, grid_tolerance=grid_tolerance,
        angular_spacing=angular_grid_spacing, audit_knn=grid_audit_knn)
    if grid.n_locations != len(location_names):
        raise ValueError("Coordinates do not match location count.")
    print(f"[stgan-cnn] grid audit: {grid.metadata}", flush=True)
    minimum, scale = _feature_minmax(train)
    calibration_with_context, calibration_times_with_context = prepend_training_context_to_test(
        train,
        calibration,
        train_timestamps=train_timestamps,
        test_timestamps=calibration_timestamps,
        context_steps=trend_steps,
        materialize=False,
    )
    test_with_context, test_timestamps_with_context = prepend_training_context_to_test(
        calibration_with_context,
        test,
        train_timestamps=calibration_times_with_context,
        test_timestamps=test_timestamps,
        context_steps=trend_steps,
        materialize=False,
    )
    cache_root = (Path(normalized_cache_dir) if normalized_cache_dir is not None else
                  Path(checkpoint_path).resolve().parent / "normalized_cache" if checkpoint_path else None)
    normalized = bool(cache_normalized and cache_root is not None)
    if normalized:
        train = normalized_memmap(train, minimum, scale, cache_root / "train.npy")
        calibration_with_context = normalized_memmap(calibration_with_context, minimum, scale, cache_root / "calibration.npy")
        test_with_context = normalized_memmap(test_with_context, minimum, scale, cache_root / "test.npy")
    train_fit = STGANWindowDataset(
        train,
        train_timestamps,
        grid,
        feature_minimum=minimum,
        feature_scale=scale,
        recent_steps=recent_steps,
        trend_steps=trend_steps,
        stride=1,
        normalized=normalized,
    )
    calibration_score_data = STGANWindowDataset(
        calibration_with_context, calibration_times_with_context, grid,
        feature_minimum=minimum, feature_scale=scale, recent_steps=recent_steps,
        trend_steps=trend_steps, stride=1, normalized=normalized,
    )
    test_score_data = STGANWindowDataset(
        test_with_context,
        test_timestamps_with_context,
        grid,
        feature_minimum=minimum,
        feature_scale=scale,
        recent_steps=recent_steps,
        trend_steps=trend_steps,
        stride=score_stride,
        normalized=normalized,
    )
    if min(len(train_fit), len(calibration_score_data), len(test_score_data)) == 0:
        raise ValueError("STGAN split has no complete regular context window.")

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "STGAN was configured for CUDA, but PyTorch cannot see a CUDA GPU. "
            "Use device='cpu' explicitly only for a smoke test."
        )
    torch_device = torch.device(device)
    model_config = {
        "n_features": len(feature_names),
        "hidden_size": hidden_size,
        "n_layers": n_layers,
        "cnn_channels": cnn_channels,
        "cnn_layers": cnn_layers,
        "patch_size": patch_size,
        "kernel_size": kernel_size,
        "time_feature_size": 31,
        "dropout_enabled": dropout_enabled,
        "dropout_p": dropout_p,
    }
    model = STGAN(**model_config).to(torch_device)
    generator_optimizer = torch.optim.Adam(model.generator.parameters(), lr=lr)
    discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=lr)
    def grid_payload():
        return {**grid.metadata, "node_indices": grid.node_indices,
                "valid_mask": grid.valid_mask, "latitudes": np.asarray(latitudes),
                "longitudes": np.asarray(longitudes)}
    checkpoint_resolved = (
        None if checkpoint_path is None else Path(checkpoint_path).resolve()
    )
    if checkpoint_resolved is not None:
        checkpoint_resolved.parent.mkdir(parents=True, exist_ok=True)

    def cpu_state_dict() -> dict:
        return {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        }

    generator = torch.Generator()
    generator.manual_seed(seed)
    full_training_product = (
        train_samples_per_epoch is None or train_samples_per_epoch <= 0
    )
    sampling_description = (
        ("complete_block_shuffled_time_location_product" if shuffle_mode == "block"
         else "complete_shuffled_time_location_product") if full_training_product
        else "replacement_sampled_domain_adaptation"
    )
    if full_training_product:
        sampler = EpochShuffleSampler(train_fit, mode=shuffle_mode, seed=seed,
            block_size=shuffle_block_size, legacy_rng=shuffle_mode != "block")
        shuffle = False
    else:
        sampler = RandomSampler(
            train_fit,
            replacement=True,
            num_samples=int(train_samples_per_epoch),
            generator=generator,
        )
        shuffle = False
    train_workers = num_workers if train_num_workers is None else train_num_workers
    score_workers = num_workers if score_num_workers is None else score_num_workers
    inference_batch_size = batch_size if score_batch_size is None else score_batch_size
    loader_options = dict(num_workers=train_workers, persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor, pin_memory=pin_memory, device=torch_device,
        vectorized=execution_mode == "optimized")
    train_loader = make_loader(
        train_fit,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        generator=torch.Generator().manual_seed(seed),
        **loader_options,
    )

    batches_per_epoch = len(train_loader)
    progress_interval = log_interval if log_interval is not None else max(100, batches_per_epoch // 20)
    preparation_seconds = perf_counter() - preparation_start
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
        torch.cuda.reset_peak_memory_stats(torch_device)
    train_start = perf_counter()
    history = []
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_start = perf_counter()
            totals = DeviceLossTotals(torch_device)
            for batch_index, (
                recent,
                trend,
                mask,
                time_features,
                observed,
                _,
                _,
            ) in enumerate(train_loader, start=1):
                recent = recent.to(torch_device, non_blocking=True)
                trend = trend.to(torch_device, non_blocking=True)
                mask = mask.to(torch_device, non_blocking=True)
                time_features = time_features.to(torch_device, non_blocking=True)
                observed = observed.to(torch_device, non_blocking=True)
                generator_total, discriminator_total = gan_train_step(
                    model, (recent, trend, mask, time_features, observed),
                    generator_optimizer, discriminator_optimizer,
                    reconstruction_weight=generator_reconstruction_weight,
                    reuse_generator=False,
                    share_history=execution_mode == "optimized")
                batch_n = recent.shape[0]
                totals.update(generator_total, discriminator_total, batch_n)
                if batch_index % progress_interval == 0 or batch_index == batches_per_epoch:
                    g_value, d_value = totals.means_since_last_log()
                    print(
                        f"[stgan] epoch={epoch}/{epochs} "
                        f"batch={batch_index}/{batches_per_epoch} "
                        f"D_mean={d_value:.6f} G_mean={g_value:.6f}",
                        flush=True,
                    )
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            g_mean, d_mean = totals.means()
            history.append({"epoch": epoch, "generator_loss": g_mean,
                            "discriminator_loss": d_mean,
                            "samples": totals.samples, "seconds": perf_counter() - epoch_start})
            if checkpoint_resolved is not None:
                pd.DataFrame(history).to_csv(checkpoint_resolved.parent / "training_history.csv", index=False)
                epoch_path = checkpoint_resolved.with_name(
                    f"{checkpoint_resolved.stem}_epoch_{epoch}{checkpoint_resolved.suffix}"
                )
                torch.save(
                    {
                        "format_version": 2,
                        "model_class": "STGAN_CONVGRU",
                        "window_config": {"recent_steps": recent_steps, "trend_steps": trend_steps},
                        "timestep_hours": timestep_hours,
                        "splits": splits,
                        "model_state_dict": cpu_state_dict(),
                        "model_config": model_config,
                        "mc_config": {"mc_dropout_enabled": mc_dropout_enabled, "mc_samples": mc_samples},
                        "completed_epochs": epoch,
                        "normalization": {
                            "kind": "training_only_feature_minmax_to_minus_one_one",
                            "minimum": minimum,
                            "scale": scale,
                        },
                        "grid": grid_payload(),
                        "seed": seed,
                    },
                    epoch_path,
                )

        training_seconds = perf_counter() - train_start

    finally:
        close_loader(train_loader)
    score_root = (Path(score_dir) if score_dir is not None else
                  checkpoint_resolved.parent / "scores" if checkpoint_resolved else None)
    score_loader_options = {k:v for k,v in loader_options.items() if k != "device"}
    score_loader_options["num_workers"] = score_workers
    # Finish and release calibration scratch before allocating test draws.
    if score_root is not None:
        score_root.mkdir(parents=True, exist_ok=True)
    calibration_start = perf_counter()
    with TemporaryDirectory(prefix=".mc_calibration_", dir=score_root) as scratch:
        calibration_root = score_root / "calibration" if save_raw_mc and score_root else Path(scratch)
        cg, cd, calibration_features, calibration_store = score_components(
            model, calibration_score_data, batch_size=inference_batch_size, device=torch_device,
            n_features=len(feature_names), storage=score_storage, output_dir=calibration_root,
            memory_limit_mb=score_memory_limit_mb, loader_options=score_loader_options,
            mc_dropout_enabled=mc_dropout_enabled, mc_samples=mc_samples,
            save_raw_mc=save_raw_mc, share_history=execution_mode == "optimized")
        try:
            score_normalization = {
                **fit_calibration_ranges(cg, cd, chunk_size=score_chunk_size),
                "fit_period": "calibration_only",
                "fit_axes": ["M", "T", "N"],
                "calibration_start": str(calibration_score_data.target_timestamps[0]),
                "calibration_end": str(calibration_score_data.target_timestamps[-1]),
                "calibration_shape": list(cg.shape),
                "mc_samples": mc_samples,
                "effective_mc_samples": mc_samples if mc_dropout_enabled else 1,
                "component_weight": 1.0,
                "clipping": False,
                "frozen_during_test": True,
            }
        finally:
            calibration_store.close()
        del cg, cd, calibration_features
    calibration_seconds = perf_counter() - calibration_start
    scoring_start = perf_counter()
    test_generator, test_discriminator, test_features, score_store = score_components(
        model, test_score_data, batch_size=inference_batch_size, device=torch_device,
        n_features=len(feature_names), storage=score_storage, output_dir=score_root,
        memory_limit_mb=score_memory_limit_mb, loader_options=score_loader_options,
        mc_dropout_enabled=mc_dropout_enabled, mc_samples=mc_samples,
        save_raw_mc=save_raw_mc,
        share_history=execution_mode == "optimized")
    try:
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        scoring_seconds = perf_counter() - scoring_start
        performance = {"preparation_seconds": preparation_seconds,
            "calibration_seconds": calibration_seconds,
            "training_seconds": training_seconds, "scoring_seconds": scoring_seconds,
            "training_samples_per_second": sum(r["samples"] for r in history) / training_seconds,
            "scoring_samples_per_second": len(test_score_data) / scoring_seconds,
            "peak_cuda_memory_bytes": (torch.cuda.max_memory_allocated(torch_device)
                                       if torch_device.type == "cuda" else None)}
        # The test only applies the already frozen calibration transform.
        (test_scores, anomaly_std, test_generator, test_discriminator,
         generator_range, discriminator_range) = normalize_mc_scores(
            test_generator, test_discriminator, score_store,
            normalization=score_normalization, chunk_size=score_chunk_size)
        score_store.discard_temporary_raw()
        reference_hyperparameters_used = all(
            (
                epochs == REFERENCE_CONFIG.epochs,
                batch_size == REFERENCE_CONFIG.batch_size,
                lr == REFERENCE_CONFIG.learning_rate,
                generator_reconstruction_weight
                == REFERENCE_CONFIG.generator_reconstruction_weight,
                hidden_size == REFERENCE_CONFIG.hidden_size,
                n_layers == REFERENCE_CONFIG.n_layers,
                patch_size == REFERENCE_CONFIG.patch_size,
                kernel_size == REFERENCE_CONFIG.kernel_size,
                cnn_channels == REFERENCE_CONFIG.cnn_channels,
                cnn_layers == REFERENCE_CONFIG.cnn_layers,
                recent_steps == REFERENCE_CONFIG.recent_steps,
                trend_steps == REFERENCE_CONFIG.trend_steps,
                score_stride == REFERENCE_CONFIG.score_stride,
                seed == REFERENCE_SEED,
                full_training_product,
                shuffle_mode != "block",
                not dropout_enabled,
                not mc_dropout_enabled,
            )
        )

        if checkpoint_resolved is not None:
            torch.save(
                {
                    "format_version": 2,
                    "model_class": "STGAN_CONVGRU",
                    "window_config": {"recent_steps": recent_steps, "trend_steps": trend_steps},
                    "timestep_hours": timestep_hours,
                    "splits": splits,
                    "model_state_dict": cpu_state_dict(),
                    "model_config": model_config,
                    "mc_config": {"mc_dropout_enabled": mc_dropout_enabled, "mc_samples": mc_samples},
                    "normalization": {
                        "kind": "training_only_feature_minmax_to_minus_one_one",
                        "minimum": minimum,
                        "scale": scale,
                    },
                    "score_normalization": score_normalization,
                    "grid": grid_payload(),
                    "parameter_counts": model.parameter_counts(),
                    "locations": location_names,
                    "features": feature_names,
                    "training": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "generator_reconstruction_weight": generator_reconstruction_weight,
                        "train_samples_per_epoch": train_samples_per_epoch,
                        "sampling": sampling_description,
                        "shuffle_mode": shuffle_mode,
                        "shuffle_block_size": shuffle_block_size,
                        "discriminator_targets": {
                            "real_normal": 0,
                            "generated_fake": 1,
                        },
                        "seed": seed,
                        "test_labels_used": False,
                        "test_context": {
                            "source": "preceding_calibration_history",
                            "steps": trend_steps,
                            "targets": "test_timestamps_only",
                        },
                    },
                    "environment": runtime_environment(),
                },
                checkpoint_resolved,
            )

        return STGANResult(
            _store=score_store,
            test_timestamps=test_score_data.target_timestamps,
            location_names=location_names,
            feature_names=feature_names,
            test_scores=test_scores,
            anomaly_std=anomaly_std,
            test_feature_scores=test_features,
            test_generator_scores=test_generator,
            test_discriminator_scores=test_discriminator,
            metadata={
                "splits": splits,
                "dropout_enabled": dropout_enabled,
                "dropout_p": dropout_p,
                "mc_dropout_enabled": mc_dropout_enabled,
                "mc_samples": mc_samples,
                "save_raw_mc": save_raw_mc,
                "effective_mc_samples": mc_samples if mc_dropout_enabled else 1,
                "uncertainty_definition": "population_std_of_anomaly_scores_ddof_0",
                "execution_mode": execution_mode,
                "runtime": {"num_workers": num_workers, "persistent_workers": persistent_workers,
                    "train_num_workers": train_workers, "score_num_workers": score_workers,
                    "train_batch_size": batch_size, "score_batch_size": inference_batch_size,
                    "log_interval": progress_interval,
                    "prefetch_factor": prefetch_factor, "pin_memory": pin_memory,
                    "normalized_disk_cache": normalized, "shuffle_mode": shuffle_mode,
                    "score_storage": score_store.backend, "score_chunk_size": score_chunk_size,
                    "save_raw_mc": save_raw_mc,
                    "raw_storage_policy": "retained" if save_raw_mc else "temporary_until_normalized",
                    "shuffle_block_size": shuffle_block_size,
                    "shuffle_order_equivalent_to_legacy": shuffle_mode != "block"},
                "backend": f"stgan_convgru_lstm_{dataset_name}",
                "dataset": dataset_name,
                "timestep_hours": timestep_hours,
                "trend_hours": trend_steps * timestep_hours,
                "source_branch": "feat/stgan-paper",
                "source_commit": "777df6bc6deddeccafbf806bd1c380f79ea146a1",
                "parameter_counts": model.parameter_counts(),
                "performance": performance,
                "alignment_reference": "TNNLS_2022_and_official_dleyan_STGAN",
                "alignment_policy": ALIGNMENT_POLICY,
                "paper_reference": {
                    "doi": "10.1109/TNNLS.2021.3136171",
                    "repository": "https://github.com/dleyan/STGAN",
                    "repository_commit_verified": "20d2f6b365ea003500a57737647a846fb41267aa",
                },
                "paper_alignment": {
                    "generator_discriminator_architecture": False,
                    "adversarial_losses_and_targets": True,
                    "score_equation": True,
                    "reference_hyperparameters": reference_hyperparameters_used,
                    "complete_training_product": full_training_product,
                    "domain_adaptations": [
                        "convgru_2d_gates_with_mask_instead_of_graph_convolutional_gates",
                        "pointwise_1x1_projections_instead_of_remaining_graph_convolutions",
                        "chronological_train_calibration_test_with_past_only_context",
                        "shared_score_ranges_fitted_on_calibration_only_without_clipping",
                        "target_feature_residuals_for_diagnostics",
                    ] + (["generator_spatial_temporal_fusion_dropout"] if dropout_enabled else [])
                      + (["mc_score_mean_and_population_std"] if mc_dropout_enabled else []),
                },
                "normalization": "training_only_feature_minmax",
                "score_normalization": score_normalization,
                "test_labels_used": False,
                "grid": grid.metadata,
                "reconstruction_reduction": "mean_valid_cells_and_features_per_sample_then_mean_samples",
                "test_context": {
                    "source": "preceding_calibration_history",
                    "steps": trend_steps,
                    "targets": "test_timestamps_only",
                },
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "generator_reconstruction_weight": generator_reconstruction_weight,
                "hidden_size": hidden_size,
                "n_layers": n_layers,
                "cnn_channels": cnn_channels,
                "cnn_layers": cnn_layers,
                "patch_size": patch_size,
                "kernel_size": kernel_size,
                "recent_steps": recent_steps,
                "trend_steps": trend_steps,
                "score_stride": score_stride,
                "train_samples_per_epoch": train_samples_per_epoch,
                "training_sampling": sampling_description,
                "discriminator_output_semantics": "anomaly_probability_real_0_fake_1",
                "score_definition": ("mc_mean_of_calibration_normalized_generator_plus_discriminator_gap"
                                     if mc_dropout_enabled else "calibration_normalized_generator_plus_discriminator_gap"),
                "feature_score_definition": ("mc_mean_target_node_squared_prediction_error"
                                             if mc_dropout_enabled else "target_node_squared_prediction_error"),
                "checkpoint": None if checkpoint_resolved is None else str(checkpoint_resolved),
                "device": str(torch_device),
                "seed": seed,
                "environment": runtime_environment(),
            },
        )
    except BaseException:
        score_store.close()
        raise
