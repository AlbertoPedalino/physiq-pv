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
from .gat import STGANGAT
from .graph import grid_edge_index
from .graph_data import STGANGraphDataset
from .result import STGANResult
from .loading import make_loader, close_loader
from .training import gan_train_step, DeviceLossTotals
from .monitoring import ValidationMonitor
from .pca_reference import load_pca_reference
from .mmd import MMDMonitor, load_mmd_reference
from .objective import validation_objective
from .precision import validate_precision
from .sampling import EpochShuffleSampler
from .seasonal import fit_seasonal_climatology, seasonal_memmap
from .scoring import (component_statistics, score_components, normalize_mc_scores, fit_calibration_ranges,
                      summarize_raw_mc_components, normalize_paper_mc_scores)


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
    if payload.get("format_version") != 2 or payload.get("model_class") not in ("STGAN_CONVGRU", "STGAN_GAT"):
        raise ValueError(f"Unsupported STGAN checkpoint: {resolved}")
    model_type = STGANGAT if payload["model_class"] == "STGAN_GAT" else STGAN
    model = model_type(**payload["model_config"]).to(torch_device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    payload.setdefault("precision", "fp32")
    return model, payload


def fit_and_score_stgan(
    train: np.ndarray,
    test: np.ndarray,
    *,
    train_timestamps: pd.DatetimeIndex,
    test_timestamps: pd.DatetimeIndex,
    location_names: tuple[str, ...],
    feature_names: tuple[str, ...],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    calibration: np.ndarray | None = None,
    calibration_timestamps: pd.DatetimeIndex | None = None,
    validation: np.ndarray | None = None,  # Held out of training; read only by the per-epoch monitoring.
    validation_timestamps: pd.DatetimeIndex | None = None,
    epochs: int = REFERENCE_CONFIG.epochs,
    batch_size: int | None = None,
    lr: float = REFERENCE_CONFIG.learning_rate,
    discriminator_lr_ratio: float = REFERENCE_CONFIG.discriminator_lr_ratio,
    generator_learning_rate: float | None = REFERENCE_CONFIG.generator_learning_rate,
    discriminator_learning_rate: float | None = REFERENCE_CONFIG.discriminator_learning_rate,
    adam_beta1: float = REFERENCE_CONFIG.adam_beta1,
    monitoring_timestamps: int = REFERENCE_CONFIG.monitoring_timestamps,
    monitoring_feature_mmd_every_n_epochs: int = REFERENCE_CONFIG.monitoring_feature_mmd_every_n_epochs,
    monitoring_feature_mmd_samples: int = REFERENCE_CONFIG.monitoring_feature_mmd_samples,
    pca_reference: str | Path | None = None,  # Directory written by build_pca_reference: loaded, never refitted.
    mmd_objective_window: int = REFERENCE_CONFIG.mmd_objective_window,
    mmd_reference: str | Path | None = None,  # Directory written by build_mmd_reference: loaded, never rebuilt.
    skip_final_scoring: bool = False,  # Training and validation only: the test is not scored.
    discriminator_generator_update_ratio: str = REFERENCE_CONFIG.discriminator_generator_update_ratio,
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
    time_encoding: str = REFERENCE_CONFIG.time_encoding,
    normalization: str = REFERENCE_CONFIG.normalization,
    seasonal_window_days: int = REFERENCE_CONFIG.seasonal_window_days,
    score_stride: int = REFERENCE_CONFIG.score_stride,
    train_samples_per_epoch: int | None = REFERENCE_CONFIG.train_samples_per_epoch,
    device: str = "cuda",
    precision: str = REFERENCE_CONFIG.precision,
    seed: int = REFERENCE_SEED,
    checkpoint_path: str | Path | None = None,
    resume_from: str | Path | None = None,  # Epoch checkpoint; training continues after it.
    on_epoch=None,  # Optional callback receiving a copy of the completed epoch metrics.
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
    score_mode: str = "calibrated",
    timestep_hours: float = 1.0,
    dataset_name: str = "pvgis",
    angular_grid_spacing: float | None = None,
    grid_audit_knn: bool = True,
    dropout_enabled: bool = REFERENCE_CONFIG.dropout_enabled,
    dropout_p: float = REFERENCE_CONFIG.dropout_p,
    mc_dropout_enabled: bool = REFERENCE_CONFIG.mc_dropout_enabled,
    mc_samples: int = REFERENCE_CONFIG.mc_samples,
    save_raw_mc: bool = REFERENCE_CONFIG.save_raw_mc,
    spatial_encoder: str = REFERENCE_CONFIG.spatial_encoder,
    gat_hidden_dim: int = REFERENCE_CONFIG.gat_hidden_dim,
    gat_heads: int = REFERENCE_CONFIG.gat_heads,
    gat_layers: int = REFERENCE_CONFIG.gat_layers,
    gat_recurrence: str = REFERENCE_CONFIG.gat_recurrence,
    discriminator_chunk_size: int = REFERENCE_CONFIG.discriminator_chunk_size,
    trend_chunk_size: int = REFERENCE_CONFIG.trend_chunk_size,
) -> STGANResult:
    """Train and score, optionally using separate calibration for legacy runs."""
    import torch
    from torch.utils.data import RandomSampler

    validate_precision(precision, device)

    graph_mode = spatial_encoder == "gat"
    if batch_size is None:
        batch_size = 1 if graph_mode else REFERENCE_CONFIG.batch_size
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if score_mode not in ("calibrated", "paper"):
        raise ValueError("score_mode must be calibrated or paper.")
    if score_mode == "calibrated" and (calibration is None or calibration_timestamps is None):
        raise ValueError("Calibrated scoring requires calibration data and timestamps.")
    if score_mode == "paper" and (calibration is not None or calibration_timestamps is not None):
        raise ValueError("Paper scoring does not use calibration data.")
    # Data held out of training: the calibration period of calibrated scoring, or a validation
    # period. A validation is read only by the per-epoch monitoring and by the validation
    # objective: it never fits score ranges, thresholds or the normalization of the test.
    holdout, holdout_timestamps, holdout_name = calibration, calibration_timestamps, "calibration"
    if validation is not None or validation_timestamps is not None:
        if validation is None or validation_timestamps is None or calibration is not None:
            raise ValueError("Validation needs data and timestamps, and excludes calibration.")
        holdout, holdout_timestamps, holdout_name = validation, validation_timestamps, "validation"
    if pca_reference is not None and normalization == "seasonal":
        raise ValueError("The PCA reference uses the min-max normalization: set normalization=minmax.")
    if mmd_reference is not None and (pca_reference is None or holdout_name != "validation"):
        raise ValueError("The MMD reference needs the PCA reference it was built on and a validation period.")
    if type(skip_final_scoring) is not bool:
        raise ValueError("skip_final_scoring must be a boolean.")
    datasets = [("train", train_timestamps, train)]
    if holdout is not None:
        datasets.append((holdout_name, holdout_timestamps, holdout))
    datasets.append(("test", test_timestamps, test))
    if any(data.ndim != 3 for _, _, data in datasets):
        raise ValueError("STGAN arrays must be [time,location,feature].")
    expected_tail = (len(location_names), len(feature_names))
    if any(tuple(data.shape[1:]) != expected_tail for _, _, data in datasets):
        raise ValueError(f"STGAN arrays must end in {expected_tail}.")
    if not np.isfinite(timestep_hours) or timestep_hours <= 0:
        raise ValueError("timestep_hours must be finite and positive.")
    for name, times, data in datasets:
        if len(times) != len(data) or len(times) < 1 or np.any(np.diff(times.as_unit("ns").asi8) != pd.Timedelta(hours=timestep_hours).value):
            raise ValueError(f"STGAN {name} timestamps must match the data and form a contiguous {timestep_hours:g}h series.")
        validation_rows = max(1, (16 * 1024**2) // (int(np.prod(data.shape[1:])) * data.dtype.itemsize))
        for start in range(0, len(data), validation_rows):
            if not np.isfinite(data[start:start + validation_rows]).all():
                raise ValueError(f"Observed {name} values must be finite; the mask represents absent sites only.")
    splits = {name: {"start": str(times[0]), "end": str(times[-1]), "timestamps": len(times)}
              for name, times, _ in datasets}
    seed_everything(seed)
    validated_config = STGANCNNConfig(epochs=epochs, batch_size=batch_size, learning_rate=lr, precision=precision,
        discriminator_lr_ratio=discriminator_lr_ratio,
        generator_learning_rate=generator_learning_rate, discriminator_learning_rate=discriminator_learning_rate,
        adam_beta1=adam_beta1,
        monitoring_timestamps=monitoring_timestamps, monitoring_feature_mmd_every_n_epochs=monitoring_feature_mmd_every_n_epochs,
        monitoring_feature_mmd_samples=monitoring_feature_mmd_samples,
        mmd_objective_window=mmd_objective_window,
        discriminator_generator_update_ratio=discriminator_generator_update_ratio,
        generator_reconstruction_weight=generator_reconstruction_weight,
        hidden_size=hidden_size, n_layers=n_layers, cnn_channels=cnn_channels,
        cnn_layers=cnn_layers, patch_size=patch_size, kernel_size=kernel_size,
        recent_steps=recent_steps,
        trend_steps=trend_steps, time_encoding=time_encoding, score_stride=score_stride,
        normalization=normalization, seasonal_window_days=seasonal_window_days,
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
        spatial_encoder=spatial_encoder, gat_hidden_dim=gat_hidden_dim,
        gat_heads=gat_heads, gat_layers=gat_layers, gat_recurrence=gat_recurrence,
        discriminator_chunk_size=discriminator_chunk_size,
        trend_chunk_size=trend_chunk_size,
        score_memory_limit_mb=score_memory_limit_mb, score_chunk_size=score_chunk_size)
    preparation_start = perf_counter()
    grid = build_spatial_grid(latitudes, longitudes, patch_size=patch_size,
        grid_crs=grid_crs, grid_spacing=grid_spacing, grid_tolerance=grid_tolerance,
        angular_spacing=angular_grid_spacing, audit_knn=grid_audit_knn)
    if grid.n_locations != len(location_names):
        raise ValueError("Coordinates do not match location count.")
    edges = grid_edge_index(grid.row_indices, grid.column_indices) if graph_mode else None
    if graph_mode:
        grid.metadata.update(adjacency_used=True, adjacency_storage="sparse_edge_index",
                             patches_used_by="discriminator_only")
    graph_metadata = ({"nodes": grid.n_locations, "directed_edges_including_self": edges.shape[1],
                       "neighborhood": "8_immediate_grid_neighbors_plus_self",
                       "gat_layers": gat_layers, "gat_heads": gat_heads,
                       "gat_hidden_dim_per_head": gat_hidden_dim,
                       "gat_recurrence": gat_recurrence,
                       "batch_unit": "global_timestamps"} if graph_mode else None)
    print(f"[stgan-{spatial_encoder}] grid audit: {grid.metadata}; graph={graph_metadata}", flush=True)
    seasonal = normalization == "seasonal"
    if not seasonal:
        minimum, scale = _feature_minmax(train)
    if holdout is not None:
        holdout_with_context, holdout_times_with_context = prepend_training_context_to_test(
            train, holdout, train_timestamps=train_timestamps,
            test_timestamps=holdout_timestamps, context_steps=trend_steps,
            materialize=False)
    else:
        holdout_with_context = None
        holdout_times_with_context = None
    test_with_context, test_timestamps_with_context = prepend_training_context_to_test(
        holdout_with_context if holdout is not None else train,
        test,
        train_timestamps=holdout_times_with_context if holdout is not None else train_timestamps,
        test_timestamps=test_timestamps,
        context_steps=trend_steps,
        materialize=False,
    )
    cache_root = (Path(normalized_cache_dir) if normalized_cache_dir is not None else
                  Path(checkpoint_path).resolve().parent / "normalized_cache" if checkpoint_path else None)
    normalized = bool(cache_normalized and cache_root is not None) and not seasonal
    seasonal_metadata = None
    if seasonal:
        # Standardised values live in the disk cache; the datasets then apply the
        # usual train-only min-max on top, so targets keep the generator's range.
        if cache_root is None:
            raise ValueError("Seasonal normalization needs checkpoint_path or normalized_cache_dir.")
        climatology = fit_seasonal_climatology(train, train_timestamps, cache_root,
                                               window_days=seasonal_window_days)
        try:
            seasonal_metadata = climatology.metadata
            train = seasonal_memmap(train, train_timestamps, climatology, cache_root / "train.npy")
            if holdout_with_context is not None:
                holdout_with_context = seasonal_memmap(holdout_with_context,
                    holdout_times_with_context, climatology, cache_root / f"{holdout_name}.npy")
            test_with_context = seasonal_memmap(test_with_context, test_timestamps_with_context,
                                                climatology, cache_root / "test.npy")
        finally:
            climatology.close()
        minimum, scale = _feature_minmax(train)
    if normalized:
        train = normalized_memmap(train, minimum, scale, cache_root / "train.npy")
        if holdout_with_context is not None:
            holdout_with_context = normalized_memmap(holdout_with_context, minimum, scale, cache_root / f"{holdout_name}.npy")
        test_with_context = normalized_memmap(test_with_context, minimum, scale, cache_root / "test.npy")
    normalization_kind = ("training_only_seasonal_standardisation_then_feature_minmax_to_minus_one_one"
                          if seasonal else "training_only_feature_minmax_to_minus_one_one")
    dataset_type = STGANGraphDataset if graph_mode else STGANWindowDataset
    train_fit = dataset_type(
        train,
        train_timestamps,
        grid,
        feature_minimum=minimum,
        feature_scale=scale,
        recent_steps=recent_steps,
        trend_steps=trend_steps,
        stride=1,
        normalized=normalized,
        time_encoding=time_encoding, longitudes=longitudes,
    )
    holdout_score_data = (dataset_type(
        holdout_with_context, holdout_times_with_context, grid,
        feature_minimum=minimum, feature_scale=scale, recent_steps=recent_steps,
        trend_steps=trend_steps, stride=1, normalized=normalized,
        time_encoding=time_encoding, longitudes=longitudes,
    ) if holdout_with_context is not None else None)
    test_score_data = dataset_type(
        test_with_context,
        test_timestamps_with_context,
        grid,
        feature_minimum=minimum,
        feature_scale=scale,
        recent_steps=recent_steps,
        trend_steps=trend_steps,
        stride=score_stride,
        normalized=normalized,
        time_encoding=time_encoding, longitudes=longitudes,
    )
    if not len(train_fit) or not len(test_score_data) or (holdout_score_data is not None and not len(holdout_score_data)):
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
        "time_feature_size": train_fit.time_feature_size,
        "dropout_enabled": dropout_enabled,
        "dropout_p": dropout_p,
    }
    if graph_mode:
        model_config.update(edge_index=edges, node_indices=grid.node_indices,
            recent_steps=recent_steps, gat_hidden_dim=gat_hidden_dim, gat_heads=gat_heads,
            gat_layers=gat_layers, discriminator_chunk_size=discriminator_chunk_size,
            trend_chunk_size=trend_chunk_size, gat_recurrence=gat_recurrence)
    model_type = STGANGAT if graph_mode else STGAN
    model_class = "STGAN_GAT" if graph_mode else "STGAN_CONVGRU"
    model = model_type(**model_config).to(torch_device)
    # From here on lr is the generator's effective rate, whichever field gave it.
    lr = validated_config.effective_generator_learning_rate
    discriminator_lr = validated_config.effective_discriminator_learning_rate
    learning_rates = {"learning_rate": lr, "generator_learning_rate": lr,
                      "discriminator_learning_rate": discriminator_lr,
                      "adam_beta1": adam_beta1,
                      # Diagnostic: the optimizers use the two rates above.
                      "discriminator_lr_ratio": (discriminator_lr_ratio if discriminator_learning_rate is None
                                                 else discriminator_lr / lr)}
    # Optimizer steps per batch; unrelated to the learning rates above.
    update_steps = {"discriminator_generator_update_ratio": discriminator_generator_update_ratio,
                    "discriminator_updates_per_batch": validated_config.discriminator_updates_per_batch,
                    "generator_updates_per_batch": validated_config.generator_updates_per_batch}
    betas = (adam_beta1, 0.999)
    generator_optimizer = torch.optim.Adam(model.generator.parameters(), lr=lr, betas=betas)
    discriminator_optimizer = torch.optim.Adam(
        model.discriminator.parameters(), lr=discriminator_lr, betas=betas)
    def grid_payload():
        return {**grid.metadata, "graph": graph_metadata, "node_indices": grid.node_indices,
                "valid_mask": grid.valid_mask, "latitudes": np.asarray(latitudes),
                "longitudes": np.asarray(longitudes)}
    checkpoint_resolved = (
        None if checkpoint_path is None else Path(checkpoint_path).resolve()
    )
    if checkpoint_resolved is not None:
        checkpoint_resolved.parent.mkdir(parents=True, exist_ok=True)

    def optimizer_steps(optimizer) -> int:
        """Effective optimizer.step() calls so far, read from Adam's own counter."""
        return max((int(state["step"]) for state in optimizer.state.values()), default=0)

    def cpu_state_dict() -> dict:
        return {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        }

    # Fixed PCA feature space, fitted once on training data before any run. It is loaded and
    # checked against this run's variables, locations and normalization; it is never fitted here
    # and nothing of the training, of the score or of the evaluation uses it.
    pca_summary = mmd_monitor = None
    if pca_reference is not None:
        loaded_pca = load_pca_reference(pca_reference)
        loaded_pca.check(minimum=minimum, scale=scale, feature_names=feature_names, n_locations=len(location_names))
        pca_summary = loaded_pca.summary()
        if mmd_reference is not None:
            # MMD in that PCA space, on validation subsets and with a bandwidth fixed once, before any
            # run: validation observations against their reconstructions, every epoch. Read-only too.
            mmd_monitor = MMDMonitor(holdout_score_data, load_mmd_reference(mmd_reference), loaded_pca,
                batch_size=batch_size if score_batch_size is None else score_batch_size,
                device=torch_device, precision=precision)
        del loaded_pca
    mmd_fingerprint = None if mmd_monitor is None else mmd_monitor.reference.fingerprint
    completed_epochs = 0
    prior_history = []
    resume_metadata = None
    monitoring_state = None
    if resume_from is not None:
        resume_path = Path(resume_from).resolve()
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if "completed_epochs" not in payload:
            raise ValueError(f"Not an epoch checkpoint (no completed_epochs): {resume_path}")
        completed_epochs = int(payload["completed_epochs"])
        if completed_epochs > epochs:
            raise ValueError(f"Checkpoint has {completed_epochs} completed epochs, but epochs={epochs}.")
        # Keys absent from older checkpoints are not compared.
        expected = {"format_version": 2, "model_class": model_class, "seed": seed,
                    "time_encoding": time_encoding, "timestep_hours": timestep_hours,
                    "window_config": {"recent_steps": recent_steps, "trend_steps": trend_steps},
                    "generator_reconstruction_weight": generator_reconstruction_weight}
        mismatched = [name for name, value in expected.items() if payload.get(name, value) != value]
        # Checkpoints written before time_encoding stored annual_cycle: False is the
        # one-hot encoding; True (one-hot plus annual phase) can no longer be built.
        legacy_annual = payload.get("annual_cycle") if "time_encoding" not in payload else None
        if legacy_annual is True or (legacy_annual is False and time_encoding != "onehot"):
            mismatched.append("time_encoding")
        saved = payload["normalization"]
        if (np.shape(saved["minimum"]) != minimum.shape
                or not np.array_equal(saved["minimum"], minimum)
                or not np.array_equal(saved["scale"], scale)):
            mismatched.append("normalization")
        if saved.get("kind", normalization_kind) != normalization_kind:
            mismatched.append("normalization kind")
        # Checkpoints written before this option used the pointwise recurrence.
        if graph_mode and payload["model_config"].get("gat_recurrence", "pointwise") != gat_recurrence:
            mismatched.append("gat_recurrence")
        # Every checkpoint written before this option used one D and one G step per batch.
        if payload.get("discriminator_generator_update_ratio", "1:1") != discriminator_generator_update_ratio:
            mismatched.append("discriminator_generator_update_ratio")
        if "learning_rates" in payload and any(
                payload["learning_rates"].get(name) != learning_rates[name]
                for name in ("generator_learning_rate", "discriminator_learning_rate")):
            mismatched.append("learning_rates")
        # The training period, and whether a validation period is held out of it.
        if "splits" in payload and (payload["splits"].get("train") != splits["train"]
                                    or ("validation" in payload["splits"]) != ("validation" in splits)):
            mismatched.append("splits")
        # Adam.load_state_dict restores LR as well as moments. Reject a mismatch
        # rather than silently training with rates different from the run config.
        for name in ("generator", "discriminator"):
            state = payload.get(f"{name}_optimizer_state_dict")
            if state is not None and any(
                    group["lr"] != learning_rates[f"{name}_learning_rate"]
                    for group in state["param_groups"]):
                mismatched.append(f"{name}_learning_rate")
            # The same holds for the betas: every earlier checkpoint was trained with 0.9.
            if state is not None and "adam_beta1" not in mismatched and any(
                    group["betas"][0] != adam_beta1 for group in state["param_groups"]):
                mismatched.append("adam_beta1")
        # The rolling MMD continues across the interruption only on the same reference.
        if "mmd_reference" in payload and payload["mmd_reference"] != mmd_fingerprint:
            mismatched.append("mmd_reference")
        if mismatched:
            raise ValueError(f"Resume checkpoint does not match this run: {mismatched}")
        model.load_state_dict(payload["model_state_dict"])
        monitoring_state = payload.get("monitoring")
        optimizer_restored = "generator_optimizer_state_dict" in payload
        if optimizer_restored:
            generator_optimizer.load_state_dict(payload["generator_optimizer_state_dict"])
            discriminator_optimizer.load_state_dict(payload["discriminator_optimizer_state_dict"])
        history_path = resume_path.parent / "training_history.csv"
        if history_path.is_file():
            saved_history = pd.read_csv(history_path)
            prior_history = saved_history[saved_history["epoch"] <= completed_epochs].to_dict("records")
        resume_metadata = {"checkpoint": str(resume_path), "completed_epochs": completed_epochs,
                           "optimizer_state_restored": optimizer_restored}
        print(f"[stgan] resume from {resume_path}: completed_epochs={completed_epochs}/{epochs} "
              f"optimizer_state={'restored' if optimizer_restored else 'absent, Adam moments restart'}",
              flush=True)
        del payload

    monitor = None
    if holdout_name == "validation" and monitoring_timestamps:
        monitor = ValidationMonitor(holdout_score_data, timestamps=monitoring_timestamps,
            batch_size=batch_size if score_batch_size is None else score_batch_size,
            device=torch_device, precision=precision, feature_mmd_every_n_epochs=monitoring_feature_mmd_every_n_epochs,
            feature_mmd_samples=monitoring_feature_mmd_samples, state=monitoring_state)
    print(f"[stgan] Adam lr_G={lr:g} lr_D={discriminator_lr:g} "
          f"lr_D/lr_G={learning_rates['discriminator_lr_ratio']:g} updates_D:G={discriminator_generator_update_ratio} "
          f"reconstruction_weight={generator_reconstruction_weight:g} beta1={adam_beta1:g}",
          flush=True)
    generator = torch.Generator()
    generator.manual_seed(seed + completed_epochs)
    full_training_product = (
        train_samples_per_epoch is None or train_samples_per_epoch <= 0
    )
    sampling_description = (
        ("complete_block_shuffled_time_location_product" if shuffle_mode == "block"
         else "complete_shuffled_time_location_product") if full_training_product
        else "replacement_sampled_domain_adaptation"
    )
    if graph_mode:
        sampling_description = ("complete_shuffled_timestamps_all_nodes" if full_training_product
                                else "replacement_sampled_timestamps_all_nodes")
    if full_training_product:
        sampler = EpochShuffleSampler(train_fit, mode=shuffle_mode, seed=seed,
            block_size=shuffle_block_size, legacy_rng=shuffle_mode != "block")
        sampler.skip_epochs(completed_epochs)
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
        for epoch in range(completed_epochs + 1, epochs + 1):
            model.train()
            epoch_start = perf_counter()
            totals = DeviceLossTotals(torch_device)
            steps_before = optimizer_steps(discriminator_optimizer), optimizer_steps(generator_optimizer)
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
                generator_total, discriminator_total, terms = gan_train_step(
                    model, (recent, trend, mask, time_features, observed),
                    generator_optimizer, discriminator_optimizer,
                    reconstruction_weight=generator_reconstruction_weight,
                    reuse_generator=False,
                    share_history=execution_mode == "optimized", precision=precision,
                    discriminator_steps=update_steps["discriminator_updates_per_batch"],
                    generator_steps=update_steps["generator_updates_per_batch"], return_terms=True)
                batch_n = recent.shape[0]
                totals.update(generator_total, discriminator_total, batch_n, terms)
                if batch_index % progress_interval == 0 or batch_index == batches_per_epoch:
                    g_value, d_value = totals.means_since_last_log()
                    print(
                        f"[stgan] precision={precision} epoch={epoch}/{epochs} "
                        f"batch={batch_index}/{batches_per_epoch} "
                        f"D_mean={d_value:.6f} G_mean={g_value:.6f}",
                        flush=True,
                    )
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            g_mean, d_mean = totals.means()
            record = {"epoch": epoch, "generator_loss": g_mean, "discriminator_loss": d_mean,
                      "samples": totals.samples, "seconds": perf_counter() - epoch_start,
                      "discriminator_updates": optimizer_steps(discriminator_optimizer) - steps_before[0],
                      "generator_updates": optimizer_steps(generator_optimizer) - steps_before[1],
                      **totals.term_means()}
            if monitor is not None:
                # Reads the model only: RNG streams, modes and weights are left as found.
                monitoring_start = perf_counter()
                record.update(monitor.evaluate(model, epoch))
                record["validation_seconds"] = perf_counter() - monitoring_start
            if mmd_monitor is not None:
                # Reads the model only, like the monitoring above.
                mmd_start = perf_counter()
                record.update(mmd_monitor.evaluate(model))
                earlier = [float(past["validation_pca_mmd_mean"]) for past in prior_history + history
                           if past.get("validation_pca_mmd_mean") is not None
                           and np.isfinite(past["validation_pca_mmd_mean"])]
                # The sweep objective: the mean over the last epochs, all of them when fewer are available.
                record["validation_pca_mmd_rolling_mean"] = float(np.mean(
                    (earlier + [record["validation_pca_mmd_mean"]])[-mmd_objective_window:]))
                record["validation_pca_mmd_seconds"] = perf_counter() - mmd_start
            history.append(record)
            if on_epoch is not None:
                on_epoch(dict(history[-1]))
            if checkpoint_resolved is not None:
                pd.DataFrame(prior_history + history).to_csv(checkpoint_resolved.parent / "training_history.csv", index=False)
                epoch_path = checkpoint_resolved.with_name(
                    f"{checkpoint_resolved.stem}_epoch_{epoch}{checkpoint_resolved.suffix}"
                )
                torch.save(
                    {
                        "format_version": 2, "precision": precision,
                        "model_class": model_class,
                        "window_config": {"recent_steps": recent_steps, "trend_steps": trend_steps},
                        "time_encoding": time_encoding,
                        "timestep_hours": timestep_hours,
                        "splits": splits,
                        "model_state_dict": cpu_state_dict(),
                        "generator_optimizer_state_dict": generator_optimizer.state_dict(),
                        "discriminator_optimizer_state_dict": discriminator_optimizer.state_dict(),
                        "model_config": model_config,
                        "mc_config": {"mc_dropout_enabled": mc_dropout_enabled, "mc_samples": mc_samples},
                        "completed_epochs": epoch,
                        "learning_rates": learning_rates,
                        "monitoring": None if monitor is None else monitor.state(),
                        "pca_reference": None if pca_summary is None else pca_summary["fingerprint"],
                        "mmd_reference": mmd_fingerprint,
                        "discriminator_generator_update_ratio": discriminator_generator_update_ratio,
                        "generator_reconstruction_weight": generator_reconstruction_weight,
                        "normalization": {
                            "kind": normalization_kind,
                            "minimum": minimum,
                            "scale": scale,
                            "seasonal": seasonal_metadata,
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
    if score_root is not None:
        score_root.mkdir(parents=True, exist_ok=True)
    score_normalization = None
    calibration_seconds = 0.0
    if score_mode == "calibrated":
        calibration_start = perf_counter()
        with TemporaryDirectory(prefix=".mc_calibration_", dir=score_root) as scratch:
            calibration_root = score_root / "calibration" if save_raw_mc and score_root else Path(scratch)
            cg, cd, calibration_features, calibration_store = score_components(
                model, holdout_score_data, batch_size=inference_batch_size, device=torch_device,
                n_features=len(feature_names), storage=score_storage, output_dir=calibration_root,
                memory_limit_mb=score_memory_limit_mb, loader_options=score_loader_options,
                mc_dropout_enabled=mc_dropout_enabled, mc_samples=mc_samples,
                save_raw_mc=save_raw_mc, share_history=execution_mode == "optimized", precision=precision)
            try:
                score_normalization = {
                    **fit_calibration_ranges(cg, cd, chunk_size=score_chunk_size),
                    "fit_period": "calibration_only",
                    "fit_axes": ["M", "T", "N"],
                    "calibration_start": str(holdout_score_data.target_timestamps[0]),
                    "calibration_end": str(holdout_score_data.target_timestamps[-1]),
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
    # Complete validation, once, after training: MC-mean reconstruction error of every valid point.
    # It reads the model only and gives the random streams back, so the test scoring is unchanged.
    objective = None
    if holdout_name == "validation":
        objective = validation_objective(
            model, holdout_score_data, batch_size=inference_batch_size, device=torch_device,
            loader_options=score_loader_options, mc_dropout_enabled=mc_dropout_enabled, mc_samples=mc_samples,
            share_history=execution_mode == "optimized", precision=precision, chunk_size=score_chunk_size)
    pca_mmd_metadata = None if mmd_monitor is None else {
        **mmd_monitor.reference.summary(), "objective_window": mmd_objective_window,
        "objective": "mean_of_validation_pca_mmd_mean_over_the_last_objective_window_epochs",
        "validation_pca_mmd_rolling_mean": next(
            (float(past["validation_pca_mmd_rolling_mean"]) for past in reversed(prior_history + history)
             if past.get("validation_pca_mmd_rolling_mean") is not None
             and np.isfinite(past["validation_pca_mmd_rolling_mean"])), None)}
    if skip_final_scoring:
        # Training and validation only (smoke tests, timing of the per-epoch validation): the test
        # is not scored. Epoch checkpoints and the training history are already saved; no score, no
        # score normalization and no final checkpoint are written.
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        if score_root is not None and not any(score_root.iterdir()):
            score_root.rmdir()
        print("[stgan] final scoring skipped: training and validation only", flush=True)
        return STGANResult(
            test_timestamps=test_score_data.target_timestamps, location_names=location_names,
            feature_names=feature_names, test_scores=None, test_feature_scores=None,
            test_generator_scores=None, test_discriminator_scores=None,
            metadata={
                "final_scoring": "skipped", "precision": precision, "splits": splits, "score_mode": score_mode,
                "dataset": dataset_name, "timestep_hours": timestep_hours, "execution_mode": execution_mode,
                "resume": resume_metadata,
                "monitoring": None if monitor is None else monitor.metadata(),
                "validation_objective": objective, "pca_reference": pca_summary, "pca_mmd": pca_mmd_metadata,
                "parameter_counts": model.parameter_counts(), "environment": runtime_environment(),
                "performance": {"precision": precision, "preparation_seconds": preparation_seconds,
                    "training_seconds": training_seconds,
                    "training_samples_per_second": sum(r["samples"] for r in history) / training_seconds,
                    "peak_cuda_memory_bytes": (torch.cuda.max_memory_allocated(torch_device)
                                               if torch_device.type == "cuda" else None)},
                "epochs": epochs, "batch_size": batch_size, **learning_rates, **update_steps,
                "generator_reconstruction_weight": generator_reconstruction_weight,
                "recent_steps": recent_steps, "trend_steps": trend_steps,
                "train_samples_per_epoch": train_samples_per_epoch,
                "score_normalization": None, "score_statistics": None, "test_labels_used": False})
    scoring_start = perf_counter()
    test_generator, test_discriminator, test_features, score_store = score_components(
        model, test_score_data, batch_size=inference_batch_size, device=torch_device,
        n_features=len(feature_names), storage=score_storage, output_dir=score_root,
        memory_limit_mb=score_memory_limit_mb, loader_options=score_loader_options,
        mc_dropout_enabled=mc_dropout_enabled, mc_samples=mc_samples,
        save_raw_mc=save_raw_mc,
        share_history=execution_mode == "optimized", precision=precision)
    try:
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        scoring_seconds = perf_counter() - scoring_start
        performance = {"precision": precision, "preparation_seconds": preparation_seconds,
            **({"calibration_seconds": calibration_seconds} if score_mode == "calibrated" else {}),
            "training_seconds": training_seconds, "scoring_seconds": scoring_seconds,
            "training_samples_per_second": sum(r["samples"] for r in history) / training_seconds,
            "scoring_samples_per_second": len(test_score_data) / scoring_seconds,
            "peak_cuda_memory_bytes": (torch.cuda.max_memory_allocated(torch_device)
                                       if torch_device.type == "cuda" else None)}
        if score_mode == "calibrated":
            (test_scores, anomaly_std, test_generator, test_discriminator,
             _, _) = normalize_mc_scores(
                test_generator, test_discriminator, score_store,
                normalization=score_normalization, chunk_size=score_chunk_size)
        else:
            raw_generator, raw_discriminator = test_generator, test_discriminator
            (test_generator, generator_std, test_discriminator,
             discriminator_std, component_covariance) = summarize_raw_mc_components(
                raw_generator, raw_discriminator, score_store, chunk_size=score_chunk_size)
            test_scores, anomaly_std, score_normalization = normalize_paper_mc_scores(
                test_generator, generator_std, test_discriminator,
                discriminator_std, component_covariance, score_store,
                chunk_size=score_chunk_size, raw_generator=raw_generator,
                raw_discriminator=raw_discriminator)
        score_store.discard_temporary_raw()
        # Descriptive statistics of the unnormalized components and of the final score.
        # They describe the saved maps; none of them is used by the score itself.
        score_statistics = {name: component_statistics(values, chunk_size=score_chunk_size)
                            for name, values in (("reconstruction_raw", test_generator),
                                                 ("discriminator_raw", test_discriminator),
                                                 ("anomaly", test_scores))}
        reference_hyperparameters_used = all(
            (
                not graph_mode,
                epochs == REFERENCE_CONFIG.epochs,
                batch_size == REFERENCE_CONFIG.batch_size,
                lr == REFERENCE_CONFIG.learning_rate,
                discriminator_lr == lr,
                discriminator_generator_update_ratio == REFERENCE_CONFIG.discriminator_generator_update_ratio,
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
                time_encoding == "onehot", not seasonal,
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
                    "format_version": 2, "precision": precision,
                    "model_class": model_class,
                    "window_config": {"recent_steps": recent_steps, "trend_steps": trend_steps},
                    "time_encoding": time_encoding,
                    "timestep_hours": timestep_hours,
                    "splits": splits,
                    "model_state_dict": cpu_state_dict(),
                    "model_config": model_config,
                    "mc_config": {"mc_dropout_enabled": mc_dropout_enabled, "mc_samples": mc_samples},
                    "normalization": {
                        "kind": normalization_kind,
                        "minimum": minimum,
                        "scale": scale,
                        "seasonal": seasonal_metadata,
                    },
                    "score_normalization": score_normalization,
                    "score_statistics": score_statistics,
                    "grid": grid_payload(),
                    "parameter_counts": model.parameter_counts(),
                    "locations": location_names,
                    "features": feature_names,
                    "training": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        **learning_rates,
                        **update_steps,
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
                            "source": f"preceding_{holdout_name}_history" if holdout is not None else "preceding_training_history",
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
                "precision": precision,
                "splits": splits,
                "score_mode": score_mode,
                "dropout_enabled": dropout_enabled,
                "dropout_p": dropout_p,
                "mc_dropout_enabled": mc_dropout_enabled,
                "mc_samples": mc_samples,
                "save_raw_mc": save_raw_mc,
                "effective_mc_samples": mc_samples if mc_dropout_enabled else 1,
                "uncertainty_definition": "population_std_of_anomaly_scores_ddof_0",
                "execution_mode": execution_mode,
                "resume": resume_metadata,
                "monitoring": None if monitor is None else monitor.metadata(),
                "validation_objective": objective,
                "pca_reference": pca_summary,
                "pca_mmd": pca_mmd_metadata,
                "runtime": {"num_workers": num_workers, "persistent_workers": persistent_workers,
                    "train_num_workers": train_workers, "score_num_workers": score_workers,
                    "train_batch_size": batch_size, "score_batch_size": inference_batch_size,
                    "log_interval": progress_interval,
                    "prefetch_factor": prefetch_factor, "pin_memory": pin_memory,
                    "normalized_disk_cache": normalized, "shuffle_mode": shuffle_mode,
                    "batch_unit": "global_timestamps" if graph_mode else "time_location_patches",
                    "trend_chunk_size": trend_chunk_size if graph_mode else None,
                    "discriminator_chunk_size": discriminator_chunk_size if graph_mode else None,
                    "score_storage": score_store.backend, "score_chunk_size": score_chunk_size,
                    "save_raw_mc": save_raw_mc,
                    "raw_storage_policy": "retained" if save_raw_mc else "temporary_until_aggregation",
                    "shuffle_block_size": shuffle_block_size,
                    "shuffle_order_equivalent_to_legacy": not graph_mode and shuffle_mode != "block"},
                "backend": f"stgan_{spatial_encoder}_lstm_{dataset_name}",
                "spatial_encoder": spatial_encoder,
                "graph": graph_metadata,
                "dataset": dataset_name,
                "timestep_hours": timestep_hours,
                "trend_hours": trend_steps * timestep_hours,
                "time_encoding": time_encoding,
                "time_feature_size": model_config["time_feature_size"],
                "source_branch": "feat/stgan-paper",
                "source_commit": "777df6bc6deddeccafbf806bd1c380f79ea146a1",
                "parameter_counts": model.parameter_counts(),
                "performance": performance,
                "alignment_reference": "TNNLS_2022_and_official_dleyan_STGAN",
                "alignment_policy": ("global_sparse_gat_original_trend_lstm_patch_discriminator_losses_and_score"
                                     if graph_mode else ALIGNMENT_POLICY),
                "paper_reference": {
                    "doi": "10.1109/TNNLS.2021.3136171",
                    "repository": "https://github.com/dleyan/STGAN",
                    "repository_commit_verified": "20d2f6b365ea003500a57737647a846fb41267aa",
                },
                "paper_alignment": {
                    "generator_discriminator_architecture": False,
                    "adversarial_losses_and_targets": True,
                    "score_equation": score_mode == "paper",
                    "reference_hyperparameters": reference_hyperparameters_used,
                    "complete_training_product": full_training_product,
                    "domain_adaptations": [
                        ("two_layer_global_sparse_gat_with_pointwise_recent_gru"
                         if graph_mode else "convgru_2d_gates_with_mask_instead_of_graph_convolutional_gates"),
                        "pointwise_1x1_projections_instead_of_remaining_graph_convolutions",
                        (f"chronological_train_{holdout_name}_test_with_past_only_context" if holdout is not None
                         else "chronological_train_test_with_past_only_context"),
                        ("shared_score_ranges_fitted_on_calibration_only_without_clipping" if score_mode == "calibrated"
                         else "global_test_mc_mean_component_minmax_then_sum"),
                        "target_feature_residuals_for_diagnostics",
                    ] + (["generator_spatial_temporal_fusion_dropout"] if dropout_enabled else [])
                      + (["cyclic_local_solar_time_and_annual_phase_instead_of_weekday_hour_onehot"]
                         if time_encoding == "cyclic" else [])
                      + (["inputs_standardised_per_location_day_of_year_and_time_of_day"] if seasonal else [])
                      + (["mc_score_mean_and_population_std"] if mc_dropout_enabled else []),
                },
                "normalization": ("training_only_seasonal_standardisation_then_feature_minmax"
                                  if seasonal else "training_only_feature_minmax"),
                "seasonal_normalization": seasonal_metadata,
                "score_normalization": score_normalization,
                "score_statistics": score_statistics,
                "test_labels_used": False,
                "grid": grid.metadata,
                "reconstruction_reduction": "mean_valid_cells_and_features_per_sample_then_mean_samples",
                "test_context": {
                    "source": f"preceding_{holdout_name}_history" if holdout is not None else "preceding_training_history",
                    "steps": trend_steps,
                    "targets": "test_timestamps_only",
                },
                "epochs": epochs,
                "batch_size": batch_size,
                **learning_rates,
                **update_steps,
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
                "score_definition": (("mc_mean_of_calibration_normalized_generator_plus_discriminator_gap"
                                      if mc_dropout_enabled else "calibration_normalized_generator_plus_discriminator_gap")
                                     if score_mode == "calibrated" else
                                     "global_test_normalized_generator_plus_discriminator_gap_lambda_1"),
                "component_covariance_file": ("scores/component_covariance.npy" if score_mode == "paper" else None),
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
