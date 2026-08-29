"""Leakage-safe training, checkpointing, and scoring for geographical STGAN."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..common import runtime_environment, seed_everything
from .config import ALIGNMENT_POLICY, REFERENCE_CONFIG, REFERENCE_SEED
from .data import STGANWindowDataset, prepend_training_context_to_test
from .graph import build_geographical_subgraphs
from .model import STGAN
from .result import STGANResult


def _feature_minmax(data: np.ndarray, chunk_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.full(data.shape[2], np.inf, dtype=np.float64)
    maximum = np.full(data.shape[2], -np.inf, dtype=np.float64)
    for start in range(0, data.shape[0], chunk_size):
        block = np.asarray(data[start : start + chunk_size], dtype=np.float64)
        minimum = np.minimum(minimum, np.min(block, axis=(0, 1)))
        maximum = np.maximum(maximum, np.max(block, axis=(0, 1)))
    scale = maximum - minimum
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    if not np.isfinite(minimum).all():
        raise ValueError("STGAN training data contain non-finite values.")
    return minimum.astype(np.float32), scale.astype(np.float32)


def _component_range(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("STGAN anomaly component has no finite values.")
    minimum = float(np.min(finite))
    scale = float(np.max(finite) - minimum)
    return minimum, scale if scale >= 1e-8 else 1.0


def _normalise_component(values: np.ndarray, parameters: tuple[float, float]) -> np.ndarray:
    minimum, scale = parameters
    return (values - minimum) / scale


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
    if payload.get("format_version") != 1 or payload.get("model_class") != "STGAN":
        raise ValueError(f"Unsupported STGAN checkpoint: {resolved}")
    model = STGAN(**payload["model_config"]).to(torch_device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
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
    epochs: int = REFERENCE_CONFIG.epochs,
    batch_size: int = REFERENCE_CONFIG.batch_size,
    lr: float = REFERENCE_CONFIG.learning_rate,
    generator_reconstruction_weight: float = REFERENCE_CONFIG.generator_reconstruction_weight,
    hidden_size: int = REFERENCE_CONFIG.hidden_size,
    n_layers: int = REFERENCE_CONFIG.n_layers,
    subgraph_size: int = REFERENCE_CONFIG.subgraph_size,
    recent_steps: int = REFERENCE_CONFIG.recent_steps,
    trend_steps: int = REFERENCE_CONFIG.trend_steps,
    score_stride: int = REFERENCE_CONFIG.score_stride,
    train_samples_per_epoch: int | None = REFERENCE_CONFIG.train_samples_per_epoch,
    device: str = "cuda",
    seed: int = REFERENCE_SEED,
    checkpoint_path: str | Path | None = None,
) -> STGANResult:
    """Fit STGAN without anomaly labels and return location/feature scores."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, RandomSampler

    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if train.ndim != 3 or test.ndim != 3:
        raise ValueError("STGAN train/test arrays must be [time,location,feature].")
    expected_tail = (len(location_names), len(feature_names))
    if tuple(train.shape[1:]) != expected_tail or tuple(test.shape[1:]) != expected_tail:
        raise ValueError(f"STGAN arrays must end in {expected_tail}.")
    seed_everything(seed)
    graph = build_geographical_subgraphs(
        latitudes,
        longitudes,
        subgraph_size=subgraph_size,
    )
    minimum, scale = _feature_minmax(train)
    test_with_context, test_timestamps_with_context = prepend_training_context_to_test(
        train,
        test,
        train_timestamps=train_timestamps,
        test_timestamps=test_timestamps,
        context_steps=trend_steps,
    )
    train_fit = STGANWindowDataset(
        train,
        train_timestamps,
        graph,
        feature_minimum=minimum,
        feature_scale=scale,
        recent_steps=recent_steps,
        trend_steps=trend_steps,
        stride=1,
    )
    test_score_data = STGANWindowDataset(
        test_with_context,
        test_timestamps_with_context,
        graph,
        feature_minimum=minimum,
        feature_scale=scale,
        recent_steps=recent_steps,
        trend_steps=trend_steps,
        stride=score_stride,
    )
    if min(len(train_fit), len(test_score_data)) == 0:
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
        "subgraph_size": graph.subgraph_size,
        "time_feature_size": 31,
    }
    model = STGAN(**model_config).to(torch_device)
    generator_optimizer = torch.optim.Adam(model.generator.parameters(), lr=lr)
    discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=lr)
    binary_loss = nn.BCELoss()
    reconstruction_loss = nn.MSELoss()
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
    if full_training_product:
        sampler = None
        shuffle = True
    else:
        sampler = RandomSampler(
            train_fit,
            replacement=True,
            num_samples=int(train_samples_per_epoch),
            generator=generator,
        )
        shuffle = False
    train_loader = DataLoader(
        train_fit,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=False,
    )

    batches_per_epoch = len(train_loader)
    progress_interval = max(1, batches_per_epoch // 20)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_index, (
            recent,
            trend,
            adjacency,
            time_features,
            observed,
            _,
            _,
        ) in enumerate(train_loader, start=1):
            recent = recent.to(torch_device)
            trend = trend.to(torch_device)
            adjacency = adjacency.to(torch_device)
            time_features = time_features.to(torch_device)
            observed = observed.to(torch_device)
            # Paper/repository convention: discriminator output is anomaly
            # probability, hence real/normal=0 and generated/fake=1.
            normal = torch.zeros((recent.shape[0], 1), device=torch_device)
            generated_target = torch.ones_like(normal)

            discriminator_optimizer.zero_grad()
            with torch.no_grad():
                generated = model.generator(recent, trend, adjacency, time_features)
            real_sequence = torch.cat((recent, observed[:, None]), dim=1)
            fake_sequence = torch.cat((recent, generated[:, None]), dim=1)
            discriminator_total = 0.5 * (
                binary_loss(model.discriminator(real_sequence, adjacency), normal)
                + binary_loss(
                    model.discriminator(fake_sequence, adjacency), generated_target
                )
            )
            discriminator_total.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad()
            for parameter in model.discriminator.parameters():
                parameter.requires_grad_(False)
            generated = model.generator(recent, trend, adjacency, time_features)
            fake_sequence = torch.cat((recent, generated[:, None]), dim=1)
            generator_total = (
                generator_reconstruction_weight
                * reconstruction_loss(generated, observed)
                + binary_loss(model.discriminator(fake_sequence, adjacency), normal)
            )
            generator_total.backward()
            generator_optimizer.step()
            for parameter in model.discriminator.parameters():
                parameter.requires_grad_(True)
            if batch_index % progress_interval == 0 or batch_index == batches_per_epoch:
                print(
                    f"[stgan] epoch={epoch}/{epochs} "
                    f"batch={batch_index}/{batches_per_epoch} "
                    f"D={discriminator_total.item():.6f} G={generator_total.item():.6f}",
                    flush=True,
                )
        if checkpoint_resolved is not None:
            epoch_path = checkpoint_resolved.with_name(
                f"{checkpoint_resolved.stem}_epoch_{epoch}{checkpoint_resolved.suffix}"
            )
            torch.save(
                {
                    "format_version": 1,
                    "model_class": "STGAN",
                    "model_state_dict": cpu_state_dict(),
                    "model_config": model_config,
                    "completed_epochs": epoch,
                    "normalization": {
                        "kind": "training_only_feature_minmax_to_minus_one_one",
                        "minimum": minimum,
                        "scale": scale,
                    },
                    "graph": {
                        "kind": graph.topology,
                        "node_indices": graph.node_indices,
                        "normalized_adjacency": graph.normalized_adjacency,
                        "sigma_km": graph.sigma_km,
                    },
                    "seed": seed,
                },
                epoch_path,
            )

    def score(dataset: STGANWindowDataset, *, include_features: bool):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        shape = (len(dataset.targets), len(location_names))
        generator_scores = np.empty(shape, dtype=np.float32)
        discriminator_scores = np.empty(shape, dtype=np.float32)
        feature_scores = (
            np.empty(shape + (len(feature_names),), dtype=np.float32)
            if include_features
            else None
        )
        model.eval()
        with torch.no_grad():
            for recent, trend, adjacency, time_features, observed, time_pos, loc in loader:
                recent = recent.to(torch_device)
                trend = trend.to(torch_device)
                adjacency = adjacency.to(torch_device)
                time_features = time_features.to(torch_device)
                observed = observed.to(torch_device)
                _, real_score, fake_score, squared_error = model.components(
                    recent, trend, adjacency, time_features, observed
                )
                time_np = time_pos.numpy()
                loc_np = loc.numpy()
                generator_scores[time_np, loc_np] = (
                    squared_error.mean(dim=(1, 2)).cpu().numpy()
                )
                discriminator_scores[time_np, loc_np] = (
                    (real_score - fake_score).squeeze(-1).cpu().numpy()
                )
                if feature_scores is not None:
                    # node_indices[:, 0] is always the target location itself.
                    feature_scores[time_np, loc_np, :] = (
                        squared_error[:, 0, :].cpu().numpy()
                    )
        return generator_scores, discriminator_scores, feature_scores

    test_generator, test_discriminator, test_features = score(
        test_score_data, include_features=True
    )
    assert test_features is not None
    # Section VII-D normalizes the two terms before Eq. (10). The public
    # repository exports only the raw test components, so the minimal faithful
    # interpretation is one global min-max transform per component on the
    # complete test time-location product, followed by lambda=1.
    generator_range = _component_range(test_generator)
    discriminator_range = _component_range(test_discriminator)
    test_scores = _normalise_component(
        test_generator, generator_range
    ) + _normalise_component(test_discriminator, discriminator_range)
    reference_hyperparameters_used = all(
        (
            epochs == REFERENCE_CONFIG.epochs,
            batch_size == REFERENCE_CONFIG.batch_size,
            lr == REFERENCE_CONFIG.learning_rate,
            generator_reconstruction_weight
            == REFERENCE_CONFIG.generator_reconstruction_weight,
            hidden_size == REFERENCE_CONFIG.hidden_size,
            n_layers == REFERENCE_CONFIG.n_layers,
            subgraph_size == REFERENCE_CONFIG.subgraph_size,
            recent_steps == REFERENCE_CONFIG.recent_steps,
            trend_steps == REFERENCE_CONFIG.trend_steps,
            score_stride == REFERENCE_CONFIG.score_stride,
            seed == REFERENCE_SEED,
            full_training_product,
        )
    )

    if checkpoint_resolved is not None:
        torch.save(
            {
                "format_version": 1,
                "model_class": "STGAN",
                "model_state_dict": cpu_state_dict(),
                "model_config": model_config,
                "normalization": {
                    "kind": "training_only_feature_minmax_to_minus_one_one",
                    "minimum": minimum,
                    "scale": scale,
                },
                "score_normalization": {
                    "fit_period": "complete_test_time_location_product",
                    "generator": generator_range,
                    "discriminator": discriminator_range,
                    "component_weight": 1.0,
                },
                "graph": {
                    "kind": graph.topology,
                    "node_indices": graph.node_indices,
                    "normalized_adjacency": graph.normalized_adjacency,
                    "sigma_km": graph.sigma_km,
                    "latitudes": np.asarray(latitudes),
                    "longitudes": np.asarray(longitudes),
                },
                "locations": location_names,
                "features": feature_names,
                "training": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "generator_reconstruction_weight": generator_reconstruction_weight,
                    "train_samples_per_epoch": train_samples_per_epoch,
                    "sampling": (
                        "complete_shuffled_time_location_product"
                        if full_training_product
                        else "replacement_sampled_pvgis_adaptation"
                    ),
                    "discriminator_targets": {
                        "real_normal": 0,
                        "generated_fake": 1,
                    },
                    "seed": seed,
                    "test_labels_used": False,
                    "test_context": {
                        "source": "training_tail_only",
                        "steps": trend_steps,
                        "targets": "test_timestamps_only",
                    },
                },
                "environment": runtime_environment(),
            },
            checkpoint_resolved,
        )

    return STGANResult(
        test_timestamps=test_score_data.target_timestamps,
        location_names=location_names,
        feature_names=feature_names,
        test_scores=test_scores.astype(np.float32),
        test_feature_scores=test_features,
        test_generator_scores=test_generator,
        test_discriminator_scores=test_discriminator,
        metadata={
            "backend": "stgan_paper_pvgis",
            "alignment_reference": "TNNLS_2022_and_official_dleyan_STGAN",
            "alignment_policy": ALIGNMENT_POLICY,
            "paper_reference": {
                "doi": "10.1109/TNNLS.2021.3136171",
                "repository": "https://github.com/dleyan/STGAN",
                "repository_commit_verified": "20d2f6b365ea003500a57737647a846fb41267aa",
            },
            "paper_alignment": {
                "generator_discriminator_architecture": True,
                "adversarial_losses_and_targets": True,
                "score_equation": True,
                "reference_hyperparameters": reference_hyperparameters_used,
                "complete_training_product": full_training_product,
                "pvgis_domain_adaptations": [
                    "directed_geographical_knn_instead_of_physical_traffic_network_edges",
                    "historical_2005_2018_to_test_2019_split",
                    "target_feature_residuals_for_diagnostics",
                ],
            },
            "normalization": "training_only_feature_minmax",
            "score_normalization": "global_test_component_minmax",
            "test_labels_used": False,
            "graph": {
                "subgraph_size": graph.subgraph_size,
                "sigma_km": graph.sigma_km,
                "topology": graph.topology,
                "directed_edge_count": graph.directed_edge_count,
                "dense_runtime_adjacency": False,
            },
            "test_context": {
                "source": "training_tail_only",
                "steps": trend_steps,
                "targets": "test_timestamps_only",
            },
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "generator_reconstruction_weight": generator_reconstruction_weight,
            "hidden_size": hidden_size,
            "n_layers": n_layers,
            "recent_steps": recent_steps,
            "trend_steps": trend_steps,
            "score_stride": score_stride,
            "train_samples_per_epoch": train_samples_per_epoch,
            "training_sampling": (
                "complete_shuffled_time_location_product"
                if full_training_product
                else "replacement_sampled_pvgis_adaptation"
            ),
            "discriminator_output_semantics": "anomaly_probability_real_0_fake_1",
            "score_definition": "test_normalized_generator_plus_discriminator_gap",
            "feature_score_definition": "target_node_squared_prediction_error",
            "checkpoint": None if checkpoint_resolved is None else str(checkpoint_resolved),
            "device": str(torch_device),
            "seed": seed,
            "environment": runtime_environment(),
        },
    )
