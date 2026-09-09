"""CNN spatial ablation; retain the temporal LSTM and adversarial protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class STGANCNNConfig:
    epochs: int = 6
    batch_size: int = 256
    learning_rate: float = 1e-3
    generator_reconstruction_weight: float = 500.0
    hidden_size: int = 64
    n_layers: int = 2
    cnn_channels: int = 32
    cnn_layers: int = 2
    patch_size: int = 3
    recent_steps: int = 1
    trend_steps: int = 7 * 24
    score_stride: int = 1
    # Zero means the complete shuffled time-location Cartesian product, as in
    # the paper repository. Positive values enable an explicit PVGIS scaling
    # adaptation through replacement sampling.
    train_samples_per_epoch: int = 0
    grid_crs: str = "EPSG:32632"
    grid_spacing: float = 5000.0
    grid_tolerance: float = 25.0

    def __post_init__(self):
        import math
        for name in ("epochs", "batch_size", "hidden_size", "n_layers", "cnn_channels",
                     "cnn_layers", "trend_steps", "score_stride"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.patch_size not in (1, 3, 5):
            raise ValueError("patch_size must be 1, 3 or 5.")
        if self.recent_steps != 1:
            raise ValueError("This spatial CNN ablation requires recent_steps=1; history uses the LSTM.")
        if self.train_samples_per_epoch < 0:
            raise ValueError("train_samples_per_epoch must be non-negative.")
        for name in ("learning_rate", "generator_reconstruction_weight", "grid_spacing", "grid_tolerance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")

    def to_dict(self) -> dict:
        return asdict(self)


REFERENCE_CONFIG = STGANCNNConfig()
REFERENCE_SEED = 20
ALIGNMENT_POLICY = "cnn_spatial_ablation_with_original_lstm_losses_and_score"
