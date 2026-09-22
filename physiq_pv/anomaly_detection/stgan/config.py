"""Grid ConvGRU adaptation; retain the trend LSTM and adversarial protocol."""

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
    cnn_channels: int = 32  # Hidden channels per ConvGRU layer.
    cnn_layers: int = 2  # Stacked ConvGRU layers; n_layers controls the trend LSTM.
    patch_size: int = 3
    kernel_size: int = 3  # ConvGRU gates in both G and D; independent of patch_size.
    recent_steps: int = 1  # Preserve the reference repository's hourly adaptation.
    trend_steps: int = 7 * 24
    score_stride: int = 1
    # Zero means the complete shuffled time-location Cartesian product, as in
    # the paper repository. Positive values enable an explicit PVGIS scaling
    # adaptation through replacement sampling.
    train_samples_per_epoch: int = 0
    grid_crs: str = "EPSG:32632"
    grid_spacing: float = 5000.0
    grid_tolerance: float = 25.0
    # Runtime controls: no change to the scientific hyperparameters above.
    num_workers: int = 0
    train_num_workers: int | None = None  # None inherits the compatible common setting.
    score_num_workers: int | None = None
    score_batch_size: int | None = None  # None preserves the previous inference batch.
    log_interval: int | None = None  # None: about 20 logs/epoch, at least 100 batches apart.
    persistent_workers: bool = True
    prefetch_factor: int = 2
    pin_memory: bool = True
    cache_normalized: bool = True
    shuffle_mode: str = "global"  # block is explicit: same samples, different order.
    shuffle_block_size: int = 262144
    execution_mode: str = "optimized"  # legacy is an equivalence/debug path.
    score_storage: str = "auto"
    score_memory_limit_mb: int = 1024
    score_chunk_size: int = 65536
    dropout_enabled: bool = True
    dropout_p: float = 0.2
    mc_dropout_enabled: bool = True
    mc_samples: int = 20
    save_raw_mc: bool = False  # Keep raw (M,T,N) components after aggregation.

    def __post_init__(self):
        import math
        for name in ("dropout_enabled", "mc_dropout_enabled", "save_raw_mc"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean.")
        if not math.isfinite(self.dropout_p) or not 0 <= self.dropout_p < 1:
            raise ValueError("dropout_p must be finite and in [0, 1).")
        if type(self.mc_samples) is not int or self.mc_samples < 1:
            raise ValueError("mc_samples must be a positive integer.")
        for name in ("epochs", "batch_size", "hidden_size", "n_layers", "cnn_channels",
                     "cnn_layers", "recent_steps", "trend_steps", "score_stride"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.patch_size not in (1, 3, 5):
            raise ValueError("patch_size must be 1, 3 or 5.")
        if type(self.kernel_size) is not int or self.kernel_size not in (1, 3, 5):
            raise ValueError("kernel_size must be an integer: 1, 3 or 5.")
        if self.trend_steps < self.recent_steps:
            raise ValueError("Require trend_steps >= recent_steps >= 1.")
        if self.train_samples_per_epoch < 0:
            raise ValueError("train_samples_per_epoch must be non-negative.")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer.")
        for name in ("train_num_workers", "score_num_workers"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be None or a non-negative integer.")
        for name in ("score_batch_size", "log_interval"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be None or a positive integer.")
        if type(self.prefetch_factor) is not int or self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be a positive integer.")
        if self.shuffle_mode not in ("legacy", "global", "block"):
            raise ValueError("shuffle_mode must be legacy, global or block.")
        if type(self.shuffle_block_size) is not int or self.shuffle_block_size < 1:
            raise ValueError("shuffle_block_size must be a positive integer.")
        if self.execution_mode not in ("legacy", "optimized"):
            raise ValueError("execution_mode must be legacy or optimized.")
        if self.score_storage not in ("auto", "memory", "memmap"):
            raise ValueError("score_storage must be auto, memory or memmap.")
        for name in ("score_memory_limit_mb", "score_chunk_size"):
            if type(getattr(self,name)) is not int or getattr(self,name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("learning_rate", "generator_reconstruction_weight", "grid_spacing", "grid_tolerance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def train_batch_size(self) -> int:
        """Explicit name for the existing training setting; checkpoint key stays compatible."""
        return self.batch_size


REFERENCE_CONFIG = STGANCNNConfig()
REFERENCE_SEED = 20
ALIGNMENT_POLICY = "convgru_grid_with_original_trend_lstm_losses_and_score"
