"""Grid ConvGRU adaptation; retain the trend LSTM and adversarial protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re


def parse_update_ratio(value) -> tuple[int, int]:
    """Optimizer steps per batch from "D:G": "2:1" is two D steps, then one G step."""
    match = re.fullmatch(r"([1-9][0-9]*):([1-9][0-9]*)", value) if type(value) is str else None
    if match is None:
        raise ValueError(
            "discriminator_generator_update_ratio must be a string \"D:G\" of positive integers, "
            f"such as '1:1', '2:1' or '1:2' (quote it in YAML); got {value!r}.")
    steps = int(match[1]), int(match[2])
    if math.gcd(*steps) != 1:
        raise ValueError(f"discriminator_generator_update_ratio must be in lowest terms; got {value!r}.")
    return steps


@dataclass(frozen=True)
class STGANCNNConfig:
    epochs: int = 6
    batch_size: int = 256
    learning_rate: float = 1e-3  # Generator LR, unless generator_learning_rate is given.
    discriminator_lr_ratio: float = 1.0  # lr_D / lr_G, unless discriminator_learning_rate is given.
    # Independent Adam rates. None leaves the two fields above in charge, so earlier
    # configurations keep their meaning; giving a rate in both forms is an error.
    generator_learning_rate: float | None = None
    discriminator_learning_rate: float | None = None
    # Optimizer steps per batch, "D:G": "2:1" updates D twice then G once, "1:2" updates
    # D once then G twice. A count of updates, unrelated to discriminator_lr_ratio.
    discriminator_generator_update_ratio: str = "1:1"
    # ERA5: train through 2003 and keep 2004 out of training as the validation year. It is read
    # only by the per-epoch monitoring below and by the validation objective: no loss, no score
    # range, no threshold and no result on the test uses it.
    validation_holdout: bool = False
    monitoring_timestamps: int = 32  # Validation timestamps (every location) checked each epoch; 0 = off.
    monitoring_feature_mmd_every_n_epochs: int = 1  # 0 disables the MMD.
    monitoring_feature_mmd_samples: int = 1024  # Feature vectors per set in the MMD.
    # Epochs averaged in validation/pca_mmd_rolling_mean, the MMD in the fixed PCA space (stgan/mmd.py).
    mmd_objective_window: int = 5
    generator_reconstruction_weight: float = 500.0
    hidden_size: int = 64
    n_layers: int = 2
    cnn_channels: int = 32  # Hidden channels per ConvGRU layer.
    cnn_layers: int = 2  # Stacked ConvGRU layers; n_layers controls the trend LSTM.
    patch_size: int = 3
    kernel_size: int = 3  # ConvGRU gates in both G and D; independent of patch_size.
    recent_steps: int = 1  # One preceding sample; its duration follows the dataset cadence.
    trend_steps: int = 7 * 24
    # onehot: paper weekday+hour. cyclic: sine/cosine of per-location local solar
    # time and of the position in the year (no weekday).
    time_encoding: str = "onehot"
    # minmax: train-only feature min-max. seasonal: first standardise per location,
    # day of year (+- seasonal_window_days) and time of day, then the same min-max.
    normalization: str = "minmax"
    seasonal_window_days: int = 15
    score_stride: int = 1
    # Zero means the complete shuffled time-location Cartesian product, as in
    # the paper repository. Positive values enable an explicit PVGIS scaling
    # adaptation through replacement sampling.
    train_samples_per_epoch: int = 0
    grid_crs: str = "EPSG:32632"
    grid_spacing: float = 5000.0
    grid_tolerance: float = 25.0
    # Runtime controls: no change to the scientific hyperparameters above.
    precision: str = "fp32"  # bf16 uses native CUDA autocast; weights stay FP32.
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
    spatial_encoder: str = "convgru"
    gat_hidden_dim: int = 16  # Per head in the first (concatenating) layer.
    gat_heads: int = 4
    gat_layers: int = 2
    discriminator_chunk_size: int = 256
    trend_chunk_size: int = 256  # Same node-wise LSTM, bounded activation memory.

    def __post_init__(self):
        parse_update_ratio(self.discriminator_generator_update_ratio)
        if self.precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16.")
        if self.spatial_encoder not in ("convgru", "gat"):
            raise ValueError("spatial_encoder must be convgru or gat.")
        if self.time_encoding not in ("onehot", "cyclic"):
            raise ValueError("time_encoding must be onehot or cyclic.")
        if self.normalization not in ("minmax", "seasonal"):
            raise ValueError("normalization must be minmax or seasonal.")
        if type(self.seasonal_window_days) is not int or not 0 <= self.seasonal_window_days <= 183:
            raise ValueError("seasonal_window_days must be an integer between 0 and 183.")
        for name in ("gat_hidden_dim", "gat_heads", "discriminator_chunk_size", "trend_chunk_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.gat_layers) is not int or self.gat_layers != 2:
            raise ValueError("This experiment requires gat_layers=2.")
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
        for name in ("learning_rate", "discriminator_lr_ratio", "generator_reconstruction_weight", "grid_spacing", "grid_tolerance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("generator_learning_rate", "discriminator_learning_rate"):
            value = getattr(self, name)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be None or finite and positive.")
        if self.generator_learning_rate is not None and self.learning_rate not in (
                type(self).learning_rate, self.generator_learning_rate):
            raise ValueError("Give the generator rate once: learning_rate or generator_learning_rate.")
        if (self.discriminator_learning_rate is not None
                and self.discriminator_lr_ratio != type(self).discriminator_lr_ratio):
            raise ValueError("Give the discriminator rate once: discriminator_learning_rate or discriminator_lr_ratio.")
        if not math.isfinite(self.effective_discriminator_learning_rate) or self.effective_discriminator_learning_rate <= 0:
            raise ValueError("Effective discriminator learning rate must be finite and positive.")
        if type(self.validation_holdout) is not bool:
            raise ValueError("validation_holdout must be a boolean.")
        for name, minimum in (("monitoring_timestamps", 0), ("monitoring_feature_mmd_every_n_epochs", 0),
                              ("monitoring_feature_mmd_samples", 2), ("mmd_objective_window", 1)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def effective_generator_learning_rate(self) -> float:
        return self.learning_rate if self.generator_learning_rate is None else self.generator_learning_rate

    @property
    def effective_discriminator_learning_rate(self) -> float:
        if self.discriminator_learning_rate is not None:
            return self.discriminator_learning_rate
        return self.effective_generator_learning_rate * self.discriminator_lr_ratio

    @property
    def discriminator_updates_per_batch(self) -> int:
        return parse_update_ratio(self.discriminator_generator_update_ratio)[0]

    @property
    def generator_updates_per_batch(self) -> int:
        return parse_update_ratio(self.discriminator_generator_update_ratio)[1]

    @property
    def train_batch_size(self) -> int:
        """Explicit name for the existing training setting; checkpoint key stays compatible."""
        return self.batch_size


@dataclass(frozen=True)
class STGANGATConfig(STGANCNNConfig):
    spatial_encoder: str = "gat"
    batch_size: int = 1  # Full graph timestamps, not independent location patches.
    score_batch_size: int | None = 1
    # Centers that D processes together: memory and speed only, the same patches, losses and
    # optimizer steps for any value. 10611 = 81 x 131: the whole ERA5 grid of a timestamp at once.
    discriminator_chunk_size: int = 10611


REFERENCE_CONFIG = STGANCNNConfig()
REFERENCE_SEED = 20
ALIGNMENT_POLICY = "convgru_grid_with_original_trend_lstm_losses_and_score"
