"""Configuration for the PVGIS adaptation of paper STGAN.

The architecture and adversarial losses follow Deng et al., *Graph
Convolutional Adversarial Networks for Spatiotemporal Anomaly Detection*.
PVGIS-specific sampling is explicit because fourteen years times 1,149
locations is much larger than either benchmark used in the paper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class STGANReferenceConfig:
    epochs: int = 6
    batch_size: int = 256
    learning_rate: float = 1e-3
    generator_reconstruction_weight: float = 500.0
    hidden_size: int = 64
    n_layers: int = 2
    subgraph_size: int = 9
    recent_steps: int = 1
    trend_steps: int = 7 * 24
    score_stride: int = 1
    # Zero means the complete shuffled time-location Cartesian product, as in
    # the paper repository. Positive values enable an explicit PVGIS scaling
    # adaptation through replacement sampling.
    train_samples_per_epoch: int = 0
    def to_dict(self) -> dict:
        return asdict(self)


REFERENCE_CONFIG = STGANReferenceConfig()
REFERENCE_SEED = 20
ALIGNMENT_POLICY = "paper_and_official_repo_with_explicit_pvgis_scaling"
