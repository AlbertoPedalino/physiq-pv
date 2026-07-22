"""Reference configuration used for the MTGFlow base implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class MTGFlowReferenceConfig:
    """Hyperparameters reported by the paper and its official implementation."""

    epochs: int = 40
    window_size: int = 60
    train_stride: int = 10
    score_stride: int = 10
    batch_size: int = 256
    learning_rate: float = 2e-3
    weight_decay: float = 5e-4
    n_blocks: int = 2
    hidden_size: int = 32
    n_hidden: int = 1
    attention_dropout: float = 0.2
    iqr_k: float = 1.5
    entity_threshold_scale: float = 0.8

    def to_dict(self) -> dict:
        return asdict(self)


REFERENCE_CONFIG = MTGFlowReferenceConfig()
REFERENCE_SEEDS = (15, 16, 17, 18, 19)


def reference_protocol_deviations(
    settings: Mapping[str, object],
    seeds: tuple[int, ...],
    preparation_verified: bool,
) -> list[str]:
    """Describe departures from reference values present in ``settings``."""
    reference = REFERENCE_CONFIG.to_dict()
    aliases = {"lr": "learning_rate"}
    deviations = []
    for name, actual in settings.items():
        reference_name = aliases.get(name, name)
        if reference_name in reference and actual != reference[reference_name]:
            deviations.append(
                f"{name}={actual!r} (reference {reference[reference_name]!r})"
            )
    if tuple(seeds) != REFERENCE_SEEDS:
        deviations.append(
            f"seeds={list(seeds)!r} (reference suite {list(REFERENCE_SEEDS)!r})"
        )
    if not preparation_verified:
        deviations.append("preparation metadata not verified")
    return deviations
