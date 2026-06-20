"""Gaussian input-noise injection for the PVGIS run.

Used to build the SDE-Net pseudo-OOD batch (training inputs + Gaussian noise) on
which the diffusion net g is pushed high.
"""
from __future__ import annotations

from typing import List, Optional

import torch


# The SDE-Net pseudo-OOD construction is x + epsilon over the complete input.
# Keep this public constant empty so helper callers retain that paper behaviour.
NOISE_EXCLUDED_FEATURES: tuple[str, ...] = ()


def build_noise_feature_indices(feature_names: List[str]) -> List[int]:
    """Channel indices eligible for input-noise injection.

    The paper perturbs every channel. The order matches the dataset's channel
    order (resolve_feature_set order), so index i corresponds to feature_names[i].
    """
    return [
        i for i, name in enumerate(feature_names)
        if name not in NOISE_EXCLUDED_FEATURES
    ]


def inject_input_noise(
    x: torch.Tensor,
    noise_idx: Optional[torch.Tensor],
    noise_std: float,
    noise_prob: float,
) -> torch.Tensor:
    """Add Gaussian noise to selected feature channels of a TRAINING batch only.

    x is (B, N, seq_len, C). For each sample (first dim) a Bernoulli(noise_prob)
    gate decides whether that sample is perturbed; perturbed samples get
    N(0, noise_std) added to the channels in `noise_idx` (excluded channels and
    the target y are untouched). No-op when noise is disabled or no sample is
    gated. The caller must NEVER apply this in eval/validation/inference.
    """
    if (
        noise_idx is None
        or noise_idx.numel() == 0
        or noise_std <= 0.0
        or noise_prob <= 0.0
    ):
        return x
    B = x.shape[0]
    gate = torch.rand(B, device=x.device) < noise_prob  # (B,)
    if not bool(gate.any()):
        return x
    x = x.clone()
    sub = x.index_select(-1, noise_idx)                    # (B, N, seq, len(idx))
    noise = torch.randn_like(sub) * float(noise_std)
    noise = noise * gate.view(B, 1, 1, 1).to(noise.dtype)  # ungated samples -> 0
    x.index_copy_(-1, noise_idx, sub + noise)
    return x

