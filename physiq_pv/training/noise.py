"""Train-only input-noise injection (random + anomaly-aware) for the PVGIS run."""
from __future__ import annotations

from typing import List, Optional

import torch


# Feature channels that must NOT receive input noise. sin_elev/cos_elev are a
# cyclical (sin, cos) encoding of solar elevation: perturbing them independently
# breaks the sin²+cos²≈1 relation and shifts the encoded angle in an
# uncontrolled way, so they are excluded from noise injection by default.
NOISE_EXCLUDED_FEATURES = ("sin_elev", "cos_elev")


def build_noise_feature_indices(feature_names: List[str]) -> List[int]:
    """Channel indices eligible for input-noise injection.

    Excludes the cyclical sin_elev/cos_elev encoding (see NOISE_EXCLUDED_FEATURES);
    every other (continuous, normalised/bounded) channel is perturbable. The order
    matches the dataset's channel order (resolve_feature_set order), so index i
    corresponds to feature_names[i].
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


def inject_input_noise_anomaly(
    x: torch.Tensor,
    noise_idx: Optional[torch.Tensor],
    anomaly_mask: torch.Tensor,
    normal_std: float,
    normal_prob: float,
    anomaly_std: float,
    anomaly_prob: float,
) -> torch.Tensor:
    """Anomaly-aware input noise: per-(sample, node) std/prob from `anomaly_mask`.

    x is (B, N, seq_len, C); anomaly_mask is a (B, N) bool tensor (True =
    rare_or_extreme). Anomalous cells use (anomaly_std, anomaly_prob); the rest
    use (normal_std, normal_prob) — so setting the normal pair to 0 perturbs ONLY
    anomalies, while a larger anomaly pair perturbs them more strongly. The
    Bernoulli gate is per (sample, node); the perturbed channels are `noise_idx`
    (sin_elev/cos_elev excluded). Train-only; targets untouched.

    ATTENTION: this couples training to anomaly labels — it is NOT an
    eval-only-labels configuration.
    """
    if noise_idx is None or noise_idx.numel() == 0:
        return x
    B, N = x.shape[0], x.shape[1]
    mask = anomaly_mask.to(device=x.device, dtype=torch.bool)
    std_map = torch.where(
        mask,
        torch.as_tensor(float(anomaly_std), device=x.device),
        torch.as_tensor(float(normal_std), device=x.device),
    )  # (B, N)
    prob_map = torch.where(
        mask,
        torch.as_tensor(float(anomaly_prob), device=x.device),
        torch.as_tensor(float(normal_prob), device=x.device),
    )  # (B, N)
    gate = torch.rand(B, N, device=x.device) < prob_map
    if not bool(gate.any()):
        return x
    x = x.clone()
    sub = x.index_select(-1, noise_idx)                       # (B, N, seq, len(idx))
    noise = torch.randn_like(sub) * std_map[:, :, None, None]
    noise = noise * gate[:, :, None, None].to(noise.dtype)    # ungated cells -> 0
    x.index_copy_(-1, noise_idx, sub + noise)
    return x


