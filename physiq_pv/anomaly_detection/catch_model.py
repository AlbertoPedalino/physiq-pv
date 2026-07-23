"""Neural building blocks for the CATCH multivariate anomaly detector.

The implementation follows Wu et al. (ICLR 2025): an input window is
normalised per instance, transformed with an FFT, split into frequency
patches, fused across channels through learned binary masks, and reconstructed
in both frequency and time domains.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CATCHModelOutput:
    """Outputs required by reconstruction training and anomaly scoring."""

    reconstruction: torch.Tensor
    frequency_reconstruction: torch.Tensor
    clustering_loss: torch.Tensor
    regularization_loss: torch.Tensor
    masks: torch.Tensor
    mask_probabilities: torch.Tensor


class ReversibleInstanceNorm(nn.Module):
    """Stateless reversible instance normalisation for ``(B, T, C)`` data."""

    def __init__(self, n_channels: int, *, affine: bool = False, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_channels))
            self.bias = nn.Parameter(torch.zeros(n_channels))

    def normalize(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = values.mean(dim=1, keepdim=True).detach()
        scale = torch.sqrt(
            values.var(dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()
        normalized = (values - mean) / scale
        if self.affine:
            normalized = normalized * self.weight + self.bias
        return normalized, mean, scale

    def denormalize(
        self, values: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        if self.affine:
            values = (values - self.bias) / (self.weight + self.eps**2)
        return values * scale + mean


class ChannelMaskGenerator(nn.Module):
    """Generate one differentiable binary channel mask per frequency patch."""

    def __init__(self, input_size: int, n_channels: int) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.projection = nn.Linear(input_size, n_channels, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = torch.sigmoid(self.projection(patches)).clamp(1e-6, 1 - 1e-6)
        # Equation 4 defines a Bernoulli resample and the official implementation
        # keeps using Gumbel-Softmax in evaluation mode.  Log-probabilities retain
        # D as the actual Bernoulli probability (the repository's log-odds pair
        # would unintentionally square the odds).
        logits = torch.stack(
            (torch.log(probabilities), torch.log1p(-probabilities)), dim=-1
        )
        mask = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)[..., 0]

        identity = torch.eye(
            self.n_channels, device=patches.device, dtype=patches.dtype
        ).unsqueeze(0)
        off_diagonal = 1.0 - identity
        mask = mask * off_diagonal + identity
        return mask, probabilities


class ChannelMaskedAttention(nn.Module):
    """Multi-head channel attention constrained by a patch-wise binary mask."""

    def __init__(
        self,
        cf_dim: int,
        n_heads: int,
        *,
        head_dim: int,
        dropout: float,
        temperature: float,
    ) -> None:
        super().__init__()
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.temperature = float(temperature)
        inner_dim = n_heads * head_dim
        self.to_q = nn.Linear(cf_dim, inner_dim)
        self.to_k = nn.Linear(cf_dim, inner_dim)
        self.to_v = nn.Linear(cf_dim, inner_dim)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, cf_dim), nn.Dropout(dropout))

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, width = values.shape

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch, channels, self.n_heads, self.head_dim
            ).permute(0, 2, 1, 3)

        query = split_heads(self.to_q(values))
        key = split_heads(self.to_k(values))
        value = split_heads(self.to_v(values))
        raw_scores = torch.einsum("bhid,bhjd->bhij", query, key)
        scaled_scores = raw_scores / math.sqrt(self.head_dim)
        expanded_mask = mask[:, None, :, :]
        large_negative = -math.log(1e10)
        masked_scores = (
            scaled_scores * expanded_mask
            + (1.0 - expanded_mask) * large_negative
        )
        attention = torch.softmax(masked_scores, dim=-1)
        fused = torch.einsum("bhij,bhjd->bhid", attention, value)
        fused = fused.permute(0, 2, 1, 3).reshape(
            batch, channels, self.n_heads * self.head_dim
        )

        # Equation 9 uses the same QK^T similarities as channel attention.
        # Average the multi-head scores to obtain the paper's (N, N) matrix.
        logits = raw_scores.mean(dim=1) / self.temperature
        stable_logits = logits - logits.max(dim=-1, keepdim=True).values
        exponentials = torch.exp(stable_logits)
        positive_sum = (exponentials * mask).sum(dim=-1).clamp_min(1e-12)
        all_sum = exponentials.sum(dim=-1).clamp_min(1e-12)
        clustering = -torch.log(positive_sum / all_sum).mean()
        return self.to_out(fused), clustering


class CATCHBlock(nn.Module):
    """Pre-normalised channel-masked Transformer block."""

    def __init__(
        self,
        cf_dim: int,
        n_heads: int,
        d_ff: int,
        *,
        head_dim: int,
        dropout: float,
        temperature: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(cf_dim)
        self.attention = ChannelMaskedAttention(
            cf_dim,
            n_heads,
            head_dim=head_dim,
            dropout=dropout,
            temperature=temperature,
        )
        self.feed_forward_norm = nn.LayerNorm(cf_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(cf_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, cf_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, clustering = self.attention(
            self.attention_norm(values), mask
        )
        values = values + attended
        values = values + self.feed_forward(self.feed_forward_norm(values))
        return values, clustering


class ResidualFlattenHead(nn.Module):
    """Repository-faithful residual MLP used for real/imaginary spectra."""

    def __init__(
        self,
        input_width: int,
        seq_len: int,
        *,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_layers < 0:
            raise ValueError("n_layers must be >= 0")
        self.residual_layers = nn.ModuleList(
            nn.Linear(input_width, input_width) for _ in range(n_layers)
        )
        self.output = nn.Linear(input_width, seq_len)
        # Kept for state/API compatibility. The official non-individual head
        # defines this module but does not apply it in forward().
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.flatten(start_dim=-2)
        for layer in self.residual_layers:
            values = values + F.relu(layer(values))
        return self.output(values)


class CATCHModel(nn.Module):
    """Channel-aware frequency-patching reconstruction network."""

    def __init__(
        self,
        n_channels: int,
        *,
        seq_len: int = 192,
        patch_size: int = 16,
        patch_stride: int = 8,
        cf_dim: int = 64,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 2,
        head_dim: int = 64,
        d_ff: int = 256,
        dropout: float = 0.2,
        head_dropout: float = 0.1,
        head_layers: int = 3,
        temperature: float = 0.07,
        affine_revin: bool = False,
        mask_source: str = "projected",
    ) -> None:
        super().__init__()
        if n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        if seq_len < 2 or not 1 <= patch_size <= seq_len:
            raise ValueError("patch_size must be in [1, seq_len]")
        if patch_stride < 1:
            raise ValueError("patch_stride must be >= 1")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if cf_dim < 1 or d_model < 1 or head_dim < 1:
            raise ValueError("cf_dim, d_model, and head_dim must be >= 1")
        if mask_source not in {"projected", "raw"}:
            raise ValueError("mask_source must be 'projected' (paper) or 'raw' (repo)")

        self.n_channels = int(n_channels)
        self.seq_len = int(seq_len)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.patch_count = 1 + (seq_len - patch_size) // patch_stride
        self.cf_dim = int(cf_dim)
        self.d_model = int(d_model)
        self.head_dim = int(head_dim)
        self.mask_source = mask_source
        self.revin = ReversibleInstanceNorm(n_channels, affine=affine_revin)
        mask_width = cf_dim if mask_source == "projected" else 2 * patch_size
        self.mask_generator = ChannelMaskGenerator(mask_width, n_channels)
        self.patch_projection = nn.Sequential(
            nn.Linear(2 * patch_size, cf_dim), nn.Dropout(dropout)
        )
        self.blocks = nn.ModuleList(
            CATCHBlock(
                cf_dim,
                n_heads,
                d_ff,
                head_dim=head_dim,
                dropout=dropout,
                temperature=temperature,
            )
            for _ in range(n_layers)
        )
        spectrum_width = 2 * d_model
        self.frequency_projection = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(cf_dim, spectrum_width)
        )
        self.real_projection = nn.Linear(spectrum_width, spectrum_width)
        self.imag_projection = nn.Linear(spectrum_width, spectrum_width)
        flattened_width = self.patch_count * spectrum_width
        self.real_head = ResidualFlattenHead(
            flattened_width,
            seq_len,
            n_layers=head_layers,
            dropout=head_dropout,
        )
        self.imag_head = ResidualFlattenHead(
            flattened_width,
            seq_len,
            n_layers=head_layers,
            dropout=head_dropout,
        )
        self.ircom = nn.Linear(2 * seq_len, seq_len)

    def forward(self, values: torch.Tensor) -> CATCHModelOutput:
        if values.ndim != 3:
            raise ValueError("values must have shape (batch, time, channels)")
        if values.shape[1:] != (self.seq_len, self.n_channels):
            raise ValueError(
                f"expected (*, {self.seq_len}, {self.n_channels}), got {tuple(values.shape)}"
            )

        normalized, mean, scale = self.revin.normalize(values)
        spectrum = torch.fft.fft(normalized.permute(0, 2, 1), dim=-1)
        real_patches = spectrum.real.unfold(-1, self.patch_size, self.patch_stride)
        imag_patches = spectrum.imag.unfold(-1, self.patch_size, self.patch_stride)
        patches = torch.cat((real_patches, imag_patches), dim=-1)
        patches = patches.permute(0, 2, 1, 3)
        batch, patch_count, channels, width = patches.shape
        flat_patches = patches.reshape(batch * patch_count, channels, width)

        hidden = self.patch_projection(flat_patches)
        mask_input = hidden if self.mask_source == "projected" else flat_patches
        masks, probabilities = self.mask_generator(mask_input)
        clustering_losses: list[torch.Tensor] = []
        for block in self.blocks:
            hidden, clustering = block(hidden, masks)
            clustering_losses.append(clustering)

        hidden = self.frequency_projection(hidden)
        hidden = hidden.reshape(batch, patch_count, channels, -1).permute(0, 2, 1, 3)
        real = self.real_head(self.real_projection(hidden))
        imag = self.imag_head(self.imag_projection(hidden))
        reconstructed_spectrum = torch.complex(real, imag)
        inverse = torch.fft.ifft(reconstructed_spectrum, dim=-1)
        reconstructed_normalized = self.ircom(
            torch.cat((inverse.real, inverse.imag), dim=-1)
        ).permute(0, 2, 1)
        reconstruction = self.revin.denormalize(
            reconstructed_normalized, mean, scale
        )

        identity = torch.eye(
            channels, device=values.device, dtype=values.dtype
        ).unsqueeze(0)
        regularization = (
            torch.linalg.vector_norm(
                (identity - masks).flatten(start_dim=-2), ord=2, dim=-1
            )
            / channels
        ).mean()
        return CATCHModelOutput(
            reconstruction=reconstruction,
            frequency_reconstruction=reconstructed_spectrum.permute(0, 2, 1),
            clustering_loss=torch.stack(clustering_losses).mean(),
            regularization_loss=regularization,
            masks=masks.reshape(batch, patch_count, channels, channels),
            mask_probabilities=probabilities.reshape(
                batch, patch_count, channels, channels
            ),
        )


def frequency_reconstruction_loss(
    reconstructed_spectrum: torch.Tensor, normalized_values: torch.Tensor
) -> torch.Tensor:
    """Equation 15: sum of real- and imaginary-spectrum L1 losses."""

    target = torch.fft.fft(normalized_values, dim=1)
    difference = reconstructed_spectrum - target
    return difference.real.abs().mean() + difference.imag.abs().mean()


def frequency_point_error(
    reconstructed: torch.Tensor,
    observed: torch.Tensor,
    *,
    patch_size: int,
    patch_stride: int = 1,
) -> torch.Tensor:
    """Distribute patch-level FFT errors back to their covered timestamps.

    This implements the point-granularity frequency scoring described in
    Section 3.5 and Appendix A.6 of the paper.
    """

    if reconstructed.shape != observed.shape or observed.ndim != 3:
        raise ValueError("reconstructed and observed must share shape (B, T, C)")
    length = observed.shape[1]
    if not 1 <= patch_size <= length:
        raise ValueError("patch_size must be in [1, window length]")
    if patch_stride < 1:
        raise ValueError("patch_stride must be >= 1")

    starts = list(range(0, length - patch_size + 1, patch_stride))
    covered_length = patch_size + (len(starts) - 1) * patch_stride
    padding_length = length - covered_length
    reconstructed_patches = torch.stack(
        [reconstructed[:, start : start + patch_size] for start in starts], dim=1
    )
    observed_patches = torch.stack(
        [observed[:, start : start + patch_size] for start in starts], dim=1
    )
    spectral_difference = (
        torch.fft.fft(reconstructed_patches, dim=2)
        - torch.fft.fft(observed_patches, dim=2)
    )
    patch_errors = (
        spectral_difference.real.abs().mean(dim=2)
        + spectral_difference.imag.abs().mean(dim=2)
    )

    point_errors = observed.new_zeros(observed.shape)
    counts = observed.new_zeros((1, length, 1))
    for patch_index, start in enumerate(starts):
        point_errors[:, start : start + patch_size] += patch_errors[
            :, patch_index
        ].unsqueeze(1)
        counts[:, start : start + patch_size] += 1

    # Algorithm 2 scores a non-covered suffix as its own (shorter) patch.
    if padding_length:
        tail_difference = (
            torch.fft.fft(reconstructed[:, -padding_length:], dim=1)
            - torch.fft.fft(observed[:, -padding_length:], dim=1)
        )
        tail_error = (
            tail_difference.real.abs().mean(dim=1)
            + tail_difference.imag.abs().mean(dim=1)
        )
        point_errors[:, covered_length:] = tail_error.unsqueeze(1)
        counts[:, covered_length:] = 1
    return point_errors / counts.clamp_min(1)
