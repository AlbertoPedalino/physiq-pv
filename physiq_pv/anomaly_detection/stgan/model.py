"""ConvGRU + trend LSTM GAN on geographic patches with explicit missing cells."""
from __future__ import annotations

import torch
from torch import nn


def masked_cell_mean(values, mask):
    """Per-sample mean over observed cells/features, then caller averages samples."""
    valid = mask.to(dtype=values.dtype)
    numerator = torch.where(valid.bool(), values, 0.0).sum(dim=(1, 2, 3))
    denominator = valid.sum(dim=(1, 2, 3)) * values.shape[1]
    return numerator / denominator.clamp_min(1)


def _masked_inputs(values, mask):
    # Mask both real and generated values, including NaNs in unavailable cells.
    return torch.cat((torch.where(mask.bool(), values, 0.0), mask.to(values.dtype)), dim=1)


class ConvGRUCell(nn.Module):
    """Paper GCGRU gates with configurable spatial convolutions."""

    def __init__(self, n_features, channels, kernel_size=3):
        super().__init__()
        if type(kernel_size) is not int or kernel_size not in (1, 3, 5):
            raise ValueError("kernel_size must be an integer: 1, 3 or 5.")
        # Each layer receives the validity mask as an additional input channel.
        joint_channels = n_features + 1 + channels
        padding = kernel_size // 2
        self.reset = nn.Conv2d(joint_channels, channels, kernel_size, padding=padding)
        self.update = nn.Conv2d(joint_channels, channels, kernel_size, padding=padding)
        self.candidate = nn.Conv2d(joint_channels, channels, kernel_size, padding=padding)

    def forward(self, values, hidden, mask):
        values = _masked_inputs(values, mask)
        hidden = torch.where(mask.bool(), hidden, 0.0)
        joint = torch.cat((values, hidden), dim=1)
        reset = torch.sigmoid(self.reset(joint))
        update = torch.sigmoid(self.update(joint))
        candidate = torch.tanh(self.candidate(torch.cat((values, reset * hidden), dim=1)))
        next_hidden = update * hidden + (1.0 - update) * candidate
        # Missing cells must not acquire state that can leak into valid neighbors
        # in a later layer or time step. Convolution padding is separate from this.
        return torch.where(mask.bool(), next_hidden, 0.0)


class ConvGRU(nn.Module):
    """Encode [batch,time,features,height,width]; reset state for each window."""

    def __init__(self, n_features, channels, layers, kernel_size=3):
        super().__init__()
        if min(n_features, channels, layers) < 1:
            raise ValueError("ConvGRU dimensions and layer count must be positive.")
        self.n_features = n_features
        self.channels = channels
        self.layers = nn.ModuleList(
            ConvGRUCell(n_features if index == 0 else channels, channels, kernel_size)
            for index in range(layers)
        )

    def forward(self, sequence, mask):
        if sequence.ndim != 5 or sequence.shape[1] < 1:
            raise ValueError("ConvGRU requires nonempty [batch,time,features,height,width] input.")
        batch, _, features, height, width = sequence.shape
        if features != self.n_features or mask.shape != (batch, 1, height, width):
            raise ValueError("ConvGRU input features or validity mask shape do not match.")
        hidden = [sequence.new_zeros((batch, self.channels, height, width))
                  for _ in self.layers]
        for time_index in range(sequence.shape[1]):
            output = sequence[:, time_index]
            for index, layer in enumerate(self.layers):
                hidden[index] = layer(output, hidden[index], mask)
                output = hidden[index]
        return output


class STGANGenerator(nn.Module):
    def __init__(self, n_features, hidden_size, n_layers, cnn_channels, cnn_layers,
                 time_feature_size=31, kernel_size=3):
        super().__init__()
        self.recent_encoder = ConvGRU(n_features, cnn_channels, cnn_layers, kernel_size)
        self.trend_encoder = nn.LSTM(n_features, hidden_size, num_layers=n_layers,
                                    batch_first=True)
        self.time_projection = nn.Sequential(nn.Linear(time_feature_size, hidden_size), nn.ReLU())
        self.output_projection = nn.Sequential(
            nn.Conv2d(cnn_channels + 2 * hidden_size, n_features, 1), nn.Tanh())

    def forward(self, recent, trend, mask, time_features):
        spatial = self.recent_encoder(recent, mask)
        temporal, _ = self.trend_encoder(trend)
        h, w = spatial.shape[-2:]
        temporal = temporal[:, -1, :, None, None].expand(-1, -1, h, w)
        calendar = self.time_projection(time_features)[:, :, None, None].expand(-1, -1, h, w)
        predicted = self.output_projection(torch.cat((spatial, temporal, calendar), dim=1))
        return torch.where(mask.bool(), predicted, 0.0)


class STGANDiscriminator(nn.Module):
    def __init__(self, n_features, hidden_size, cnn_channels, cnn_layers, patch_size,
                 kernel_size=3):
        super().__init__()
        self.sequence_encoder = ConvGRU(n_features, cnn_channels, cnn_layers, kernel_size)
        self.sequence_projection = nn.Sequential(
            nn.Linear(patch_size**2 * cnn_channels, hidden_size), nn.ReLU())
        self.current_projection = nn.Sequential(
            nn.Conv2d(n_features + 1, hidden_size, 1), nn.Sigmoid())
        self.output = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(),
                                    nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, sequence, mask):
        if sequence.ndim != 5 or sequence.shape[1] < 2:
            raise ValueError("Discriminator requires recent history plus current data.")
        return self.score_current(self.encode_history(sequence[:, :-1], mask), sequence[:, -1], mask)

    def encode_history(self, recent, mask):
        """Parameter-dependent history: share only until the next D update."""
        historical = self.sequence_encoder(recent, mask)
        historical = torch.where(mask.bool(), historical, 0.0)
        return self.sequence_projection(historical.flatten(start_dim=1))

    def score_current(self, historical, current, mask):
        current = self.current_projection(_masked_inputs(current, mask))
        current = current.masked_fill(~mask.bool(), -torch.inf).amax(dim=(2, 3))
        return self.output(torch.cat((current, historical), dim=1))

    def score_pair(self, recent, observed, predicted, mask):
        historical = self.encode_history(recent, mask)
        return (self.score_current(historical, observed, mask),
                self.score_current(historical, predicted, mask))


class STGAN(nn.Module):
    """Grid ConvGRU implementation; original GCGRU lives on feat/stgan-paper."""
    def __init__(self, *, n_features, hidden_size=64, n_layers=2,
                 cnn_channels=32, cnn_layers=2, patch_size=3, time_feature_size=31,
                 kernel_size=3):
        super().__init__()
        if min(n_features, hidden_size, n_layers, cnn_channels, cnn_layers) < 1:
            raise ValueError("Model dimensions and layer counts must be positive.")
        if patch_size not in (1, 3, 5):
            raise ValueError("patch_size must be 1, 3 or 5.")
        self.generator = STGANGenerator(n_features, hidden_size, n_layers, cnn_channels,
                                       cnn_layers, time_feature_size, kernel_size)
        self.discriminator = STGANDiscriminator(n_features, hidden_size, cnn_channels,
                                               cnn_layers, patch_size, kernel_size)

    def components(self, recent, trend, mask, time_features, observed, *, share_history=True):
        predicted = self.generator(recent, trend, mask, time_features)
        if share_history:
            real_score, fake_score = self.discriminator.score_pair(recent, observed, predicted, mask)
        else:
            real_score = self.discriminator(torch.cat((recent, observed[:, None]), dim=1), mask)
            fake_score = self.discriminator(torch.cat((recent, predicted[:, None]), dim=1), mask)
        errors = torch.where(mask.bool(), predicted - observed, 0.0).square()
        return predicted, real_score, fake_score, errors

    def parameter_counts(self):
        return {"generator": sum(p.numel() for p in self.generator.parameters()),
                "discriminator": sum(p.numel() for p in self.discriminator.parameters())}
