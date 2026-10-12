"""ConvGRU + trend LSTM GAN on geographic patches with explicit missing cells."""
from __future__ import annotations

from contextlib import contextmanager

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


@contextmanager
def dropout_disabled(module):
    """Run `module` without dropout whatever its mode; every dropout layer gets its mode back."""
    layers = [layer for layer in module.modules() if isinstance(layer, nn.modules.dropout._DropoutNd)]
    states = [layer.training for layer in layers]
    try:
        for layer in layers:
            layer.training = False
        yield
    finally:
        for layer, state in zip(layers, states):
            layer.training = state


class STGANGenerator(nn.Module):
    def __init__(self, n_features, hidden_size, n_layers, cnn_channels, cnn_layers,
                 time_feature_size=31, kernel_size=3, dropout_enabled=True, dropout_p=0.2):
        super().__init__()
        if type(dropout_enabled) is not bool or not 0 <= dropout_p < 1:
            raise ValueError("Require boolean dropout_enabled and dropout_p in [0, 1).")
        self.spatial_dropout = nn.Dropout(dropout_p) if dropout_enabled else nn.Identity()
        self.temporal_dropout = nn.Dropout(dropout_p) if dropout_enabled else nn.Identity()
        self.fusion_dropout = nn.Dropout(dropout_p) if dropout_enabled else nn.Identity()
        self.recent_encoder = ConvGRU(n_features, cnn_channels, cnn_layers, kernel_size)
        self.trend_encoder = nn.LSTM(n_features, hidden_size, num_layers=n_layers,
                                    batch_first=True)
        self.time_projection = nn.Sequential(nn.Linear(time_feature_size, hidden_size), nn.ReLU())
        self.output_projection = nn.Sequential(
            nn.Conv2d(cnn_channels + 2 * hidden_size, n_features, 1), nn.Tanh())

    def encode(self, recent, trend, mask, time_features):
        """forward up to its dropouts. Nothing here is stochastic, so MC scoring
        computes it once per batch; forward itself is unchanged and stays the reference."""
        temporal, _ = self.trend_encoder(trend)
        return self.recent_encoder(recent, mask), temporal[:, -1], self.time_projection(time_features)

    def decode(self, spatial, temporal, calendar, mask):
        """The rest of forward: its three dropouts, in the same order, and the output."""
        spatial = self.spatial_dropout(spatial)
        h, w = spatial.shape[-2:]
        temporal = self.temporal_dropout(temporal)[:, :, None, None].expand(-1, -1, h, w)
        calendar = calendar[:, :, None, None].expand(-1, -1, h, w)
        fused = self.fusion_dropout(torch.cat((spatial, temporal, calendar), dim=1))
        return torch.where(mask.bool(), self.output_projection(fused), 0.0)

    def forward(self, recent, trend, mask, time_features):
        spatial = self.spatial_dropout(self.recent_encoder(recent, mask))
        temporal, _ = self.trend_encoder(trend)
        h, w = spatial.shape[-2:]
        temporal = self.temporal_dropout(temporal[:, -1])[:, :, None, None].expand(-1, -1, h, w)
        calendar = self.time_projection(time_features)[:, :, None, None].expand(-1, -1, h, w)
        fused = self.fusion_dropout(torch.cat((spatial, temporal, calendar), dim=1))
        predicted = self.output_projection(fused)
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

    def forward(self, sequence, mask, *, return_logits=False):
        if sequence.ndim != 5 or sequence.shape[1] < 2:
            raise ValueError("Discriminator requires recent history plus current data.")
        return self.score_current(self.encode_history(sequence[:, :-1], mask), sequence[:, -1], mask,
                                  return_logits=return_logits)

    def encode_history(self, recent, mask):
        """Parameter-dependent history: share only until the next D update."""
        historical = self.sequence_encoder(recent, mask)
        historical = torch.where(mask.bool(), historical, 0.0)
        return self.sequence_projection(historical.flatten(start_dim=1))

    def score_current(self, historical, current, mask, *, return_logits=False):
        current = self.current_projection(_masked_inputs(current, mask))
        current = current.masked_fill(~mask.bool(), -torch.inf).amax(dim=(2, 3))
        fused = torch.cat((current, historical), dim=1)
        # Preserve module indices/state_dict keys of existing checkpoints.
        for index in range(len(self.output) - 1):
            fused = self.output[index](fused)
        logits = fused.float()
        return logits if return_logits else self.output[-1](logits)

    def score_pair(self, recent, observed, predicted, mask, *, return_logits=False):
        historical = self.encode_history(recent, mask)
        return (self.score_current(historical, observed, mask, return_logits=return_logits),
                self.score_current(historical, predicted, mask, return_logits=return_logits))

    def penultimate(self, historical, current, mask):
        """Activations entering the final linear layer: D's last learned representation.

        Monitoring only. score_current stays the reference for the score; the two
        agree, since score_from_penultimate(penultimate(...)) is its output.
        """
        current = self.current_projection(_masked_inputs(current, mask))
        current = current.masked_fill(~mask.bool(), -torch.inf).amax(dim=(2, 3))
        fused = torch.cat((current, historical), dim=1)
        for layer in self.output[:-2]:
            fused = layer(fused)
        return fused

    def score_from_penultimate(self, features):
        return self.output[-1](self.output[-2](features).float())


class STGAN(nn.Module):
    """Grid ConvGRU implementation; original GCGRU lives on feat/stgan-paper."""
    def __init__(self, *, n_features, hidden_size=64, n_layers=2,
                 cnn_channels=32, cnn_layers=2, patch_size=3, time_feature_size=31,
                 kernel_size=3, dropout_enabled=True, dropout_p=0.2, discriminator_input_dropout=True,
                 reconstruction_feature_weights=None):
        super().__init__()
        if min(n_features, hidden_size, n_layers, cnn_channels, cnn_layers) < 1:
            raise ValueError("Model dimensions and layer counts must be positive.")
        if patch_size not in (1, 3, 5):
            raise ValueError("patch_size must be 1, 3 or 5.")
        if type(discriminator_input_dropout) is not bool:
            raise ValueError("discriminator_input_dropout must be a boolean.")
        self.discriminator_input_dropout = discriminator_input_dropout
        self.set_reconstruction_feature_weights(reconstruction_feature_weights, n_features)
        self.generator = STGANGenerator(n_features, hidden_size, n_layers, cnn_channels,
                                       cnn_layers, time_feature_size, kernel_size,
                                       dropout_enabled, dropout_p)
        self.discriminator = STGANDiscriminator(n_features, hidden_size, cnn_channels,
                                               cnn_layers, patch_size, kernel_size)

    def set_reconstruction_feature_weights(self, weights, n_features):
        """Weight of each variable in the reconstruction error of the loss and of the score.

        None is the plain mean over the variables, as in the paper. The weights are part of the
        model configuration, not of its learned state.
        """
        if weights is not None:
            weights = torch.as_tensor(weights, dtype=torch.float32)
            if weights.shape != (n_features,) or not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("reconstruction_feature_weights must be one finite positive value per variable.")
        self.register_buffer("reconstruction_feature_weights", weights, persistent=False)

    def weighted_errors(self, errors):
        """Squared errors [sample,feature,...] times the weight of their variable; the same tensor without weights."""
        weights = self.reconstruction_feature_weights
        if weights is None:
            return errors
        return errors * weights.to(errors.dtype).view(1, -1, *([1] * (errors.ndim - 2)))

    def discriminator_input(self, recent, trend, mask, time_features):
        """The reconstruction D judges: G's output, without G's dropout unless
        discriminator_input_dropout. Then D cannot tell it by the dropout noise."""
        if self.discriminator_input_dropout:
            return self.generator(recent, trend, mask, time_features).float()
        with dropout_disabled(self.generator):
            return self.generator(recent, trend, mask, time_features).float()

    def reconstructions(self, recent, trend, mask, time_features):
        """The reconstruction of the reconstruction loss and the one D judges, from one encoding.

        They are the same tensor with discriminator_input_dropout. Without it the first keeps
        G's dropout and the second has none; both carry gradients to G.
        """
        if self.discriminator_input_dropout:
            generated = self.generator(recent, trend, mask, time_features).float()
            return generated, generated
        encoded = self.generator.encode(recent, trend, mask, time_features)
        generated = self.generator.decode(*encoded, mask).float()
        with dropout_disabled(self.generator):
            return generated, self.generator.decode(*encoded, mask).float()

    def monitoring_outputs(self, recent, trend, mask, time_features, observed):
        """One forward for monitoring, per sample: the two score components and D's
        penultimate activations of the observation and of its reconstruction."""
        predicted = self.generator(recent, trend, mask, time_features).float()
        historical = self.discriminator.encode_history(recent, mask)
        real = self.discriminator.penultimate(historical, observed, mask)
        fake = self.discriminator.penultimate(historical, predicted, mask)
        difference = self.discriminator.score_from_penultimate(real) - self.discriminator.score_from_penultimate(fake)
        errors = torch.where(mask.bool(), predicted - observed, 0.0).square()
        return masked_cell_mean(self.weighted_errors(errors), mask), difference.squeeze(1), real, fake

    def reconstructed_cells(self, recent, trend, mask, time_features, observed):
        """Observed and generated values [sample,feature] of the target cell of every sample."""
        centre = mask.shape[-1] // 2
        generated = self.generator(recent, trend, mask, time_features).float()
        return observed[:, :, centre, centre], generated[:, :, centre, centre]

    def reconstruction_valid(self, mask):
        """Per sample: its reconstruction error averages at least one cell (the mask of masked_cell_mean)."""
        return mask.bool().flatten(1).any(dim=1)

    def score_draws(self, recent, trend, mask, time_features, observed, samples, *, share_history=True):
        """`samples` scoring draws [samples,B,2+F], with the draw-independent work done once.

        Each draw is the patch-mean squared error (with the weights of the variables,
        when the model has them), D(real) - D(generated) and the target-cell squared
        errors, never weighted. Dropout sits after G's encoders and D has none:
        G's encoders, D's history encoding and D's score of the observation are the
        same in every draw. Each draw applies the dropouts in forward's order, so
        the values are those of repeated components calls. Without
        discriminator_input_dropout D judges the dropout-free reconstruction, as in
        training: its component is then the same in every draw.
        """
        center = mask.shape[-1] // 2
        def pack(real, fake, errors):
            return torch.cat((masked_cell_mean(self.weighted_errors(errors), mask)[:, None], real - fake,
                              errors[:, :, center, center]), dim=1)
        if not share_history:
            clean_fake = None
            if not self.discriminator_input_dropout:
                clean = self.discriminator_input(recent, trend, mask, time_features)
                clean_fake = self.discriminator(torch.cat((recent, clean[:, None]), dim=1), mask)
            draws = []
            for _ in range(samples):
                real, fake, errors = self.components(recent, trend, mask, time_features, observed,
                                                     share_history=False)[1:]
                draws.append(pack(real, fake if clean_fake is None else clean_fake, errors))
            return torch.stack(draws)
        encoded = self.generator.encode(recent, trend, mask, time_features)
        historical = self.discriminator.encode_history(recent, mask)
        real = self.discriminator.score_current(historical, observed, mask)
        clean_fake = None
        if not self.discriminator_input_dropout:
            with dropout_disabled(self.generator):
                clean_fake = self.discriminator.score_current(
                    historical, self.generator.decode(*encoded, mask).float(), mask)
        draws = []
        for _ in range(samples):
            predicted = self.generator.decode(*encoded, mask).float()
            fake = self.discriminator.score_current(historical, predicted, mask) if clean_fake is None else clean_fake
            draws.append(pack(real, fake, torch.where(mask.bool(), predicted - observed, 0.0).square()))
        return torch.stack(draws)

    def observation_scores(self, recent, mask, observed):
        """D's score of the observation, one value per sample: the first term of the
        discriminator component. D has no dropout, so no scoring draw changes it."""
        historical = self.discriminator.encode_history(recent, mask)
        return self.discriminator.score_current(historical, observed, mask).float().squeeze(1)

    def components(self, recent, trend, mask, time_features, observed, *, share_history=True):
        predicted = self.generator(recent, trend, mask, time_features).float()
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
