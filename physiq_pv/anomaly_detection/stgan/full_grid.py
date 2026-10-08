"""Full-grid training variant of the grid ConvGRU STGAN: one sample is one complete field.

The patch model (model.py, data.py) takes as a sample one target cell with its patch. Here a
sample is one target timestamp with every cell of the lattice:

    recent [B,T,F,H,W] -> ConvGRU over the whole field -> reconstruction [B,F,H,W]

The modules and their parameters are those of the patch model, with the same state_dict keys;
what changes is the unit they are applied to. This is a distinct training variant, not an
optimization of the patch one:

- an optimizer step uses a batch of timestamps (every cell of each) instead of a batch of
  cells, so an epoch has about n_locations times fewer steps;
- G's convolutions run over the whole field: its receptive field is no longer cut at the
  border of one patch, and it reconstructs every cell once instead of once per patch.

D stays local. The patch of every cell is gathered from the field and the patch discriminator
scores all of them in one batched pass, with the masks and the zeros of the patch dataset: its
output for a cell is that of the patch discriminator on that cell's patch. The ConvGRU of D
pads inside each patch, so sliding it over the field would not give that value.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .data import STGANWindowDataset
from .grid import SpatialGrid
from .model import STGAN, STGANDiscriminator, STGANGenerator, masked_cell_mean


def lattice(grid: SpatialGrid):
    """Height, width and the flat [H*W] cell of every location of the grid."""
    rows, columns = np.asarray(grid.row_indices), np.asarray(grid.column_indices)
    height, width = int(rows.max()) + 1, int(columns.max()) + 1
    return height, width, (rows * width + columns).astype(np.int64)


class STGANFullGridDataset(STGANWindowDataset):
    """One sample per target timestamp: the window of every location, laid out on the lattice.

    For B timestamps fetch_batch returns recent [B,T,F,H,W], trend [B,S,N,F] (one sequence per
    location, time-major as stored), mask [B,1,H,W], calendar [B,31] (onehot) or [B,N,4]
    (cyclic), observed [B,F,H,W], then the target position and the location of each of the
    B*N cells, in the order of the model's per-cell outputs. The values are those of the patch
    dataset: same normalization, zero where the lattice has no location.
    """

    def __init__(self, data, timestamps, grid, **options):
        super().__init__(data, timestamps, grid, **options)
        self.height, self.width, self.cells = lattice(grid)
        occupied = np.zeros(self.height * self.width, dtype=np.float32)
        occupied[self.cells] = 1.0
        self.field_mask = occupied.reshape(1, 1, self.height, self.width)

    def __len__(self) -> int:
        return len(self.targets)

    def fetch_batch(self, indices):
        positions = np.asarray(indices, dtype=np.int64)
        if positions.ndim != 1 or not len(positions) or np.any(positions < 0) or np.any(positions >= len(self)):
            raise IndexError("Expected a nonempty batch of valid sample indices.")
        targets, count, steps = self.targets[positions], len(positions), self.recent_steps + 1
        # One contiguous read per timestamp: the trend history, then the target. The recent
        # steps are the end of that history.
        window = np.stack([self._normalise(np.asarray(self.data[int(target) - self.trend_steps:int(target) + 1]))
                           for target in targets])
        field = np.zeros((count, steps, window.shape[-1], self.height * self.width), dtype=np.float32)
        field[..., self.cells] = window[:, -steps:].transpose(0, 1, 3, 2)
        field = field.reshape(count, steps, -1, self.height, self.width)
        locations = np.arange(self.n_locations)
        calendar = (self._calendar(targets, locations) if self.time_encoding == "onehot"
                    else self._calendar(targets[:, None], locations[None]))
        arrays = (field[:, :-1], window[:, :-1], np.repeat(self.field_mask, count, axis=0),
                  calendar, field[:, -1], np.repeat(positions, self.n_locations), np.tile(locations, count))
        return tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in arrays)

    def __getitem__(self, item: int):
        """One timestamp without the batch axis; loaders use fetch_batch."""
        batch = self.fetch_batch([item])
        return (*(values[0] for values in batch[:5]), *batch[5:])


class GridLayout(nn.Module):
    """Where each location sits on the [H,W] lattice and which cells form its patch.

    The buffers are not persistent: a full-grid model has the state_dict of the patch model.
    """

    def __init__(self, grid: SpatialGrid):
        super().__init__()
        self.height, self.width, cells = lattice(grid)
        self.n_locations, self.patch_size = grid.n_locations, grid.patch_size
        # Absent patch cells read any existing cell and are zeroed by the patch mask.
        nodes = np.maximum(grid.node_indices, 0).reshape(-1)
        for name, values in (("cell_index", cells), ("patch_index", cells[nodes]),
                             ("patch_mask", grid.valid_mask[:, None].astype(np.float32))):
            self.register_buffer(name, torch.from_numpy(np.ascontiguousarray(values)), persistent=False)

    def field(self, values):
        """Per-location values [B,N,C] on the lattice [B,C,H,W]; zero where there is no location."""
        batch, _, channels = values.shape
        canvas = values.new_zeros((batch, channels, self.height * self.width))
        return canvas.index_copy(2, self.cell_index, values.transpose(1, 2)).view(
            batch, channels, self.height, self.width)

    def cells(self, field):
        """A field [B,C,H,W] at every location: [B*N,C], timestamp-major."""
        return field.flatten(2).index_select(2, self.cell_index).transpose(1, 2).flatten(0, 1)

    def patch_masks(self, batch):
        """The patch validity mask of every location, for `batch` timestamps: [B*N,1,P,P]."""
        return self.patch_mask.repeat(batch, 1, 1, 1)

    def patches(self, field):
        """The patch of every location of a field [B,(T,)C,H,W]: [B*N,(T,)C,P,P], absent cells zero.

        Row b*N + n is the patch the patch dataset builds for location n at timestamp b.
        """
        size = self.patch_size
        values = field.flatten(-2).index_select(-1, self.patch_index)
        values = values.unflatten(-1, (self.n_locations, size, size)).movedim(-3, 1).flatten(0, 1)
        mask = self.patch_masks(len(field)).bool()
        return torch.where(mask if values.ndim == 4 else mask[:, None], values, 0.0)


class FullGridGenerator(STGANGenerator):
    """The patch generator's modules on complete fields: [B,T,F,H,W] -> [B,F,H,W].

    The ConvGRU slides over the whole lattice. The trend LSTM reads one sequence per location
    and the time projection one calendar vector per timestamp (onehot) or per location
    (cyclic); both are then placed on the lattice, where the patch model repeats the target
    cell's over its patch. The three dropouts are applied in the order of the patch forward.
    """

    def __init__(self, layout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout = layout

    def encode(self, recent, trend, mask, time_features):
        batch, steps, locations, features = trend.shape
        temporal, _ = self.trend_encoder(trend.transpose(1, 2).reshape(batch * locations, steps, features))
        return (self.recent_encoder(recent, mask), temporal[:, -1].reshape(batch, locations, -1),
                self.time_projection(time_features))

    def decode(self, spatial, temporal, calendar, mask):
        spatial = self.spatial_dropout(spatial)
        temporal = self.layout.field(self.temporal_dropout(temporal))
        calendar = (self.layout.field(calendar) if calendar.ndim == 3
                    else calendar[:, :, None, None].expand(-1, -1, *spatial.shape[-2:]))
        fused = self.fusion_dropout(torch.cat((spatial, temporal, calendar), dim=1))
        return torch.where(mask.bool(), self.output_projection(fused), 0.0)

    def forward(self, recent, trend, mask, time_features):
        return self.decode(*self.encode(recent, trend, mask, time_features), mask)


class FullGridDiscriminator(STGANDiscriminator):
    """The patch discriminator on the patch of every cell of a field: one local score per cell.

    Inputs are fields and outputs have one row per (timestamp, location), timestamp-major. A
    row is what the patch discriminator returns for that cell's patch: same weights, same
    patch mask, same zeros. The field mask argument is not used.
    """

    def __init__(self, layout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout = layout

    def encode_history(self, recent, mask):
        return super().encode_history(self.layout.patches(recent), self.layout.patch_masks(len(recent)))

    def score_current(self, historical, current, mask, *, return_logits=False):
        return super().score_current(historical, self.layout.patches(current),
                                     self.layout.patch_masks(len(current)), return_logits=return_logits)

    def penultimate(self, historical, current, mask):
        return super().penultimate(historical, self.layout.patches(current), self.layout.patch_masks(len(current)))


class FullGridSTGAN(STGAN):
    """STGAN whose sample is a complete field; per-cell outputs keep the patch definitions.

    Every score component has one row per (timestamp, location): the reconstruction component
    is the mean squared error over the cell's patch, D's component is its local output on that
    patch, and the feature errors are the cell's own.
    """

    def __init__(self, *, grid, n_features, hidden_size=64, n_layers=2, cnn_channels=32, cnn_layers=2,
                 patch_size=3, time_feature_size=31, kernel_size=3, dropout_enabled=True, dropout_p=0.2):
        nn.Module.__init__(self)
        if min(n_features, hidden_size, n_layers, cnn_channels, cnn_layers) < 1:
            raise ValueError("Model dimensions and layer counts must be positive.")
        if patch_size not in (1, 3, 5) or grid.patch_size != patch_size:
            raise ValueError("patch_size must be 1, 3 or 5 and match the grid.")
        self.layout = GridLayout(grid)
        self.generator = FullGridGenerator(self.layout, n_features, hidden_size, n_layers, cnn_channels,
                                           cnn_layers, time_feature_size, kernel_size,
                                           dropout_enabled, dropout_p)
        self.discriminator = FullGridDiscriminator(self.layout, n_features, hidden_size, cnn_channels,
                                                   cnn_layers, patch_size, kernel_size)

    def reconstruction_score(self, errors, mask):
        return masked_cell_mean(self.layout.patches(errors), self.layout.patch_masks(len(errors)))

    def target_cells(self, values):
        return self.layout.cells(values)

    def reconstruction_valid(self, mask):
        return self.layout.patch_masks(len(mask)).bool().flatten(1).any(dim=1)
