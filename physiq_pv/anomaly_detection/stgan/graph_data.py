"""One sample is a timestamp and ALL valid nodes; never patches for G."""
import numpy as np
import torch

from .data import STGANWindowDataset


class STGANGraphDataset(STGANWindowDataset):
    def __len__(self):
        return len(self.targets)

    def fetch_batch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("Expected a nonempty batch of valid timestamp indices.")
        targets = self.targets[indices]
        nodes = np.arange(self.n_locations, dtype=np.int64)
        # Paired time/location indexing also works on disk-backed ContextArray.
        recent = self._normalise(self.data[
            (targets[:, None] - self.recent_offsets)[..., None], nodes[None, None], :])
        trend = self._normalise(self.data[
            (targets[:, None] - self.trend_offsets)[..., None], nodes[None, None], :])
        trend = trend.transpose(0, 2, 1, 3)
        observed = self._normalise(self.data[targets[:, None], nodes[None], :])
        shape = (len(indices), self.n_locations)
        arrays = (recent, trend, np.ones((*shape, 1), dtype=np.float32),
                  self.time_features[targets], observed,
                  np.broadcast_to(indices[:, None], shape), np.broadcast_to(nodes, shape))
        return tuple(torch.from_numpy(np.array(a, copy=True, order="C")) for a in arrays)

    def __getitem__(self, index):
        return tuple(value[0] for value in self.fetch_batch([index]))
