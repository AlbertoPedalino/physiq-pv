"""Global GAT Generator with the original patch Discriminator and reductions."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .graph import TwoLayerGAT
from .model import ConvGRU, STGANGenerator, STGANDiscriminator, STGAN, masked_cell_mean


class STGANGATGenerator(STGANGenerator):
    def __init__(self, n_features, hidden_size, n_layers, cnn_channels, cnn_layers,
                 edge_index, n_nodes, recent_steps=1, gat_hidden_dim=16, gat_heads=4,
                 gat_layers=2, time_feature_size=31, dropout_enabled=True, dropout_p=.2,
                 trend_chunk_size=256):
        # Keep trend, calendar, dropouts and the literal 1x1 output projection.
        super().__init__(n_features, hidden_size, n_layers, cnn_channels, cnn_layers,
                         time_feature_size, 1, dropout_enabled, dropout_p)
        if any(type(x) is not int or x < 1 for x in (n_nodes, recent_steps)):
            raise ValueError("n_nodes and recent_steps must be positive integers.")
        self.n_nodes, self.recent_steps = n_nodes, recent_steps
        if type(trend_chunk_size) is not int or trend_chunk_size < 1:
            raise ValueError("trend_chunk_size must be a positive integer.")
        self.trend_chunk_size = trend_chunk_size
        self.recent_encoder = TwoLayerGAT(n_features, cnn_channels, gat_hidden_dim,
                                         gat_heads, edge_index, gat_layers)
        # Existing GRU equations, pointwise in space: GAT owns all spatial mixing.
        self.recent_temporal = (ConvGRU(cnn_channels, cnn_channels, cnn_layers, kernel_size=1)
                                if recent_steps > 1 else None)

    def _trend_last(self, values):
        sequence, _ = self.trend_encoder(values)
        return sequence[:, -1]

    def encode_trend(self, values):
        if len(values) <= self.trend_chunk_size:
            return self._trend_last(values)
        # LSTM has no inter-node state or internal dropout. Chunking preserves
        # its equations; checkpoint only recomputes activations during backward.
        chunks = []
        for block in values.split(self.trend_chunk_size):
            last = (checkpoint(self._trend_last, block, use_reentrant=False)
                    if torch.is_grad_enabled() else self._trend_last(block))
            chunks.append(last)
        return torch.cat(chunks)

    def forward(self, recent, trend, mask, time_features):
        if recent.ndim != 4 or recent.shape[1:3] != (self.recent_steps, self.n_nodes):
            raise ValueError("GAT recent must be [B,recent_steps,N,F] for the fixed global graph.")
        batch, steps, nodes, features = recent.shape
        if (trend.ndim != 4 or trend.shape[:2] != (batch, nodes)
                or trend.shape[-1] != features or mask.shape != (batch, nodes, 1)):
            raise ValueError("GAT trend/mask must preserve the global node axis.")
        spatial = self.recent_encoder(recent.reshape(batch * steps, nodes, features))
        channels = spatial.shape[-1]
        spatial = spatial.reshape(batch, steps, nodes, channels)
        if self.recent_temporal is not None:
            spatial = self.recent_temporal(spatial.permute(0, 1, 3, 2)[..., None],
                                          mask.transpose(1, 2)[..., None]).squeeze(-1).transpose(1, 2)
        else:
            spatial = spatial[:, 0]
        spatial = self.spatial_dropout(spatial)
        temporal = self.encode_trend(trend.reshape(batch * nodes, trend.shape[2], features))
        temporal = self.temporal_dropout(temporal).reshape(batch, nodes, -1)
        calendar = self.time_projection(time_features)[:, None].expand(-1, nodes, -1)
        fused = self.fusion_dropout(torch.cat((spatial, temporal, calendar), dim=-1))
        predicted = self.output_projection(fused.transpose(1, 2)[..., None]).squeeze(-1).transpose(1, 2)
        return torch.where(mask.bool(), predicted, 0.0)


class STGANGAT(STGAN):
    """D is unchanged; bounded patch adapters sit outside the global Generator."""
    global_graph = True

    def __init__(self, *, n_features, edge_index, node_indices, hidden_size=64, n_layers=2,
                 cnn_channels=32, cnn_layers=2, patch_size=3, time_feature_size=31,
                 kernel_size=3, dropout_enabled=True, dropout_p=.2, recent_steps=1,
                 gat_hidden_dim=16, gat_heads=4, gat_layers=2, discriminator_chunk_size=256,
                 trend_chunk_size=256):
        nn.Module.__init__(self)
        if type(discriminator_chunk_size) is not int or discriminator_chunk_size < 1:
            raise ValueError("discriminator_chunk_size must be a positive integer.")
        indices = torch.as_tensor(node_indices, dtype=torch.long)
        if (patch_size not in (1, 3, 5) or indices.ndim != 3 or not len(indices)
                or indices.shape[1:] != (patch_size, patch_size)):
            raise ValueError("Discriminator patch indices do not match patch_size.")
        if (indices.min() < -1 or indices.max() >= len(indices)
                or not torch.equal(indices[:, patch_size//2, patch_size//2],
                                   torch.arange(len(indices), device=indices.device))):
            raise ValueError("Patch indices must preserve every center and the global node order.")
        edges = torch.as_tensor(edge_index)
        if (edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2
                or not edges.shape[1] or edges.min() < 0 or edges.max() >= len(indices)):
            raise ValueError("edge_index must be int64 [2,E] over the global valid nodes.")
        self.register_buffer("node_indices", indices.clone())
        self.discriminator_chunk_size = discriminator_chunk_size
        self.generator = STGANGATGenerator(n_features, hidden_size, n_layers, cnn_channels,
            cnn_layers, edge_index, len(indices), recent_steps, gat_hidden_dim, gat_heads,
            gat_layers, time_feature_size, dropout_enabled, dropout_p, trend_chunk_size)
        self.discriminator = STGANDiscriminator(n_features, hidden_size, cnn_channels,
                                               cnn_layers, patch_size, kernel_size)

    def patch_batches(self, recent, observed, predicted):
        """Yield original D inputs in flattened (batch,node) order, O(chunk) RAM."""
        batch, steps, nodes, features = recent.shape
        size = self.node_indices.shape[-1]
        for start in range(0, batch * nodes, self.discriminator_chunk_size):
            ids = torch.arange(start, min(start + self.discriminator_chunk_size, batch * nodes),
                               device=recent.device)
            times, centers = ids // nodes, ids % nodes
            indices = self.node_indices[centers]
            valid = (indices >= 0)[:, None]
            safe = indices.clamp_min(0).flatten(1)
            history = recent[times[:, None, None], torch.arange(steps, device=recent.device)[None, :, None],
                             safe[:, None]].reshape(-1, steps, size, size, features).permute(0, 1, 4, 2, 3)
            def gather(values):
                patch = values[times[:, None], safe].reshape(-1, size, size, features).permute(0, 3, 1, 2)
                return torch.where(valid, patch, 0.0)
            yield ids, torch.where(valid[:, None], history, 0.0), gather(observed), gather(predicted), valid

    def score_draw(self, recent, trend, mask, calendar, observed, *, share_history=True):
        predicted = self.generator(recent, trend, mask, calendar)
        parts = []
        center = self.node_indices.shape[-1] // 2
        for _, history, real_patch, fake_patch, valid in self.patch_batches(recent, observed, predicted):
            real, fake = self.discriminator.score_pair(history, real_patch, fake_patch, valid) if share_history else (
                self.discriminator(torch.cat((history, real_patch[:, None]), 1), valid),
                self.discriminator(torch.cat((history, fake_patch[:, None]), 1), valid))
            errors = (fake_patch - real_patch).square()
            parts.append(torch.cat((masked_cell_mean(errors, valid)[:, None], real - fake,
                                    errors[:, :, center, center]), dim=1))
        return torch.cat(parts)

    def components(self, recent, trend, mask, time_features, observed, *, share_history=True):
        """Diagnostic API: predictions/errors [B,N,F], D scores [B,N,1].

        Production scoring uses score_draw to also preserve the original
        patch-mean reconstruction component, with bounded D working memory.
        """
        predicted = self.generator(recent, trend, mask, time_features)
        real_scores, fake_scores = [], []
        for _, history, real_patch, fake_patch, valid in self.patch_batches(recent, observed, predicted):
            real, fake = self.discriminator.score_pair(history, real_patch, fake_patch, valid) if share_history else (
                self.discriminator(torch.cat((history, real_patch[:, None]), 1), valid),
                self.discriminator(torch.cat((history, fake_patch[:, None]), 1), valid))
            real_scores.append(real)
            fake_scores.append(fake)
        shape = (*predicted.shape[:2], 1)
        errors = torch.where(mask.bool(), predicted - observed, 0.0).square()
        return predicted, torch.cat(real_scores).reshape(shape), torch.cat(fake_scores).reshape(shape), errors
