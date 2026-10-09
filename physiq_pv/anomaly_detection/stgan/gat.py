"""Global GAT Generator with the original patch Discriminator and reductions."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .graph import GATGRU, TwoLayerGAT
from .model import ConvGRU, STGANGenerator, STGANDiscriminator, STGAN, masked_cell_mean


class STGANGATGenerator(STGANGenerator):
    def __init__(self, n_features, hidden_size, n_layers, cnn_channels, cnn_layers,
                 edge_index, n_nodes, recent_steps=1, gat_hidden_dim=16, gat_heads=4,
                 gat_layers=2, time_feature_size=31, dropout_enabled=True, dropout_p=.2,
                 trend_chunk_size=256, gat_recurrence="pointwise"):
        # Keep trend, calendar, dropouts and the literal 1x1 output projection.
        super().__init__(n_features, hidden_size, n_layers, cnn_channels, cnn_layers,
                         time_feature_size, 1, dropout_enabled, dropout_p)
        if any(type(x) is not int or x < 1 for x in (n_nodes, recent_steps)):
            raise ValueError("n_nodes and recent_steps must be positive integers.")
        self.n_nodes, self.recent_steps = n_nodes, recent_steps
        if type(trend_chunk_size) is not int or trend_chunk_size < 1:
            raise ValueError("trend_chunk_size must be a positive integer.")
        self.trend_chunk_size = trend_chunk_size
        if gat_recurrence not in ("pointwise", "gated"):
            raise ValueError("gat_recurrence must be pointwise or gated.")
        self.gat_recurrence = gat_recurrence
        if gat_recurrence == "gated":
            # The GCGRU of the paper with attention in place of the graph convolution: the
            # spatial operator sits inside the gates and acts on the input and on the
            # state at every recent step. gat_hidden_dim and gat_layers are not used.
            self.recent_encoder = GATGRU(n_features, cnn_channels, cnn_layers, gat_heads, edge_index)
        else:
            self.recent_encoder = TwoLayerGAT(n_features, cnn_channels, gat_hidden_dim,
                                             gat_heads, edge_index, gat_layers)
            # Apply GRU gates even for one recent step, as the CNN and reference do.
            # GAT owns the spatial mixing, so the temporal gates are pointwise.
            self.recent_temporal = ConvGRU(cnn_channels, cnn_channels, cnn_layers, kernel_size=1)

    def _trend_last(self, values):
        sequence, _ = self.trend_encoder(values)
        # Own only the last state; a view would retain the entire sequence per chunk.
        return sequence[:, -1].clone()

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

    def encode(self, recent, trend, mask, time_features):
        """forward up to its dropouts. Nothing here is stochastic, so MC scoring
        computes it once per batch; forward itself is unchanged and stays the reference."""
        batch, steps, nodes, features = recent.shape
        if self.gat_recurrence == "gated":
            spatial = self.recent_encoder(recent, mask)
        else:
            spatial = self.recent_encoder(recent.reshape(batch * steps, nodes, features))
            spatial = spatial.reshape(batch, steps, nodes, spatial.shape[-1])
            spatial = self.recent_temporal(spatial.permute(0, 1, 3, 2)[..., None],
                                           mask.transpose(1, 2)[..., None]).squeeze(-1).transpose(1, 2)
        temporal = self.encode_trend(trend.reshape(batch * nodes, trend.shape[2], features))
        calendar = self.time_projection(time_features)
        if calendar.ndim == 2:  # Shared calendar [B,F]; per-node features are [B,N,F].
            calendar = calendar[:, None].expand(-1, nodes, -1)
        elif calendar.shape[:2] != (batch, nodes):
            raise ValueError("Per-node time features must be [B,N,F].")
        return spatial, temporal, calendar

    def decode(self, spatial, temporal, calendar, mask):
        """The rest of forward: its three dropouts, in the same order, and the output."""
        spatial = self.spatial_dropout(spatial)
        temporal = self.temporal_dropout(temporal).reshape(*spatial.shape[:2], -1)
        fused = self.fusion_dropout(torch.cat((spatial, temporal, calendar), dim=-1))
        predicted = self.output_projection(fused.transpose(1, 2)[..., None]).squeeze(-1).transpose(1, 2)
        return torch.where(mask.bool(), predicted, 0.0)

    def forward(self, recent, trend, mask, time_features):
        if recent.ndim != 4 or recent.shape[1:3] != (self.recent_steps, self.n_nodes):
            raise ValueError("GAT recent must be [B,recent_steps,N,F] for the fixed global graph.")
        batch, steps, nodes, features = recent.shape
        if (trend.ndim != 4 or trend.shape[:2] != (batch, nodes)
                or trend.shape[-1] != features or mask.shape != (batch, nodes, 1)):
            raise ValueError("GAT trend/mask must preserve the global node axis.")
        if self.gat_recurrence == "gated":
            spatial = self.recent_encoder(recent, mask)
        else:
            spatial = self.recent_encoder(recent.reshape(batch * steps, nodes, features))
            channels = spatial.shape[-1]
            spatial = spatial.reshape(batch, steps, nodes, channels)
            spatial = self.recent_temporal(spatial.permute(0, 1, 3, 2)[..., None],
                                           mask.transpose(1, 2)[..., None]).squeeze(-1).transpose(1, 2)
        spatial = self.spatial_dropout(spatial)
        temporal = self.encode_trend(trend.reshape(batch * nodes, trend.shape[2], features))
        temporal = self.temporal_dropout(temporal).reshape(batch, nodes, -1)
        calendar = self.time_projection(time_features)
        if calendar.ndim == 2:  # Shared calendar [B,F]; per-node features are [B,N,F].
            calendar = calendar[:, None].expand(-1, nodes, -1)
        elif calendar.shape[:2] != (batch, nodes):
            raise ValueError("Per-node time features must be [B,N,F].")
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
                 trend_chunk_size=256, gat_recurrence="pointwise"):
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
        self._patch_plan = None  # Static chunk structures of the last batch shape; see patch_plan.
        self.generator = STGANGATGenerator(n_features, hidden_size, n_layers, cnn_channels,
            cnn_layers, edge_index, len(indices), recent_steps, gat_hidden_dim, gat_heads,
            gat_layers, time_feature_size, dropout_enabled, dropout_p, trend_chunk_size,
            gat_recurrence)
        self.discriminator = STGANDiscriminator(n_features, hidden_size, cnn_channels,
                                               cnn_layers, patch_size, kernel_size)

    def patch_plan(self, batch, steps, nodes, device):
        """Per chunk, what depends on the grid and on the batch shape only: the flattened
        (batch,node) ids of its centers, their timestamp, the nodes of every patch (holes and
        borders clamped to node 0) and which of those cells exist.

        Built once and reused while shape, chunk size and device stay the same: nothing here
        depends on a value of the batch. Returns the recent-step index and the chunks.
        """
        key = (batch, steps, nodes, self.discriminator_chunk_size, torch.device(device))
        if self._patch_plan is None or self._patch_plan[0] != key:
            chunks = []
            for start in range(0, batch * nodes, self.discriminator_chunk_size):
                ids = torch.arange(start, min(start + self.discriminator_chunk_size, batch * nodes), device=device)
                times, centers = ids // nodes, ids % nodes
                indices = self.node_indices[centers]
                chunks.append((ids, times, indices.clamp_min(0).flatten(1), (indices >= 0)[:, None]))
            self._patch_plan = (key, torch.arange(steps, device=device)[None, :, None], chunks)
        return self._patch_plan[1:]

    def gather_patch(self, values, times, safe, valid):
        """Patches [chunk,F,size,size] of values [B,N,F] around the centers of one chunk; absent cells are 0."""
        size = self.node_indices.shape[-1]
        patch = values[times[:, None], safe].reshape(-1, size, size, values.shape[-1]).permute(0, 3, 1, 2)
        return torch.where(valid, patch, 0.0)

    def patch_inputs(self, recent, observed):
        """Yield, per chunk, the D inputs that no weight changes: the plan of the chunk, the
        history patches and the observed patches. Training gathers them once per batch."""
        batch, steps, nodes, features = recent.shape
        size = self.node_indices.shape[-1]
        step_index, chunks = self.patch_plan(batch, steps, nodes, recent.device)
        for ids, times, safe, valid in chunks:
            history = recent[times[:, None, None], step_index,
                             safe[:, None]].reshape(-1, steps, size, size, features).permute(0, 1, 4, 2, 3)
            yield (ids, times, safe, valid, torch.where(valid[:, None], history, 0.0),
                   self.gather_patch(observed, times, safe, valid))

    def patch_batches(self, recent, observed, predicted):
        """Yield original D inputs in flattened (batch,node) order, O(chunk) RAM."""
        for ids, times, safe, valid, history, real_patch in self.patch_inputs(recent, observed):
            yield ids, history, real_patch, self.gather_patch(predicted, times, safe, valid), valid

    def score_draw(self, recent, trend, mask, calendar, observed, *, share_history=True):
        predicted = self.generator(recent, trend, mask, calendar).float()
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

    def monitoring_outputs(self, recent, trend, mask, calendar, observed):
        """One forward for monitoring, per center in (batch, node) order: the two score
        components and D's penultimate activations of the observation and of its reconstruction."""
        predicted = self.generator(recent, trend, mask, calendar).float()
        parts = []
        for _, history, real_patch, fake_patch, valid in self.patch_batches(recent, observed, predicted):
            historical = self.discriminator.encode_history(history, valid)
            real = self.discriminator.penultimate(historical, real_patch, valid)
            fake = self.discriminator.penultimate(historical, fake_patch, valid)
            difference = (self.discriminator.score_from_penultimate(real)
                          - self.discriminator.score_from_penultimate(fake))
            parts.append((masked_cell_mean((fake_patch - real_patch).square(), valid), difference.squeeze(1),
                          real, fake))
        return tuple(torch.cat(values) for values in zip(*parts))

    def reconstructed_cells(self, recent, trend, mask, calendar, observed):
        """Observed and generated values [batch*node,feature], in flattened (batch,node) order."""
        generated = self.generator(recent, trend, mask, calendar).float()
        return observed.reshape(-1, observed.shape[-1]), generated.reshape(-1, generated.shape[-1])

    def reconstruction_valid(self, mask):
        """Per (batch,node) point: its patch, the mask of its reconstruction error, has at least one cell."""
        return (self.node_indices >= 0).flatten(1).any(dim=1).repeat(mask.shape[0])

    def score_draws(self, recent, trend, mask, calendar, observed, samples, *, share_history=True):
        """Stack of `samples` score_draw results, with the draw-independent work done once.

        Dropout sits after G's encoders and D has none: G's encoders, D's history
        encoding and D's score of the observation are the same in every draw. Each
        draw applies the dropouts in forward's order, so the values are those of
        repeated score_draw calls.
        """
        if not share_history:
            return torch.stack([self.score_draw(recent, trend, mask, calendar, observed, share_history=False)
                                for _ in range(samples)])
        encoded = self.generator.encode(recent, trend, mask, calendar)
        center = self.node_indices.shape[-1] // 2
        chunks = []
        for _, times, safe, valid, history, real_patch in self.patch_inputs(recent, observed):
            historical = self.discriminator.encode_history(history, valid)
            chunks.append((times, safe, real_patch, valid, historical,
                           self.discriminator.score_current(historical, real_patch, valid)))
        draws = []
        for _ in range(samples):
            predicted = self.generator.decode(*encoded, mask).float()
            parts = []
            for times, safe, real_patch, valid, historical, real in chunks:
                # Same gather as patch_batches, for the generated values only.
                fake_patch = self.gather_patch(predicted, times, safe, valid)
                fake = self.discriminator.score_current(historical, fake_patch, valid)
                errors = (fake_patch - real_patch).square()
                parts.append(torch.cat((masked_cell_mean(errors, valid)[:, None], real - fake,
                                        errors[:, :, center, center]), dim=1))
            draws.append(torch.cat(parts))
        return torch.stack(draws)

    def components(self, recent, trend, mask, time_features, observed, *, share_history=True):
        """Diagnostic API: predictions/errors [B,N,F], D scores [B,N,1].

        Production scoring uses score_draw to also preserve the original
        patch-mean reconstruction component, with bounded D working memory.
        """
        predicted = self.generator(recent, trend, mask, time_features).float()
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
