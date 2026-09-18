"""Bounded-memory raw scoring and global (never per-chunk) normalization."""
from __future__ import annotations
from pathlib import Path
import tempfile
import numpy as np
import torch
from .loading import make_loader, close_loader
from .model import masked_cell_mean


class ScoreStore:
    """Own RAM/memmap outputs; anonymous disk storage lives until result.close()."""
    def __init__(self, *, root=None, backend="memory", memory_limit_mb=1024):
        self.root = Path(root) if root is not None else None
        self.backend = backend
        self.memory_limit_mb = memory_limit_mb
        self.arrays = []
        self._temporary = None

    def directory(self):
        if self.root is None:
            self._temporary = tempfile.TemporaryDirectory(prefix="stgan_scores_")
            self.root = Path(self._temporary.name)
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def allocate(self, name, shape, dtype=np.float32, *, disk=False):
        if disk or self.backend == "memmap":
            array = np.lib.format.open_memmap(self.directory() / (name + ".npy"),
                mode="w+", dtype=dtype, shape=shape)
        else:
            array = np.empty(shape, dtype=dtype)
        self.arrays.append(array)
        return array

    def close(self):
        for array in self.arrays:
            mapping = getattr(array, "_mmap", None)
            if mapping is not None and not mapping.closed:
                array.flush()
                mapping.close()
        self.arrays.clear()
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None


def component_range(values, chunk_size=65536):
    flat = values.reshape(-1)
    minimum, maximum = np.inf, -np.inf
    for start in range(0, flat.size, chunk_size):
        block = np.asarray(flat[start:start+chunk_size])
        finite = block[np.isfinite(block)]
        if len(finite):
            minimum = min(minimum, np.min(finite))
            maximum = max(maximum, np.max(finite))
    if not np.isfinite(minimum):
        raise ValueError("STGAN anomaly component has no finite values.")
    minimum = float(minimum)
    # Preserve NumPy's float32 subtraction/rounding from the original function.
    scale = float(np.asarray(maximum, dtype=values.dtype) - minimum)
    return minimum, scale if scale >= 1e-8 else 1.0


def normalize_scores(generator, discriminator, store, *, chunk_size=65536):
    generator_range = component_range(generator, chunk_size)
    discriminator_range = component_range(discriminator, chunk_size)
    scores = store.allocate("test_scores", generator.shape)
    gf, df, sf = generator.reshape(-1), discriminator.reshape(-1), scores.reshape(-1)
    for start in range(0, sf.size, chunk_size):
        end = start + chunk_size
        sf[start:end] = ((np.asarray(gf[start:end]) - generator_range[0]) / generator_range[1]
                        + (np.asarray(df[start:end]) - discriminator_range[0]) / discriminator_range[1])
    if isinstance(scores, np.memmap):
        scores.flush()
    return scores, generator_range, discriminator_range


def score_components(model, dataset, *, batch_size, device, n_features,
                     storage="auto", output_dir=None, memory_limit_mb=1024,
                     loader_options=None, share_history=True):
    shape = (len(dataset.targets), dataset.n_locations)
    if storage not in ("auto", "memory", "memmap"):
        raise ValueError("score storage must be auto, memory or memmap")
    required = int(np.prod(shape)) * (n_features + 3) * 4
    backend = ("memmap" if required > memory_limit_mb * 1024**2 else "memory") if storage == "auto" else storage
    store = ScoreStore(root=output_dir, backend=backend, memory_limit_mb=memory_limit_mb)
    loader = None
    try:
        generator = store.allocate("generator_scores", shape)
        discriminator = store.allocate("discriminator_scores", shape)
        features = store.allocate("feature_scores", shape + (n_features,))
        loader = make_loader(dataset, batch_size=batch_size, shuffle=False,
                             device=device, **(loader_options or {}))
        model.eval()
        with torch.no_grad():
            for batch in loader:
                recent, trend, mask, calendar, observed = (x.to(device, non_blocking=True) for x in batch[:5])
                _, real, fake, errors = model.components(recent, trend, mask, calendar, observed,
                                                        share_history=share_history)
                center = dataset.grid.patch_size // 2
                # One D2H transfer, after all reductions; inputs/score math unchanged.
                packed = torch.cat((masked_cell_mean(errors, mask)[:, None], real-fake,
                                    errors[:, :, center, center]), dim=1).cpu().numpy()
                if not np.isfinite(packed).all():
                    raise FloatingPointError("Non-finite detector outputs; no partial ranking will be exported.")
                time, location = batch[-2].numpy(), batch[-1].numpy()
                generator[time, location] = packed[:, 0]
                discriminator[time, location] = packed[:, 1]
                features[time, location] = packed[:, 2:]
        for array in store.arrays:
            if isinstance(array, np.memmap):
                array.flush()
        return generator, discriminator, features, store
    except torch.cuda.OutOfMemoryError as exc:
        store.close()
        raise torch.cuda.OutOfMemoryError(
            f"STGAN scoring ran out of CUDA memory with score_batch_size={batch_size}. "
            "Reduce --score-batch-size (for example, halve it) and retry scoring. "
            "The training batch size and score definition are unchanged."
        ) from exc
    except BaseException:
        store.close()
        raise
    finally:
        close_loader(loader)
