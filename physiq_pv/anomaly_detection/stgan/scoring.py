"""Bounded-memory raw scoring and global (never per-chunk) normalization."""
from __future__ import annotations
from pathlib import Path
from contextlib import contextmanager
import tempfile
import numpy as np
import torch
from .loading import make_loader, close_loader
from .model import masked_cell_mean


@contextmanager
def scoring_mode(model, mc_dropout_enabled=False):
    """Only G's dropout is stochastic; restore every module even after failure."""
    states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        if mc_dropout_enabled:
            for module in model.generator.modules():
                if isinstance(module, torch.nn.modules.dropout._DropoutNd):
                    module.train()
        yield
    finally:
        for module, training in states:
            module.training = training


class ScoreStore:
    """Own RAM/memmap outputs; anonymous disk storage lives until result.close()."""
    def __init__(self, *, root=None, backend="memory", memory_limit_mb=1024):
        self.root = Path(root) if root is not None else None
        self.backend = backend
        self.memory_limit_mb = memory_limit_mb
        self.arrays = []
        self._temporary = None
        self._raw_arrays = []
        self._raw_temporary = None

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

    def allocate_raw(self, name, shape, *, save=False):
        """Keep raw draws only on request; otherwise own temporary components.

        Disk scratch stays on the output volume, not the system temp volume.
        Its lifetime ends after normalization (or on any scoring failure).
        The backend and peak memory estimate include raw arrays in both modes.
        """
        if save:
            return self.allocate(name, shape)
        if self.backend == "memmap":
            if self._raw_temporary is None:
                self._raw_temporary = tempfile.TemporaryDirectory(
                    prefix=".mc_raw_", dir=self.directory())
            array = np.lib.format.open_memmap(Path(self._raw_temporary.name)/(name+".npy"),
                mode="w+", dtype=np.float32, shape=shape)
        else:
            array = np.empty(shape, dtype=np.float32)
        self._raw_arrays.append(array)
        return array

    def discard_temporary_raw(self):
        """Invalidate temporary raw arrays after aggregation; keep debug exports."""
        for array in self._raw_arrays:
            mapping = getattr(array, "_mmap", None)
            if mapping is not None and not mapping.closed:
                mapping.close()
        self._raw_arrays.clear()
        if self._raw_temporary is not None:
            self._raw_temporary.cleanup()
            self._raw_temporary = None

    def close(self):
        self.discard_temporary_raw()
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
    """Calibration extrema (actual min/max, including constant components)."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    flat = values.reshape(-1)
    minimum, maximum = np.inf, -np.inf
    for start in range(0, flat.size, chunk_size):
        block = np.asarray(flat[start:start+chunk_size])
        if not np.isfinite(block).all():
            raise ValueError("Calibration components must be finite.")
        minimum = min(minimum, np.min(block))
        maximum = max(maximum, np.max(block))
    if not np.isfinite(minimum):
        raise ValueError("STGAN anomaly component has no finite values.")
    return float(minimum), float(maximum)


def fit_calibration_ranges(generator, discriminator, *, chunk_size=65536):
    """Fit shared ranges ONLY on calibration draws, before test scoring."""
    if generator.shape != discriminator.shape or generator.ndim not in (2, 3):
        raise ValueError("Expected matching (T,N) or (M,T,N) calibration components")
    r_min, r_max = component_range(generator, chunk_size)
    d_min, d_max = component_range(discriminator, chunk_size)
    return dict(r_min=r_min, r_max=r_max, d_min=d_min, d_max=d_max)


def _frozen_parameters(normalization):
    parameters = []
    for prefix in ("r", "d"):
        minimum, maximum = (float(normalization[prefix+suffix]) for suffix in ("_min", "_max"))
        if not np.isfinite([minimum, maximum]).all() or maximum < minimum:
            raise ValueError("Invalid calibration normalization extrema")
        scale = maximum - minimum
        parameters.append((minimum, scale if scale >= 1e-8 else 1.0))
    return parameters


def normalize_scores(generator, discriminator, store, *, normalization, chunk_size=65536):
    generator_range, discriminator_range = _frozen_parameters(normalization)
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
                     loader_options=None, share_history=True,
                     mc_dropout_enabled=False, mc_samples=20, save_raw_mc=False):
    """Return components valid until store cleanup; raw persistence is opt-in."""
    if type(save_raw_mc) is not bool:
        raise ValueError("save_raw_mc must be a boolean")
    if type(mc_samples) is not int or mc_samples < 1:
        raise ValueError("mc_samples must be a positive integer")
    samples = mc_samples if mc_dropout_enabled else 1
    shape = (len(dataset.targets), dataset.n_locations)
    if storage not in ("auto", "memory", "memmap"):
        raise ValueError("score storage must be auto, memory or memmap")
    required = int(np.prod(shape)) * (n_features + 2 * samples + 4) * 4
    backend = ("memmap" if required > memory_limit_mb * 1024**2 else "memory") if storage == "auto" else storage
    store = ScoreStore(root=output_dir, backend=backend, memory_limit_mb=memory_limit_mb)
    loader = None
    try:
        raw_shape = (samples, *shape) if mc_dropout_enabled else shape
        generator = store.allocate_raw("generator_scores", raw_shape, save=save_raw_mc)
        discriminator = store.allocate_raw("discriminator_scores", raw_shape, save=save_raw_mc)
        generator_samples = generator if mc_dropout_enabled else generator[None]
        discriminator_samples = discriminator if mc_dropout_enabled else discriminator[None]
        features = store.allocate("feature_scores", shape + (n_features,))
        loader = make_loader(dataset, batch_size=batch_size, shuffle=False,
                             device=device, **(loader_options or {}))
        with scoring_mode(model, mc_dropout_enabled), torch.no_grad():
            for batch in loader:
                recent, trend, mask, calendar, observed = (x.to(device, non_blocking=True) for x in batch[:5])
                center = dataset.grid.patch_size // 2
                draws = []
                for _ in range(samples):
                    # Each draw recomputes all of G, including both encoders.
                    if getattr(model, "global_graph", False):
                        draws.append(model.score_draw(recent, trend, mask, calendar, observed,
                                                      share_history=share_history))
                        continue
                    _, real, fake, errors = model.components(recent, trend, mask, calendar, observed,
                                                            share_history=share_history)
                    draws.append(torch.cat((masked_cell_mean(errors, mask)[:, None], real-fake,
                                            errors[:, :, center, center]), dim=1))
                # One D2H transfer per batch; only reduced outputs retain the MC axis.
                packed = torch.stack(draws).cpu().numpy()
                if not np.isfinite(packed).all():
                    raise FloatingPointError("Non-finite detector outputs; no partial ranking will be exported.")
                time, location = batch[-2].numpy().reshape(-1), batch[-1].numpy().reshape(-1)
                generator_samples[:, time, location] = packed[:, :, 0]
                discriminator_samples[:, time, location] = packed[:, :, 1]
                features[time, location] = packed[:, :, 2:].mean(axis=0)
        for array in store.arrays + store._raw_arrays:
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


def normalize_mc_scores(generator, discriminator, store, *, normalization, chunk_size=65536):
    """Apply frozen calibration ranges to every draw, without fitting or clipping.

    ``normalization`` contains r_min/r_max/d_min/d_max from calibration only.
    Population std (ddof=0)
    is accumulated in float64 using Welford; M=1 has exactly zero uncertainty.
    RAM is O(chunk_size), irrespective of M or the number of test locations.
    """
    if generator.shape != discriminator.shape or generator.ndim not in (2, 3):
        raise ValueError("Expected matching (T,N) or (M,T,N) score components")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    gs = generator[None] if generator.ndim == 2 else generator
    ds = discriminator[None] if discriminator.ndim == 2 else discriminator
    samples, *shape = gs.shape
    gr, dr = _frozen_parameters(normalization)
    mean = store.allocate("anomaly_mean", tuple(shape))
    std = store.allocate("anomaly_std", tuple(shape))
    gmean = store.allocate("generator_mean", tuple(shape))
    dmean = store.allocate("discriminator_mean", tuple(shape))
    gf, df = gs.reshape(samples, -1), ds.reshape(samples, -1)
    mf, sf = mean.reshape(-1), std.reshape(-1)
    for start in range(0, mf.size, chunk_size):
        end = min(start + chunk_size, mf.size)
        average = np.zeros(end-start, np.float64)
        m2 = np.zeros_like(average)
        ga, da = np.zeros_like(average), np.zeros_like(average)
        for index in range(samples):
            g, d = np.asarray(gf[index, start:end]), np.asarray(df[index, start:end])
            score = (g-gr[0])/gr[1] + (d-dr[0])/dr[1]
            delta = score-average
            average += delta/(index+1)
            m2 += delta*(score-average)
            ga += g
            da += d
        mf[start:end] = average
        sf[start:end] = np.sqrt(np.maximum(m2/samples, 0))
        gmean.reshape(-1)[start:end] = ga/samples
        dmean.reshape(-1)[start:end] = da/samples
    for array in (mean, std, gmean, dmean):
        if isinstance(array, np.memmap):
            array.flush()
    return mean, std, gmean, dmean, gr, dr
