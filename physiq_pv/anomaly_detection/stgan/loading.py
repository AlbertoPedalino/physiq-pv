"""DataLoader setup shared by training, inference and performance tests."""
from __future__ import annotations

import mmap
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler, SequentialSampler

from .data import ContextArray

RELEASE_EVERY = 16  # batches a loading process reads between two releases of its file pages


def release_pages(*arrays):
    """Detach from this process the pages of the read-only memory-mapped files it has read.

    The pages stay in the page cache of the system and are read again from there when they are needed, so
    the values read are the same; they only stop counting in the memory of the process (its RSS), which the
    out-of-memory killer of Linux reads to choose the process it kills. Maps written in place (modes r+ and
    w+, shared with the file) keep what was written, which is in the page cache of the file; copy-on-write
    maps (mode c) would lose it and are left alone, as are other arrays and systems without madvise (Windows).
    """
    for array in arrays:
        if isinstance(array, ContextArray):
            release_pages(array.prefix, array.test)
        elif isinstance(array, np.memmap) and array.mode in ("r", "r+", "w+") and hasattr(mmap, "MADV_DONTNEED"):
            mapping = getattr(array, "_mmap", None)
            if mapping is not None and not mapping.closed:
                mapping.madvise(mmap.MADV_DONTNEED)


class BatchedWindows(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.read = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        batch = self.dataset.fetch_batch(indices)
        # The batch holds its own copies: the file pages read for it are no longer needed by this process.
        self.read += 1
        if self.read % RELEASE_EVERY == 0:
            release_pages(self.dataset.data, getattr(self.dataset, "input_data", None))
        return batch


def already_batched(batch):
    return batch


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def close_loader(loader):
    """Release persistent workers before closing/removing their disk mappings."""
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()
        loader._iterator = None


def make_loader(dataset, *, batch_size, sampler=None, shuffle=False,
                num_workers=0, persistent_workers=True, prefetch_factor=2,
                pin_memory=True, device="cpu", generator=None, vectorized=True):
    options = dict(num_workers=num_workers,
                   pin_memory=bool(pin_memory and torch.device(device).type == "cuda"),
                   worker_init_fn=seed_worker, generator=generator)
    if num_workers:
        options.update(persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)
    if vectorized:
        if sampler is not None and shuffle:
            raise ValueError("sampler and shuffle are mutually exclusive")
        if sampler is None:
            sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        return DataLoader(BatchedWindows(dataset), batch_size=None,
                          sampler=BatchSampler(sampler, batch_size, drop_last=False),
                          collate_fn=already_batched, **options)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle,
                      drop_last=False, **options)
