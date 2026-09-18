"""DataLoader setup shared by training, inference and performance tests."""
from __future__ import annotations

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler, SequentialSampler


class BatchedWindows(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self.dataset.fetch_batch(indices)


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
