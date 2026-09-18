"""Reproducible complete sampling without a Python list of all samples."""
from __future__ import annotations
import torch
from torch.utils.data import Sampler, RandomSampler


def _mix(value):
    value = (value + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def permuted_block(index, count, key):
    """Keyed Feistel bijection + cycle walking: O(1) storage for block order.

    This is a pseudorandom block ordering, NOT a uniform draw from all count!
    permutations. Every block is nevertheless visited exactly once.
    """
    if count <= 1:
        return 0
    half = max(1, ((count - 1).bit_length() + 1) // 2)
    mask = (1 << half) - 1
    value = index
    while True:
        left, right = value >> half, value & mask
        for round_index in range(6):
            left, right = right, left ^ (_mix(right ^ _mix(key + round_index)) & mask)
        value = (left << half) | right
        if value < count:
            return value


class EpochShuffleSampler(Sampler):
    """Modes: exact legacy, global tensor/chunk conversion, bounded block shuffle.

    global is order-equivalent to legacy, but retains an O(N) int64 tensor.
    block uses O(block_size) memory and changes the distribution/order only,
    never the set or multiplicity of samples. Workers receive indices from the
    parent DataLoader; worker count cannot change this order.
    """
    def __init__(self, data_source, *, mode="global", seed=20, block_size=262144,
                 legacy_rng=False, index_chunk_size=4096):
        if mode not in ("legacy", "global", "block"):
            raise ValueError("shuffle mode must be legacy, global or block")
        if block_size < 1 or index_chunk_size < 1:
            raise ValueError("shuffle block/chunk size must be positive")
        self.size = len(data_source)
        self.mode, self.seed, self.block_size = mode, int(seed), int(block_size)
        self.legacy_rng, self.index_chunk_size = legacy_rng, index_chunk_size
        self.epoch = 0

    def __len__(self):
        return self.size

    def _indices(self, permutation, offset=0):
        for start in range(0, len(permutation), self.index_chunk_size):
            for index in permutation[start:start+self.index_chunk_size].tolist():
                yield offset + index

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1
        if self.legacy_rng:
            # Reproduce old DataLoader base_seed then RandomSampler seed draws.
            # The actual worker loader uses a separate generator, so persistent
            # workers cannot alter subsequent epoch permutations.
            torch.empty((), dtype=torch.int64).random_()
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = _mix(self.seed + epoch) & ((1 << 63) - 1)
        generator = torch.Generator().manual_seed(seed)
        if self.mode == "legacy":
            yield from RandomSampler(range(self.size), generator=generator)
        elif self.mode == "global":
            yield from self._indices(torch.randperm(self.size, generator=generator))
        else:
            blocks = (self.size + self.block_size - 1) // self.block_size
            for position in range(blocks):
                block = permuted_block(position, blocks, seed)
                start = block * self.block_size
                length = min(self.block_size, self.size - start)
                local_generator = torch.Generator().manual_seed(_mix(seed ^ block) & ((1 << 63) - 1))
                yield from self._indices(torch.randperm(length, generator=local_generator), start)
