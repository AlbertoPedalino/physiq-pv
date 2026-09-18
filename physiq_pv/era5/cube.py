"""Geographical score mapping and frame-local morphology; independent of STGAN."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np
from scipy import ndimage


@dataclass
class CubeGrid:
    latitudes: np.ndarray  # descending, north -> south
    longitudes: np.ndarray  # ascending, west -> east
    rows: np.ndarray  # one entry per model location, in model order
    cols: np.ndarray
    spacing: float = 0.5

    def __post_init__(self):
        self.latitudes = np.asarray(self.latitudes, dtype=float)
        self.longitudes = np.asarray(self.longitudes, dtype=float)
        raw_rows, raw_cols = np.asarray(self.rows), np.asarray(self.cols)
        if any(a.ndim != 1 or not len(a) or not np.isfinite(a).all()
               for a in (self.latitudes, self.longitudes, raw_rows, raw_cols)):
            raise ValueError("Grid axes and mappings must be finite nonempty vectors")
        if not np.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError("Positive spacing required")
        if not np.allclose(np.diff(self.latitudes), -self.spacing) or not np.allclose(np.diff(self.longitudes), self.spacing):
            raise ValueError("Require regular north-to-south / west-to-east axes")
        if np.any(np.abs(self.latitudes) > 90) or np.any(np.abs(self.longitudes) > 180):
            raise ValueError("Invalid geographical coordinates")
        self.rows, self.cols = raw_rows.astype(np.int64), raw_cols.astype(np.int64)
        if not np.array_equal(raw_rows, self.rows) or not np.array_equal(raw_cols, self.cols):
            raise ValueError("Grid row/col must be integers")
        if self.rows.shape != self.cols.shape or np.any(self.rows < 0) or np.any(self.cols < 0) or np.any(self.rows >= self.shape[0]) or np.any(self.cols >= self.shape[1]):
            raise ValueError("Invalid location -> grid mapping")
        if len(np.unique(self.rows * self.shape[1] + self.cols)) != len(self.rows):
            raise ValueError("Duplicate location in geographical cube")

    @property
    def shape(self):
        return len(self.latitudes), len(self.longitudes)

    @property
    def valid_mask(self):
        valid = np.zeros(self.shape, dtype=bool)
        valid[self.rows, self.cols] = True
        return valid

    @property
    def cell_area_km2(self):
        north = np.deg2rad(np.minimum(90, self.latitudes + self.spacing / 2))
        south = np.deg2rad(np.maximum(-90, self.latitudes - self.spacing / 2))
        areas = 6371.0088**2 * np.deg2rad(self.spacing) * (np.sin(north) - np.sin(south))
        return np.broadcast_to(areas[:, None], self.shape)

    @classmethod
    def from_locations(cls, latitude, longitude, *, area=None, spacing=0.5):
        latitude, longitude = np.asarray(latitude), np.asarray(longitude)
        north, west, south, east = area or (latitude.max(), longitude.min(), latitude.min(), longitude.max())
        if not (north >= south and east >= west):
            raise ValueError("Invalid area")
        nr, nc = (north - south) / spacing, (east - west) / spacing
        if not np.allclose([nr, nc], np.rint([nr, nc])):
            raise ValueError("Area must align with the grid")
        rows, cols = (north - latitude) / spacing, (longitude - west) / spacing
        if not np.allclose(rows, np.rint(rows), atol=1e-5) or not np.allclose(cols, np.rint(cols), atol=1e-5):
            raise ValueError("Locations are off the geographical lattice")
        return cls(north - np.arange(round(nr) + 1) * spacing,
                   west + np.arange(round(nc) + 1) * spacing,
                   np.rint(rows), np.rint(cols), spacing)

    def frame(self, scores):
        scores = np.asarray(scores)
        if scores.shape != self.rows.shape:
            raise ValueError("Scores must follow the grid's location order")
        result = np.full(self.shape, np.nan, dtype=np.result_type(scores.dtype, np.float32))
        result[self.rows, self.cols] = scores
        return result

    def to_dict(self):
        return {"latitudes": self.latitudes.tolist(), "longitudes": self.longitudes.tolist(),
                "rows": self.rows.tolist(), "cols": self.cols.tolist(), "spacing": self.spacing}


def score_cube_chunks(scores, grid: CubeGrid, chunk_size=32):
    if chunk_size < 1 or scores.ndim != 2 or scores.shape[1] != len(grid.rows):
        raise ValueError("Expected scores (T,N) and positive chunk_size")
    for start in range(0, len(scores), chunk_size):
        block = np.full((min(chunk_size, len(scores)-start), *grid.shape), np.nan,
                        dtype=np.result_type(scores.dtype, np.float32))
        block[:, grid.rows, grid.cols] = scores[start:start+len(block)]
        yield start, block


def disk_percentile(scores, percentile, directory, chunk_size=32):
    """Exact global linear percentile, using an on-disk 1D partition workspace.

    RAM explicitly allocated is O(chunk*N). NumPy partitions the contiguous
    memmap in place; operating-system file cache can still use free RAM.
    """
    if not 0 <= percentile <= 100 or chunk_size < 1:
        raise ValueError("Invalid percentile/chunk_size")
    count = sum(int(np.isfinite(scores[s:s+chunk_size]).sum()) for s in range(0, len(scores), chunk_size))
    if count == 0:
        raise ValueError("No finite scores for percentile")
    with tempfile.TemporaryDirectory(prefix="percentile_", dir=directory) as temporary:
        values = np.memmap(Path(temporary)/"values.bin", mode="w+", dtype=np.float64, shape=(count,))
        try:
            cursor = 0
            for start in range(0, len(scores), chunk_size):
                block = np.asarray(scores[start:start+chunk_size])
                finite = block[np.isfinite(block)]
                values[cursor:cursor+len(finite)] = finite
                cursor += len(finite)
            position = (count - 1) * percentile / 100
            low, high = int(np.floor(position)), int(np.ceil(position))
            values.partition((low, high))
            return float(values[low] + (values[high] - values[low]) * (position-low))
        finally:
            values._mmap.close()


def morphology(frame, threshold, *, kernel_size=3, opening_iterations=1, closing_iterations=1):
    if kernel_size < 1 or kernel_size % 2 == 0 or min(opening_iterations, closing_iterations) < 0:
        raise ValueError("Require odd positive kernel and nonnegative iteration counts")
    valid = np.isfinite(frame)
    raw = valid & (frame > threshold)  # strict threshold: ties may reduce selected fraction
    structure = np.ones((kernel_size, kernel_size), dtype=bool)
    opened = (ndimage.binary_opening(raw, structure, iterations=opening_iterations, mask=valid)
              if opening_iterations else raw.copy()) & valid
    closed = (ndimage.binary_closing(opened, structure, iterations=closing_iterations, mask=valid)
              if closing_iterations else opened.copy()) & valid
    return raw, opened, closed


def spatial_clusters(frame, binary, grid: CubeGrid, *, min_cells=1):
    """8-connected components. Bboxes use cell-centre coordinates."""
    labels, _ = ndimage.label(binary & np.isfinite(frame), structure=np.ones((3, 3)))
    result = []
    for label, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        rr, cc = np.nonzero(labels[slices] == label)
        rr, cc = rr + slices[0].start, cc + slices[1].start
        if len(rr) < min_cells:
            labels[rr, cc] = 0
            continue
        areas = grid.cell_area_km2[rr, cc]
        values = frame[rr, cc]
        result.append({"label": label, "flat_cells": rr * grid.shape[1] + cc,
                       "cells": len(rr), "area_km2": float(areas.sum()),
                       "centroid_lat": float(np.average(grid.latitudes[rr], weights=areas)),
                       "centroid_lon": float(np.average(grid.longitudes[cc], weights=areas)),
                       "north": float(grid.latitudes[rr].max()), "south": float(grid.latitudes[rr].min()),
                       "west": float(grid.longitudes[cc].min()), "east": float(grid.longitudes[cc].max()),
                       "mean_score": float(values.mean(dtype=np.float64)), "max_score": float(values.max())})
    return labels, result
