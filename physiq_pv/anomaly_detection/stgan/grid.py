"""Validate geographical lattices and gather masked patches; no graph is built."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


@dataclass(frozen=True)
class SpatialGrid:
    node_indices: np.ndarray  # [location,row,column], -1 means absent
    valid_mask: np.ndarray
    row_indices: np.ndarray
    column_indices: np.ndarray
    metadata: dict

    @property
    def n_locations(self):
        return len(self.node_indices)

    @property
    def patch_size(self):
        return self.node_indices.shape[1]

    def location_frame(self, location_names):
        counts = self.valid_mask.sum(axis=(1, 2)).astype(int)
        return pd.DataFrame({
            "location": location_names, "grid_row": self.row_indices,
            "grid_column": self.column_indices, "n_valid_cells": counts,
            "complete_patch": counts == self.patch_size**2,
        })


def _angular_spacing(values):
    # Validate an angular lattice without calling it a metric 5 km grid.
    axis = np.unique(np.round(values, 5))
    if len(axis) < 2:
        raise ValueError("Cannot infer spacing from a single coordinate.")
    step = float(np.median(np.diff(axis)))
    if step < 1e-4:
        raise ValueError("Coordinates do not identify a regular angular axis.")
    return step


def _indices(x, y, spacing, tolerance):
    dx, dy = spacing
    col = np.rint((x - x.min()) / dx).astype(np.int64)
    row = np.rint((y.max() - y) / dy).astype(np.int64)
    residual = max(float(np.max(np.abs(x - (x.min() + col * dx)))),
                   float(np.max(np.abs(y - (y.max() - row * dy)))))
    if residual > tolerance:
        raise ValueError(f"Maximum lattice residual {residual:.6g} exceeds {tolerance}.")
    if len(set(zip(row.tolist(), col.tolist()))) != len(x):
        raise ValueError("Multiple locations map to the same grid cell.")
    if (int(row.max()) + 1) * (int(col.max()) + 1) > max(100, len(x) * 100):
        raise ValueError("Inferred lattice is implausibly sparse; specify the source CRS.")
    return row, col, residual


def build_spatial_grid(latitudes, longitudes, *, patch_size=3, grid_crs="auto",
                       grid_spacing=5000.0, grid_tolerance=25.0):
    """Verify a lattice before assigning fixed compass-oriented cells.

    Auto tests common projected CRSs for Piedmont, then a regular lat/lon
    lattice. Irregular clouds fail rather than being reshaped into images.
    Explicit projected CRSs must use metres.
    """
    if patch_size not in (1, 3, 5):
        raise ValueError("patch_size must be 1, 3 or 5.")
    lat, lon = np.asarray(latitudes, float), np.asarray(longitudes, float)
    if lat.ndim != 1 or lat.shape != lon.shape or not len(lat):
        raise ValueError("Latitude/longitude must be nonempty equal-length vectors.")
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("Coordinates must be finite.")
    if np.any(np.abs(lat) > 90) or np.any(np.abs(lon) > 180):
        raise ValueError("Invalid latitude/longitude.")
    if not np.isfinite(grid_spacing) or not np.isfinite(grid_tolerance) or not 0 < grid_tolerance < grid_spacing / 4:
        raise ValueError("Require finite spacing > 0 and 0 < tolerance < spacing/4.")
    candidates = (["EPSG:32632", "EPSG:3035", "EPSG:3857", "EPSG:4326"]
                  if grid_crs == "auto" else [grid_crs])
    failures = {}
    for candidate in candidates:
        try:
            crs = CRS.from_user_input(candidate)
            if crs.to_epsg() == 4326:
                x, y = lon, lat
                spacing = (_angular_spacing(lon), _angular_spacing(lat))
                tolerance, unit = 2e-5, "degrees"
            else:
                if not crs.is_projected or any(a.unit_name != "metre" for a in crs.axis_info):
                    raise ValueError("Use a projected CRS with metre units or EPSG:4326.")
                x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
                x, y = np.asarray(x), np.asarray(y)
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise ValueError("Projection produced non-finite coordinates.")
                spacing, tolerance, unit = (grid_spacing, grid_spacing), grid_tolerance, "m"
            rows, cols, residual = _indices(x, y, spacing, tolerance)
            break
        except ValueError as exc:
            failures[str(candidate)] = str(exc)
    else:
        raise ValueError(f"No verified regular grid. Supply the source --grid-crs and "
                         f"--grid-spacing; coordinates were not rearranged. {failures}")

    lookup = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    nodes = np.full((len(lat), patch_size, patch_size), -1, dtype=np.int64)
    radius = patch_size // 2
    for i, (r, c) in enumerate(zip(rows, cols)):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nodes[i, dr + radius, dc + radius] = lookup.get((r + dr, c + dc), -1)
    valid = nodes >= 0
    complete = valid.all(axis=(1, 2))
    # Coordinate-only audit of input sets; no adjacency is constructed.
    matches = 0
    for i in np.flatnonzero(complete):
        a = np.sin(np.radians(lat - lat[i]) / 2)**2 + (
            np.cos(np.radians(lat[i])) * np.cos(np.radians(lat))
            * np.sin(np.radians(lon - lon[i]) / 2)**2)
        nearest = np.argsort(a, kind="stable")[:patch_size**2]
        matches += set(nearest.tolist()) == set(nodes[i].ravel().tolist())
    metadata = {
        "crs": crs.to_string(), "spacing_x": spacing[0], "spacing_y": spacing[1],
        "spacing_unit": unit, "max_lattice_residual": residual,
        "tolerance": tolerance, "orientation": "rows grid-north to south; columns grid-west to east",
        "patch_size": patch_size, "n_locations": len(lat),
        "n_complete_patches": int(complete.sum()), "n_incomplete_patches": int((~complete).sum()),
        "complete_patches_matching_knn": int(matches), "adjacency_used": False,
        "missing_cell_policy": "zero_after_normalization_plus_binary_mask",
        "auto_candidates_rejected": failures,
    }
    return SpatialGrid(nodes, valid, rows, cols, metadata)
