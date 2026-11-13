"""The neighbourhood within a distance in km, at the same timestamp, on the regular latitude-longitude grid of ERA5.

The zone of a concordant cluster is every cell whose centre lies less than `radius_km` from a cell of the cluster at the
same timestamp (Bourdin et al. 2022, GMD, Sect. 2.4: points at the same time "located closer than 300 km"; Flaounas et
al. 2023, WCD, Sect. 2.1), the distance being the great-circle (haversine) distance between cell centres on a sphere of
radius 6371 km. A cell within the radius of more than one cluster goes to the cluster with the nearest cell, ties to the
lowest cluster number (the affiliation zones of Huet et al. 2022, there in time, here in space at one timestamp). On a
grid of constant spacing in degrees the disc spans the same rows everywhere but more columns towards the pole, so the
offsets are listed row by row of latitude.
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between points given in degrees (arrays broadcast)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlam = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def row_offsets(latitudes, lon_step_deg, radius_km):
    """For every row r of the grid, the offsets (dr, dc) of the cells whose centre lies less than `radius_km` from a
    cell of row r, with their distance, sorted by distance (then dr, then dc). `latitudes` are the row latitudes in
    degrees, in the order of the rows; the columns are `lon_step_deg` apart. Returns a list of arrays [n, 3] of (dr, dc,
    distance), one per row; rows outside the grid are left out."""
    latitudes = np.asarray(latitudes, dtype=np.float64)
    height = len(latitudes)
    out = []
    for r in range(height):
        found = []
        for target in range(height):
            dr = target - r
            if haversine_km(latitudes[r], 0.0, latitudes[target], 0.0) >= radius_km:
                continue  # even the cell straight north or south is too far
            dc = 0
            while True:
                distance = haversine_km(latitudes[r], 0.0, latitudes[target], dc * lon_step_deg)
                if distance >= radius_km:
                    break
                found.append((dr, dc, distance))
                if dc:
                    found.append((dr, -dc, distance))
                dc += 1
        found.sort(key=lambda item: (item[2], item[0], item[1]))
        out.append(np.array(found, dtype=np.float64).reshape(-1, 3))
    return out


def nearest_within(labels, offsets):
    """For a block of labels [time, row, column] (0 where no cluster), the label of the nearest labelled cell within the
    radius of `offsets` (row_offsets) at the same timestamp, ties to the lowest label; 0 where no labelled cell is within
    the radius. The cells of a cluster are their own nearest (distance 0)."""
    labels = np.asarray(labels)
    times, height, width = labels.shape
    owner = np.zeros(labels.shape, dtype=labels.dtype)
    best = np.full((times, width), np.inf)
    for r in range(height):
        best.fill(np.inf)
        row_owner = owner[:, r, :]
        for dr, dc, distance in offsets[r]:
            dr, dc = int(dr), int(dc)
            source = labels[:, r + dr, :]
            candidate = np.zeros((times, width), dtype=labels.dtype)
            if dc >= 0:
                candidate[:, :width - dc] = source[:, dc:]
            else:
                candidate[:, -dc:] = source[:, :width + dc]
            better = (candidate > 0) & ((distance < best) | ((distance == best) & (candidate < row_owner)))
            row_owner[better] = candidate[better]
            best[better] = distance
    return owner


def owner_lattice(labels, grid, radius_km, chunk=256):
    """The nearest cluster within `radius_km` of every value of a lattice of cluster numbers [time, row, column], with
    how many rows and columns the disc reaches at most. `grid` holds the row latitudes and column longitudes."""
    latitudes = np.asarray(grid["latitudes"], dtype=np.float64)
    lon_step = float(abs(grid["longitudes"][1] - grid["longitudes"][0]))
    offsets = row_offsets(latitudes, lon_step, radius_km)
    reach_rows = int(max(np.abs(o[:, 0]).max() for o in offsets if len(o)))
    reach_cols = int(max(np.abs(o[:, 1]).max() for o in offsets if len(o)))
    owner = np.zeros(labels.shape, dtype=labels.dtype)
    for start in range(0, labels.shape[0], chunk):
        owner[start:start + chunk] = nearest_within(labels[start:start + chunk], offsets)
    return owner, reach_rows, reach_cols


def radius_box(row, reach_rows, reach_cols, shape):
    """The box of a cluster (a row of clusters.csv) holding its zone: its own timestamps, its rows and columns widened
    by the reach of the disc, cut at the edges of the grid."""
    return (int(row.t0), int(row.t1), max(int(row.row0) - reach_rows, 0), min(int(row.row1) + reach_rows, shape[0] - 1),
            max(int(row.col0) - reach_cols, 0), min(int(row.col1) + reach_cols, shape[1] - 1))


def within(mask, offsets):
    """The cells less than the radius of `offsets` from a True cell of `mask` [time, row, column] at the same timestamp
    (a dilation by a disc whose width in columns depends on the latitude)."""
    labels = np.asarray(mask).astype(np.int8)
    return nearest_within(labels, offsets) > 0
