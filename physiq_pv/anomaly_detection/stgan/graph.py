"""Geographical subgraphs for the PVGIS adaptation of STGAN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EARTH_RADIUS_KM = 6371.0088


def haversine_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> np.ndarray:
    """Return great-circle distances in kilometres with broadcast semantics."""
    lat1_r = np.radians(np.asarray(lat1, dtype=np.float64))
    lon1_r = np.radians(np.asarray(lon1, dtype=np.float64))
    lat2_r = np.radians(np.asarray(lat2, dtype=np.float64))
    lon2_r = np.radians(np.asarray(lon2, dtype=np.float64))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + (
        np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass(frozen=True)
class GeographicalSubgraphs:
    """Fixed-size local graphs used by the paper's per-target-node batches."""

    node_indices: np.ndarray
    normalized_adjacency: np.ndarray
    sigma_km: float
    directed_edge_count: int
    topology: str

    @property
    def n_locations(self) -> int:
        return int(self.node_indices.shape[0])

    @property
    def subgraph_size(self) -> int:
        return int(self.node_indices.shape[1])


def _validate_coordinates(lats: np.ndarray, lons: np.ndarray) -> None:
    if lats.ndim != 1 or lons.ndim != 1 or lats.shape != lons.shape:
        raise ValueError("Latitude and longitude must be same-length 1-D arrays.")
    if lats.size < 2:
        raise ValueError("STGAN requires at least two geographical locations.")
    if not np.isfinite(lats).all() or not np.isfinite(lons).all():
        raise ValueError("STGAN coordinates must all be finite.")
    if np.any(np.abs(lats) > 90.0) or np.any(np.abs(lons) > 180.0):
        raise ValueError("Invalid latitude or longitude range.")


def build_geographical_subgraphs(
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    subgraph_size: int = 9,
    sigma_km: float | None = None,
    chunk_size: int = 256,
) -> GeographicalSubgraphs:
    """Build directed geographical KNN subgraphs for the PVGIS adaptation.

    Only a ``chunk_size x N`` distance block is materialised. Runtime graph
    convolutions therefore operate on ``subgraph_size x subgraph_size``
    matrices rather than a dense ``N x N`` adjacency.
    """
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    _validate_coordinates(lats, lons)
    if subgraph_size < 2:
        raise ValueError("subgraph_size must be at least two.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    n_locations = lats.size
    local_size = min(int(subgraph_size), n_locations)
    k = local_size - 1
    nearest = np.empty((n_locations, k), dtype=np.int64)
    nearest_distances = np.empty((n_locations, k), dtype=np.float64)

    for start in range(0, n_locations, chunk_size):
        stop = min(start + chunk_size, n_locations)
        distances = haversine_km(
            lats[start:stop, None],
            lons[start:stop, None],
            lats[None, :],
            lons[None, :],
        )
        rows = np.arange(stop - start)
        distances[rows, np.arange(start, stop)] = np.inf
        candidates = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        candidate_distances = np.take_along_axis(distances, candidates, axis=1)
        order = np.argsort(candidate_distances, axis=1)
        nearest[start:stop] = np.take_along_axis(candidates, order, axis=1)
        nearest_distances[start:stop] = np.take_along_axis(
            candidate_distances, order, axis=1
        )

    # The paper defines a directed graph. With no physical PVGIS connectivity,
    # the closest available surrogate is one outgoing edge per nearest node.
    neighbour_sets = [set(row.tolist()) for row in nearest]
    finite_edges = nearest_distances[np.isfinite(nearest_distances)]
    if finite_edges.size == 0:
        raise ValueError("No finite geographical neighbour distances were found.")
    # Equation (1) and the official loader use the standard deviation of
    # graph-edge distances. The median fallback only handles a degenerate
    # coordinate layout where the paper definition would divide by zero.
    if sigma_km is None:
        # torch.std in the official loader uses the sample standard deviation.
        resolved_sigma = (
            float(np.std(finite_edges, ddof=1)) if finite_edges.size > 1 else 0.0
        )
    else:
        resolved_sigma = float(sigma_km)
    if not np.isfinite(resolved_sigma) or resolved_sigma <= 1e-6:
        resolved_sigma = float(np.median(finite_edges))
    if not np.isfinite(resolved_sigma) or resolved_sigma <= 1e-6:
        resolved_sigma = 1.0

    node_indices = np.concatenate(
        (np.arange(n_locations, dtype=np.int64)[:, None], nearest), axis=1
    )
    adjacency = np.zeros(
        (n_locations, local_size, local_size), dtype=np.float32
    )
    directed_edges = sum(len(targets) for targets in neighbour_sets)
    for target in range(n_locations):
        nodes = node_indices[target]
        local_lats = lats[nodes]
        local_lons = lons[nodes]
        local_distances = haversine_km(
            local_lats[:, None],
            local_lons[:, None],
            local_lats[None, :],
            local_lons[None, :],
        )
        for i, source_node in enumerate(nodes):
            for j, destination_node in enumerate(nodes):
                if i == j:
                    continue
                if int(destination_node) in neighbour_sets[int(source_node)]:
                    adjacency[target, i, j] = np.exp(
                        -(local_distances[i, j] ** 2) / (resolved_sigma**2)
                    )

        # Official STGAN adds self loops and applies D^-1/2 A D^-1/2.
        with_self = adjacency[target] + np.eye(local_size, dtype=np.float32)
        inverse_sqrt_degree = np.power(
            np.maximum(with_self.sum(axis=1), 1e-6), -0.5
        )
        adjacency[target] = (
            inverse_sqrt_degree[:, None]
            * with_self
            * inverse_sqrt_degree[None, :]
        )

    return GeographicalSubgraphs(
        node_indices=node_indices,
        normalized_adjacency=adjacency,
        sigma_km=resolved_sigma,
        directed_edge_count=directed_edges,
        topology="directed_geographical_knn",
    )
