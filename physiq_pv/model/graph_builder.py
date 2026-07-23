import numpy as np
import torch


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_graph(
    lats: np.ndarray,
    lons: np.ndarray,
    max_dist_km: float = 50.0,
    distance_scale_km: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build an undirected geographic graph with a Gaussian distance prior.

    Every node is guaranteed at least one neighbour. Nodes isolated by the
    distance threshold are connected to their nearest plant.

    Returns:
        edge_index: (2, E) int64  — bidirectional pairs
        edge_weight: (E,) float32
    """
    n = len(lats)
    if n < 2:
        raise ValueError("build_graph requires at least two nodes")
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    if not (np.isfinite(lats).all() and np.isfinite(lons).all()):
        raise ValueError("Graph coordinates must be finite")
    if max_dist_km <= 0:
        raise ValueError("max_dist_km must be positive")
    if distance_scale_km is None:
        distance_scale_km = max_dist_km / 2.0
    if distance_scale_km <= 0:
        raise ValueError("distance_scale_km must be positive")

    distances = np.full((n, n), np.inf, dtype=float)
    undirected_edges: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(lats[i], lons[i], lats[j], lons[j])
            distances[i, j] = distances[j, i] = d
            if d < max_dist_km:
                undirected_edges[(i, j)] = d

    degree = np.zeros(n, dtype=int)
    for i, j in undirected_edges:
        degree[i] += 1
        degree[j] += 1
    for i in np.flatnonzero(degree == 0):
        if degree[i] > 0:
            continue
        j = int(np.argmin(distances[i]))
        pair = (min(i, j), max(i, j))
        undirected_edges[pair] = distances[i, j]
        degree[i] += 1
        degree[j] += 1

    src: list[int] = []
    dst: list[int] = []
    weights: list[float] = []
    for (i, j), distance in sorted(undirected_edges.items()):
        weight = max(
            float(np.exp(-0.5 * (distance / float(distance_scale_km)) ** 2)),
            1e-6,
        )
        src.extend([i, j])
        dst.extend([j, i])
        weights.extend([weight, weight])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight
