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
    Build a geographic graph with a bounded Gaussian distance prior.

    Returns:
        edge_index: directed edges including self-loops
        edge_weight: Gaussian prior in (0, 1], consumed in log-space by GAT
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    if lats.ndim != 1 or lons.ndim != 1 or len(lats) != len(lons):
        raise ValueError("lats and lons must be one-dimensional arrays of equal length.")
    if len(lats) == 0:
        raise ValueError("At least one graph node is required.")
    if not np.isfinite(lats).all() or not np.isfinite(lons).all():
        raise ValueError("Graph coordinates must be finite.")
    if max_dist_km <= 0:
        raise ValueError("max_dist_km must be positive.")
    if distance_scale_km is None:
        distance_scale_km = max_dist_km / 2.0
    if distance_scale_km <= 0:
        raise ValueError("distance_scale_km must be positive.")

    n = len(lats)
    distances = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(distances, 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            distance = haversine_km(lats[i], lons[i], lats[j], lons[j])
            distances[i, j] = distances[j, i] = distance

    undirected: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] <= max_dist_km:
                undirected.add((i, j))

    # Every isolated node receives its nearest-neighbour connection.
    if n > 1:
        degree = np.zeros(n, dtype=np.int64)
        for i, j in undirected:
            degree[i] += 1
            degree[j] += 1
        for i in np.flatnonzero(degree == 0):
            candidates = distances[i].copy()
            candidates[i] = np.inf
            j = int(np.argmin(candidates))
            undirected.add((min(i, j), max(i, j)))

    src, dst, weights = [], [], []
    for i, j in sorted(undirected):
        prior = max(
            float(np.exp(-0.5 * (distances[i, j] / distance_scale_km) ** 2)),
            1e-6,
        )
        src.extend((i, j))
        dst.extend((j, i))
        weights.extend((prior, prior))
    for i in range(n):
        src.append(i)
        dst.append(i)
        weights.append(1.0)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    incoming = torch.bincount(edge_index[1], minlength=n)
    if int(incoming.min()) < 1:
        raise RuntimeError("Graph construction left a node without an incoming edge.")
    return edge_index, edge_weight
