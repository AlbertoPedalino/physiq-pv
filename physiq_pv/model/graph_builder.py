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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build undirected graph: edges where Haversine distance < max_dist_km.
    Edge weight = 1 / dist_km (closer plants → stronger coupling).

    Fallback: if no edges exist, connect each node to its nearest neighbour.

    Returns:
        edge_index: (2, E) int64  — bidirectional pairs
        edge_weight: (E,) float32
    """
    n = len(lats)
    src, dst, weights = [], [], []

    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(lats[i], lons[i], lats[j], lons[j])
            if d < max_dist_km:
                src += [i, j]
                dst += [j, i]
                w = 1.0 / (d + 1e-6)
                weights += [w, w]

    if not src:
        # Nearest-neighbour fallback — track seen pairs to avoid duplicates
        seen: set[tuple[int, int]] = set()
        for i in range(n):
            dists = [
                haversine_km(lats[i], lons[i], lats[j], lons[j]) if i != j else 1e9
                for j in range(n)
            ]
            j = int(np.argmin(dists))
            pair = (min(i, j), max(i, j))
            if pair not in seen:
                seen.add(pair)
                w = 1.0 / (dists[j] + 1e-6)
                src += [i, j]
                dst += [j, i]
                weights += [w, w]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight
