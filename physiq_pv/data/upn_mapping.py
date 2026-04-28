"""Stub: UPN → (lat, lon) mapping. Synthetic coords used until real mapping arrives."""
import numpy as np
from physiq_pv.data.synthetic_generator import _coords, N_PLANTS


def get_upn_coords(upn_list: list[str] | None = None) -> dict[str, tuple[float, float]]:
    """
    Return {upn: (lat, lon)} mapping.
    upn_list=None → all synthetic plants.

    Real implementation: read a CSV/DB table keyed on UPN codes.
    """
    lats, lons = _coords()
    synthetic = {f"UPN_{i:03d}": (float(lats[i]), float(lons[i])) for i in range(N_PLANTS)}
    if upn_list is not None:
        return {k: v for k, v in synthetic.items() if k in upn_list}
    return synthetic


def get_coords_array(upn_list: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (lats, lons) arrays in plant order."""
    mapping = get_upn_coords(upn_list)
    coords = list(mapping.values())
    lats = np.array([c[0] for c in coords])
    lons = np.array([c[1] for c in coords])
    return lats, lons
