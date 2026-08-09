"""STGAN and its geographical PVGIS training/scoring integration."""

from .config import ALIGNMENT_POLICY, REFERENCE_CONFIG, REFERENCE_SEED
from .data import (
    AlignedPVGISCubes,
    STGANWindowDataset,
    calendar_features,
    load_aligned_manifest_cubes,
    regular_target_indices,
)
from .graph import GeographicalSubgraphs, build_geographical_subgraphs, haversine_km
from .model import (
    GCGRU,
    GCGRUCell,
    GraphConvolution,
    STGAN,
    STGANDiscriminator,
    STGANGenerator,
)
from .pipeline import fit_and_score_stgan, load_stgan_checkpoint
from .result import STGANResult

__all__ = [
    "ALIGNMENT_POLICY",
    "AlignedPVGISCubes",
    "GCGRU",
    "GCGRUCell",
    "GeographicalSubgraphs",
    "GraphConvolution",
    "REFERENCE_CONFIG",
    "REFERENCE_SEED",
    "STGAN",
    "STGANDiscriminator",
    "STGANGenerator",
    "STGANResult",
    "STGANWindowDataset",
    "build_geographical_subgraphs",
    "calendar_features",
    "fit_and_score_stgan",
    "haversine_km",
    "load_aligned_manifest_cubes",
    "load_stgan_checkpoint",
    "regular_target_indices",
]
