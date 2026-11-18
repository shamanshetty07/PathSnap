"""Candidate generation for PathSnap map matching.

Identifies nearby plausible road segments (arcs) for every GPS observation
using a KD-tree spatial index and exact point-to-segment distance calculations.
"""

from .candidate_generator import (
    Candidate,
    CandidateGenerationResult,
    generate_candidates,
    run_trip,
    save_candidates_csv,
    save_candidates_jsonl,
)
from .candidate_validation import ReferenceCoverageReport, evaluate_reference_coverage
from .geometry import (
    EARTH_RADIUS_M,
    LocalProjection,
    gps_bearing_degrees,
    haversine_distance_m,
    project_point_on_segment,
    segment_bearing_meters,
)
from .spatial_index import IndexedSegment, RoadNetworkIndex

__all__ = [
    "Candidate",
    "CandidateGenerationResult",
    "ReferenceCoverageReport",
    "IndexedSegment",
    "RoadNetworkIndex",
    "LocalProjection",
    "generate_candidates",
    "run_trip",
    "save_candidates_csv",
    "save_candidates_jsonl",
    "evaluate_reference_coverage",
    "haversine_distance_m",
    "project_point_on_segment",
    "segment_bearing_meters",
    "gps_bearing_degrees",
    "EARTH_RADIUS_M",
]

__version__ = "0.1.0"
