"""PathSnap exploration package.

Reference route reconstruction and visualisation components.
"""

from .data_loader import TripData, TripDataError, load_trip
from .route_reconstruction import ReconstructedRoute, reconstruct_reference_route
from .route_validation import (
    DisconnectedTransition,
    RouteValidationReport,
    validate_continuity,
    validate_route_references,
)

__all__ = [
    "TripData",
    "TripDataError",
    "load_trip",
    "ReconstructedRoute",
    "reconstruct_reference_route",
    "DisconnectedTransition",
    "RouteValidationReport",
    "validate_continuity",
    "validate_route_references",
]

__version__ = "0.1.0"
