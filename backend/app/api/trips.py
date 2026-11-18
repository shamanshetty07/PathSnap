"""Trip management API endpoints."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Path as PathParam, Query

from pathlib import Path

from app.config import settings
from app.utils.validators import validate_trip_id, validate_split
from app.schemas.trip import TripSummary, TripMetadata

router = APIRouter(tags=["trips"])
logger = logging.getLogger(__name__)


def _discover_trips(split: str = "student_test") -> list[TripSummary]:
    """Discover trips available for frontend evaluation from the dataset directory.

    Only returns trips from the student_test split (permitted for frontend evaluation).
    Does not expose reference routes, training labels, or internal dataset paths.
    """
    data_dir = settings.base_dir / split

    if not data_dir.is_dir():
        return []

    trips = []
    for trip_dir in sorted(data_dir.iterdir()):
        if not trip_dir.is_dir():
            continue
        # Only include numerically-named trip directories
        if not trip_dir.name.isdigit():
            continue

        trip_id = trip_dir.name
        # Check that essential files exist (track at minimum)
        track_file = trip_dir / f"{trip_id}.track"
        if not track_file.is_file():
            continue

        trips.append(TripSummary(trip_id=trip_id, status="available"))

    return trips


@router.get(
    "/api/trips",
    response_model=list[TripSummary],
    summary="List available trips",
    description="Return the trips available for frontend evaluation.",
)
def list_trips(split: str = Query("student_test", description="Dataset split")) -> list[TripSummary]:
    """List trips available for frontend evaluation.

    Discovers trips from the dataset directory. Only returns trips from splits
    permitted for frontend evaluation. Does not expose reference routes,
    training labels, or internal dataset paths.
    """
    validate_split(split)
    trips = _discover_trips(split)
    logger.info("Found %d trips in split %s", len(trips), split)
    return trips


@router.get(
    "/api/trips/{trip_id}",
    response_model=TripMetadata,
    summary="Get trip metadata",
    description="Return basic information about the selected trip.",
)
def get_trip_metadata(trip_id: str = PathParam(...)) -> TripMetadata:
    """Get basic information about a selected trip.

    Returns trip ID, number of GPS observations, dataset availability,
    and whether map matching has already been performed.
    Does not return ground-truth information.
    """
    validated_id = validate_trip_id(trip_id)

    # Check if map matching has been performed
    results_dir = Path(settings.results_dir) / "train" / validated_id
    map_matching_done = results_dir.is_dir() if settings.results_dir else False

    # Check dataset availability by looking for track file
    track_file = Path(settings.base_dir) / "train" / validated_id / f"{validated_id}.track"
    dataset_available = track_file.is_file()

    return TripMetadata(
        trip_id=validated_id,
        observation_count=0,  # Will be filled when data is loaded
        dataset_available=dataset_available,
        map_matching_performed=map_matching_done,
    )