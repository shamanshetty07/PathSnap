"""Trip service containing business logic for trip operations."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import settings
from app.utils.validators import validate_trip_id, validate_split
from app.schemas.trip import TripSummary, TripMetadata

logger = logging.getLogger(__name__)


def discover_trips(split: str = "student_test") -> list[TripSummary]:
    """Discover trips available for frontend evaluation."""
    validate_split(split)
    data_dir = settings.base_dir / split

    if not data_dir.is_dir():
        return []

    trips = []
    for trip_dir in sorted(data_dir.iterdir()):
        if not trip_dir.is_dir():
            continue
        if not trip_dir.name.isdigit():
            continue

        trip_id = trip_dir.name
        track_file = trip_dir / f"{trip_id}.track"
        if not track_file.is_file():
            continue

        trips.append(TripSummary(trip_id=trip_id, status="available"))

    return trips


def get_trip_metadata(trip_id: str) -> TripMetadata:
    """Get basic metadata about a trip."""
    validated_id = validate_trip_id(trip_id)

    # Check dataset availability
    track_file = Path(settings.base_dir) / "train" / validated_id / f"{validated_id}.track"
    dataset_available = track_file.is_file()

    # Check if map matching has been performed
    results_dir = Path(settings.results_dir) / "train" / validated_id
    map_matching_done = results_dir.is_dir()

    # For observation count, we'd need to load the trip data
    # For now, return 0 as placeholder (will be filled by API layer)
    observation_count = 0

    return TripMetadata(
        trip_id=validated_id,
        observation_count=observation_count,
        dataset_available=dataset_available,
        map_matching_performed=map_matching_done,
    )