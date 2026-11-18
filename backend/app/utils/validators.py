"""Validation utilities for PathSnap API inputs."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import HTTPException, status


def validate_trip_id(trip_id: str) -> str:
    """Validate and sanitize a trip ID to prevent path traversal.

    Returns the cleaned trip ID.
    Raises HTTPException if the trip ID is invalid.
    """
    if not trip_id or not trip_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Trip ID must not be empty",
        )

    # Strip whitespace and check for path traversal
    cleaned = trip_id.strip()

    # Reject absolute paths or traversal sequences
    if re.search(r"[\\/\.\]{2,}", cleaned) or cleaned.startswith("/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid trip ID: path traversal detected",
        )

    # Only allow alphanumeric and hyphens/underscores
    if not re.match(r"^[A-Za-z0-9_-]+$", cleaned):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid trip ID: must contain only alphanumeric characters, hyphens, or underscores",
        )

    return cleaned


def validate_split(split: str) -> str:
    """Validate a dataset split name."""
    if split not in ("train", "student_test"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid split: {split}. Must be one of: train, student_test",
        )
    return split


class TripNotFoundError(HTTPException):
    """Raised when a trip ID is not found in the dataset."""

    def __init__(self, trip_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip not found: {trip_id}",
        )


class MapMatchingNotPerformedError(HTTPException):
    """Raised when map matching has not been performed for a trip."""

    def __init__(self, trip_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Map matching not performed for trip: {trip_id}",
        )


class ModelNotLoadedError(HTTPException):
    """Raised when the ML model is not available."""

    def __init__(self, detail: str = "Trained model not available"):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )


class DuplicateProcessingError(HTTPException):
    """Raised when a map-matching request is already in progress."""

    def __init__(self, trip_id: str):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Map matching already in progress for trip: {trip_id}",
        )