"""Schema for trip listing and metadata."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TripSummary(BaseModel):
    """Summary of an available trip for frontend evaluation."""
    trip_id: str = Field(..., description="Unique trip identifier")
    status: str = Field("available", description="Availability status")


class TripMetadata(BaseModel):
    """Basic metadata about a selected trip."""
    trip_id: str = Field(..., description="Unique trip identifier")
    observation_count: int = Field(..., description="Number of GPS observations")
    dataset_available: bool = Field(..., description="Whether dataset files are available")
    map_matching_performed: bool = Field(
        False, description="Whether map matching has been performed for this trip"
    )