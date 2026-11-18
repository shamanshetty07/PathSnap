"""Schema for map matching request and response."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MapMatchingRequest(BaseModel):
    """Request body for map matching (currently uses path params only)."""
    search_radius_m: float | None = Field(
        None, description="Override default search radius in metres"
    )
    max_candidates: int | None = Field(
        None, description="Override default max candidates per observation"
    )
    gps_gap_seconds: float | None = Field(
        None, description="Override GPS gap threshold in seconds"
    )


class MapMatchingStatistics(BaseModel):
    """Statistics from map matching execution."""
    total_observations: int = Field(..., description="Total GPS observations")
    decoded_observations: int = Field(
        ..., description="Number of observations successfully decoded"
    )
    fallback_matches: int = Field(
        0, description="Matches using fallback candidate selection"
    )
    missing_candidates: int = Field(
        0, description="Observations with no candidate generation"
    )
    sequence_count: int = Field(
        ..., description="Number of decoded sequences"
    )


class DecodedObservation(BaseModel):
    """A single decoded observation in the matched route."""
    observation_index: int = Field(
        ..., description="Index of the GPS observation"
    )
    timestamp: int = Field(
        ..., description="Unix timestamp of the observation"
    )
    arc_id: int = Field(
        ..., description="Matched road arc ID"
    )
    emission_probability: float = Field(
        ..., description="Random Forest emission probability"
    )
    decoded: bool = Field(
        ..., description="Whether the observation was HMM-decoded"
    )
    sequence_id: int = Field(
        ..., description="Sequence identifier for route continuity"
    )


class MapMatchingResponse(BaseModel):
    """Response from map matching endpoint."""
    trip_id: str = Field(..., description="Trip identifier")
    status: str = Field(..., description="Pipeline execution status")
    statistics: MapMatchingStatistics = Field(
        ..., description="Execution statistics"
    )
    route: list[DecodedObservation] = Field(
        default_factory=list, description="Reconstructed route observations"
    )