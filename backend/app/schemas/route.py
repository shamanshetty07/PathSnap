"""Schema for route results and geometry responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RouteObservation(BaseModel):
    """A single observation in the reconstructed route."""
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


class RouteResult(BaseModel):
    """Complete route result for a trip."""
    trip_id: str = Field(..., description="Trip identifier")
    total_observations: int = Field(
        ..., description="Total number of GPS observations"
    )
    route: list[RouteObservation] = Field(
        default_factory=list, description="Reconstructed route observations"
    )


class RouteGeometrySegment(BaseModel):
    """A single segment of reconstructed route geometry."""
    type: str = Field("LineString", description="GeoJSON geometry type")
    coordinates: list[list[float]] = Field(
        ..., description="GeoJSON coordinates [longitude, latitude]"
    )


class RouteGeometryResponse(BaseModel):
    """GeoJSON geometry response for frontend map visualization."""
    trip_id: str = Field(..., description="Trip identifier")
    geometry: RouteGeometrySegment = Field(
        ..., description="GeoJSON MultiLineString geometry"
    )


class RouteSegmentsResponse(BaseModel):
    """Route segments response for disconnected sequences."""
    trip_id: str = Field(..., description="Trip identifier")
    geometry_type: str = Field(
        "MultiLineString", description="GeoJSON geometry type"
    )
    coordinates: list[list[float]] = Field(
        ..., description="GeoJSON coordinates [[lon, lat], ...]"
    )
    segment_count: int = Field(
        ..., description="Number of route segments"
    )