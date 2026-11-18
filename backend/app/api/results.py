"""Route results API endpoint."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.utils.validators import validate_trip_id

router = APIRouter(tags=["results"])
logger = logging.getLogger(__name__)


@router.get(
    "/api/results/{trip_id}",
    response_model=dict,
    summary="Retrieve reconstructed route",
    description="Retrieve the reconstructed route from the saved CSV.",
)
def get_route_results(trip_id: str) -> dict:
    """Retrieve the reconstructed route from the saved CSV file.

    Returns observation index, timestamp, matched arc ID,
    emission probability, decoded status, and sequence ID.
    """
    validated_id = validate_trip_id(trip_id)

    csv_path = Path(settings.results_dir) / "student_test" / validated_id / "matched_route.csv"

    if not csv_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Route results not found for trip: {validated_id}",
        )

    import pandas as pd

    df = pd.read_csv(csv_path)

    # Build route observations
    route = []
    for _, row in df.iterrows():
        obs = {
            "observation_index": int(row["observation_index"]),
            "timestamp": int(row["timestamp"]) if pd.notna(row["timestamp"]) else 0,
            "arc_id": int(row["arc_id"]) if pd.notna(row["arc_id"]) else 0,
            "emission_probability": float(row["emission_probability"])
            if pd.notna(row["emission_probability"])
            else 0.0,
            "decoded": bool(row["decoded"]) if "decoded" in row.columns else True,
            "sequence_id": int(row["sequence_id"]) if "sequence_id" in row.columns else 0,
        }
        route.append(obs)

    return {
        "trip_id": validated_id,
        "total_observations": len(df),
        "route": route,
    }


@router.get(
    "/api/results/{trip_id}/geometry",
    response_model=dict,
    summary="Retrieve route geometry",
    description="Return the reconstructed route as GeoJSON coordinates.",
)
def get_route_geometry(trip_id: str) -> dict:
    """Return the reconstructed route as GeoJSON coordinates.

    Uses actual road-network node coordinates to convert matched arc IDs
    into geographic coordinates suitable for frontend map visualization.
    Respects longitude/latitude ordering [longitude, latitude].
    Preserves sequence boundaries and handles missing or invalid arc IDs.
    """
    validated_id = validate_trip_id(trip_id)

    csv_path = Path(settings.results_dir) / "student_test" / validated_id / "matched_route.csv"

    if not csv_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Route results not found for trip: {validated_id}",
        )

    import pandas as pd
    from models.hmm import RoadNetwork

    # Load the road network for the trip
    try:
        from exploration.data_loader import load_trip
        trip_data = load_trip("student_test", validated_id, str(settings.base_dir))
        nodes = trip_data.nodes.copy()
        if "node_id" not in nodes.columns:
            nodes = nodes.reset_index().rename(columns={"index": "node_id"})

        arcs = trip_data.arcs.copy()
        arcs = arcs.rename(columns={"from_node": "from_node_id", "to_node": "to_node_id"})

        network = RoadNetwork(arcs=arcs, nodes=nodes)
        coord_map = network.coordinates  # {node_id: (lon, lat)}
    except Exception as exc:
        logger.warning("Failed to load road network: %s", exc)
        coord_map = {}

    # Load route CSV
    df = pd.read_csv(csv_path)

    # Build geometry from matched arcs
    segments: list[list[list[float]]] = []
    current_segment: list[list[float]] = []

    for _, row in df.iterrows():
        arc_id = row["arc_id"]

        if pd.isna(arc_id) or int(arc_id) not in coord_map:
            # Save current segment if exists, then start new one
            if current_segment:
                segments.append(current_segment)
                current_segment = []
            continue

        lon, lat = coord_map[int(arc_id)]

        # Check if this arc connects to the previous one
        if current_segment:
            prev_lon, prev_lat = current_segment[-1]
            # Simple check: if the new point is different from the previous, add it
            if abs(lon - prev_lon) > 1e-8 or abs(lat - prev_lat) > 1e-8:
                current_segment.append([lon, lat])
            # else: skip duplicate (same location)
        else:
            current_segment = [[lon, lat]]

    # Don't forget the last segment
    if current_segment:
        segments.append(current_segment)

    # Build GeoJSON MultiLineString
    # Only include segments that have at least 2 points
    valid_segments = [s for s in segments if len(s) >= 2]

    geometry = {
        "type": "MultiLineString",
        "coordinates": valid_segments
        if valid_segments
        else [],  # Empty if no valid segments
    }

    return {
        "trip_id": validated_id,
        "geometry": geometry,
    }