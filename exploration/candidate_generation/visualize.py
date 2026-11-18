"""Folium visualisation for candidate generation.

Displays:

* Raw GPS observations as blue points.
* Candidate road segments as green lines (limited per observation for
  readability).
* Candidate projected points as orange markers (limited).
* Reference route as a red polyline for visual comparison on training trips.

Folium expects (latitude, longitude); the dataset stores (longitude, latitude).
"""

from __future__ import annotations

from pathlib import Path

import folium

from ..data_loader import TripData
from .candidate_generator import Candidate, CandidateGenerationResult

# Maximum number of candidate segments to draw per observation. Keeps the map
# readable when an observation has many nearby segments.
MAX_CANDIDATE_SEGMENTS_PER_OBS = 3
# Maximum number of GPS observations to draw as individual points. If the
# track is longer, points are subsampled.
MAX_GPS_POINTS = 2000


def _subsample_indices(n: int, max_n: int) -> list[int]:
    if n <= max_n:
        return list(range(n))
    step = n / max_n
    return [int(i * step) for i in range(max_n)]


def build_candidate_map(
    trip_data: TripData,
    result: CandidateGenerationResult,
    output_path: Path,
    reference_route_segments: list[list[tuple[float, float]]] | None = None,
    max_candidate_segments_per_obs: int = MAX_CANDIDATE_SEGMENTS_PER_OBS,
    max_gps_points: int = MAX_GPS_POINTS,
) -> folium.Map:
    """Build a Folium map showing GPS observations and candidate segments.

    Parameters
    ----------
    trip_data : TripData
        Loaded trip data (used for GPS track centre and nodes).
    result : CandidateGenerationResult
        Generated candidates.
    output_path : Path
        Where to save the HTML file.
    reference_route_segments : list of (lat, lon) segments, optional
        Reconstructed reference route segments for visual comparison.
    max_candidate_segments_per_obs : int
        Max candidate segments drawn per observation.
    max_gps_points : int
        Max GPS points drawn (subsampled if necessary).

    Returns
    -------
    folium.Map
    """
    track = trip_data.track
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Centre the map on the GPS track centroid or first observation.
    if not track.empty:
        centre_lat = float(track["latitude"].mean())
        centre_lon = float(track["longitude"].mean())
    else:
        centre_lat, centre_lon = 0.0, 0.0

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=14, tiles="OpenStreetMap")

    # ---- GPS observations (blue points, subsampled) ----
    if not track.empty:
        gps_layer = folium.FeatureGroup(name="GPS observations (blue)", show=True)
        obs_indices = _subsample_indices(len(track), max_gps_points)
        for i in obs_indices:
            row = track.iloc[i]
            folium.CircleMarker(
                location=[float(row["latitude"]), float(row["longitude"])],
                radius=3,
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.5,
                weight=1,
                popup=f"Obs {i}",
                tooltip=f"Obs {i}",
            ).add_to(gps_layer)
        gps_layer.add_to(m)

    # ---- Candidate segments (green lines, limited per observation) ----
    if result.candidates:
        cand_layer = folium.FeatureGroup(name="Candidate segments (green)", show=True)
        proj_layer = folium.FeatureGroup(name="Candidate projections (orange)", show=True)

        # Group candidates by observation index.
        candidates_by_obs: dict[int, list[Candidate]] = {}
        for c in result.candidates:
            candidates_by_obs.setdefault(c.observation_index, []).append(c)

        drawn_segments = 0
        drawn_projections = 0
        max_total_segments = 300  # global cap for readability

        for obs_idx in sorted(candidates_by_obs.keys()):
            obs_cands = sorted(
                candidates_by_obs[obs_idx], key=lambda c: c.distance_m
            )[:max_candidate_segments_per_obs]
            for c in obs_cands:
                if drawn_segments >= max_total_segments:
                    break
                folium.PolyLine(
                    locations=[
                        [c.gps_latitude, c.gps_longitude],
                        [c.projected_latitude, c.projected_longitude],
                    ],
                    color="green",
                    weight=2,
                    opacity=0.6,
                    popup=(
                        f"Obs {c.observation_index}, Arc {c.arc_id}<br>"
                        f"Distance: {c.distance_m:.1f} m"
                    ),
                ).add_to(cand_layer)
                drawn_segments += 1

                folium.CircleMarker(
                    location=[c.projected_latitude, c.projected_longitude],
                    radius=3,
                    color="orange",
                    fill=True,
                    fill_color="orange",
                    fill_opacity=0.7,
                    weight=1,
                    popup=f"Proj Arc {c.arc_id}, {c.distance_m:.1f} m",
                ).add_to(proj_layer)
                drawn_projections += 1
            if drawn_segments >= max_total_segments:
                break

        cand_layer.add_to(m)
        proj_layer.add_to(m)

    # ---- Reference route (red polyline, training trips only) ----
    if reference_route_segments:
        route_layer = folium.FeatureGroup(name="Reference route (red)", show=True)
        for segment in reference_route_segments:
            if len(segment) < 2:
                continue
            folium.PolyLine(
                locations=segment,
                color="red",
                weight=3,
                opacity=0.8,
                popup="Reference route",
            ).add_to(route_layer)
        route_layer.add_to(m)

    # ---- Start / end markers ----
    if not track.empty:
        first = track.iloc[0]
        folium.Marker(
            location=[float(first["latitude"]), float(first["longitude"])],
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
            popup="GPS start",
            tooltip="Start",
        ).add_to(m)
        last = track.iloc[-1]
        folium.Marker(
            location=[float(last["latitude"]), float(last["longitude"])],
            icon=folium.Icon(color="red", icon="stop", prefix="fa"),
            popup="GPS end",
            tooltip="End",
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(output_path))
    return m


def build_map_from_result(
    trip_data: TripData,
    result: CandidateGenerationResult,
    output_path: Path,
    reference_segments: list[list[tuple[float, float]]] | None = None,
) -> folium.Map:
    """Convenience wrapper that builds and saves the candidate map."""
    return build_candidate_map(
        trip_data=trip_data,
        result=result,
        output_path=output_path,
        reference_route_segments=reference_segments,
    )
