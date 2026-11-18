"""Map matching service containing business logic for map matching operations."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.config import settings
from app.utils.validators import validate_trip_id, validate_split
from app.schemas.map_matching import (
    MapMatchingResponse,
    MapMatchingStatistics,
    DecodedObservation,
)
from app.schemas.route import RouteObservation

logger = logging.getLogger(__name__)

# In-memory lock to prevent duplicate processing
_processing_locks: dict[str, asyncio.Lock] = {}


def _run_map_matching_pipeline_sync(
    trip_id: str,
    split: str,
    search_radius_m: float,
    max_candidates: int,
    gps_gap_seconds: float,
) -> dict:
    """Run the map matching pipeline synchronously in a thread."""
    import joblib
    import pandas as pd
    import numpy as np

    from models.hmm import RoadNetwork, HiddenMarkovMapMatcher
    from exploration.data_loader import load_trip
    from exploration.candidate_generation.candidate_generator import generate_candidates
    from features.spatial_features import compute_spatial_features
    from features.temporal_features import compute_temporal_features
    from features.network_features import compute_network_features
    from features.transition_features import compute_transition_features

    # 1. Load trip data
    logger.info("Step 1: Loading trip data for %s", trip_id)
    trip_data = load_trip(split=split, trip_id=trip_id, base_dir=str(settings.base_dir))

    observations = trip_data.track.copy().reset_index(drop=True)
    observations["observation_index"] = np.arange(len(observations))

    # 2. Generate candidates
    logger.info("Step 2: Generating candidates (search_radius=%s, max_candidates=%s)",
                search_radius_m, max_candidates)
    result = generate_candidates(
        trip_data,
        search_radius_m=search_radius_m,
        max_candidates=max_candidates,
    )

    candidates = pd.DataFrame(
        [candidate.to_dict() for candidate in result.candidates]
    )

    if candidates.empty:
        logger.warning("No candidates generated for trip %s", trip_id)
        return {
            "trip_id": trip_id,
            "status": "completed",
            "statistics": {
                "total_observations": len(observations),
                "decoded_observations": 0,
                "fallback_matches": 0,
                "missing_candidates": len(observations),
                "sequence_count": 0,
            },
            "route": [],
        }

    # 3. Generate features
    logger.info("Step 3: Generating features (%d candidates)", len(candidates))
    candidates["trip_id"] = trip_data.trip_id

    # Rename columns
    candidates = candidates.rename(
        columns={
            "gps_lon": "gps_longitude",
            "gps_lat": "gps_latitude",
        }
    )

    df = compute_spatial_features(candidates, nodes_df=trip_data.nodes)
    df = compute_temporal_features(df, track=trip_data.track)
    df = compute_network_features(df, arcs_df=trip_data.arcs, nodes_df=trip_data.nodes)
    df = compute_transition_features(df, arcs_df=trip_data.arcs, nodes_df=trip_data.nodes)

    # 4. Load Random Forest model and predict emission probabilities
    logger.info("Step 4: Loading Random Forest model")
    model_path = Path(settings.model_path)
    if not model_path.is_file():
        logger.error("Model file not found: %s", model_path)
        raise RuntimeError("Trained model not available")

    pipeline_data = joblib.load(model_path)
    pipeline = pipeline_data["pipeline"]
    features = pipeline_data["features"]

    # 5. Predict emission probabilities
    logger.info("Step 5: Predicting emission probabilities")
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    df_features = df[features].copy()
    probabilities = pipeline.predict_proba(df_features)[:, 1]
    df = df.copy()
    df["emission_probability"] = probabilities

    # 6. Build road network
    logger.info("Step 6: Building road network")
    nodes = trip_data.nodes.copy()
    if "node_id" not in nodes.columns:
        nodes = nodes.reset_index().rename(columns={"index": "node_id"})

    arcs = trip_data.arcs.copy()
    arcs = arcs.rename(columns={"from_node": "from_node_id", "to_node": "to_node_id"})

    network = RoadNetwork(arcs=arcs, nodes=nodes)

    # 7. Run HMM/Viterbi
    logger.info("Step 7: Running HMM/Viterbi")
    matcher = HiddenMarkovMapMatcher(network)

    # Prepare candidates for HMM
    candidates_for_hmm = df.copy()
    candidates_for_hmm = candidates_for_hmm.rename(
        columns={
            "gps_longitude": "gps_lon",
            "gps_latitude": "gps_lat",
        }
    )

    # Ensure required columns
    required_cols = [
        "observation_index", "arc_id", "from_node_id", "to_node_id",
        "distance_along_segment_m", "segment_length_m",
        "gps_lon", "gps_lat", "timestamp", "emission_probability",
    ]
    missing = [col for col in required_cols if col not in candidates_for_hmm.columns]
    if missing:
        raise ValueError(f"Candidates missing HMM columns: {missing}")

    candidates_for_hmm["observation_index"] = candidates_for_hmm[
        "observation_index"
    ].astype(int)

    observations_for_hmm = observations.copy()
    observations_for_hmm = observations_for_hmm.rename(
        columns={
            "gps_longitude": "gps_lon",
            "gps_latitude": "gps_lat",
        }
    )

    try:
        decoded = matcher.decode(candidates_for_hmm, observations=observations_for_hmm)
    except ValueError as exc:
        if "No valid transition" in str(exc):
            logger.info("HMM transition error, splitting sequences: %s", exc)
            decoded = _handle_invalid_transitions_hmm(
                matcher, candidates_for_hmm, observations_for_hmm, gps_gap_seconds
            )
        else:
            raise

    # 8. Save results
    logger.info("Step 8: Saving results")
    output_dir = Path(settings.results_dir) / split / trip_id
    output_dir.mkdir(parents=True, exist_ok=True)

    output_cols = [
        "observation_index", "timestamp", "arc_id", "emission_probability",
        "decoded", "no_candidate", "fallback_match", "decode_failed",
        "sequence_id",
    ]

    available_output = [col for col in output_cols if col in decoded.columns]
    result_df = decoded[available_output].copy()
    result_df = result_df.sort_values("observation_index").reset_index(drop=True)
    csv_path = output_dir / "matched_route.csv"
    result_df.to_csv(csv_path, index=False, float_format="%.6g")

    # Build response statistics
    total_observations = len(observations)
    decoded_count = int(decoded["decoded"].sum())
    fallback_count = int(decoded["fallback_match"].sum())
    missing_count = int(decoded["no_candidate"].sum())
    sequence_count = int(decoded["sequence_id"].max()) + 1 if not decoded.empty else 0

    # Convert decoded observations to response format
    route_obs = []
    if not decoded.empty:
        for _, row in decoded.iterrows():
            obs = {
                "observation_index": int(row["observation_index"]),
                "timestamp": int(row["timestamp"]) if pd.notna(row["timestamp"]) else 0,
                "arc_id": int(row["arc_id"]) if pd.notna(row["arc_id"]) else 0,
                "emission_probability": float(row["emission_probability"]),
                "decoded": bool(row["decoded"]),
                "sequence_id": int(row["sequence_id"]) if pd.notna(row["sequence_id"]) else 0,
            }
            route_obs.append(obs)

    statistics = {
        "total_observations": total_observations,
        "decoded_observations": decoded_count,
        "fallback_matches": fallback_count,
        "missing_candidates": missing_count,
        "sequence_count": sequence_count,
    }

    return {
        "trip_id": trip_id,
        "status": "completed",
        "statistics": statistics,
        "route": route_obs,
    }


def _handle_invalid_transitions_hmm(
    matcher: HiddenMarkovMapMatcher,
    candidates: pd.DataFrame,
    observations: pd.DataFrame,
    gps_gap_seconds: float,
) -> pd.DataFrame:
    """Handle HMM invalid transitions by splitting sequences."""
    candidate_groups = {
        int(index): group.sort_values(
            "emission_probability", ascending=False
        ).reset_index(drop=True)
        for index, group in candidates.groupby(
            "observation_index", sort=False
        )
    }

    output_rows = []
    current_segment_groups: list[pd.DataFrame] = []
    reachable_candidates: pd.DataFrame | None = None
    sequence_id = 0
    previous_observation = None

    ordered_observations = observations.sort_values("observation_index")

    for _, observation in ordered_observations.iterrows():
        obs_index = int(observation["observation_index"])
        group = candidate_groups.get(obs_index)

        # Split at invalid or unusually large GPS time gaps
        if previous_observation is not None:
            previous_time = previous_observation.get("timestamp", 0)
            current_time = observation.get("timestamp", 0)
            if pd.notna(previous_time) and pd.notna(current_time):
                delta_time = float(current_time - previous_time)
                if delta_time <= 0 or delta_time > gps_gap_seconds:
                    if current_segment_groups:
                        rows = _decode_segment_fallback(
                            matcher, current_segment_groups, sequence_id
                        )
                        output_rows.extend(rows)
                        sequence_id += 1
                        current_segment_groups = []

        # Handle observations without candidates
        if group is None or group.empty:
            if current_segment_groups:
                rows = _decode_segment_fallback(
                    matcher, current_segment_groups, sequence_id
                )
                output_rows.extend(rows)
                sequence_id += 1
                current_segment_groups = []

            output_rows.append({
                "observation_index": obs_index,
                "arc_id": float("nan"),
                "emission_probability": 0.0,
                "decoded": False,
                "no_candidate": True,
                "fallback_match": False,
                "decode_failed": False,
                "sequence_id": sequence_id,
            })
            previous_observation = observation
            continue

        # Initialize first segment
        if not current_segment_groups:
            current_segment_groups.append(group)
            reachable_candidates = group.copy()
            previous_observation = observation
            continue

        # Check reachability
        assert reachable_candidates is not None

        reachable_indices = []
        for current_idx, current_cand in group.iterrows():
            is_reachable = False
            for _, prev_cand in reachable_candidates.iterrows():
                transition_score = matcher.transition_log_probability(
                    prev_cand, current_cand
                )
                if __import__("numpy").isfinite(transition_score):
                    is_reachable = True
                    break

        if not reachable_indices:
            # No candidate can continue - flush and start new sequence
            if current_segment_groups:
                rows = _decode_segment_fallback(
                    matcher, current_segment_groups, sequence_id
                )
                output_rows.extend(rows)
                sequence_id += 1
                current_segment_groups = []
            current_segment_groups.append(group)
            reachable_candidates = group.copy()
        else:
            current_segment_groups.append(group)
            reachable_candidates = group.loc[reachable_indices].reset_index(
                drop=True
            )

        previous_observation = observation

    # Flush final segment
    if current_segment_groups:
        rows = _decode_segment_fallback(
            matcher, current_segment_groups, sequence_id
        )
        output_rows.extend(rows)

    if not output_rows:
        return pd.DataFrame()

    result = pd.DataFrame(output_rows)
    if not result.empty:
        result = result.sort_values("observation_index").reset_index(drop=True)
    return result


def _decode_segment_fallback(
    matcher: HiddenMarkovMapMatcher,
    segment_groups: list[pd.DataFrame],
    sequence_id: int,
) -> list[dict]:
    """Decode a segment using fallback emission-based selection."""
    fallback_rows = []
    for group in segment_groups:
        if group.empty:
            continue
        best = group.sort_values(
            "emission_probability", ascending=False
        ).iloc[0]
        fallback_rows.append({
            "observation_index": int(best["observation_index"]),
            "arc_id": int(best["arc_id"]),
            "emission_probability": float(best["emission_probability"]),
            "decoded": False,
            "no_candidate": False,
            "fallback_match": True,
            "decode_failed": True,
            "sequence_id": sequence_id,
        })
    return fallback_rows


class MapMatchingService:
    """Service class for map matching operations."""

    @staticmethod
    async def run_map_matching(
        trip_id: str,
        split: str = "student_test",
        search_radius_m: float | None = None,
        max_candidates: int | None = None,
        gps_gap_seconds: float | None = None,
    ) -> MapMatchingResponse:
        """Run map matching for a trip asynchronously."""
        validated_id = validate_trip_id(trip_id)

        # Check for existing processing lock
        if validated_id in _processing_locks:
            lock = _processing_locks[validated_id]
            if not lock.locked():
                del _processing_locks[validated_id]
            else:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=409,
                    detail=f"Map matching already in progress for trip: {validated_id}",
                )

        # Acquire lock
        lock = asyncio.Lock()
        _processing_locks[validated_id] = lock

        try:
            async with lock:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    ThreadPoolExecutor(max_workers=4),
                    _run_map_matching_pipeline_sync,
                    validated_id,
                    split,
                    search_radius_m or settings.frontier_search_radius_m,
                    max_candidates or settings.max_candidates,
                    gps_gap_seconds or settings.gps_gap_seconds,
                )
                return MapMatchingResponse(**result)
        finally:
            if validated_id in _processing_locks:
                del _processing_locks[validated_id]