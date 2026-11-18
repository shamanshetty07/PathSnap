"""Complete PathSnap inference pipeline.

GPS observations
  -> Candidate generation
  -> Feature engineering
  -> Random Forest emission probabilities
  -> HMM + Viterbi
  -> Reconstructed route

The reference route is not used for inference.
"""

from __future__ import annotations

import argparse
import joblib
from pathlib import Path

import numpy as np
import pandas as pd

from exploration.data_loader import load_trip
from exploration.candidate_generation.candidate_generator import generate_candidates

from features.spatial_features import compute_spatial_features
from features.temporal_features import compute_temporal_features
from features.network_features import compute_network_features
from features.transition_features import compute_transition_features

from models.hmm import RoadNetwork, HiddenMarkovMapMatcher


MODEL_PATH = Path("outputs/models/random_forest.joblib")
OUTPUT_DIR = Path("outputs/map_matching")


def load_trip_data(
    split: str,
    trip_id: str,
    base_dir: str = "Data",
) -> tuple:
    """Load trip data and prepare GPS observations."""

    print("1. Loading trip...")

    trip_data = load_trip(
        split=split,
        trip_id=trip_id,
        base_dir=base_dir,
    )

    observations = trip_data.track.copy().reset_index(drop=True)
    observations["observation_index"] = np.arange(len(observations))

    observations = observations.rename(
        columns={
            "longitude": "gps_longitude",
            "latitude": "gps_latitude",
        }
    )

    required = {
        "observation_index",
        "timestamp",
        "gps_longitude",
        "gps_latitude",
    }

    missing = required - set(observations.columns)

    if missing:
        raise ValueError(f"GPS track is missing columns: {sorted(missing)}")

    observations = observations[
        [
            "observation_index",
            "timestamp",
            "gps_longitude",
            "gps_latitude",
        ]
    ]

    print(f"   Loaded {len(observations)} GPS observations")

    return trip_data, observations


def generate_candidates_for_trip(
    trip_data,
    search_radius_m: float = 50.0,
    max_candidates: int = 5,
) -> pd.DataFrame:
    """Generate road-segment candidates for GPS observations."""

    print("2. Generating candidates...")

    result = generate_candidates(
        trip_data,
        search_radius_m=search_radius_m,
        max_candidates=max_candidates,
    )

    candidates = pd.DataFrame(
        [candidate.to_dict() for candidate in result.candidates]
    )

    if candidates.empty:
        print("   No candidates generated")
        return candidates

    candidates["trip_id"] = trip_data.trip_id

    candidates = candidates.rename(
        columns={
            "gps_lon": "gps_longitude",
            "gps_lat": "gps_latitude",
        }
    )

    print(f"   Generated {len(candidates)} candidates")

    return candidates


def generate_features(
    candidates: pd.DataFrame,
    trip_data,
    track: pd.DataFrame,
) -> pd.DataFrame:
    """Generate the same features used during model training."""

    print("3. Generating features...")

    df = candidates.copy()

    if "gps_longitude" not in df.columns and "gps_lon" in df.columns:
        df = df.rename(columns={"gps_lon": "gps_longitude"})

    if "gps_latitude" not in df.columns and "gps_lat" in df.columns:
        df = df.rename(columns={"gps_lat": "gps_latitude"})

    df = compute_spatial_features(
        df,
        nodes_df=trip_data.nodes,
    )

    df = compute_temporal_features(
        df,
        track=trip_data.track,
    )

    df = compute_network_features(
        df,
        arcs_df=trip_data.arcs,
        nodes_df=trip_data.nodes,
    )

    df = compute_transition_features(
        df,
        arcs_df=trip_data.arcs,
        nodes_df=trip_data.nodes,
    )

    expected_prefixes = (
        "spatial_",
        "temporal_",
        "network_",
        "transition_",
    )

    actual_features = [
        column
        for column in df.columns
        if column.startswith(expected_prefixes)
    ]

    print(f"   Generated {len(actual_features)} engineered features")

    return df


def load_random_forest_model(model_path: Path = MODEL_PATH):
    """Load the trained Random Forest pipeline."""

    print("4. Loading Random Forest model...")

    data = joblib.load(model_path)

    pipeline = data["pipeline"]
    features = data["features"]

    print(f"   Model features: {len(features)} columns")
    print(f"   Pipeline type: {type(pipeline).__name__}")

    return pipeline, features


def predict_emission_probabilities(
    pipeline,
    features: list[str],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Predict candidate probabilities using the trained classifier."""

    print("5. Predicting emission probabilities...")

    missing = [column for column in features if column not in df.columns]

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    df_features = df[features].copy()

    probabilities = pipeline.predict_proba(df_features)

    classes = list(pipeline.classes_)

    if 1 not in classes:
        raise ValueError("Trained model does not contain positive class 1")

    positive_class_index = classes.index(1)

    proba = probabilities[:, positive_class_index]

    df = df.copy()
    df["emission_probability"] = proba

    print(f"   Predicted probabilities for {len(df)} candidates")

    return df


def build_road_network(trip_data) -> RoadNetwork:
    """Build the directed road network used by the HMM."""

    print("6. Building road network...")

    nodes = trip_data.nodes.copy()

    if "node_id" not in nodes.columns:
        nodes = nodes.reset_index().rename(
            columns={"index": "node_id"}
        )

    arcs = trip_data.arcs.copy()

    arcs = arcs.rename(
        columns={
            "from_node": "from_node_id",
            "to_node": "to_node_id",
        }
    )

    network = RoadNetwork(
        arcs=arcs,
        nodes=nodes,
    )

    print(
        f"   Road network: {network.node_count} nodes, "
        f"{network.edge_count} arcs"
    )

    return network


def _ensure_output_metadata(
    decoded: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Restore timestamps and explicit status flags."""

    result = decoded.copy()

    if "observation_index" not in result.columns:
        raise ValueError("HMM output is missing observation_index")

    result["observation_index"] = result["observation_index"].astype(int)

    metadata = observations[
        ["observation_index", "timestamp"]
    ].drop_duplicates("observation_index")

    result = result.drop(
        columns=["timestamp"],
        errors="ignore",
    ).merge(
        metadata,
        on="observation_index",
        how="left",
        validate="many_to_one",
    )

    default_flags = {
        "decoded": False,
        "no_candidate": False,
        "fallback_match": False,
        "decode_failed": False,
    }

    for column, default in default_flags.items():
        if column not in result.columns:
            result[column] = default

        result[column] = result[column].fillna(default).astype(bool)

    return result.sort_values(
        "observation_index"
    ).reset_index(drop=True)


def run_hmm_viterbi(
    network: RoadNetwork,
    candidates: pd.DataFrame,
    observations: pd.DataFrame,
    gps_gap_seconds: float = 30.0,
) -> pd.DataFrame:
    """Run HMM/Viterbi and preserve candidate-bearing observations."""

    print("7. Running HMM/Viterbi...")

    matcher = HiddenMarkovMapMatcher(network)

    candidates_for_hmm = candidates.copy()

    candidates_for_hmm = candidates_for_hmm.rename(
        columns={
            "gps_longitude": "gps_lon",
            "gps_latitude": "gps_lat",
        }
    )

    observations_for_hmm = observations.copy()

    observations_for_hmm = observations_for_hmm.rename(
        columns={
            "gps_longitude": "gps_lon",
            "gps_latitude": "gps_lat",
        }
    )

    required_cols = [
        "observation_index",
        "arc_id",
        "from_node_id",
        "to_node_id",
        "distance_along_segment_m",
        "segment_length_m",
        "gps_lon",
        "gps_lat",
        "timestamp",
        "emission_probability",
    ]

    missing = [
        column
        for column in required_cols
        if column not in candidates_for_hmm.columns
    ]

    if missing:
        raise ValueError(f"Candidates are missing HMM columns: {missing}")

    candidates_subset = candidates_for_hmm[required_cols].copy()

    candidates_subset["observation_index"] = (
        candidates_subset["observation_index"].astype(int)
    )

    try:
        decoded = matcher.decode(
            candidates_subset,
            observations=observations_for_hmm,
        )

        decoded = _ensure_output_metadata(
            decoded,
            observations_for_hmm,
        )

        print(
            "   HMM decoded:",
            int(decoded["decoded"].sum()),
            "observations",
        )

        return decoded

    except ValueError as exc:

        if "No valid transition" not in str(exc):
            raise

        print(f"   HMM transition error: {exc}")
        print("   Splitting at invalid transitions...")

        decoded = _handle_invalid_transitions(
            matcher,
            candidates_subset,
            observations_for_hmm,
            gps_gap_seconds,
        )

        decoded = _ensure_output_metadata(
            decoded,
            observations_for_hmm,
        )

        print(
            f"   HMM decoded: {int(decoded['decoded'].sum())}"
        )

        print(
            f"   Emission fallbacks: {int(decoded['fallback_match'].sum())}"
        )

        print(
            f"   Genuine missing candidates: {int(decoded['no_candidate'].sum())}"
        )

        return decoded


def _handle_invalid_transitions(
    matcher: HiddenMarkovMapMatcher,
    candidates: pd.DataFrame,
    observations: pd.DataFrame,
    gps_gap_seconds: float = 30.0,
) -> pd.DataFrame:
    """Split sequences when no candidate remains reachable from the start."""

    candidate_groups = {
        int(index): group.sort_values(
            "emission_probability",
            ascending=False,
        ).reset_index(drop=True)
        for index, group in candidates.groupby(
            "observation_index",
            sort=False,
        )
    }

    output_rows = []
    current_segment_groups: list[pd.DataFrame] = []
    reachable_candidates: pd.DataFrame | None = None

    sequence_id = 0
    previous_observation = None

    def flush_segment():
        nonlocal current_segment_groups, reachable_candidates, sequence_id

        if not current_segment_groups:
            reachable_candidates = None
            return

        rows = _decode_segment(
            matcher,
            current_segment_groups,
            observations,
            sequence_id=sequence_id,
        )

        output_rows.extend(rows)

        sequence_id += 1
        current_segment_groups = []
        reachable_candidates = None

    ordered_observations = observations.sort_values(
        "observation_index"
    )

    for _, observation in ordered_observations.iterrows():

        obs_index = int(observation["observation_index"])
        group = candidate_groups.get(obs_index)

        # Split at invalid or unusually large GPS time gaps.
        if previous_observation is not None:

            previous_time = pd.to_numeric(
                pd.Series([previous_observation["timestamp"]]),
                errors="coerce",
            ).iloc[0]

            current_time = pd.to_numeric(
                pd.Series([observation["timestamp"]]),
                errors="coerce",
            ).iloc[0]

            if pd.notna(previous_time) and pd.notna(current_time):

                delta_time = float(current_time - previous_time)

                if delta_time <= 0 or delta_time > gps_gap_seconds:
                    flush_segment()

        # Preserve observations for which candidate generation failed.
        if group is None or group.empty:

            flush_segment()

            output_rows.append({
                "observation_index": obs_index,
                "arc_id": np.nan,
                "emission_probability": 0.0,
                "decoded": False,
                "no_candidate": True,
                "fallback_match": False,
                "decode_failed": False,
                "sequence_id": sequence_id,
            })

            previous_observation = observation
            continue

        # The first observation initializes all reachable states.
        if not current_segment_groups:

            current_segment_groups.append(group)
            reachable_candidates = group.copy()

            previous_observation = observation
            continue

        assert reachable_candidates is not None

        # Keep only current candidates reachable from a previously
        # reachable candidate. This tracks complete path reachability,
        # rather than merely checking adjacent-group connectivity.
        reachable_indices = []

        for current_index, current_candidate in group.iterrows():

            is_reachable = False

            for _, previous_candidate in reachable_candidates.iterrows():

                transition_score = matcher.transition_log_probability(
                    previous_candidate,
                    current_candidate,
                )

                if np.isfinite(transition_score):
                    is_reachable = True
                    break

            if is_reachable:
                reachable_indices.append(current_index)

        if not reachable_indices:

            # No candidate can continue the current sequence.
            flush_segment()

            # Start a new sequence at the current observation.
            current_segment_groups.append(group)
            reachable_candidates = group.copy()

        else:

            current_segment_groups.append(group)

            reachable_candidates = group.loc[
                reachable_indices
            ].reset_index(drop=True)

        previous_observation = observation

    flush_segment()

    return pd.DataFrame(output_rows)


def _decode_segment(
    matcher: HiddenMarkovMapMatcher,
    segment_groups: list[pd.DataFrame],
    observations: pd.DataFrame,
    sequence_id: int = 0,
) -> list[dict]:
    """Decode one continuous sequence.

    If Viterbi fails, retain the best emission candidate as a fallback.
    Such a result is not reported as an HMM-decoded match.
    """

    try:

        decoded = matcher.viterbi_decode(segment_groups)

        if isinstance(decoded, pd.DataFrame):
            rows = decoded.to_dict("records")
        else:
            rows = list(decoded)

        for row in rows:

            row.setdefault("decoded", True)
            row.setdefault("no_candidate", False)
            row.setdefault("fallback_match", False)
            row.setdefault("decode_failed", False)

            row["sequence_id"] = sequence_id

        return rows

    except ValueError as exc:

        print(
            f"   Viterbi failed for sequence {sequence_id}: {exc}"
        )

        fallback_rows = []

        for group in segment_groups:

            if group.empty:
                continue

            best = group.sort_values(
                "emission_probability",
                ascending=False,
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


def save_results(
    decoded: pd.DataFrame,
    split: str,
    trip_id: str,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Save the reconstructed route."""

    print("8. Saving reconstructed route...")

    out_path = output_dir / split / trip_id / "matched_route.csv"

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_cols = [
        "observation_index",
        "timestamp",
        "arc_id",
        "emission_probability",
        "decoded",
        "no_candidate",
        "fallback_match",
        "decode_failed",
        "sequence_id",
    ]

    available_output = [
        column
        for column in output_cols
        if column in decoded.columns
    ]

    result = decoded[available_output].copy()

    result = result.sort_values(
        "observation_index"
    ).reset_index(drop=True)

    result.to_csv(
        out_path,
        index=False,
        float_format="%.6g",
    )

    print(f"   Saved {len(result)} rows to {out_path}")

    return out_path


def run_pipeline(
    split: str,
    trip_id: str,
    base_dir: str = "Data",
    model_path: Path = MODEL_PATH,
    search_radius_m: float = 50.0,
    max_candidates: int = 5,
    gps_gap_seconds: float = 30.0,
) -> Path:
    """Run the complete map-matching inference pipeline."""

    trip_data, observations = load_trip_data(
        split,
        trip_id,
        base_dir,
    )

    candidates = generate_candidates_for_trip(
        trip_data,
        search_radius_m=search_radius_m,
        max_candidates=max_candidates,
    )

    if candidates.empty:
        raise ValueError("No candidates generated for this trip")

    df = generate_features(
        candidates,
        trip_data,
        trip_data.track,
    )

    pipeline, features = load_random_forest_model(model_path)

    df = predict_emission_probabilities(
        pipeline,
        features,
        df,
    )

    network = build_road_network(trip_data)

    decoded = run_hmm_viterbi(
        network,
        df,
        observations,
        gps_gap_seconds=gps_gap_seconds,
    )

    out_path = save_results(
        decoded,
        split,
        trip_id,
    )

    return out_path


def main(argv=None):

    parser = argparse.ArgumentParser(
        description="PathSnap map matching inference pipeline"
    )

    parser.add_argument(
        "--split",
        required=True,
        help="Dataset split: train or student_test",
    )

    parser.add_argument(
        "--trip",
        required=True,
        help="Trip identifier",
    )

    parser.add_argument(
        "--base-dir",
        default="Data",
    )

    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
    )

    parser.add_argument(
        "--search-radius",
        type=float,
        default=50.0,
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--gps-gap-seconds",
        type=float,
        default=30.0,
    )

    args = parser.parse_args(argv)

    out_path = run_pipeline(
        split=args.split,
        trip_id=args.trip,
        base_dir=args.base_dir,
        model_path=Path(args.model),
        search_radius_m=args.search_radius,
        max_candidates=args.max_candidates,
        gps_gap_seconds=args.gps_gap_seconds,
    )

    print(f"\nPipeline complete. Output: {out_path}")


if __name__ == "__main__":
    main()