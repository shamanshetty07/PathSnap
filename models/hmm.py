"""Hidden Markov Model utilities for PathSnap map matching.

The Random Forest supplies emission probabilities.
Road-network connectivity and GPS movement supply transition probabilities.

This module does not use reference routes or .route labels during inference.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict

import numpy as np
import pandas as pd


EARTH_RADIUS_M = 6_371_000.0
EPSILON = 1e-12


def haversine_distance(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> float:
    """Calculate geodesic distance between two longitude/latitude points."""
    lon1, lat1, lon2, lat2 = map(
        math.radians, [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


class RoadNetwork:
    """Directed weighted road graph with bounded Dijkstra search."""

    def __init__(
        self,
        arcs: pd.DataFrame,
        nodes: pd.DataFrame,
    ) -> None:

        required_arcs = {"from_node_id", "to_node_id"}
        required_nodes = {"node_id", "longitude", "latitude"}

        if not required_arcs.issubset(arcs.columns):
            raise ValueError(
                f"Arcs dataframe must contain: {required_arcs}"
            )

        if not required_nodes.issubset(nodes.columns):
            raise ValueError(
                f"Nodes dataframe must contain: {required_nodes}"
            )

        if nodes.empty:
            raise ValueError("Nodes dataframe is empty")

        node_table = nodes[
            ["node_id", "longitude", "latitude"]
        ].copy()

        for column in ("node_id", "longitude", "latitude"):
            node_table[column] = pd.to_numeric(
                node_table[column],
                errors="coerce",
            )

        if node_table.isna().any().any():
            raise ValueError("Nodes contain invalid IDs or coordinates")

        if node_table["node_id"].duplicated().any():
            raise ValueError("Duplicate node IDs found")

        invalid_coordinates = (
            ~node_table["longitude"].between(-180, 180)
            | ~node_table["latitude"].between(-90, 90)
        )

        if invalid_coordinates.any():
            raise ValueError("Invalid longitude or latitude values")

        self.coordinates = {
            int(row.node_id): (
                float(row.longitude),
                float(row.latitude),
            )
            for row in node_table.itertuples(index=False)
        }

        self.graph: dict[int, list[tuple[int, float]]] = defaultdict(list)

        self.edge_count = 0
        self.skipped_arc_count = 0

        for row in arcs.itertuples(index=False):

            if pd.isna(row.from_node_id) or pd.isna(row.to_node_id):
                self.skipped_arc_count += 1
                continue

            source = int(row.from_node_id)
            target = int(row.to_node_id)

            if (
                source not in self.coordinates
                or target not in self.coordinates
            ):
                self.skipped_arc_count += 1
                continue

            lon1, lat1 = self.coordinates[source]
            lon2, lat2 = self.coordinates[target]

            length = haversine_distance(
                lon1,
                lat1,
                lon2,
                lat2,
            )

            self.graph[source].append((target, length))
            self.edge_count += 1

        self._distance_cache: dict[tuple[int, int], float] = {}

    @property
    def node_count(self) -> int:
        return len(self.coordinates)

    def shortest_distance(
        self,
        source: int,
        target: int,
        max_distance: float = math.inf,
    ) -> float:
        """Return directed shortest-path distance in metres."""

        source = int(source)
        target = int(target)
        max_distance = float(max_distance)

        if max_distance < 0:
            return math.inf

        if (
            source not in self.coordinates
            or target not in self.coordinates
        ):
            return math.inf

        if source == target:
            return 0.0

        cache_key = (source, target)

        if cache_key in self._distance_cache:
            distance = self._distance_cache[cache_key]

            return (
                distance
                if distance <= max_distance
                else math.inf
            )

        distances = {source: 0.0}
        queue = [(0.0, source)]

        while queue:

            distance, node = heapq.heappop(queue)

            if distance > distances.get(node, math.inf):
                continue

            if distance > max_distance:
                break

            if node == target:
                self._distance_cache[cache_key] = distance
                return distance

            for neighbour, edge_length in self.graph.get(node, []):

                new_distance = distance + edge_length

                if new_distance > max_distance:
                    continue

                if new_distance < distances.get(
                    neighbour,
                    math.inf,
                ):
                    distances[neighbour] = new_distance

                    heapq.heappush(
                        queue,
                        (new_distance, neighbour),
                    )

        return math.inf


class HiddenMarkovMapMatcher:
    """HMM map matcher using candidate emissions and road transitions."""

    def __init__(
        self,
        network: RoadNetwork,
        emission_floor: float = 1e-8,
        transition_sigma_m: float = 50.0,
        max_speed_mps: float = 55.0,
        speed_tolerance_m: float = 30.0,
        disconnected_penalty: float = 15.0,
        gps_gap_seconds: float = 30.0,
    ) -> None:

        self.network = network
        self.emission_floor = emission_floor
        self.transition_sigma_m = transition_sigma_m
        self.max_speed_mps = max_speed_mps
        self.speed_tolerance_m = speed_tolerance_m
        self.disconnected_penalty = disconnected_penalty
        self.gps_gap_seconds = gps_gap_seconds

    @staticmethod
    def _validate_candidates(candidates: pd.DataFrame) -> None:

        required = {
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
        }

        missing = required - set(candidates.columns)

        if missing:
            raise ValueError(
                f"Candidate dataframe is missing columns: {sorted(missing)}"
            )

    def emission_log_probability(
        self,
        candidates: pd.DataFrame,
    ) -> np.ndarray:
        """Convert Random Forest probabilities into log emission scores."""

        probabilities = candidates[
            "emission_probability"
        ].to_numpy(dtype=float)

        probabilities = np.clip(
            probabilities,
            self.emission_floor,
            1.0,
        )

        return np.log(probabilities)

    def _network_distance(
        self,
        previous: pd.Series,
        current: pd.Series,
        max_distance: float,
    ) -> float:
        """Return directed network distance between two candidate positions."""

        previous_arc = int(previous.arc_id)
        current_arc = int(current.arc_id)

        previous_along = float(
            previous.distance_along_segment_m
        )

        current_along = float(
            current.distance_along_segment_m
        )

        previous_length = float(previous.segment_length_m)

        # Both observations lie on the same directed arc.
        if previous_arc == current_arc:

            difference = current_along - previous_along

            # Allow small backward movement caused by GPS noise.
            if difference >= -10.0:

                distance = (
                    max(0.0, difference)
                    + 2.0 * max(0.0, -difference)
                )

                return (
                    distance
                    if distance <= max_distance
                    else math.inf
                )

            # A substantial backward move on the same directed arc
            # is invalid.
            return math.inf

        previous_end = int(previous.to_node_id)
        current_start = int(current.from_node_id)

        remaining_previous = max(
            0.0,
            previous_length - previous_along,
        )

        current_progress = max(
            0.0,
            current_along,
        )

        available = (
            max_distance
            - remaining_previous
            - current_progress
        )

        if available < 0:
            return math.inf

        middle_distance = self.network.shortest_distance(
            previous_end,
            current_start,
            available,
        )

        if not math.isfinite(middle_distance):
            return math.inf

        total_distance = (
            remaining_previous
            + middle_distance
            + current_progress
        )

        return (
            total_distance
            if total_distance <= max_distance
            else math.inf
        )

    def transition_log_probability(
        self,
        previous: pd.Series,
        current: pd.Series,
    ) -> float:
        """Calculate the log transition score between two candidate states."""

        gps_distance = haversine_distance(
            float(previous.gps_lon),
            float(previous.gps_lat),
            float(current.gps_lon),
            float(current.gps_lat),
        )

        delta_time = (
            float(current.timestamp)
            - float(previous.timestamp)
        )

        # Invalid or non-increasing timestamps.
        if delta_time <= 0:
            return -math.inf

        max_distance = (
            self.max_speed_mps * delta_time
            + self.speed_tolerance_m
        )

        network_distance = self._network_distance(
            previous,
            current,
            max_distance,
        )

        # No valid directed path or physically possible transition.
        if not math.isfinite(network_distance):
            return -math.inf

        difference = abs(
            network_distance - gps_distance
        )

        # Gaussian-style transition likelihood.
        score = -0.5 * (
            difference / self.transition_sigma_m
        ) ** 2

        network_speed = network_distance / delta_time

        if network_speed > self.max_speed_mps:
            score -= self.disconnected_penalty

        return float(score)

    @staticmethod
    def _normalise_log_scores(
        scores: np.ndarray,
    ) -> np.ndarray:
        """Numerically stable log-softmax."""

        if len(scores) == 0:
            return scores.astype(float)

        maximum = np.max(scores)

        if not np.isfinite(maximum):
            return np.full_like(
                scores,
                -math.log(len(scores)),
                dtype=float,
            )

        shifted = scores - maximum

        return shifted - np.log(
            np.exp(shifted).sum()
        )

    def viterbi_decode(
        self,
        candidate_groups: list[pd.DataFrame],
    ) -> pd.DataFrame:
        """Decode one continuous sequence of GPS candidate groups with Viterbi."""

        if not candidate_groups:
            return pd.DataFrame()

        groups = []

        for index, group in enumerate(candidate_groups):

            if group is None or group.empty:
                raise ValueError(
                    f"Candidate group {index} is empty; "
                    "split sequences before decoding"
                )

            groups.append(
                group.reset_index(drop=True)
            )

        # Initial emission probabilities.
        scores = np.asarray(
            self.emission_log_probability(groups[0]),
            dtype=float,
        )

        if scores.shape != (len(groups[0]),):
            raise ValueError(
                "Emission function must return one score per candidate"
            )

        if not np.isfinite(scores).any():
            raise ValueError(
                "First candidate group has no finite emission scores"
            )

        backpointers = [
            np.full(len(groups[0]), -1, dtype=int)
        ]

        # Viterbi forward pass.
        for time_index in range(1, len(groups)):

            previous_group = groups[time_index - 1]
            current_group = groups[time_index]

            emissions = np.asarray(
                self.emission_log_probability(current_group),
                dtype=float,
            )

            if emissions.shape != (len(current_group),):
                raise ValueError(
                    "Emission function must return one score per candidate"
                )

            next_scores = np.full(
                len(current_group),
                -np.inf,
                dtype=float,
            )

            pointers = np.full(
                len(current_group),
                -1,
                dtype=int,
            )

            for current_position, (_, current_candidate) in enumerate(
                current_group.iterrows()
            ):

                best_score = -np.inf
                best_previous = -1

                for previous_position, (_, previous_candidate) in enumerate(
                    previous_group.iterrows()
                ):

                    if not np.isfinite(scores[previous_position]):
                        continue

                    transition_score = self.transition_log_probability(
                        previous_candidate,
                        current_candidate,
                    )

                    if not np.isfinite(transition_score):
                        continue

                    total_score = (
                        scores[previous_position]
                        + transition_score
                        + emissions[current_position]
                    )

                    if total_score > best_score:
                        best_score = total_score
                        best_previous = previous_position

                next_scores[current_position] = best_score
                pointers[current_position] = best_previous

            if not np.isfinite(next_scores).any():
                raise ValueError(
                    "No valid transition between candidate groups "
                    f"{time_index - 1} and {time_index}; split the sequence"
                )

            scores = next_scores
            backpointers.append(pointers)

        # Backtracking.
        state = int(np.argmax(scores))
        best_path = [state]

        for time_index in range(len(groups) - 1, 0, -1):

            state = int(backpointers[time_index][state])

            if state < 0:
                raise RuntimeError(
                    "Broken Viterbi backpointer at observation group "
                    f"{time_index}"
                )

            best_path.append(state)

        best_path.reverse()

        decoded_rows = []

        for group, candidate_position in zip(groups, best_path):

            selected = group.iloc[candidate_position].to_dict()
            selected["decoded"] = True

            decoded_rows.append(selected)

        return pd.DataFrame(decoded_rows)

    def decode(
        self,
        candidates: pd.DataFrame,
        observations: pd.DataFrame | None = None,
        gps_gap_seconds: float = 30.0,
    ) -> pd.DataFrame:
        """Decode candidates while splitting at missing observations or invalid transitions."""

        if gps_gap_seconds <= 0:
            raise ValueError(
                "gps_gap_seconds must be greater than zero"
            )

        self._validate_candidates(candidates)

        # Prepare the complete GPS observation list.
        if observations is not None:

            required = {
                "observation_index",
                "timestamp",
                "gps_lon",
                "gps_lat",
            }

            missing = required - set(observations.columns)

            if missing:
                raise ValueError(
                    f"Missing observation columns: {sorted(missing)}"
                )

            if observations["observation_index"].duplicated().any():
                raise ValueError(
                    "Duplicate observation indices"
                )

            observation_table = observations.sort_values(
                "observation_index"
            ).reset_index(drop=True)

        else:

            observation_table = (
                candidates[
                    [
                        "observation_index",
                        "timestamp",
                        "gps_lon",
                        "gps_lat",
                    ]
                ]
                .drop_duplicates("observation_index")
                .sort_values("observation_index")
                .reset_index(drop=True)
            )

        # Group candidates by GPS observation.
        candidate_groups = {
            int(index): group.sort_values(
                "arc_id"
            ).reset_index(drop=True)

            for index, group in candidates.groupby(
                "observation_index",
                sort=False,
            )
        }

        output_rows = []
        current_segment = []
        previous_observation = None
        sequence_id = 0

        def flush_segment() -> None:

            nonlocal current_segment, sequence_id

            if not current_segment:
                return

            decoded = self.viterbi_decode(
                [group for _, group in current_segment]
            )

            for (_, group), (_, decoded_row) in zip(
                current_segment,
                decoded.iterrows(),
            ):

                row = decoded_row.to_dict()

                row["decoded"] = True
                row["no_candidate"] = False
                row["sequence_id"] = sequence_id

                output_rows.append(row)

            sequence_id += 1
            current_segment = []

        # Process observations chronologically.
        for _, observation in observation_table.iterrows():

            observation_index = int(
                observation["observation_index"]
            )

            group = candidate_groups.get(observation_index)

            # Handle observations without candidates.
            if group is None or group.empty:

                flush_segment()

                missing_row = observation.to_dict()

                missing_row["decoded"] = False
                missing_row["no_candidate"] = True
                missing_row["sequence_id"] = None

                output_rows.append(missing_row)

                previous_observation = observation
                continue

            # Handle large GPS gaps or invalid timestamps.
            if previous_observation is not None:

                delta_time = (
                    float(observation["timestamp"])
                    - float(previous_observation["timestamp"])
                )

                if (
                    delta_time <= 0
                    or delta_time > gps_gap_seconds
                ):
                    flush_segment()

            # Check whether any transition connects the two groups.
            if current_segment:

                previous_group = current_segment[-1][1]

                valid_transition = any(
                    np.isfinite(
                        self.transition_log_probability(
                            previous_candidate,
                            current_candidate,
                        )
                    )
                    for _, previous_candidate in previous_group.iterrows()
                    for _, current_candidate in group.iterrows()
                )

                if not valid_transition:
                    flush_segment()

            current_segment.append(
                (observation_index, group)
            )

            previous_observation = observation

        # Decode the final segment.
        flush_segment()

        result = pd.DataFrame(output_rows)

        if not result.empty:
            result = result.sort_values(
                "observation_index"
            ).reset_index(drop=True)

        return result