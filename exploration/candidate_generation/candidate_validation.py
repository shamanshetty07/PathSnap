"""Candidate generation quality evaluation.

Evaluates candidate-generation quality for training trips using the reference
route. The reference route is used *only* for evaluation, never for generating
candidates.

Because there is no observation-to-route alignment (GPS observation indices do
not correspond to reference-route arc indices), we use two defensible
coverage metrics that do not require such an alignment:

1. Reference-arc coverage: the fraction of unique reference-route arc IDs that
   appear in at least one observation's candidate set. This answers: "are the
   reference-route arcs discoverable by candidate generation?"

2. Observation reference coverage: the fraction of GPS observations that have
   at least one candidate arc belonging to the reference route. This answers:
   "are the GPS observations near the reference route?"

A low score on either metric does not necessarily indicate a bug: GPS
observations are noisy and may not fall within the search radius of the exact
reference arc. These metrics flag gross failures (e.g. search radius too
small, index broken) rather than measuring map-matching accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

from .candidate_generator import CandidateGenerationResult


@dataclass
class ReferenceCoverageReport:
    """Outcome of evaluating candidate coverage against the reference route."""

    trip_id: str
    split: str
    num_reference_arcs: int
    unique_reference_arcs: int
    reference_arcs_covered: int
    reference_arc_coverage: float
    num_observations: int
    observations_with_reference_candidate: int
    observation_reference_coverage: float
    num_candidates: int
    candidate_arcs_in_reference: int
    candidate_reference_overlap: float
    search_radius_m: float
    max_candidates: int
    messages: list[str] = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = []


def evaluate_reference_coverage(
    result: CandidateGenerationResult,
    reference_arc_ids: list[int],
) -> ReferenceCoverageReport:
    """Evaluate how well candidates cover the reference route.

    Parameters
    ----------
    result : CandidateGenerationResult
        Candidate-generation output for a trip.
    reference_arc_ids : list[int]
        Ordered reference arc IDs from ``.route`` (training data only).

    Returns
    -------
    ReferenceCoverageReport
        Coverage metrics and messages.
    """
    messages: list[str] = []
    reference_set = set(reference_arc_ids)
    num_reference_arcs = len(reference_arc_ids)
    unique_reference_arcs = len(reference_set)

    # Candidate arc IDs present in the generated candidates.
    candidate_arc_ids = {c.arc_id for c in result.candidates}
    num_candidates = len(result.candidates)

    # 1. Reference-arc coverage: fraction of unique reference arcs that appear
    #    in at least one observation's candidate set.
    covered_reference_arcs = reference_set & candidate_arc_ids
    reference_arcs_covered = len(covered_reference_arcs)
    if unique_reference_arcs > 0:
        reference_arc_coverage = reference_arcs_covered / unique_reference_arcs
    else:
        reference_arc_coverage = 0.0

    if reference_arc_coverage < 0.5:
        messages.append(
            f"Low reference-arc coverage ({reference_arc_coverage:.2%}): consider "
            f"increasing search radius (currently {result.search_radius_m} m)."
        )

    # 2. Observation reference coverage: fraction of observations that have at
    #    least one candidate arc belonging to the reference route.
    obs_with_ref = {
        c.observation_index
        for c in result.candidates
        if c.arc_id in reference_set
    }
    observations_with_reference_candidate = len(obs_with_ref)
    if result.num_observations > 0:
        observation_reference_coverage = (
            observations_with_reference_candidate / result.num_observations
        )
    else:
        observation_reference_coverage = 0.0

    if observation_reference_coverage < 0.5:
        messages.append(
            f"Low observation reference coverage "
            f"({observation_reference_coverage:.2%}): most GPS observations are "
            f"not within {result.search_radius_m} m of any reference-route arc."
        )

    # 3. Candidate-reference overlap: fraction of all candidates whose arc is
    #    on the reference route.
    candidate_arcs_in_reference = sum(
        1 for c in result.candidates if c.arc_id in reference_set
    )
    if num_candidates > 0:
        candidate_reference_overlap = candidate_arcs_in_reference / num_candidates
    else:
        candidate_reference_overlap = 0.0

    return ReferenceCoverageReport(
        trip_id=result.trip_id,
        split=result.split,
        num_reference_arcs=num_reference_arcs,
        unique_reference_arcs=unique_reference_arcs,
        reference_arcs_covered=reference_arcs_covered,
        reference_arc_coverage=reference_arc_coverage,
        num_observations=result.num_observations,
        observations_with_reference_candidate=observations_with_reference_candidate,
        observation_reference_coverage=observation_reference_coverage,
        num_candidates=num_candidates,
        candidate_arcs_in_reference=candidate_arcs_in_reference,
        candidate_reference_overlap=candidate_reference_overlap,
        search_radius_m=result.search_radius_m,
        max_candidates=result.max_candidates,
        messages=messages,
    )
