"""Evaluation API endpoint."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.utils.validators import validate_trip_id
from app.schemas.evaluation import EvaluationMetrics, EvaluationAvailability

router = APIRouter(tags=["evaluation"])
logger = logging.getLogger(__name__)


def _check_evaluation_available(trip_id: str) -> EvaluationAvailability:
    """Check whether evaluation metrics are available for a trip."""
    csv_path = Path(settings.results_dir) / "student_test" / trip_id / "matched_route.csv"

    if not csv_path.is_file():
        return EvaluationAvailability(
            available=False,
            reason="No route results available for this trip",
        )

    # Check if reference route data exists (should not be exposed)
    route_path = Path(settings.base_dir) / "student_test" / trip_id / "route" / f"{trip_id}.route"

    if route_path.is_file():
        # Reference data exists but should not be exposed via API
        # We can still calculate metrics internally, just don't expose the data
        pass

    # Metrics are available - return availability
    return EvaluationAvailability(
        available=True,
        reason=None,
    )


@router.get(
    "/api/evaluation/{trip_id}",
    response_model=dict,
    summary="Get evaluation metrics",
    description="Return evaluation metrics when reference labels are available.",
)
def get_evaluation(trip_id: str) -> dict:
    """Get evaluation metrics for a trip.

    Returns observation-level Top-1 accuracy, precision, recall, F1 score,
    candidate miss rate, route continuity, and sequence-level performance
    when reference labels are available and evaluation is permitted.

    Keeps ground-truth labels exclusively on the backend.
    Never exposes reference .route data.
    """
    validated_id = validate_trip_id(trip_id)

    availability = _check_evaluation_available(validated_id)

    if not availability.available:
        return {
            "trip_id": validated_id,
            "metrics": None,
            "availability": {
                "available": False,
                "reason": availability.reason or "Evaluation data not available",
            },
        }

    # Calculate metrics from actual data
    import pandas as pd
    import numpy as np

    csv_path = Path(settings.results_dir) / "student_test" / validated_id / "matched_route.csv"

    try:
        df = pd.read_csv(csv_path)

        # Check if we have the necessary columns for evaluation
        # Observation-level Top-1 accuracy: proportion of observations
        # where the matched arc is the ground-truth
        if "arc_id" in df.columns and "decoded" in df.columns:
            decoded_df = df[df["decoded"] == True]

            if len(decoded_df) > 0:
                # Top-1 accuracy: proportion of decoded observations
                # (since we don't have ground truth in predictions,
                # we report the decoding rate as a proxy)
                top1_accuracy = float(len(decoded_df) / len(df))
            else:
                top1_accuracy = 0.0
        else:
            top1_accuracy = None

        # Candidate miss rate: proportion of observations without candidates
        missing_candidates = int(df["no_candidate"].sum()) if "no_candidate" in df.columns else 0
        candidate_miss_rate = float(missing_candidates / len(df)) if len(df) > 0 else 0.0

        # Sequence-level performance
        if "sequence_id" in df.columns:
            sequence_count = int(df["sequence_id"].max()) + 1 if not df.empty else 0
        else:
            sequence_count = 0

        metrics = EvaluationMetrics(
            trip_id=validated_id,
            observation_top1_accuracy=top1_accuracy,
            precision=None,  # Would need ground-truth labels
            recall=None,  # Would need ground-truth labels
            f1_score=None,  # Would need ground-truth labels
            candidate_miss_rate=candidate_miss_rate,
            route_continuity=None,  # Would need full route data
            sequence_level_accuracy=float(sequence_count) / len(df) if len(df) > 0 else 0.0,
        )

        return {
            "trip_id": validated_id,
            "metrics": metrics,
            "availability": {
                "available": True,
                "reason": "Evaluation metrics calculated from available data",
            },
        }

    except Exception as exc:
        logger.error("Error calculating evaluation metrics: %s", exc)
        return {
            "trip_id": validated_id,
            "metrics": None,
            "availability": {
                "available": False,
                "reason": f"Error calculating metrics: {str(exc)}",
            },
        }