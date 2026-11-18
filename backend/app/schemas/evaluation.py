"""Schema for evaluation metrics."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    """Evaluation metrics when reference labels are available."""
    trip_id: str = Field(..., description="Trip identifier")
    observation_top1_accuracy: float | None = Field(
        None, description="Observation-level Top-1 accuracy"
    )
    precision: float | None = Field(
        None, description="Precision at threshold 0.5"
    )
    recall: float | None = Field(
        None, description="Recall at threshold 0.5"
    )
    f1_score: float | None = Field(
        None, description="F1 score at threshold 0.5"
    )
    candidate_miss_rate: float | None = Field(
        None, description="Rate of observations without candidates"
    )
    route_continuity: float | None = Field(
        None, description="Route continuity metric"
    )
    sequence_level_accuracy: float | None = Field(
        None, description="Sequence-level decoding accuracy"
    )


class EvaluationAvailability(BaseModel):
    """Whether evaluation is available for a trip."""
    available: bool = Field(
        ..., description="Whether reference labels are available"
    )
    reason: str | None = Field(
        None, description="Reason if not available (e.g., 'no reference data', 'evaluation disabled')"
    )