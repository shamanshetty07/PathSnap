"""Health check API endpoint."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get(
    "/api/health",
    response_model=dict,
    summary="Health check",
    description="Check whether the PathSnap API backend is operational.",
)
def health_check() -> dict:
    """Return the health status of the PathSnap API."""
    return {
        "status": "healthy",
        "service": "PathSnap API",
        "version": "1.0.0",
    }