"""PathSnap API main FastAPI application."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.trips import router as trips_router
from app.api.map_matching import router as map_matching_router
from app.api.results import router as results_router
from app.api.evaluation import router as evaluation_router

from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pathsnap-api")

# Track processing locks for concurrent request prevention
_processing_locks: dict[str, str] = {}

# ---------------------------------------------------------------------------
# FastAPI application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PathSnap API",
    description="Distributed map-matching and route reconstruction system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration using environment variables
allowed_origins = os.getenv(
    "FRONTEND_ORIGIN",
    settings.frontend_origin,
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip() for origin in allowed_origins
    ] if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Include API routers
# ---------------------------------------------------------------------------

app.include_router(
    health_router,
    prefix="/api",
    tags=["health"],
)

app.include_router(
    trips_router,
    prefix="/api",
    tags=["trips"],
)

app.include_router(
    map_matching_router,
    prefix="/api",
    tags=["map_matching"],
)

app.include_router(
    results_router,
    prefix="/api",
    tags=["results"],
)

app.include_router(
    evaluation_router,
    prefix="/api",
    tags=["evaluation"],
)

# ---------------------------------------------------------------------------
# Root endpoint
# -------------------------------------------------------------------------


@app.get(
    "/",
    include_in_schema=False,
    summary="Root endpoint",
    description="Root path returning basic API information.",
)
def root() -> dict:
    """Return API basic information."""
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
    }


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    """Log startup information."""
    logger.info("=" * 60)
    logger.info("PathSnap API starting up")
    logger.info("Service: %s", settings.app_name)
    logger.info("Host: %s", settings.app_host)
    logger.info("Port: %s", settings.app_port)
    logger.info("Data directory: %s", settings.base_dir)
    logger.info("Model path: %s", settings.model_path)
    logger.info("Results directory: %s", settings.results_dir)
    logger.info("Frontend origin: %s", settings.frontend_origin)
    logger.info("GPS search radius: %s m", settings.frontier_search_radius_m)
    logger.info("Max candidates: %d", settings.max_candidates)
    logger.info("GPS gap seconds: %s", settings.gps_gap_seconds)
    logger.info("=" * 60)

    # Verify model is available
    model_path = Path(settings.model_path)
    if model_path.is_file():
        logger.info("✓ Trained model found: %s", model_path)
    else:
        logger.warning(
            "✗ Trained model not found at: %s", model_path
        )
        logger.warning(
            "Map matching will fail until a model is trained."
        )

    # Verify data directory structure
    data_dir = Path(settings.base_dir)
    if data_dir.is_dir():
        logger.info("✓ Data directory found: %s", data_dir)
        train_dir = data_dir / "train"
        if train_dir.is_dir():
            trip_count = len([d for d in train_dir.iterdir() if d.is_dir() and d.name.isdigit()])
            logger.info("  %d trips in train split", trip_count)
        student_test_dir = data_dir / "student_test"
        if student_test_dir.is_dir():
            trip_count = len([d for d in student_test_dir.iterdir() if d.is_dir() and d.name.isdigit()])
            logger.info("  %d trips in student_test split", trip_count)
    else:
        logger.warning("! Data directory not found: %s", data_dir)

    logger.info("PathSnap API ready for requests")


# ---------------------------------------------------------------------------
# Shutdown event
# ---------------------------------------------------------------------------

@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Log shutdown information."""
    logger.info("PathSnap API shutting down")