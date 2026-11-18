"""PathSnap API configuration via environment variables."""

from __future__ import annotations

import os
from pathlib import Path


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val is not None else default


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val is not None else default


class Settings:
    """Singleton-style settings loaded from environment."""

    app_name: str = "PathSnap API"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    data_dir: str = "Data"
    model_path: str = "outputs/models/random_forest.joblib"
    results_dir: str = "outputs/map_matching"
    frontier_search_radius_m: float = 50.0
    max_candidates: int = 5
    gps_gap_seconds: float = 30.0
    frontend_origin: str = "http://localhost:5173"
    secret_key: str = "dev-secret-key-change-in-production"

    @property
    def base_dir(self) -> Path:
        return Path(self.data_dir)

    @property
    def output_dir(self) -> Path:
        return Path(self.results_dir)


settings = Settings()