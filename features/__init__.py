"""PathSnap feature engineering package.

Transforms GPS observations and generated candidates into ML-ready features
(one row = one GPS observation x one candidate arc). See ``features.schema``.
"""

from .spatial_features import compute_spatial_features, get_spatial_feature_names
from .temporal_features import compute_temporal_features, get_temporal_feature_names
from .network_features import compute_network_features, get_network_feature_names
from .transition_features import compute_transition_features, get_transition_feature_names

__all__ = [
    "compute_spatial_features", "compute_temporal_features",
    "compute_network_features", "compute_transition_features",
    "get_spatial_feature_names", "get_temporal_feature_names",
    "get_network_feature_names", "get_transition_feature_names",
]

__version__ = "0.2.0"
