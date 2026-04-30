"""PIMAP prediction module.

Provides ML-based pressure ulcer risk prediction using:
- XGBoost model (primary, externally trained)
- Feature extraction from FHIR patient data
- Pluggable predictor interface for future model updates
"""

from .predictor import Predictor, XGBoostPredictor, MockPredictor, get_predictor
from .feature_extractor import FeatureExtractor

__all__ = [
    "Predictor",
    "XGBoostPredictor",
    "MockPredictor",
    "FeatureExtractor",
    "get_predictor"
]
