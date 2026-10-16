"""PIMAP prediction module.

Provides ML-based pressure ulcer risk prediction using:
- XGBoost model (primary, externally trained)
- Feature extraction from FHIR patient data
- Pluggable predictor interface for future model updates
"""

from .predictor import (
    Predictor,
    XGBoostPredictor,
    MockPredictor,
    get_predictor,
    PredictionResult,
)
from .feature_extractor import FeatureExtractor, forward_fill_braden, BRADEN_FIELDS
from .xgb_feature_extractor import XGBoostFeatureExtractor
from .models import (
    load_xgboost_model,
    load_model_schema,
    get_available_models,
    ModelLoadError,
    SchemaLoadError,
)

__all__ = [
    "Predictor",
    "XGBoostPredictor",
    "MockPredictor",
    "PredictionResult",
    "get_predictor",
    "FeatureExtractor",
    "XGBoostFeatureExtractor",
    "forward_fill_braden",
    "BRADEN_FIELDS",
    "load_xgboost_model",
    "load_model_schema",
    "get_available_models",
    "ModelLoadError",
    "SchemaLoadError",
]
