"""ML predictor module for pressure ulcer risk assessment.

Provides a pluggable predictor interface that can be swapped between:
- XGBoostPredictor: Production predictor using trained XGBoost model
- MockPredictor: Placeholder for testing/demo without a trained model
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class PredictionResult:
    """Container for prediction output."""

    patient_id: str
    risk_score: float
    risk_level: str
    confidence: float
    timestamp: datetime
    model_version: str
    features_used: Dict[str, Any]
    imputed_features: list


class Predictor(ABC):
    """Abstract base class for pressure ulcer risk predictors."""

    @abstractmethod
    def predict(self, patient_id: str, features: Dict[str, Any]) -> PredictionResult:
        """Generate a risk prediction for a patient.

        Args:
            patient_id: Patient identifier
            features: Dictionary of feature values

        Returns:
            PredictionResult with risk score and metadata
        """
        pass

    @abstractmethod
    def get_model_version(self) -> str:
        """Return the model version identifier."""
        pass


class XGBoostPredictor(Predictor):
    """Production predictor using XGBoost model for HAPI prediction.
    
    Uses a trained XGBoost model to predict pressure ulcer risk across
    three categories: No HAPI, Low-severity HAPI, and High-severity HAPI.
    
    The model expects 415 features computed via XGBoostFeatureExtractor.
    """
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        model_name: str = "hapi_xgb",
        version: str = "1",
        model_path: Optional[str] = None,
        schema_path: Optional[str] = None,
    ):
        """Initialize the XGBoost predictor.
        
        Args:
            model_dir: Directory containing model files
            model_name: Base name of model (default: hapi_xgb)
            version: Model version number (default: 1)
            model_path: Direct path to model file (alternative to model_dir)
            schema_path: Direct path to schema file (alternative to model_dir)
        
        Raises:
            ModelLoadError: If model file not found or load fails
            SchemaLoadError: If schema validation fails
        """
        self.model_dir = model_dir
        self.model_name = model_name
        self.version = version
        self._model = None
        self._schema = None
        self._feature_names = []
        self._target_labels = {}
        self._decision_rule = None
        
        if model_dir:
            self._load_from_dir(model_dir, model_name, version)
        elif model_path and schema_path:
            self._load_from_paths(model_path, schema_path)
    
    def _load_from_dir(self, model_dir: str, model_name: str, version: str):
        """Load model and schema from directory using naming convention."""
        from .models import load_xgboost_model, load_model_schema
        from pathlib import Path
        
        self._model, self._schema = load_xgboost_model(model_dir, model_name, version)
        self._setup_from_schema()
    
    def _load_from_paths(self, model_path: str, schema_path: str):
        """Load model and schema from explicit file paths."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "xgboost is required for XGBoostPredictor. "
                "Install with: pip install xgboost"
            )
        
        from .models import load_model_schema
        from pathlib import Path
        
        self._model = xgb.XGBClassifier()
        self._model.load_model(model_path)
        self._schema = load_model_schema(Path(schema_path))
        self._setup_from_schema()
    
    def _setup_from_schema(self):
        """Setup internal state from loaded schema."""
        self._feature_names = self._schema['feature_names']
        self._target_labels = self._schema.get('target', {})
        self._decision_rule = self._schema.get('decision_rule')
    
    def predict(
        self,
        patient_id: str,
        features: Dict[str, Any],
        prediction_time: Optional[datetime] = None,
    ) -> PredictionResult:
        """Generate a risk prediction using the XGBoost model.

        Args:
            patient_id: Patient identifier
            features: Dictionary containing:
                - 'demographics': Patient demographics dict
                - 'vitals': List of vitals/lab records
                Or pre-computed feature dict from XGBoostFeatureExtractor
            prediction_time: Optional reference time for temporal feature
                windows. Defaults to now. Must be timezone-aware if given.
                Use the most recent vitals timestamp when the data is
                historical or date-shifted.

        Returns:
            PredictionResult with risk_level and all class probabilities
        """
        if self._model is None:
            raise NotImplementedError(
                "XGBoostPredictor requires a trained model. "
                "Provide model_dir during initialization, or use MockPredictor for testing."
            )

        feature_dict = self._prepare_features(features, prediction_time)
        
        import pandas as pd
        df = pd.DataFrame([feature_dict], columns=self._feature_names)
        
        proba = self._model.predict_proba(df)[0]
        
        p0, p1, p2 = proba[0], proba[1], proba[2]
        
        risk_level = self._apply_decision_rule(p0, p1, p2)
        
        risk_score = max(p0, p1, p2)
        
        return PredictionResult(
            patient_id=patient_id,
            risk_score=float(risk_score),
            risk_level=risk_level,
            confidence=self._calculate_confidence(proba),
            timestamp=datetime.now(),
            model_version=self.get_model_version(),
            features_used={
                **features,
                'probabilities': {
                    'no_hapi': float(p0),
                    'low_severity_hapi': float(p1),
                    'advanced_severity_hapi': float(p2),
                },
            },
            imputed_features=features.get("_imputed_fields", []),
        )
    
    def _prepare_features(
        self,
        features: Dict[str, Any],
        prediction_time: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Prepare features for model input.

        If features already contains all feature_names keys, use directly.
        Otherwise, extract using XGBoostFeatureExtractor.
        """
        if all(k in features for k in self._feature_names):
            return {k: features.get(k, float('nan')) for k in self._feature_names}

        demographics = features.get('demographics', {})
        vitals = features.get('vitals', [])

        if not demographics and 'patient_id' in features:
            demographics = features
            vitals = features.get('vitals_records', [])

        from .xgb_feature_extractor import XGBoostFeatureExtractor

        extractor = XGBoostFeatureExtractor(self._feature_names, prediction_time)
        return extractor.extract_features(demographics, vitals)
    
    def _apply_decision_rule(self, p0: float, p1: float, p2: float) -> str:
        """Apply the two-stage decision rule from schema.
        
        Decision rule:
        1. p_any = p1 + p2
        2. If p_any < t_pi, predict "No HAPI"
        3. Otherwise, compute p_adv_given_pi = p2 / (p1 + p2)
        4. If p_adv_given_pi >= t_adv, predict "High-severity HAPI"
        5. Otherwise predict "Low-severity HAPI"
        """
        if self._decision_rule:
            t_pi = self._decision_rule.get('t_pi', 0.20)
            t_adv = self._decision_rule.get('t_adv', 0.05)
        else:
            t_pi = 0.20
            t_adv = 0.05
        
        p_any = p1 + p2
        
        if p_any < t_pi:
            return "No HAPI"
        
        p_adv_given_pi = p2 / p_any if p_any > 0 else 0
        
        if p_adv_given_pi >= t_adv:
            return "High-severity HAPI"
        else:
            return "Low-severity HAPI"
    
    def _calculate_confidence(self, proba: np.ndarray) -> float:
        """Calculate confidence based on entropy of probability distribution.
        
        Lower entropy (more certain distribution) = higher confidence.
        """
        proba = proba[proba > 0]
        if len(proba) == 0:
            return 0.5
        
        import math
        entropy = -sum(p * math.log(p) for p in proba)
        max_entropy = math.log(3)
        
        confidence = 1.0 - (entropy / max_entropy)
        return float(max(0.0, min(1.0, confidence)))
    
    def get_model_version(self) -> str:
        """Return the model version identifier."""
        return f"{self.model_name}_v{self.version}"


class MockPredictor(Predictor):
    """Mock predictor for testing and demo purposes.

    Uses Braden scale heuristic as a fallback when no trained model is available.
    """

    def __init__(self):
        """Initialize the mock predictor."""
        self._model_version = "mock-braden-heuristic-v1"

    def predict(self, patient_id: str, features: Dict[str, Any]) -> PredictionResult:
        """Generate a risk prediction using Braden scale heuristic.

        The Braden Scale for Pressure Ulcer Risk Assessment uses six subscales:
        - Sensory perception (1-4)
        - Moisture (1-4)
        - Activity (1-4)
        - Mobility (1-4)
        - Nutrition (1-4)
        - Friction and shear (1-3)

        Total score range: 6-23
        Risk categories:
        - High risk: ≤ 12
        - Moderate risk: 13-16
        - Low risk: > 16

        NOTE: This method expects NORMALIZED features from FeatureExtractor.
        The Braden subscales are normalized to [0,1] range by FeatureExtractor:
        - friction_shear divided by 3.0
        - all other subscales divided by 4.0
        
        We denormalize by multiplying back before summing to get raw Braden scores.
        """
        braden_total = (
            round(features.get("braden_sensory_perception", 0.75) * 4) +
            round(features.get("braden_moisture", 0.75) * 4) +
            round(features.get("braden_activity", 0.5) * 4) +
            round(features.get("braden_mobility", 0.5) * 4) +
            round(features.get("braden_nutrition", 0.75) * 4) +
            round(features.get("braden_friction_shear", 0.67) * 3)
        )

        if braden_total <= 12:
            risk_score = 0.9
            risk_level = "High"
        elif braden_total <= 16:
            risk_score = 0.6
            risk_level = "Moderate"
        else:
            risk_score = 0.2
            risk_level = "Low"

        confidence = 0.5

        return PredictionResult(
            patient_id=patient_id,
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=confidence,
            timestamp=datetime.now(),
            model_version=self._model_version,
            features_used=features,
            imputed_features=features.get("_imputed_fields", []),
        )

    def get_model_version(self) -> str:
        """Return the model version identifier."""
        return self._model_version


def get_predictor(
    model_dir: Optional[str] = None,
    model_name: str = "hapi_xgb",
    version: str = "1",
    model_path: Optional[str] = None,
    schema_path: Optional[str] = None,
) -> Predictor:
    """Factory function to get the appropriate predictor.

    Args:
        model_dir: Directory containing model files (preferred method)
        model_name: Base name of model (default: hapi_xgb)
        version: Model version number (default: 1)
        model_path: Direct path to model file (alternative)
        schema_path: Direct path to schema file (required if model_path given)

    Returns:
        XGBoostPredictor if model files found, else MockPredictor
    """
    if model_dir or (model_path and schema_path):
        return XGBoostPredictor(
            model_dir=model_dir,
            model_name=model_name,
            version=version,
            model_path=model_path,
            schema_path=schema_path,
        )
    return MockPredictor()
