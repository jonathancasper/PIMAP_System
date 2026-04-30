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
    """Production predictor using XGBoost model.

    PLACEHOLDER: This class requires a trained XGBoost model file.
    Implementation will be completed when the model is ready.

    Expected model file format: XGBoost binary or JSON format
    Feature order must match FeatureExtractor output
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the XGBoost predictor.

        Args:
            model_path: Path to the trained XGBoost model file

        Raises:
            FileNotFoundError: If model_path is provided but file doesn't exist
            ImportError: If xgboost is not installed
        """
        self.model_path = model_path
        self._model = None
        self._feature_names = [
            "icu_stay_duration",
            "braden_friction_shear",
            "braden_mobility",
            "braden_moisture",
            "braden_sensory_perception",
            "braden_nutrition",
            "braden_activity",
            "arterial_o2_saturation",
            "arterial_blood_pressure_systolic",
            "glucose_whole_blood",
            "albumin",
            "total_bilirubin",
            "total_protein",
            "daily_weight",
        ]

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load the XGBoost model from file.

        PLACEHOLDER: Implement when model is ready.
        """
        try:
            import xgboost as xgb

            self._model = xgb.Booster()
            self._model.load_model(model_path)
        except ImportError:
            raise ImportError(
                "xgboost is required for XGBoostPredictor. "
                "Install with: pip install xgboost"
            )

    def predict(self, patient_id: str, features: Dict[str, Any]) -> PredictionResult:
        """Generate a risk prediction using the XGBoost model.

        PLACEHOLDER: Currently raises NotImplementedError.
        Will be implemented when the trained model is available.
        """
        if self._model is None:
            raise NotImplementedError(
                "XGBoostPredictor requires a trained model. "
                "Provide model_path during initialization, or use MockPredictor for testing."
            )

        feature_vector = self._prepare_features(features)

        import xgboost as xgb

        dmatrix = xgb.DMatrix(feature_vector.reshape(1, -1))
        risk_score = float(self._model.predict(dmatrix)[0])

        risk_level = self._score_to_level(risk_score)

        return PredictionResult(
            patient_id=patient_id,
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=self._calculate_confidence(risk_score),
            timestamp=datetime.now(),
            model_version=self.get_model_version(),
            features_used=features,
            imputed_features=features.get("_imputed_fields", []),
        )

    def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare feature vector in the order expected by the model."""
        return np.array(
            [
                features.get("icu_stay_duration", 3) / 30.0,
                features.get("braden_friction_shear", 2) / 3.0,
                features.get("braden_mobility", 2) / 4.0,
                features.get("braden_moisture", 3) / 4.0,
                features.get("braden_sensory_perception", 3) / 4.0,
                features.get("braden_nutrition", 3) / 4.0,
                features.get("braden_activity", 2) / 4.0,
                features.get("arterial_o2_saturation", 95) / 100.0,
                features.get("arterial_blood_pressure_systolic", 120) / 200.0,
                features.get("glucose_whole_blood", 110) / 300.0,
                features.get("albumin", 3.5) / 5.0,
                features.get("total_bilirubin", 1.0) / 5.0,
                features.get("total_protein", 6.0) / 10.0,
                features.get("daily_weight", 70) / 200.0,
            ]
        )

    def _score_to_level(self, score: float) -> str:
        """Convert numeric risk score to risk level category."""
        if score >= 0.7:
            return "High"
        elif score >= 0.4:
            return "Moderate"
        else:
            return "Low"

    def _calculate_confidence(self, score: float) -> float:
        """Calculate confidence based on distance from decision boundaries.

        Higher confidence when score is far from 0.4 and 0.7 thresholds.
        """
        distances = [abs(score - 0.4), abs(score - 0.7)]
        min_distance = min(distances)
        return min(1.0, min_distance * 2.5)

    def get_model_version(self) -> str:
        """Return the model version identifier."""
        if self.model_path:
            return f"xgboost-{self.model_path}"
        return "xgboost-uninitialized"


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
        """
        braden_total = sum(
            [
                features.get("braden_sensory_perception", 3),
                features.get("braden_moisture", 3),
                features.get("braden_activity", 2),
                features.get("braden_mobility", 2),
                features.get("braden_nutrition", 3),
                features.get("braden_friction_shear", 2),
            ]
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


def get_predictor(model_path: Optional[str] = None) -> Predictor:
    """Factory function to get the appropriate predictor.

    Args:
        model_path: Optional path to XGBoost model file.
                   If provided, returns XGBoostPredictor.
                   If None, returns MockPredictor.

    Returns:
        Predictor instance
    """
    if model_path:
        return XGBoostPredictor(model_path)
    return MockPredictor()
