"""Unit tests for pimap_predict module.

Tests for predictor and feature extractor components.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pimap_predict.predictor import (
    MockPredictor,
    XGBoostPredictor,
    PredictionResult,
    get_predictor,
)
from pimap_predict.feature_extractor import FeatureExtractor, FeatureConfig


class TestMockPredictor(unittest.TestCase):
    """Tests for MockPredictor (Braden heuristic)."""

    def setUp(self):
        self.predictor = MockPredictor()

    def test_high_risk_prediction(self):
        """Test high risk prediction with low Braden scores."""
        features = {
            "braden_sensory_perception": 1,
            "braden_moisture": 1,
            "braden_activity": 1,
            "braden_mobility": 1,
            "braden_nutrition": 1,
            "braden_friction_shear": 1,
        }

        result = self.predictor.predict("patient-1", features)

        self.assertEqual(result.patient_id, "patient-1")
        self.assertEqual(result.risk_level, "High")
        self.assertGreaterEqual(result.risk_score, 0.7)
        self.assertEqual(result.model_version, "mock-braden-heuristic-v1")

    def test_moderate_risk_prediction(self):
        """Test moderate risk prediction."""
        features = {
            "braden_sensory_perception": 3,
            "braden_moisture": 3,
            "braden_activity": 2,
            "braden_mobility": 2,
            "braden_nutrition": 3,
            "braden_friction_shear": 2,
        }

        result = self.predictor.predict("patient-2", features)

        self.assertEqual(result.risk_level, "Moderate")
        self.assertGreaterEqual(result.risk_score, 0.4)
        self.assertLess(result.risk_score, 0.7)

    def test_low_risk_prediction(self):
        """Test low risk prediction with high Braden scores."""
        features = {
            "braden_sensory_perception": 4,
            "braden_moisture": 4,
            "braden_activity": 4,
            "braden_mobility": 4,
            "braden_nutrition": 4,
            "braden_friction_shear": 3,
        }

        result = self.predictor.predict("patient-3", features)

        self.assertEqual(result.risk_level, "Low")
        self.assertLess(result.risk_score, 0.4)

    def test_imputed_fields_tracked(self):
        """Test that imputed fields are tracked in result."""
        features = {
            "braden_sensory_perception": 3,
            "braden_moisture": 3,
            "braden_activity": 2,
            "_imputed_fields": ["albumin", "icu_stay_duration"],
        }

        result = self.predictor.predict("patient-4", features)
        self.assertIn("albumin", result.imputed_features)
        self.assertIn("icu_stay_duration", result.imputed_features)


class TestXGBoostPredictor(unittest.TestCase):
    """Tests for XGBoostPredictor."""

    def test_not_implemented_without_model(self):
        """Test that prediction raises error without model."""
        predictor = XGBoostPredictor()

        with self.assertRaises(NotImplementedError):
            predictor.predict("patient-1", {})

    def test_model_version_without_path(self):
        """Test model version string without model path."""
        predictor = XGBoostPredictor()
        self.assertEqual(predictor.get_model_version(), "xgboost-uninitialized")

    def test_model_version_with_path(self):
        """Test model version string with model path."""
        try:
            import xgboost

            predictor = XGBoostPredictor(model_path="/path/to/model.json")
            self.assertIn("/path/to/model.json", predictor.get_model_version())
        except ImportError:
            self.skipTest("xgboost not installed")


class TestGetPredictor(unittest.TestCase):
    """Tests for predictor factory function."""

    def test_returns_mock_without_path(self):
        """Test that get_predictor returns MockPredictor without model path."""
        predictor = get_predictor()
        self.assertIsInstance(predictor, MockPredictor)

    def test_returns_xgboost_with_path(self):
        """Test that get_predictor returns XGBoostPredictor with model path."""
        try:
            import xgboost

            predictor = get_predictor(model_path="/path/to/model.json")
            self.assertIsInstance(predictor, XGBoostPredictor)
        except ImportError:
            self.skipTest("xgboost not installed")


class TestFeatureExtractor(unittest.TestCase):
    """Tests for FeatureExtractor."""

    def setUp(self):
        self.extractor = FeatureExtractor()

    def test_extract_basic_features(self):
        """Test basic feature extraction."""
        vitals = {
            "patient_id": "patient-1",
            "timestamp": "2024-01-15T10:00:00Z",
            "icu_stay_duration": 7,
            "braden_sensory_perception": 3,
            "braden_moisture": 3,
            "braden_activity": 2,
            "braden_mobility": 2,
            "braden_nutrition": 3,
            "braden_friction_shear": 2,
            "arterial_o2_saturation": 95,
            "arterial_blood_pressure_systolic": 120,
            "glucose_whole_blood": 110,
            "albumin": 3.5,
            "total_bilirubin": 1.0,
            "total_protein": 6.0,
            "daily_weight": 70,
            "_imputed_fields": [],
            "_data_source": "epic_fhir",
        }

        features = self.extractor.extract(vitals)

        self.assertEqual(features["_patient_id"], "patient-1")
        self.assertEqual(features["_timestamp"], "2024-01-15T10:00:00Z")
        self.assertGreaterEqual(features["icu_stay_duration"], 0)
        self.assertLessEqual(features["icu_stay_duration"], 1)

    def test_extract_with_missing_values(self):
        """Test feature extraction with missing values (imputation)."""
        vitals = {
            "patient_id": "patient-2",
            "timestamp": "2024-01-15T10:00:00Z",
            "_imputed_fields": ["albumin", "icu_stay_duration"],
        }

        features = self.extractor.extract(vitals)

        self.assertIn("albumin", features["_imputed_fields"])
        self.assertIn("icu_stay_duration", features["_imputed_fields"])

    def test_to_vector(self):
        """Test converting features to vector."""
        features = {
            "icu_stay_duration": 0.23,
            "braden_friction_shear": 0.67,
            "braden_mobility": 0.5,
            "braden_moisture": 0.75,
            "braden_sensory_perception": 0.75,
            "braden_nutrition": 0.75,
            "braden_activity": 0.5,
            "arterial_o2_saturation": 0.95,
            "arterial_blood_pressure_systolic": 0.6,
            "glucose_whole_blood": 0.4,
            "albumin": 0.7,
            "total_bilirubin": 0.2,
            "total_protein": 0.6,
            "daily_weight": 0.35,
        }

        vector = self.extractor.to_vector(features)
        self.assertEqual(len(vector), 14)

    def test_custom_config(self):
        """Test feature extractor with custom configuration."""
        config = FeatureConfig(
            icu_duration_max_days=60.0,
            weight_max_kg=150.0,
        )
        extractor = FeatureExtractor(config)

        vitals = {"icu_stay_duration": 30}
        features = extractor.extract(vitals)

        self.assertEqual(features["icu_stay_duration"], 0.5)

    def test_normalize_string_value(self):
        """Test normalizing string values."""
        result = self.extractor._normalize("7 days", 30.0)
        self.assertEqual(result, 7.0 / 30.0)

    def test_normalize_invalid_value(self):
        """Test normalizing invalid values returns default."""
        result = self.extractor._normalize("invalid", 100.0)
        self.assertEqual(result, 0.5)


if __name__ == "__main__":
    unittest.main()
