"""Unit tests for pimap_predict module.

Tests for predictor and feature extractor components.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pimap_predict.predictor import (
    MockPredictor,
    XGBoostPredictor,
    PredictionResult,
    get_predictor,
)
from pimap_predict.feature_extractor import FeatureExtractor, FeatureConfig
from pimap_predict.xgb_feature_extractor import XGBoostFeatureExtractor


class TestMockPredictor(unittest.TestCase):
    """Tests for MockPredictor (Braden heuristic)."""

    def setUp(self):
        self.predictor = MockPredictor()

    def test_high_risk_prediction(self):
        """Test high risk prediction with low Braden scores."""
        features = {
            "braden_sensory_perception": 0.25,
            "braden_moisture": 0.25,
            "braden_activity": 0.25,
            "braden_mobility": 0.25,
            "braden_nutrition": 0.25,
            "braden_friction_shear": 0.33,
        }

        result = self.predictor.predict("patient-1", features)

        self.assertEqual(result.patient_id, "patient-1")
        self.assertEqual(result.risk_level, "High")
        self.assertGreaterEqual(result.risk_score, 0.7)
        self.assertEqual(result.model_version, "mock-braden-heuristic-v1")

    def test_moderate_risk_prediction(self):
        """Test moderate risk prediction."""
        features = {
            "braden_sensory_perception": 0.75,
            "braden_moisture": 0.75,
            "braden_activity": 0.5,
            "braden_mobility": 0.5,
            "braden_nutrition": 0.75,
            "braden_friction_shear": 0.67,
        }

        result = self.predictor.predict("patient-2", features)

        self.assertEqual(result.risk_level, "Moderate")
        self.assertGreaterEqual(result.risk_score, 0.4)
        self.assertLess(result.risk_score, 0.7)

    def test_low_risk_prediction(self):
        """Test low risk prediction with high Braden scores."""
        features = {
            "braden_sensory_perception": 1.0,
            "braden_moisture": 1.0,
            "braden_activity": 1.0,
            "braden_mobility": 1.0,
            "braden_nutrition": 1.0,
            "braden_friction_shear": 1.0,
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
        self.assertEqual(predictor.get_model_version(), "hapi_xgb_v1")

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
        self.assertEqual(len(vector), 15)

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


class TestXGBoostFeatureExtractor(unittest.TestCase):
    """Tests for XGBoostFeatureExtractor temporal aggregation."""

    def setUp(self):
        self.feature_names = [
            "Height_MEAN_YearAgo",
            "age_years",
            "is_male",
            "eth_White",
            "eth_Black/African American",
            "eth_Other / Unknown",
            "smoker_any",
            "SBP_MIN_DayAgo",
            "SBP_MAX_DayAgo",
            "SBP_MEAN_DayAgo",
            "Creatinine_MIN_DayAgo",
            "Creatinine_MEAN_DayAgo",
            "Glucose_MAX_DayAgo",
            "FiO2_MIN_DayAgo",
            "FiO2_MAX_DayAgo",
            "FiO2_MEAN_DayAgo",
            "FiO2_TWA_DayAgo",
            "SpO2_TWA_DayAgo",
            "weight_twa_DayAgo",
            "neq_twa_DayAgo",
            "length_of_stay_to_incident",
            "fecal incontinence_BOOL_OR_DayAgo",
            "Paralysis_Dx_BOOL_OR_Ever",
            "has_SBP_DayAgo",
            "has_Creatinine_DayAgo",
            "Reposition_Count_SUM_DayAgo",
            "Fluid_In_SUM_HalfWeekAgo",
            "Ventilation_Status_MAX_DayAgo",
        ]
        self.now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    def _create_vitals_record(self, hours_ago, **kwargs):
        """Helper to create a vitals record at a specific time."""
        ts = self.now - timedelta(hours=hours_ago)
        record = {
            "timestamp": ts.isoformat(),
        }
        record.update(kwargs)
        return record

    def test_extract_demographics(self):
        """Test demographic feature extraction."""
        extractor = XGBoostFeatureExtractor(self.feature_names, self.now)
        
        demo = {
            "dob": "1970-01-15T00:00:00Z",
            "gender": "M",
            "ethnicity": "WHITE",
            "height_cm": 175.0,
        }
        
        features = extractor.extract_features(demo, [])
        
        self.assertAlmostEqual(features["age_years"], 54.0, places=1)
        self.assertEqual(features["is_male"], 1.0)
        self.assertEqual(features["eth_White"], 1.0)
        self.assertEqual(features["eth_Black/African American"], 0.0)
        self.assertEqual(features["Height_MEAN_YearAgo"], 175.0)

    def test_extract_vitals_aggregation(self):
        """Test vital signs temporal aggregation."""
        extractor = XGBoostFeatureExtractor(self.feature_names, self.now)
        
        demo = {"dob": "1970-01-15", "gender": "M", "ethnicity": "WHITE"}
        
        vitals = [
            self._create_vitals_record(1, arterial_blood_pressure_systolic=120),
            self._create_vitals_record(2, arterial_blood_pressure_systolic=130),
            self._create_vitals_record(3, arterial_blood_pressure_systolic=110),
            self._create_vitals_record(30, arterial_blood_pressure_systolic=140),
        ]
        
        features = extractor.extract_features(demo, vitals)
        
        self.assertEqual(features["SBP_MIN_DayAgo"], 110.0)
        self.assertEqual(features["SBP_MAX_DayAgo"], 130.0)
        self.assertAlmostEqual(features["SBP_MEAN_DayAgo"], 120.0, places=1)

    def test_extract_labs_aggregation(self):
        """Test lab value temporal aggregation."""
        extractor = XGBoostFeatureExtractor(self.feature_names, self.now)
        
        demo = {"dob": "1970-01-15", "gender": "M", "ethnicity": "WHITE"}
        
        vitals = [
            self._create_vitals_record(6, creatinine=1.0),
            self._create_vitals_record(12, creatinine=1.5),
            self._create_vitals_record(18, creatinine=1.3),
        ]
        
        features = extractor.extract_features(demo, vitals)
        
        self.assertEqual(features["Creatinine_MIN_DayAgo"], 1.0)
        self.assertAlmostEqual(features["Creatinine_MEAN_DayAgo"], 1.27, places=2)

    def test_extract_diagnosis_flags(self):
        """Test diagnosis BOOL_OR aggregation."""
        extractor = XGBoostFeatureExtractor(self.feature_names, self.now)
        
        demo = {"dob": "1970-01-15", "gender": "M", "ethnicity": "WHITE"}
        
        vitals = [
            self._create_vitals_record(6, stool_incontinent=True),
            self._create_vitals_record(12, stool_incontinent=False),
            self._create_vitals_record(48, paralysis_diagnosis=True),
        ]
        
        features = extractor.extract_features(demo, vitals)
        
        self.assertEqual(features["fecal incontinence_BOOL_OR_DayAgo"], 1.0)
        self.assertEqual(features["Paralysis_Dx_BOOL_OR_Ever"], 1.0)

    def test_extract_indicator_features(self):
        """Test has_* indicator features."""
        extractor = XGBoostFeatureExtractor(self.feature_names, self.now)
        
        demo = {"dob": "1970-01-15", "gender": "M", "ethnicity": "WHITE"}
        
        vitals = [
            self._create_vitals_record(6, arterial_blood_pressure_systolic=120),
            self._create_vitals_record(48, creatinine=1.2),
        ]
        
        features = extractor.extract_features(demo, vitals)
        
        self.assertEqual(features["has_SBP_DayAgo"], 1.0)
        self.assertEqual(features["has_Creatinine_DayAgo"], 0.0)

    def test_time_window_partitioning(self):
        """Test that records are correctly partitioned by time window."""
        extractor = XGBoostFeatureExtractor(self.feature_names, self.now)
        
        demo = {"dob": "1970-01-15", "gender": "M", "ethnicity": "WHITE"}
        
        vitals = [
            self._create_vitals_record(12, glucose=100),
            self._create_vitals_record(36, glucose=110),
            self._create_vitals_record(100, glucose=120),
            self._create_vitals_record(200, glucose=130),
        ]
        
        features = extractor.extract_features(demo, vitals)
        
        self.assertEqual(features["Glucose_MAX_DayAgo"], 100.0)

    def test_missing_features_are_nan(self):
        """Test that missing features return NaN."""
        extractor = XGBoostFeatureExtractor(self.feature_names, self.now)
        
        demo = {"dob": "1970-01-15", "gender": "M", "ethnicity": "WHITE"}
        
        features = extractor.extract_features(demo, [])
        
        import math
        self.assertTrue(math.isnan(features["SBP_MIN_DayAgo"]))
        self.assertTrue(math.isnan(features["Creatinine_MIN_DayAgo"]))

    def test_smoker_extraction(self):
        """Test smoker status extraction from demographics."""
        extractor = XGBoostFeatureExtractor(self.feature_names, self.now)
        
        demo = {"dob": "1970-01-15", "gender": "M", "ethnicity": "WHITE", "smoker_any": True}
        features = extractor.extract_features(demo, [])
        self.assertEqual(features["smoker_any"], 1.0)
        
        demo = {"dob": "1970-01-15", "gender": "M", "ethnicity": "WHITE", "smoker_any": False}
        features = extractor.extract_features(demo, [])
        self.assertEqual(features["smoker_any"], 0.0)
        
        demo = {"dob": "1970-01-15", "gender": "M", "ethnicity": "WHITE"}
        features = extractor.extract_features(demo, [])
        self.assertEqual(features["smoker_any"], 0.0)

    def test_interventions_aggregation(self):
        """Test nursing intervention aggregation."""
        extractor = XGBoostFeatureExtractor(self.feature_names, self.now)
        
        demo = {"dob": "1970-01-15", "gender": "M", "ethnicity": "WHITE"}
        
        vitals = [
            self._create_vitals_record(6, repositioned=True),
            self._create_vitals_record(12, repositioned=True),
            self._create_vitals_record(18, repositioned=False),
            self._create_vitals_record(24, intake_ml=1000),
            self._create_vitals_record(48, intake_ml=500),
        ]
        
        features = extractor.extract_features(demo, vitals)
        
        self.assertEqual(features["Reposition_Count_SUM_DayAgo"], 2.0)
        self.assertEqual(features["Fluid_In_SUM_HalfWeekAgo"], 1500.0)

    def test_ventilation_status(self):
        """Test ventilation status extraction."""
        extractor = XGBoostFeatureExtractor(self.feature_names, self.now)
        
        demo = {"dob": "1970-01-15", "gender": "M", "ethnicity": "WHITE"}
        
        vitals = [
            self._create_vitals_record(6, on_invasive_ventilation=True),
            self._create_vitals_record(12, on_non_invasive_ventilation=True),
        ]
        
        features = extractor.extract_features(demo, vitals)
        
        self.assertEqual(features["Ventilation_Status_MAX_DayAgo"], 2.0)

    def test_fio2_percent_normalization(self):
        """Test FiO2 percent-to-fraction normalization."""
        extractor = XGBoostFeatureExtractor(self.feature_names, self.now)
        
        demo = {"dob": "1970-01-15", "gender": "M", "ethnicity": "WHITE"}
        
        vitals = [
            self._create_vitals_record(6, fio2=40),
            self._create_vitals_record(12, fio2=60),
            self._create_vitals_record(18, fio2=0.5),
        ]
        
        features = extractor.extract_features(demo, vitals)
        
        self.assertAlmostEqual(features["FiO2_MIN_DayAgo"], 0.4, places=2)
        self.assertAlmostEqual(features["FiO2_MAX_DayAgo"], 0.6, places=2)
        self.assertAlmostEqual(features["FiO2_MEAN_DayAgo"], 0.5, places=2)

    def test_trapezoidal_twa(self):
        """Test trapezoidal time-weighted average."""
        extractor = XGBoostFeatureExtractor(self.feature_names, self.now)
        
        demo = {"dob": "1970-01-15", "gender": "M", "ethnicity": "WHITE"}
        
        vitals = [
            self._create_vitals_record(1, spo2=98),
            self._create_vitals_record(13, spo2=96),
            self._create_vitals_record(23, spo2=94),
        ]
        
        features = extractor.extract_features(demo, vitals)
        
        area1 = (98 + 96) / 2 * 12
        area2 = (96 + 94) / 2 * 10
        twa = (area1 + area2) / 24
        
        self.assertAlmostEqual(features["SpO2_TWA_DayAgo"], twa, places=1)

    def test_twa_single_point(self):
        """Test TWA with single measurement returns that value."""
        extractor = XGBoostFeatureExtractor(self.feature_names, self.now)
        
        demo = {"dob": "1970-01-15", "gender": "M", "ethnicity": "WHITE"}
        
        vitals = [
            self._create_vitals_record(6, spo2=97),
        ]
        
        features = extractor.extract_features(demo, vitals)
        
        self.assertEqual(features["SpO2_TWA_DayAgo"], 97.0)

    def test_twa_fio2_normalization(self):
        """Test TWA with FiO2 percent normalization."""
        extractor = XGBoostFeatureExtractor(self.feature_names, self.now)
        
        demo = {"dob": "1970-01-15", "gender": "M", "ethnicity": "WHITE"}
        
        vitals = [
            self._create_vitals_record(6, fio2=40),
            self._create_vitals_record(18, fio2=60),
        ]
        
        features = extractor.extract_features(demo, vitals)
        
        v1 = 0.4
        v2 = 0.6
        area = (v1 + v2) / 2 * 12
        twa = area / 24
        
        self.assertAlmostEqual(features["FiO2_TWA_DayAgo"], twa, places=2)

    def test_neq_twa_defaults_to_zero(self):
        """Test neq_twa defaults to 0."""
        extractor = XGBoostFeatureExtractor(self.feature_names, self.now)
        
        demo = {"dob": "1970-01-15", "gender": "M", "ethnicity": "WHITE"}
        
        features = extractor.extract_features(demo, [])
        
        self.assertEqual(features["neq_twa_DayAgo"], 0.0)

    def test_length_of_stay_hours(self):
        """Test length_of_stay_to_incident uses hours with log transform."""
        extractor = XGBoostFeatureExtractor(self.feature_names, self.now)

        import math

        demo = {
            "dob": "1970-01-15",
            "gender": "M",
            "ethnicity": "WHITE",
            "admission_time": (self.now - timedelta(hours=48)).isoformat(),
        }

        features = extractor.extract_features(demo, [])

        self.assertAlmostEqual(features["length_of_stay_to_incident"], math.log(48.0), places=3)


class TestForwardFillHistoricalStayBoundary(unittest.TestCase):
    """Regression tests: Braden forward-fill must never cross ICU-stay
    boundaries (per jc_rewrite_curation_plan.txt Part 1)."""

    @staticmethod
    def _record(ts, stay_id=None, **fields):
        rec = {"timestamp": ts, "stay_id": stay_id}
        rec.update(fields)
        return rec

    def _forward_fill(self):
        from scripts.generate_mimic_predictions import forward_fill_historical
        return forward_fill_historical

    def test_t1_fill_does_not_cross_stay_boundary(self):
        """T1: fields absent in the newest stay stay absent, even though an
        older stay has Braden data."""
        forward_fill_historical = self._forward_fill()

        records = [
            self._record("2024-01-01T10:00:00", stay_id=1,
                         braden_moisture=2, braden_activity=3),
            self._record("2024-01-05T09:00:00", stay_id=2),  # fresh stay
        ]
        target = datetime(2024, 1, 5, 12, 0)

        result = forward_fill_historical(
            records, target, target_stay_id=2)

        # Stay 2 has no Braden at/before target: must return None, never
        # borrow stay 1 values across the boundary.
        self.assertIsNone(result)

    def test_t2_fresh_admission_without_braden_returns_none(self):
        """T2: fresh admission (no in-stay Braden yet) -> None -> NoData."""
        forward_fill_historical = self._forward_fill()

        records = [
            self._record("2024-01-01T08:00:00", stay_id=1,
                         braden_moisture=2, braden_activity=3),
            self._record("2024-01-05T09:00:00", stay_id=2, heart_rate=88),
        ]
        target = datetime(2024, 1, 5, 10, 0)

        result = forward_fill_historical(
            records, target, target_stay_id=2)

        self.assertIsNone(result)

    def test_t3_in_stay_fill_works_and_tracks_imputed_fields(self):
        """T3: within one stay, older records supply missing subscores and
        _imputed_fields is populated."""
        forward_fill_historical = self._forward_fill()

        records = [
            self._record("2024-01-05T08:00:00", stay_id=7,
                         braden_moisture=2, braden_activity=3),
            self._record("2024-01-05T09:00:00", stay_id=7,
                         braden_activity=1),
        ]
        target = datetime(2024, 1, 5, 9, 0)

        result = forward_fill_historical(
            records, target, target_stay_id=7)

        self.assertIsNotNone(result)
        self.assertEqual(result["braden_moisture"], 2)  # filled in-stay
        self.assertEqual(result["braden_activity"], 1)  # present on newest
        self.assertIn("braden_moisture", result["_imputed_fields"])
        self.assertNotIn("braden_activity", result["_imputed_fields"])

    def test_no_stay_id_keeps_legacy_behavior(self):
        """Records without stay_id (e.g. Epic path) still forward-fill
        across the whole record list."""
        forward_fill_historical = self._forward_fill()

        records = [
            self._record("2024-01-01T10:00:00", braden_moisture=2),
            self._record("2024-01-05T09:00:00"),
        ]
        target = datetime(2024, 1, 5, 12, 0)

        result = forward_fill_historical(records, target)

        self.assertIsNotNone(result)
        self.assertEqual(result["braden_moisture"], 2)


class TestMimicDataSourceStayFiltering(unittest.TestCase):
    """Live-path guard: get_vitals must not mix ICU stays (plan Part 1)."""

    def test_get_vitals_restricts_to_newest_stay(self):
        import gzip
        import json
        import tempfile
        from pimap_dashboard.api.data_source import MimicDataSource

        with tempfile.TemporaryDirectory() as tmp:
            vitals = [
                {"patient_id": "p1", "stay_id": 1,
                 "timestamp": "2024-01-01T10:00:00", "heart_rate": 80},
                {"patient_id": "p1", "stay_id": 1,
                 "timestamp": "2024-01-01T12:00:00", "heart_rate": 82},
                {"patient_id": "p1", "stay_id": 2,
                 "timestamp": "2024-02-01T10:00:00", "heart_rate": 90},
                {"patient_id": "p1", "stay_id": 2,
                 "timestamp": "2024-02-01T12:00:00", "heart_rate": 92},
            ]
            with open(os.path.join(tmp, "patients_curated.json"), "w") as f:
                json.dump([], f)
            with gzip.open(os.path.join(tmp, "vitals_curated.json.gz"), "wt") as f:
                json.dump(vitals, f)

            source = MimicDataSource(tmp)
            result = source.get_vitals("p1", max_records=10)

            self.assertTrue(result)
            self.assertTrue(all(r["stay_id"] == 2 for r in result))
            self.assertEqual(len(result), 2)  # newest stay's records only

    def test_get_vitals_without_stay_id_unaffected(self):
        """Records lacking stay_id (Epic-style) are returned unfiltered."""
        import gzip
        import json
        import tempfile
        from pimap_dashboard.api.data_source import MimicDataSource

        with tempfile.TemporaryDirectory() as tmp:
            vitals = [
                {"patient_id": "p1", "timestamp": "2024-01-01T10:00:00"},
                {"patient_id": "p1", "timestamp": "2024-01-05T10:00:00"},
            ]
            with open(os.path.join(tmp, "patients_curated.json"), "w") as f:
                json.dump([], f)
            with gzip.open(os.path.join(tmp, "vitals_curated.json.gz"), "wt") as f:
                json.dump(vitals, f)

            source = MimicDataSource(tmp)
            result = source.get_vitals("p1", max_records=10)

            self.assertEqual(len(result), 2)


class TestVitalsHistoryFullStay(unittest.TestCase):
    """get_vitals with max_records=None returns the full newest stay."""

    def _make_source(self, tmp):
        import gzip
        import json
        from pimap_dashboard.api.data_source import MimicDataSource

        vitals = []
        for i in range(25):
            vitals.append({
                "patient_id": "p1", "stay_id": 7,
                "timestamp": f"2024-01-01T{10 + i:02d}:00:00",
            })
        with open(os.path.join(tmp, "patients_curated.json"), "w") as f:
            json.dump([{"patient_id": "p1", "gender": "F"}], f)
        with gzip.open(os.path.join(tmp, "vitals_curated.json.gz"), "wt") as f:
            json.dump(vitals, f)
        return MimicDataSource(tmp)

    def test_full_stay_returned_when_unlimited(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            source = self._make_source(tmp)
            result = source.get_vitals("p1", max_records=None)
            self.assertEqual(len(result), 25)

    def test_explicit_limit_still_applies(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            source = self._make_source(tmp)
            result = source.get_vitals("p1", max_records=5)
            self.assertEqual(len(result), 5)

    def test_get_patient_returns_record(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            source = self._make_source(tmp)
            patient = source.get_patient("p1")
            self.assertIsNotNone(patient)
            self.assertEqual(patient["patient_id"], "p1")
            self.assertIsNone(source.get_patient("missing"))


class TestEpicDataSourceGetPatient(unittest.TestCase):
    """EpicDataSource.get_patient scopes the FHIR query to one patient."""

    def test_get_patient_uses_scoped_patient_ids(self):
        from pimap_dashboard.api.data_source import EpicDataSource

        source = EpicDataSource.__new__(EpicDataSource)
        source._client = Mock()
        source._client.get_all_dashboard_patients.return_value = [
            {"patient_id": "p9", "gender": "M"}
        ]

        patient = source.get_patient("p9")

        source._client.get_all_dashboard_patients.assert_called_once_with(
            patient_ids=["p9"]
        )
        self.assertEqual(patient["patient_id"], "p9")
        self.assertIsNone(source.get_patient("p404"))


class TestVitalsHandlerMaxRecords(unittest.TestCase):
    """Query parameter parsing for the vitals Lambda handler."""

    def test_absent_param_means_full_stay(self):
        from pimap_dashboard.api.vitals import _parse_max_records

        self.assertIsNone(_parse_max_records({}))
        self.assertIsNone(_parse_max_records({"queryStringParameters": None}))

    def test_valid_param(self):
        from pimap_dashboard.api.vitals import _parse_max_records

        self.assertEqual(
            _parse_max_records({"queryStringParameters": {"maxRecords": "50"}}), 50
        )

    def test_invalid_param_means_full_stay(self):
        from pimap_dashboard.api.vitals import _parse_max_records

        self.assertIsNone(
            _parse_max_records({"queryStringParameters": {"maxRecords": "abc"}})
        )
        self.assertIsNone(
            _parse_max_records({"queryStringParameters": {"maxRecords": "-3"}})
        )


class TestXGBoostPredictionTime(unittest.TestCase):
    """XGBoostPredictor accepts a prediction_time for historical data."""

    MODEL_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_input"
    )

    def _skip_if_model_missing(self):
        if not (
            os.path.exists(os.path.join(self.MODEL_DIR, "hapi_xgb_v1.json"))
            and os.path.exists(os.path.join(self.MODEL_DIR, "hapi_xgb_schema_v1.json"))
        ):
            self.skipTest("hapi_xgb_v1 model files not present")

    def test_predict_with_demographics_bundle(self):
        self._skip_if_model_missing()
        try:
            import xgboost  # noqa: F401
        except ImportError:
            self.skipTest("xgboost not installed")

        predictor = XGBoostPredictor(model_dir=self.MODEL_DIR)
        demo = {"dob": "2110-03-01", "gender": "M", "height_cm": 180.0}
        vitals = [
            {
                "timestamp": "2169-01-15T04:04:00",
                "heart_rate": 88,
                "spo2": 95,
                "daily_weight": 97.4,
                "creatinine": 1.1,
            },
        ]
        prediction_time = datetime(2169, 1, 15, 5, 0, 0, tzinfo=timezone.utc)

        result = predictor.predict(
            "p1",
            {"demographics": demo, "vitals": vitals},
            prediction_time=prediction_time,
        )

        self.assertIn(
            result.risk_level,
            ["No HAPI", "Low-severity HAPI", "High-severity HAPI"],
        )
        self.assertEqual(result.model_version, "hapi_xgb_v1")
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.risk_score, float)

    def test_age_computed_from_naive_dob(self):
        """Naive (date-only) dob must not break against aware prediction time."""
        from pimap_predict.xgb_feature_extractor import XGBoostFeatureExtractor

        extractor = XGBoostFeatureExtractor(
            ["age_years"], prediction_time=datetime(2169, 1, 15, tzinfo=timezone.utc)
        )
        features = extractor.extract_features({"dob": "2106-01-01"}, [])
        self.assertAlmostEqual(features["age_years"], 63.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
