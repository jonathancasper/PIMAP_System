"""Unit tests for pimap_epic module.

Tests for FHIR client and authentication components.
Note: Most tests require Epic sandbox access or mocking.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pimap_epic.fhir_client import (
    EpicFHIRClient,
    FHIRRequestError,
    _parse_fhir_datetime,
)
from pimap_epic.auth import EpicAuthError


class TestFHIRClient(unittest.TestCase):
    """Tests for EpicFHIRClient."""

    def test_parse_fhir_datetime(self):
        """Test FHIR datetime parsing."""
        dt = _parse_fhir_datetime("2024-01-15T10:30:00Z")
        self.assertEqual(dt.year, 2024)
        self.assertEqual(dt.month, 1)
        self.assertEqual(dt.day, 15)

        dt_with_tz = _parse_fhir_datetime("2024-01-15T10:30:00+00:00")
        self.assertEqual(dt_with_tz.year, 2024)

    def test_build_value_map(self):
        """Test building value map from observations."""
        mock_auth = Mock()
        mock_auth.get_access_token.return_value = "test-token"
        client = EpicFHIRClient(mock_auth)

        entries = [
            {
                "resource": {
                    "effectiveDateTime": "2024-01-15T10:00:00Z",
                    "valueQuantity": {"value": 120},
                }
            },
            {
                "resource": {
                    "effectiveDateTime": "2024-01-15T11:00:00Z",
                    "valueQuantity": {"value": 125},
                }
            },
        ]

        value_map = client._build_value_map(entries)
        self.assertEqual(len(value_map), 2)
        self.assertEqual(value_map["2024-01-15T10:00:00Z"], 120)

    def test_find_nearest(self):
        """Test finding nearest value in map."""
        mock_auth = Mock()
        mock_auth.get_access_token.return_value = "test-token"
        client = EpicFHIRClient(mock_auth)

        value_map = {
            "2024-01-15T10:00:00Z": 120,
            "2024-01-15T12:00:00Z": 125,
            "2024-01-15T14:00:00Z": 130,
        }

        nearest = client._find_nearest(value_map, "2024-01-15T11:30:00Z")
        self.assertEqual(nearest, 125)

        nearest = client._find_nearest(value_map, "2024-01-15T09:00:00Z")
        self.assertEqual(nearest, 120)

    def test_find_nearest_empty_map(self):
        """Test finding nearest with empty map."""
        mock_auth = Mock()
        mock_auth.get_access_token.return_value = "test-token"
        client = EpicFHIRClient(mock_auth)

        nearest = client._find_nearest({}, "2024-01-15T11:30:00Z")
        self.assertIsNone(nearest)

    def test_transform_patient(self):
        """Test patient transformation."""
        mock_auth = Mock()
        mock_auth.get_access_token.return_value = "test-token"
        client = EpicFHIRClient(mock_auth)

        fhir_patient = {
            "id": "patient-123",
            "name": [{"given": ["John"], "family": "Doe"}],
            "birthDate": "1980-05-15",
            "gender": "male",
            "extension": [],
        }

        result = client._transform_patient(fhir_patient)
        self.assertEqual(result["patient_id"], "patient-123")
        self.assertEqual(result["first_name"], "John")
        self.assertEqual(result["last_name"], "Doe")
        self.assertEqual(result["gender"], "M")
        self.assertEqual(result["dob"], "1980-05-15")


class TestFHIRWriter(unittest.TestCase):
    """Tests for FHIR write-back functionality."""

    def test_build_observation_resource(self):
        """Test building FHIR Observation resource."""
        from pimap_epic.fhir_writer import EpicFHIRWriter
        from datetime import datetime, timezone

        mock_client = Mock()
        writer = EpicFHIRWriter(mock_client)

        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        obs = writer._build_observation_resource(
            patient_id="patient-123",
            risk_score=0.75,
            risk_level="High",
            model_version="xgboost-v1",
            timestamp=ts,
        )

        self.assertEqual(obs["resourceType"], "Observation")
        self.assertEqual(obs["status"], "final")
        self.assertEqual(obs["subject"]["reference"], "Patient/patient-123")
        self.assertEqual(obs["valueQuantity"]["value"], 0.75)

    def test_build_document_reference_resource(self):
        """Test building FHIR DocumentReference resource."""
        from pimap_epic.fhir_writer import EpicFHIRWriter
        from datetime import datetime, timezone

        mock_client = Mock()
        writer = EpicFHIRWriter(mock_client)

        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        doc = writer._build_document_reference_resource(
            patient_id="patient-123",
            note_text="High risk for pressure ulcer",
            note_type="risk_assessment",
            timestamp=ts,
        )

        self.assertEqual(doc["resourceType"], "DocumentReference")
        self.assertEqual(doc["status"], "current")
        self.assertEqual(doc["subject"]["reference"], "Patient/patient-123")

    def test_write_methods_raise_error(self):
        """Test that write methods raise NotImplementedError."""
        from pimap_epic.fhir_writer import EpicFHIRWriter, FHIRWriteError

        mock_client = Mock()
        writer = EpicFHIRWriter(mock_client)

        with self.assertRaises(FHIRWriteError):
            writer.write_risk_observation("patient-123", 0.75, "High", "v1")

        with self.assertRaises(FHIRWriteError):
            writer.write_clinical_note("patient-123", "Test note")


if __name__ == "__main__":
    unittest.main()
