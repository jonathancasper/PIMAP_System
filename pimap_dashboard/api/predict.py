"""Lambda handler for pressure ulcer prediction endpoint."""

import json
import sys
import os
from datetime import datetime

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from pimap_epic import EpicAuth, EpicFHIRClient
from pimap_predict import FeatureExtractor, get_predictor

_fhir_client = None
_predictor = None
_feature_extractor = None


def _get_client():
    global _fhir_client
    if _fhir_client is None:
        auth = EpicAuth()
        _fhir_client = EpicFHIRClient(auth)
    return _fhir_client


def _get_predictor():
    global _predictor
    if _predictor is None:
        model_path = os.environ.get("XGBOOST_MODEL_PATH")
        _predictor = get_predictor(model_path)
    return _predictor


def _get_feature_extractor():
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = FeatureExtractor()
    return _feature_extractor


def predict_pressure_ulcer(event, context):
    """Lambda handler: POST /patients/{patient_id}/predict

    Generates a pressure ulcer risk prediction for a patient.
    Uses XGBoost model if available, otherwise falls back to Braden heuristic.
    """
    try:
        patient_id = event["pathParameters"]["patient_id"]

        client = _get_client()
        vitals = client.get_patient_vitals(patient_id, max_records=1)

        if not vitals:
            return {
                "statusCode": 400,
                "headers": {"Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": "No vitals data found"}),
            }

        extractor = _get_feature_extractor()
        features = extractor.extract(vitals[0])

        predictor = _get_predictor()
        result = predictor.predict(patient_id, features)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(
                {
                    "patient_id": result.patient_id,
                    "prediction_score": result.risk_score,
                    "risk_level": result.risk_level,
                    "confidence": result.confidence,
                    "timestamp": result.timestamp.isoformat(),
                    "model_version": result.model_version,
                    "data_source": "epic_fhir",
                    "imputed_fields": result.imputed_features,
                }
            ),
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)}),
        }
