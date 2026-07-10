"""Lambda handler for pressure ulcer prediction endpoint."""

import json
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from .data_source import get_data_source
from pimap_predict import FeatureExtractor, get_predictor, forward_fill_braden

PREDICTION_LOOKBACK_DAYS = 7

_predictor = None
_feature_extractor = None


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

        data_source = get_data_source()
        vitals = data_source.get_vitals(patient_id, max_records=500)

        if not vitals:
            return {
                "statusCode": 400,
                "headers": {"Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": "No vitals data found"}),
            }

        cutoff = _get_lookback_cutoff(vitals[0])
        recent_vitals = [
            v for v in vitals if v.get("timestamp", "") >= cutoff
        ]

        if not recent_vitals:
            recent_vitals = vitals[:1]

        enriched = forward_fill_braden(recent_vitals)

        if enriched is None:
            data_source_name = "mimic_iv" if os.environ.get("DATA_SOURCE") == "mimic" else "epic_fhir"
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                },
                "body": json.dumps(
                    {
                        "patient_id": patient_id,
                        "prediction_score": None,
                        "risk_level": "NoData",
                        "confidence": 0.0,
                        "timestamp": datetime.now().isoformat(),
                        "model_version": "mock-braden-heuristic-v1",
                        "data_source": data_source_name,
                        "imputed_fields": [],
                        "message": "No Braden assessment available within lookback window",
                    }
                ),
            }

        extractor = _get_feature_extractor()
        features = extractor.extract(enriched)

        predictor = _get_predictor()
        result = predictor.predict(patient_id, features)

        data_source_name = "mimic_iv" if os.environ.get("DATA_SOURCE") == "mimic" else "epic_fhir"

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
                    "data_source": data_source_name,
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


def _get_lookback_cutoff(most_recent_record: dict) -> str:
    """Get cutoff timestamp for lookback window.

    Returns ISO timestamp string for PREDICTION_LOOKBACK_DAYS before
    the most recent record's timestamp.
    """
    ts_str = most_recent_record.get("timestamp", "")
    try:
        most_recent = datetime.fromisoformat(ts_str)
        cutoff = most_recent - timedelta(days=PREDICTION_LOOKBACK_DAYS)
        return cutoff.isoformat()
    except (ValueError, TypeError):
        return ""
