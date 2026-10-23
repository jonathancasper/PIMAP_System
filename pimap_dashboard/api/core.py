"""Shared PIMAP dashboard API cores.

Plain functions returning (status_code, body_dict) for each dashboard
endpoint. Two adapters share this module so behavior can never drift
between deployments:

- AWS Lambda handlers (pimap_dashboard/api/*.py): parse the API Gateway
  event, call the core, serialize the body.
- Vercel serverless functions (api/**.py): parse the request path/query,
  call the core, serialize the body.

The data source is controlled by the DATA_SOURCE environment variable
('mimic' by default - the MIMIC-IV demo; set 'epic' for the Epic FHIR
integration).
"""

import json
import os
from datetime import datetime, timedelta, timezone

from .data_source import get_data_source


def _get_lookback_days():
    try:
        return max(1, int(os.environ.get("PREDICTION_LOOKBACK_DAYS", "7")))
    except (TypeError, ValueError):
        return 7


def parse_max_records(raw):
    """Parse an optional maxRecords query-string value.

    Returns None (no limit; all records for the newest ICU stay) when
    the value is absent or invalid.
    """
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


# --- patients -----------------------------------------------------------------
def list_patients():
    """GET /api/patients - list all patients."""
    try:
        data_source = get_data_source()
        patients = data_source.get_patients()
        return 200, patients
    except Exception as e:
        return 500, {"error": str(e)}


# --- vitals -------------------------------------------------------------------
def get_vitals(patient_id, max_records=None):
    """GET /api/patients/{id}/vitals - vitals history + stored predictions."""
    try:
        data_source = get_data_source()
        vitals = data_source.get_vitals(patient_id, max_records=max_records)
        predictions = data_source.get_predictions(patient_id)
        return 200, {"vitals": vitals, "predictions": predictions}
    except Exception as e:
        return 500, {"error": str(e)}


# --- actions ------------------------------------------------------------------
def record_action(patient_id, body, put_action=None):
    """POST /api/patients/{id}/actions - record a staff action.

    `put_action(patient_id, timestamp, action_taken, staff_id)` stores
    the action; adapters inject a DynamoDB-backed implementation on AWS
    and an in-memory implementation elsewhere. Returns the body dict for
    a successful recording (callers serialize).
    """
    try:
        action_taken = (body or {}).get("action_taken")
        staff_id = (body or {}).get("staff_id")

        if not action_taken:
            return 400, {"error": "action_taken is required"}

        timestamp = datetime.now().isoformat() + "Z"

        if put_action is not None:
            put_action(patient_id, timestamp, action_taken, staff_id or "")

        return 200, {
            "message": "Action recorded successfully",
            "patient_id": patient_id,
            "action_taken": action_taken,
        }
    except Exception as e:
        return 500, {"error": str(e)}


# --- predict ------------------------------------------------------------------
_predictor = None
_feature_extractor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        from pimap_predict import get_predictor as _factory

        model_path = os.environ.get("XGBOOST_MODEL_PATH")
        _predictor = _factory(model_path)
    return _predictor


def _get_feature_extractor():
    global _feature_extractor
    if _feature_extractor is None:
        from pimap_predict import FeatureExtractor

        _feature_extractor = FeatureExtractor()
    return _feature_extractor


def _parse_prediction_time(timestamp):
    """Parse a vitals timestamp into a timezone-aware datetime.

    The XGBoost feature extractor computes temporal windows relative
    to this time. Demo data (e.g. MIMIC) is date-shifted, so "now" is
    not a valid reference; the newest vitals record is.
    """
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _get_lookback_cutoff(most_recent_record):
    """ISO timestamp PREDICTION_LOOKBACK_DAYS before the newest record."""
    ts_str = most_recent_record.get("timestamp", "")
    try:
        most_recent = datetime.fromisoformat(ts_str)
        cutoff = most_recent - timedelta(days=_get_lookback_days())
        return cutoff.isoformat()
    except (ValueError, TypeError):
        return ""


def predict(patient_id):
    """POST /api/patients/{id}/predict - generate a risk prediction.

    Uses the XGBoost model if available, otherwise the Braden-based
    mock predictor.
    """
    from pimap_predict import XGBoostPredictor, forward_fill_braden

    try:
        data_source = get_data_source()
        vitals = data_source.get_vitals(patient_id, max_records=500)

        if not vitals:
            return 400, {"error": "No vitals data found"}

        predictor = _get_predictor()
        data_source_name = (
            "mimic_iv" if os.environ.get("DATA_SOURCE", "mimic").lower() == "mimic"
            else "epic_fhir"
        )

        if isinstance(predictor, XGBoostPredictor):
            # The XGBoost model derives its own 415 temporal features from
            # demographics plus the vitals history, so pass the records
            # directly (no Braden forward-fill or lookback filtering).
            patient_info = data_source.get_patient(patient_id) or {}
            prediction_time = _parse_prediction_time(
                vitals[0].get("timestamp", "")
            )
            features = {"demographics": patient_info, "vitals": vitals}
            result = predictor.predict(
                patient_id, features, prediction_time=prediction_time
            )
        else:
            cutoff = _get_lookback_cutoff(vitals[0])
            recent_vitals = [
                v for v in vitals if v.get("timestamp", "") >= cutoff
            ]

            if not recent_vitals:
                recent_vitals = vitals[:1]

            enriched = forward_fill_braden(recent_vitals)

            if enriched is None:
                return 200, {
                    "patient_id": patient_id,
                    "prediction_score": None,
                    "risk_level": "NoData",
                    "confidence": 0.0,
                    "timestamp": datetime.now().isoformat(),
                    "model_version": "mock-braden-heuristic-v1",
                    "data_source": data_source_name,
                    "imputed_fields": [],
                    "message": (
                        "No Braden assessment available within lookback window"
                    ),
                }

            extractor = _get_feature_extractor()
            features = extractor.extract(enriched)
            result = predictor.predict(patient_id, features)

        probabilities = None
        if isinstance(getattr(result, "features_used", None), dict):
            probabilities = result.features_used.get("probabilities")

        return 200, {
            "patient_id": result.patient_id,
            "prediction_score": result.risk_score,
            "risk_level": result.risk_level,
            "probabilities": probabilities,
            "confidence": result.confidence,
            "timestamp": result.timestamp.isoformat(),
            "model_version": result.model_version,
            "data_source": data_source_name,
            "imputed_fields": result.imputed_features,
        }

    except Exception as e:
        return 500, {"error": str(e)}