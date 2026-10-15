"""Lambda handler for patient vitals endpoint."""

import json
import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from .data_source import get_data_source


def _parse_max_records(event: dict):
    """Parse the optional maxRecords query string parameter.

    Returns None (no limit; all records for the newest ICU stay) when
    the parameter is absent or invalid.
    """
    params = event.get("queryStringParameters") or {}
    raw = params.get("maxRecords")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def get_patient_vitals(event, context):
    """Lambda handler: GET /patients/{patient_id}/vitals

    Returns vitals history and stored predictions for a specific patient
    from the configured data source. By default all records from the
    newest ICU stay are returned; pass ?maxRecords=N to limit.
    """
    try:
        patient_id = event["pathParameters"]["patient_id"]
        data_source = get_data_source()
        vitals = data_source.get_vitals(
            patient_id, max_records=_parse_max_records(event)
        )
        predictions = data_source.get_predictions(patient_id)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({
                "vitals": vitals,
                "predictions": predictions,
            }),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)}),
        }
