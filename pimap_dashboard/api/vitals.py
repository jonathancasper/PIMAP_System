"""Lambda handler for patient vitals endpoint."""

import json
import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from .data_source import get_data_source


def get_patient_vitals(event, context):
    """Lambda handler: GET /patients/{patient_id}/vitals

    Returns vitals history and stored predictions for a specific patient
    from the configured data source.
    """
    try:
        patient_id = event["pathParameters"]["patient_id"]
        data_source = get_data_source()
        vitals = data_source.get_vitals(patient_id, max_records=20)
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
