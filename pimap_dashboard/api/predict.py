"""Lambda handler for pressure ulcer prediction: POST /patients/{id}/predict.

Thin adapter around pimap_dashboard.api.core.predict.
"""

import json
import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from .core import predict


def predict_pressure_ulcer(event, context):
    """Lambda handler: POST /patients/{patient_id}/predict
    (also routed at /api/patients/{patient_id}/predict).
    """
    status, body = predict(event["pathParameters"]["patient_id"])
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }