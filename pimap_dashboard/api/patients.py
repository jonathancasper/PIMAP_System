"""Lambda handler for patient list endpoint: GET /patients.

Thin adapter around pimap_dashboard.api.core.list_patients.
"""

import json
import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from .core import list_patients


CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Allow-Methods": "*",
}


def get_patients(event, context):
    """Lambda handler: GET /patients (also routed at /api/patients)."""
    status, body = list_patients()
    return {
        "statusCode": status,
        "headers": CORS_HEADERS if status == 200 else {"Access-Control-Allow-Origin": "*"},
        "body": json.dumps(body),
    }