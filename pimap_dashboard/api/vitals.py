"""Lambda handler for patient vitals endpoint: GET /patients/{id}/vitals.

Thin adapter around pimap_dashboard.api.core.get_vitals.
"""

import json
import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from .core import get_vitals, parse_max_records


def get_patient_vitals(event, context):
    """Lambda handler: GET /patients/{patient_id}/vitals
    (also routed at /api/patients/{patient_id}/vitals).

    By default all records for the newest ICU stay are returned; pass
    ?maxRecords=N to limit.
    """
    params = event.get("queryStringParameters") or {}
    max_records = parse_max_records(params.get("maxRecords"))
    status, body = get_vitals(
        event["pathParameters"]["patient_id"], max_records=max_records
    )
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }