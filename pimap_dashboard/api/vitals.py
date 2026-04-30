"""Lambda handler for patient vitals endpoint."""

import json
import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from pimap_epic import EpicAuth, EpicFHIRClient

_fhir_client = None


def _get_client():
    global _fhir_client
    if _fhir_client is None:
        auth = EpicAuth()
        _fhir_client = EpicFHIRClient(auth)
    return _fhir_client


def get_patient_vitals(event, context):
    """Lambda handler: GET /patients/{patient_id}/vitals

    Returns vitals history for a specific patient from Epic FHIR.
    """
    try:
        patient_id = event["pathParameters"]["patient_id"]
        client = _get_client()
        vitals = client.get_patient_vitals(patient_id, max_records=20)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(vitals),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)}),
        }
