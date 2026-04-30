"""Lambda handler for patient list endpoint."""

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


def get_patients(event, context):
    """Lambda handler: GET /patients

    Returns list of all patients from Epic FHIR.
    """
    try:
        client = _get_client()
        patients = client.get_all_dashboard_patients()

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "*",
            },
            "body": json.dumps(patients),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)}),
        }
