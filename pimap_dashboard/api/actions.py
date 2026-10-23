"""Lambda handler for staff actions endpoint: POST /patients/{id}/actions.

Thin adapter around pimap_dashboard.api.core.record_action, with a
DynamoDB-backed action store (VitalsTable).
"""

import json
import sys
import os
import boto3
from datetime import datetime

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from .core import record_action


dynamodb = boto3.resource("dynamodb")
vitals_table = dynamodb.Table(os.environ.get("VITALS_TABLE", "patient_vitals"))


def _put_action(patient_id, timestamp, action_taken, staff_id):
    vitals_table.put_item(
        Item={
            "patient_id": patient_id,
            "timestamp": timestamp,
            "actions_taken": action_taken,
            "action_by_staff": staff_id,
            "action_timestamp": timestamp,
        }
    )


def update_action(event, context):
    """Lambda handler: POST /patients/{patient_id}/actions
    (also routed at /api/patients/{patient_id}/actions).
    """
    try:
        patient_id = event["pathParameters"]["patient_id"]
        body = json.loads(event["body"] or "{}")
        status, result = record_action(
            patient_id, body, put_action=_put_action
        )
    except Exception as e:
        status, result = 500, {"error": str(e)}
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(result),
    }