"""Lambda handler for staff actions endpoint."""

import json
import boto3
import os
from datetime import datetime

dynamodb = boto3.resource("dynamodb")
vitals_table = dynamodb.Table(os.environ.get("VITALS_TABLE", "patient_vitals"))


def update_action(event, context):
    """Lambda handler: POST /patients/{patient_id}/actions

    Records a staff action for a patient.
    """
    try:
        patient_id = event["pathParameters"]["patient_id"]
        body = json.loads(event["body"])

        action_taken = body.get("action_taken")
        staff_id = body.get("staff_id")

        if not action_taken:
            return {
                "statusCode": 400,
                "headers": {"Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": "action_taken is required"}),
            }

        timestamp = datetime.now().isoformat() + "Z"

        vitals_table.put_item(
            Item={
                "patient_id": patient_id,
                "timestamp": timestamp,
                "actions_taken": action_taken,
                "action_by_staff": staff_id or "",
                "action_timestamp": timestamp,
            }
        )

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(
                {
                    "message": "Action recorded successfully",
                    "patient_id": patient_id,
                    "action_taken": action_taken,
                }
            ),
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)}),
        }
