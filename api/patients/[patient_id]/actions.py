"""Vercel serverless function: POST /api/patients/{patient_id}/actions.

Thin adapter around pimap_dashboard.api.core.record_action, with an
in-memory action store (the AWS deployment uses DynamoDB instead).
"""

import sys
import os
import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Add project root to sys.path so pimap_dashboard is importable
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'),
)

from pimap_dashboard.api.core import record_action

# In-memory action store (per instance lifetime)
_actions = []


def _put_action(patient_id, timestamp, action_taken, staff_id):
    _actions.append({
        "patient_id": patient_id,
        "timestamp": timestamp,
        "actions_taken": action_taken,
        "action_by_staff": staff_id,
        "action_timestamp": timestamp,
    })


class handler(BaseHTTPRequestHandler):
    """Handle POST /api/patients/{patient_id}/actions."""

    def do_POST(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        patient_id = params.get('patient_id', [None])[0]

        if not patient_id:
            self._send_json(400, {"error": "patient_id is required"})
            return

        try:
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length) or b'{}')
        except (ValueError, json.JSONDecodeError):
            body = {}

        status, result = record_action(patient_id, body, put_action=_put_action)
        self._send_json(status, result)

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self._send_json(200, {})

    def _send_json(self, status, data):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())