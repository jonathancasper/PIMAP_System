"""Vercel serverless function: POST /api/patients/{patient_id}/predict.

Thin adapter around pimap_dashboard.api.core.predict (XGBoost model when
XGBOOST_MODEL_PATH is configured, Braden mock otherwise).
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

from pimap_dashboard.api.core import predict


class handler(BaseHTTPRequestHandler):
    """Handle POST /api/patients/{patient_id}/predict."""

    def do_POST(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        patient_id = params.get('patient_id', [None])[0]

        if not patient_id:
            self._send_json(400, {"error": "patient_id is required"})
            return

        status, body = predict(patient_id)
        self._send_json(status, body)

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