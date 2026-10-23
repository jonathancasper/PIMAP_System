"""Vercel serverless function: GET /api/patients/{patient_id}/vitals.

Thin adapter around pimap_dashboard.api.core.get_vitals.
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

from pimap_dashboard.api.core import get_vitals, parse_max_records


class handler(BaseHTTPRequestHandler):
    """Handle GET /api/patients/{patient_id}/vitals."""

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        patient_id = params.get('patient_id', [None])[0]

        if not patient_id:
            self._send_json(400, {"error": "patient_id is required"})
            return

        max_records = parse_max_records(
            (params.get('maxRecords') or [None])[0]
        )
        status, body = get_vitals(patient_id, max_records=max_records)
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