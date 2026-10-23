"""Vercel serverless function: GET /api/patients.

Thin adapter around pimap_dashboard.api.core.list_patients.
"""

import sys
import os
import json
from http.server import BaseHTTPRequestHandler

# Add project root to sys.path so pimap_dashboard is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pimap_dashboard.api.core import list_patients


class handler(BaseHTTPRequestHandler):
    """Handle GET /api/patients - list all patients."""

    def do_GET(self):
        status, body = list_patients()
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