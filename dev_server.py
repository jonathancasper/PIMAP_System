"""Local development server that emulates Vercel routing.

Run with: python3 dev_server.py [--port PORT]
Serves the dashboard (pimap_dashboard/frontend/index.html) at
http://localhost:3000 with the API mounted under /api/patients/...,
matching the Vercel deployment's routing.

Environment:
  DATA_SOURCE        'mimic' (default - MIMIC-IV demo) or 'epic'
  XGBOOST_MODEL_PATH optional directory containing hapi_xgb_v1.json +
                     hapi_xgb_schema_v1.json (enables the XGBoost model)
"""

import sys
import os
import re
import json
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

if not os.environ.get('MIMIC_DATA_PATH'):
    os.environ['MIMIC_DATA_PATH'] = os.path.join(PROJECT_ROOT, 'data', 'mimic_demo')

from pimap_dashboard.api.core import (
    list_patients,
    get_vitals,
    predict,
    record_action,
    parse_max_records,
)
from pimap_dashboard.api import data_source as _data_source_module

FRONTEND_INDEX = os.path.join(PROJECT_ROOT, 'pimap_dashboard', 'frontend', 'index.html')

# In-memory action store (AWS deployment uses DynamoDB instead)
_actions = []


def _put_action(patient_id, timestamp, action_taken, staff_id):
    _actions.append({
        "patient_id": patient_id,
        "timestamp": timestamp,
        "actions_taken": action_taken,
        "action_by_staff": staff_id,
        "action_timestamp": timestamp,
    })


class DevHandler(SimpleHTTPRequestHandler):
    """Routes /api/* requests to the shared dashboard API cores and
    serves the frontend for everything else."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == '/api/patients':
            self._handle(list_patients)
            return

        m_vitals = re.match(r'^/api/patients/([^/]+)/vitals$', path)
        if m_vitals:
            pid = m_vitals.group(1)
            max_records = parse_max_records((params.get('maxRecords') or [None])[0])
            self._handle(lambda: get_vitals(pid, max_records=max_records))
            return

        if path == '/' or path == '/index.html':
            self._serve_index()
            return

        self._handle(lambda: (404, {"error": f"Not found: {path}"}))

    def do_POST(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        m = re.match(r'^/api/patients/([^/]+)/(predict|actions)$', parsed.path)
        if not m:
            self._handle(lambda: (404, {"error": f"Not found: {parsed.path}"}))
            return

        patient_id = m.group(1)

        try:
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length) or b'{}')
        except (ValueError, json.JSONDecodeError):
            body = {}

        if m.group(2) == 'predict':
            self._handle(lambda: predict(patient_id))
        else:
            self._handle(
                lambda: record_action(patient_id, body, put_action=_put_action)
            )

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

    def _handle(self, fn):
        status, body = fn()
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(payload)

    def _serve_index(self):
        with open(FRONTEND_INDEX, 'rb') as f:
            payload = f.read()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        print(f"[dev] {self.address_string()} {fmt % args}")


def main():
    parser = argparse.ArgumentParser(
        description="Local dev server emulating Vercel routing"
    )
    parser.add_argument('--port', type=int, default=3000)
    args = parser.parse_args()

    # Reset any cached data source so this process picks up env vars
    _data_source_module.reset_data_source()

    print(f"Serving dashboard at http://localhost:{args.port}/")
    print(f"  DATA_SOURCE        = {os.environ.get('DATA_SOURCE', 'mimic')}")
    print(f"  XGBOOST_MODEL_PATH = {os.environ.get('XGBOOST_MODEL_PATH', '(unset)')}")
    server = HTTPServer(('localhost', args.port), DevHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == '__main__':
    main()