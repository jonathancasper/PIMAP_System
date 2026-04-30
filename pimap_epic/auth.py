"""Epic FHIR OAuth 2.0 authentication pipeline for PIMAP.

Adapted from pimap_cert/demo/epic_auth.py. Implements the Backend System OAuth 2.0
flow (client_credentials with JWT bearer assertion) for Epic's FHIR API.

Configuration (checked in order):
  Private key: EPIC_PRIVATE_KEY env var (raw PEM) > EPIC_PRIVATE_KEY_PATH env var > default paths
  Client ID:   EPIC_CLIENT_ID env var > ~/.config/epic_fhir/client_id
"""

import jwt
import os
import uuid
import time
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path


DEFAULT_TOKEN_ENDPOINT = "https://fhir.epic.com/interconnect-fhir-oauth/oauth2/token"

DEFAULT_JWT_KID = "pimap-2026-02-13"
DEFAULT_JWT_JKU = "https://genome-test.soe.ucsc.edu/~jcasper/pimap/jwks.json"

TOKEN_REFRESH_MARGIN_SECONDS = 60

_DEFAULT_KEY_PATHS = [
    Path(__file__).resolve().parent / "privatekey.pem",
    Path(__file__).resolve().parent.parent / "pimap_cert" / "privatekey.pem",
]


class EpicAuthError(Exception):
    """Raised when authentication with Epic fails."""


def load_private_key(
    env_var_content="EPIC_PRIVATE_KEY",
    env_var_path="EPIC_PRIVATE_KEY_PATH",
):
    """Load the private key from env var (raw PEM), env var (file path), or default paths."""
    raw = os.environ.get(env_var_content, "").strip()
    if raw:
        return raw

    path_str = os.environ.get(env_var_path, "").strip()
    if path_str:
        p = Path(path_str)
        if p.is_file():
            return p.read_text()
        raise EpicAuthError(f"Private key file not found: {p}")

    for p in _DEFAULT_KEY_PATHS:
        if p.is_file():
            return p.read_text()

    raise EpicAuthError(
        f"Private key not found. Set {env_var_content} (raw PEM content), "
        f"{env_var_path} (file path), or place privatekey.pem in the project directory."
    )


def load_client_id(env_var="EPIC_CLIENT_ID", config_path=None):
    """Load the Epic client ID from environment or ~/.config/epic_fhir/client_id."""
    value = os.environ.get(env_var, "").strip()
    if value:
        return value

    if config_path is None:
        config_path = Path.home() / ".config" / "epic_fhir" / "client_id"
    else:
        config_path = Path(config_path)

    if config_path.is_file():
        value = config_path.read_text().strip()
        if value:
            return value

    raise EpicAuthError(
        f"Epic client ID not found. Set the {env_var} environment variable "
        f"or write the ID to {config_path}"
    )


class EpicAuth:
    """Manages OAuth 2.0 client-credentials authentication against Epic FHIR.

    Generates a signed JWT, exchanges it for a bearer token, and caches the
    token until it expires (minus a safety margin).
    """

    def __init__(
        self,
        client_id=None,
        private_key=None,
        private_key_path=None,
        token_endpoint=DEFAULT_TOKEN_ENDPOINT,
        jwt_kid=DEFAULT_JWT_KID,
        jwt_jku=DEFAULT_JWT_JKU,
    ):
        self.client_id = client_id or load_client_id()
        self.token_endpoint = token_endpoint
        self.jwt_kid = jwt_kid
        self.jwt_jku = jwt_jku

        if private_key:
            self._private_key = private_key
        elif private_key_path:
            self._private_key = Path(private_key_path).read_text()
        else:
            self._private_key = load_private_key()

        self._access_token = None
        self._token_expires_at = 0.0

    def get_access_token(self):
        """Return a valid bearer token, refreshing automatically if needed."""
        if self._token_is_valid():
            return self._access_token
        self._refresh_token()
        return self._access_token

    def _token_is_valid(self):
        return (
            self._access_token is not None
            and time.time() < self._token_expires_at - TOKEN_REFRESH_MARGIN_SECONDS
        )

    def _build_jwt(self):
        now = datetime.now(timezone.utc)
        iat = int(now.timestamp())
        payload = {
            "iss": self.client_id,
            "sub": self.client_id,
            "aud": self.token_endpoint,
            "jti": str(uuid.uuid4()),
            "exp": int((now + timedelta(minutes=5)).timestamp()),
            "nbf": iat,
            "iat": iat,
        }
        headers = {
            "alg": "RS384",
            "typ": "JWT",
            "kid": self.jwt_kid,
            "jku": self.jwt_jku,
        }
        return jwt.encode(
            payload,
            self._private_key,
            algorithm="RS384",
            headers=headers,
        )

    def _refresh_token(self):
        assertion = self._build_jwt()
        data = {
            "grant_type": "client_credentials",
            "client_assertion_type": (
                "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
            ),
            "client_assertion": assertion,
        }
        resp = requests.post(
            self.token_endpoint,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if resp.status_code != 200:
            raise EpicAuthError(
                f"Token request failed ({resp.status_code}): {resp.text}"
            )
        body = resp.json()
        if "access_token" not in body:
            raise EpicAuthError(f"No access_token in response: {body}")

        self._access_token = body["access_token"]
        expires_in = body.get("expires_in", 3600)
        self._token_expires_at = time.time() + expires_in
