#!/usr/bin/env python3
"""Epic authentication diagnostic tool.

Tests the Epic FHIR OAuth 2.0 authentication flow and provides detailed
debugging information for troubleshooting deployment issues.

Usage:
    python scripts/test_epic_auth.py [--verbose] [--dry-run] [--fingerprint-only]

Exit codes:
    0 - Success (auth and FHIR call both work)
    1 - Auth failed
    2 - Auth succeeded but FHIR call failed
    3 - Configuration error
    4 - Network error
"""

import sys
import os
import argparse
import hashlib
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import jwt

from pimap_epic.auth import EpicAuth, EpicAuthError, load_client_id, load_private_key
from pimap_epic.fhir_client import EpicFHIRClient, FHIRRequestError, SANDBOX_PATIENT_IDS


def get_config_fingerprint(client_id, private_key, token_endpoint, jwt_kid, jwt_jku):
    """Generate SHA256 fingerprints of configuration components."""
    client_id_hash = hashlib.sha256(client_id.encode()).hexdigest()[:16]
    private_key_hash = hashlib.sha256(private_key.encode()).hexdigest()[:16]
    full_string = f"{client_id}|{private_key}|{token_endpoint}|{jwt_kid}|{jwt_jku}"
    full_hash = hashlib.sha256(full_string.encode()).hexdigest()[:16]
    
    return {
        "client_id_hash": client_id_hash,
        "private_key_hash": private_key_hash,
        "full_config_hash": full_hash,
    }


def get_client_id_source():
    """Determine where client ID was loaded from."""
    if os.environ.get("EPIC_CLIENT_ID", "").strip():
        return "EPIC_CLIENT_ID env var"
    
    config_path = Path.home() / ".config" / "epic_fhir" / "client_id"
    if config_path.is_file():
        return f"~/.config/epic_fhir/client_id"
    
    return "unknown"


def get_private_key_source():
    """Determine where private key was loaded from."""
    if os.environ.get("EPIC_PRIVATE_KEY", "").strip():
        return "EPIC_PRIVATE_KEY env var"
    
    path_str = os.environ.get("EPIC_PRIVATE_KEY_PATH", "").strip()
    if path_str:
        return f"EPIC_PRIVATE_KEY_PATH env var ({path_str})"
    
    default_paths = [
        Path(__file__).resolve().parent.parent / "pimap_epic" / "privatekey.pem",
        Path(__file__).resolve().parent.parent / "pimap_cert" / "privatekey.pem",
    ]
    for p in default_paths:
        if p.is_file():
            return f"default path ({p})"
    
    return "unknown"


def check_jwks_url(jwks_url):
    """Check if JWKS URL is accessible."""
    try:
        resp = requests.get(jwks_url, timeout=10)
        if resp.status_code == 200:
            try:
                jwks = resp.json()
                keys = jwks.get("keys", [])
                return True, f"accessible ({len(keys)} key(s) found)"
            except:
                return True, "accessible (invalid JSON)"
        else:
            return False, f"HTTP {resp.status_code}"
    except requests.RequestException as e:
        return False, str(e)


def inspect_jwt(auth):
    """Extract JWT payload and headers without sending."""
    assertion = auth._build_jwt()
    
    header = jwt.get_unverified_header(assertion)
    payload = jwt.decode(assertion, options={"verify_signature": False})
    
    return header, payload, assertion


def mask_token(token, show_first=20, show_last=20):
    """Return truncated version of token for display."""
    if len(token) <= show_first + show_last + 10:
        return token[:10] + "..."
    return f"{token[:show_first]}...{token[-show_last:]}"


def test_fhir_call(client, patient_id):
    """Test FHIR API call."""
    try:
        patient = client.get_patient(patient_id)
        names = patient.get("name", [{}])
        n = names[0] if names else {}
        given = n.get("given", [])
        first_name = given[0] if given else "Unknown"
        last_name = n.get("family", "Unknown")
        dob = patient.get("birthDate", "Unknown")
        return True, f"Patient found: {first_name} {last_name} (DOB: {dob})"
    except FHIRRequestError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def validate_private_key(private_key):
    """Check if private key is valid for RS384 signing."""
    try:
        test_payload = {"test": "data"}
        jwt.encode(test_payload, private_key, algorithm="RS384")
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose Epic FHIR authentication issues"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed JWT and request/response information"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and inspect JWT but don't send to Epic"
    )
    parser.add_argument(
        "--fingerprint-only",
        action="store_true",
        help="Output only the configuration fingerprint"
    )
    args = parser.parse_args()
    
    print("=== Epic Auth Diagnostic ===\n")
    
    try:
        client_id = load_client_id()
        private_key = load_private_key()
    except EpicAuthError as e:
        print(f"Configuration:\n  ERROR - {e}\n")
        print("Result: FAILED - Configuration error")
        sys.exit(3)
    
    client_id_source = get_client_id_source()
    private_key_source = get_private_key_source()
    valid_key = validate_private_key(private_key)
    
    if args.fingerprint_only:
        auth = EpicAuth(client_id=client_id, private_key=private_key)
        fingerprint = get_config_fingerprint(
            client_id, private_key, auth.token_endpoint, auth.jwt_kid, auth.jwt_jku
        )
        print(fingerprint["full_config_hash"])
        sys.exit(0)
    
    print("Configuration:")
    print(f"  Client ID: loaded from {client_id_source}")
    print(f"  Private key: loaded from {private_key_source}")
    print(f"  Private key: {'valid RS384 key' if valid_key else 'INVALID (not a valid RS384 key)'}")
    
    if not valid_key:
        print("\nResult: FAILED - Invalid private key")
        sys.exit(3)
    
    auth = EpicAuth(client_id=client_id, private_key=private_key)
    
    print(f"  Token endpoint: {auth.token_endpoint}")
    print(f"  JWT kid: {auth.jwt_kid}")
    print(f"  JWT jku: {auth.jwt_jku}")
    
    fingerprint = get_config_fingerprint(
        client_id, private_key, auth.token_endpoint, auth.jwt_kid, auth.jwt_jku
    )
    
    print("\nFingerprint:")
    print(f"  Client ID hash: {fingerprint['client_id_hash']}")
    print(f"  Private key hash: {fingerprint['private_key_hash']}")
    print(f"  Full config hash: {fingerprint['full_config_hash']}")
    
    jwks_accessible, jwks_status = check_jwks_url(auth.jwt_jku)
    print(f"  JWKS URL: {jwks_status}")
    
    if not jwks_accessible:
        print("\n  WARNING: JWKS URL is not accessible. Epic cannot verify your JWT signature.")
    
    if args.dry_run:
        if args.verbose:
            header, payload, assertion = inspect_jwt(auth)
            print("\nJWT Payload:")
            for key in ["iss", "sub", "aud", "jti", "iat", "exp", "nbf"]:
                if key in payload:
                    val = payload[key]
                    if key in ["iat", "exp", "nbf"]:
                        from datetime import datetime, timezone
                        dt = datetime.fromtimestamp(val, tz=timezone.utc)
                        print(f"  {key}: {val} ({dt.isoformat()})")
                    else:
                        print(f"  {key}: {val}")
            
            print("\nJWT Headers:")
            for key, val in header.items():
                print(f"  {key}: {val}")
            
            print(f"\nSigned JWT: {mask_token(assertion)}")
        
        print("\nResult: DRY RUN - JWT built successfully (not sent to Epic)")
        sys.exit(0)
    
    if args.verbose:
        header, payload, assertion = inspect_jwt(auth)
        print("\nJWT Payload:")
        for key in ["iss", "sub", "aud", "jti", "iat", "exp", "nbf"]:
            if key in payload:
                val = payload[key]
                if key in ["iat", "exp", "nbf"]:
                    from datetime import datetime, timezone
                    dt = datetime.fromtimestamp(val, tz=timezone.utc)
                    print(f"  {key}: {val} ({dt.isoformat()})")
                else:
                    print(f"  {key}: {val}")
        
        print("\nJWT Headers:")
        for key, val in header.items():
            print(f"  {key}: {val}")
        
        print(f"\nSigned JWT: {mask_token(assertion)}")
        
        print("\nToken Request:")
        print(f"  POST {auth.token_endpoint}")
        print("  Content-Type: application/x-www-form-urlencoded")
        print("  Body:")
        print("    grant_type=client_credentials")
        print("    client_assertion_type=urn:ietf:params:oauth:client-assertion-type:jwt-bearer")
        print(f"    client_assertion={mask_token(assertion, 30, 30)}")
    
    try:
        token = auth.get_access_token()
        print("\nToken Response:")
        print("  Status: 200 OK")
        print(f"  Access token received: {mask_token(token)}")
        print(f"  Expires in: {int(auth._token_expires_at - auth._token_expires_at + 3600)} seconds (approx)")
    except EpicAuthError as e:
        print("\nToken Response:")
        error_str = str(e)
        if "401" in error_str:
            print("  Status: 401 Unauthorized")
        elif "403" in error_str:
            print("  Status: 403 Forbidden")
        elif "400" in error_str:
            print("  Status: 400 Bad Request")
        else:
            print(f"  ERROR: {error_str}")
        
        if args.verbose and "(" in error_str:
            try:
                body_start = error_str.index("(")
                body_str = error_str[body_start:]
                print(f"  Body: {body_str}")
            except:
                pass
        
        print(f"\nResult: FAILED - {error_str}")
        sys.exit(1)
    except requests.RequestException as e:
        print(f"\nToken Response:")
        print(f"  Network error: {e}")
        print(f"\nResult: FAILED - Network error")
        sys.exit(4)
    
    print("\nFHIR API Test:")
    test_patient_id = SANDBOX_PATIENT_IDS[0]
    print(f"  Fetching patient: {test_patient_id}")
    
    client = EpicFHIRClient(auth)
    
    try:
        success, message = test_fhir_call(client, test_patient_id)
        if success:
            print(f"  Status: 200 OK")
            print(f"  {message}")
            print("\nResult: SUCCESS")
            sys.exit(0)
        else:
            print(f"  Status: ERROR")
            print(f"  {message}")
            print("\nResult: FAILED - FHIR call failed (auth succeeded but token may be invalid)")
            sys.exit(2)
    except requests.RequestException as e:
        print(f"  Network error: {e}")
        print("\nResult: FAILED - Network error during FHIR call")
        sys.exit(4)


if __name__ == "__main__":
    main()
