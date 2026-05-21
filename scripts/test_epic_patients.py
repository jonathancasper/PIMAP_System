#!/usr/bin/env python3
"""Epic patients endpoint diagnostic tool.

Tests the dashboard get_patients function to diagnose API issues.

Usage:
    python scripts/test_epic_patients.py [--verbose]

Exit codes:
    0 - Success
    1 - Failed to fetch patients
    2 - Failed to transform patients
    3 - Configuration error
    4 - Network error
"""

import sys
import os
import argparse
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pimap_epic.auth import EpicAuth, EpicAuthError
from pimap_epic.fhir_client import EpicFHIRClient, FHIRRequestError, SANDBOX_PATIENT_IDS


def test_patients_endpoint(verbose=False):
    """Test fetching and transforming patients."""
    print("=== Epic Patients Endpoint Diagnostic ===\n")
    
    try:
        print("Configuration:")
        auth = EpicAuth()
        print(f"  Client ID: loaded successfully")
        print(f"  Private key: loaded successfully")
        print(f"  Token endpoint: {auth.token_endpoint}")
    except EpicAuthError as e:
        print(f"  ERROR - {e}\n")
        print("Result: FAILED - Configuration error")
        return 3
    
    client = EpicFHIRClient(auth)
    
    print(f"\nFetching patients...")
    print(f"  Patient IDs: {len(SANDBOX_PATIENT_IDS)} sandbox patients")
    
    if verbose:
        print(f"  IDs: {', '.join(SANDBOX_PATIENT_IDS[:3])}...")
    
    try:
        patients = client.get_all_dashboard_patients()
        print(f"  Status: Success")
        print(f"  Patients fetched: {len(patients)}")
    except FHIRRequestError as e:
        print(f"  Status: FAILED")
        print(f"  Error: {e}")
        if verbose:
            print(f"\nFull traceback:")
            traceback.print_exc()
        print(f"\nResult: FAILED - FHIR request error")
        return 1
    except EpicAuthError as e:
        print(f"  Status: FAILED")
        print(f"  Auth error: {e}")
        if verbose:
            print(f"\nFull traceback:")
            traceback.print_exc()
        print(f"\nResult: FAILED - Authentication error")
        return 1
    except Exception as e:
        print(f"  Status: FAILED")
        print(f"  Unexpected error: {type(e).__name__}: {e}")
        if verbose:
            print(f"\nFull traceback:")
            traceback.print_exc()
        print(f"\nResult: FAILED - Unexpected error")
        return 1
    
    if len(patients) == 0:
        print("\nWarning: No patients returned")
        print("Result: SUCCESS (but empty patient list)")
        return 0
    
    print(f"\nSample patient:")
    sample = patients[0]
    print(f"  ID: {sample.get('patient_id', 'N/A')}")
    print(f"  Name: {sample.get('first_name', 'N/A')} {sample.get('last_name', 'N/A')}")
    print(f"  DOB: {sample.get('dob', 'N/A')}")
    print(f"  Gender: {sample.get('gender', 'N/A')}")
    print(f"  Data source: {sample.get('data_source', 'N/A')}")
    
    if verbose:
        print(f"\nFull patient data:")
        print(json.dumps(patients, indent=2))
    
    print(f"\nResult: SUCCESS - Fetched {len(patients)} patients")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose Epic patients endpoint issues"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information and full patient data"
    )
    args = parser.parse_args()
    
    exit_code = test_patients_endpoint(verbose=args.verbose)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
