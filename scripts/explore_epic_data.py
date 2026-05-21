#!/usr/bin/env python3
"""Explore available FHIR data for Epic sandbox patients.

This script fetches raw FHIR resources to help discover what data is actually
available, which fields are populated, and how to query for additional info.

Usage:
    python scripts/explore_epic_data.py [--patient-id ID] [--resource TYPE] [--verbose]

Examples:
    # Explore all patients summary
    python scripts/explore_epic_data.py

    # Deep dive on one patient
    python scripts/explore_epic_data.py --patient-id erXuFYUfucBZaryVksYEcMg3

    # Check what Observations exist for a patient
    python scripts/explore_epic_data.py --patient-id erXuFYUfucBZaryVksYEcMg3 --resource Observation

    # Check Encounter data
    python scripts/explore_epic_data.py --patient-id erXuFYUfucBZaryVksYEcMg3 --resource Encounter
"""

import sys
import os
import argparse
import json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pimap_epic.auth import EpicAuth, EpicAuthError
from pimap_epic.fhir_client import EpicFHIRClient, FHIRRequestError, SANDBOX_PATIENT_IDS


def explore_patient(client, patient_id, verbose=False):
    """Explore a single patient's available data."""
    print(f"\n{'='*60}")
    print(f"Patient: {patient_id}")
    print('='*60)

    try:
        patient = client.get_patient(patient_id)
        print(f"\n--- Patient Resource ---")
        print(f"  ID: {patient.get('id')}")
        print(f"  Name: {patient.get('name', [{}])[0].get('given', [''])[0]} {patient.get('name', [{}])[0].get('family', '')}")
        print(f"  DOB: {patient.get('birthDate', 'N/A')}")
        print(f"  Gender: {patient.get('gender', 'N/A')}")

        extensions = patient.get("extension", [])
        for ext in extensions:
            url = ext.get("url", "")
            if "race" in url or "ethnicity" in url:
                for sub in ext.get("extension", []):
                    if sub.get("url") == "text":
                        print(f"  {url.split('/')[-1]}: {sub.get('valueString', 'N/A')}")

        if verbose:
            print(f"\n  Full resource:")
            print(json.dumps(patient, indent=2))

    except FHIRRequestError as e:
        print(f"  ERROR fetching patient: {e}")

    try:
        bundle = client.get("Encounter", params={"patient": patient_id, "_count": 5})
        entries = bundle.get("entry", [])
        print(f"\n--- Encounters ({len(entries)} found) ---")
        for entry in entries:
            enc = entry.get("resource", {})
            status = enc.get("status", "unknown")
            period = enc.get("period", {})
            start = period.get("start", "N/A")
            locations = enc.get("location", [])
            loc_display = locations[0].get("location", {}).get("display", "N/A") if locations else "N/A"
            print(f"  Status: {status}, Start: {start}, Location: {loc_display}")
            if verbose:
                print(json.dumps(enc, indent=2))
    except FHIRRequestError as e:
        print(f"  ERROR fetching encounters: {e}")

    try:
        bundle = client.get("Observation", params={"patient": patient_id, "_count": 50})
        entries = bundle.get("entry", [])
        print(f"\n--- Observations ({len(entries)} found) ---")

        code_counts = Counter()
        for entry in entries:
            obs = entry.get("resource", {})
            codings = obs.get("code", {}).get("coding", [])
            for coding in codings:
                code = coding.get("code", "unknown")
                display = coding.get("display", "")[:40]
                code_counts[(code, display)] += 1

        print(f"  Observation types by LOINC code:")
        for (code, display), count in code_counts.most_common(15):
            print(f"    {code}: {count}x - {display}")

        if verbose:
            print(f"\n  Sample observation:")
            if entries:
                print(json.dumps(entries[0].get("resource", {}), indent=2))

    except FHIRRequestError as e:
        print(f"  ERROR fetching observations: {e}")

    try:
        bundle = client.get("Condition", params={"patient": patient_id, "_count": 20})
        entries = bundle.get("entry", [])
        print(f"\n--- Conditions ({len(entries)} found) ---")
        for entry in entries:
            cond = entry.get("resource", {})
            code = cond.get("code", {}).get("coding", [{}])[0].get("display", "N/A")
            status = cond.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "unknown")
            print(f"  {status}: {code[:60]}")
            if verbose:
                print(json.dumps(cond, indent=2))
    except FHIRRequestError as e:
        print(f"  ERROR (or no access) fetching conditions: {e}")


def explore_resource_type(client, patient_id, resource_type, verbose=False):
    """Explore a specific resource type for a patient."""
    print(f"\n{'='*60}")
    print(f"Resource: {resource_type} for patient {patient_id}")
    print('='*60)

    try:
        bundle = client.get(resource_type, params={"patient": patient_id, "_count": 20})
        entries = bundle.get("entry", [])
        print(f"Found {len(entries)} {resource_type} resources")

        if verbose:
            for i, entry in enumerate(entries[:3]):
                print(f"\n--- Entry {i+1} ---")
                print(json.dumps(entry.get("resource", {}), indent=2))
        else:
            print("\nSummary of resource types/IDs:")
            for entry in entries:
                res = entry.get("resource", {})
                res_id = res.get("id", "unknown")
                res_type = res.get("resourceType", "unknown")
                print(f"  {res_type}/{res_id}")

    except FHIRRequestError as e:
        print(f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="Explore Epic FHIR sandbox data")
    parser.add_argument("--patient-id", help="Specific patient ID to explore")
    parser.add_argument("--resource", help="Specific resource type to query (e.g., Observation, Encounter, Condition)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full JSON responses")
    args = parser.parse_args()

    print("=== Epic FHIR Data Explorer ===\n")

    try:
        auth = EpicAuth()
        client = EpicFHIRClient(auth)
        print("Connected to Epic FHIR sandbox")
    except EpicAuthError as e:
        print(f"Authentication failed: {e}")
        sys.exit(1)

    if args.patient_id and args.resource:
        explore_resource_type(client, args.patient_id, args.resource, args.verbose)
    elif args.patient_id:
        explore_patient(client, args.patient_id, args.verbose)
    else:
        print(f"\nExploring {len(SANDBOX_PATIENT_IDS)} sandbox patients...\n")
        for pid in SANDBOX_PATIENT_IDS[:5]:
            explore_patient(client, pid, args.verbose)

    print("\n" + "="*60)
    print("Tips:")
    print("  - Use --patient-id to focus on one patient")
    print("  - Use --resource Observation to see all observation types")
    print("  - Use --resource Encounter to see admission/location data")
    print("  - Use --verbose to see full JSON responses")
    print("="*60)


if __name__ == "__main__":
    main()
