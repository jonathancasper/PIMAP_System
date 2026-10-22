"""Adjust MIMIC demo dates so patients appear as current ICU stays.

Shifts all timestamps so each patient's most recent vitals record
falls within the past 1-3 hours. Preserves the relative spacing between
records and shifts DOB by the same delta to keep age consistent.

Usage:
    python3 scripts/adjust_mimic_dates.py [--patients PATH] [--vitals PATH]
                                          [--predictions PATH] [--output-dir DIR]

Defaults:
    --patients    data/mimic_demo/patients_curated.json
    --vitals      data/mimic_demo/vitals_curated.json.gz
    --predictions data/mimic_demo/predictions_curated.json
    --output-dir  data/mimic_demo/

Outputs (in --output-dir):
    dateshifted_patients_curated.json
    dateshifted_vitals_curated.json
    dateshifted_predictions_curated.json   (if --predictions exists)
"""

import argparse
import gzip
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List


def parse_timestamp(ts: str) -> datetime:
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts}")


def format_timestamp(dt: datetime, original: str) -> str:
    if "T" not in original:
        return dt.strftime("%Y-%m-%d")
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def compute_shift(most_recent_ts: datetime) -> timedelta:
    now = datetime.now()
    offset_minutes = random.randint(60, 180)
    target = now - timedelta(minutes=offset_minutes)
    target = target.replace(
        minute=most_recent_ts.minute,
        second=most_recent_ts.second,
    )
    return target - most_recent_ts


def adjust_patients(
    patients: List[Dict[str, Any]], shifts: Dict[str, timedelta]
) -> List[Dict[str, Any]]:
    adjusted = []
    for p in patients:
        pid = p["patient_id"]
        delta = shifts.get(pid)
        if delta is None:
            adjusted.append(p)
            continue
        new_p = dict(p)
        if new_p.get("dob"):
            dob = parse_timestamp(new_p["dob"])
            new_p["dob"] = format_timestamp(dob + delta, new_p["dob"])
        adjusted.append(new_p)
    return adjusted


def adjust_vitals(
    vitals: List[Dict[str, Any]], shifts: Dict[str, timedelta]
) -> List[Dict[str, Any]]:
    adjusted = []
    for v in vitals:
        pid = v["patient_id"]
        delta = shifts.get(pid)
        if delta is None:
            adjusted.append(v)
            continue
        new_v = dict(v)
        if new_v.get("timestamp"):
            ts = parse_timestamp(new_v["timestamp"])
            new_v["timestamp"] = format_timestamp(ts + delta, new_v["timestamp"])
        adjusted.append(new_v)
    return adjusted


def adjust_predictions(
    predictions: List[Dict[str, Any]], shifts: Dict[str, timedelta]
) -> List[Dict[str, Any]]:
    """Shift prediction timestamps by the same per-patient delta as vitals.

    Only the 'timestamp' field is shifted; risk scores and other fields
    are preserved unchanged.
    """
    adjusted = []
    for p in predictions:
        pid = p.get("patient_id")
        delta = shifts.get(pid)
        if delta is None:
            adjusted.append(p)
            continue
        new_p = dict(p)
        if new_p.get("timestamp"):
            ts = parse_timestamp(new_p["timestamp"])
            new_p["timestamp"] = format_timestamp(ts + delta, new_p["timestamp"])
        adjusted.append(new_p)
    return adjusted


def compute_shifts(vitals: List[Dict[str, Any]]) -> Dict[str, timedelta]:
    by_patient: Dict[str, List[str]] = {}
    for v in vitals:
        pid = v["patient_id"]
        ts = v.get("timestamp")
        if ts:
            by_patient.setdefault(pid, []).append(ts)

    shifts = {}
    for pid, timestamps in by_patient.items():
        parsed = [parse_timestamp(t) for t in timestamps]
        most_recent = max(parsed)
        shifts[pid] = compute_shift(most_recent)
    return shifts


def main():
    parser = argparse.ArgumentParser(
        description="Adjust MIMIC demo dates to appear as current ICU stays"
    )
    parser.add_argument(
        "--patients",
        default="data/mimic_demo/patients_curated.json",
        help="Path to patients_curated.json",
    )
    parser.add_argument(
        "--vitals",
        default="data/mimic_demo/vitals_curated.json.gz",
        help="Path to vitals_curated.json.gz (or .json)",
    )
    parser.add_argument(
        "--predictions",
        default="data/mimic_demo/predictions_curated.json",
        help="Path to predictions_curated.json (shifted to dateshifted_*)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/mimic_demo/",
        help="Directory for output files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible shifts",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    base = Path(__file__).resolve().parent.parent
    patients_path = base / args.patients if not Path(args.patients).is_absolute() else Path(args.patients)
    vitals_path = base / args.vitals if not Path(args.vitals).is_absolute() else Path(args.vitals)
    predictions_path = base / args.predictions if not Path(args.predictions).is_absolute() else Path(args.predictions)
    output_dir = base / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    with open(patients_path) as f:
        patients = json.load(f)

    if vitals_path.suffix == ".gz":
        with gzip.open(vitals_path, "rt") as f:
            vitals = json.load(f)
    else:
        with open(vitals_path) as f:
            vitals = json.load(f)

    shifts = compute_shifts(vitals)

    new_patients = adjust_patients(patients, shifts)
    new_vitals = adjust_vitals(vitals, shifts)

    output_dir.mkdir(parents=True, exist_ok=True)

    patients_out = output_dir / "dateshifted_patients_curated.json"
    with open(patients_out, "w") as f:
        json.dump(new_patients, f, indent=2)

    vitals_out = output_dir / "dateshifted_vitals_curated.json"
    with open(vitals_out, "w") as f:
        json.dump(new_vitals, f, indent=2)

    predictions_out = output_dir / "dateshifted_predictions_curated.json"
    if predictions_path.exists():
        with open(predictions_path) as f:
            predictions = json.load(f)
        new_predictions = adjust_predictions(predictions, shifts)
        with open(predictions_out, "w") as f:
            json.dump(new_predictions, f, indent=2)
    else:
        predictions_out = None

    print(f"Shifted {len(shifts)} patients")
    print(f"Patients written to: {patients_out}")
    print(f"Vitals written to:   {vitals_out}")
    if predictions_out is not None:
        print(f"Predictions written to: {predictions_out}")
    else:
        print(f"No predictions file found at {predictions_path}; skipped")
    print()
    print("Per-patient shifts:")
    for pid in sorted(shifts):
        delta = shifts[pid]
        days = delta.total_seconds() / 86400
        print(f"  {pid}: {delta} ({days:+.1f} days)")


if __name__ == "__main__":
    main()
