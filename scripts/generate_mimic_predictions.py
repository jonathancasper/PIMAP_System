"""Generate historical predictions for MIMIC demo patients.

For each timestamp in the vitals data, runs the predictor as if that
timestamp were "now" — using only vitals records at or before that point.
This simulates what predictions would have been made in real-time.

Usage:
    python3 scripts/generate_mimic_predictions.py [--vitals PATH] [--output PATH]
                                                  [--predictor {mock,xgboost}]
                                                  [--patients PATH] [--model-dir DIR]

Outputs predictions_curated.json with prediction records for each timestamp.

The default vitals input matches the file the dashboard serves
(vitals_curated.json.gz) so prediction timestamps join with dashboard
vitals timestamps.
"""

import argparse
import gzip
import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pimap_predict import (
    BRADEN_FIELDS,
    FeatureExtractor,
    XGBoostPredictor,
    get_predictor,
)


def parse_timestamp(ts: str) -> datetime:
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts}")


# MIMIC raw vitals field names -> FeatureExtractor/predictor field names
# (same mapping as pimap_dashboard.api.data_source.MimicDataSource._map_fields)
FIELD_MAPPING = {
    "braden_sensory": "braden_sensory_perception",
    "spo2": "arterial_o2_saturation",
    "glucose": "glucose_whole_blood",
}


def apply_field_mapping(record: Dict[str, Any]) -> Dict[str, Any]:
    """Add mapped aliases so FeatureExtractor sees canonical field names."""
    mapped = dict(record)
    for raw, canonical in FIELD_MAPPING.items():
        if raw in mapped and canonical not in mapped:
            mapped[canonical] = mapped[raw]
    return mapped


def forward_fill_historical(
    records: List[Dict[str, Any]],
    target_timestamp: datetime,
    target_stay_id: Any = None,
) -> Optional[Dict[str, Any]]:
    """Forward-fill Braden values using only records before target_timestamp.

    Unlike the regular forward_fill_braden, this only considers records
    at or before the target timestamp, simulating historical prediction.

    If target_stay_id is given, the candidate pool is restricted to
    records from that ICU stay, so forward-fill never crosses a stay
    boundary. A fresh admission with no in-stay Braden data at or before
    the target time returns None (caller emits a NoData prediction).
    """
    eligible = [
        r for r in records
        if parse_timestamp(r.get("timestamp", "")) <= target_timestamp
    ]
    if target_stay_id is not None:
        eligible = [r for r in eligible if r.get("stay_id") == target_stay_id]

    if not eligible:
        return None

    eligible.sort(key=lambda r: parse_timestamp(r.get("timestamp", "")), reverse=True)

    enriched = dict(eligible[0])

    has_any_braden = False
    for field in BRADEN_FIELDS:
        if enriched.get(field) is not None:
            has_any_braden = True
            continue
        for record in eligible[1:]:
            if record.get(field) is not None:
                enriched[field] = record[field]
                has_any_braden = True
                break

    if not has_any_braden:
        return None

    imputed = list(enriched.get("_imputed_fields", []))
    for field in BRADEN_FIELDS:
        if enriched.get(field) is not None and eligible[0].get(field) is None:
            imputed.append(field)
    enriched["_imputed_fields"] = imputed

    return enriched


def generate_predictions(
    vitals: List[Dict[str, Any]],
    predictor_type: str = "mock",
    patients: Optional[List[Dict[str, Any]]] = None,
    model_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generate predictions for each timestamp using historical forward-fill.

    For the mock predictor, Braden values are forward-filled from records
    at or before the target timestamp (within the same ICU stay).

    For the XGBoost predictor, all records at or before the target
    timestamp (within the same ICU stay) are passed to the model's
    temporal feature extractor, with the prediction time set to the
    target timestamp so time windows match the historical moment.
    """

    by_patient: Dict[str, List[Dict[str, Any]]] = {}
    for v in vitals:
        by_patient.setdefault(v["patient_id"], []).append(v)

    for records in by_patient.values():
        records.sort(key=lambda r: parse_timestamp(r.get("timestamp", "")))

    demographics_by_id = {}
    for p in patients or []:
        demographics_by_id[str(p.get("patient_id"))] = p

    if predictor_type == "xgboost":
        if not model_dir:
            raise SystemExit(
                "xgboost predictor requires --model-dir or XGBOOST_MODEL_PATH"
            )
        predictor = get_predictor(model_dir)
        if not isinstance(predictor, XGBoostPredictor):
            raise SystemExit(
                f"Could not load XGBoost model from {model_dir}; "
                "check that hapi_xgb_v1.json and hapi_xgb_schema_v1.json exist there"
            )
    else:
        predictor = get_predictor()

    extractor = FeatureExtractor()

    predictions = []

    for pid, records in sorted(by_patient.items()):
        print(f"  patient {pid}: {len(records)} records...", flush=True)
        mapped_records = [apply_field_mapping(r) for r in records]
        parsed_records = [
            (parse_timestamp(r.get("timestamp", "")), r) for r in records
        ]
        demographics = demographics_by_id.get(str(pid), {})

        for record in mapped_records:
            ts = parse_timestamp(record.get("timestamp"))

            if predictor_type == "xgboost":
                stay_id = record.get("stay_id")
                history = [
                    r for (t, r) in parsed_records
                    if t <= ts and r.get("stay_id") == stay_id
                ]
                result = predictor.predict(
                    pid,
                    {"demographics": demographics, "vitals": history},
                    prediction_time=ts.replace(tzinfo=timezone.utc),
                )
            else:
                enriched = forward_fill_historical(
                    mapped_records, ts, target_stay_id=record.get("stay_id")
                )

                if enriched is None:
                    predictions.append({
                        "patient_id": pid,
                        "timestamp": record.get("timestamp"),
                        "risk_level": "NoData",
                        "risk_score": None,
                        "confidence": 0.0,
                        "model_version": predictor.get_model_version(),
                        "imputed_fields": [],
                    })
                    continue

                features = extractor.extract(enriched)
                result = predictor.predict(pid, features)

            predictions.append({
                "patient_id": pid,
                "timestamp": record.get("timestamp"),
                "risk_level": result.risk_level,
                "risk_score": result.risk_score,
                "confidence": result.confidence,
                "model_version": result.model_version,
                "imputed_fields": result.imputed_features,
            })

    predictions.sort(key=lambda p: (p["patient_id"], parse_timestamp(p["timestamp"])))

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Generate historical predictions for MIMIC demo data"
    )
    parser.add_argument(
        "--vitals",
        default="data/mimic_demo/vitals_curated.json.gz",
        help="Path to vitals JSON file (must match what the dashboard serves)",
    )
    parser.add_argument(
        "--patients",
        default="data/mimic_demo/patients_curated.json",
        help="Path to patients JSON file (demographics for the XGBoost model)",
    )
    parser.add_argument(
        "--output",
        default="data/mimic_demo/predictions_curated.json",
        help="Path for output predictions file",
    )
    parser.add_argument(
        "--predictor",
        choices=["mock", "xgboost"],
        default="mock",
        help="Predictor type to use",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("XGBOOST_MODEL_PATH", "model_input"),
        help="Directory containing hapi_xgb_v1.json and hapi_xgb_schema_v1.json "
        "(used only with --predictor xgboost; defaults to XGBOOST_MODEL_PATH)",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    vitals_path = base / args.vitals if not Path(args.vitals).is_absolute() else Path(args.vitals)
    patients_path = base / args.patients if not Path(args.patients).is_absolute() else Path(args.patients)
    output_path = base / args.output if not Path(args.output).is_absolute() else Path(args.output)

    if vitals_path.suffix == ".gz":
        with gzip.open(vitals_path, "rt") as f:
            vitals = json.load(f)
    else:
        with open(vitals_path) as f:
            vitals = json.load(f)

    print(f"Loaded {len(vitals)} vitals records")

    if args.predictor == "xgboost":
        with open(patients_path) as f:
            patients = json.load(f)
        print(f"Loaded {len(patients)} patient records for demographics")
    else:
        patients = None

    predictions = generate_predictions(
        vitals,
        predictor_type=args.predictor,
        patients=patients,
        model_dir=args.model_dir,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Generated {len(predictions)} predictions")
    print(f"Output: {output_path}")

    from collections import Counter
    level_counts = Counter(p["risk_level"] for p in predictions)
    print("\nRisk level distribution:")
    for level, count in sorted(level_counts.items()):
        print(f"  {level}: {count}")


if __name__ == "__main__":
    main()