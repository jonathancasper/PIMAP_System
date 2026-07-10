#!/usr/bin/env python3
"""
Curate MIMIC-IV demo patients for PIMAP demonstration.

Selects a subset of patients with interesting clinical patterns:
- High risk (low Braden scores)
- Declining Braden scores over time
- Good Braden but low blood pressure (mixed signals)
- Low risk (stable high Braden)

Outputs curated JSON files suitable for Lambda bundling.
"""

import json
import gzip
from pathlib import Path
from datetime import datetime
import pandas as pd

SELECTED_PATIENTS = [
    # High Risk (bad Braden, has BP)
    '10015931',  # Braden 6-18, declined 15->11
    '10038081',  # Braden 6-12, declined 12->6
    '10007818',  # Braden 8-14, low BP (62)
    '10005817',  # Braden 8-20, low BP (65)
    '10039708',  # Braden 9-22, declined 19->14
    '10021487',  # Braden 9-18
    
    # Declining
    '10035631',  # 20->9 severe decline
    '10031757',  # 18->13
    
    # Good Braden + Low BP (mixed signals)
    '10020187',  # Braden 17, SBP 95
    '10012853',  # Braden 16, SBP 75
    '10026354',  # Braden 16, SBP 85
    
    # Low Risk (stable high Braden)
    '10022880',  # 19-19
    '10002930',  # 18-22
    '10029484',  # 18-18
    '10013049',  # 18-20
    '10009049',  # 18-21
    
    # One patient without BP data (for variety)
    '10006053',  # 8-12
]


def load_icustays(mimic_path: Path) -> pd.DataFrame:
    """Load ICU stays table."""
    stays_path = mimic_path / "icu" / "icustays.csv.gz"
    df = pd.read_csv(stays_path, compression='gzip')
    df['intime'] = pd.to_datetime(df['intime'])
    df['outtime'] = pd.to_datetime(df['outtime'])
    return df


def calculate_icu_duration(vital_timestamp: str, icu_intime: datetime) -> int:
    """Calculate days since ICU admission."""
    try:
        vt = datetime.fromisoformat(vital_timestamp.replace('Z', '+00:00'))
        delta = vt - icu_intime
        days = max(0, delta.days)
        return days
    except:
        return 0


def main():
    project_root = Path(__file__).parent.parent
    mimic_path = Path("/path/to/mimic-iv-clinical-database-demo-2.2")
    output_dir = project_root / "data" / "mimic_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading full MIMIC data...")
    with open(output_dir / "patients_mimic_demo.json") as f:
        all_patients = json.load(f)
    with open(output_dir / "vitals_mimic_demo.json") as f:
        all_vitals = json.load(f)
    
    print("Loading ICU stays...")
    icustays = load_icustays(mimic_path)
    
    # Create stay_id to ICU intime mapping
    stay_to_intime = {}
    for _, row in icustays.iterrows():
        stay_to_intime[row['stay_id']] = row['intime']
    
    print(f"Filtering to {len(SELECTED_PATIENTS)} selected patients...")
    
    # Filter patients
    curated_patients = [p for p in all_patients if p['patient_id'] in SELECTED_PATIENTS]
    
    # Filter and enhance vitals
    curated_vitals = []
    for v in all_vitals:
        if v['patient_id'] in SELECTED_PATIENTS:
            # Calculate ICU stay duration
            stay_id = v.get('stay_id')
            if stay_id and stay_id in stay_to_intime:
                icu_duration = calculate_icu_duration(v['timestamp'], stay_to_intime[stay_id])
            else:
                icu_duration = 0
            
            # Add ICU stay duration
            v['icu_stay_duration'] = icu_duration
            curated_vitals.append(v)
    
    print(f"Selected {len(curated_patients)} patients, {len(curated_vitals)} vital records")
    
    # Write patients
    patients_output = output_dir / "patients_curated.json"
    with open(patients_output, 'w') as f:
        json.dump(curated_patients, f, indent=2)
    print(f"Wrote {patients_output}")
    
    # Write gzipped vitals
    vitals_output = output_dir / "vitals_curated.json.gz"
    with gzip.open(vitals_output, 'wt') as f:
        json.dump(curated_vitals, f)
    print(f"Wrote {vitals_output} ({vitals_output.stat().st_size / 1024:.1f} KB)")
    
    # Summary
    print("\nPatient summary:")
    for p in curated_patients:
        p_vitals = [v for v in curated_vitals if v['patient_id'] == p['patient_id']]
        print(f"  {p['patient_id']}: {len(p_vitals)} records, floor={p.get('floor', '?')[:30]}")


if __name__ == '__main__':
    main()
