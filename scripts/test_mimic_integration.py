#!/usr/bin/env python3
"""
Test script for MIMIC data integration.

Usage:
    python scripts/test_mimic_integration.py
"""

import sys
import os
import json

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ['DATA_SOURCE'] = 'mimic'

from pimap_dashboard.api.data_source import get_data_source, reset_data_source
from pimap_dashboard.api.patients import get_patients
from pimap_dashboard.api.vitals import get_patient_vitals
from pimap_dashboard.api.predict import predict_pressure_ulcer
from pimap_predict import FeatureExtractor


def test_data_source():
    print("Testing data source factory...")
    reset_data_source()
    ds = get_data_source()
    assert ds is not None
    assert type(ds).__name__ == 'MimicDataSource'
    print("  ✓ Data source created correctly")


def test_patients():
    print("\nTesting patients endpoint...")
    event = {}
    context = {}
    result = get_patients(event, context)
    assert result['statusCode'] == 200
    patients = json.loads(result['body'])
    assert len(patients) > 0
    print(f"  ✓ Returned {len(patients)} patients")


def test_vitals():
    print("\nTesting vitals endpoint...")
    reset_data_source()
    ds = get_data_source()
    patients = ds.get_patients()
    patient_id = patients[0]['patient_id']
    
    event = {'pathParameters': {'patient_id': patient_id}}
    context = {}
    result = get_patient_vitals(event, context)
    assert result['statusCode'] == 200
    vitals = json.loads(result['body'])
    assert len(vitals) > 0
    print(f"  ✓ Returned {len(vitals)} vital records")
    
    return vitals


def test_field_mapping(vitals):
    print("\nTesting field mapping...")
    assert 'icu_stay_duration' in vitals[0]
    assert 'braden_sensory_perception' in vitals[0] or 'braden_sensory' in vitals[0]
    print("  ✓ Field names mapped correctly")


def test_feature_extraction(vitals):
    print("\nTesting feature extraction...")
    extractor = FeatureExtractor()
    
    for v in vitals:
        if v.get('braden_sensory_perception'):
            features = extractor.extract(v)
            assert 'braden_sensory_perception' in features
            assert 'icu_stay_duration' in features
            print(f"  ✓ Feature extraction successful")
            return
    
    print("  ⚠ No records with Braden scores found")


def test_prediction():
    print("\nTesting prediction endpoint...")
    reset_data_source()
    ds = get_data_source()
    patients = ds.get_patients()
    patient_id = patients[0]['patient_id']
    
    event = {'pathParameters': {'patient_id': patient_id}}
    context = {}
    result = predict_pressure_ulcer(event, context)
    assert result['statusCode'] == 200
    prediction = json.loads(result['body'])
    assert 'risk_level' in prediction
    assert 'prediction_score' in prediction
    assert prediction['data_source'] == 'mimic_iv'
    print(f"  ✓ Prediction: {prediction['risk_level']} ({prediction['prediction_score']:.2f})")


def main():
    print("=" * 60)
    print("MIMIC Data Integration Test")
    print("=" * 60)
    
    try:
        test_data_source()
        test_patients()
        vitals = test_vitals()
        test_field_mapping(vitals)
        test_feature_extraction(vitals)
        test_prediction()
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
