"""Integration example for using XGBoost HAPI predictor.

This example demonstrates:
1. Loading the model with schema
2. Extracting features from patient data
3. Running prediction

Model files should be placed in model_input/ with naming convention:
  - Model: hapi_xgb_v1.json
  - Schema: hapi_xgb_schema_v1.json
"""

import json
from datetime import datetime, timezone
from pathlib import Path

# Import prediction components
from pimap_predict import (
    XGBoostPredictor,
    XGBoostFeatureExtractor,
    load_xgboost_model,
    get_available_models,
)


def example_prediction():
    """Example of running HAPI prediction for a patient."""
    
    # 1. Check available models
    model_dir = Path("model_input")
    available = get_available_models(str(model_dir))
    print(f"Available models: {available}")
    
    # 2. Initialize predictor with model directory
    # This will automatically load model and schema
    predictor = XGBoostPredictor(
        model_dir=str(model_dir),
        model_name="hapi_xgb",
        version="1",
    )
    
    # 3. Prepare patient data (normally from FHIR client or database)
    # This would be extracted from your EHR/FHIR data
    
    # Option A: Pre-computed features (recommended for production)
    # Extract features using XGBoostFeatureExtractor
    schema = load_xgboost_model(str(model_dir), "hapi_xgb", "1")[1]
    feature_names = schema['feature_names']
    
    # Example: Load from demo data
    with open("data/mimic_demo/vitals_mimic_demo.json") as f:
        vitals_data = json.load(f)
    with open("data/mimic_demo/patients_mimic_demo.json") as f:
        patients_data = json.load(f)
    
    patient_id = "10018328"
    patient_demo = next(
        (p for p in patients_data if p['patient_id'] == patient_id), 
        None
    )
    patient_vitals = [
        v for v in vitals_data 
        if v['patient_id'] == patient_id
    ][:50]
    
    # Find latest timestamp for prediction time
    timestamps = []
    for v in patient_vitals:
        ts = v.get('timestamp')
        if ts:
            ts_dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
            timestamps.append(ts_dt)
    
    prediction_time = max(timestamps) if timestamps else datetime.now(timezone.utc)
    
    # Extract features
    extractor = XGBoostFeatureExtractor(
        feature_names, 
        prediction_time=prediction_time
    )
    features = extractor.extract_features(patient_demo, patient_vitals)
    
    # 4. Run prediction
    result = predictor.predict(
        patient_id=patient_id,
        features={
            'demographics': patient_demo,
            'vitals': patient_vitals,
        }
    )
    
    # 5. Access results
    print(f"\nPatient: {result.patient_id}")
    print(f"Risk Level: {result.risk_level}")
    print(f"Risk Score: {result.risk_score:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Model Version: {result.model_version}")
    
    # 6. Access full probability distribution
    probs = result.features_used.get('probabilities', {})
    print("\nClass Probabilities:")
    print(f"  No HAPI: {probs.get('no_hapi', 0):.3f}")
    print(f"  Low-severity HAPI: {probs.get('low_severity_hapi', 0):.3f}")
    print(f"  Advanced-severity HAPI: {probs.get('advanced_severity_hapi', 0):.3f}")
    
    return result


def example_with_mock_predictor():
    """Example using MockPredictor when model files not available."""
    from pimap_predict import MockPredictor
    
    predictor = MockPredictor()
    
    result = predictor.predict(
        patient_id="demo-patient",
        features={
            "braden_sensory_perception": 0.5,
            "braden_moisture": 0.5,
            "braden_activity": 0.5,
            "braden_mobility": 0.5,
            "braden_nutrition": 0.5,
            "braden_friction_shear": 0.5,
        }
    )
    
    print(f"Mock prediction: {result.risk_level}")
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("HAPI XGBoost Predictor - Integration Example")
    print("=" * 60)
    
    # Try with real model, fall back to mock
    try:
        result = example_prediction()
    except Exception as e:
        print(f"\nModel not available ({e})")
        print("Using MockPredictor instead...\n")
        result = example_with_mock_predictor()
