PIMAP (Pervasive Injury Monitoring And Prevention)

License:
Author: Sam Mansfield

PIMAP is a system framework for pressure ulcer risk prediction consisting of four 
components: Sense, Store, Analyze, and Visualize. The system integrates with Epic 
EHR systems via FHIR R4 APIs to fetch patient data and write risk assessments.

================================================================================
DIRECTORY STRUCTURE
================================================================================

pimap/
  Core sensor pipeline components for pressure bandage data processing.
  
  pimapsensetcp.py        - TCP sensor input
  pimapsenseudp.py        - UDP sensor input
  pimapsensesentinel.py   - Sentinel file watcher
  pimapstorekafka.py      - Kafka storage backend
  pimapanalyzeobjectivemobility.py  - Mobility analysis from pressure data
  pimapanalyzeheatmap.py  - Heatmap analysis
  pimapvisualizepltgraph.py  - Matplotlib visualization
  pimapvisualizeheatmap.py   - Heatmap visualization
  pimaputilities.py       - Shared utilities for PIMAP samples/metrics

pimap_epic/
  Epic EHR integration via FHIR R4.
  
  auth.py                 - OAuth 2.0 backend system authentication
  fhir_client.py          - Read patients and observations from Epic
  fhir_writer.py          - Write risk assessments to Epic (stub)

pimap_predict/
  ML-based pressure ulcer risk prediction.
  
  predictor.py            - XGBoost and Mock (Braden) predictors
  feature_extractor.py    - Transform FHIR data to model features

pimap_dashboard/
  Demo dashboard for visualization.
  
  api/
    patients.py           - GET /patients Lambda handler
    vitals.py             - GET /patients/{id}/vitals Lambda handler
    actions.py            - POST /patients/{id}/actions Lambda handler
    predict.py            - POST /patients/{id}/predict Lambda handler
  frontend/
    index.html            - Dashboard HTML

deployment/
  Deployment configurations.
  
  aws/
    template.yaml         - AWS SAM template for Lambda deployment

tests/
  Unit tests for all modules.
  
  testpimaputilities.py
  testpimapsensetcp.py
  testpimapsenseudp.py
  testpimapstorekafka.py
  testpimapanalyzeobjectivemobility.py
  testpimapvisualizepltgraph.py
  testpimapepic.py        - Tests for Epic integration
  testpimappredict.py     - Tests for prediction module

examples/
  Example scripts demonstrating PIMAP usage.

requirements.txt
  Python dependencies. Install with:
    pip install -r requirements.txt

================================================================================
PREREQUISITES
================================================================================

1. Python 3.9+
   - Install from python.org or use pyenv

2. AWS CLI (for deployment)
   - Install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
   - Configure: aws configure

3. SAM CLI (for AWS Lambda deployment)
   - Install: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html

4. Epic FHIR credentials
   - Client ID: Set EPIC_CLIENT_ID environment variable
   - Private key: Set EPIC_PRIVATE_KEY or EPIC_PRIVATE_KEY_PATH

================================================================================
LOCAL DEVELOPMENT (Demo Sandbox)
================================================================================

The demo uses Epic's public FHIR sandbox with synthetic patient data.

1. Install dependencies:
   pip install -r requirements.txt

2. Set up Epic credentials:
   export EPIC_CLIENT_ID="your-client-id"
   export EPIC_PRIVATE_KEY_PATH="/path/to/privatekey.pem"
   
   Or store client ID in:
   ~/.config/epic_fhir/client_id

3. Run tests:
   PYTHONPATH=. python3 tests/testpimapepic.py
   PYTHONPATH=. python3 tests/testpimappredict.py

4. Start local API (requires SAM CLI):
   cd deployment/aws
   sam build
   sam local start-api

5. Open dashboard:
   Open pimap_dashboard/frontend/index.html in a browser
   (Update API_URL in the HTML to point to your local endpoint)

================================================================================
AWS DEPLOYMENT
================================================================================

1. Build and deploy:
   cd deployment/aws
   sam build
   sam deploy --guided

2. The guided deployment will prompt for:
   - Stack name: pimap-dashboard
   - AWS Region: us-west-2 (or your preferred region)
   - EpicClientId: Your Epic FHIR client ID
   - Confirm changes: Y
   - Allow SAM CLI IAM role creation: Y

3. After deployment, note the API Gateway URL from outputs.

4. Deploy frontend:
   - Upload pimap_dashboard/frontend/index.html to S3
   - Or serve from any static host
   - Update API_URL in the HTML to your API Gateway URL

================================================================================
ENVIRONMENT VARIABLES
================================================================================

EPIC_CLIENT_ID
  Epic FHIR OAuth client ID (required for Epic integration)

EPIC_PRIVATE_KEY
  Private key content as PEM string (for Lambda deployment)

EPIC_PRIVATE_KEY_PATH
  Path to private key file (alternative to EPIC_PRIVATE_KEY)

XGBOOST_MODEL_PATH
  Path to trained XGBoost model file (optional)
  If not set, uses MockPredictor with Braden heuristic

VITALS_TABLE
  DynamoDB table name for staff actions (default: patient_vitals)

================================================================================
API ENDPOINTS
================================================================================

GET /patients
  Returns list of patients from Epic FHIR sandbox

GET /patients/{patient_id}/vitals
  Returns vitals history for a patient

POST /patients/{patient_id}/predict
  Generate pressure ulcer risk prediction
  
  Response:
  {
    "patient_id": "string",
    "prediction_score": 0.75,
    "risk_level": "High|Moderate|Low",
    "confidence": 0.8,
    "timestamp": "2024-01-15T10:00:00Z",
    "model_version": "string",
    "imputed_fields": ["albumin", "icu_stay_duration"]
  }

POST /patients/{patient_id}/actions
  Record a staff action for a patient
  
  Body:
  {
    "action_taken": "Repositioned patient",
    "staff_id": "nurse-123"
  }

================================================================================
EPIC SANDBOX PATIENTS
================================================================================

The demo uses Epic's public sandbox with these test patients:

- erXuFYUfucBZaryVksYEcMg3 (Camila Maria Lopez)
- eq081-VQEgP8drUUqCWzHfw3 (Derrick Lin)
- e63wRTbPfr1p8UW81d8Seiw3 (Theodore Mychart)
- eAB3mDIBBcyUKviyzrxsnAw3 (Desiree Caroline Powell)
- And more...

See pimap_epic/fhir_client.py for the full list.

================================================================================
PREDICTION MODELS
================================================================================

MockPredictor (default)
  Uses Braden Scale heuristic for risk assessment.
  No training required. Suitable for demos and testing.
  
  Risk levels:
  - High: Braden total <= 12 (90% risk)
  - Moderate: Braden total 13-16 (60% risk)
  - Low: Braden total > 16 (20% risk)

XGBoostPredictor
  Uses externally trained XGBoost model.
  Requires model file and xgboost package.
  
  To use:
  1. Train model and save to file
  2. Set XGBOOST_MODEL_PATH environment variable
  3. Deploy or restart application

================================================================================
FHIR WRITE-BACK (FUTURE)
================================================================================

The fhir_writer.py module contains stubs for writing risk assessments back to 
Epic. Options documented:

1. Observation - Recommended for risk scores (triggers BPAs)
2. ClinicalImpression - For detailed clinical assessments
3. DocumentReference - For narrative clinical notes

Implementation requires:
- Write-enabled OAuth scope (system/Observation.write)
- Epic production or sandbox with write access
- Testing with real or synthetic patient data

================================================================================
LICENSE
================================================================================

See LICENSE file for details.
