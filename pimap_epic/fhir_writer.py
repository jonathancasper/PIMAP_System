"""FHIR R4 write-back module for Epic EHR systems.

This module provides functionality to write risk assessments and clinical notes
back to Epic via FHIR R4 APIs.

IMPORTANT: This module is currently a STUB. The actual write functionality requires:
1. Write-enabled OAuth scopes (system/Observation.write or similar)
2. Epic sandbox or production environment with write access
3. Testing with real or synthetic patient data

EPIC WRITE-BACK OPTIONS
=======================

When writing risk assessments to Epic, there are several FHIR resource types to consider:

1. OBSERVATION (RECOMMENDED for risk scores)
   - Standard FHIR resource for clinical measurements and assessments
   - Can be queried by other systems and used in clinical decision support
   - Epic supports Observation.create for custom assessments
   - Can include reference ranges, interpretation (high/low), and status
   - Example: A pressure ulcer risk score as a numeric observation

   Advantages:
   - Structured data that can trigger Epic BestPractice Alerts (BPAs)
   - Queryable and can be used in flowsheets
   - Standard FHIR approach, most interoperable

2. CLINICALIMPRESSION
   - Designed specifically for clinical assessments and risk predictions
   - Can capture investigation results, summary, and prognosis
   - More expressive for clinical reasoning
   - Less commonly used in Epic integrations

3. DOCUMENTREFERENCE (Clinical Notes)
   - Epic's DocumentReference.Create (Clinical Notes) R4 interface
   - Good for narrative clinical notes and reports
   - Can include structured content via attachments
   - More flexible but less structured

4. CONDITION
   - For documenting diagnosed conditions (not predictions)
   - Not appropriate for risk scores

5. HL7v2 INCOMING MDM (TRANSCRIPTIONS) INTERFACE
   - Legacy HL7v2 approach, more flexible
   - Requires MDM interface setup with the hospital
   - Not FHIR-based, requires separate infrastructure

RECOMMENDATION
--------------

For pressure ulcer risk assessments:
1. Use Observation for the numeric risk score (enables BPAs and queries)
2. Optionally write a DocumentReference for detailed clinical notes

The Observation approach integrates best with Epic's clinical workflows and
allows hospitals to set up triggers/alerts based on risk thresholds.

REFERENCES
----------
- Epic FHIR R4 Documentation: https://fhir.epic.com/
- FHIR R4 Observation: https://www.hl7.org/fhir/observation.html
- FHIR R4 ClinicalImpression: https://www.hl7.org/fhir/clinicalimpression.html
- FHIR R4 DocumentReference: https://www.hl7.org/fhir/documentreference.html
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any


class FHIRWriteError(Exception):
    """Raised when a FHIR write operation fails."""


class EpicFHIRWriter:
    """Writes risk assessments and clinical notes to Epic via FHIR R4.

    This class provides methods to create FHIR resources for risk assessments.
    Currently stubbed - requires write-enabled OAuth scope to function.
    """

    def __init__(self, fhir_client):
        """Initialize the writer with an authenticated FHIR client.

        Args:
            fhir_client: An EpicFHIRClient instance with write-enabled auth
        """
        self.client = fhir_client

    def write_risk_observation(
        self,
        patient_id: str,
        risk_score: float,
        risk_level: str,
        model_version: str,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Write a risk score as a FHIR Observation.

        STUB: This method is not yet implemented. Requires:
        - Write-enabled OAuth scope (system/Observation.write)
        - Epic environment with write access

        Args:
            patient_id: Epic FHIR patient ID
            risk_score: Numeric risk score (0.0 to 1.0)
            risk_level: Risk category ("Low", "Moderate", "High")
            model_version: Version identifier for the prediction model
            timestamp: Optional timestamp (defaults to now)

        Returns:
            Dict with created observation ID and status

        Raises:
            FHIRWriteError: If the write operation fails
        """
        raise FHIRWriteError(
            "write_risk_observation is not yet implemented. "
            "Requires write-enabled OAuth scope and Epic write access. "
            "See module docstring for implementation notes."
        )

    def write_clinical_note(
        self,
        patient_id: str,
        note_text: str,
        note_type: str = "risk_assessment",
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Write a clinical note as a FHIR DocumentReference.

        STUB: This method is not yet implemented. Requires:
        - Write-enabled OAuth scope (system/DocumentReference.write)
        - Epic environment with write access

        Args:
            patient_id: Epic FHIR patient ID
            note_text: Clinical note content
            note_type: Type of note (for categorization)
            timestamp: Optional timestamp (defaults to now)

        Returns:
            Dict with created document ID and status

        Raises:
            FHIRWriteError: If the write operation fails
        """
        raise FHIRWriteError(
            "write_clinical_note is not yet implemented. "
            "Requires write-enabled OAuth scope and Epic write access. "
            "See module docstring for implementation notes."
        )

    def _build_observation_resource(
        self,
        patient_id: str,
        risk_score: float,
        risk_level: str,
        model_version: str,
        timestamp: datetime,
    ) -> Dict[str, Any]:
        """Build a FHIR Observation resource for a risk assessment.

        This is a helper method that constructs the FHIR resource structure.
        It can be used for testing and validation before write is enabled.

        Args:
            patient_id: Epic FHIR patient ID
            risk_score: Numeric risk score (0.0 to 1.0)
            risk_level: Risk category ("Low", "Moderate", "High")
            model_version: Version identifier for the prediction model
            timestamp: Observation timestamp

        Returns:
            FHIR Observation resource as a dict
        """
        interpretation_map = {
            "High": "H",
            "Moderate": "A",
            "Low": "N",
        }

        observation = {
            "resourceType": "Observation",
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "survey",
                            "display": "Survey",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "XXXXX-X",
                        "display": "Pressure ulcer risk score",
                    }
                ],
                "text": "Pressure Ulcer Risk Assessment",
            },
            "subject": {
                "reference": f"Patient/{patient_id}",
            },
            "effectiveDateTime": timestamp.isoformat(),
            "issued": datetime.now(timezone.utc).isoformat(),
            "valueQuantity": {
                "value": risk_score,
                "unit": "risk probability",
                "system": "http://unitsofmeasure.org",
                "code": "{probability}",
            },
            "interpretation": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                            "code": interpretation_map.get(risk_level, "N"),
                            "display": f"{risk_level} Risk",
                        }
                    ]
                }
            ],
            "note": [
                {
                    "text": f"Pressure ulcer risk prediction generated by PIMAP model {model_version}",
                }
            ],
        }

        return observation

    def _build_document_reference_resource(
        self,
        patient_id: str,
        note_text: str,
        note_type: str,
        timestamp: datetime,
    ) -> Dict[str, Any]:
        """Build a FHIR DocumentReference resource for a clinical note.

        This is a helper method that constructs the FHIR resource structure.
        It can be used for testing and validation before write is enabled.

        Args:
            patient_id: Epic FHIR patient ID
            note_text: Clinical note content
            note_type: Type of note (for categorization)
            timestamp: Document timestamp

        Returns:
            FHIR DocumentReference resource as a dict
        """
        doc_ref = {
            "resourceType": "DocumentReference",
            "status": "current",
            "type": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "11488-4",
                        "display": "Consult note",
                    }
                ],
                "text": "Pressure Ulcer Risk Assessment Note",
            },
            "subject": {
                "reference": f"Patient/{patient_id}",
            },
            "date": timestamp.isoformat(),
            "content": [
                {
                    "attachment": {
                        "contentType": "text/plain",
                        "data": note_text.encode("utf-8").decode("utf-8"),
                    }
                }
            ],
        }

        return doc_ref
