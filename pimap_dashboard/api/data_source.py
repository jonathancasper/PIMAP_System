"""Data source abstraction for PIMAP dashboard.

Provides a unified interface for retrieving patient data from different sources:
- Epic FHIR (production)
- MIMIC-IV demo (demonstration)

The active source is controlled by the DATA_SOURCE environment variable.
"""

import os
import json
import gzip
from typing import Dict, List, Any, Optional, Protocol
from pathlib import Path


class DataSource(Protocol):
    """Protocol defining the data source interface."""

    def get_patients(self) -> List[Dict[str, Any]]:
        """Return list of all patients."""
        ...

    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Return a single patient record, or None if not found."""
        ...

    def get_vitals(
        self, patient_id: str, max_records: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return vitals history for a specific patient.

        max_records limits the number of records returned; None means
        no limit (all records, restricted to the newest ICU stay where
        applicable).
        """
        ...

    def get_predictions(self, patient_id: str) -> List[Dict[str, Any]]:
        """Return stored predictions for a specific patient."""
        ...


class EpicDataSource:
    """Data source backed by Epic FHIR API."""
    
    def __init__(self):
        from pimap_epic import EpicAuth, EpicFHIRClient
        self._client = EpicFHIRClient(EpicAuth())
    
    def get_patients(self) -> List[Dict[str, Any]]:
        return self._client.get_all_dashboard_patients()

    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        patients = self._client.get_all_dashboard_patients(patient_ids=[patient_id])
        for patient in patients:
            if str(patient.get("patient_id")) == str(patient_id):
                return patient
        return None

    def get_vitals(
        self, patient_id: str, max_records: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        count = max_records if max_records is not None else 20
        return self._client.get_patient_vitals(patient_id, max_records=count)

    def get_predictions(self, patient_id: str) -> List[Dict[str, Any]]:
        return []


class MimicDataSource:
    """Data source backed by curated MIMIC-IV demo data."""
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = Path(data_path) if data_path else self._default_data_path()
        self._patients: Optional[List[Dict]] = None
        self._vitals: Optional[List[Dict]] = None
        self._vitals_by_patient: Optional[Dict[str, List[Dict]]] = None
        self._predictions: Optional[List[Dict]] = None
        self._predictions_by_patient: Optional[Dict[str, List[Dict]]] = None
    
    def _default_data_path(self) -> Path:
        """Return default path to MIMIC demo data."""
        return Path(__file__).parent.parent.parent / "data" / "mimic_demo"
    
    def _load_patients(self) -> List[Dict[str, Any]]:
        """Load patients from curated JSON file.

        Prefers the date-shifted variant (current-ICU-stay dates, ages
        display correctly for demos); falls back to the original curated
        file.
        """
        if self._patients is None:
            patients_path = self.data_path / "dateshifted_patients_curated.json"
            if not patients_path.exists():
                patients_path = self.data_path / "patients_curated.json"
            with open(patients_path) as f:
                self._patients = json.load(f)
        return self._patients
    
    def _load_vitals(self) -> List[Dict[str, Any]]:
        """Load vitals from curated JSON file.

        Prefers the date-shifted variant (current-ICU-stay dates, ages
        display correctly for demos); falls back to the original curated
        gzipped file.
        """
        if self._vitals is None:
            vitals_path = self.data_path / "dateshifted_vitals_curated.json"
            if not vitals_path.exists():
                vitals_path = self.data_path / "vitals_curated.json.gz"
            if vitals_path.suffix == ".gz":
                with gzip.open(vitals_path, 'rt') as f:
                    self._vitals = json.load(f)
            else:
                with open(vitals_path) as f:
                    self._vitals = json.load(f)
        return self._vitals
    
    def _load_predictions(self) -> List[Dict[str, Any]]:
        """Load predictions from curated JSON file.

        Prefers the date-shifted variant (timestamps aligned with the
        dateshifted vitals); falls back to the original curated file.
        """
        if self._predictions is None:
            pred_path = self.data_path / "dateshifted_predictions_curated.json"
            if not pred_path.exists():
                pred_path = self.data_path / "predictions_curated.json"
            if pred_path.exists():
                with open(pred_path) as f:
                    self._predictions = json.load(f)
            else:
                self._predictions = []
        return self._predictions
    
    def _build_predictions_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build index of predictions by patient ID."""
        if self._predictions_by_patient is None:
            predictions = self._load_predictions()
            self._predictions_by_patient = {}
            for record in predictions:
                patient_id = record.get('patient_id')
                if patient_id not in self._predictions_by_patient:
                    self._predictions_by_patient[patient_id] = []
                self._predictions_by_patient[patient_id].append(record)
        return self._predictions_by_patient
    
    def _build_vitals_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build index of vitals by patient ID."""
        if self._vitals_by_patient is None:
            vitals = self._load_vitals()
            self._vitals_by_patient = {}
            for record in vitals:
                patient_id = record.get('patient_id')
                if patient_id not in self._vitals_by_patient:
                    self._vitals_by_patient[patient_id] = []
                self._vitals_by_patient[patient_id].append(record)
            for patient_id in self._vitals_by_patient:
                self._vitals_by_patient[patient_id].sort(
                    key=lambda r: r.get('timestamp', ''),
                    reverse=True
                )
        return self._vitals_by_patient
    
    def get_patients(self) -> List[Dict[str, Any]]:
        """Return list of all patients."""
        patients = self._load_patients()
        unique_patients = {}
        for p in patients:
            if p['patient_id'] not in unique_patients:
                unique_patients[p['patient_id']] = p
        return list(unique_patients.values())

    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Return a single patient record, or None if not found."""
        for patient in self.get_patients():
            if str(patient.get('patient_id')) == str(patient_id):
                return patient
        return None

    def get_vitals(
        self, patient_id: str, max_records: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return vitals history for a specific patient with field mapping.

        Records are restricted to the newest record's ICU stay (when
        stay_id is present) before slicing to max_records, so a patient
        with multiple stays never mixes timelines or forward-fills
        across a stay boundary. When max_records is None, all records
        from the newest stay are returned.
        """
        index = self._build_vitals_index()
        records = index.get(str(patient_id), [])
        if records:
            newest_stay_id = records[0].get('stay_id')
            if newest_stay_id is not None:
                records = [r for r in records if r.get('stay_id') == newest_stay_id]
        raw_vitals = records if max_records is None else records[:max_records]
        
        mapped_vitals = []
        for v in raw_vitals:
            mapped = self._map_fields(v)
            mapped_vitals.append(mapped)
        
        return mapped_vitals
    
    def get_predictions(self, patient_id: str) -> List[Dict[str, Any]]:
        """Return stored predictions for a specific patient."""
        index = self._build_predictions_index()
        return index.get(str(patient_id), [])
    
    def _map_fields(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Add canonical MIMIC field-name aliases for FeatureExtractor.

        Raw field names are preserved (the XGBoost feature extractor reads
        e.g. 'glucose'/'spo2' directly); canonical aliases
        ('glucose_whole_blood', etc.) are added alongside so the mock
        FeatureExtractor finds its expected names. This matches the
        mapping behavior of scripts/generate_mimic_predictions.py so
        live and stored predictions agree.
        """
        field_aliases = {
            'braden_sensory': 'braden_sensory_perception',
            'spo2': 'arterial_o2_saturation',
            'glucose': 'glucose_whole_blood',
        }

        mapped = dict(record)
        for raw_key, canonical_key in field_aliases.items():
            if raw_key in mapped and canonical_key not in mapped:
                mapped[canonical_key] = mapped[raw_key]
        return mapped


_data_source: Optional[DataSource] = None


def get_data_source() -> DataSource:
    """Factory function returning the configured data source.
    
    Reads DATA_SOURCE environment variable (default: 'epic').
    Valid values: 'epic', 'mimic'
    """
    global _data_source
    
    if _data_source is not None:
        return _data_source
    
    source_name = os.environ.get('DATA_SOURCE', 'epic').lower()
    
    if source_name == 'epic':
        _data_source = EpicDataSource()
    elif source_name == 'mimic':
        mimic_path = os.environ.get('MIMIC_DATA_PATH')
        _data_source = MimicDataSource(mimic_path)
    else:
        raise ValueError(f"Unknown DATA_SOURCE: {source_name}. Use 'epic' or 'mimic'.")
    
    return _data_source


def reset_data_source():
    """Reset the cached data source (useful for testing)."""
    global _data_source
    _data_source = None
