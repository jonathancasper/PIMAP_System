"""Feature extraction module for pressure ulcer risk prediction.

Transforms FHIR patient/vitals data into feature vectors suitable for
the ML prediction model.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    icu_duration_max_days: float = 30.0
    weight_max_kg: float = 200.0
    bp_systolic_max: float = 200.0
    o2_sat_max: float = 100.0
    glucose_max: float = 300.0
    albumin_max: float = 5.0
    bilirubin_max: float = 5.0
    protein_max: float = 10.0


FEATURE_NAMES = [
    "icu_stay_duration",
    "braden_friction_shear",
    "braden_mobility",
    "braden_moisture",
    "braden_sensory_perception",
    "braden_nutrition",
    "braden_activity",
    "arterial_o2_saturation",
    "arterial_blood_pressure_systolic",
    "glucose_whole_blood",
    "albumin",
    "total_bilirubin",
    "total_protein",
    "daily_weight",
]


class FeatureExtractor:
    """Extracts and normalizes features from patient vitals data.

    Takes the flat record format from EpicFHIRClient and produces
    a normalized feature vector for the ML model.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize the feature extractor.

        Args:
            config: Optional configuration for normalization bounds
        """
        self.config = config or FeatureConfig()

    def extract(self, vitals_record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from a single vitals record.

        Args:
            vitals_record: Flat record from EpicFHIRClient.get_patient_vitals()

        Returns:
            Dictionary with normalized features and metadata
        """
        features = {
            "icu_stay_duration": self._normalize(
                vitals_record.get("icu_stay_duration", 3),
                self.config.icu_duration_max_days,
            ),
            "braden_friction_shear": self._normalize(
                vitals_record.get("braden_friction_shear", 2), 3.0
            ),
            "braden_mobility": self._normalize(
                vitals_record.get("braden_mobility", 2), 4.0
            ),
            "braden_moisture": self._normalize(
                vitals_record.get("braden_moisture", 3), 4.0
            ),
            "braden_sensory_perception": self._normalize(
                vitals_record.get("braden_sensory_perception", 3), 4.0
            ),
            "braden_nutrition": self._normalize(
                vitals_record.get("braden_nutrition", 3), 4.0
            ),
            "braden_activity": self._normalize(
                vitals_record.get("braden_activity", 2), 4.0
            ),
            "arterial_o2_saturation": self._normalize(
                vitals_record.get("arterial_o2_saturation", 95), self.config.o2_sat_max
            ),
            "arterial_blood_pressure_systolic": self._normalize(
                vitals_record.get("arterial_blood_pressure_systolic", 120),
                self.config.bp_systolic_max,
            ),
            "glucose_whole_blood": self._normalize(
                vitals_record.get("glucose_whole_blood", 110), self.config.glucose_max
            ),
            "albumin": self._normalize(
                vitals_record.get("albumin", 3.5), self.config.albumin_max
            ),
            "total_bilirubin": self._normalize(
                vitals_record.get("total_bilirubin", 1.0), self.config.bilirubin_max
            ),
            "total_protein": self._normalize(
                vitals_record.get("total_protein", 6.0), self.config.protein_max
            ),
            "daily_weight": self._normalize(
                vitals_record.get("daily_weight", 70), self.config.weight_max_kg
            ),
        }

        features["_imputed_fields"] = vitals_record.get("_imputed_fields", [])
        features["_data_source"] = vitals_record.get("_data_source", "unknown")
        features["_patient_id"] = vitals_record.get("patient_id", "")
        features["_timestamp"] = vitals_record.get("timestamp", "")

        return features

    def extract_batch(
        self, vitals_records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract features from multiple vitals records.

        Args:
            vitals_records: List of flat records from EpicFHIRClient

        Returns:
            List of feature dictionaries
        """
        return [self.extract(record) for record in vitals_records]

    def to_vector(self, features: Dict[str, Any]) -> List[float]:
        """Convert feature dict to ordered list for model input.

        Args:
            features: Feature dictionary from extract()

        Returns:
            Ordered list of feature values
        """
        return [features.get(name, 0.0) for name in FEATURE_NAMES]

    def _normalize(self, value: Any, max_value: float) -> float:
        """Normalize a value to [0, 1] range.

        Args:
            value: Raw value (may be string or numeric)
            max_value: Maximum value for normalization

        Returns:
            Normalized value in [0, 1] range
        """
        try:
            if isinstance(value, str):
                if "day" in value.lower():
                    value = float(value.split()[0])
                else:
                    value = float(value)
            return min(1.0, max(0.0, float(value) / max_value))
        except (ValueError, TypeError):
            return 0.5
