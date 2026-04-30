"""FHIR R4 client for Epic, fetching patient data and observations for the PIMAP dashboard.

Transforms FHIR resources into the flat record format the dashboard expects.
Observations that cannot be fetched (missing data or unsupported codes) are
imputed with clinically reasonable defaults and flagged in _imputed_fields.
"""

import requests
from datetime import datetime, timedelta, timezone

from .auth import EpicAuth


DEFAULT_FHIR_BASE = "https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4"

LOINC_BP_PANEL = "85354-9"
LOINC_SYSTOLIC = "8480-6"
LOINC_DIASTOLIC = "8462-4"
LOINC_BODY_WEIGHT = "29463-7"
LOINC_O2_SAT = "2708-6"
LOINC_HEIGHT = "8302-2"
LOINC_GLUCOSE = "2345-7"
LOINC_ALBUMIN = "1751-7"
LOINC_TOTAL_BILIRUBIN = "1975-2"
LOINC_TOTAL_PROTEIN = "2885-2"

SANDBOX_PATIENT_IDS = [
    "erXuFYUfucBZaryVksYEcMg3",
    "eq081-VQEgP8drUUqCWzHfw3",
    "e63wRTbPfr1p8UW81d8Seiw3",
    "eAB3mDIBBcyUKviyzrxsnAw3",
    "eIXesllypH3M9tAA5WdJftQ3",
    "e0w0LEDCYtfckT6N.CkJKCw3",
    "eTjDDWfopD0BnRlyEO2mGZQ3",
    "e3fr4nj0o2ClexQf3ERo3rA3",
    "enh2Q1c0oNRtWzXArnG4tKw3",
    "eBSae5lBhhsLxb.2HF7P2ng3",
    "e3ioqdpm4AN6AdB7FsIT7aA3",
    "eMKXfzrACilPfpKlx65qSxQ3",
    "em6D.pvHknAMENeHgzrlW.A3",
    "eYUL4WwHBjAQEv6YcK29oqA3",
    "evTMIl9S3BwJBhHmqWOWjAg3",
]

_IMPUTE_DEFAULTS = {
    "daily_weight": 70.0,
    "arterial_o2_saturation": 96,
    "glucose_whole_blood": 110,
    "total_bilirubin": 1.0,
    "albumin": 3.5,
    "total_protein": 6.5,
    "braden_sensory_perception": 3,
    "braden_moisture": 3,
    "braden_activity": 3,
    "braden_mobility": 3,
    "braden_nutrition": 3,
    "braden_friction_shear": 2,
    "icu_stay_duration": 3,
}

_ALWAYS_IMPUTED = [
    "braden_sensory_perception",
    "braden_moisture",
    "braden_activity",
    "braden_mobility",
    "braden_nutrition",
    "braden_friction_shear",
    "icu_stay_duration",
]


class FHIRRequestError(Exception):
    """Raised when a FHIR API request fails."""


class EpicFHIRClient:
    """Authenticated client for Epic's FHIR R4 API with PIMAP data helpers."""

    def __init__(self, auth, base_url=DEFAULT_FHIR_BASE):
        self.auth = auth
        self.base_url = base_url.rstrip("/")

    def get(self, path, params=None):
        """Authenticated GET against the FHIR base URL."""
        url = f"{self.base_url}/{path}"
        token = self.auth.get_access_token()
        resp = requests.get(
            url,
            params=params,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/fhir+json",
            },
        )
        if resp.status_code != 200:
            raise FHIRRequestError(
                f"GET {url} failed ({resp.status_code}): {resp.text}"
            )
        return resp.json()

    def search_patients(self, patient_ids=None, count=50):
        """Search for Patient resources by FHIR ID list."""
        params = {"_count": count}
        if patient_ids:
            params["_id"] = ",".join(patient_ids)
        bundle = self.get("Patient", params=params)
        return [
            e
            for e in bundle.get("entry", [])
            if e.get("resource", {}).get("resourceType") == "Patient"
        ]

    def get_patient(self, patient_id):
        return self.get(f"Patient/{patient_id}")

    def get_all_dashboard_patients(self, patient_ids=None):
        """Fetch all sandbox patients and transform to dashboard format."""
        ids = patient_ids or SANDBOX_PATIENT_IDS
        entries = self.search_patients(patient_ids=ids, count=len(ids) + 10)
        return [self._transform_patient(e["resource"]) for e in entries]

    def _transform_patient(self, resource):
        """Transform a FHIR Patient resource to the dashboard's expected format."""
        names = resource.get("name", [{}])
        n = names[0] if names else {}
        given = n.get("given", [])
        first_name = given[0] if given else ""
        last_name = n.get("family", "")

        gender_map = {"male": "M", "female": "F", "other": "O", "unknown": "U"}
        gender = gender_map.get(resource.get("gender", "unknown"), "U")

        ethnicity = ""
        race = ""
        for ext in resource.get("extension", []):
            url = ext.get("url", "")
            if "us-core-ethnicity" in url:
                for sub in ext.get("extension", []):
                    if sub.get("url") == "text":
                        ethnicity = sub.get("valueString", "")
            elif "us-core-race" in url:
                for sub in ext.get("extension", []):
                    if sub.get("url") == "text":
                        race = sub.get("valueString", "")

        if ethnicity and ethnicity not in ("Unknown", "Not Hispanic or Latino"):
            display_ethnicity = ethnicity
        elif race:
            display_ethnicity = race
        else:
            display_ethnicity = "Unknown"

        return {
            "patient_id": resource.get("id", ""),
            "first_name": first_name,
            "last_name": last_name,
            "dob": resource.get("birthDate", ""),
            "gender": gender,
            "ethnicity": display_ethnicity,
            "admission_time": "",
            "height": 0,
            "height_ft": 0,
            "height_in": 0,
            "floor": 0,
            "room": "",
            "fracture_diagnosis": False,
            "diarrhea_diagnosis": False,
            "spinal_cord_injury_diagnosis": False,
            "data_source": "epic_fhir",
        }

    def _get_observations(self, patient_id, code=None, last_n_days=None, count=100):
        """Fetch Observation resources, optionally filtered by LOINC code and date."""
        params = {"patient": patient_id, "_count": count}
        if code:
            params["code"] = code
        if last_n_days is not None:
            since = datetime.now(timezone.utc) - timedelta(days=last_n_days)
            params["date"] = f"ge{since.strftime('%Y-%m-%d')}"
        try:
            bundle = self.get("Observation", params=params)
        except FHIRRequestError:
            return []
        return [
            e
            for e in bundle.get("entry", [])
            if e.get("resource", {}).get("resourceType") == "Observation"
        ]

    def get_blood_pressure(self, patient_id, last_n_days=None, count=100):
        return self._get_observations(patient_id, LOINC_BP_PANEL, last_n_days, count)

    def get_body_weight(self, patient_id, last_n_days=None, count=100):
        return self._get_observations(patient_id, LOINC_BODY_WEIGHT, last_n_days, count)

    def get_o2_saturation(self, patient_id, last_n_days=None, count=100):
        return self._get_observations(patient_id, LOINC_O2_SAT, last_n_days, count)

    def get_glucose(self, patient_id, last_n_days=None, count=100):
        return self._get_observations(patient_id, LOINC_GLUCOSE, last_n_days, count)

    def get_albumin(self, patient_id, last_n_days=None, count=100):
        return self._get_observations(patient_id, LOINC_ALBUMIN, last_n_days, count)

    def get_total_bilirubin(self, patient_id, last_n_days=None, count=100):
        return self._get_observations(
            patient_id, LOINC_TOTAL_BILIRUBIN, last_n_days, count
        )

    def get_total_protein(self, patient_id, last_n_days=None, count=100):
        return self._get_observations(
            patient_id, LOINC_TOTAL_PROTEIN, last_n_days, count
        )

    def get_patient_vitals(self, patient_id, max_records=20):
        """Fetch all available vitals from Epic, combine into dashboard-format records.

        Uses BP observations as the time spine (one record per BP reading).
        Other observation types are matched to the nearest BP timestamp.
        Missing values are imputed with defaults and flagged in _imputed_fields.
        """
        bp_entries = self.get_blood_pressure(
            patient_id, last_n_days=None, count=max_records
        )
        weight_entries = self.get_body_weight(patient_id, last_n_days=None, count=100)
        o2_entries = self.get_o2_saturation(patient_id, last_n_days=None, count=100)
        glucose_entries = self.get_glucose(patient_id, last_n_days=None, count=100)
        albumin_entries = self.get_albumin(patient_id, last_n_days=None, count=100)
        bilirubin_entries = self.get_total_bilirubin(
            patient_id, last_n_days=None, count=100
        )
        protein_entries = self.get_total_protein(
            patient_id, last_n_days=None, count=100
        )

        weight_map = self._build_value_map(weight_entries)
        o2_map = self._build_value_map(o2_entries)
        glucose_map = self._build_value_map(glucose_entries)
        albumin_map = self._build_value_map(albumin_entries)
        bilirubin_map = self._build_value_map(bilirubin_entries)
        protein_map = self._build_value_map(protein_entries)

        vitals_records = []
        for bp_entry in bp_entries:
            obs = bp_entry.get("resource", {})
            timestamp = obs.get("effectiveDateTime", "")

            systolic = None
            for comp in obs.get("component", []):
                codings = comp.get("code", {}).get("coding", [])
                code = codings[0].get("code", "") if codings else ""
                if code == LOINC_SYSTOLIC:
                    systolic = comp.get("valueQuantity", {}).get("value")
            if systolic is None:
                continue

            weight = self._find_nearest(weight_map, timestamp)
            o2_sat = self._find_nearest(o2_map, timestamp)
            glucose = self._find_nearest(glucose_map, timestamp)
            albumin = self._find_nearest(albumin_map, timestamp)
            bilirubin = self._find_nearest(bilirubin_map, timestamp)
            protein = self._find_nearest(protein_map, timestamp)

            imputed = list(_ALWAYS_IMPUTED)
            fetched_vals = {
                "daily_weight": weight,
                "arterial_o2_saturation": o2_sat,
                "glucose_whole_blood": glucose,
                "albumin": albumin,
                "total_bilirubin": bilirubin,
                "total_protein": protein,
            }
            for field, val in fetched_vals.items():
                if val is None:
                    imputed.append(field)

            record = {
                "patient_id": patient_id,
                "timestamp": timestamp,
                "arterial_blood_pressure_systolic": systolic,
                "daily_weight": weight
                if weight is not None
                else _IMPUTE_DEFAULTS["daily_weight"],
                "arterial_o2_saturation": o2_sat
                if o2_sat is not None
                else _IMPUTE_DEFAULTS["arterial_o2_saturation"],
                "glucose_whole_blood": glucose
                if glucose is not None
                else _IMPUTE_DEFAULTS["glucose_whole_blood"],
                "total_bilirubin": bilirubin
                if bilirubin is not None
                else _IMPUTE_DEFAULTS["total_bilirubin"],
                "albumin": albumin
                if albumin is not None
                else _IMPUTE_DEFAULTS["albumin"],
                "total_protein": protein
                if protein is not None
                else _IMPUTE_DEFAULTS["total_protein"],
                "braden_sensory_perception": _IMPUTE_DEFAULTS[
                    "braden_sensory_perception"
                ],
                "braden_moisture": _IMPUTE_DEFAULTS["braden_moisture"],
                "braden_activity": _IMPUTE_DEFAULTS["braden_activity"],
                "braden_mobility": _IMPUTE_DEFAULTS["braden_mobility"],
                "braden_nutrition": _IMPUTE_DEFAULTS["braden_nutrition"],
                "braden_friction_shear": _IMPUTE_DEFAULTS["braden_friction_shear"],
                "icu_stay_duration": _IMPUTE_DEFAULTS["icu_stay_duration"],
                "pu_prediction_score": None,
                "actions_taken": "",
                "last_predicted_by_model": "",
                "pu": "",
                "_imputed_fields": imputed,
                "_data_source": "epic_fhir",
            }
            vitals_records.append(record)

        vitals_records.sort(key=lambda r: r["timestamp"], reverse=True)
        return vitals_records[:max_records]

    def _build_value_map(self, entries):
        """Build a {timestamp_str: numeric_value} map from Observation entries."""
        result = {}
        for entry in entries:
            obs = entry.get("resource", {})
            ts = obs.get("effectiveDateTime", "")
            val = obs.get("valueQuantity", {}).get("value")
            if ts and val is not None:
                result[ts] = val
        return result

    def _find_nearest(self, value_map, target_ts):
        """Find the value whose timestamp is closest to target_ts."""
        if not value_map:
            return None
        try:
            target_dt = _parse_fhir_datetime(target_ts)
        except (ValueError, TypeError):
            return None

        best_val = None
        best_delta = None
        for ts, val in value_map.items():
            try:
                dt = _parse_fhir_datetime(ts)
                delta = abs(dt - target_dt)
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_val = val
            except (ValueError, TypeError):
                continue
        return best_val


def _parse_fhir_datetime(ts):
    """Parse a FHIR datetime string into a timezone-aware datetime."""
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)
