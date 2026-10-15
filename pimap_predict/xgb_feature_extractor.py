"""Feature extraction for XGBoost HAPI prediction model.

Transforms patient vitals data into the 415-feature format required
by the trained XGBoost model, with temporal aggregation across
multiple time windows.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import math


TIME_WINDOWS = {
    'DayAgo': timedelta(hours=24),
    'TwoDaysAgo': timedelta(hours=48),
    'HalfWeekAgo': timedelta(hours=84),
    'WeekAgo': timedelta(hours=168),
    'TwoWeeksAgo': timedelta(hours=336),
    'YearAgo': timedelta(days=365),
}

FIO2_FIELDS = {'FiO2', 'fio2'}

TWA_SOURCE_FIELDS = {
    'FiO2': 'fio2',
    'SpO2': 'spo2',
    'weight': 'daily_weight',
}

ETHNICITY_ONEHOT = {
    'eth_White': ['WHITE', 'WHITE - BRAZILIAN', 'WHITE - RUSSIAN', 'WHITE - OTHER'],
    'eth_Black/African American': ['BLACK/AFRICAN AMERICAN', 'BLACK/CAPE VERDEAN', 'BLACK/HAITIAN'],
    'eth_Asian': ['ASIAN', 'ASIAN - CHINESE', 'ASIAN - KOREAN', 'ASIAN - SOUTH EAST ASIAN'],
    'eth_Native American/Alaska Native': ['AMERICAN INDIAN/ALASKA NATIVE'],
    'eth_Pacific Islander': ['NATIVE HAWAIIAN/PACIFIC ISLANDER'],
    'eth_Other / Unknown': ['OTHER', 'UNKNOWN', 'UNABLE TO OBTAIN', 'PATIENT DECLINED TO ANSWER'],
}


LAB_FIELD_MAPPING = {
    'creatinine': 'Creatinine',
    'bun': 'BUN',
    'egfr': 'eGFR',
    'hemoglobin': 'Hemoglobin',
    'hematocrit': 'Hematocrit',
    'wbc': 'WBC',
    'sodium': 'Sodium',
    'potassium': 'Potassium',
    'chloride': 'Chloride',
    'magnesium': 'Magnesium',
    'phosphorus': 'Phosphorus',
    'calcium': 'Calcium_Total',
    'glucose': 'Glucose',
    'albumin': 'Albumin_Serum',
    'ast': 'AST',
    'bicarbonate': 'Bicarbonate',
    'inr': 'INR',
    'pt': 'PT',
    'lactate': 'Lactate',
    'total_bilirubin': 'Bilirubin_Total',
}


VITAL_FIELD_MAPPING = {
    'arterial_blood_pressure_systolic': 'SBP',
    'arterial_blood_pressure_diastolic': 'DBP',
    'arterial_blood_pressure_mean': 'MAP',
    'heart_rate': 'HR',
    'temperature_f': 'Temp',
    'fio2': 'FiO2',
    'spo2': 'SpO2',
    'pao2': 'PaO2',
    'pao2_fio2_ratio': 'PaO2/FiO2 Ratio',
}


DIAGNOSIS_FIELD_MAPPING = {
    'stool_incontinent': 'fecal incontinence',
    'bed_confinement_diagnosis': 'Bed_or_Wheelchair_Dx',
    'paralysis_diagnosis': 'Paralysis_Dx',
    'gait_abnormality_diagnosis': 'Gait_Limitation_Dx',
    'fracture_diagnosis': 'Fracture_Dx',
}


def _normalize_fio2(value: float) -> float:
    """Normalize FiO2 to fraction (0-1) if given as percent (0-100)."""
    if value > 1.0:
        return value / 100.0
    return value


class XGBoostFeatureExtractor:
    """Extract features for HAPI XGBoost model from patient vitals data.
    
    Performs temporal aggregation across time windows to produce
    the 415 features required by the model.
    """
    
    def __init__(self, feature_names: List[str], prediction_time: Optional[datetime] = None):
        self.feature_names = feature_names
        self.prediction_time = prediction_time or datetime.now(timezone.utc)
        self._build_feature_spec()
    
    def _build_feature_spec(self):
        self.needs_min = {}
        self.needs_max = {}
        self.needs_mean = {}
        self.needs_sum = {}
        self.needs_bool_or = {}
        self.needs_twa = {}
        self.needs_indicator = {}
        self.single_value = {}
        
        for fname in self.feature_names:
            if fname.startswith('has_'):
                self.needs_indicator[fname] = fname
            elif '_MIN_' in fname:
                base, window = self._parse_agg_feature(fname, '_MIN_')
                self.needs_min[fname] = (base, window)
            elif '_MAX_' in fname:
                base, window = self._parse_agg_feature(fname, '_MAX_')
                self.needs_max[fname] = (base, window)
            elif '_MEAN_' in fname:
                base, window = self._parse_agg_feature(fname, '_MEAN_')
                self.needs_mean[fname] = (base, window)
            elif '_SUM_' in fname:
                base, window = self._parse_agg_feature(fname, '_SUM_')
                self.needs_sum[fname] = (base, window)
            elif '_BOOL_OR_' in fname:
                base, window = self._parse_agg_feature(fname, '_BOOL_OR_')
                self.needs_bool_or[fname] = (base, window)
            elif '_TWA_' in fname:
                base, window = self._parse_agg_feature(fname, '_TWA_')
                self.needs_twa[fname] = (base, window)
            else:
                self.single_value[fname] = fname
    
    def _parse_agg_feature(self, fname: str, agg_type: str) -> Tuple[str, str]:
        parts = fname.split(agg_type)
        if len(parts) == 2:
            return parts[0], parts[1]
        return fname, ''
    
    def extract_features(
        self,
        patient_demographics: Dict[str, Any],
        vitals_records: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        features = {}
        
        for fname in self.feature_names:
            features[fname] = float('nan')
        
        self._extract_demographics(patient_demographics, features)
        
        records_by_window = self._partition_records_by_window(vitals_records)
        
        self._extract_vitals(records_by_window, features)
        self._extract_labs(records_by_window, features)
        self._extract_diagnoses(records_by_window, features)
        self._extract_interventions(records_by_window, features)
        self._extract_ventilation(records_by_window, features)
        self._extract_fluids(records_by_window, features)
        self._extract_indicators(records_by_window, features)
        self._compute_twa(records_by_window, features)
        self._compute_derived_features(patient_demographics, features, vitals_records)
        
        return features
    
    def _extract_demographics(self, demo: Dict[str, Any], features: Dict[str, float]):
        for eth_col, eth_values in ETHNICITY_ONEHOT.items():
            features[eth_col] = 0.0
            patient_eth = (demo.get('ethnicity') or '').upper()
            for val in eth_values:
                if val in patient_eth:
                    features[eth_col] = 1.0
                    break
        
        if 'age_years' in self.feature_names:
            dob = demo.get('dob')
            if dob:
                try:
                    birth = datetime.fromisoformat(dob.replace('Z', '+00:00'))
                    if birth.tzinfo is None and self.prediction_time.tzinfo is not None:
                        birth = birth.replace(tzinfo=timezone.utc)
                    age_days = (self.prediction_time - birth).days
                    features['age_years'] = age_days / 365.25
                except (ValueError, TypeError, OverflowError):
                    pass
        
        if 'is_male' in self.feature_names:
            features['is_male'] = 1.0 if demo.get('gender') == 'M' else 0.0
        
        if 'Height_MEAN_YearAgo' in self.feature_names:
            height = demo.get('height_cm')
            if height is not None:
                features['Height_MEAN_YearAgo'] = float(height)
        
        if 'smoker_any' in self.feature_names:
            features['smoker_any'] = 1.0 if demo.get('smoker_any', False) else 0.0
    
    def _partition_records_by_window(
        self, records: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        result = {window: [] for window in TIME_WINDOWS}
        result['Ever'] = list(records)
        
        for record in records:
            ts = self._parse_timestamp(record.get('timestamp'))
            if ts is None:
                continue
            
            age = self.prediction_time - ts
            if age.total_seconds() < 0:
                continue
            
            for window_name, window_delta in TIME_WINDOWS.items():
                if age <= window_delta:
                    result[window_name].append(record)
        
        return result
    
    def _parse_timestamp(self, ts: Any) -> Optional[datetime]:
        if ts is None:
            return None
        try:
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return ts
            ts_str = str(ts).replace('Z', '+00:00')
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None
    
    def _extract_vitals(self, records_by_window: Dict, features: Dict[str, float]):
        for vital_field, vital_prefix in VITAL_FIELD_MAPPING.items():
            self._aggregate_numeric_field(
                records_by_window, features, vital_field, vital_prefix
            )
    
    def _extract_labs(self, records_by_window: Dict, features: Dict[str, float]):
        for lab_field, lab_prefix in LAB_FIELD_MAPPING.items():
            self._aggregate_numeric_field(
                records_by_window, features, lab_field, lab_prefix
            )
    
    def _aggregate_numeric_field(
        self,
        records_by_window: Dict,
        features: Dict[str, float],
        source_field: str,
        target_prefix: str,
    ):
        for window_name, records in records_by_window.items():
            values = []
            for r in records:
                val = r.get(source_field)
                if val is not None and not math.isnan(val) if isinstance(val, float) else val is not None:
                    try:
                        v = float(val)
                        if target_prefix in FIO2_FIELDS:
                            v = _normalize_fio2(v)
                        values.append(v)
                    except (ValueError, TypeError):
                        pass
            
            if not values:
                continue
            
            min_key = f"{target_prefix}_MIN_{window_name}"
            max_key = f"{target_prefix}_MAX_{window_name}"
            mean_key = f"{target_prefix}_MEAN_{window_name}"
            
            if min_key in self.feature_names:
                features[min_key] = float(min(values))
            if max_key in self.feature_names:
                features[max_key] = float(max(values))
            if mean_key in self.feature_names:
                features[mean_key] = float(sum(values) / len(values))
    
    def _extract_diagnoses(self, records_by_window: Dict, features: Dict[str, float]):
        for diag_field, diag_prefix in DIAGNOSIS_FIELD_MAPPING.items():
            for window_name, records in records_by_window.items():
                bool_key = f"{diag_prefix}_BOOL_OR_{window_name}"
                if bool_key in self.feature_names:
                    has_diagnosis = any(
                        r.get(diag_field, False) is True
                        for r in records
                    )
                    features[bool_key] = 1.0 if has_diagnosis else 0.0
        
        neuro_windows = {'HalfWeekAgo', 'WeekAgo', 'TwoWeeksAgo'}
        neuro_fields = ['stroke_diagnosis', 'tbi_diagnosis', 'peripheral_neuropathy_diagnosis']
        
        for window_name in neuro_windows:
            if window_name not in records_by_window:
                continue
            bool_key = f"Neuro_Impairment_NonParalytic_BOOL_OR_{window_name}"
            if bool_key in self.feature_names:
                has_neuro = any(
                    any(r.get(f, False) for f in neuro_fields)
                    for r in records_by_window[window_name]
                )
                features[bool_key] = 1.0 if has_neuro else 0.0
    
    def _extract_interventions(self, records_by_window: Dict, features: Dict[str, float]):
        reposition_counts = {}
        for window_name, records in records_by_window.items():
            count = sum(1 for r in records if r.get('repositioned', False) is True)
            if count > 0:
                reposition_counts[window_name] = count
        
        for window_name, count in reposition_counts.items():
            sum_key = f"Reposition_Count_SUM_{window_name}"
            if sum_key in self.feature_names:
                features[sum_key] = float(count)
        
        restraint_windows = {'HalfWeekAgo', 'WeekAgo'}
        for window_name in restraint_windows:
            if window_name not in records_by_window:
                continue
            sum_key = f"Restraint_Any_SUM_{window_name}"
            if sum_key in self.feature_names:
                count = sum(1 for r in records_by_window[window_name] 
                           if r.get('on_restraints', False) is True)
                features[sum_key] = float(count)
        
        paralysis_windows = {'DayAgo', 'HalfWeekAgo', 'WeekAgo', 'TwoWeeksAgo'}
        paralysis_drugs = ['cisatracurium_volume_ml', 'rocuronium_volume_ml', 'vecuronium_volume_ml']
        
        for window_name in paralysis_windows:
            if window_name not in records_by_window:
                continue
            bool_key = f"Pharmacologic_Paralysis_BOOL_OR_{window_name}"
            if bool_key in self.feature_names:
                has_paralysis = any(
                    any(r.get(drug) and r.get(drug) > 0 for drug in paralysis_drugs)
                    for r in records_by_window[window_name]
                )
                features[bool_key] = 1.0 if has_paralysis else 0.0
    
    def _extract_ventilation(self, records_by_window: Dict, features: Dict[str, float]):
        vent_key = "Ventilation_Status_MAX_DayAgo"
        if vent_key in self.feature_names and 'DayAgo' in records_by_window:
            records = records_by_window['DayAgo']
            max_status = 0
            for r in records:
                status = 0
                if r.get('on_tracheostomy', False):
                    status = 3
                elif r.get('on_invasive_ventilation', False):
                    status = 2
                elif r.get('on_non_invasive_ventilation', False):
                    status = 1
                max_status = max(max_status, status)
            features[vent_key] = float(max_status)
        
        cc_device_key = "Critical_Care_Device_BOOL_OR_HalfWeekAgo"
        if cc_device_key in self.feature_names and 'HalfWeekAgo' in records_by_window:
            features[cc_device_key] = 0.0
    
    def _extract_fluids(self, records_by_window: Dict, features: Dict[str, float]):
        fluid_in_windows = {'HalfWeekAgo', 'WeekAgo'}
        for window_name in fluid_in_windows:
            if window_name not in records_by_window:
                continue
            sum_key = f"Fluid_In_SUM_{window_name}"
            if sum_key in self.feature_names:
                total = sum(
                    r.get('intake_ml', 0) or 0
                    for r in records_by_window[window_name]
                )
                features[sum_key] = float(total)
        
        fluid_out_windows = {'HalfWeekAgo', 'WeekAgo', 'TwoWeeksAgo'}
        for window_name in fluid_out_windows:
            if window_name not in records_by_window:
                continue
            sum_key = f"Fluid_Out_SUM_{window_name}"
            if sum_key in self.feature_names:
                total = sum(
                    r.get('urine_output_ml', 0) or 0
                    for r in records_by_window[window_name]
                )
                features[sum_key] = float(total)
    
    def _extract_indicators(self, records_by_window: Dict, features: Dict[str, float]):
        for fname in self.feature_names:
            if not fname.startswith('has_'):
                continue
            
            parts = fname.split('_', 1)
            if len(parts) < 2:
                continue
            
            measurement = parts[1].rsplit('_', 1)[0]
            window = parts[1].rsplit('_', 1)[1]
            
            if window not in records_by_window:
                features[fname] = 0.0
                continue
            
            field_map = {**LAB_FIELD_MAPPING, **VITAL_FIELD_MAPPING}
            reverse_map = {v: k for k, v in field_map.items()}
            
            source_field = reverse_map.get(measurement)
            if source_field is None:
                source_field = measurement.lower()
            
            has_value = any(
                r.get(source_field) is not None
                for r in records_by_window[window]
            )
            features[fname] = 1.0 if has_value else 0.0
    
    def _compute_twa(self, records_by_window: Dict, features: Dict[str, float]):
        """Compute time-weighted averages using trapezoidal rule.
        
        TWA = integral(value dt) / time_window
        
        For discrete measurements, this uses trapezoidal integration:
        TWA = sum((v_i + v_{i+1})/2 * (t_{i+1} - t_i)) / time_window_duration
        """
        for twa_prefix, source_field in TWA_SOURCE_FIELDS.items():
            for window_name, window_delta in TIME_WINDOWS.items():
                twa_key = f"{twa_prefix}_TWA_{window_name}"
                if twa_key not in self.feature_names:
                    continue
                
                records = records_by_window.get(window_name, [])
                if not records:
                    continue
                
                points = []
                for r in records:
                    val = r.get(source_field)
                    ts = self._parse_timestamp(r.get('timestamp'))
                    if val is not None and ts is not None:
                        try:
                            v = float(val)
                            if twa_prefix in FIO2_FIELDS:
                                v = _normalize_fio2(v)
                            points.append((ts, v))
                        except (ValueError, TypeError):
                            continue
                
                if not points:
                    continue
                
                points.sort(key=lambda p: p[0])
                
                if len(points) == 1:
                    features[twa_key] = points[0][1]
                    continue
                
                total_weighted = 0.0
                for i in range(len(points) - 1):
                    t1, v1 = points[i]
                    t2, v2 = points[i + 1]
                    dt = (t2 - t1).total_seconds()
                    total_weighted += (v1 + v2) / 2.0 * dt
                
                window_seconds = window_delta.total_seconds()
                if window_seconds > 0:
                    features[twa_key] = total_weighted / window_seconds
        
        for window_name in TIME_WINDOWS:
            neq_key = f"neq_twa_{window_name}"
            if neq_key in self.feature_names:
                features[neq_key] = 0.0
    
    def _compute_derived_features(
        self,
        demo: Dict[str, Any],
        features: Dict[str, float],
        records: List[Dict[str, Any]],
    ):
        if 'BMI_MEAN_WeekAgo' in self.feature_names:
            height_cm = demo.get('height_cm')
            if height_cm:
                weights = []
                for r in records:
                    w = r.get('daily_weight')
                    if w is not None:
                        try:
                            weights.append(float(w))
                        except (ValueError, TypeError):
                            pass
                
                if weights:
                    avg_weight = sum(weights) / len(weights)
                    height_m = height_cm / 100.0
                    bmi = avg_weight / (height_m * height_m)
                    features['BMI_MEAN_WeekAgo'] = bmi
        
        if 'length_of_stay_to_incident' in self.feature_names:
            admission = demo.get('admission_time')
            if admission:
                try:
                    admit_dt = datetime.fromisoformat(admission.replace('Z', '+00:00'))
                    los_hours = (self.prediction_time - admit_dt).total_seconds() / 3600
                    features['length_of_stay_to_incident'] = math.log(max(0.1, los_hours))
                except (ValueError, TypeError):
                    pass
