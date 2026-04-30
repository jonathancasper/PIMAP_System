"""PIMAP Dashboard module.

Provides Lambda handlers for the demo dashboard API and frontend assets.
"""

from .api.patients import get_patients
from .api.vitals import get_patient_vitals
from .api.actions import update_action
from .api.predict import predict_pressure_ulcer

__all__ = [
    "get_patients",
    "get_patient_vitals",
    "update_action",
    "predict_pressure_ulcer",
]
