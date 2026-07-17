"""PIMAP Dashboard module.

Provides Lambda handlers for the demo dashboard API and frontend assets.
"""

__all__ = [
    "get_patients",
    "get_patient_vitals",
    "update_action",
    "predict_pressure_ulcer",
]


def __getattr__(name):
    if name == "get_patients":
        from .api.patients import get_patients
        return get_patients
    elif name == "get_patient_vitals":
        from .api.vitals import get_patient_vitals
        return get_patient_vitals
    elif name == "update_action":
        from .api.actions import update_action
        return update_action
    elif name == "predict_pressure_ulcer":
        from .api.predict import predict_pressure_ulcer
        return predict_pressure_ulcer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
