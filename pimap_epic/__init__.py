"""PIMAP Epic integration module.

Provides FHIR R4 client for Epic EHR systems, including:
- OAuth 2.0 authentication (Backend System flow)
- Patient and observation data retrieval
- Risk assessment write-back (stub)
"""

from .auth import EpicAuth, EpicAuthError
from .fhir_client import EpicFHIRClient, FHIRRequestError

__all__ = [
    "EpicAuth",
    "EpicAuthError",
    "EpicFHIRClient",
    "FHIRRequestError",
]
