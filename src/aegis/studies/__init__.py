"""Public, dependency-light study configuration API."""

from aegis.harness.types import RunSettings, StudyDefinition, StudyMode, StudyResult

from .loader import load_study

__all__ = [
    "RunSettings",
    "StudyDefinition",
    "StudyMode",
    "StudyResult",
    "load_study",
]
