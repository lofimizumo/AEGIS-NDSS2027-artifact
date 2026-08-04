"""Strict standard-library loader for external study definitions."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, NoReturn

from aegis.harness.types import JsonValue, StudyDefinition, StudyMode

_NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_REQUIRED_FIELDS = frozenset(
    {
        "name",
        "title",
        "mode",
        "models",
        "datasets",
        "defenses",
        "attacks",
        "defaults",
        "quick_defaults",
        "validation_defaults",
        "output_filename",
    }
)
_OPTIONAL_FIELDS = frozenset({"fixed_sentences", "behavior"})


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key!r}")
        result[key] = value
    return result


def _validate_json_value(value: object, location: str) -> JsonValue:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [
            _validate_json_value(item, f"{location}[{index}]") for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{location} must use string keys")
        return {key: _validate_json_value(item, f"{location}.{key}") for key, item in value.items()}
    raise ValueError(f"{location} contains an unsupported JSON value")


def _string_list(document: dict[str, Any], field: str) -> tuple[str, ...]:
    value = document[field]
    if not isinstance(value, list) or not all(
        isinstance(item, str) and bool(item) for item in value
    ):
        raise ValueError(f"{field} must be an array of non-empty strings")
    return tuple(value)


def _object(document: dict[str, Any], field: str) -> dict[str, JsonValue]:
    value = document[field]
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    validated = _validate_json_value(value, field)
    if not isinstance(validated, dict):
        raise ValueError(f"{field} must be a JSON object")
    return validated


def _fixed_sentences(document: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    value = document.get("fixed_sentences", {})
    if not isinstance(value, dict):
        raise ValueError("fixed_sentences must be a JSON object")
    result: dict[str, tuple[str, ...]] = {}
    for dataset, sentences in value.items():
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("fixed_sentences must use non-empty dataset names")
        if not isinstance(sentences, list) or not all(
            isinstance(sentence, str) for sentence in sentences
        ):
            raise ValueError(f"fixed_sentences.{dataset} must be an array of strings")
        result[dataset] = tuple(sentences)
    return result


def _output_filename(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or not value.endswith(".json")
    ):
        raise ValueError("output_filename must be a safe local JSON filename")
    return value


def load_study(path: str | Path) -> StudyDefinition:
    """Load and strictly validate one external JSON study definition."""

    source = Path(path)
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load study definition {source}: {error}") from error

    if not isinstance(raw, dict):
        raise ValueError("study definition must be a JSON object")
    document = raw
    fields = set(document)
    missing = _REQUIRED_FIELDS - fields
    unknown = fields - _REQUIRED_FIELDS - _OPTIONAL_FIELDS
    if missing:
        raise ValueError(f"missing required fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"unknown fields: {', '.join(sorted(unknown))}")

    name = document["name"]
    if not isinstance(name, str) or _NAME_PATTERN.fullmatch(name) is None:
        raise ValueError("name must use lower-case kebab syntax")
    title = document["title"]
    if not isinstance(title, str) or not title:
        raise ValueError("title must be a non-empty string")
    try:
        mode = StudyMode(document["mode"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported study mode: {document['mode']!r}") from error

    behavior = _object(document, "behavior") if "behavior" in document else {}
    definition = StudyDefinition(
        name=name,
        title=title,
        mode=mode,
        models=_string_list(document, "models"),
        datasets=_string_list(document, "datasets"),
        defenses=_string_list(document, "defenses"),
        attacks=_string_list(document, "attacks"),
        defaults=_object(document, "defaults"),
        quick_defaults=_object(document, "quick_defaults"),
        validation_defaults=_object(document, "validation_defaults"),
        output_filename=_output_filename(document["output_filename"]),
        fixed_sentences=_fixed_sentences(document),
        behavior=behavior,
    )
    _validate_json_value(document, "study")
    return definition


__all__ = ["load_study"]
