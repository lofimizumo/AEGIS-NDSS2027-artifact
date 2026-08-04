"""Typed, dependency-light study abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class StudyMode(str, Enum):
    """Execution shape used by the unified runner."""

    TRAIN_ATTACK = "train_attack"
    ABLATION = "ablation"
    CONVERGENCE = "convergence"
    PARETO = "pareto"
    QUALITATIVE = "qualitative"
    SIDE_CHANNEL = "side_channel"
    FIXED_GRID = "fixed_grid"


@dataclass(frozen=True)
class StudyDefinition:
    """Validated external definition of one study protocol."""

    name: str
    title: str
    mode: StudyMode
    models: tuple[str, ...]
    datasets: tuple[str, ...]
    defenses: tuple[str, ...]
    attacks: tuple[str, ...]
    defaults: dict[str, JsonValue]
    quick_defaults: dict[str, JsonValue]
    validation_defaults: dict[str, JsonValue]
    output_filename: str
    fixed_sentences: dict[str, tuple[str, ...]] = field(default_factory=dict)
    behavior: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSettings:
    """Resolved, typed configuration passed to the runner."""

    study: StudyDefinition
    models: tuple[str, ...]
    datasets: tuple[str, ...]
    defenses: tuple[str, ...]
    attacks: tuple[str, ...]
    seed: int = 42
    quick: bool = False
    validate: bool = False
    model_dir: Path | None = None
    cache_dir: Path | None = None
    output_dir: Path = Path("results")
    device: str | None = None
    parameters: dict[str, JsonValue] = field(default_factory=dict)
    output_name: str | None = None


@dataclass(frozen=True)
class StudyResult:
    """Result payload and the path written by the harness."""

    study: str
    output_path: Path
    payload: dict[str, Any]


__all__ = [
    "JsonScalar",
    "JsonValue",
    "RunSettings",
    "StudyDefinition",
    "StudyMode",
    "StudyResult",
]
