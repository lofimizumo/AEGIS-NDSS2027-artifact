"""Defense implementations for AEGIS."""

from .aegis import AegisDefense
from .base import Defense, GradientDefenseHook
from .baselines import (
    DEFENSES,
    GaussianNoiseDefense,
    GradientPruningDefense,
    NoDefense,
    SoteriaDefense,
    make_defense,
)

__all__ = [
    "DEFENSES",
    "AegisDefense",
    "Defense",
    "GaussianNoiseDefense",
    "GradientDefenseHook",
    "GradientPruningDefense",
    "NoDefense",
    "SoteriaDefense",
    "make_defense",
]
