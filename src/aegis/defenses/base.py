"""Common interfaces for gradient defenses."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch.nn as nn


@runtime_checkable
class GradientDefenseHook(Protocol):
    """Attack-facing view of a defense.

    Existing inversion attacks call ``flood_gradients`` after constructing a
    victim gradient.  The protocol keeps that small compatibility surface
    without coupling attacks to a concrete defense.
    """

    def flood_gradients(self, target: nn.Module | None = None) -> None:
        """Perturb gradients on ``target`` or on the captured model."""


class _AfterBackwardHook:
    def __init__(self, defense: Defense, model: nn.Module) -> None:
        self._defense = defense
        self._model = model

    def flood_gradients(self, target: nn.Module | None = None) -> None:
        self._defense.after_backward(target if target is not None else self._model)


class Defense:
    """Base class for defenses used during training and attack evaluation."""

    name = "base"
    needs_per_sample_grads = False

    def pre_train(self, model: nn.Module) -> None:
        """Apply one-time model changes before optimizer construction."""

    def after_backward(self, model: nn.Module) -> None:
        """Modify gradients after backward and before the optimizer step."""

    def attack_hook(self, model: nn.Module) -> GradientDefenseHook | None:
        """Return the hook that reproduces the defense on attack gradients."""

        return _AfterBackwardHook(self, model)


__all__ = ["Defense", "GradientDefenseHook"]
