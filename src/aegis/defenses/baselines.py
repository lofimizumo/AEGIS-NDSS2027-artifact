"""Baseline gradient defenses and their public factory."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from .aegis import AegisDefense
from .base import Defense


class NoDefense(Defense):
    """No-op baseline."""

    name = "none"

    def attack_hook(self, model: nn.Module) -> None:
        return None


class GaussianNoiseDefense(Defense):

    def __init__(self, noise_std: float) -> None:
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        self.noise_std = noise_std
        self.name = f"gaussian_noise_{noise_std:g}"

    def after_backward(self, model: nn.Module) -> None:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.add_(torch.randn_like(parameter.grad) * self.noise_std)


class GradientPruningDefense(Defense):

    def __init__(
        self,
        percent: float,
        *,
        target_sample_size: int = 1_000_000,
    ) -> None:
        if not 0 <= percent <= 100:
            raise ValueError("percent must be between 0 and 100")
        if target_sample_size <= 0:
            raise ValueError("target_sample_size must be positive")
        self.percent = percent
        self.target_sample_size = target_sample_size
        self.name = f"gradient_pruning_{percent:g}"

    def after_backward(self, model: nn.Module) -> None:
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None and parameter.grad.numel() > 100
        ]
        if not gradients:
            return

        total_elements = sum(gradient.numel() for gradient in gradients)
        samples: list[torch.Tensor] = []
        for gradient in gradients:
            sample_count = max(
                1,
                int(self.target_sample_size * (gradient.numel() / total_elements)),
            )
            sample_count = min(sample_count, gradient.numel())
            indices = torch.randint(
                0,
                gradient.numel(),
                (sample_count,),
                device=gradient.device,
            )
            samples.append(gradient.abs().view(-1)[indices])

        global_sample = torch.cat(samples)
        threshold = torch.quantile(global_sample.float(), self.percent / 100).to(gradients[0].dtype)
        for gradient in gradients:
            gradient.mul_((gradient.abs() >= threshold).to(gradient.dtype))


class SoteriaDefense(Defense):
    name = "soteria"

    def __init__(self, drop_frac: float = 0.10) -> None:
        if not 0 <= drop_frac <= 1:
            raise ValueError("drop_frac must be between 0 and 1")
        self.drop_frac = drop_frac
        self._target: nn.Linear | None = None
        self._handle: torch.utils.hooks.RemovableHandle | None = None
        self._model: nn.Module | None = None

    def pre_train(self, model: nn.Module) -> None:
        if self._model is model and self._handle is not None:
            return
        self.close()

        candidates = [
            (module.weight.numel(), module)
            for module in model.modules()
            if isinstance(module, nn.Linear)
        ]
        if not candidates:
            self._model = model
            return
        _, target = max(candidates, key=lambda candidate: candidate[0])
        self._target = target
        self._model = model

        def mask_input_gradient(
            module: nn.Module,
            grad_input: tuple[torch.Tensor | None, ...],
            grad_output: tuple[torch.Tensor | None, ...],
        ) -> tuple[torch.Tensor | None, ...] | None:
            del module, grad_output
            if not grad_input or grad_input[0] is None:
                return None
            input_gradient = grad_input[0]
            flat = input_gradient.abs().flatten()
            count = int(self.drop_frac * flat.numel())
            if count == 0:
                return None
            threshold = torch.kthvalue(
                flat,
                max(flat.numel() - count, 1),
            ).values
            mask = (input_gradient.abs() < threshold).to(input_gradient.dtype)
            return (input_gradient * mask, *grad_input[1:])

        self._handle = target.register_full_backward_hook(mask_input_gradient)

    def close(self) -> None:
        """Remove the installed backward hook, if any."""

        if self._handle is not None:
            self._handle.remove()
        self._handle = None
        self._target = None
        self._model = None


DefenseFactory = Callable[[], Defense]

DEFENSES = (
    "none",
    "gaussian_noise_0.001",
    "gaussian_noise_0.01",
    "gaussian_noise_0.1",
    "gradient_pruning_50",
    "gradient_pruning_90",
    "gradient_pruning_99",
    "soteria",
    "aegis",
)

_STATIC_FACTORIES: dict[str, DefenseFactory] = {
    "none": NoDefense,
    "aegis": AegisDefense,
    "soteria": SoteriaDefense,
}


def make_defense(name: str) -> Defense:
    """Construct a defense from its canonical slug."""

    factory = _STATIC_FACTORIES.get(name)
    if factory is not None:
        return factory()

    if name in DEFENSES and name.startswith("gaussian_noise_"):
        return GaussianNoiseDefense(float(name.removeprefix("gaussian_noise_")))

    if name in DEFENSES and name.startswith("gradient_pruning_"):
        return GradientPruningDefense(float(name.removeprefix("gradient_pruning_")))

    choices = ", ".join(DEFENSES)
    raise ValueError(f"unknown defense {name!r}; expected one of: {choices}")


__all__ = [
    "DEFENSES",
    "GaussianNoiseDefense",
    "GradientPruningDefense",
    "NoDefense",
    "SoteriaDefense",
    "make_defense",
]
