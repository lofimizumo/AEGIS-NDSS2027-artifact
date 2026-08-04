"""Inline/custom GRAB hybrid gradient-inversion attack."""

from __future__ import annotations

import random
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn

from .helpers import (
    AttackResult,
    GradientDefense,
    apply_defense,
    compute_text_metrics,
    empty_result,
    get_hidden_size,
    get_input_embeddings,
    get_last_block_parameters,
    grad_context,
    math_attention_context,
)


def _gradient_vector(
    model: nn.Module,
    input_ids: torch.Tensor,
    parameters: list[nn.Parameter],
    defense: GradientDefense | None,
) -> torch.Tensor:
    device = input_ids.device
    autocast = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()
    model.zero_grad()
    with autocast:
        loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    apply_defense(defense)
    gradient = torch.cat(
        [
            parameter.grad.detach().reshape(-1).float()
            if parameter.grad is not None
            else torch.zeros(parameter.numel(), device=device, dtype=torch.float32)
            for parameter in parameters
        ]
    )
    model.zero_grad()
    return gradient


def _recovery_loss(
    model: nn.Module,
    dummy_embeddings: torch.Tensor,
    target_ids: torch.Tensor,
    parameters: list[nn.Parameter],
    observed_gradients: list[torch.Tensor],
    l1_weights: list[float],
) -> torch.Tensor:
    model.zero_grad()
    try:
        with math_attention_context():
            output = model(inputs_embeds=dummy_embeddings, labels=target_ids)
    except TypeError:
        return dummy_embeddings.new_tensor(float("inf"))
    gradients = torch.autograd.grad(
        output.loss,
        parameters,
        create_graph=True,
        allow_unused=True,
    )
    loss = dummy_embeddings.new_zeros(())
    for gradient, observed, alpha in zip(gradients, observed_gradients, l1_weights, strict=True):
        if gradient is not None:
            difference = gradient - observed.to(dummy_embeddings.device)
            loss = loss + difference.pow(2).sum() + alpha * difference.abs().sum()
    model.zero_grad()
    return loss


def run_grab(
    model: nn.Module,
    tokenizer: Any,
    sentence: str,
    device: str,
    defense: GradientDefense | None = None,
    n_h: int = 3,
    n_c: int = 200,
    n_d: int = 3,
    n_b: int = 2,
    n_e: int = 10,
    lr: float = 0.01,
    alpha_l1: float = 0.01,
    max_tokens: int = 32,
) -> AttackResult:
    """Run alternating continuous token-bag and discrete ordering recovery."""
    model.eval()
    input_ids = tokenizer(sentence, return_tensors="pt").to(device)["input_ids"]
    input_ids = input_ids[:, :max_tokens]
    length = input_ids.shape[1]
    true_ids = input_ids.squeeze(0).cpu()
    embeddings = get_input_embeddings(model)
    parameters = get_last_block_parameters(model, n=3)
    if not parameters:
        return empty_result("grab", "no trainable matching parameters")

    with grad_context(embeddings.weight):
        model.zero_grad()
        model(input_ids=input_ids, labels=input_ids).loss.backward()
        apply_defense(defense)
        observed = [
            parameter.grad.detach().clone()
            if parameter.grad is not None
            else torch.zeros_like(parameter)
            for parameter in parameters
        ]
    model.zero_grad()
    observed_flat = torch.cat([gradient.reshape(-1).float() for gradient in observed])
    count = len(parameters)
    alphas = [alpha_l1 * (count - index) / max(count, 1) for index in range(count)]

    def discrete_cosine(token_ids: list[int]) -> float:
        trial = _gradient_vector(
            model,
            torch.tensor([token_ids], device=device),
            parameters,
            defense,
        )
        denominator = trial.norm() * observed_flat.norm()
        return (trial @ observed_flat / denominator).item() if denominator > 1e-12 else -1.0

    dtype = next(model.parameters()).dtype
    hidden_size = get_hidden_size(model)
    best_initial_loss = float("inf")
    discrete_embeddings = torch.randn(1, length, hidden_size, device=device, dtype=dtype)
    for _ in range(n_e):
        candidate = torch.randn(1, length, hidden_size, device=device, dtype=dtype)
        try:
            model.zero_grad()
            model(inputs_embeds=candidate.detach(), labels=input_ids).loss.backward()
            proxy = sum(
                (
                    (parameter.grad if parameter.grad is not None else torch.zeros_like(parameter))
                    - observed[index].to(device)
                )
                .pow(2)
                .sum()
                .item()
                for index, parameter in enumerate(parameters)
            )
            model.zero_grad()
        except Exception:
            proxy = float("inf")
        if proxy < best_initial_loss:
            best_initial_loss = proxy
            discrete_embeddings = candidate.detach().clone()

    best_cosine = -2.0
    best_ids = list(range(length))
    for _ in range(n_h):
        dummy = discrete_embeddings.clone().detach().requires_grad_(True)
        optimizer = torch.optim.AdamW([dummy], lr=lr, fused=torch.cuda.is_available())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.89)
        best_continuous_loss = float("inf")
        stale = 0
        for _ in range(n_c):
            optimizer.zero_grad()
            loss = _recovery_loss(model, dummy, input_ids, parameters, observed, alphas)
            if not loss.isfinite():
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_([dummy], max_norm=1.0)
            optimizer.step()
            scheduler.step()
            value = loss.item()
            if best_continuous_loss - value > 1e-4:
                best_continuous_loss = value
                stale = 0
            else:
                stale += 1
                if stale >= 10:
                    break

        with torch.no_grad():
            continuous_ids = (
                torch.cdist(
                    dummy.detach().squeeze(0).float(),
                    embeddings.weight.detach().float(),
                )
                .argmin(dim=1)
                .tolist()
            )
        continuous_cosine = discrete_cosine(continuous_ids)
        if continuous_cosine > best_cosine:
            best_cosine = continuous_cosine
            best_ids = continuous_ids[:]

        token_set = list(dict.fromkeys(continuous_ids))[:10]
        initial_beams = [continuous_ids[:]] + [
            random.sample(continuous_ids, length) for _ in range(min(n_e * 2, 10))
        ]
        beams = sorted(
            [(discrete_cosine(beam), beam[:]) for beam in initial_beams],
            key=lambda item: -item[0],
        )[:n_b]
        previous = beams[0][0]
        for _ in range(n_d):
            candidates = list(beams)
            for _, beam in beams:
                for position in range(length):
                    for token in token_set:
                        if token != beam[position]:
                            trial = beam.copy()
                            trial[position] = token
                            candidates.append((discrete_cosine(trial), trial))
            beams = sorted(candidates, key=lambda item: -item[0])[:n_b]
            if beams[0][0] - previous < 1e-4:
                break
            previous = beams[0][0]

        discrete_cos, discrete_ids = beams[0]
        if discrete_cos > best_cosine:
            best_cosine = discrete_cos
            best_ids = discrete_ids[:]
        with torch.no_grad():
            discrete_embeddings = (
                torch.stack([embeddings.weight[token] for token in discrete_ids])
                .unsqueeze(0)
                .detach()
            )

    result = compute_text_metrics(torch.tensor(best_ids, dtype=torch.long), true_ids, tokenizer)
    result["attack"] = "grab"
    return result


__all__ = ["run_grab"]
