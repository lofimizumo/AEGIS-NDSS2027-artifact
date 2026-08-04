"""FedSpy-LLM B=1 gradient-inversion attack."""

from __future__ import annotations

from itertools import permutations
from typing import Any

import torch
import torch.nn as nn

from .helpers import (
    AttackResult,
    GradientDefense,
    apply_defense,
    compute_text_metrics,
    get_input_embeddings,
    grad_context,
)


def _embedding_gradient(
    model: nn.Module,
    input_ids: torch.Tensor,
    defense: GradientDefense | None,
) -> torch.Tensor:
    embeddings = get_input_embeddings(model)
    model.zero_grad()
    with grad_context(embeddings.weight):
        model(input_ids=input_ids, labels=input_ids).loss.backward()
        apply_defense(defense)
        gradient = (
            embeddings.weight.grad.detach().clone()
            if embeddings.weight.grad is not None
            else torch.zeros_like(embeddings.weight)
        )
    model.zero_grad()
    return gradient


def _full_gradient(
    model: nn.Module,
    input_ids: torch.Tensor,
    defense: GradientDefense | None,
) -> torch.Tensor:
    model.zero_grad()
    model(input_ids=input_ids, labels=input_ids).loss.backward()
    apply_defense(defense)
    gradients = [
        parameter.grad.detach().flatten()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    model.zero_grad()
    return torch.cat(gradients)


def run_fedspy(
    model: nn.Module,
    tokenizer: Any,
    sentence: str,
    device: str,
    defense: GradientDefense | None = None,
    max_len: int = 32,
    max_perm_t: int = 8,
) -> AttackResult:
    """Recover active embedding rows, then calibrate their sequence order."""
    model.eval()
    input_ids = tokenizer(sentence, return_tensors="pt").to(device)["input_ids"]
    input_ids = input_ids[:, :max_len]
    length = input_ids.shape[1]
    true_ids = input_ids.squeeze(0).cpu()

    embedding_gradient = _embedding_gradient(model, input_ids, defense)
    row_norms = embedding_gradient.norm(dim=1)
    threshold = 0.01 * row_norms.max()
    candidates = (row_norms > threshold).nonzero(as_tuple=True)[0]
    if candidates.numel() == 0:
        result = compute_text_metrics(torch.zeros(length, dtype=torch.long), true_ids, tokenizer)
        result["attack"] = "fedspy_llm"
        return result

    candidate_norms = row_norms[candidates]
    token_ids = candidates[candidate_norms.argsort(descending=True)].cpu().tolist()
    while len(token_ids) < length:
        token_ids.append(token_ids[len(token_ids) % len(candidates)])
    token_ids = token_ids[:length]

    observed = _full_gradient(model, input_ids, defense)
    observed = observed / (observed.norm() + 1e-8)
    if length <= max_perm_t:
        best_score = -1e9
        best_order = token_ids[:]
        for ordering in permutations(range(length)):
            trial_ids = [token_ids[index] for index in ordering]
            trial = _full_gradient(model, torch.tensor([trial_ids], device=device), None)
            score = (trial * observed).sum().item()
            if score > best_score:
                best_score = score
                best_order = trial_ids
        token_ids = best_order
    else:
        remaining = token_ids[:]
        fixed: list[int] = []
        for position in range(length):
            best_score = -1e9
            best_token = remaining[0]
            pad_token = remaining[0]
            for candidate in dict.fromkeys(remaining):
                trial_ids = fixed + [candidate] + [pad_token] * (length - position - 1)
                trial = _full_gradient(model, torch.tensor([trial_ids], device=device), None)
                score = (trial * observed).sum().item()
                if score > best_score:
                    best_score = score
                    best_token = candidate
            fixed.append(best_token)
            remaining.remove(best_token)
        token_ids = fixed

    result = compute_text_metrics(torch.tensor(token_ids, dtype=torch.long), true_ids, tokenizer)
    result["attack"] = "fedspy_llm"
    return result


__all__ = ["run_fedspy"]
