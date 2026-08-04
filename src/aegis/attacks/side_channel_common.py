"""Shared machinery for adaptive gradient side-channel probes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .helpers import (
    AttackResult,
    GradientDefense,
    apply_defense,
    compute_text_metrics,
    get_hidden_size,
    get_input_embeddings,
    grad_context,
    language_model_loss,
)


@dataclass
class ProbeContext:
    model: nn.Module
    tokenizer: Any
    true_ids: torch.Tensor
    embedding_weight: torch.Tensor
    hidden_size: int
    top_t: int


def collect_probe_context(
    model: nn.Module,
    tokenizer: Any,
    sentence: str,
    device: str,
    defense: GradientDefense | None,
    max_tokens: int,
) -> ProbeContext:
    model.eval()
    encoded = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = encoded["input_ids"][:, :max_tokens]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask[:, :max_tokens]
    true_ids = input_ids.squeeze(0).detach().cpu()
    embeddings = get_input_embeddings(model)
    model.zero_grad(set_to_none=True)
    with grad_context(embeddings.weight):
        language_model_loss(model, tokenizer, input_ids, attention_mask).backward()
        apply_defense(defense)
    return ProbeContext(
        model=model,
        tokenizer=tokenizer,
        true_ids=true_ids,
        embedding_weight=embeddings.weight.detach(),
        hidden_size=get_hidden_size(model),
        top_t=int(true_ids.numel()),
    )


def finish_probe_context(context: ProbeContext) -> None:
    context.model.zero_grad(set_to_none=True)


def error_result(error: str) -> AttackResult:
    return {
        "r1": 0.0,
        "r2": 0.0,
        "rl": 0.0,
        "meteor": 0.0,
        "error": error,
    }


def matrix_embedding_probe(
    gradient: torch.Tensor,
    context: ProbeContext,
) -> AttackResult:
    matrix = gradient.detach().cpu().float()
    if matrix.dim() != 2:
        return error_result("matrix gradient is not 2-D")
    if matrix.shape[1] == context.hidden_size:
        oriented = matrix
    elif matrix.shape[0] == context.hidden_size:
        oriented = matrix.T
    else:
        return error_result(f"no hidden-size dimension in gradient shape {tuple(matrix.shape)}")
    embeddings = context.embedding_weight.detach().cpu().float()
    embeddings /= embeddings.norm(dim=1, keepdim=True) + 1e-10
    scores = (embeddings @ oriented.T).norm(dim=1)
    top_ids = scores.topk(context.top_t, largest=True).indices
    recovered = top_ids[scores[top_ids].argsort(descending=True)]
    return compute_text_metrics(recovered, context.true_ids, context.tokenizer)


def random_matrix_probe(
    gradient: torch.Tensor,
    context: ProbeContext,
) -> AttackResult:
    gradient = gradient.detach().float()
    scale = max(gradient.std().item(), 1e-8)
    return matrix_embedding_probe(torch.randn_like(gradient) * scale, context)


Probe = Callable[[ProbeContext, bool], dict[str, AttackResult]]


def run_single_probe(
    probe: Probe,
    model: nn.Module,
    tokenizer: Any,
    sentence: str,
    device: str,
    defense: GradientDefense | None = None,
    max_tokens: int = 32,
    random_baseline: bool = False,
) -> dict[str, AttackResult]:
    context = collect_probe_context(model, tokenizer, sentence, device, defense, max_tokens)
    try:
        return probe(context, random_baseline)
    finally:
        finish_probe_context(context)


__all__ = [
    "ProbeContext",
    "collect_probe_context",
    "error_result",
    "finish_probe_context",
    "matrix_embedding_probe",
    "random_matrix_probe",
    "run_single_probe",
]
