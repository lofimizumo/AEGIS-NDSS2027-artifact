"""Adaptive side-channel probe for LayerNorm scale gradients."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .helpers import GradientDefense, compute_text_metrics
from .side_channel_common import (
    ProbeContext,
    error_result,
    run_single_probe,
)


def _probe(gradients: list[torch.Tensor], context: ProbeContext) -> dict[str, Any]:
    usable = [
        gradient.detach().cpu().float().flatten() for gradient in gradients if gradient is not None
    ]
    usable = [
        gradient
        for gradient in usable
        if gradient.numel() == context.embedding_weight.shape[1] and gradient.norm().item() > 1e-10
    ]
    if not usable:
        return error_result("no hidden-size LayerNorm gradients")
    directions = torch.stack([gradient / (gradient.norm() + 1e-10) for gradient in usable])
    embeddings = context.embedding_weight.detach().cpu().float()
    embeddings /= embeddings.norm(dim=1, keepdim=True) + 1e-10
    scores = (embeddings @ directions.T).norm(dim=1)
    top_ids = scores.topk(context.top_t).indices
    recovered = top_ids[scores[top_ids].argsort(descending=True)]
    return compute_text_metrics(recovered, context.true_ids, context.tokenizer)


def probe_layernorm(
    context: ProbeContext, random_baseline: bool = False
) -> dict[str, dict[str, Any]]:
    gradients = [
        module.weight.grad
        for module in context.model.modules()
        if isinstance(module, nn.LayerNorm)
        and module.weight is not None
        and module.weight.grad is not None
    ]
    output = {"layernorm": _probe(gradients, context)}
    if random_baseline:
        random_gradients = [
            torch.randn_like(gradient) * max(gradient.detach().float().std().item(), 1e-8)
            for gradient in gradients
        ]
        output["layernorm_null"] = _probe(random_gradients, context)
    return output


def run_layernorm(
    model: nn.Module,
    tokenizer: Any,
    sentence: str,
    device: str,
    defense: GradientDefense | None = None,
    max_tokens: int = 32,
    random_baseline: bool = False,
):
    return run_single_probe(
        probe_layernorm,
        model,
        tokenizer,
        sentence,
        device,
        defense,
        max_tokens,
        random_baseline,
    )


__all__ = ["probe_layernorm", "run_layernorm"]
