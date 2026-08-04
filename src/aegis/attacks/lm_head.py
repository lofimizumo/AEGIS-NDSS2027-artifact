"""Adaptive side-channel probe for output LM-head gradients."""

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


def untie_lm_head(model: nn.Module) -> bool:
    """Clone the output head so its gradients are independent of embeddings."""
    output = model.get_output_embeddings()
    if output is None or not hasattr(output, "weight"):
        return False
    weight = output.weight.detach()
    new_head = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    new_head.weight.data.copy_(weight.float())
    new_head.to(device=weight.device, dtype=weight.dtype)
    model.set_output_embeddings(new_head)
    if hasattr(model, "config"):
        model.config.tie_word_embeddings = False
    return True


def probe_lm_head(
    context: ProbeContext, random_baseline: bool = False
) -> dict[str, dict[str, Any]]:
    output_head = context.model.get_output_embeddings()
    gradient = (
        output_head.weight.grad
        if output_head is not None and hasattr(output_head, "weight")
        else None
    )
    if gradient is None:
        result = error_result("no LM-head gradient")
    elif gradient.dim() != 2:
        result = error_result("LM-head gradient is not 2-D")
    else:
        scores = gradient.detach().cpu().float().norm(dim=1)
        top_ids = scores.topk(context.top_t).indices
        recovered = top_ids[scores[top_ids].argsort(descending=True)]
        result = compute_text_metrics(recovered, context.true_ids, context.tokenizer)
    output = {"lm_head": result}
    if random_baseline:
        if gradient is None:
            null_result = error_result("no LM-head gradient")
        else:
            scores = torch.rand(gradient.shape[0])
            top_ids = scores.topk(context.top_t).indices
            recovered = top_ids[scores[top_ids].argsort(descending=True)]
            null_result = compute_text_metrics(recovered, context.true_ids, context.tokenizer)
        output["lm_head_null"] = null_result
    return output


def run_lm_head(
    model: nn.Module,
    tokenizer: Any,
    sentence: str,
    device: str,
    defense: GradientDefense | None = None,
    max_tokens: int = 32,
    random_baseline: bool = False,
):
    return run_single_probe(
        probe_lm_head,
        model,
        tokenizer,
        sentence,
        device,
        defense,
        max_tokens,
        random_baseline,
    )


__all__ = ["probe_lm_head", "run_lm_head", "untie_lm_head"]
