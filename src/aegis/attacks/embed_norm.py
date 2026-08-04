"""Embedding-gradient row-norm attack."""

from __future__ import annotations

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
    language_model_loss,
)


def run_embed_norm(
    model: nn.Module,
    tokenizer: Any,
    sentence: str,
    device: str,
    defense: GradientDefense | None = None,
    max_tokens: int = 32,
) -> AttackResult:
    """Recover tokens from the largest input-embedding gradient rows."""
    model.eval()
    encoded = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = encoded["input_ids"][:, :max_tokens]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask[:, :max_tokens]
    true_ids = input_ids.squeeze(0).cpu()
    embeddings = get_input_embeddings(model)

    model.zero_grad()
    with grad_context(embeddings.weight):
        language_model_loss(model, tokenizer, input_ids, attention_mask).backward()
        apply_defense(defense)
        gradient = (
            embeddings.weight.grad.detach().cpu().float().clone()
            if embeddings.weight.grad is not None
            else torch.zeros_like(embeddings.weight.detach().cpu().float())
        )
    model.zero_grad()
    recovered = gradient.norm(dim=1).topk(input_ids.shape[1]).indices
    result = compute_text_metrics(recovered, true_ids, tokenizer)
    result["attack"] = "embed_norm"
    return result


__all__ = ["run_embed_norm"]
