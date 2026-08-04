"""MLP-gradient subspace attack from the adaptive-attack experiment."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .helpers import (
    AttackResult,
    GradientDefense,
    apply_defense,
    compute_text_metrics,
    detect_architecture,
    get_hidden_size,
    get_input_embeddings,
    get_transformer_blocks,
    grad_context,
    language_model_loss,
)


def _get_expansion_layer(model: nn.Module, block_index: int) -> nn.Module | None:
    blocks = get_transformer_blocks(model)
    if block_index >= len(blocks):
        return None
    block = blocks[block_index]
    architecture = detect_architecture(model)
    if architecture == "gpt2":
        return getattr(block.mlp, "c_fc", None)
    if architecture in ("llama", "gemma"):
        mlp = getattr(block, "mlp", None)
        return getattr(mlp, "gate_proj", getattr(mlp, "up_proj", None)) if mlp is not None else None
    if architecture in ("bert_mlm", "roberta_mlm"):
        return getattr(block.intermediate, "dense", None)
    return None


def _embedding_fallback(
    model: nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    true_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    defense: GradientDefense | None,
) -> AttackResult:
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
    result["attack"] = "mlp_svd"
    result["mlp_svd_fallback"] = "embed_norm"
    return result


def run_mlp_svd(
    model: nn.Module,
    tokenizer: Any,
    sentence: str,
    device: str,
    defense: GradientDefense | None = None,
    n_sv: int = 10,
    max_tokens: int = 32,
) -> AttackResult:
    """Score vocabulary embeddings against the first MLP weight gradient.

    ``n_sv`` is retained for compatibility with the experiment interface; the
    active implementation scores with the full gradient projection.
    """
    del n_sv
    model.eval()
    encoded = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = encoded["input_ids"][:, :max_tokens]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask[:, :max_tokens]
    true_ids = input_ids.squeeze(0).cpu()
    expansion = _get_expansion_layer(model, 0)
    if expansion is None or getattr(expansion, "weight", None) is None:
        return _embedding_fallback(
            model,
            tokenizer,
            input_ids,
            true_ids,
            attention_mask,
            defense,
        )

    embeddings = get_input_embeddings(model)
    model.zero_grad()
    with grad_context(embeddings.weight):
        language_model_loss(model, tokenizer, input_ids, attention_mask).backward()
        apply_defense(defense)
        gradient = (
            expansion.weight.grad.detach().cpu().float().clone()
            if expansion.weight.grad is not None
            else None
        )
    model.zero_grad()
    if gradient is None or gradient.norm().item() < 1e-10:
        return _embedding_fallback(
            model,
            tokenizer,
            input_ids,
            true_ids,
            attention_mask,
            defense,
        )

    hidden_size = get_hidden_size(model)
    if gradient.shape[0] == hidden_size and gradient.shape[1] != hidden_size:
        gradient = gradient.T
    table = embeddings.weight.detach().cpu().float()
    normalized = table / (table.norm(dim=1, keepdim=True) + 1e-10)
    scores = (normalized @ gradient.T).norm(dim=1)
    top_ids = scores.topk(input_ids.shape[1], largest=True).indices
    recovered = top_ids[scores[top_ids].argsort(descending=True)]
    result = compute_text_metrics(recovered, true_ids, tokenizer)
    result["attack"] = "mlp_svd"
    return result


__all__ = ["run_mlp_svd"]
