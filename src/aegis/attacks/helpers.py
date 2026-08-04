"""Shared utilities for gradient-inversion attacks."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any, Protocol

import torch
import torch.nn as nn

from aegis.evaluation import apply_mlm_labels
from aegis.models.architecture import (
    detect_architecture,
    get_final_layer_norm,
    get_hidden_size,
    get_input_embedding,
    get_transformer_blocks,
    is_masked_lm,
    unwrap_peft,
)

AttackResult = dict[str, Any]
get_input_embeddings = get_input_embedding
unwrap_peft_model = unwrap_peft


class GradientDefense(Protocol):
    """Structural interface used by attacks after back-propagation."""

    def flood_gradients(self, model: nn.Module | None = None) -> None: ...


@contextmanager
def enable_grad_temporarily(*tensors: torch.Tensor) -> Iterator[None]:
    states = [tensor.requires_grad for tensor in tensors]
    for tensor in tensors:
        tensor.requires_grad_(True)
    try:
        yield
    finally:
        for tensor, state in zip(tensors, states, strict=True):
            tensor.requires_grad_(state)


def grad_context(tensor: torch.Tensor):
    return enable_grad_temporarily(tensor) if not tensor.requires_grad else nullcontext()


def apply_defense(
    defense: GradientDefense | None,
    model: nn.Module | None = None,
) -> None:
    if defense is None:
        return
    if model is None:
        defense.flood_gradients()
    else:
        defense.flood_gradients(model)


def is_aegis_defense(defense: GradientDefense | None) -> bool:
    """Identify AEGIS-family defenses without importing the defense package."""
    if defense is None:
        return False
    class_name = type(defense).__name__.lower()
    return "aegis" in class_name


def empty_result(attack: str, error: str | None = None) -> AttackResult:
    result: AttackResult = {
        "r1": 0.0,
        "r2": 0.0,
        "rl": 0.0,
        "meteor": 0.0,
        "rec_text": "",
        "ref_text": "",
        "attack": attack,
    }
    if error is not None:
        result["error"] = error
    return result


def compute_text_metrics(
    recovered_ids: torch.Tensor,
    true_ids: torch.Tensor,
    tokenizer: Any,
) -> AttackResult:
    """Compute the standard ROUGE and METEOR result fields."""
    rec_text = tokenizer.decode(recovered_ids.tolist(), skip_special_tokens=True).strip()
    ref_text = tokenizer.decode(true_ids.tolist(), skip_special_tokens=True).strip()
    r1 = r2 = rl = meteor = 0.0

    if rec_text and ref_text:
        try:
            from rouge_score import rouge_scorer

            scores = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=False
            ).score(ref_text, rec_text)
            r1 = round(scores["rouge1"].fmeasure, 4)
            r2 = round(scores["rouge2"].fmeasure, 4)
            rl = round(scores["rougeL"].fmeasure, 4)
        except ImportError:
            pass

        try:
            from nltk.translate.meteor_score import meteor_score

            meteor = round(meteor_score([ref_text.split()], rec_text.split()), 4)
        except (ImportError, LookupError, TypeError, ValueError):
            pass

    return {
        "r1": r1,
        "r2": r2,
        "rl": rl,
        "meteor": meteor,
        "rec_text": rec_text,
        "ref_text": ref_text,
    }


def language_model_loss(
    model: nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
):
    if is_masked_lm(model):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        masked_input, labels = apply_mlm_labels(input_ids, attention_mask, tokenizer)
        return model(
            input_ids=masked_input,
            attention_mask=attention_mask,
            labels=labels,
        ).loss
    labels = (
        torch.where(
            attention_mask.bool(),
            input_ids,
            torch.full_like(input_ids, -100),
        )
        if attention_mask is not None
        else input_ids
    )
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    ).loss


def get_last_block_parameters(model: nn.Module, n: int = 3) -> list[nn.Parameter]:
    blocks = get_transformer_blocks(model)
    params = [
        parameter
        for block in blocks[-n:]
        for parameter in block.parameters()
        if parameter.requires_grad
    ]
    params.extend(
        parameter
        for parameter in get_final_layer_norm(model).parameters()
        if parameter.requires_grad
    )
    return params


try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def math_attention_context():
        return sdpa_kernel(SDPBackend.MATH)

except ImportError:

    def math_attention_context():
        if hasattr(torch.backends, "cuda"):
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
            )
        return nullcontext()


__all__ = [
    "AttackResult",
    "GradientDefense",
    "apply_defense",
    "apply_mlm_labels",
    "compute_text_metrics",
    "detect_architecture",
    "empty_result",
    "enable_grad_temporarily",
    "get_hidden_size",
    "get_input_embeddings",
    "get_last_block_parameters",
    "get_transformer_blocks",
    "grad_context",
    "is_aegis_defense",
    "is_masked_lm",
    "language_model_loss",
    "math_attention_context",
    "unwrap_peft_model",
]
