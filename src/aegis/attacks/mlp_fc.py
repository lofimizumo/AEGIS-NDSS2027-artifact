"""Adaptive side-channel probe for the first MLP expansion gradient."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .helpers import GradientDefense, detect_architecture, get_transformer_blocks
from .side_channel_common import (
    ProbeContext,
    error_result,
    matrix_embedding_probe,
    random_matrix_probe,
    run_single_probe,
)


def _parameter(model: nn.Module) -> torch.nn.Parameter | None:
    block = get_transformer_blocks(model)[0]
    architecture = detect_architecture(model)
    if architecture == "gpt2":
        module = getattr(block.mlp, "c_fc", None)
    elif architecture in ("llama", "gemma"):
        mlp = getattr(block, "mlp", None)
        module = getattr(mlp, "gate_proj", getattr(mlp, "up_proj", None))
    elif architecture in ("bert_mlm", "roberta_mlm", "deberta_mlm"):
        module = getattr(block.intermediate, "dense", None)
    elif architecture == "opt":
        module = getattr(block, "fc1", None)
    else:
        module = None
    weight = getattr(module, "weight", None)
    return weight if isinstance(weight, torch.nn.Parameter) else None


def probe_mlp_fc(context: ProbeContext, random_baseline: bool = False) -> dict[str, dict[str, Any]]:
    parameter = _parameter(context.model)
    result = (
        matrix_embedding_probe(parameter.grad, context)
        if parameter is not None and parameter.grad is not None
        else error_result("no mlp_fc gradient")
    )
    output = {"mlp_fc": result}
    if random_baseline:
        output["mlp_fc_null"] = (
            random_matrix_probe(parameter.grad, context)
            if parameter is not None and parameter.grad is not None
            else error_result("no mlp_fc gradient")
        )
    return output


def run_mlp_fc(
    model: nn.Module,
    tokenizer: Any,
    sentence: str,
    device: str,
    defense: GradientDefense | None = None,
    max_tokens: int = 32,
    random_baseline: bool = False,
):
    return run_single_probe(
        probe_mlp_fc,
        model,
        tokenizer,
        sentence,
        device,
        defense,
        max_tokens,
        random_baseline,
    )


__all__ = ["probe_mlp_fc", "run_mlp_fc"]
