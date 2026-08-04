"""SOMP B=1 subspace-guided token recovery."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .dager import DagerArgs, DagerModelWrapper, filter_l1
from .helpers import (
    AttackResult,
    GradientDefense,
    apply_defense,
    compute_text_metrics,
    detect_architecture,
    get_input_embeddings,
    unwrap_peft_model,
)


def run_somp(
    model: nn.Module,
    tokenizer: Any,
    sentence: str,
    device: str,
    defense: GradientDefense | None = None,
    beam_width: int = 12,
    max_len: int = 32,
) -> AttackResult:
    """Run DAGER-style pooling followed by LM-guided beam search."""
    model.eval()
    base_model = unwrap_peft_model(model)
    architecture = detect_architecture(base_model)
    input_ids = tokenizer(sentence, return_tensors="pt").to(device)["input_ids"]
    input_ids = input_ids[:, :max_len]
    length = input_ids.shape[1]
    true_ids = input_ids.squeeze(0).cpu()
    was_bf16 = next(base_model.parameters()).dtype == torch.bfloat16
    if was_bf16:
        base_model.float()
    try:
        return _run_somp(
            model,
            base_model,
            tokenizer,
            input_ids,
            true_ids,
            length,
            device,
            architecture,
            defense,
            beam_width,
        )
    finally:
        if was_bf16:
            base_model.bfloat16()


def _run_somp(
    model: nn.Module,
    base_model: nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    true_ids: torch.Tensor,
    length: int,
    device: str,
    architecture: str,
    defense: GradientDefense | None,
    beam_width: int,
) -> AttackResult:
    q_suffixes = (
        "attn.c_attn.weight",
        "self_attn.q_proj.weight",
        "attention.self.query.weight",
    )
    q_parameters = [
        parameter
        for name, parameter in base_model.named_parameters()
        if any(name.endswith(suffix) for suffix in q_suffixes)
    ]
    embeddings = get_input_embeddings(base_model)
    parameters = q_parameters + [embeddings.weight]
    states = {id(parameter): parameter.requires_grad for parameter in parameters}
    for parameter in parameters:
        parameter.requires_grad_(True)
    try:
        base_model.zero_grad()
        model(input_ids=input_ids, labels=input_ids).loss.backward()
        apply_defense(defense)
        gradients = {
            name: (
                parameter.grad.detach().cpu().float().clone()
                if parameter.grad is not None
                else None
            )
            for name, parameter in base_model.named_parameters()
            if any(name.endswith(suffix) for suffix in q_suffixes)
        }
        embedding_gradient = (
            embeddings.weight.grad.detach().clone()
            if embeddings.weight.grad is not None
            else torch.zeros_like(embeddings.weight)
        )
        base_model.zero_grad()
    finally:
        for parameter in parameters:
            parameter.requires_grad_(states[id(parameter)])

    args = DagerArgs(device, architecture)
    wrapper = DagerModelWrapper(base_model, tokenizer, architecture, device, args)
    _, spans = wrapper.get_matrices_expansions(gradients, min_B=length)
    if len(spans) < 1:
        top_ids = (
            embedding_gradient.norm(dim=1)
            .topk(min(500, embedding_gradient.shape[0]))
            .indices.tolist()
        )
        position_candidates = [top_ids] * length
    else:
        _, candidate_ids, _, _ = filter_l1(args, wrapper, spans, sequence_length=length)
        position_candidates = (
            candidate_ids
            if len(candidate_ids) == length
            else [
                candidate_ids[position] if position < len(candidate_ids) else []
                for position in range(length)
            ]
        )

    pool = {token for candidates in position_candidates for token in candidates}
    bos = tokenizer.bos_token_id or tokenizer.eos_token_id or 0
    beams: list[tuple[list[int], float]] = [([bos], 0.0)]
    with torch.no_grad():
        for position in range(length - 1):
            candidates = (
                position_candidates[position + 1]
                if position + 1 < len(position_candidates)
                else list(pool)
            )
            if not candidates:
                candidates = list(pool) or [bos]
            candidate_tensor = torch.tensor(candidates, device=device)
            extensions: list[tuple[list[int], float]] = []
            for sequence, log_probability in beams:
                context = torch.tensor([sequence], device=device)
                log_probs = functional.log_softmax(
                    model(input_ids=context).logits[:, -1, :], dim=-1
                ).squeeze(0)
                local = log_probs[candidate_tensor]
                count = min(beam_width * 2, len(candidates))
                for local_index in local.topk(count).indices.tolist():
                    token = candidates[local_index]
                    extensions.append(
                        (
                            sequence + [token],
                            log_probability + log_probs[token].item(),
                        )
                    )
            extensions.sort(key=lambda item: -item[1])
            beams = extensions[:beam_width]

    best = beams[0][0] if beams else [bos] * length
    if len(best) < length:
        best += [tokenizer.eos_token_id or 0] * (length - len(best))
    recovered = torch.tensor(best[:length], dtype=torch.long)
    result = compute_text_metrics(recovered, true_ids, tokenizer)
    result["attack"] = "somp"
    return result


__all__ = ["run_somp"]
