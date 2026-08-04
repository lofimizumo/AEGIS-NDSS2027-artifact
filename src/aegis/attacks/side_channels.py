"""Grouped execution of the adaptive side-channel attacks."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .helpers import AttackResult, GradientDefense
from .layernorm import probe_layernorm
from .lm_head import probe_lm_head
from .mlp_fc import probe_mlp_fc
from .mlp_proj import probe_mlp_proj
from .side_channel_common import (
    collect_probe_context,
    finish_probe_context,
)

CHANNELS = ("mlp_fc", "mlp_proj", "layernorm", "lm_head")
_PROBES = {
    "mlp_fc": probe_mlp_fc,
    "mlp_proj": probe_mlp_proj,
    "layernorm": probe_layernorm,
    "lm_head": probe_lm_head,
}


def run_sentence_probes(
    model: nn.Module,
    tokenizer: Any,
    sentence: str,
    defense: GradientDefense | None,
    channels: list[str],
    max_tokens: int,
    random_baseline: bool,
    device: str | None = None,
) -> dict[str, AttackResult]:
    """Run selected probes from one shared victim backward pass."""
    unknown = set(channels) - set(CHANNELS)
    if unknown:
        raise ValueError(f"unknown side-channel probes: {sorted(unknown)}")
    if device is None:
        device = str(next(model.parameters()).device)
    context = collect_probe_context(model, tokenizer, sentence, device, defense, max_tokens)
    try:
        results: dict[str, AttackResult] = {}
        for channel in channels:
            results.update(_PROBES[channel](context, random_baseline))
        return results
    finally:
        finish_probe_context(context)


__all__ = ["CHANNELS", "run_sentence_probes"]
