"""Architecture-aware accessors for supported Hugging Face models."""

from __future__ import annotations

from typing import Any, cast

import torch.nn as nn

from ..config import Architecture

_MASKED_LM_ARCHITECTURES = frozenset({"bert_mlm", "roberta_mlm", "deberta_mlm"})


def detect_architecture(model: nn.Module) -> Architecture:
    """Detect the logical transformer architecture family."""

    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", "") or ""
    if model_type == "gpt2" or hasattr(model, "transformer"):
        return "gpt2"
    if model_type == "opt":
        return "opt"
    if model_type == "phi":
        return "phi"
    if model_type in ("llama", "mistral"):
        return "llama"
    if model_type in ("gemma", "gemma2"):
        return "gemma"
    if model_type == "bert":
        return "bert_mlm"
    if model_type == "roberta":
        return "roberta_mlm"
    if "deberta" in model_type:
        return "deberta_mlm"
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "decoder"):
        return "opt"
    if inner is not None and hasattr(inner, "embed_tokens"):
        return "llama"
    return "gpt2"


def is_masked_lm(model: nn.Module) -> bool:
    """Return whether ``model`` is one of the supported encoder MLMs."""

    return detect_architecture(model) in _MASKED_LM_ARCHITECTURES


def unwrap_peft(model: nn.Module) -> nn.Module:
    """Strip a PEFT/LoRA wrapper and expose its Hugging Face model."""

    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "model"):
        return cast(nn.Module, base_model.model)
    return model


def get_input_embedding(model: nn.Module) -> nn.Embedding:
    """Return the input token embedding for a supported model."""

    get_embeddings = getattr(model, "get_input_embeddings", None)
    embedding = get_embeddings() if get_embeddings is not None else None
    if embedding is not None:
        return cast(nn.Embedding, embedding)

    unwrapped = unwrap_peft(model)
    architecture = detect_architecture(unwrapped)
    if architecture == "gpt2":
        return cast(nn.Embedding, unwrapped.transformer.wte)  # type: ignore[attr-defined]
    if architecture == "opt":
        return cast(nn.Embedding, unwrapped.model.decoder.embed_tokens)  # type: ignore[attr-defined]
    return cast(nn.Embedding, unwrapped.model.embed_tokens)  # type: ignore[attr-defined]


def get_transformer_blocks(model: nn.Module) -> nn.ModuleList:
    """Return the transformer block list for a supported architecture."""

    unwrapped = unwrap_peft(model)
    architecture = detect_architecture(unwrapped)
    if architecture == "gpt2":
        return cast(nn.ModuleList, unwrapped.transformer.h)  # type: ignore[attr-defined]
    if architecture == "opt":
        return cast(nn.ModuleList, unwrapped.model.decoder.layers)  # type: ignore[attr-defined]
    if architecture == "bert_mlm":
        return cast(nn.ModuleList, unwrapped.bert.encoder.layer)  # type: ignore[attr-defined]
    if architecture == "roberta_mlm":
        return cast(nn.ModuleList, unwrapped.roberta.encoder.layer)  # type: ignore[attr-defined]
    if architecture == "deberta_mlm":
        return cast(nn.ModuleList, unwrapped.deberta.encoder.layer)  # type: ignore[attr-defined]
    return cast(nn.ModuleList, unwrapped.model.layers)  # type: ignore[attr-defined]


def get_block_attention_module(block: nn.Module) -> nn.Module | None:
    """Return a transformer's self-attention submodule, if present."""

    for attribute in ("attn", "self_attn", "attention"):
        module = getattr(block, attribute, None)
        if module is not None:
            return cast(nn.Module, module)
    return None


def get_hidden_size(model: nn.Module) -> int:
    """Return the configured hidden size, preserving the 768 fallback."""

    config = model.config
    for attribute in ("n_embd", "hidden_size", "d_model"):
        value = getattr(config, attribute, None)
        if value is not None:
            return int(value)
    return 768


def get_final_layer_norm(model: nn.Module) -> nn.Module:
    """Return the final layer normalization module."""

    unwrapped: Any = unwrap_peft(model)
    architecture = detect_architecture(unwrapped)
    if architecture == "gpt2":
        return cast(nn.Module, unwrapped.transformer.ln_f)
    if architecture == "opt":
        return cast(nn.Module, unwrapped.model.decoder.final_layer_norm)
    if architecture == "phi":
        return cast(nn.Module, unwrapped.model.final_layernorm)
    if architecture == "bert_mlm":
        return cast(nn.Module, unwrapped.bert.encoder.layer[-1].output.LayerNorm)
    if architecture == "roberta_mlm":
        return cast(nn.Module, unwrapped.roberta.encoder.layer[-1].output.LayerNorm)
    if architecture == "deberta_mlm":
        return cast(nn.Module, unwrapped.deberta.encoder.layer[-1].output.LayerNorm)
    return cast(nn.Module, unwrapped.model.norm)

