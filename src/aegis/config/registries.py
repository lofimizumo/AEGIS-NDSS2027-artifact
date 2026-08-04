"""Portable model and dataset registries for AEGIS studies."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypeAlias

Architecture: TypeAlias = Literal[
    "gpt2",
    "bert_mlm",
    "roberta_mlm",
    "deberta_mlm",
    "llama",
    "gemma",
    "phi",
    "opt",
]
DatasetType: TypeAlias = Literal["classification", "generation"]

_PORTABLE_MODEL_PATHS: dict[str, str] = {
    "tiny_gpt2": "sshleifer/tiny-gpt2",
    "gpt2": "gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
    "gpt2-large": "gpt2-large",
    "gpt2-xl": "openai-community/gpt2-xl",
    "bert-base": "bert-base-uncased",
    "bert-large": "bert-large-uncased",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "deberta-v3-small": "microsoft/deberta-v3-small",
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "deberta-v3-large": "microsoft/deberta-v3-large",
    "llama-2-7b": "unsloth/Llama-2-7b",
    "llama-2-13b": "unsloth/Llama-2-13b",
    "llama-3.1-8b": "unsloth/Llama-3.1-8B",
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-9b": "google/gemma-2-9b",
    "phi-2": "microsoft/phi-2",
    "opt-1.3b": "facebook/opt-1.3b",
}


def build_model_paths(
    llm_dir: str | Path | None = None,
    *,
    overrides: Mapping[str, str | Path] | None = None,
) -> dict[str, str]:
    """Build the model registry without binding it to a workspace.

    Passing ``llm_dir`` restores local paths for tiny GPT-2
    (``tiny_gpt2``) and GPT-2 Medium (``gpt2_medium_weights``). Missing local
    directories are handled by :func:`aegis.models.load_model` using the
    corresponding Hugging Face fallback.
    """

    paths = dict(_PORTABLE_MODEL_PATHS)
    if llm_dir is not None:
        root = Path(llm_dir).expanduser()
        paths["tiny_gpt2"] = str(root / "tiny_gpt2")
        paths["gpt2-medium"] = str(root / "gpt2_medium_weights")
    if overrides:
        paths.update({name: str(path) for name, path in overrides.items()})
    return paths


MODEL_PATHS: dict[str, str] = build_model_paths()

MODEL_HF_FALLBACKS: dict[str, str] = {
    "tiny_gpt2": "sshleifer/tiny-gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
}

MODEL_ARCH: dict[str, Architecture] = {
    "tiny_gpt2": "gpt2",
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2",
    "gpt2-large": "gpt2",
    "gpt2-xl": "gpt2",
    "bert-base": "bert_mlm",
    "bert-large": "bert_mlm",
    "roberta-base": "roberta_mlm",
    "roberta-large": "roberta_mlm",
    "deberta-v3-small": "deberta_mlm",
    "deberta-v3-base": "deberta_mlm",
    "deberta-v3-large": "deberta_mlm",
    "llama-2-7b": "llama",
    "llama-2-13b": "llama",
    "llama-3.1-8b": "llama",
    "gemma-2-2b": "gemma",
    "gemma-2-9b": "gemma",
    "phi-2": "phi",
    "opt-1.3b": "opt",
}

MODEL_USE_LORA: set[str] = set()

MODEL_USE_BF16: set[str] = {
    "llama-2-7b",
    "llama-2-13b",
    "llama-3.1-8b",
    "gemma-2-9b",
    "phi-2",
    "opt-1.3b",
}

MODEL_BATCH_SIZE: dict[str, int] = {
    "gpt2-xl": 2,
    "bert-large": 4,
    "roberta-large": 4,
    "deberta-v3-base": 4,
    "deberta-v3-large": 2,
    "llama-2-7b": 4,
    "llama-2-13b": 2,
    "llama-3.1-8b": 4,
    "gemma-2-9b": 2,
    "phi-2": 4,
    "opt-1.3b": 8,
}

MODEL_N_ATTACK: dict[str, int] = {
    "llama-2-13b": 5,
}

LORA_TARGETS: dict[Architecture, list[str]] = {
    "llama": ["q_proj", "v_proj"],
}

DATASET_TYPE: dict[str, DatasetType] = {
    "rotten_tomatoes": "classification",
    "emotion": "classification",
    "financial_phrasebank": "classification",
    "wikitext2": "generation",
    "dialogsum": "generation",
    "cnn_dailymail": "generation",
}

DATASET_LABEL_NAMES: dict[str, list[str]] = {
    "rotten_tomatoes": ["negative", "positive"],
    "emotion": ["sadness", "joy", "love", "anger", "fear", "surprise"],
    "financial_phrasebank": ["negative", "neutral", "positive"],
}

def supports_dager_attack(model_key: str) -> bool:
    """Return whether DAGER is supported for a registered model architecture."""

    return MODEL_ARCH.get(model_key) not in ("roberta_mlm", "deberta_mlm")
