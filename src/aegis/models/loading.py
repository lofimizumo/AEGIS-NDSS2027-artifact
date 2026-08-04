"""Architecture-aware model loading for AEGIS studies."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..config import (
    LORA_TARGETS,
    MODEL_ARCH,
    MODEL_BATCH_SIZE,
    MODEL_HF_FALLBACKS,
    MODEL_PATHS,
    MODEL_USE_BF16,
    MODEL_USE_LORA,
)
from .architecture import unwrap_peft

_EXPECTED_LOCAL_DIRECTORY = {
    "tiny_gpt2": "tiny_gpt2",
    "gpt2-medium": "gpt2_medium_weights",
}


def load_attack_compatible_gpt2(model_dir: str | Path) -> nn.Module:
    """Load a GPT-2 language model with eager attention.

    Eager attention preserves the higher-order derivatives required by the
    gradient-inversion attacks and replaces the recovered legacy loader.
    """

    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(
        str(model_dir),
        attn_implementation="eager",
    )
    model.eval()
    return model


def _resolve_model_path(
    model_name: str,
    model_paths: Mapping[str, str | Path],
    model_fallbacks: Mapping[str, str],
) -> str:
    path = str(model_paths[model_name])
    fallback = model_fallbacks.get(model_name)
    expected_directory = _EXPECTED_LOCAL_DIRECTORY.get(model_name)
    is_expected_local_path = (
        expected_directory is not None
        and Path(path).name == expected_directory
        and path != fallback
    )
    if fallback is not None and is_expected_local_path and not Path(path).expanduser().exists():
        print(f"  [INFO] local path not found ({path}), using HF: {fallback}")
        return fallback
    return path


def load_model(
    model_name: str,
    *,
    model_paths: Mapping[str, str | Path] = MODEL_PATHS,
    model_fallbacks: Mapping[str, str] = MODEL_HF_FALLBACKS,
) -> nn.Module:
    """Load a registered model, applying the configured dtype and LoRA policy.

    ``model_paths`` makes local model roots or alternate Hub IDs injectable.
    Known missing local GPT-2 paths fall back to their portable Hub IDs.
    """

    from transformers import AutoModelForCausalLM

    path = _resolve_model_path(model_name, model_paths, model_fallbacks)
    architecture = MODEL_ARCH[model_name]
    dtype = torch.bfloat16 if model_name in MODEL_USE_BF16 else torch.float32

    if model_name in MODEL_USE_LORA:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as error:
            raise RuntimeError("peft not installed: pip install peft") from error

        base = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(dtype)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=LORA_TARGETS.get(architecture, ["q_proj", "v_proj"]),
            bias="none",
        )
        model = get_peft_model(base, lora_config)
        print(f"  [LoRA] {model_name}: ", end="")
        model.print_trainable_parameters()
        return model

    if architecture in ("bert_mlm", "roberta_mlm", "deberta_mlm"):
        from transformers import AutoModelForMaskedLM

        return AutoModelForMaskedLM.from_pretrained(
            path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    if architecture == "gpt2":
        if model_name == "gpt2-xl":
            return AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
        return load_attack_compatible_gpt2(path)

    use_safetensors = architecture != "opt"
    return AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        trust_remote_code=True,
        use_safetensors=use_safetensors,
    ).to(dtype)


def warmup_bert_sequence_classifier(
    mlm_model: nn.Module,
    tokenizer: Any,
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    num_labels: int,
    device: str | torch.device,
    model_name: str,
    epochs: int = 2,
    lr: float = 2e-4,
    *,
    model_paths: Mapping[str, str | Path] = MODEL_PATHS,
) -> nn.Module:
    """Build and warm a BERT sequence head for the experiment's attack model."""

    from transformers import BertForSequenceClassification

    mlm: Any = unwrap_peft(mlm_model)
    if not hasattr(mlm, "bert"):
        raise TypeError("warmup_bert_sequence_classifier: expected BERT MLM")

    config = deepcopy(mlm.config)
    config.num_labels = int(num_labels)
    sequence_model = BertForSequenceClassification(config)
    sequence_model.bert.load_state_dict(mlm.bert.state_dict(), strict=False)

    if getattr(sequence_model.bert, "pooler", None) is not None:
        pretrained_id = getattr(mlm.config, "_name_or_path", None) or model_paths.get(model_name)
        if pretrained_id:
            try:
                from transformers import BertModel

                reference_bert = BertModel.from_pretrained(
                    str(pretrained_id),
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
                if getattr(reference_bert, "pooler", None) is not None:
                    sequence_model.bert.pooler.load_state_dict(
                        reference_bert.pooler.state_dict(),
                        strict=True,
                    )
                del reference_bert
            except Exception as error:
                print(
                    "    [BERT seq_class] pretrained pooler copy failed "
                    f"({pretrained_id}): {error}",
                    flush=True,
                )

    sequence_model.to(device)
    for parameter in sequence_model.bert.embeddings.parameters():
        parameter.requires_grad = False
    for parameter in sequence_model.bert.encoder.parameters():
        parameter.requires_grad = False

    pooler_parameters: list[nn.Parameter] = []
    if sequence_model.bert.pooler is not None:
        for parameter in sequence_model.bert.pooler.parameters():
            parameter.requires_grad = True
            pooler_parameters.append(parameter)
    for parameter in sequence_model.classifier.parameters():
        parameter.requires_grad = True

    sequence_model.train()
    optimizer = torch.optim.AdamW(
        pooler_parameters + list(sequence_model.classifier.parameters()),
        lr=lr,
        weight_decay=0.01,
    )
    batch_size = MODEL_BATCH_SIZE.get(model_name, 8)
    sample_count = len(train_texts)
    print(
        f"    [BERT seq_class] head warm-up: {epochs} epoch(s), "
        f"n={sample_count}, batch={batch_size} (CE / pooler+classifier, encoder frozen)",
        flush=True,
    )
    for _ in range(epochs):
        order = list(range(sample_count))
        random.shuffle(order)
        for start in range(0, sample_count, batch_size):
            indices = order[start : start + batch_size]
            texts = [train_texts[index] for index in indices]
            labels = torch.tensor(
                [train_labels[index] for index in indices],
                device=device,
                dtype=torch.long,
            )
            encoding = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoding = {key: value.to(device) for key, value in encoding.items()}
            optimizer.zero_grad(set_to_none=True)
            output = sequence_model(**encoding, labels=labels)
            output.loss.backward()
            optimizer.step()

    sequence_model.eval()
    for parameter in sequence_model.bert.parameters():
        parameter.requires_grad = True
    for parameter in sequence_model.classifier.parameters():
        parameter.requires_grad = True
    return sequence_model
