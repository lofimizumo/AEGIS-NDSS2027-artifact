"""Utility metrics for AEGIS studies."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from ..models.architecture import is_masked_lm


def apply_mlm_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
    mask_prob: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Hugging Face-style 80/10/10 MLM masking."""

    device = input_ids.device
    labels = input_ids.clone()
    masked_input = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mask_prob, device=device)
    probability_matrix.masked_fill_(attention_mask == 0, 0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("MLM requires tokenizer.mask_token_id")

    replaced_indices = (
        torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool() & masked_indices
    )
    masked_input[replaced_indices] = mask_token_id

    random_indices = (
        torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool()
        & masked_indices
        & ~replaced_indices
    )
    if random_indices.any():
        random_count = int(random_indices.sum().item())
        masked_input[random_indices] = torch.randint(
            0,
            len(tokenizer),
            (random_count,),
            dtype=torch.long,
            device=device,
        )

    return masked_input, labels


def eval_loss_ppl(
    model: nn.Module,
    tokenizer: Any,
    sentences: Sequence[str],
    device: str | torch.device,
) -> tuple[float, float]:
    """Return mean held-out cross-entropy and perplexity."""

    model.eval()
    total_loss = 0.0
    evaluated_count = 0
    with torch.no_grad():
        for sentence in sentences:
            encoding = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=64,
            ).to(device)
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            if input_ids.shape[1] < 2:
                continue
            if is_masked_lm(model):
                masked_input, labels = apply_mlm_labels(
                    input_ids,
                    attention_mask,
                    tokenizer,
                )
                if (labels == -100).all():
                    continue
                output = model(
                    input_ids=masked_input,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            else:
                output = model(input_ids=input_ids, labels=input_ids)

            loss = output.loss.item()
            if math.isnan(loss):
                continue
            total_loss += loss
            evaluated_count += 1

    if evaluated_count == 0:
        return float("inf"), float("inf")
    mean_loss = total_loss / evaluated_count
    return round(mean_loss, 4), round(math.exp(mean_loss), 2)


def eval_classification_utility(
    model: nn.Module,
    tokenizer: Any,
    sentences: Sequence[str],
    labels: Sequence[int],
    label_names: Sequence[str],
    device: str | torch.device,
) -> tuple[float, float]:
    """Return label-token accuracy and macro-F1 for a causal LM."""

    model.eval()
    predictions: list[int] = []
    label_token_ids: list[int] = []
    for label_name in label_names:
        tokens = tokenizer.encode(f" {label_name}", add_special_tokens=False)
        label_token_ids.append(tokens[0] if tokens else 0)

    with torch.no_grad():
        for sentence in sentences:
            input_ids = tokenizer(sentence, return_tensors="pt")["input_ids"].to(device)
            logits = model(input_ids=input_ids).logits[0, -1]
            scores = [logits[token_id].item() for token_id in label_token_ids]
            predictions.append(int(scores.index(max(scores))))

    class_count = len(label_names)
    correct = sum(
        prediction == target for prediction, target in zip(predictions, labels, strict=True)
    )
    accuracy = correct / len(labels) if labels else 0.0

    f1_scores: list[float] = []
    for class_index in range(class_count):
        true_positive = sum(
            prediction == class_index and target == class_index
            for prediction, target in zip(predictions, labels, strict=True)
        )
        false_positive = sum(
            prediction == class_index and target != class_index
            for prediction, target in zip(predictions, labels, strict=True)
        )
        false_negative = sum(
            prediction != class_index and target == class_index
            for prediction, target in zip(predictions, labels, strict=True)
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )
        f1_scores.append(
            2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        )

    macro_f1 = sum(f1_scores) / class_count if class_count > 0 else 0.0
    return round(accuracy, 4), round(macro_f1, 4)


__all__ = [
    "apply_mlm_labels",
    "eval_classification_utility",
    "eval_loss_ppl",
]
