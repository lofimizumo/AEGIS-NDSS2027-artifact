"""Dataset sampling and splitting for AEGIS studies."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypedDict


class DatasetSplits(TypedDict):
    """Text and optional labels for the experiment's train/test split."""

    train: list[str]
    test: list[str]
    train_labels: list[int] | None
    test_labels: list[int] | None


DatasetLoader = Callable[..., Iterable[Mapping[str, Any]]]


def _word_count(text: str) -> int:
    return len(text.split())


def load_dataset_sentences(
    name: str,
    n_total: int,
    max_words: int = 30,
    seed: int = 42,
    *,
    dataset_loader: DatasetLoader | None = None,
) -> DatasetSplits:
    """Download, sample, and return the configured text splits.

    ``dataset_loader`` defaults to ``datasets.load_dataset`` and can be
    injected for alternate caches, mirrors, or offline dataset providers.
    """

    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset

    rng = random.Random(seed)
    sentences: list[str] = []
    labels: list[int] = []

    if name == "rotten_tomatoes":
        dataset = dataset_loader("rotten_tomatoes", split="train")
        by_class: dict[int, list[tuple[str, int]]] = {0: [], 1: []}
        for example in dataset:
            label = int(example["label"])
            by_class[label].append((str(example["text"]), label))
        per_class = n_total // 2
        for class_items in by_class.values():
            rng.shuffle(class_items)
            for text, label in class_items[:per_class]:
                sentences.append(text)
                labels.append(label)

    elif name == "emotion":
        dataset = dataset_loader("dair-ai/emotion", split="train")
        by_class = {index: [] for index in range(6)}
        for example in dataset:
            label = int(example["label"])
            by_class[label].append((str(example["text"]), label))
        per_class = n_total // 6
        for class_items in by_class.values():
            rng.shuffle(class_items)
            for text, label in class_items[:per_class]:
                sentences.append(text)
                labels.append(label)

    elif name == "financial_phrasebank":
        dataset = dataset_loader(
            "takala/financial_phrasebank",
            "sentences_allagree",
            split="train",
            trust_remote_code=True,
        )
        by_class = {0: [], 1: [], 2: []}
        for example in dataset:
            label = int(example["label"])
            by_class[label].append((str(example["sentence"]), label))
        smallest_class = min(len(items) for items in by_class.values())
        per_class = min(n_total // 3, smallest_class)
        for class_items in by_class.values():
            rng.shuffle(class_items)
            for text, label in class_items[:per_class]:
                sentences.append(text)
                labels.append(label)

    elif name == "wikitext2":
        dataset = dataset_loader("wikitext", "wikitext-2-raw-v1", split="train")
        candidates: list[str] = []
        for example in dataset:
            for line in str(example["text"]).split("\n"):
                line = line.strip()
                if 5 <= _word_count(line) < max_words:
                    candidates.append(line)
        rng.shuffle(candidates)
        sentences = candidates[:n_total]

    elif name == "dialogsum":
        dataset = dataset_loader("knkarthick/dialogsum", split="train")
        candidates = []
        for example in dataset:
            for utterance in str(example["dialogue"]).split("\n"):
                utterance = utterance.strip()
                if ":" in utterance:
                    utterance = utterance.split(":", 1)[-1].strip()
                if 3 <= _word_count(utterance) < max_words:
                    candidates.append(utterance)
        rng.shuffle(candidates)
        sentences = candidates[:n_total]

    elif name == "cnn_dailymail":
        dataset = dataset_loader("cnn_dailymail", "3.0.0", split="train")
        candidates = []
        for example in dataset:
            for sentence in str(example["highlights"]).split("\n"):
                sentence = sentence.strip().lstrip("- ").strip()
                if 5 <= _word_count(sentence) < max_words:
                    candidates.append(sentence)
            if len(candidates) >= n_total * 5:
                break
        rng.shuffle(candidates)
        sentences = candidates[:n_total]

    elif name == "ag_news":
        dataset = dataset_loader("ag_news", split="train")
        candidates = []
        for example in dataset:
            text = str(example["text"]).strip()
            for sentence in text.split(". "):
                sentence = sentence.strip()
                if sentence and 5 <= _word_count(sentence) < max_words:
                    candidates.append(sentence)
            if len(candidates) >= n_total * 5:
                break
        rng.shuffle(candidates)
        sentences = candidates[:n_total]

    else:
        raise ValueError(f"Unknown dataset: {name!r}")

    combined: list[tuple[str, int | None]]
    if labels:
        combined = list(zip(sentences, labels, strict=True))
    else:
        combined = list(zip(sentences, [None] * len(sentences), strict=True))
    rng.shuffle(combined)

    split_index = int(len(combined) * 0.8)
    train_items = combined[:split_index]
    test_items = combined[split_index:]
    train_labels = [int(label) for _, label in train_items if label is not None] if labels else None
    test_labels = [int(label) for _, label in test_items if label is not None] if labels else None

    return {
        "train": [sentence for sentence, _ in train_items],
        "test": [sentence for sentence, _ in test_items],
        "train_labels": train_labels,
        "test_labels": test_labels,
    }
