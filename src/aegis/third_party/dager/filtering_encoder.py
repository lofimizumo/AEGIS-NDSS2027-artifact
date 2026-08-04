"""Encoder filtering stage adapted from official DAGER.

Modified only for package-relative imports and local naming.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .functional import check_if_in_span


def filter_encoder(
    args: Any,
    model_wrapper: Any,
    second_span: torch.Tensor,
    length: int,
    token_type: int,
    candidate_ids: list[list[int]],
    sentence_filter: list[list[int]],
    approximate_filter: list[list[int]],
    approximate_scores: list[Any],
    max_ids: int,
    batch_size: int,
) -> tuple[list[list[int]], list[float]]:
    predicted = [[-1] * (length + 1) for _ in range(batch_size)]
    scores = [torch.inf for _ in range(batch_size)]
    candidate_ids = [row.copy() for row in candidate_ids]
    for index in range(length):
        if max_ids >= 0:
            candidate_ids[index] = candidate_ids[index][: min(max_ids, len(candidate_ids[index]))]

    for sentence in sentence_filter:
        for index, token in enumerate(sentence):
            if (
                token != model_wrapper.start_token
                and len(candidate_ids[index]) > 1
                and token in candidate_ids[index]
            ):
                candidate_ids[index].remove(token)

    index_ranges = [
        list(range(len(row))) for index, row in enumerate(candidate_ids) if index < length
    ]
    total = int(np.prod([len(row) for row in index_ranges]))
    max_width = max(len(row) for row in candidate_ids[:length])
    padded = torch.tensor(
        [row + [-1] * (max_width - len(row)) for row in candidate_ids[:length]]
    ).to(args.device)
    combinations = iter(itertools.product(*index_ranges))

    progress = tqdm(total=total)
    passed = 0
    while passed < args.maxC:
        if total < args.maxC:
            batch_indices = [item for item in itertools.islice(combinations, args.parallel)]
            indices = torch.tensor(np.array(batch_indices)).to(args.device)
            if indices.shape[0] == 0:
                break
        else:
            indices = torch.cat(
                [torch.randint(len(row), (args.parallel, 1)) for row in candidate_ids[:length]],
                dim=1,
            ).to(args.device)
        passed += indices.shape[0]
        sentences = padded[torch.arange(padded.shape[0], device=args.device), indices]
        eos = (
            torch.tensor([model_wrapper.eos_token], device=args.device)
            .reshape(1, 1)
            .repeat(indices.shape[0], 1)
        )
        sentences = torch.cat((sentences, eos), dim=1)
        if model_wrapper.is_bert():
            token_types = torch.ones_like(sentences) * token_type
            layer_input = model_wrapper.get_layer_inputs(sentences, token_types)[0]
        else:
            layer_input = model_wrapper.get_layer_inputs(sentences)[0]
        distances = check_if_in_span(second_span, layer_input, args.dist_norm).mean(dim=1)

        for index, approximate in enumerate(approximate_filter):
            extended = torch.tensor(
                approximate + [-1] * (length + 1 - len(approximate)),
                device=args.device,
            )
            duplicate = (extended == sentences).sum(1) >= (length + 1) * args.distinct_thresh
            distances[duplicate & (distances > approximate_scores[index])] = torch.inf
        for index in range(batch_size):
            duplicate = (torch.tensor(predicted[index], device=args.device) == sentences).sum(
                1
            ) >= (length + 1) * args.distinct_thresh
            distances[duplicate & (distances > scores[index])] = torch.inf

        batch_sentences: list[list[int]] = []
        batch_scores: list[float] = []
        for _ in range(batch_size):
            best_index = torch.argmin(distances)
            best_sentence = sentences[best_index]
            batch_sentences.append(best_sentence.tolist())
            batch_scores.append(distances[best_index].item())
            similar = (best_sentence == sentences).sum(1) >= (length + 1) * args.distinct_thresh
            distances[similar] = torch.inf
        for candidate, score in zip(batch_sentences, batch_scores, strict=True):
            if score > scores[-1]:
                break
            insertion = 0
            while score > scores[insertion]:
                insertion += 1
            predicted = predicted[:insertion] + [candidate] + predicted[insertion:-1]
            scores = scores[:insertion] + [score] + scores[insertion:-1]
        progress.update(indices.shape[0])
    progress.close()
    return predicted, scores


__all__ = ["filter_encoder"]
