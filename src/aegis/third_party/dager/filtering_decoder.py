"""Decoder filtering stage from official DAGER.

Modified for package-relative imports and formatting.
"""

from __future__ import annotations

import copy
import itertools
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .functional import check_if_in_span, filter_outliers, get_span_dists


def filter_decoder(
    args: Any,
    model_wrapper: Any,
    spans: list[torch.Tensor],
    candidate_ids: list[list[int]],
    max_ids: int = -1,
):
    second_span = spans[1]
    candidate_ids = copy.deepcopy(candidate_ids)
    for index in range(len(candidate_ids)):
        if max_ids >= 0:
            candidate_ids[index] = candidate_ids[index][: min(max_ids, len(candidate_ids[index]))]
    if args.pad == "right":
        batch = torch.tensor(candidate_ids[0]).unsqueeze(1)
    else:
        start_ids = candidate_ids[0].copy()
        if model_wrapper.start_token is not None:
            start_ids = [model_wrapper.start_token]
        batch = torch.tensor(start_ids).unsqueeze(1)

    incorrect_flags = torch.zeros_like(batch).squeeze(1)
    scores = (
        check_if_in_span(
            second_span,
            model_wrapper.get_layer_inputs(batch.to(args.device))[0],
            args.dist_norm,
        )
        .mean(dim=1)
        .cpu()
    )
    predicted: list[list[int]] = []
    predicted_scores: list[float] = []
    top_incorrect = [[] for _ in range(args.batch_size)]
    top_incorrect_scores = [torch.inf for _ in range(args.batch_size)]

    position = 1
    while True:
        best_incorrect_at_length = [[] for _ in range(args.batch_size)]
        best_incorrect_scores_at_length = [torch.inf for _ in range(args.batch_size)]
        if len(batch) == 0 or (not model_wrapper.has_rope() and position >= len(candidate_ids)):
            break
        if model_wrapper.has_rope():
            endings = torch.tensor(candidate_ids[0])
            endings = endings[endings != model_wrapper.pad_token]
        else:
            endings = torch.tensor(candidate_ids[position])

        combinations = iter(itertools.product(range(batch.shape[0]), range(len(endings))))
        next_batch: list[torch.Tensor] = []
        next_scores: list[torch.Tensor] = []
        next_incorrect: list[torch.Tensor] = []
        distance_chunks: list[torch.Tensor] = []
        complete = args.defense_noise is None
        current_sentence = 0
        progress = tqdm(total=batch.shape[0] * endings.shape[0])

        while True:
            batch_indices: list[int] = []
            ending_indices: list[int] = []
            chunk_size = max(args.parallel // endings.shape[0], 1) * endings.shape[0]
            for _ in range(chunk_size):
                item = next(combinations, None)
                if item is None:
                    break
                batch_indices.append(item[0])
                ending_indices.append(item[1])
            batch_index_tensor = torch.tensor(np.array(batch_indices))
            ending_index_tensor = torch.tensor(np.array(ending_indices))
            if batch_index_tensor.shape[0] == 0 and complete:
                break
            if batch_index_tensor.shape[0] == 0:
                all_indices = np.array(
                    list(itertools.product(range(batch.shape[0]), range(len(endings))))
                )
                new_batch = torch.cat(
                    (
                        batch[all_indices[:, 0]].clone().long(),
                        endings[all_indices[:, 1]].clone().long().unsqueeze(1),
                    ),
                    dim=-1,
                ).to(args.device)
                new_incorrect = incorrect_flags[all_indices[:, 0]].to(args.device)
                distances = torch.cat(distance_chunks)
                distances, correct = filter_outliers(
                    distances,
                    stage="sequence",
                    std_thrs=args.l2_std_thrs,
                    maxB=args.batch_size,
                )
                complete = True
            else:
                new_batch = (
                    torch.cat(
                        (
                            batch[batch_index_tensor],
                            endings[ending_index_tensor].unsqueeze(1),
                        ),
                        dim=-1,
                    )
                    .int()
                    .to(args.device)
                )
                new_incorrect = incorrect_flags[batch_index_tensor].to(args.device)
                if args.defense_noise is None:
                    distances, correct = filter_decoder_step(
                        args,
                        model_wrapper,
                        spans,
                        new_batch,
                        position,
                    )
                else:
                    distance_chunks.append(
                        filter_decoder_step(
                            args,
                            model_wrapper,
                            spans,
                            new_batch,
                            position,
                        )
                    )
                    continue

            if position > 1:
                complete_batches = torch.where(~correct.reshape(-1, endings.shape[0]).any(dim=1))[0]
                for prediction_index in complete_batches:
                    source_index = current_sentence + prediction_index
                    if not incorrect_flags[source_index]:
                        predicted.append(batch[source_index].tolist())
                        predicted_scores.append(scores[source_index].item())

            next_batch.append(new_batch[correct].cpu())
            position_scores = (
                distances[:, 1:].mean(dim=1) if model_wrapper.has_bos() else distances.mean(dim=1)
            )
            next_scores.append(position_scores[correct].cpu())
            next_incorrect.append(new_incorrect[correct].cpu())
            current_sentence += len(batch_indices) // endings.shape[0]

            failed_sentences = new_batch[~correct]
            failed_scores = (
                distances[~correct, 1:].mean(dim=1)
                if model_wrapper.has_bos()
                else distances[~correct].mean(dim=1)
            )
            if len(failed_sentences) == 0:
                continue
            length_sentences: list[list[int]] = []
            length_scores: list[float] = []
            for _ in range(args.batch_size):
                best_index = torch.argmin(failed_scores)
                best_score = failed_scores[best_index]
                best_sentence = failed_sentences[best_index]
                length_sentences.append(best_sentence.cpu().tolist())
                length_scores.append(best_score.item())
                similar = (best_sentence == failed_sentences).sum(1) >= (
                    position + 1
                ) * args.distinct_thresh
                failed_scores[similar] = torch.inf

            for candidate, score in zip(length_sentences, length_scores, strict=True):
                if score > best_incorrect_scores_at_length[-1]:
                    break
                insertion = 0
                while score > best_incorrect_scores_at_length[insertion]:
                    insertion += 1
                best_incorrect_at_length = (
                    best_incorrect_at_length[:insertion]
                    + [candidate]
                    + best_incorrect_at_length[insertion:-1]
                )
                best_incorrect_scores_at_length = (
                    best_incorrect_scores_at_length[:insertion]
                    + [score]
                    + best_incorrect_scores_at_length[insertion:-1]
                )
            progress.update(new_batch.shape[0])

        batch = torch.cat(next_batch)
        if len(batch) == 0:
            break
        incorrect_flags = torch.cat(next_incorrect)
        scores = torch.cat(next_scores)
        if position != len(candidate_ids) - 1 and best_incorrect_at_length[0]:
            batch = torch.cat((batch, torch.tensor(best_incorrect_at_length)))
            scores = torch.cat((scores, torch.tensor(best_incorrect_scores_at_length)))
            incorrect_flags = torch.cat(
                (
                    incorrect_flags,
                    torch.ones(len(best_incorrect_at_length)),
                )
            )
        top_incorrect_scores += best_incorrect_scores_at_length
        top_incorrect += best_incorrect_at_length
        progress.close()
        position += 1

    for index in range(batch.shape[0]):
        predicted.append(batch[index].cpu().tolist())
        predicted_scores.append(scores[index].item())
    return predicted, predicted_scores, top_incorrect, top_incorrect_scores


def filter_decoder_step(
    args: Any,
    model_wrapper: Any,
    spans: list[torch.Tensor],
    batch: torch.Tensor,
    position: int,
):
    attention_mask = torch.where(batch != model_wrapper.pad_token, 1, 0)
    if args.defense_noise is not None:
        inputs = model_wrapper.get_layer_inputs(
            batch,
            attention_mask=attention_mask,
            layers=args.n_layers - 1,
        )
        return get_span_dists(args, model_wrapper, spans, inputs, stage="sequence")
    layer_input = model_wrapper.get_layer_inputs(batch, attention_mask=attention_mask)[0]
    distances = check_if_in_span(spans[1], layer_input, args.dist_norm)
    within_span = distances < args.l2_span_thresh
    if model_wrapper.has_rope():
        allowed = torch.tensor(
            [model_wrapper.pad_token, model_wrapper.start_token],
            device=args.device,
        )
        within_span = within_span | torch.isin(batch, allowed)
        if position > 1:
            repeats = (batch[:, -2] == model_wrapper.start_token) & torch.isin(
                batch[:, -1], batch[:, 1]
            )
            correct = within_span.all(dim=1) & ~repeats
        else:
            correct = within_span.all(dim=1)
    else:
        correct = within_span.all(dim=1)
    return distances, correct


__all__ = ["filter_decoder", "filter_decoder_step"]
