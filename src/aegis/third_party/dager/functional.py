"""Active numerical utilities adapted from the official DAGER repository.

Modified for package-relative use and trimmed to the functions used by AEGIS.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def get_layer_decomp(
    gradient: torch.Tensor,
    B: int | None = None,
    tol: float | None = None,
    upcast: bool = False,
) -> tuple[int, torch.Tensor]:
    array = gradient.detach().cpu().numpy()
    if upcast:
        array = array.astype(np.float32)
    if B is None:
        B = int(np.linalg.matrix_rank(array, tol=tol))
    _, _, vectors = torch.svd_lowrank(torch.tensor(array), q=B, niter=10)
    result = vectors.T.half() if upcast else vectors.T
    return B, torch.as_tensor(result).detach()


def check_if_in_span(span: torch.Tensor, vectors: torch.Tensor, norm: str = "l2") -> torch.Tensor:
    vectors = vectors / vectors.pow(2).sum(-1, keepdim=True).sqrt()
    projection = torch.einsum("ik,ij,...j->...k", span, span, vectors)
    residual = projection - vectors
    if norm == "l2":
        return residual.pow(2).sum(-1).sqrt()
    if norm == "l1":
        return residual.abs().mean(-1)
    raise ValueError(f"unsupported span norm: {norm}")


def get_top_b_in_span(
    span: torch.Tensor,
    vectors: torch.Tensor,
    batch_size: int,
    threshold: float,
    norm: str,
) -> tuple[torch.Tensor, ...]:
    del batch_size
    distances = check_if_in_span(span, vectors, norm)
    selected = torch.where(distances < threshold)
    _, indices = torch.sort(distances[selected])
    return tuple(part[indices] for part in selected)


def filter_outliers(
    distances: Any,
    stage: str = "token",
    std_thrs: float | None = None,
    maxB: int | None = None,
):
    if std_thrs is None:
        selected = torch.tensor(distances.argsort()[:maxB])
        mask = torch.zeros_like(distances).bool()
        mask[selected] = True
    elif maxB is None:
        normalized = (distances - distances.mean()) / distances.std()
        mask = normalized < -std_thrs
        selected = torch.tensor(np.nonzero(mask)[:, 0])
    else:
        mask = torch.zeros_like(distances).bool()
        mask[torch.tensor(distances.argsort()[:maxB])] = True
        normalized = (distances - distances.mean()) / distances.std()
        mask &= normalized < -std_thrs
        selected = torch.tensor(np.nonzero(mask)[:, 0])
    if stage == "token":
        return selected
    return torch.tensor(distances).unsqueeze(1), torch.tensor(mask)


def get_span_distances(
    args: Any,
    model_wrapper: Any,
    spans: list[torch.Tensor],
    embeddings: list[torch.Tensor] | torch.Tensor,
    position: int = 0,
    stage: str = "token",
) -> torch.Tensor:
    distances: list[torch.Tensor] = []
    if stage == "token":
        distances.append(check_if_in_span(spans[0], embeddings, args.dist_norm).T)
        sentences = torch.arange(embeddings.shape[1]).unsqueeze(1).to(model_wrapper.args.device)
        layer_embeddings = model_wrapper.get_layer_inputs(sentences, layers=args.n_layers - 1)
    else:
        layer_embeddings = [item.to(model_wrapper.args.device) for item in embeddings]
    if position == 0:
        for index in range(model_wrapper.args.n_layers - 1):
            distances.append(
                check_if_in_span(
                    spans[index + 1],
                    layer_embeddings[index],
                    args.dist_norm,
                )
            )
    joined = torch.cat(distances, axis=1)
    logits = torch.log(joined) - torch.log(1 - joined)
    return logits.mean(axis=1).cpu().detach()


# Original names retained for compatibility with the extracted DAGER adapter.
get_top_B_in_span = get_top_b_in_span
get_span_dists = get_span_distances

__all__ = [
    "check_if_in_span",
    "filter_outliers",
    "get_layer_decomp",
    "get_span_distances",
    "get_span_dists",
    "get_top_B_in_span",
    "get_top_b_in_span",
]
