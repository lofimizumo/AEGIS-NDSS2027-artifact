"""Canonical attack-name registry for methods presented in the paper."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .dager import run_dager
from .embed_norm import run_embed_norm
from .fedspy import run_fedspy
from .grab import run_grab
from .layernorm import run_layernorm
from .lm_head import run_lm_head
from .mlp_fc import run_mlp_fc
from .mlp_proj import run_mlp_proj
from .mlp_svd import run_mlp_svd
from .somp import run_somp

AttackCallable = Callable[..., Any]

ATTACKS: dict[str, AttackCallable] = {
    "dager": run_dager,
    "grab": run_grab,
    "mlp_svd": run_mlp_svd,
    "embed_norm": run_embed_norm,
    "mlp_fc": run_mlp_fc,
    "mlp_proj": run_mlp_proj,
    "layernorm": run_layernorm,
    "lm_head": run_lm_head,
    "somp": run_somp,
    "fedspy": run_fedspy,
    "fedspy_llm": run_fedspy,
}


def get_attack(name: str) -> AttackCallable:
    try:
        return ATTACKS[name]
    except KeyError as error:
        raise KeyError(f"unknown attack {name!r}; available: {', '.join(ATTACKS)}") from error


__all__ = ["ATTACKS", "AttackCallable", "get_attack"]
