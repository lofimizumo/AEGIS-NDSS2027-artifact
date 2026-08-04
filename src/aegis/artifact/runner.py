"""Self-contained, deterministic artifact exercise for the AEGIS defense."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from contextlib import redirect_stdout, suppress
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

SCHEMA = "aegis.artifact.result"
VERSION = 3
PROFILES = ("kick-the-tires", "scaled-reproduction")
CHANNELS = ("attention", "embedding", "mlp")

_BASE_TOP_LEVEL_FIELDS = {
    "schema",
    "version",
    "profile",
    "device",
    "runtime",
    "utility",
    "channels",
}
_UTILITY_FIELDS = {"loss_before", "loss_after", "loss_delta"}
_CHECK_FIELDS = {"value", "threshold", "passed"}
_REPRODUCTION_FIELDS = {
    "attack",
    "claim",
    "open_token_recall",
    "protected_token_recall",
    "open_threshold",
    "protect_threshold",
    "relative_reduction",
    "passed",
}

_CHANNEL_THRESHOLD = 0.1
_OPEN_TOKEN_RECALL_THRESHOLD = 0.9
_PROTECTED_TOKEN_RECALL_THRESHOLD = 0.05
_REPRODUCTION_CLAIM = (
    "AEGIS suppresses embedding-gradient token-set leakage relative to an undefended baseline"
)

# Fixed local token IDs only. No tokenizer, model weights, dataset, cache, or
# network service is consulted by the runner.
# Special IDs match _build_model: pad=0, bos=2, eos=3. Recall scores content
# tokens only so the causal-LM final-position (EOS) zero-grad artifact is excluded.
_SPECIAL_TOKEN_IDS = frozenset({0, 2, 3})
_CONTENT_TOKEN_POOL = (5, 7, 11, 13, 17, 19, 23, 29, 31)

# Override scaled-reproduction step count for fast unit/integration tests.
_SCALED_STEPS_ENV = "AEGIS_ARTIFACT_SCALED_STEPS"


def _make_token_sequence(offset: int, length: int) -> tuple[int, ...]:
    if length < 3:
        raise ValueError("token sequence length must be at least 3")
    body = [
        _CONTENT_TOKEN_POOL[(index + offset) % len(_CONTENT_TOKEN_POOL)]
        for index in range(length - 2)
    ]
    return (2, *body, 3)


_KICK_TOKEN_IDS = (
    (2, 5, 7, 11, 13, 17, 19, 3),
    (2, 23, 7, 29, 11, 5, 17, 3),
    (2, 13, 31, 7, 19, 11, 5, 3),
    (2, 17, 5, 23, 13, 29, 7, 3),
)
_SCALED_SEQUENCE_LENGTH = 64
_SCALED_TOKEN_IDS = tuple(
    _make_token_sequence(offset, _SCALED_SEQUENCE_LENGTH) for offset in (0, 2, 4, 6)
)
# Kept as an alias for older tests that import the short smoke sequences.
_FIXED_TOKEN_IDS = _KICK_TOKEN_IDS


@dataclass(frozen=True)
class _Profile:
    layers: int
    hidden_size: int
    heads: int
    steps: int
    learning_rate: float
    seed: int
    vocab_size: int
    positions: int
    token_rows: tuple[tuple[int, ...], ...]


_PROFILE_CONFIGS = {
    "kick-the-tires": _Profile(
        layers=1,
        hidden_size=16,
        heads=2,
        steps=1,
        learning_rate=0.03,
        seed=1729,
        vocab_size=32,
        positions=16,
        token_rows=_KICK_TOKEN_IDS,
    ),
    "scaled-reproduction": _Profile(
        # Larger offline micro-model + many CPU steps (~5 minutes on a commodity
        # laptop; allow up to ~10 minutes on slower evaluator hosts).
        layers=4,
        hidden_size=32,
        heads=4,
        steps=60_000,
        learning_rate=0.02,
        seed=2718,
        vocab_size=128,
        positions=_SCALED_SEQUENCE_LENGTH,
        token_rows=_SCALED_TOKEN_IDS,
    ),
}


def _profile_steps(profile_name: str, profile: _Profile) -> int:
    if profile_name != "scaled-reproduction":
        return profile.steps
    override = os.environ.get(_SCALED_STEPS_ENV)
    if override is None or override.strip() == "":
        return profile.steps
    try:
        steps = int(override)
    except ValueError as error:
        raise ValueError(
            f"{_SCALED_STEPS_ENV} must be a positive integer, got {override!r}"
        ) from error
    if steps < 1:
        raise ValueError(f"{_SCALED_STEPS_ENV} must be >= 1, got {steps}")
    return steps


def _require_finite_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path} must be a JSON number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{path} must be finite")
    return converted


def _require_exact_fields(value: object, fields: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    if set(value) != fields:
        missing = sorted(fields - set(value), key=repr)
        extra = sorted(set(value) - fields, key=repr)
        raise ValueError(f"{path} has invalid fields (missing={missing}, extra={extra})")
    return value


def _relative_reduction(open_recall: float, protected_recall: float) -> float:
    if open_recall > 0:
        return 1.0 - protected_recall / open_recall
    return 1.0 if protected_recall == 0 else 0.0


def _validate_reproduction(reproduction: object) -> None:
    check = _require_exact_fields(reproduction, _REPRODUCTION_FIELDS, "reproduction")
    if check["attack"] != "embed_norm":
        raise ValueError("reproduction.attack must be 'embed_norm'")
    if not isinstance(check["claim"], str) or not check["claim"].strip():
        raise ValueError("reproduction.claim must be a non-empty string")

    open_recall = _require_finite_number(
        check["open_token_recall"],
        "reproduction.open_token_recall",
    )
    protected_recall = _require_finite_number(
        check["protected_token_recall"],
        "reproduction.protected_token_recall",
    )
    open_threshold = _require_finite_number(
        check["open_threshold"],
        "reproduction.open_threshold",
    )
    protect_threshold = _require_finite_number(
        check["protect_threshold"],
        "reproduction.protect_threshold",
    )
    relative = _require_finite_number(
        check["relative_reduction"],
        "reproduction.relative_reduction",
    )
    if not 0.0 <= open_recall <= 1.0:
        raise ValueError("reproduction.open_token_recall must be in [0, 1]")
    if not 0.0 <= protected_recall <= 1.0:
        raise ValueError("reproduction.protected_token_recall must be in [0, 1]")
    if not math.isclose(open_threshold, _OPEN_TOKEN_RECALL_THRESHOLD, abs_tol=1e-12):
        raise ValueError(
            f"reproduction.open_threshold must be {_OPEN_TOKEN_RECALL_THRESHOLD}"
        )
    if not math.isclose(protect_threshold, _PROTECTED_TOKEN_RECALL_THRESHOLD, abs_tol=1e-12):
        raise ValueError(
            f"reproduction.protect_threshold must be {_PROTECTED_TOKEN_RECALL_THRESHOLD}"
        )
    expected_relative = _relative_reduction(open_recall, protected_recall)
    if not math.isclose(relative, expected_relative, abs_tol=1e-7):
        raise ValueError("reproduction.relative_reduction does not match recalls")
    if check["passed"] is not True:
        raise ValueError("reproduction.passed must be true")
    if open_recall < open_threshold:
        raise ValueError("reproduction.open_token_recall does not meet its threshold")
    if protected_recall > protect_threshold:
        raise ValueError("reproduction.protected_token_recall exceeds its threshold")
    if not protected_recall < open_recall:
        raise ValueError("reproduction requires protected_token_recall < open_token_recall")


def validate_result(payload: object) -> None:
    """Validate the strict, JSON-safe public artifact result schema."""

    if not isinstance(payload, dict):
        raise ValueError("result must be an object")
    profile = payload.get("profile")
    expected_fields = set(_BASE_TOP_LEVEL_FIELDS)
    if profile == "scaled-reproduction":
        expected_fields.add("reproduction")
    result = _require_exact_fields(payload, expected_fields, "result")

    if result["schema"] != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA!r}")
    if type(result["version"]) is not int or result["version"] != VERSION:
        raise ValueError(f"version must be {VERSION}")
    if result["profile"] not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    if not isinstance(result["device"], str) or not result["device"]:
        raise ValueError("device must be a non-empty string")

    runtime = _require_finite_number(result["runtime"], "runtime")
    if runtime < 0:
        raise ValueError("runtime must be non-negative")

    utility = _require_exact_fields(result["utility"], _UTILITY_FIELDS, "utility")
    loss_before = _require_finite_number(utility["loss_before"], "utility.loss_before")
    loss_after = _require_finite_number(utility["loss_after"], "utility.loss_after")
    loss_delta = _require_finite_number(utility["loss_delta"], "utility.loss_delta")
    if not math.isclose(loss_after - loss_before, loss_delta, abs_tol=1e-7):
        raise ValueError("utility.loss_delta must equal loss_after - loss_before")

    channels = _require_exact_fields(result["channels"], set(CHANNELS), "channels")
    for name in CHANNELS:
        check = _require_exact_fields(channels[name], _CHECK_FIELDS, f"channels.{name}")
        measured = _require_finite_number(check["value"], f"channels.{name}.value")
        threshold = _require_finite_number(
            check["threshold"],
            f"channels.{name}.threshold",
        )
        if measured < 0 or threshold < 0:
            raise ValueError(f"channels.{name} values must be non-negative")
        if not math.isclose(threshold, _CHANNEL_THRESHOLD, abs_tol=1e-12):
            raise ValueError(
                f"channels.{name}.threshold must be {_CHANNEL_THRESHOLD}"
            )
        if check["passed"] is not True:
            raise ValueError(f"channels.{name}.passed must be true")
        if measured < threshold:
            raise ValueError(f"channels.{name}.value does not meet its threshold")

    if profile == "scaled-reproduction":
        _validate_reproduction(result["reproduction"])


def _build_model(config: _Profile, device: str) -> Any:
    from transformers import GPT2Config, GPT2LMHeadModel

    model_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.positions,
        n_ctx=config.positions,
        n_embd=config.hidden_size,
        n_layer=config.layers,
        n_head=config.heads,
        n_inner=config.hidden_size * 2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    return GPT2LMHeadModel(model_config).to(device)


def _module_gradient(module: Any) -> Any:
    import torch

    pieces = [
        (
            parameter.grad.detach().float().reshape(-1)
            if parameter.grad is not None
            else torch.zeros_like(parameter, dtype=torch.float32).reshape(-1)
        )
        for parameter in module.parameters()
    ]
    if not pieces:
        return torch.zeros(1, dtype=torch.float32)
    return torch.cat(pieces)


def _gradient_channels(model: Any) -> dict[str, Any]:
    from aegis.models import get_input_embedding, get_transformer_blocks

    first_block = get_transformer_blocks(model)[0]
    return {
        "attention": _module_gradient(first_block.attn),
        "embedding": _module_gradient(get_input_embedding(model)),
        "mlp": _module_gradient(first_block.mlp),
    }


def _relative_gradient_change(clean: Any, protected: Any) -> float:
    denominator = max(float(clean.norm().item()), 1e-12)
    return float((protected - clean).norm().item()) / denominator


def _loss(model: Any, input_ids: Any) -> Any:
    import torch

    attention_mask = torch.ones_like(input_ids)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    ).loss


def _evaluate_loss(model: Any, input_ids: Any) -> float:
    import torch

    model.eval()
    with torch.no_grad():
        value = float(_loss(model, input_ids).item())
    model.train()
    return value


def _content_token_ids(token_ids: Any) -> set[int]:
    return {
        int(token)
        for token in token_ids
        if int(token) not in _SPECIAL_TOKEN_IDS
    }


def _token_set_recall(row_norms: Any, true_token_ids: set[int]) -> float:
    if not true_token_ids:
        return 0.0
    top_count = len(true_token_ids)
    recovered = {int(index) for index in row_norms.topk(top_count).indices.tolist()}
    return float(len(recovered & true_token_ids) / top_count)


def _embedding_row_norms(
    model: Any,
    batch: Any,
    *,
    defense: Any | None = None,
    flood_seed: int | None = None,
) -> Any:
    import torch

    from aegis.models import get_input_embedding

    embedding = get_input_embedding(model)
    model.zero_grad(set_to_none=True)
    _loss(model, batch).backward()
    if defense is not None:
        assert flood_seed is not None
        torch.manual_seed(flood_seed)
        defense.after_backward(model)
    if embedding.weight.grad is None:
        return torch.zeros(embedding.weight.shape[0], dtype=torch.float32)
    return embedding.weight.grad.detach().float().norm(dim=1)


def _run_reproduction(profile: _Profile, device: str) -> dict[str, Any]:
    """Scaled Channel-2 style embed-norm token-set recovery (offline, no tokenizer)."""

    import torch

    from aegis.defenses import AegisDefense

    torch.manual_seed(profile.seed)
    open_model = _build_model(profile, device)
    protected_model = _build_model(profile, device)
    protected_model.load_state_dict(open_model.state_dict())
    open_model.train()
    protected_model.train()

    batch = torch.tensor([profile.token_rows[0]], dtype=torch.long, device=device)
    true_token_ids = _content_token_ids(batch.view(-1).tolist())

    open_norms = _embedding_row_norms(open_model, batch)
    open_recall = round(_token_set_recall(open_norms, true_token_ids), 8)

    defense = AegisDefense(protected_model)
    with redirect_stdout(StringIO()):
        defense.apply_defense_pre_training()
    protected_norms = _embedding_row_norms(
        protected_model,
        batch,
        defense=defense,
        flood_seed=profile.seed + 100,
    )
    protected_recall = round(_token_set_recall(protected_norms, true_token_ids), 8)

    relative = round(_relative_reduction(open_recall, protected_recall), 8)
    passed = (
        open_recall >= _OPEN_TOKEN_RECALL_THRESHOLD
        and protected_recall <= _PROTECTED_TOKEN_RECALL_THRESHOLD
        and protected_recall < open_recall
    )
    return {
        "attack": "embed_norm",
        "claim": _REPRODUCTION_CLAIM,
        "open_token_recall": open_recall,
        "protected_token_recall": protected_recall,
        "open_threshold": _OPEN_TOKEN_RECALL_THRESHOLD,
        "protect_threshold": _PROTECTED_TOKEN_RECALL_THRESHOLD,
        "relative_reduction": relative,
        "passed": passed,
    }


def _run_exercise(
    profile: _Profile,
    device: str,
    *,
    steps: int,
    show_progress: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    import torch
    from tqdm.auto import tqdm

    from aegis.defenses import AegisDefense

    torch.manual_seed(profile.seed)
    clean_model = _build_model(profile, device)
    protected_model = _build_model(profile, device)
    protected_model.load_state_dict(clean_model.state_dict())
    clean_model.train()
    protected_model.train()

    # Use the same constructor defaults as the research harness / paper settings.
    defense = AegisDefense(protected_model)
    with redirect_stdout(StringIO()):
        defense.apply_defense_pre_training()

    clean_optimizer = torch.optim.SGD(clean_model.parameters(), lr=profile.learning_rate)
    protected_optimizer = torch.optim.SGD(
        (parameter for parameter in protected_model.parameters() if parameter.requires_grad),
        lr=profile.learning_rate,
    )
    rows = torch.tensor(profile.token_rows, dtype=torch.long, device=device)
    loss_before = _evaluate_loss(protected_model, rows)
    measurements = {name: [] for name in CHANNELS}

    step_iter = range(steps)
    if show_progress:
        step_iter = tqdm(
            step_iter,
            total=steps,
            desc="Scaled training steps",
            unit="step",
            leave=True,
            dynamic_ncols=True,
            mininterval=0.5,
        )

    for step in step_iter:
        batch = rows[step % len(rows) : (step % len(rows)) + 1]

        clean_optimizer.zero_grad(set_to_none=True)
        _loss(clean_model, batch).backward()
        clean_gradients = _gradient_channels(clean_model)

        protected_optimizer.zero_grad(set_to_none=True)
        _loss(protected_model, batch).backward()
        torch.manual_seed(profile.seed + 100 + step)
        defense.after_backward(protected_model)
        protected_gradients = _gradient_channels(protected_model)

        for name in CHANNELS:
            measurements[name].append(
                _relative_gradient_change(clean_gradients[name], protected_gradients[name])
            )

        clean_optimizer.step()
        protected_optimizer.step()

    loss_after = _evaluate_loss(protected_model, rows)
    utility = {
        "loss_before": round(loss_before, 8),
        "loss_after": round(loss_after, 8),
    }
    utility["loss_delta"] = round(utility["loss_after"] - utility["loss_before"], 8)
    channel_values = {
        name: round(sum(values) / len(values), 8) for name, values in measurements.items()
    }
    return utility, channel_values


def run_artifact_profile(
    profile: str,
    device: str = "cpu",
    *,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Run a local GPT-2 micro-model and return measured AEGIS checks."""

    if profile not in _PROFILE_CONFIGS:
        raise ValueError(f"profile must be one of {PROFILES}")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string")

    import torch

    try:
        resolved_device = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid torch device {device!r}") from error
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if resolved_device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")

    started = time.perf_counter()
    prior_threads = torch.get_num_threads()
    config = _PROFILE_CONFIGS[profile]
    steps = _profile_steps(profile, config)
    progress = bool(show_progress and profile == "scaled-reproduction" and steps > 1)
    try:
        torch.set_num_threads(1)
        if progress:
            eta_note = (
                "target ~5 minutes; "
                if steps >= 10_000
                else ""
            )
            print(
                f"Running {steps:,} offline CPU steps "
                f"({eta_note}progress bar shows ETA)...",
                flush=True,
            )
        utility, values = _run_exercise(
            config,
            str(resolved_device),
            steps=steps,
            show_progress=progress,
        )
        reproduction = None
        if profile == "scaled-reproduction":
            if show_progress:
                print("Measuring open vs AEGIS content-token embed-norm recall...", flush=True)
            reproduction = _run_reproduction(config, str(resolved_device))
    finally:
        torch.set_num_threads(prior_threads)

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "profile": profile,
        "device": str(resolved_device),
        "runtime": round(time.perf_counter() - started, 6),
        "utility": utility,
        "channels": {
            name: {
                "value": values[name],
                "threshold": _CHANNEL_THRESHOLD,
                "passed": values[name] >= _CHANNEL_THRESHOLD,
            }
            for name in CHANNELS
        },
    }
    if reproduction is not None:
        result["reproduction"] = reproduction
    validate_result(result)
    return result


def write_result(payload: object, output_dir: Path) -> Path:
    """Atomically write one validated result with strict JSON encoding."""

    validate_result(payload)
    result = payload
    assert isinstance(result, dict)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{result['profile']}.json"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_dir,
        prefix=f".{result['profile']}-",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(result, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
    return target


__all__ = ["PROFILES", "VERSION", "run_artifact_profile", "validate_result", "write_result"]
