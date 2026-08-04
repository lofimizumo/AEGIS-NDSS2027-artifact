"""Reusable fine-tuning engine for causal, masked, and classification models."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

import torch
import torch.nn as nn

from aegis.defenses import Defense, NoDefense
from aegis.evaluation import apply_mlm_labels
from aegis.models import is_masked_lm


class TraceRecord(TypedDict):
    step: int
    epoch: int
    train_loss: float
    val_loss: float
    val_ppl: float


class TimingRecord(TypedDict):
    step: int
    epoch: int
    seconds: float
    train_loss: float
    skipped: bool


TraceCallback = Callable[[TraceRecord], None]
TimingCallback = Callable[[TimingRecord], None]
LogCallback = Callable[[str], None]
EvaluationCallback = Callable[
    [nn.Module, Any, Sequence[str], Sequence[int] | None, str],
    tuple[float, float],
]
OptimizerFactory = Callable[[list[nn.Parameter], float], torch.optim.Optimizer]


@dataclass(frozen=True)
class FineTuneConfig:
    """Configuration shared by the supported training modes."""

    epochs: int = 1
    learning_rate: float = 5e-5
    batch_size: int = 8
    max_length: int = 64
    pad_to_multiple_of: int | None = 8
    max_steps: int | None = None
    mask_probability: float = 0.15
    shuffle: bool = True
    seed: int | None = None

    # Epoch-level validation and best-checkpoint restoration.
    patience: int = 0
    restore_best: bool = True

    # Step-level convergence trace.
    eval_every: int | None = None
    early_stop_patience: int = 0
    gradient_clip_norm: float | None = None

    # Prefer 8-bit AdamW for very large models unless a study selects SGD.
    large_model_threshold: int = 8_000_000_000
    large_model_optimizer: Literal["adamw8bit", "sgd"] = "adamw8bit"
    fused_adamw: bool = True
    apply_defense_pre_training: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 0:
            raise ValueError("epochs must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when supplied")
        if not 0 <= self.mask_probability <= 1:
            raise ValueError("mask_probability must be between 0 and 1")
        if self.patience < 0 or self.early_stop_patience < 0:
            raise ValueError("patience values must be non-negative")
        if self.eval_every is not None and self.eval_every <= 0:
            raise ValueError("eval_every must be positive when supplied")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when supplied")


@dataclass
class FineTuneCallbacks:
    """Optional observers for progress, convergence traces, and step timing."""

    trace: TraceCallback | None = None
    timing: TimingCallback | None = None
    log: LogCallback | None = print


@dataclass
class FineTuneResult:
    """Summary returned by every fine-tuning mode."""

    steps: int
    epochs_completed: int
    average_step_seconds: float
    trace: list[TraceRecord] = field(default_factory=list)
    stopped_early: bool = False
    stop_reason: str | None = None
    best_validation_loss: float | None = None



def evaluate_loss(
    model: nn.Module,
    tokenizer: Any,
    sentences: Sequence[str],
    labels: Sequence[int] | None,
    device: str,
    *,
    max_length: int = 64,
    mask_probability: float = 0.15,
) -> tuple[float, float]:
    """Evaluate mean loss and its exponential for all supported objectives."""

    if labels is not None and len(labels) != len(sentences):
        raise ValueError("validation labels must align with validation sentences")

    model.eval()
    losses: list[float] = []
    masked_lm = labels is None and is_masked_lm(model)
    with torch.no_grad():
        for index, sentence in enumerate(sentences):
            encoded = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            if labels is not None:
                target = torch.tensor(
                    [labels[index]],
                    dtype=torch.long,
                    device=input_ids.device,
                )
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=target,
                )
            elif masked_lm:
                masked_input, target = apply_mlm_labels(
                    input_ids,
                    attention_mask,
                    tokenizer,
                    mask_prob=mask_probability,
                )
                if (target == -100).all():
                    continue
                output = model(
                    input_ids=masked_input,
                    attention_mask=attention_mask,
                    labels=target,
                )
            else:
                target = input_ids.clone()
                target[attention_mask == 0] = -100
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=target,
                )

            loss = float(output.loss.item())
            if math.isfinite(loss):
                losses.append(loss)

    if not losses:
        return float("inf"), float("inf")
    mean_loss = sum(losses) / len(losses)
    try:
        perplexity = math.exp(mean_loss)
    except OverflowError:
        perplexity = float("inf")
    return round(mean_loss, 4), round(perplexity, 2)


class FineTuneEngine:
    """One training loop with selectable validation, trace, and timing behavior."""

    def __init__(
        self,
        config: FineTuneConfig | None = None,
        *,
        callbacks: FineTuneCallbacks | None = None,
        evaluator: EvaluationCallback | None = None,
        optimizer_factory: OptimizerFactory | None = None,
    ) -> None:
        self.config = config or FineTuneConfig()
        self.callbacks = callbacks or FineTuneCallbacks()
        self.evaluator = evaluator
        self.optimizer_factory = optimizer_factory

    def run(
        self,
        model: nn.Module,
        tokenizer: Any,
        train_sentences: Sequence[str],
        *,
        device: str,
        defense: Defense | None = None,
        train_labels: Sequence[int] | None = None,
        validation_sentences: Sequence[str] | None = None,
        validation_labels: Sequence[int] | None = None,
    ) -> FineTuneResult:
        """Fine-tune ``model`` and return timing and convergence information."""

        config = self.config
        active_defense = defense or NoDefense()
        if config.apply_defense_pre_training:
            active_defense.pre_train(model)

        examples = self._prepare_examples(train_sentences, train_labels)
        if validation_labels is not None and validation_sentences is None:
            raise ValueError("validation_labels require validation_sentences")
        if (
            train_labels is not None
            and validation_sentences is not None
            and validation_labels is None
        ):
            raise ValueError("classification validation requires validation_labels")
        if (
            validation_labels is not None
            and validation_sentences is not None
            and len(validation_labels) != len(validation_sentences)
        ):
            raise ValueError("validation labels must align with validation sentences")
        if (config.patience > 0 or config.eval_every is not None) and not validation_sentences:
            raise ValueError("validation sentences are required for configured validation")

        model.train()
        model.to(device)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("model has no trainable parameters after defense pre-training")
        optimizer = self._make_optimizer(model, trainable, device)

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError("tokenizer requires either a pad_token or eos_token")
            tokenizer.pad_token = tokenizer.eos_token

        device_string = str(device)
        is_cuda = device_string.startswith("cuda")
        model_dtype = next(model.parameters()).dtype
        model_bfloat16 = model_dtype == torch.bfloat16
        use_scaler = is_cuda and not model_bfloat16
        amp_dtype = torch.bfloat16 if model_bfloat16 else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        masked_lm = train_labels is None and is_masked_lm(model)
        randomizer = random.Random(config.seed) if config.seed is not None else random

        result = FineTuneResult(steps=0, epochs_completed=0, average_step_seconds=0.0)
        total_step_seconds = 0.0
        best_step_metric = float("inf")
        best_validation_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        no_improvement_epochs = 0
        no_improvement_evaluations = 0

        if config.eval_every is not None and validation_sentences is not None:
            validation_loss, validation_ppl = self._evaluate(
                model,
                tokenizer,
                validation_sentences,
                validation_labels,
                device_string,
            )
            self._record_trace(
                result,
                step=0,
                epoch=0,
                train_loss=float("nan"),
                validation_loss=validation_loss,
                validation_ppl=validation_ppl,
            )
            best_step_metric = validation_ppl
            self._log(f"    step 0  val_ppl={validation_ppl:.2f}")
            model.train()

        stop_training = False
        for epoch in range(config.epochs):
            if config.shuffle:
                randomizer.shuffle(examples)
            epoch_losses: list[float] = []
            model.train()

            for start in range(0, len(examples), config.batch_size):
                batch = examples[start : start + config.batch_size]
                started_at = time.perf_counter()
                encoded = tokenizer(
                    [sentence for sentence, _ in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=config.max_length,
                    pad_to_multiple_of=config.pad_to_multiple_of,
                ).to(device_string)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(encoded["input_ids"])

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    "cuda",
                    enabled=is_cuda,
                    dtype=amp_dtype,
                ):
                    if train_labels is not None:
                        targets = torch.tensor(
                            [label for _, label in batch],
                            dtype=torch.long,
                            device=encoded["input_ids"].device,
                        )
                        output = model(
                            input_ids=encoded["input_ids"],
                            attention_mask=attention_mask,
                            labels=targets,
                        )
                    elif masked_lm:
                        masked_input, targets = apply_mlm_labels(
                            encoded["input_ids"],
                            attention_mask,
                            tokenizer,
                            mask_prob=config.mask_probability,
                        )
                        output = model(
                            input_ids=masked_input,
                            attention_mask=attention_mask,
                            labels=targets,
                        )
                    else:
                        targets = encoded["input_ids"].clone()
                        targets[attention_mask == 0] = -100
                        output = model(
                            input_ids=encoded["input_ids"],
                            attention_mask=attention_mask,
                            labels=targets,
                        )

                scaler.scale(output.loss).backward()
                # Unscale before defense flooding so noise/pruning/AEGIS see true grads.
                scaler.unscale_(optimizer)
                active_defense.after_backward(model)

                skipped = False
                if config.gradient_clip_norm is not None:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        trainable,
                        max_norm=config.gradient_clip_norm,
                    )
                    if not torch.isfinite(total_norm):
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        skipped = True

                if not skipped:
                    scaler.step(optimizer)
                    scaler.update()

                train_loss = float(output.loss.item()) if not skipped else float("nan")
                elapsed = time.perf_counter() - started_at
                total_step_seconds += elapsed
                result.steps += 1
                if math.isfinite(train_loss):
                    epoch_losses.append(train_loss)
                self._record_timing(
                    step=result.steps,
                    epoch=epoch + 1,
                    seconds=elapsed,
                    train_loss=train_loss,
                    skipped=skipped,
                )

                if (
                    not skipped
                    and config.eval_every is not None
                    and result.steps % config.eval_every == 0
                    and validation_sentences is not None
                ):
                    validation_loss, validation_ppl = self._evaluate(
                        model,
                        tokenizer,
                        validation_sentences,
                        validation_labels,
                        device_string,
                    )
                    self._record_trace(
                        result,
                        step=result.steps,
                        epoch=epoch + 1,
                        train_loss=train_loss,
                        validation_loss=validation_loss,
                        validation_ppl=validation_ppl,
                    )
                    self._log(
                        f"    step {result.steps}  train={train_loss:.3f}  "
                        f"val_ppl={validation_ppl:.2f}"
                    )
                    if validation_ppl < best_step_metric:
                        best_step_metric = validation_ppl
                        no_improvement_evaluations = 0
                    else:
                        no_improvement_evaluations += 1
                    if (
                        config.early_stop_patience > 0
                        and no_improvement_evaluations >= config.early_stop_patience
                    ):
                        result.stopped_early = True
                        result.stop_reason = "step_validation_patience"
                        stop_training = True
                    model.train()

                if config.max_steps is not None and result.steps >= config.max_steps:
                    result.stopped_early = True
                    result.stop_reason = "max_steps"
                    stop_training = True
                if stop_training:
                    break

            result.epochs_completed = epoch + 1
            mean_train_loss = (
                sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
            )

            if config.patience > 0 and validation_sentences is not None:
                validation_loss, validation_ppl = self._evaluate(
                    model,
                    tokenizer,
                    validation_sentences,
                    validation_labels,
                    device_string,
                )
                improved = validation_loss < best_validation_loss
                if improved:
                    best_validation_loss = validation_loss
                    no_improvement_epochs = 0
                    best_state = self._clone_state_dict(
                        model,
                        move_to_cpu=sum(p.numel() for p in model.parameters())
                        > config.large_model_threshold,
                    )
                else:
                    no_improvement_epochs += 1
                status = (
                    "  best"
                    if improved
                    else f"  no-improve={no_improvement_epochs}/{config.patience}"
                )
                self._log(
                    f"    epoch {epoch + 1}/{config.epochs}  "
                    f"train={mean_train_loss:.4f}  val={validation_loss:.4f}{status}"
                )
                if no_improvement_epochs >= config.patience:
                    result.stopped_early = True
                    result.stop_reason = "epoch_validation_patience"
                    stop_training = True
            else:
                self._log(f"    epoch {epoch + 1}/{config.epochs}  loss={mean_train_loss:.4f}")
            if stop_training:
                break

        if config.restore_best and best_state is not None:
            model.load_state_dict(best_state)
        result.best_validation_loss = (
            best_validation_loss
            if config.patience > 0 and math.isfinite(best_validation_loss)
            else None
        )
        result.average_step_seconds = total_step_seconds / max(result.steps, 1)
        return result

    @staticmethod
    def _prepare_examples(
        sentences: Sequence[str],
        labels: Sequence[int] | None,
    ) -> list[tuple[str, int | None]]:
        if labels is not None and len(labels) != len(sentences):
            raise ValueError("training labels must align with training sentences")
        examples = [
            (sentence, labels[index] if labels is not None else None)
            for index, sentence in enumerate(sentences)
            if len(sentence.split()) >= 2
        ]
        if not examples:
            raise ValueError("no training sentences contain at least two words")
        return examples

    def _make_optimizer(
        self,
        model: nn.Module,
        trainable: list[nn.Parameter],
        device: str,
    ) -> torch.optim.Optimizer:
        config = self.config
        if self.optimizer_factory is not None:
            return self.optimizer_factory(trainable, config.learning_rate)

        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count > config.large_model_threshold:
            if config.large_model_optimizer == "sgd":
                return torch.optim.SGD(
                    trainable,
                    lr=config.learning_rate,
                    momentum=0.0,
                )
            try:
                import bitsandbytes as bnb
            except ImportError:
                return torch.optim.SGD(
                    trainable,
                    lr=config.learning_rate,
                    momentum=0.0,
                )
            return bnb.optim.AdamW8bit(trainable, lr=config.learning_rate)

        use_fused = (
            config.fused_adamw and str(device).startswith("cuda") and torch.cuda.is_available()
        )
        return torch.optim.AdamW(
            trainable,
            lr=config.learning_rate,
            fused=use_fused,
        )

    def _evaluate(
        self,
        model: nn.Module,
        tokenizer: Any,
        sentences: Sequence[str],
        labels: Sequence[int] | None,
        device: str,
    ) -> tuple[float, float]:
        if self.evaluator is not None:
            return self.evaluator(model, tokenizer, sentences, labels, device)
        return evaluate_loss(
            model,
            tokenizer,
            sentences,
            labels,
            device,
            max_length=self.config.max_length,
            mask_probability=self.config.mask_probability,
        )

    def _record_trace(
        self,
        result: FineTuneResult,
        *,
        step: int,
        epoch: int,
        train_loss: float,
        validation_loss: float,
        validation_ppl: float,
    ) -> None:
        record = TraceRecord(
            step=step,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=validation_loss,
            val_ppl=validation_ppl,
        )
        result.trace.append(record)
        if self.callbacks.trace is not None:
            self.callbacks.trace(record)

    def _record_timing(
        self,
        *,
        step: int,
        epoch: int,
        seconds: float,
        train_loss: float,
        skipped: bool,
    ) -> None:
        if self.callbacks.timing is None:
            return
        self.callbacks.timing(
            TimingRecord(
                step=step,
                epoch=epoch,
                seconds=seconds,
                train_loss=train_loss,
                skipped=skipped,
            )
        )

    def _log(self, message: str) -> None:
        if self.callbacks.log is not None:
            self.callbacks.log(message)

    @staticmethod
    def _clone_state_dict(
        model: nn.Module,
        *,
        move_to_cpu: bool,
    ) -> dict[str, torch.Tensor]:
        if move_to_cpu:
            return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        return {key: value.detach().clone() for key, value in model.state_dict().items()}


def fine_tune(
    model: nn.Module,
    tokenizer: Any,
    train_sentences: Sequence[str],
    *,
    device: str,
    defense: Defense | None = None,
    config: FineTuneConfig | None = None,
    callbacks: FineTuneCallbacks | None = None,
    evaluator: EvaluationCallback | None = None,
    optimizer_factory: OptimizerFactory | None = None,
    train_labels: Sequence[int] | None = None,
    validation_sentences: Sequence[str] | None = None,
    validation_labels: Sequence[int] | None = None,
) -> FineTuneResult:
    """Convenience entry point for :class:`FineTuneEngine`."""

    engine = FineTuneEngine(
        config,
        callbacks=callbacks,
        evaluator=evaluator,
        optimizer_factory=optimizer_factory,
    )
    return engine.run(
        model,
        tokenizer,
        train_sentences,
        device=device,
        defense=defense,
        train_labels=train_labels,
        validation_sentences=validation_sentences,
        validation_labels=validation_labels,
    )


__all__ = [
    "EvaluationCallback",
    "FineTuneCallbacks",
    "FineTuneConfig",
    "FineTuneEngine",
    "FineTuneResult",
    "LogCallback",
    "OptimizerFactory",
    "TimingCallback",
    "TimingRecord",
    "TraceCallback",
    "TraceRecord",
    "evaluate_loss",
    "fine_tune",
]
