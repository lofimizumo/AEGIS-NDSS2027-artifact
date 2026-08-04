"""Unified execution engine for externally defined AEGIS studies."""

from __future__ import annotations

import inspect
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .types import RunSettings, StudyMode, StudyResult

_METRIC_KEYS = ("r1", "r2", "rl", "meteor")


class StudyRunner:
    """Load resources lazily and execute one resolved study."""

    def __init__(self, settings: RunSettings) -> None:
        self.settings = settings
        self.study = settings.study
        self._runtime_cache: dict[str, Any] | None = None

    def run(self) -> StudyResult:
        """Execute the configured study and atomically write its JSON result."""

        self._configure_cache()
        self._seed_everything()
        started = time.perf_counter()
        dispatch = {
            StudyMode.TRAIN_ATTACK: self._run_train_attack,
            StudyMode.ABLATION: self._run_ablation,
            StudyMode.CONVERGENCE: self._run_convergence,
            StudyMode.PARETO: self._run_pareto,
            StudyMode.QUALITATIVE: self._run_qualitative,
            StudyMode.SIDE_CHANNEL: self._run_side_channel,
            StudyMode.FIXED_GRID: self._run_fixed_grid,
        }
        payload = dispatch[self.study.mode]()
        payload["elapsed"] = round(time.perf_counter() - started, 1)
        payload.setdefault("config", self._config_payload())
        output_name = self.settings.output_name or self.study.output_filename
        output_path = self.settings.output_dir / output_name
        self._write_json_atomic(output_path, payload)
        return StudyResult(self.study.name, output_path, payload)

    def _runtime(self) -> dict[str, Any]:
        if self._runtime_cache is not None:
            return self._runtime_cache

        import torch
        from datasets import load_dataset as hf_load_dataset
        from transformers import AutoTokenizer

        from aegis.attacks import get_attack, run_sentence_probes, untie_lm_head
        from aegis.config import (
            DATASET_LABEL_NAMES,
            DATASET_TYPE,
            MODEL_ARCH,
            MODEL_BATCH_SIZE,
            MODEL_N_ATTACK,
            build_model_paths,
            supports_dager_attack,
        )
        from aegis.data import load_dataset_sentences
        from aegis.defenses import AegisDefense, NoDefense, make_defense
        from aegis.evaluation import eval_classification_utility, eval_loss_ppl
        from aegis.models import (
            is_masked_lm,
            load_model,
            warmup_bert_sequence_classifier,
        )
        from aegis.training import FineTuneConfig, fine_tune

        self._runtime_cache = {
            "torch": torch,
            "hf_load_dataset": hf_load_dataset,
            "AutoTokenizer": AutoTokenizer,
            "get_attack": get_attack,
            "run_sentence_probes": run_sentence_probes,
            "untie_lm_head": untie_lm_head,
            "DATASET_LABEL_NAMES": DATASET_LABEL_NAMES,
            "DATASET_TYPE": DATASET_TYPE,
            "MODEL_ARCH": MODEL_ARCH,
            "MODEL_BATCH_SIZE": MODEL_BATCH_SIZE,
            "MODEL_N_ATTACK": MODEL_N_ATTACK,
            "model_paths": build_model_paths(self.settings.model_dir),
            "supports_dager_attack": supports_dager_attack,
            "load_dataset_sentences": load_dataset_sentences,
            "AegisDefense": AegisDefense,
            "NoDefense": NoDefense,
            "make_defense": make_defense,
            "eval_classification_utility": eval_classification_utility,
            "eval_loss_ppl": eval_loss_ppl,
            "is_masked_lm": is_masked_lm,
            "load_model": load_model,
            "warmup_bert_sequence_classifier": warmup_bert_sequence_classifier,
            "FineTuneConfig": FineTuneConfig,
            "fine_tune": fine_tune,
        }
        return self._runtime_cache

    def _configure_cache(self) -> None:
        if self.settings.cache_dir is None:
            return
        cache = str(self.settings.cache_dir.expanduser().resolve())
        os.environ["HF_HOME"] = cache
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(cache) / "hub")
        os.environ["HF_DATASETS_CACHE"] = str(Path(cache) / "datasets")

    def _seed_everything(self) -> None:
        random.seed(self.settings.seed)
        try:
            import numpy as np

            np.random.seed(self.settings.seed)
        except ImportError:
            pass
        self._runtime()["torch"].manual_seed(self.settings.seed)

    def _device(self) -> str:
        if self.settings.device:
            return self.settings.device
        torch = self._runtime()["torch"]
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_tokenizer_model(self, model_name: str) -> tuple[Any, Any]:
        runtime = self._runtime()
        path = runtime["model_paths"][model_name]
        use_fast = runtime["MODEL_ARCH"][model_name] != "deberta_mlm"
        tokenizer = runtime["AutoTokenizer"].from_pretrained(
            path,
            trust_remote_code=True,
            use_fast=use_fast,
            cache_dir=(
                str(self.settings.cache_dir) if self.settings.cache_dir is not None else None
            ),
        )
        architecture = runtime["MODEL_ARCH"][model_name]
        if tokenizer.pad_token is None:
            if architecture in ("bert_mlm", "roberta_mlm", "deberta_mlm"):
                tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.eos_token
        model = runtime["load_model"](
            model_name,
            model_paths=runtime["model_paths"],
        ).to(self._device())
        return tokenizer, model

    def _load_dataset(self, name: str, size: int) -> dict[str, Any]:
        runtime = self._runtime()

        def loader(*args: Any, **kwargs: Any) -> Any:
            if self.settings.cache_dir is not None:
                kwargs.setdefault(
                    "cache_dir",
                    str(self.settings.cache_dir / "datasets"),
                )
            return runtime["hf_load_dataset"](*args, **kwargs)

        return runtime["load_dataset_sentences"](
            name,
            size,
            seed=self.settings.seed,
            dataset_loader=loader,
        )

    def _training_config(self, model_name: str, *, convergence: bool = False) -> Any:
        runtime = self._runtime()
        configured_batch = self.settings.parameters.get("batch_size")
        batch_size = (
            int(configured_batch)
            if configured_batch is not None
            else int(runtime["MODEL_BATCH_SIZE"].get(model_name, 8))
        )
        optimizer = self._behavior(
            "large_model_optimizer",
            self.settings.parameters.get("large_model_optimizer", "adamw8bit"),
        )
        return runtime["FineTuneConfig"](
            epochs=int(self.settings.parameters.get("epochs", 1)),
            learning_rate=float(self.settings.parameters.get("lr", 5e-5)),
            batch_size=batch_size,
            max_steps=self._optional_int(self.settings.parameters.get("max_steps")),
            seed=self.settings.seed,
            patience=(
                0 if convergence else int(self.settings.parameters.get("early_stop_patience", 0))
            ),
            eval_every=(
                int(self.settings.parameters.get("eval_every", 5)) if convergence else None
            ),
            early_stop_patience=(
                int(self.settings.parameters.get("early_stop_patience", 0)) if convergence else 0
            ),
            gradient_clip_norm=1.0 if convergence else None,
            large_model_optimizer=str(optimizer),
        )

    def _make_defense(
        self,
        name: str,
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> Any:
        runtime = self._runtime()
        configured = self._defense_parameters(name)
        configured.update(parameters or {})
        if name == "aegis" and configured:
            return runtime["AegisDefense"](**configured)
        defense = runtime["make_defense"](name)
        if not configured:
            return defense
        try:
            return type(defense)(**configured)
        except TypeError as error:
            raise ValueError(f"defense {name!r} does not accept configured parameters") from error

    def _train_condition(
        self,
        model_name: str,
        dataset_name: str,
        dataset: dict[str, Any],
        defense_name: str,
        attacks: Sequence[str],
        *,
        defense: Any | None = None,
    ) -> dict[str, Any]:
        runtime = self._runtime()
        tokenizer, model = self._load_tokenizer_model(model_name)
        pretrain_loss, pretrain_ppl = runtime["eval_loss_ppl"](
            model,
            tokenizer,
            dataset["test"],
            self._device(),
        )
        active_defense = defense if defense is not None else self._make_defense(defense_name)
        tune_result = runtime["fine_tune"](
            model,
            tokenizer,
            dataset["train"],
            device=self._device(),
            defense=active_defense,
            config=self._training_config(model_name),
            validation_sentences=dataset["test"],
        )
        test_loss, test_ppl = runtime["eval_loss_ppl"](
            model,
            tokenizer,
            dataset["test"],
            self._device(),
        )

        result_style = str(self._behavior("result_style", "standard"))
        if result_style in {"baseline", "compact", "large_model"}:
            row: dict[str, Any] = {
                "model": model_name,
                "dataset": dataset_name,
                "defense": defense_name,
                "ppl_pre": pretrain_ppl,
                "ppl": test_ppl,
                "avg_step_s": round(tune_result.average_step_seconds, 4),
            }
        else:
            row = {
                "model": model_name,
                "dataset": dataset_name,
                "defense": defense_name,
                "pretrain_loss": pretrain_loss,
                "pretrain_ppl": pretrain_ppl,
                "test_loss": test_loss,
                "ppl": test_ppl,
            }

        labels = dataset.get("test_labels")
        if (
            runtime["DATASET_TYPE"].get(dataset_name) == "classification"
            and labels is not None
            and not runtime["is_masked_lm"](model)
        ):
            accuracy, f1 = runtime["eval_classification_utility"](
                model,
                tokenizer,
                dataset["test"],
                labels,
                runtime["DATASET_LABEL_NAMES"][dataset_name],
                self._device(),
            )
            row["acc"] = accuracy
            row["f1"] = f1

        attack_count = int(
            runtime["MODEL_N_ATTACK"].get(
                model_name,
                self.settings.parameters.get("n_attack", 10),
            )
        )
        runnable_attacks = list(attacks)
        if "dager" in runnable_attacks and not runtime["supports_dager_attack"](model_name):
            runnable_attacks.remove("dager")
            row["dager"] = None
            row["dager_skipped"] = True

        sequence_attack_model = None
        attack_labels = None
        train_labels = dataset.get("train_labels")
        if (
            bool(self._behavior("warmup_sequence_classifier", False))
            and "dager" in runnable_attacks
            and runtime["MODEL_ARCH"].get(model_name) == "bert_mlm"
            and runtime["DATASET_TYPE"].get(dataset_name) == "classification"
            and train_labels is not None
        ):
            sequence_attack_model = runtime["warmup_bert_sequence_classifier"](
                model,
                tokenizer,
                dataset["train"],
                train_labels,
                len(runtime["DATASET_LABEL_NAMES"][dataset_name]),
                self._device(),
                model_name,
                epochs=int(self.settings.parameters.get("bert_seq_head_epochs", 2)),
                model_paths=runtime["model_paths"],
            )
            attack_labels = train_labels[:attack_count]

        summaries = self._attack_summaries(
            model,
            tokenizer,
            dataset["train"][:attack_count],
            runnable_attacks,
            active_defense.attack_hook(model),
            labels=attack_labels,
            sequence_attack_model=sequence_attack_model,
        )
        row.update(summaries)
        if sequence_attack_model is not None:
            self._release_model(sequence_attack_model)

        if bool(self._behavior("include_mean_r1", False)) or result_style in {
            "baseline",
            "compact",
            "large_model",
        }:
            observed = [
                value["r1"]
                for attack in attacks
                if isinstance((value := row.get(attack)), dict)
                and isinstance(value.get("r1"), int | float)
            ]
            if observed:
                row["mean_r1"] = round(sum(observed) / len(observed), 4)

        self._release_model(model)
        return row

    def _attack_summaries(
        self,
        model: Any,
        tokenizer: Any,
        sentences: Sequence[str],
        attacks: Sequence[str],
        defense_hook: Any,
        *,
        labels: Sequence[int] | None = None,
        sequence_attack_model: Any | None = None,
    ) -> dict[str, Any]:
        collected: dict[str, list[dict[str, Any]]] = {name: [] for name in attacks}
        errors: dict[str, list[str]] = {name: [] for name in attacks}
        for index, sentence in enumerate(sentences):
            for attack_name in attacks:
                try:
                    extras: dict[str, Any] = {}
                    if attack_name == "dager" and sequence_attack_model is not None:
                        extras["seq_attack_model"] = sequence_attack_model
                        if labels is not None:
                            extras["cls_label"] = int(labels[index])
                    result = self._invoke_attack(
                        attack_name,
                        model,
                        tokenizer,
                        sentence,
                        defense_hook,
                        extra_kwargs=extras,
                    )
                except Exception as error:
                    errors[attack_name].append(f"{type(error).__name__}: {error}")
                    continue
                if isinstance(result, dict):
                    collected[attack_name].append(result)

        output: dict[str, Any] = {}
        for attack_name, results in collected.items():
            metrics = self._aggregate_metrics(results)
            if metrics:
                output[attack_name] = metrics
            elif errors[attack_name]:
                output[attack_name] = {"errors": errors[attack_name]}
        return output

    def _invoke_attack(
        self,
        name: str,
        model: Any,
        tokenizer: Any,
        sentence: str,
        defense_hook: Any,
        *,
        extra_kwargs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        function = self._runtime()["get_attack"](name)
        parameters = self.settings.parameters
        candidates = {
            "max_tokens": parameters.get("max_tokens", 32),
            "max_len": parameters.get("max_tokens", 32),
            "n_h": parameters.get("grab_nh", 3),
            "n_c": parameters.get("grab_nc", 200),
            "n_d": parameters.get("grab_nd", 3),
            "n_b": parameters.get("grab_nb", 2),
            "n_e": parameters.get("grab_ne", 10),
            "lr": parameters.get("grab_lr", 0.01),
            "alpha_l1": parameters.get("grab_alpha_l1", 0.01),
            "n_sv": parameters.get("mlpsvd_n_sv", 10),
        }
        candidates.update(extra_kwargs or {})
        accepted = inspect.signature(function).parameters
        kwargs = {
            key: value for key, value in candidates.items() if key in accepted and value is not None
        }
        return function(
            model,
            tokenizer,
            sentence,
            self._device(),
            defense_hook,
            **kwargs,
        )

    def _run_train_attack(self) -> dict[str, Any]:
        validation_strategy = str(self._behavior("validation_strategy", "")).replace("-", "_")
        if self.settings.validate and validation_strategy in {
            "adaptive_attack",
            "attack_blocking",
            "defense_blocking",
        }:
            return self._run_attack_validation()

        size = int(self.settings.parameters.get("dataset_size", 1000))
        datasets = {name: self._load_dataset(name, size) for name in self.settings.datasets}
        results = [
            self._train_condition(
                model,
                dataset,
                datasets[dataset],
                defense,
                self.settings.attacks,
            )
            for model in self.settings.models
            for dataset in self.settings.datasets
            for defense in self.settings.defenses
        ]
        return {"results": results}

    def _run_attack_validation(self) -> dict[str, Any]:
        model_name = self.settings.models[0]
        dataset_name = self.settings.datasets[0]
        dataset = self._load_dataset(
            dataset_name,
            int(self.settings.parameters.get("dataset_size", 50)),
        )
        count = int(self.settings.parameters.get("n_attack", 5))
        sentences = [sentence for sentence in dataset["train"] if len(sentence.split()) >= 3][
            :count
        ]
        attack_name = str(
            self._behavior(
                "validation_attack",
                self.settings.attacks[0] if self.settings.attacks else "mlp_svd",
            )
        )
        open_variant = str(self._behavior("validation_open_variant", "a4"))
        protected_name = str(self._behavior("validation_protected_defense", "aegis"))
        conditions = (
            (open_variant, self._ablation_defense(open_variant)),
            (protected_name, None),
        )
        results: list[dict[str, Any]] = []
        for defense_name, configured_defense in conditions:
            tokenizer, model = self._load_tokenizer_model(model_name)
            defense = (
                configured_defense
                if configured_defense is not None
                else self._make_defense(defense_name)
            )
            defense.pre_train(model)
            summary = self._attack_summaries(
                model,
                tokenizer,
                sentences,
                (attack_name,),
                defense.attack_hook(model),
            )
            results.append(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "defense": defense_name,
                    **summary,
                }
            )
            self._release_model(model)

        open_r1 = float(results[0].get(attack_name, {}).get("r1", 0.0))
        protected_r1 = float(results[1].get(attack_name, {}).get("r1", 0.0))
        open_threshold = float(self.settings.parameters.get("validation_open_threshold", 0.3))
        protected_threshold = float(
            self.settings.parameters.get("validation_protected_threshold", 0.05)
        )
        weak_threshold = float(self.settings.parameters.get("validation_weak_threshold", 0.1))
        similarity_margin = float(
            self.settings.parameters.get("validation_similarity_margin", 0.02)
        )
        passed = (open_r1 > open_threshold and protected_r1 < protected_threshold) or (
            open_r1 < weak_threshold and abs(open_r1 - protected_r1) < similarity_margin
        )
        return {
            "passed": passed,
            "validation_strategy": str(self._behavior("validation_strategy", "")),
            "attack": attack_name,
            "results": results,
        }

    def _run_ablation(self) -> dict[str, Any]:
        variants = self._string_list(self.settings.parameters.get("variants")) or [
            f"a{i}" for i in range(8)
        ]
        size = int(self.settings.parameters.get("dataset_size", 1000))
        dataset_name = self.settings.datasets[0]
        dataset = self._load_dataset(dataset_name, size)
        results = []
        for variant in variants:
            row = self._train_condition(
                self.settings.models[0],
                dataset_name,
                dataset,
                variant,
                self.settings.attacks,
                defense=self._ablation_defense(variant),
            )
            row["variant"] = row.pop("defense")
            row["variant_label"] = self._variant_label(variant)
            row["components"] = self._variant_components(variant)
            row.pop("pretrain_loss", None)
            row.pop("test_loss", None)
            results.append(row)

        payload: dict[str, Any] = {
            "model": self.settings.models[0],
            "dataset": dataset_name,
            "variants": variants,
            "results": results,
        }
        validation_strategy = str(self._behavior("validation_strategy", "")).replace("-", "_")
        if self.settings.validate and validation_strategy in {
            "ablation",
            "ablation_components",
        }:
            payload["passed"] = self._validate_ablation(results)
            payload["validation_strategy"] = str(self._behavior("validation_strategy", ""))
        return payload

    def _validate_ablation(self, results: Sequence[Mapping[str, Any]]) -> bool:
        attack_name = str(
            self._behavior(
                "validation_attack",
                self.settings.attacks[0] if self.settings.attacks else "dager",
            )
        )
        open_variant = str(self._behavior("validation_open_variant", "a0"))
        blocked_variants = self._string_list(
            self._behavior("validation_blocked_variants", ["a4", "a7"])
        )
        open_threshold = float(self.settings.parameters.get("validation_open_threshold", 0.5))
        blocked_threshold = float(self.settings.parameters.get("validation_blocked_threshold", 0.1))
        by_variant = {str(row["variant"]): row for row in results}
        open_score = by_variant.get(open_variant, {}).get(attack_name, {}).get("r1", 0.0)
        blocked_scores = [
            by_variant.get(variant, {}).get(attack_name, {}).get("r1", 1.0)
            for variant in blocked_variants
        ]
        return float(open_score) > open_threshold and all(
            float(score) < blocked_threshold for score in blocked_scores
        )

    def _ablation_defense(self, variant: str) -> Any:
        runtime = self._runtime()
        components = self._variant_components(variant)
        if not any(components.values()):
            return runtime["NoDefense"]()
        parameters = {
            "flood_scale": 1.0 if components["embed"] else 0.0,
            "real_token_retain": 0.3,
            "mlp_flood_scale": 8.0 if components["mlp"] else 0.0,
            "mlp_retain": 0.3,
            "ln_flood_scale": 4.0 if components["embed"] else 0.0,
        }
        aegis_type = runtime["AegisDefense"]
        freeze = components["freeze"]

        class AegisAblationDefense:
            name = variant

            def __init__(self) -> None:
                self.inner: Any | None = None

            def pre_train(self, model: Any) -> None:
                self.inner = aegis_type(model, **parameters)
                if freeze:
                    self.inner.apply_defense_pre_training()

            def after_backward(self, model: Any) -> None:
                if self.inner is None:
                    raise RuntimeError("ablation defense was not initialized")
                self.inner.flood_gradients(model)

            def attack_hook(self, model: Any) -> Any:
                if self.inner is None:
                    self.inner = aegis_type(model, **parameters)
                return self.inner

        return AegisAblationDefense()

    def _run_convergence(self) -> dict[str, Any]:
        runtime = self._runtime()
        model_name = self.settings.models[0]
        dataset_name = self.settings.datasets[0]
        defense_name = self.settings.defenses[0]
        dataset = self._load_dataset(
            dataset_name,
            int(self.settings.parameters.get("dataset_size", 1000)),
        )
        tokenizer, model = self._load_tokenizer_model(model_name)
        defense = self._make_defense(defense_name)
        result = runtime["fine_tune"](
            model,
            tokenizer,
            dataset["train"],
            device=self._device(),
            defense=defense,
            config=self._training_config(model_name, convergence=True),
            validation_sentences=dataset["test"],
        )
        payload = {
            "model": model_name,
            "dataset": dataset_name,
            "defense": defense_name,
            "epochs": int(self.settings.parameters.get("epochs", 20)),
            "lr": float(self.settings.parameters.get("lr", 5e-5)),
            "batch_size": self._training_config(model_name).batch_size,
            "eval_every": int(self.settings.parameters.get("eval_every", 5)),
            "early_stop_patience": int(self.settings.parameters.get("early_stop_patience", 0)),
            "trace": result.trace,
        }
        self._release_model(model)
        return payload

    def _run_pareto(self) -> dict[str, Any]:
        model_name = self.settings.models[0]
        dataset_name = self.settings.datasets[0]
        grid = self._float_list(self.settings.parameters.get("grid"))
        dataset = self._load_dataset(
            dataset_name,
            int(self.settings.parameters.get("dataset_size", 500)),
        )
        results = []
        for attention_scale in grid:
            for mlp_scale in grid:
                defense = self._pareto_defense(attention_scale, mlp_scale)
                row = self._train_condition(
                    model_name,
                    dataset_name,
                    dataset,
                    self.settings.defenses[0],
                    self.settings.attacks,
                    defense=defense,
                )
                cell: dict[str, Any] = {
                    "attn_scale": attention_scale,
                    "mlp_scale": mlp_scale,
                    "ppl": row["ppl"],
                }
                for attack in self.settings.attacks:
                    value = row.get(attack)
                    if isinstance(value, dict) and isinstance(
                        value.get("r1"),
                        int | float,
                    ):
                        cell[attack] = value["r1"]
                results.append(cell)
        return {
            "model": model_name,
            "dataset": dataset_name,
            "grid": grid,
            "results": results,
        }

    def _pareto_defense(
        self,
        attention_scale: float,
        mlp_scale: float,
    ) -> Any:
        runtime = self._runtime()
        aegis_type = runtime["AegisDefense"]
        torch = runtime["torch"]
        from aegis.models import (
            get_block_attention_module,
            get_transformer_blocks,
        )

        class AegisParetoFloodingDefense:
            name = "aegis_pareto"

            def __init__(self) -> None:
                self.inner: Any | None = None

            def pre_train(self, model: Any) -> None:
                self.inner = aegis_type(
                    model,
                    flood_scale=1.0,
                    real_token_retain=0.3,
                    mlp_flood_scale=mlp_scale,
                    mlp_retain=0.3,
                    ln_flood_scale=0.0,
                )

                def flood_attention(target: Any) -> None:
                    if attention_scale <= 0:
                        return
                    for block in get_transformer_blocks(target):
                        module = get_block_attention_module(block)
                        if module is None:
                            continue
                        for child in module.modules():
                            weight = getattr(child, "weight", None)
                            if weight is None or weight.grad is None or weight.dim() < 2:
                                continue
                            gradient = weight.grad.float()
                            if gradient.norm().item() < 1e-10:
                                continue
                            std = gradient.std().item()
                            gradient *= 0.3
                            gradient += torch.randn_like(gradient) * std * attention_scale
                            weight.grad.copy_(gradient.to(weight.grad.dtype))

                self.inner._flood_attention_projection_grads = flood_attention

            def after_backward(self, model: Any) -> None:
                if self.inner is None:
                    raise RuntimeError("Pareto defense was not initialized")
                self.inner.flood_gradients(model)

            def attack_hook(self, model: Any) -> Any:
                return self.inner

        return AegisParetoFloodingDefense()

    def _run_qualitative(self) -> dict[str, Any]:
        dataset_name = self.settings.datasets[0]
        model_name = self.settings.models[0]
        dataset = self._load_dataset(
            dataset_name,
            int(self.settings.parameters.get("dataset_size", 200)),
        )
        tokenizer, temporary_model = self._load_tokenizer_model(model_name)
        self._release_model(temporary_model)
        sentences = self._pick_sentences(
            tokenizer,
            dataset["test"],
            int(self.settings.parameters.get("n_sentences", 3)),
            int(self.settings.parameters.get("min_tokens", 6)),
            int(self.settings.parameters.get("max_tokens", 18)),
        )
        metric_names = self._string_list(self._behavior("reconstruction_metrics", ["r1"]))
        rows: list[dict[str, Any]] = []
        for defense_name in self.settings.defenses:
            _, model = self._load_tokenizer_model(model_name)
            defense = self._make_defense(defense_name)
            defense.pre_train(model)
            hook = defense.attack_hook(model)
            for attack_name in self.settings.attacks:
                for index, sentence in enumerate(sentences):
                    started = time.perf_counter()
                    result = self._invoke_attack(
                        attack_name,
                        model,
                        tokenizer,
                        sentence,
                        hook,
                    )
                    row: dict[str, Any] = {
                        "dataset": dataset_name,
                        "defense": defense_name,
                        "attack": attack_name,
                        "sentence_idx": index,
                        "original": sentence,
                        "reconstructed": result.get("rec_text", ""),
                        "elapsed": round(time.perf_counter() - started, 1),
                    }
                    for metric in metric_names:
                        if isinstance(result.get(metric), int | float):
                            row[metric] = float(result[metric])
                    rows.append(row)
            self._release_model(model)
        return {
            "dataset": dataset_name,
            "model": model_name,
            "results": rows,
        }

    def _run_side_channel(self) -> dict[str, Any]:
        conditions = [self._side_channel_condition(defense) for defense in self.settings.defenses]
        if not self.settings.validate:
            return {"results": conditions}

        open_name = str(self._behavior("validation_open_defense", "none"))
        protected_name = str(self._behavior("validation_protected_defense", "aegis"))
        by_defense = {str(condition["defense"]): condition for condition in conditions}
        if open_name not in by_defense or protected_name not in by_defense:
            raise ValueError(
                "side-channel validation requires configured open and protected defenses"
            )
        open_condition = by_defense[open_name]
        protected_condition = by_defense[protected_name]
        verdict: dict[str, Any] = {}
        for channel in self.settings.attacks:
            open_r1 = (
                (open_condition.get("summary") or {}).get(channel) or {}
            ).get("r1")
            protected_r1 = (
                (protected_condition.get("summary") or {}).get(channel) or {}
            ).get("r1")
            if open_r1 is None or protected_r1 is None:
                raise ValueError(
                    f"side-channel validation missing r1 for channel {channel!r}; "
                    "no probe sentences were scored"
                )
            open_null = (
                (open_condition.get("summary") or {}).get(f"{channel}_null") or {}
            ).get("r1", 0.0)
            protected_null = (
                (protected_condition.get("summary") or {}).get(f"{channel}_null") or {}
            ).get("r1", open_null)
            valid = open_r1 >= max(
                float(self.settings.parameters.get("valid_threshold", 0.1)),
                open_null + float(self.settings.parameters.get("valid_margin", 0.05)),
            )
            blocked = (not valid) or protected_r1 <= max(
                float(self.settings.parameters.get("block_threshold", 0.05)),
                protected_null + float(self.settings.parameters.get("block_margin", 0.05)),
            )
            verdict[channel] = {
                "open_r1": open_r1,
                "protected_r1": protected_r1,
                "open_null_r1": open_null,
                "protected_null_r1": protected_null,
                "valid_attack": valid,
                "blocked_if_valid": blocked,
            }
        return {
            "passed": all(value["blocked_if_valid"] for value in verdict.values()),
            "validation_strategy": str(self._behavior("validation_strategy", "side_channel")),
            "verdict": verdict,
            "results": conditions,
        }

    def _side_channel_condition(self, defense_name: str) -> dict[str, Any]:
        runtime = self._runtime()
        model_name = self.settings.models[0]
        dataset_name = self.settings.datasets[0]
        tokenizer, model = self._load_tokenizer_model(model_name)
        untied = (
            runtime["untie_lm_head"](model)
            if bool(self.settings.parameters.get("untie_head", False))
            else False
        )
        mode_parameters = {
            key: self.settings.parameters[key]
            for key in (
                "mlp_flood_scale",
                "mlp_retain",
                "ln_flood_scale",
            )
            if key in self.settings.parameters
        }
        defense = self._make_defense(
            defense_name,
            parameters=mode_parameters,
        )
        defense.pre_train(model)
        dataset = self._load_dataset(
            dataset_name,
            max(
                100,
                int(self.settings.parameters.get("n_sentences", 8)) * 10,
            ),
        )
        sentences = self._pick_sentences(
            tokenizer,
            dataset["test"] + dataset["train"],
            int(self.settings.parameters.get("n_sentences", 8)),
            2,
            int(self.settings.parameters.get("max_tokens", 32)),
        )
        per_sentence = [
            runtime["run_sentence_probes"](
                model,
                tokenizer,
                sentence,
                defense.attack_hook(model),
                list(self.settings.attacks),
                int(self.settings.parameters.get("max_tokens", 32)),
                bool(self.settings.parameters.get("random_baseline", True)),
                self._device(),
            )
            for sentence in sentences
        ]
        summary_names = list(self.settings.attacks)
        if bool(self.settings.parameters.get("random_baseline", True)):
            summary_names.extend(f"{name}_null" for name in self.settings.attacks)
        summary = {
            name: self._aggregate_metrics([item[name] for item in per_sentence if name in item])
            for name in summary_names
        }
        result = {
            "model": model_name,
            "dataset": dataset_name,
            "defense": defense_name,
            "untie_head": bool(self.settings.parameters.get("untie_head", False)),
            "untied_head_applied": bool(untied),
            "n_sentences": len(sentences),
            "summary": summary,
            "per_sentence": per_sentence,
        }
        self._release_model(model)
        return result

    def _run_fixed_grid(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        model_name = self.settings.models[0]
        tokenizer, temporary_model = self._load_tokenizer_model(model_name)
        self._release_model(temporary_model)
        sentence_sets: dict[str, list[str]] = {}
        for dataset_name in self.settings.datasets:
            fixed = self.study.fixed_sentences.get(dataset_name)
            if fixed:
                sentence_sets[dataset_name] = list(fixed)
            else:
                dataset = self._load_dataset(
                    dataset_name,
                    int(self.settings.parameters.get("dataset_size", 300)),
                )
                sentence_sets[dataset_name] = self._pick_sentences(
                    tokenizer,
                    dataset["test"] + dataset["train"],
                    int(self.settings.parameters.get("n_sentences", 5)),
                    int(self.settings.parameters.get("min_tokens", 6)),
                    int(self.settings.parameters.get("max_tokens", 18)),
                )

        include_reconstructions = bool(self._behavior("include_reconstructions", False))
        reconstruction_metrics = self._string_list(self._behavior("reconstruction_metrics", ["r1"]))
        for defense_name in self.settings.defenses:
            _, model = self._load_tokenizer_model(model_name)
            defense = self._make_defense(defense_name)
            defense.pre_train(model)
            hook = defense.attack_hook(model)
            for dataset_name, sentences in sentence_sets.items():
                for attack_name in self.settings.attacks:
                    results = [
                        self._invoke_attack(
                            attack_name,
                            model,
                            tokenizer,
                            sentence,
                            hook,
                        )
                        for sentence in sentences
                    ]
                    r1s = [
                        float(result["r1"])
                        for result in results
                        if isinstance(result.get("r1"), int | float)
                    ]
                    row: dict[str, Any] = {
                        "model": model_name,
                        "dataset": dataset_name,
                        "attack": attack_name,
                        "defense": defense_name,
                        "n": len(r1s),
                        "mean_r1": (round(sum(r1s) / len(r1s), 4) if r1s else None),
                        "r1s": [round(value, 4) for value in r1s],
                    }
                    if include_reconstructions:
                        reconstructions = []
                        for sentence, result in zip(
                            sentences,
                            results,
                            strict=True,
                        ):
                            reconstruction: dict[str, Any] = {
                                "ref": result.get("ref_text", sentence),
                                "rec": result.get("rec_text", ""),
                            }
                            for metric in reconstruction_metrics:
                                if isinstance(
                                    result.get(metric),
                                    int | float,
                                ):
                                    reconstruction[metric] = round(
                                        float(result[metric]),
                                        4,
                                    )
                            reconstructions.append(reconstruction)
                        row["reconstructions"] = reconstructions
                    rows.append(row)
            self._release_model(model)
        return {"results": rows}

    def _config_payload(self) -> dict[str, Any]:
        return {
            "study": self.study.name,
            "mode": self.study.mode.value,
            "models": list(self.settings.models),
            "datasets": list(self.settings.datasets),
            "defenses": list(self.settings.defenses),
            "attacks": list(self.settings.attacks),
            "seed": self.settings.seed,
            "quick": self.settings.quick,
            "validate": self.settings.validate,
            "model_dir": (
                str(self.settings.model_dir) if self.settings.model_dir is not None else None
            ),
            "cache_dir": (
                str(self.settings.cache_dir) if self.settings.cache_dir is not None else None
            ),
            "device": self.settings.device,
            "parameters": dict(self.settings.parameters),
            "behavior": dict(self.study.behavior),
        }

    def _behavior(self, key: str, default: Any = None) -> Any:
        return self.study.behavior.get(key, default)

    def _defense_parameters(self, name: str) -> dict[str, Any]:
        value = self.settings.parameters.get("defense_parameters")
        if not isinstance(value, Mapping):
            return {}
        configured = value.get(name)
        return dict(configured) if isinstance(configured, Mapping) else {}

    @staticmethod
    def _aggregate_metrics(
        results: Iterable[Mapping[str, Any]],
    ) -> dict[str, float]:
        rows = list(results)
        output: dict[str, float] = {}
        for key in _METRIC_KEYS:
            values = [
                float(row[key])
                for row in rows
                if isinstance(row.get(key), int | float) and math.isfinite(float(row[key]))
            ]
            if values:
                output[key] = round(sum(values) / len(values), 4)
        return output

    @staticmethod
    def _pick_sentences(
        tokenizer: Any,
        sentences: Sequence[str],
        count: int,
        minimum_tokens: int,
        maximum_tokens: int,
    ) -> list[str]:
        selected: list[str] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or sentence.startswith("=") or sentence.count("=") >= 2:
                continue
            token_count = tokenizer(
                sentence,
                return_tensors="pt",
            )["input_ids"].shape[1]
            if minimum_tokens <= token_count <= maximum_tokens:
                selected.append(sentence)
            if len(selected) >= count:
                break
        return selected

    def _release_model(self, model: Any) -> None:
        model.cpu()
        torch = self._runtime()["torch"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _write_json_atomic(
        path: Path,
        payload: Mapping[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_payload = StudyRunner._json_safe(payload)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                json.dump(safe_payload, handle, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if temporary_name is not None and Path(temporary_name).exists():
                Path(temporary_name).unlink()

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Recursively replace non-finite floats with JSON ``null``."""

        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, Mapping):
            return {str(key): StudyRunner._json_safe(item) for key, item in value.items()}
        if isinstance(value, tuple | list):
            return [StudyRunner._json_safe(item) for item in value]
        return value

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return None if value is None else int(value)

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list | tuple):
            return [str(item) for item in value]
        return [str(value)]

    @staticmethod
    def _float_list(value: Any) -> list[float]:
        if value is None:
            return []
        if isinstance(value, list | tuple):
            return [float(item) for item in value]
        return [float(value)]

    def _variant_components(self, variant: str) -> dict[str, bool]:
        configured = self._behavior("variant_components", {})
        if isinstance(configured, Mapping):
            value = configured.get(variant)
            if isinstance(value, Mapping):
                return {
                    "freeze": bool(value.get("freeze", False)),
                    "embed": bool(value.get("embed", False)),
                    "mlp": bool(value.get("mlp", False)),
                }
        index = int(variant[1:])
        return {
            "freeze": index in (1, 4, 5, 7),
            "embed": index in (2, 4, 6, 7),
            "mlp": index in (3, 5, 6, 7),
        }

    def _variant_label(self, variant: str) -> str:
        configured = self._behavior("variant_labels", {})
        if isinstance(configured, Mapping) and variant in configured:
            return str(configured[variant])
        return {
            "a0": "none",
            "a1": "freeze",
            "a2": "embed",
            "a3": "mlp",
            "a4": "freeze+embed",
            "a5": "freeze+mlp",
            "a6": "embed+mlp",
            "a7": "freeze+embed+mlp",
        }[variant]


def run_study(settings: RunSettings) -> StudyResult:
    """Execute one resolved external study configuration."""

    return StudyRunner(settings).run()


__all__ = ["StudyRunner", "run_study"]
