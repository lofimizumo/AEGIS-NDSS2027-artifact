"""Architecture-aware DAGER attack integration."""

from __future__ import annotations

import contextlib
import io
import os
import types
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from aegis.third_party.dager import (
    add_partial_forward_bert,
    add_partial_forward_gpt2,
    add_partial_forward_llama,
    filter_decoder,
    filter_encoder,
    get_layer_decomp,
    get_top_B_in_span,
)

from .helpers import (
    AttackResult,
    GradientDefense,
    apply_defense,
    apply_mlm_labels,
    compute_text_metrics,
    detect_architecture,
    get_hidden_size,
    get_input_embeddings,
    grad_context,
    is_aegis_defense,
    is_masked_lm,
    unwrap_peft_model,
)

_PATCHED: set[int] = set()


class DagerArgs:
    """Minimal namespace consumed by the official DAGER filtering utilities."""

    _L1_THRESHOLDS = {
        "gpt2": 1e-3,
        "llama": 1e-5,
        "gemma": 1e-5,
        "bert_mlm": 1e-5,
        "roberta_mlm": 1e-5,
        "deberta_mlm": 1e-5,
    }
    _RANK_TOLERANCES = {
        "gpt2": None,
        "llama": 1e-7,
        "gemma": 1e-7,
        "bert_mlm": None,
        "roberta_mlm": None,
        "deberta_mlm": None,
    }
    _PADDING = {
        "gpt2": "right",
        "llama": "left",
        "gemma": "left",
        "opt": "right",
        "phi": "left",
        "bert_mlm": "right",
        "roberta_mlm": "right",
        "deberta_mlm": "right",
    }

    def __init__(self, device: str, architecture: str):
        self.batch_size = 1
        self.defense_noise = None
        self.parallel = 10_000
        self.l2_span_thresh = 1e-3
        self.l1_span_thresh = self._L1_THRESHOLDS.get(architecture, 1e-3)
        self.distinct_thresh = 0.7
        self.reduce_incorrect = 0
        self.l2_std_thrs = 5
        self.p1_std_thrs = 5
        self.pad = self._PADDING.get(architecture, "right")
        self.dist_norm = "l2"
        self.device = device
        self.n_layers = 2
        self.rank_tol = self._RANK_TOLERANCES.get(architecture)
        self.rank_cutoff = 20
        self.max_ids = -1
        self.max_len = 1e10
        self.l1_filter = "maxB"
        self.l2_filter = "non-overlap"
        self.maxC = 10_000_000


def _causal_mask(sequence_length: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(1, 1, sequence_length, sequence_length, dtype=dtype, device=device)
    upper = torch.ones(
        sequence_length,
        sequence_length,
        device=device,
        dtype=torch.bool,
    ).triu(diagonal=1)
    return mask.masked_fill(upper, float("-inf"))


def _patch_opt(decoder: Any) -> None:
    def get_hidden_states(self, input_ids, attention_mask=None, n_layers=1):
        del attention_mask
        inputs = self.embed_tokens(input_ids)
        length = input_ids.shape[1]
        positions = torch.arange(length, dtype=torch.long, device=input_ids.device) + 2
        hidden = inputs + self.embed_positions.weight[positions].unsqueeze(0)
        if getattr(self, "project_in", None) is not None:
            hidden = self.project_in(hidden)
        mask = _causal_mask(length, hidden.dtype, input_ids.device)
        states = []
        for index, layer in enumerate(self.layers):
            states.append(layer.self_attn_layer_norm(hidden))
            if index >= n_layers:
                return states[1:]
            hidden = layer(hidden, attention_mask=mask)[0]
        return states[1:]

    decoder.get_hidden_states = types.MethodType(get_hidden_states, decoder)


def _patch_phi(phi_model: Any) -> None:
    def get_hidden_states(self, input_ids, attention_mask=None, n_layers=1):
        del attention_mask
        hidden = self.embed_tokens(input_ids)
        mask = _causal_mask(input_ids.shape[1], hidden.dtype, input_ids.device)
        states = []
        for index, layer in enumerate(self.layers):
            states.append(layer.input_layernorm(hidden))
            if index >= n_layers:
                return states[1:]
            hidden = layer(hidden, attention_mask=mask)[0]
        return states[1:]

    phi_model.get_hidden_states = types.MethodType(get_hidden_states, phi_model)


def _patch_gemma(gemma_model: Any) -> None:
    def get_hidden_states(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        n_layers=2,
    ):
        if input_ids is None:
            raise ValueError("input_ids required")
        inputs = self.embed_tokens(input_ids)
        hidden = inputs * torch.tensor(self.config.hidden_size**0.5, dtype=inputs.dtype)
        length = input_ids.shape[1]
        if position_ids is None:
            position_ids = torch.arange(
                length, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)
        cache_position = torch.arange(length, device=input_ids.device)
        mask = self._update_causal_mask(
            attention_mask,
            inputs,
            cache_position,
            past_key_values=None,
            output_attentions=False,
        )
        states = []
        for index, layer in enumerate(self.layers):
            states.append(layer.input_layernorm(hidden))
            if index >= n_layers:
                return states[1:]
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_ids=position_ids,
                use_cache=False,
                cache_position=cache_position,
            )[0]
        return states[1:]

    gemma_model.get_hidden_states = types.MethodType(get_hidden_states, gemma_model)


def _patch_roberta(roberta: Any) -> None:
    def encoder_states(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=False,
        n_layers=2,
    ):
        states = []
        for index, layer in enumerate(self.layer):
            states.append(hidden_states)
            if index >= n_layers:
                continue
            layer_mask = head_mask[index] if head_mask is not None else None
            past = past_key_values[index] if past_key_values is not None else None
            output = layer(
                hidden_states,
                attention_mask,
                layer_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past,
                output_attentions,
            )
            hidden_states = output[0]
        return states

    roberta.encoder.get_hidden_states_encoder = types.MethodType(encoder_states, roberta.encoder)

    def model_states(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        n_layers=2,
    ):
        del use_cache
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds")
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("Specify input_ids or inputs_embeds")
        batch_size, sequence_length = input_shape
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        embeddings = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, sequence_length), device=device)
        extended = self.get_extended_attention_mask(attention_mask, input_shape)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        return self.encoder.get_hidden_states_encoder(
            embeddings,
            attention_mask=extended,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_attentions=self.config.output_attentions,
            n_layers=n_layers,
        )[1:]

    roberta.get_hidden_states = types.MethodType(model_states, roberta)


class DagerModelWrapper:
    """Map supported Hugging Face architectures to official DAGER's API."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        architecture: str,
        device: str,
        args: DagerArgs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.arch = architecture
        self.args = args
        if architecture == "gpt2":
            self.start_token = None
            self.eos_token = model.config.eos_token_id
            self.pad_token = model.config.eos_token_id
            self.emb_size = model.config.n_embd
            self.embeddings_weight_nopos = model.transformer.wte.weight.unsqueeze(0)
            self._patch(model, add_partial_forward_gpt2, model.transformer)
        elif architecture == "llama":
            self.start_token = tokenizer.bos_token_id
            self.eos_token = tokenizer.eos_token_id
            self.pad_token = (
                tokenizer.unk_token_id
                if tokenizer.unk_token_id is not None
                else tokenizer.eos_token_id
            )
            self.emb_size = model.config.hidden_size
            self.embeddings_weight_nopos = model.model.embed_tokens.weight.unsqueeze(0)
            self._patch(model, add_partial_forward_llama, model.model)
        elif architecture == "opt":
            self.start_token = None
            self.eos_token = model.config.eos_token_id
            self.pad_token = model.config.eos_token_id
            self.emb_size = model.config.hidden_size
            self.embeddings_weight_nopos = model.model.decoder.embed_tokens.weight.unsqueeze(0)
            self._patch(model, _patch_opt, model.model.decoder)
        elif architecture == "phi":
            self.start_token = None
            self.eos_token = model.config.eos_token_id
            self.pad_token = model.config.eos_token_id
            self.emb_size = model.config.hidden_size
            self.embeddings_weight_nopos = model.model.embed_tokens.weight.unsqueeze(0)
            self._patch(model, _patch_phi, model.model)
        elif architecture == "gemma":
            self.start_token = tokenizer.bos_token_id
            self.eos_token = tokenizer.eos_token_id
            self.pad_token = (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            )
            self.emb_size = model.config.hidden_size
            self.embeddings_weight_nopos = model.model.embed_tokens.weight.unsqueeze(0)
            self._patch(model, _patch_gemma, model.model)
        elif architecture in ("bert_mlm", "roberta_mlm", "deberta_mlm"):
            self.start_token = tokenizer.cls_token_id
            self.eos_token = tokenizer.sep_token_id
            self.pad_token = tokenizer.pad_token_id or 0
            self.emb_size = model.config.hidden_size
            encoder = getattr(
                model,
                architecture.removesuffix("_mlm").replace("deberta", "deberta"),
            )
            self.embeddings_weight_nopos = encoder.embeddings.word_embeddings.weight.unsqueeze(0)
            if architecture == "bert_mlm":
                self._patch(model, add_partial_forward_bert, model.bert)
            elif architecture == "roberta_mlm":
                self._patch(model, _patch_roberta, model.roberta)
        else:
            raise ValueError(f"unsupported DAGER architecture: {architecture}")

    @staticmethod
    def _patch(model: nn.Module, patcher: Any, target: Any) -> None:
        if id(model) not in _PATCHED:
            patcher(target)
            _PATCHED.add(id(model))

    def has_rope(self) -> bool:
        return self.arch in ("llama", "phi", "gemma")

    def has_bos(self) -> bool:
        return self.start_token is not None

    def is_bert(self) -> bool:
        return self.arch == "bert_mlm"

    def get_embeddings(self, position: int = 0) -> torch.Tensor:
        device = self.args.device
        if self.arch == "bert_mlm":
            bert = self.model.bert
            vocab_size = bert.embeddings.word_embeddings.weight.shape[0]
            words = bert.embeddings.word_embeddings.weight.unsqueeze(0)
            positions = bert.embeddings.position_embeddings.weight[
                position : position + 1
            ].unsqueeze(1)
            token_types = bert.embeddings.token_type_embeddings(
                torch.zeros(1, vocab_size, dtype=torch.long, device=device)
            )
            return bert.embeddings.LayerNorm(words + positions + token_types).float()
        if self.arch == "roberta_mlm":
            roberta = self.model.roberta
            words = roberta.embeddings.word_embeddings.weight.unsqueeze(0)
            positions = roberta.embeddings.position_embeddings.weight[
                position : position + 1
            ].unsqueeze(1)
            return roberta.embeddings.LayerNorm(words + positions).float()
        if self.arch == "deberta_mlm":
            deberta = self.model.deberta
            words = deberta.embeddings.word_embeddings.weight.unsqueeze(0)
            layer_norm = getattr(deberta.embeddings, "LayerNorm", None)
            return layer_norm(words).float() if layer_norm is not None else words.float()
        words = self.embeddings_weight_nopos.to(device)
        if self.arch == "gpt2":
            positions = self.model.transformer.wpe.weight[position : position + 1, None, :]
            return self.model.transformer.h[0].ln_1(words + positions).float()
        if self.arch == "opt":
            positions = self.model.model.decoder.embed_positions.weight[
                position + 2 : position + 3, None, :
            ]
            return (
                self.model.model.decoder.layers[0].self_attn_layer_norm(words + positions).float()
            )
        return self.model.model.layers[0].input_layernorm(words).float()

    def get_layer_inputs(
        self,
        sentences: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        layers: int = 1,
    ) -> list[torch.Tensor]:
        device = self.args.device
        sentences = sentences.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if self.arch == "gpt2":
            result = self.model.transformer.get_hidden_states(
                input_ids=sentences,
                attention_mask=attention_mask,
                n_layers=layers,
            )
        elif self.arch in ("llama", "gemma"):
            positions = (
                torch.arange(sentences.size(1), dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(sentences.size(0), -1)
            )
            result = self.model.model.get_hidden_states(
                input_ids=sentences,
                position_ids=positions,
                attention_mask=attention_mask,
                n_layers=layers,
            )
        elif self.arch == "opt":
            result = self.model.model.decoder.get_hidden_states(
                input_ids=sentences,
                attention_mask=attention_mask,
                n_layers=layers,
            )
        elif self.arch == "phi":
            result = self.model.model.get_hidden_states(
                input_ids=sentences,
                attention_mask=attention_mask,
                n_layers=layers,
            )
        elif self.arch == "bert_mlm":
            types_tensor = (
                token_type_ids.to(device)
                if token_type_ids is not None
                else torch.zeros_like(sentences)
            )
            mask = attention_mask if attention_mask is not None else torch.ones_like(sentences)
            result = self.model.bert.get_hidden_states(
                input_ids=sentences,
                attention_mask=mask,
                token_type_ids=types_tensor,
                n_layers=layers,
            )
        elif self.arch == "roberta_mlm":
            result = self.model.roberta.get_hidden_states(
                input_ids=sentences,
                attention_mask=attention_mask,
                n_layers=layers,
            )
        else:
            raise ValueError(f"no partial forward for {self.arch}")
        return [state.float() for state in result]

    def get_matrices_expansions(
        self,
        gradients: dict[str, torch.Tensor | None],
        B: int | None = None,
        tol: float | None = None,
        min_B: int = 0,
    ) -> tuple[int | None, list[torch.Tensor]]:
        q_gradients = []
        for layer in range(64):
            gradient = self._get_q_gradient(gradients, layer)
            if gradient is not None:
                q_gradients.append(gradient)
            if len(q_gradients) >= max(self.args.n_layers, 10):
                break
        if not q_gradients:
            return None, []
        if B is None:
            B = max(
                int(np.linalg.matrix_rank(gradient.float().cpu().numpy(), tol=tol))
                for gradient in q_gradients[:10]
            )
        B = max(
            max(1, min_B),
            min(B, self.emb_size - self.args.rank_cutoff),
        )
        spans = [
            get_layer_decomp(gradient, B=B, tol=tol)[1].to(self.args.device)
            for gradient in q_gradients[: self.args.n_layers]
        ]
        return B, spans

    def _get_q_gradient(
        self,
        gradients: dict[str, torch.Tensor | None],
        layer: int,
    ) -> torch.Tensor | None:
        if self.arch == "gpt2":
            gradient = gradients.get(f"transformer.h.{layer}.attn.c_attn.weight")
            return gradient.float().T if gradient is not None else None
        if self.arch in ("llama", "phi", "gemma"):
            for prefix in (
                "model.layers",
                "base_model.model.model.layers",
            ):
                gradient = gradients.get(f"{prefix}.{layer}.self_attn.q_proj.weight")
                if gradient is not None:
                    return gradient.float()
        if self.arch == "opt":
            gradient = gradients.get(f"model.decoder.layers.{layer}.self_attn.q_proj.weight")
            return gradient.float() if gradient is not None else None
        prefix = self.arch.removesuffix("_mlm")
        if prefix in ("bert", "roberta"):
            gradient = gradients.get(f"{prefix}.encoder.layer.{layer}.attention.self.query.weight")
            return gradient.float() if gradient is not None else None
        if self.arch == "deberta_mlm":
            path = f"deberta.encoder.layer.{layer}.attention."
            for name, gradient in gradients.items():
                if (
                    gradient is not None
                    and name.startswith(path)
                    and name.endswith(".weight")
                    and ("query" in name.lower() or "q_proj" in name.lower())
                ):
                    return gradient.float()
        return None


def filter_l1(
    args: DagerArgs,
    wrapper: DagerModelWrapper,
    spans: list[torch.Tensor],
    sequence_length: int | None = None,
):
    positions: list[Any] = []
    candidate_ids: list[list[int]] = []
    candidate_types: list[list[int]] = []
    sentence_ends: list[tuple[int, int]] = []
    position = 0
    while sequence_length is None or position < sequence_length:
        with torch.no_grad():
            embeddings = wrapper.get_embeddings(position)
        _, ids_tensor = get_top_B_in_span(
            spans[0],
            embeddings,
            args.batch_size,
            args.l1_span_thresh,
            args.dist_norm,
        )
        types_tensor = torch.zeros_like(ids_tensor)
        positions_tensor = torch.ones_like(ids_tensor) * position
        ids = ids_tensor.tolist()
        candidate_types.append(types_tensor.tolist())
        if not ids or position > args.max_len:
            break
        while wrapper.eos_token in ids:
            eos_index = ids.index(wrapper.eos_token)
            sentence_ends.append((position, candidate_types[-1][eos_index]))
            ids.pop(eos_index)
            candidate_types[-1].pop(eos_index)
        candidate_ids.append(ids)
        positions.extend(positions_tensor.tolist())
        position += 1
        if wrapper.has_rope():
            break
    return positions, candidate_ids, candidate_types, sentence_ends


def _best_encoder_candidate(sentences: list[list[int]], scores: list[Any]) -> list[int] | None:
    if not sentences:
        return None
    return min(
        zip(sentences, scores, strict=True),
        key=lambda pair: float(pair[1].item() if hasattr(pair[1], "item") else pair[1]),
    )[0]


def run_dager(
    model: nn.Module,
    tokenizer: Any,
    sentence: str,
    device: str,
    defense: GradientDefense | None = None,
    max_tokens: int = 32,
    cls_label: int | None = None,
    seq_attack_model: nn.Module | None = None,
) -> AttackResult:
    """Run official DAGER L1/L2 filtering with embedding-row fallback."""
    model.eval()
    base_model = unwrap_peft_model(model)
    architecture = detect_architecture(base_model)
    is_lora = model is not base_model
    gradient_base = None
    if seq_attack_model is not None and cls_label is not None:
        seq_attack_model.eval()
        gradient_base = unwrap_peft_model(seq_attack_model)

    parameter_count = sum(p.numel() for p in base_model.parameters())
    skip_upcast = parameter_count > 10e9
    model_bf16 = next(base_model.parameters()).dtype == torch.bfloat16
    gradient_bf16 = (
        next(gradient_base.parameters()).dtype == torch.bfloat16
        if gradient_base is not None
        else False
    )
    if model_bf16 and not skip_upcast:
        base_model.float()
    if gradient_bf16:
        gradient_base.float()
    try:
        return _run_dager(
            model,
            base_model,
            tokenizer,
            sentence,
            device,
            defense,
            max_tokens,
            architecture,
            is_lora,
            gradient_base,
            cls_label,
            skip_upcast,
        )
    finally:
        if model_bf16 and not skip_upcast:
            base_model.bfloat16()
        if gradient_bf16 and gradient_base is not None:
            gradient_base.bfloat16()


def _run_dager(
    model: nn.Module,
    base_model: nn.Module,
    tokenizer: Any,
    sentence: str,
    device: str,
    defense: GradientDefense | None,
    max_tokens: int,
    architecture: str,
    is_lora: bool,
    gradient_base: nn.Module | None,
    cls_label: int | None,
    skip_upcast: bool,
) -> AttackResult:
    encoded = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = encoded["input_ids"][:, :max_tokens]
    attention_mask = encoded["attention_mask"][:, :max_tokens]
    if "token_type_ids" in encoded:
        encoded["token_type_ids"] = encoded["token_type_ids"][:, :max_tokens]
    length = input_ids.shape[1]
    true_ids = input_ids.squeeze(0).cpu()
    use_classification = gradient_base is not None and cls_label is not None
    backward_model = gradient_base if use_classification else model
    dager_model = gradient_base if use_classification else base_model
    assert backward_model is not None
    assert dager_model is not None

    backward_model.zero_grad()
    embeddings = get_input_embeddings(backward_model)
    with grad_context(embeddings.weight):
        if use_classification:
            kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": torch.tensor([cls_label], device=device, dtype=torch.long),
            }
            if "token_type_ids" in encoded:
                kwargs["token_type_ids"] = encoded["token_type_ids"]
            loss = backward_model(**kwargs).loss
        elif is_masked_lm(model):
            masked, labels = apply_mlm_labels(input_ids, attention_mask, tokenizer)
            loss = model(
                input_ids=masked,
                attention_mask=attention_mask,
                labels=labels,
            ).loss
        else:
            labels = torch.where(
                attention_mask.bool(),
                input_ids,
                torch.full_like(input_ids, -100),
            )
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            ).loss
        loss.backward()
        apply_defense(defense, backward_model if use_classification else None)
        embedding_gradient = (
            embeddings.weight.grad.detach().cpu().float().clone()
            if embeddings.weight.grad is not None
            else torch.zeros(
                backward_model.config.vocab_size,
                get_hidden_size(backward_model),
            )
        )

    q_suffixes = (
        "attn.c_attn.weight",
        "self_attn.q_proj.weight",
        "attention.self.query.weight",
        "attention.self.query_proj.weight",
    )
    gradients = {
        name: (
            parameter.grad.detach().cpu().float().clone() if parameter.grad is not None else None
        )
        for name, parameter in backward_model.named_parameters()
        if any(name.endswith(suffix) for suffix in q_suffixes)
    }
    backward_model.zero_grad()

    if not is_lora and architecture != "deberta_mlm":
        try:
            args = DagerArgs(device, architecture)
            if skip_upcast:
                args.l1_span_thresh = 4e-3
                args.l2_span_thresh = 5e-2
                args.rank_tol = None
            if defense is not None and not is_aegis_defense(defense):
                args.l1_span_thresh = 1e-2
                args.l2_span_thresh = 1e-2
            wrapper = DagerModelWrapper(dager_model, tokenizer, architecture, device, args)
            minimum_rank = 0 if architecture == "bert_mlm" else length
            _, spans = wrapper.get_matrices_expansions(gradients, min_B=minimum_rank)
            if len(spans) < 2:
                raise ValueError("could not extract two Q-gradient spans")
            _, candidate_ids, _, sentence_ends = filter_l1(args, wrapper, spans, length)
            if not candidate_ids:
                raise ValueError("DAGER L1 filter returned no candidates")
            if architecture == "bert_mlm" and not sentence_ends:
                sentence_ends = [(len(candidate_ids), 0)]
            max_ids = -1
            if len(candidate_ids[0]) > 100_000:
                max_ids = args.max_ids if args.max_ids > 0 else 10_000
            if skip_upcast and architecture != "bert_mlm":
                max_ids = 512 if max_ids < 0 else min(max_ids, 512)
            if defense is not None and not is_aegis_defense(defense):
                max_ids = 50 if max_ids < 0 else min(max_ids, 50)
            if os.environ.get("AEGIS_DAGER_DEBUG", "").lower() in (
                "1",
                "yes",
                "true",
            ):
                print(
                    f"[DAGER] arch={architecture} T={length} "
                    f"L1={args.l1_span_thresh} candidates="
                    f"{len(candidate_ids[0])} max_ids={max_ids}",
                    flush=True,
                )
            with (
                io.StringIO() as sink,
                contextlib.redirect_stdout(sink),
                contextlib.redirect_stderr(sink),
                torch.no_grad(),
            ):
                if architecture == "bert_mlm":
                    predicted: list[list[int]] = []
                    predicted_scores: list[Any] = []
                    for end, token_type in sentence_ends:
                        correct = []
                        approximate = []
                        approximate_scores = []
                        for candidate, score in zip(predicted, predicted_scores, strict=True):
                            value = float(score.item() if hasattr(score, "item") else score)
                            if value < args.l2_span_thresh:
                                correct.append(candidate)
                            else:
                                approximate.append(candidate)
                                approximate_scores.append(score)
                        new_candidates, new_scores = filter_encoder(
                            args,
                            wrapper,
                            spans[1],
                            end,
                            token_type,
                            candidate_ids,
                            correct,
                            approximate,
                            approximate_scores,
                            args.batch_size if args.l1_filter == "maxB" else -1,
                            args.batch_size,
                        )
                        predicted += new_candidates
                        predicted_scores += new_scores
                    best = _best_encoder_candidate(predicted, predicted_scores)
                    if not best:
                        raise ValueError("DAGER encoder returned no candidate")
                    recovered = torch.tensor(best, dtype=torch.long)
                    recovery_kind = "span_encoder"
                else:
                    predicted, _, incorrect, _ = filter_decoder(
                        args,
                        wrapper,
                        spans,
                        candidate_ids,
                        max_ids=max_ids,
                    )
                    best = (
                        predicted[0]
                        if predicted
                        else incorrect[0]
                        if incorrect and incorrect[0]
                        else None
                    )
                    if best is None:
                        raise ValueError("DAGER decoder returned no candidate")
                    recovered = torch.tensor(best, dtype=torch.long)
                    recovery_kind = "span_decoder"
            result = compute_text_metrics(recovered, true_ids, tokenizer)
            result["attack"] = "dager"
            result["dager_recovery"] = recovery_kind
            return result
        except Exception as error:
            if defense is None or not is_aegis_defense(defense):
                print(f"[WARN] DAGER span recovery failed: {error}", flush=True)

    norms = embedding_gradient.norm(dim=1)
    combined = 1.0 - norms / (norms.max() + 1e-10)
    candidates = combined.topk(length * 3, largest=False).indices
    token_set = candidates[norms[candidates].topk(length).indices]
    recovered = token_set[embedding_gradient[token_set].norm(dim=1).argsort(descending=True)]
    result = compute_text_metrics(recovered, true_ids, tokenizer)
    result["attack"] = "dager"
    result["dager_recovery"] = "embedding"
    return result


# Legacy private names retained for SOMP and downstream experiment compatibility.
_DagerArgs = DagerArgs
_DagerWrapper = DagerModelWrapper
_dager_filter_l1 = filter_l1

__all__ = [
    "DagerArgs",
    "DagerModelWrapper",
    "filter_l1",
    "run_dager",
]
