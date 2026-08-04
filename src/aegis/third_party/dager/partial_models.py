"""Partial-forward patches copied from the active official DAGER utilities.

The module was moved into the AEGIS package; no repository-level modules are
imported. See ``LICENSES/dager/LICENSE.md``.
"""

from __future__ import annotations

import types
from typing import Any

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask_for_sdpa,
)


def add_partial_forward_gpt2(model: Any) -> None:
    def get_hidden_states(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        n_layers=2,
    ):
        del attention_mask
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
            input_ids = input_ids.long()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("Specify input_ids or inputs_embeds")
        if position_ids is None:
            position_ids = torch.arange(input_shape[-1], dtype=torch.long, device=device).unsqueeze(
                0
            )
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        hidden_states = inputs_embeds + self.wpe(position_ids)
        if token_type_ids is not None:
            hidden_states = hidden_states + self.wte(token_type_ids.long())
        hidden_states = self.drop(hidden_states)
        all_hidden_states = []
        for index, block in enumerate(self.h):
            all_hidden_states.append(block.ln_1(hidden_states))
            if index >= n_layers or index == len(self.h) - 1:
                return all_hidden_states[1:]
            output = block(hidden_states, use_cache=False)
            hidden_states = output if isinstance(output, torch.Tensor) else output[0]
        return all_hidden_states[1:]

    model.get_hidden_states = types.MethodType(get_hidden_states, model)


def add_partial_forward_bert(model: Any) -> None:
    def get_hidden_states_encoder(
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
        all_hidden_states = []
        for index, layer in enumerate(self.layer):
            all_hidden_states.append(hidden_states)
            if index >= n_layers:
                continue
            layer_head_mask = head_mask[index] if head_mask is not None else None
            past_key_value = past_key_values[index] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                outputs = layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            hidden_states = outputs[0]
        return all_hidden_states

    model.encoder.get_hidden_states_encoder = types.MethodType(
        get_hidden_states_encoder, model.encoder
    )

    def get_hidden_states(
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
        output_attentions = self.config.output_attentions
        use_cache = (
            use_cache
            if use_cache is not None
            else self.config.use_cache
            if self.config.is_decoder
            else False
        )
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds")
        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("Specify input_ids or inputs_embeds")
        batch_size, sequence_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                token_type_ids = self.embeddings.token_type_ids[:, :sequence_length].expand(
                    batch_size, sequence_length
                )
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_length,
        )
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, sequence_length + past_length), device=device)
        use_sdpa = (
            self.attn_implementation == "sdpa"
            and self.position_embedding_type == "absolute"
            and head_mask is None
            and not output_attentions
        )
        if use_sdpa and attention_mask.dim() == 2:
            if self.config.is_decoder:
                extended_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    input_shape,
                    embedding_output,
                    past_length,
                )
            else:
                extended_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask,
                    embedding_output.dtype,
                    tgt_len=sequence_length,
                )
        else:
            extended_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        encoder_extended_mask = None
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_shape = encoder_hidden_states.size()[:-1]
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_shape, device=device)
            encoder_extended_mask = self.invert_attention_mask(encoder_attention_mask)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        return self.encoder.get_hidden_states_encoder(
            embedding_output,
            attention_mask=extended_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            n_layers=n_layers,
        )[1:]

    model.get_hidden_states = types.MethodType(get_hidden_states, model)


def add_partial_forward_llama(model: Any) -> None:
    def get_hidden_states(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        return_dict=None,
        cache_position=None,
        n_layers=2,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = (
                DynamicCache()
                if past_key_values is None
                else DynamicCache.from_legacy_cache(past_key_values)
            )
        if cache_position is None:
            seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                seen,
                seen + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = []
        for index, layer in enumerate(self.layers):
            all_hidden_states.append(layer.input_layernorm(hidden_states))
            if index >= n_layers:
                return all_hidden_states[1:]
            outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = outputs[0]
        all_hidden_states.append(self.norm(hidden_states))
        return all_hidden_states[1:]

    model.get_hidden_states = types.MethodType(get_hidden_states, model)


__all__ = [
    "add_partial_forward_bert",
    "add_partial_forward_gpt2",
    "add_partial_forward_llama",
]
