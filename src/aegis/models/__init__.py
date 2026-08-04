"""Model loading and architecture introspection for AEGIS."""

from .architecture import (
    detect_architecture,
    get_block_attention_module,
    get_final_layer_norm,
    get_hidden_size,
    get_input_embedding,
    get_transformer_blocks,
    is_masked_lm,
    unwrap_peft,
)
from .loading import load_attack_compatible_gpt2, load_model, warmup_bert_sequence_classifier

__all__ = [
    "load_attack_compatible_gpt2",
    "detect_architecture",
    "get_block_attention_module",
    "get_final_layer_norm",
    "get_hidden_size",
    "get_input_embedding",
    "get_transformer_blocks",
    "is_masked_lm",
    "load_model",
    "unwrap_peft",
    "warmup_bert_sequence_classifier",
]
