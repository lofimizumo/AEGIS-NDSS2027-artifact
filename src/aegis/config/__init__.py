"""Configuration primitives and model/dataset registries for AEGIS."""

from .registries import (
    DATASET_LABEL_NAMES,
    DATASET_TYPE,
    LORA_TARGETS,
    MODEL_ARCH,
    MODEL_BATCH_SIZE,
    MODEL_HF_FALLBACKS,
    MODEL_N_ATTACK,
    MODEL_PATHS,
    MODEL_USE_BF16,
    MODEL_USE_LORA,
    Architecture,
    DatasetType,
    build_model_paths,
    supports_dager_attack,
)

__all__ = [
    "Architecture",
    "DATASET_LABEL_NAMES",
    "DATASET_TYPE",
    "DatasetType",
    "LORA_TARGETS",
    "MODEL_ARCH",
    "MODEL_BATCH_SIZE",
    "MODEL_HF_FALLBACKS",
    "MODEL_N_ATTACK",
    "MODEL_PATHS",
    "MODEL_USE_BF16",
    "MODEL_USE_LORA",
    "build_model_paths",
    "supports_dager_attack",
]
