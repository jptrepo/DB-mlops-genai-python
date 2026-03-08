"""
Week 3: Fine-Tuning - Python Implementation

This module demonstrates LLM fine-tuning patterns equivalent to the
Rust `entrenar` crate. Covers LoRA, QLoRA, and training configuration.

Course 4, Week 3: Fine-Tuning + Production + Capstone
"""

from .finetuning import (
    BiasMode,
    CheckpointError,
    ComputeDtype,
    ConfigError,
    DataError,
    DatasetFormat,
    LoraConfig,
    LoraWeight,
    MergeMethod,
    QloraConfig,
    QuantType,
    Trainer,
    TrainingConfig,
    TrainingDataset,
    TrainingError,
    TrainingHistory,
    TrainingMetrics,
    TrainingResult,
    TrainingSample,
    merge_adapters,
)

__all__ = [
    "LoraConfig",
    "BiasMode",
    "QloraConfig",
    "QuantType",
    "ComputeDtype",
    "TrainingSample",
    "DatasetFormat",
    "TrainingDataset",
    "TrainingConfig",
    "TrainingMetrics",
    "TrainingHistory",
    "Trainer",
    "TrainingResult",
    "MergeMethod",
    "LoraWeight",
    "merge_adapters",
    "TrainingError",
    "ConfigError",
    "DataError",
    "CheckpointError",
]
