"""
Week 1: LLM Inference - Python Implementation

This module demonstrates LLM inference patterns equivalent to the
Rust `realizar` crate. Covers GGUF model metadata, tokenization,
text generation, and prompt templates.

Course 4, Week 1: Foundation Models + Prompt Engineering
"""

from .llm_inference import (
    FinishReason,
    GenerationConfig,
    GenerationOutput,
    GgufMetadata,
    KVCache,
    LlmError,
    LlmModel,
    PromptTemplate,
    QuantizationType,
    Tokenizer,
)

__all__ = [
    "QuantizationType",
    "GgufMetadata",
    "Tokenizer",
    "GenerationConfig",
    "KVCache",
    "LlmModel",
    "GenerationOutput",
    "FinishReason",
    "PromptTemplate",
    "LlmError",
]
