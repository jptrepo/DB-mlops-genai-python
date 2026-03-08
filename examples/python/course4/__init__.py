"""
Python Examples for Course 4: GenAI Engineering

This package provides Python implementations equivalent to the
Sovereign AI Stack (Rust) examples. These demonstrate the same
concepts as the Rust code but in Python.

Modules:
- week1_llm: LLM inference (realizar equivalent)
- week2_rag: RAG pipelines (trueno-rag equivalent)
- week3_finetuning: Fine-tuning (entrenar equivalent)
"""

from . import week1_llm, week2_rag, week3_finetuning

__all__ = [
    "week1_llm",
    "week2_rag",
    "week3_finetuning",
]
