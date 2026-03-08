"""
Week 2: RAG Pipeline - Python Implementation

This module demonstrates RAG (Retrieval-Augmented Generation) patterns
equivalent to the Rust `trueno-rag` crate. Covers chunking, embeddings,
retrieval, and reranking.

Course 4, Week 2: Vector Search + RAG Pipelines
"""

from .rag_pipeline import (
    Chunk,
    Chunker,
    ChunkerConfig,
    Document,
    Embedding,
    EmbeddingModel,
    HybridSearch,
    IndexedChunk,
    RagError,
    RagPipeline,
    RagResponse,
    Reranker,
    RetrievalResult,
    VectorIndex,
)

__all__ = [
    "Document",
    "Chunk",
    "ChunkerConfig",
    "Chunker",
    "Embedding",
    "EmbeddingModel",
    "VectorIndex",
    "IndexedChunk",
    "RetrievalResult",
    "Reranker",
    "RagPipeline",
    "RagResponse",
    "HybridSearch",
    "RagError",
]
