"""
RAG Pipeline with Python (trueno-rag equivalent)

Demonstrates RAG (Retrieval-Augmented Generation) patterns including:
- Document chunking with overlap
- Dense embeddings
- Vector similarity search
- Cross-encoder reranking
- Hybrid search (vector + keyword)

Course 4, Week 2: Vector Search + RAG Pipelines
"""

import math
from dataclasses import dataclass, field
from typing import Optional


class RagError(Exception):
    """Base exception for RAG operations."""

    pass


class ChunkingError(RagError):
    """Error during document chunking."""

    pass


class EmbeddingError(RagError):
    """Error during embedding generation."""

    pass


class RetrievalError(RagError):
    """Error during retrieval."""

    pass


class RerankingError(RagError):
    """Error during reranking."""

    pass


# ============================================================================
# Document and Chunking
# ============================================================================


@dataclass
class Document:
    """A document to be indexed."""

    id: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def new(cls, doc_id: str, content: str) -> "Document":
        """Create a new document."""
        return cls(id=doc_id, content=content)

    def with_metadata(self, key: str, value: str) -> "Document":
        """Add metadata and return self for chaining."""
        self.metadata[key] = value
        return self


@dataclass
class Chunk:
    """A chunk of a document."""

    id: str
    doc_id: str
    text: str
    start_idx: int
    end_idx: int
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ChunkerConfig:
    """Configuration for document chunking."""

    chunk_size: int = 512
    chunk_overlap: int = 50
    separator: str = " "

    def with_size(self, size: int) -> "ChunkerConfig":
        """Set chunk size and return self for chaining."""
        self.chunk_size = size
        return self

    def with_overlap(self, overlap: int) -> "ChunkerConfig":
        """Set chunk overlap and return self for chaining."""
        self.chunk_overlap = overlap
        return self


class Chunker:
    """Document chunker with configurable overlap."""

    def __init__(self, config: Optional[ChunkerConfig] = None) -> None:
        self.config = config or ChunkerConfig()

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into overlapping chunks."""
        words = document.content.split(self.config.separator)
        chunks = []
        word_idx = 0
        chunk_num = 0

        while word_idx < len(words):
            end_idx = min(word_idx + self.config.chunk_size, len(words))
            chunk_words = words[word_idx:end_idx]
            text = self.config.separator.join(chunk_words)

            chunks.append(
                Chunk(
                    id=f"{document.id}-chunk-{chunk_num}",
                    doc_id=document.id,
                    text=text,
                    start_idx=word_idx,
                    end_idx=end_idx,
                    metadata=document.metadata.copy(),
                )
            )

            if end_idx >= len(words):
                break

            word_idx += self.config.chunk_size - self.config.chunk_overlap
            chunk_num += 1

        return chunks

    def get_config(self) -> ChunkerConfig:
        """Get chunker configuration."""
        return self.config


# ============================================================================
# Embeddings
# ============================================================================


@dataclass
class Embedding:
    """A vector embedding."""

    vector: list[float]
    dimension: int = field(init=False)

    def __post_init__(self) -> None:
        self.dimension = len(self.vector)

    @classmethod
    def zeros(cls, dimension: int) -> "Embedding":
        """Create a zero embedding."""
        return cls(vector=[0.0] * dimension)

    def cosine_similarity(self, other: "Embedding") -> float:
        """Calculate cosine similarity with another embedding."""
        if self.dimension != other.dimension:
            return 0.0

        dot_product = sum(a * b for a, b in zip(self.vector, other.vector))
        norm_a = math.sqrt(sum(x * x for x in self.vector))
        norm_b = math.sqrt(sum(x * x for x in other.vector))

        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def normalize(self) -> None:
        """Normalize the embedding in place."""
        norm = math.sqrt(sum(x * x for x in self.vector))
        if norm > 1e-10:
            self.vector = [v / norm for v in self.vector]


class EmbeddingModel:
    """Embedding model for generating vector representations."""

    def __init__(self, dimension: int, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.dimension = dimension
        self.model_name = model_name

    def embed(self, text: str) -> Embedding:
        """Generate embedding for text (simulated)."""
        vector = [0.0] * self.dimension

        for i, word in enumerate(text.split()):
            for j, char in enumerate(word):
                idx = (ord(char) * (i + 1) + j) % self.dimension
                vector[idx] += 0.1

        # Normalize
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 1e-10:
            vector = [v / norm for v in vector]

        return Embedding(vector=vector)

    def embed_batch(self, texts: list[str]) -> list[Embedding]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension

    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name


# ============================================================================
# Vector Index
# ============================================================================


@dataclass
class IndexedChunk:
    """A chunk with its embedding."""

    chunk: Chunk
    embedding: Embedding


@dataclass
class RetrievalResult:
    """Result from retrieval."""

    chunk: Chunk
    score: float
    rank: int


class VectorIndex:
    """Vector index for similarity search."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.chunks: list[IndexedChunk] = []

    def add(self, chunk: Chunk, embedding: Embedding) -> None:
        """Add a chunk with its embedding to the index."""
        if embedding.dimension != self.dimension:
            raise EmbeddingError(f"Dimension mismatch: {embedding.dimension} vs {self.dimension}")
        self.chunks.append(IndexedChunk(chunk=chunk, embedding=embedding))

    def search(self, query_embedding: Embedding, top_k: int) -> list[RetrievalResult]:
        """Search for similar chunks."""
        scored = [
            (i, query_embedding.cosine_similarity(ic.embedding)) for i, ic in enumerate(self.chunks)
        ]

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, score) in enumerate(scored[:top_k]):
            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx].chunk,
                    score=score,
                    rank=rank + 1,
                )
            )

        return results

    def __len__(self) -> int:
        return len(self.chunks)

    def is_empty(self) -> bool:
        return len(self.chunks) == 0


# ============================================================================
# Reranker
# ============================================================================


class Reranker:
    """Cross-encoder reranker for improving retrieval quality."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Rerank results based on query relevance."""
        # Handle empty query or results
        if not results:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())

        # If no query words, just return top results
        if not query_words:
            return results[:top_k]

        scored = []
        for result in results:
            chunk_lower = result.chunk.text.lower()
            chunk_words = set(chunk_lower.split())
            overlap = len(query_words.intersection(chunk_words))
            # Combine vector score with keyword overlap
            score = result.score * 0.5 + (overlap / len(query_words)) * 0.5
            scored.append((result, score))

        # Sort by combined score
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for rank, (result, score) in enumerate(scored[:top_k]):
            reranked.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=score,
                    rank=rank + 1,
                )
            )

        return reranked

    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name


# ============================================================================
# RAG Pipeline
# ============================================================================


@dataclass
class RagResponse:
    """Response from RAG query."""

    question: str
    answer: str
    sources: list[str]
    context_length: int


class RagPipeline:
    """Complete RAG pipeline with chunking, embedding, and retrieval."""

    def __init__(self, embedding_dim: int = 384) -> None:
        self.chunker = Chunker()
        self.embedding_model = EmbeddingModel(embedding_dim, "all-MiniLM-L6-v2")
        self.index = VectorIndex(embedding_dim)
        self.reranker: Optional[Reranker] = None
        self.retrieval_top_k = 10
        self.rerank_top_k = 5

    def with_chunker(self, chunker: Chunker) -> "RagPipeline":
        """Set custom chunker and return self for chaining."""
        self.chunker = chunker
        return self

    def with_reranker(self, reranker: Reranker) -> "RagPipeline":
        """Set reranker and return self for chaining."""
        self.reranker = reranker
        return self

    def with_retrieval_k(self, k: int) -> "RagPipeline":
        """Set retrieval top-k and return self for chaining."""
        self.retrieval_top_k = k
        return self

    def with_rerank_k(self, k: int) -> "RagPipeline":
        """Set rerank top-k and return self for chaining."""
        self.rerank_top_k = k
        return self

    def ingest(self, documents: list[Document]) -> int:
        """Ingest documents into the pipeline."""
        chunk_count = 0

        for doc in documents:
            chunks = self.chunker.chunk(doc)
            for chunk in chunks:
                embedding = self.embedding_model.embed(chunk.text)
                self.index.add(chunk, embedding)
                chunk_count += 1

        return chunk_count

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a query."""
        query_embedding = self.embedding_model.embed(query)
        results = self.index.search(query_embedding, self.retrieval_top_k)

        if self.reranker:
            return self.reranker.rerank(query, results, self.rerank_top_k)
        else:
            return results[: self.rerank_top_k]

    def generate_context(self, results: list[RetrievalResult]) -> str:
        """Generate context string from retrieval results."""
        return "\n\n".join(r.chunk.text for r in results)

    def query(self, question: str) -> RagResponse:
        """Run full RAG query: retrieve + generate."""
        results = self.retrieve(question)

        # Handle empty results
        if not results:
            return RagResponse(
                question=question,
                answer="No relevant documents found.",
                sources=[],
                context_length=0,
            )

        context = self.generate_context(results)

        # Simulate answer generation
        if "machine learning" in context.lower():
            answer = (
                "Based on the context, machine learning is a field of AI "
                "that enables systems to learn from data."
            )
        elif "neural" in context.lower():
            answer = (
                "According to the retrieved documents, neural networks are "
                "computing systems inspired by biological neurons."
            )
        else:
            answer = f"Based on the retrieved context: {context[:100]}"

        return RagResponse(
            question=question,
            answer=answer,
            sources=[r.chunk.id for r in results],
            context_length=len(context),
        )

    def index_size(self) -> int:
        """Get number of chunks in index."""
        return len(self.index)


# ============================================================================
# Hybrid Search
# ============================================================================


class HybridSearch:
    """Hybrid search combining vector and keyword search."""

    def __init__(self, dimension: int, alpha: float = 0.7) -> None:
        self.vector_index = VectorIndex(dimension)
        self.keyword_index: dict[str, list[int]] = {}
        self.chunk_to_idx: dict[str, int] = {}  # Map chunk ID to original index
        self.alpha = max(0.0, min(1.0, alpha))  # Weight for vector vs keyword

    def add(self, chunk: Chunk, embedding: Embedding) -> None:
        """Add chunk to both vector and keyword indices."""
        idx = len(self.vector_index)
        self.chunk_to_idx[chunk.id] = idx

        # Add to keyword index
        for word in chunk.text.lower().split():
            if word not in self.keyword_index:
                self.keyword_index[word] = []
            self.keyword_index[word].append(idx)

        self.vector_index.add(chunk, embedding)

    def search(self, query: str, query_embedding: Embedding, top_k: int) -> list[RetrievalResult]:
        """Search using both vector and keyword matching."""
        # Vector search
        vector_results = self.vector_index.search(query_embedding, top_k * 2)

        # Keyword search
        query_lower = query.lower()
        query_words = query_lower.split()
        keyword_scores: dict[int, float] = {}

        if query_words:  # Skip keyword scoring if query is empty
            for word in query_words:
                if word in self.keyword_index:
                    for idx in self.keyword_index[word]:
                        keyword_scores[idx] = keyword_scores.get(idx, 0.0) + 1.0 / len(query_words)

        # Combine scores using chunk ID to get original index
        combined: dict[str, tuple[RetrievalResult, float]] = {}

        for result in vector_results:
            chunk_id = result.chunk.id
            original_idx = self.chunk_to_idx.get(chunk_id, -1)
            vector_score = result.score * self.alpha
            keyword_score = keyword_scores.get(original_idx, 0.0) * (1.0 - self.alpha)
            combined[chunk_id] = (result, vector_score + keyword_score)

        # Sort by combined score
        sorted_results = sorted(combined.values(), key=lambda x: x[1], reverse=True)

        final_results = []
        for rank, (result, score) in enumerate(sorted_results[:top_k]):
            final_results.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=score,
                    rank=rank + 1,
                )
            )

        return final_results

    def __len__(self) -> int:
        return len(self.vector_index)

    def is_empty(self) -> bool:
        return self.vector_index.is_empty()


# ============================================================================
# Demo
# ============================================================================


def main() -> None:
    """Run the RAG pipeline demo."""
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║     RAG Pipeline with Python - Course 4, Week 2               ║")
    print("║     Chunking, Embeddings, Retrieval, Reranking                ║")
    print("╚═══════════════════════════════════════════════════════════════╝")

    # Step 1: Create Documents
    print("\n📄 Step 1: Document Ingestion")
    documents = [
        Document.new(
            "doc1",
            "Machine learning is a subset of artificial intelligence that enables "
            "systems to learn and improve from experience. It uses algorithms to "
            "identify patterns in data and make predictions without being "
            "explicitly programmed.",
        ).with_metadata("source", "ml-guide"),
        Document.new(
            "doc2",
            "Deep learning is a specialized form of machine learning that uses "
            "neural networks with multiple layers. These networks can learn "
            "hierarchical representations of data, enabling them to solve "
            "complex problems.",
        ).with_metadata("source", "dl-guide"),
        Document.new(
            "doc3",
            "Natural language processing combines linguistics and machine learning "
            "to enable computers to understand human language. Applications include "
            "translation, sentiment analysis, and chatbots.",
        ).with_metadata("source", "nlp-guide"),
    ]

    print(f"   Created {len(documents)} documents")
    for doc in documents:
        word_count = len(doc.content.split())
        print(f"   - {doc.id}: {word_count} words, source={doc.metadata.get('source')}")

    # Step 2: Chunking
    print("\n✂️ Step 2: Document Chunking")
    chunker = Chunker(ChunkerConfig().with_size(20).with_overlap(5))
    print(f"   Chunk size: {chunker.config.chunk_size} words")
    print(f"   Overlap: {chunker.config.chunk_overlap} words")

    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)
        print(f"   {doc.id}: {len(chunks)} chunks")

    # Step 3: Embeddings
    print("\n🔢 Step 3: Generate Embeddings")
    embed_model = EmbeddingModel(dimension=64, model_name="all-MiniLM-L6-v2")
    print(f"   Model: {embed_model.get_model_name()}")
    print(f"   Dimension: {embed_model.get_dimension()}")

    # Show sample embedding
    sample_embedding = embed_model.embed("machine learning")
    print(f"   Sample embedding (first 8 dims): {sample_embedding.vector[:8]}")

    # Step 4: Build Index
    print("\n📊 Step 4: Build Vector Index")
    index = VectorIndex(dimension=64)
    for chunk in all_chunks:
        embedding = embed_model.embed(chunk.text)
        index.add(chunk, embedding)
    print(f"   Indexed {len(index)} chunks")

    # Step 5: Retrieval
    print("\n🔍 Step 5: Vector Retrieval")
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "What is NLP?",
    ]

    for query in queries:
        query_embedding = embed_model.embed(query)
        results = index.search(query_embedding, top_k=2)
        print(f'\n   Query: "{query}"')
        for result in results:
            preview = result.chunk.text[:60].replace("\n", " ") + "..."
            print(f"     [{result.rank}] Score: {result.score:.3f} - {preview}")

    # Step 6: Reranking
    print("\n🔄 Step 6: Reranking")
    reranker = Reranker()
    print(f"   Model: {reranker.get_model_name()}")

    query = "machine learning algorithms"
    query_embedding = embed_model.embed(query)
    initial_results = index.search(query_embedding, top_k=5)
    reranked_results = reranker.rerank(query, initial_results, top_k=3)

    print(f'\n   Query: "{query}"')
    print("   Before reranking:")
    for r in initial_results[:3]:
        print(f"     [{r.rank}] Score: {r.score:.3f}")
    print("   After reranking:")
    for r in reranked_results:
        print(f"     [{r.rank}] Score: {r.score:.3f}")

    # Step 7: Full RAG Pipeline
    print("\n🚀 Step 7: Complete RAG Pipeline")
    rag = (
        RagPipeline(embedding_dim=64)
        .with_chunker(Chunker(ChunkerConfig().with_size(30).with_overlap(10)))
        .with_reranker(Reranker())
        .with_retrieval_k(5)
        .with_rerank_k(3)
    )

    chunk_count = rag.ingest(documents)
    print(f"   Ingested {len(documents)} documents into {chunk_count} chunks")

    test_questions = [
        "What is machine learning?",
        "Explain deep learning",
        "How does NLP work?",
    ]

    for question in test_questions:
        response = rag.query(question)
        print(f'\n   Q: "{question}"')
        print(f"   A: {response.answer[:100]}...")
        print(f"   Sources: {response.sources}")

    # Step 8: Hybrid Search
    print("\n🔀 Step 8: Hybrid Search (Vector + Keyword)")
    hybrid = HybridSearch(dimension=64, alpha=0.7)
    print(f"   Alpha (vector weight): {hybrid.alpha}")

    for chunk in all_chunks[:6]:
        embedding = embed_model.embed(chunk.text)
        hybrid.add(chunk, embedding)

    query = "neural networks learning"
    query_embedding = embed_model.embed(query)
    hybrid_results = hybrid.search(query, query_embedding, top_k=3)

    print(f'\n   Query: "{query}"')
    for result in hybrid_results:
        preview = result.chunk.text[:50].replace("\n", " ") + "..."
        print(f"     [{result.rank}] Score: {result.score:.3f} - {preview}")

    # Summary
    print("\n═══════════════════════════════════════════════════════════════")
    print("Demo Complete!")
    print()
    print("Key concepts demonstrated:")
    print("  • Document chunking with overlap")
    print("  • Dense embeddings")
    print("  • Vector similarity search")
    print("  • Cross-encoder reranking")
    print("  • Hybrid search (vector + keyword)")
    print()
    print("Python equivalent of: Sovereign AI Stack trueno-rag")
    print("Databricks equivalent: Vector Search, RAG Pipeline")
    print("═══════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
