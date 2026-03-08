"""Tests for Python examples (Course 4: GenAI Engineering)."""

import sys
from pathlib import Path

# Add examples to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestWeek1LlmInference:
    """Tests for Week 1: LLM Inference."""

    def test_quantization_types(self):
        """Test quantization type bits per weight."""
        from examples.python.course4.week1_llm import QuantizationType

        assert QuantizationType.F16.bits_per_weight() == 16.0
        assert QuantizationType.Q8_0.bits_per_weight() == 8.0
        assert QuantizationType.Q4_K_M.bits_per_weight() == 4.5
        assert QuantizationType.Q2_K.bits_per_weight() == 2.5

    def test_gguf_metadata(self):
        """Test GGUF metadata creation."""
        from examples.python.course4.week1_llm import GgufMetadata, QuantizationType

        metadata = GgufMetadata.new("test-model", QuantizationType.Q4_K_M)
        assert metadata.model_name == "test-model"
        assert metadata.quantization == QuantizationType.Q4_K_M
        assert metadata.context_length == 4096

    def test_gguf_memory_estimation(self):
        """Test memory estimation for models."""
        from examples.python.course4.week1_llm import GgufMetadata, QuantizationType

        metadata = GgufMetadata.new("test", QuantizationType.F16)
        mem = metadata.estimated_memory_gb(7.0)
        assert mem > 0

        # Q4 should use less memory than F16
        metadata_q4 = GgufMetadata.new("test", QuantizationType.Q4_K_M)
        mem_q4 = metadata_q4.estimated_memory_gb(7.0)
        assert mem_q4 < mem

    def test_tokenizer_encode_decode(self):
        """Test tokenizer encode and decode."""
        from examples.python.course4.week1_llm import Tokenizer

        tokenizer = Tokenizer()
        tokens = tokenizer.encode("hello world")
        assert len(tokens) > 0
        assert tokens[0] == tokenizer.bos_token_id()

    def test_tokenizer_vocab_size(self):
        """Test tokenizer vocabulary size."""
        from examples.python.course4.week1_llm import Tokenizer

        tokenizer = Tokenizer()
        assert tokenizer.vocab_size() > 0

    def test_generation_config(self):
        """Test generation config builder pattern."""
        from examples.python.course4.week1_llm import GenerationConfig

        config = GenerationConfig().with_max_tokens(100).with_temperature(0.5)
        assert config.max_new_tokens == 100
        assert config.temperature == 0.5

    def test_kv_cache(self):
        """Test KV cache operations."""
        from examples.python.course4.week1_llm import KVCache

        cache = KVCache.new(capacity=10, hidden_size=64)
        assert len(cache) == 0
        assert cache.is_empty()

        cache.append([0.0] * 64, [0.0] * 64)
        assert len(cache) == 1
        assert not cache.is_empty()

        cache.clear()
        assert cache.is_empty()

    def test_llm_model_generation(self):
        """Test LLM model text generation."""
        from examples.python.course4.week1_llm import (
            GenerationConfig,
            GgufMetadata,
            LlmModel,
            QuantizationType,
        )

        metadata = GgufMetadata.new("test", QuantizationType.Q4_K_M)
        model = LlmModel(metadata)
        model.load()

        assert model.is_loaded()

        config = GenerationConfig().with_max_tokens(50)
        output = model.generate("Hello", config)

        assert len(output.text) > 0
        assert output.prompt_tokens > 0

    def test_prompt_template(self):
        """Test prompt template formatting."""
        from examples.python.course4.week1_llm import PromptTemplate

        template = PromptTemplate("Hello {name}, your task is {task}")
        assert template.get_variables() == ["name", "task"]

        result = template.format({"name": "Alice", "task": "summarize"})
        assert "Alice" in result
        assert "summarize" in result


class TestWeek2RagPipeline:
    """Tests for Week 2: RAG Pipeline."""

    def test_document_creation(self):
        """Test document creation with metadata."""
        from examples.python.course4.week2_rag import Document

        doc = Document.new("doc1", "Test content").with_metadata("source", "test")
        assert doc.id == "doc1"
        assert doc.content == "Test content"
        assert doc.metadata["source"] == "test"

    def test_chunker_config(self):
        """Test chunker configuration."""
        from examples.python.course4.week2_rag import ChunkerConfig

        config = ChunkerConfig().with_size(100).with_overlap(20)
        assert config.chunk_size == 100
        assert config.chunk_overlap == 20

    def test_chunker_chunks_document(self):
        """Test document chunking."""
        from examples.python.course4.week2_rag import Chunker, ChunkerConfig, Document

        doc = Document.new("doc1", "word " * 50)  # 50 words
        chunker = Chunker(ChunkerConfig().with_size(20).with_overlap(5))

        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
        assert all(c.doc_id == "doc1" for c in chunks)

    def test_embedding_cosine_similarity(self):
        """Test embedding cosine similarity."""
        from examples.python.course4.week2_rag import Embedding

        emb1 = Embedding(vector=[1.0, 0.0, 0.0])
        emb2 = Embedding(vector=[1.0, 0.0, 0.0])
        emb3 = Embedding(vector=[0.0, 1.0, 0.0])

        assert emb1.cosine_similarity(emb2) == 1.0
        assert emb1.cosine_similarity(emb3) == 0.0

    def test_embedding_model(self):
        """Test embedding model generates embeddings."""
        from examples.python.course4.week2_rag import EmbeddingModel

        model = EmbeddingModel(dimension=64)
        embedding = model.embed("test text")

        assert embedding.dimension == 64
        assert len(embedding.vector) == 64

    def test_vector_index(self):
        """Test vector index search."""
        from examples.python.course4.week2_rag import (
            Chunk,
            EmbeddingModel,
            VectorIndex,
        )

        index = VectorIndex(dimension=64)
        model = EmbeddingModel(dimension=64)

        chunk = Chunk(id="c1", doc_id="d1", text="test", start_idx=0, end_idx=1)
        embedding = model.embed("test")
        index.add(chunk, embedding)

        assert len(index) == 1

        query_emb = model.embed("test query")
        results = index.search(query_emb, top_k=1)
        assert len(results) == 1
        assert results[0].chunk.id == "c1"

    def test_rag_pipeline_ingest_and_query(self):
        """Test RAG pipeline ingestion and querying."""
        from examples.python.course4.week2_rag import Document, RagPipeline

        rag = RagPipeline(embedding_dim=64)
        docs = [Document.new("d1", "Machine learning is a type of AI")]
        chunks = rag.ingest(docs)

        assert chunks > 0

        response = rag.query("What is machine learning?")
        assert len(response.answer) > 0
        assert len(response.sources) > 0


class TestWeek3Finetuning:
    """Tests for Week 3: Fine-Tuning."""

    def test_lora_config(self):
        """Test LoRA configuration."""
        from examples.python.course4.week3_finetuning import LoraConfig

        lora = LoraConfig.new(8, 16)
        assert lora.rank == 8
        assert lora.alpha == 16
        assert lora.scaling_factor() == 2.0

    def test_lora_trainable_params(self):
        """Test LoRA trainable parameter estimation."""
        from examples.python.course4.week3_finetuning import LoraConfig

        lora = LoraConfig.new(8, 16).with_targets(["q_proj", "v_proj"])
        params = lora.trainable_params(4096)
        assert params > 0

    def test_qlora_config(self):
        """Test QLoRA configuration."""
        from examples.python.course4.week3_finetuning import (
            LoraConfig,
            QloraConfig,
            QuantType,
        )

        lora = LoraConfig.new(8, 16)
        qlora = QloraConfig.new(lora, 4)

        assert qlora.bits == 4
        assert qlora.quant_type == QuantType.NF4
        assert qlora.memory_reduction() < 0.5

    def test_training_sample_formats(self):
        """Test training sample formatting."""
        from examples.python.course4.week3_finetuning import TrainingSample

        sample = TrainingSample.new("Summarize", "Long text", "Short text")

        alpaca = sample.format_alpaca()
        assert "### Instruction:" in alpaca
        assert "### Response:" in alpaca

        chatml = sample.format_chatml()
        assert "<|im_start|>" in chatml
        assert "<|im_end|>" in chatml

    def test_training_dataset(self):
        """Test training dataset operations."""
        from examples.python.course4.week3_finetuning import (
            DatasetFormat,
            TrainingDataset,
            TrainingSample,
        )

        dataset = TrainingDataset(DatasetFormat.ALPACA)
        dataset.add(TrainingSample.new("Task", "Input", "Output"))

        assert len(dataset) == 1

        train, val = dataset.split(0.8)
        assert len(train) + len(val) == len(dataset)

    def test_training_config(self):
        """Test training configuration calculations."""
        from examples.python.course4.week3_finetuning import TrainingConfig

        config = TrainingConfig(batch_size=4, gradient_accumulation_steps=4, epochs=3)

        assert config.effective_batch_size() == 16
        assert config.total_steps(100) > 0

    def test_trainer(self):
        """Test trainer simulation."""
        from examples.python.course4.week3_finetuning import (
            LoraConfig,
            Trainer,
            TrainingConfig,
            TrainingDataset,
            TrainingSample,
        )

        config = TrainingConfig(epochs=1)
        lora = LoraConfig.new(8, 16)
        trainer = Trainer(config, lora)

        dataset = TrainingDataset()
        for i in range(20):
            dataset.add(TrainingSample.new(f"Task {i}", f"Input {i}", f"Output {i}"))

        result = trainer.train(dataset)

        assert result.total_steps > 0
        assert result.final_loss > 0

    def test_adapter_merging(self):
        """Test adapter merging."""
        from examples.python.course4.week3_finetuning import (
            LoraWeight,
            MergeMethod,
            merge_adapters,
        )

        adapters = [
            LoraWeight.new("adapter1", 0.5),
            LoraWeight.new("adapter2", 0.5),
        ]

        result = merge_adapters(adapters, MergeMethod.LINEAR)
        assert "Merged 2 adapters" in result


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_lora_zero_rank_raises_error(self):
        """Test that LoRA with zero rank raises an error."""
        import pytest

        from examples.python.course4.week3_finetuning import ConfigError, LoraConfig

        lora = LoraConfig(rank=0, alpha=16)
        with pytest.raises(ConfigError):
            lora.scaling_factor()

    def test_lora_zero_total_params_raises_error(self):
        """Test that trainable_ratio with zero total params raises error."""
        import pytest

        from examples.python.course4.week3_finetuning import ConfigError, LoraConfig

        lora = LoraConfig.new(8, 16)
        with pytest.raises(ConfigError):
            lora.trainable_ratio(0, 4096)

    def test_rag_empty_pipeline_query(self):
        """Test RAG pipeline query with no documents."""
        from examples.python.course4.week2_rag import RagPipeline

        rag = RagPipeline(embedding_dim=64)
        # Query without ingesting any documents
        response = rag.query("test query")
        assert response.answer == "No relevant documents found."
        assert response.sources == []
        assert response.context_length == 0

    def test_reranker_empty_query(self):
        """Test reranker with empty query."""
        from examples.python.course4.week2_rag import Chunk, Reranker, RetrievalResult

        reranker = Reranker()
        chunk = Chunk(id="c1", doc_id="d1", text="test", start_idx=0, end_idx=1)
        results = [RetrievalResult(chunk=chunk, score=0.5, rank=1)]

        # Empty query should return results unchanged
        reranked = reranker.rerank("", results, top_k=1)
        assert len(reranked) == 1

    def test_reranker_empty_results(self):
        """Test reranker with empty results."""
        from examples.python.course4.week2_rag import Reranker

        reranker = Reranker()
        reranked = reranker.rerank("test query", [], top_k=1)
        assert reranked == []

    def test_model_not_loaded_error(self):
        """Test that generating without loading raises error."""
        import pytest

        from examples.python.course4.week1_llm import (
            GenerationConfig,
            GgufMetadata,
            LlmModel,
            ModelNotLoadedError,
            QuantizationType,
        )

        metadata = GgufMetadata.new("test", QuantizationType.Q4_K_M)
        model = LlmModel(metadata)
        # Don't load the model

        config = GenerationConfig().with_max_tokens(50)
        with pytest.raises(ModelNotLoadedError):
            model.generate("Hello", config)
