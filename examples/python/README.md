# Python Examples

Python implementations equivalent to the Sovereign AI Stack (Rust) examples. These demonstrate the same concepts as the Rust code but in Python, making them more accessible for learners who prefer Python.

## Structure

```
python/
└── course4/                         # GenAI Engineering
    ├── week1_llm/                   # LLM inference
    ├── week2_rag/                   # RAG pipelines
    └── week3_finetuning/            # Fine-tuning
```

## Course 4: GenAI Engineering

### Week 1: LLM Inference

Python equivalent of `realizar` (Rust LLM inference):

- GGUF model metadata and quantization
- BPE-style tokenization
- Text generation with sampling parameters
- KV cache for efficient inference
- Prompt templates

```bash
cd course4/week1_llm && python llm_inference.py
```

### Week 2: RAG Pipeline

Python equivalent of `trueno-rag` (Rust RAG):

- Document chunking with overlap
- Dense embeddings
- Vector similarity search
- Cross-encoder reranking
- Hybrid search (vector + keyword)

```bash
cd course4/week2_rag && python rag_pipeline.py
```

### Week 3: Fine-Tuning

Python equivalent of `entrenar` (Rust fine-tuning):

- LoRA configuration (rank, alpha, targets)
- QLoRA (4-bit quantization)
- Training data formatting (Alpaca, ChatML)
- Learning rate scheduling (warmup + cosine decay)
- Adapter merging strategies

```bash
cd course4/week3_finetuning && python finetuning.py
```

## Running All Examples

```bash
# Run all Course 4 Python examples
python -m examples.python.course4.week1_llm.llm_inference
python -m examples.python.course4.week2_rag.rag_pipeline
python -m examples.python.course4.week3_finetuning.finetuning
```

Or from the example directories:

```bash
# Week 1
cd examples/python/course4/week1_llm
python llm_inference.py

# Week 2
cd examples/python/course4/week2_rag
python rag_pipeline.py

# Week 3
cd examples/python/course4/week3_finetuning
python finetuning.py
```

## Mapping to Rust (Sovereign AI Stack)

| Python Module | Rust Crate | Description |
|---------------|------------|-------------|
| `week1_llm` | `realizar` | LLM inference |
| `week2_rag` | `trueno-rag` | RAG pipelines |
| `week3_finetuning` | `entrenar` | Fine-tuning |

## Mapping to Databricks

| Python Module | Databricks Feature |
|---------------|-------------------|
| `week1_llm` | Foundation Model APIs |
| `week2_rag` | Vector Search, RAG Pipeline |
| `week3_finetuning` | Model Fine-tuning |

## Usage as a Library

These modules can also be imported and used as a library:

```python
# Week 1: LLM Inference
from examples.python.course4.week1_llm import (
    LlmModel, GgufMetadata, QuantizationType, 
    GenerationConfig, Tokenizer, PromptTemplate
)

# Create and load model
metadata = GgufMetadata.new("llama-7b", QuantizationType.Q4_K_M)
model = LlmModel(metadata)
model.load()

# Generate text
config = GenerationConfig().with_max_tokens(100)
output = model.generate("What is AI?", config)
print(output.text)

# Week 2: RAG Pipeline
from examples.python.course4.week2_rag import (
    Document, RagPipeline, ChunkerConfig, Chunker
)

# Build RAG pipeline
rag = RagPipeline(embedding_dim=384)
rag.ingest([Document.new("doc1", "Machine learning is...")])

# Query
response = rag.query("What is machine learning?")
print(response.answer)

# Week 3: Fine-Tuning
from examples.python.course4.week3_finetuning import (
    LoraConfig, TrainingConfig, TrainingDataset, 
    TrainingSample, Trainer
)

# Configure LoRA
lora = LoraConfig.new(8, 16).with_targets(["q_proj", "v_proj"])

# Create dataset
dataset = TrainingDataset()
dataset.add(TrainingSample.new("Summarize", "Long text...", "Summary"))

# Train
trainer = Trainer(TrainingConfig(), lora)
result = trainer.train(dataset)
print(f"Final loss: {result.final_loss}")
```

## Testing

Run the tests to verify the examples work correctly:

```bash
# Run all tests
pytest tests/

# Run Python example tests specifically
pytest tests/test_python_examples.py -v
```

## Design Philosophy

These Python examples follow the same patterns as their Rust counterparts:

1. **Same API design** - Method signatures and class structures match the Rust versions
2. **Builder pattern** - Configuration uses method chaining (`.with_*()`)
3. **Dataclasses** - Clean data structures with clear field definitions
4. **Type hints** - Full type annotations for clarity
5. **Educational focus** - Clear code with extensive comments and demos
