"""
LLM Inference with Python (realizar equivalent)

Demonstrates LLM inference patterns including:
- GGUF model metadata and quantization
- BPE-style tokenization
- Text generation with sampling parameters
- KV cache for efficient inference
- Prompt templates

Course 4, Week 1: Foundation Models + Prompt Engineering
"""

from dataclasses import dataclass, field
from enum import Enum


class LlmError(Exception):
    """Base exception for LLM operations."""

    pass


class ModelNotLoadedError(LlmError):
    """Model has not been loaded."""

    pass


class GenerationError(LlmError):
    """Error during text generation."""

    pass


class TokenizationError(LlmError):
    """Error during tokenization."""

    pass


class ContextLengthExceededError(LlmError):
    """Context length limit exceeded."""

    pass


# ============================================================================
# GGUF Model Metadata
# ============================================================================


class QuantizationType(Enum):
    """Quantization types for GGUF models."""

    F16 = "F16"
    Q8_0 = "Q8_0"
    Q6_K = "Q6_K"
    Q5_K_M = "Q5_K_M"
    Q4_K_M = "Q4_K_M"
    Q4_K_S = "Q4_K_S"
    Q3_K_M = "Q3_K_M"
    Q2_K = "Q2_K"

    def bits_per_weight(self) -> float:
        """Get bits per weight for this quantization type."""
        mapping = {
            QuantizationType.F16: 16.0,
            QuantizationType.Q8_0: 8.0,
            QuantizationType.Q6_K: 6.5,
            QuantizationType.Q5_K_M: 5.5,
            QuantizationType.Q4_K_M: 4.5,
            QuantizationType.Q4_K_S: 4.5,
            QuantizationType.Q3_K_M: 3.5,
            QuantizationType.Q2_K: 2.5,
        }
        return mapping[self]

    def memory_factor(self) -> float:
        """Get memory factor relative to F16."""
        return self.bits_per_weight() / 16.0


@dataclass
class GgufMetadata:
    """Metadata for a GGUF model file."""

    model_name: str
    quantization: QuantizationType
    architecture: str = "llama"
    context_length: int = 4096
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32

    @classmethod
    def new(cls, name: str, quant: QuantizationType) -> "GgufMetadata":
        """Create new metadata with default parameters."""
        return cls(model_name=name, quantization=quant)

    def estimated_memory_gb(self, params_billions: float) -> float:
        """Estimate memory usage in GB for given model size."""
        return params_billions * self.quantization.memory_factor() * 2.0


# ============================================================================
# Tokenizer
# ============================================================================


class Tokenizer:
    """Simple word-level tokenizer (BPE-style simulation)."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.reverse_vocab: dict[int, str] = {}
        self.bos_token: int = 1
        self.eos_token: int = 2
        self.pad_token: int = 0
        self.unk_token: int = 3

        # Initialize vocabulary
        self._init_vocab()

    def _init_vocab(self) -> None:
        """Initialize vocabulary with special and common tokens."""
        # Special tokens
        special_tokens = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}

        for token, token_id in special_tokens.items():
            self.vocab[token] = token_id
            self.reverse_vocab[token_id] = token

        # Common tokens (simplified vocabulary)
        common_tokens = [
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "be",
            "have",
            "has",
            "do",
            "does",
            "will",
            "can",
            "could",
            "would",
            "should",
            "hello",
            "world",
            "how",
            "what",
            "when",
            "where",
            "why",
            "who",
            "capital",
            "france",
            "paris",
            "of",
            "in",
            "on",
            "at",
            "to",
            "and",
            "or",
            "but",
            "not",
            "this",
            "that",
            "it",
            "for",
            "with",
            "AI",
            "artificial",
            "intelligence",
            "machine",
            "learning",
            "model",
            "data",
            " ",
            ".",
            ",",
            "?",
            "!",
            "\n",
        ]

        for i, token in enumerate(common_tokens):
            token_id = i + 4  # Start after special tokens
            self.vocab[token] = token_id
            self.reverse_vocab[token_id] = token

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs."""
        tokens = [self.bos_token]

        for word in text.split():
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Character-level fallback for unknown words
                for char in word:
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                    else:
                        tokens.append(self.unk_token)

            # Add space token between words
            if " " in self.vocab:
                tokens.append(self.vocab[" "])

        # Remove trailing space if present
        space_id = self.vocab.get(" ")
        if tokens and tokens[-1] == space_id:
            tokens.pop()

        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        result = []
        for token_id in token_ids:
            token = self.reverse_vocab.get(token_id)
            if token and not (token.startswith("<") and token.endswith(">")):
                result.append(token)
        return "".join(result)

    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def bos_token_id(self) -> int:
        """Get beginning-of-sequence token ID."""
        return self.bos_token

    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        return self.eos_token


# ============================================================================
# Generation Configuration
# ============================================================================


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    stop_sequences: list[str] = field(default_factory=lambda: ["</s>"])

    def with_max_tokens(self, n: int) -> "GenerationConfig":
        """Set max new tokens and return self for chaining."""
        self.max_new_tokens = n
        return self

    def with_temperature(self, t: float) -> "GenerationConfig":
        """Set temperature and return self for chaining."""
        self.temperature = t
        return self

    def with_top_p(self, p: float) -> "GenerationConfig":
        """Set top_p and return self for chaining."""
        self.top_p = p
        return self


# ============================================================================
# KV Cache
# ============================================================================


@dataclass
class KVCache:
    """Key-Value cache for efficient inference."""

    keys: list[list[float]]
    values: list[list[float]]
    seq_len: int
    capacity: int

    @classmethod
    def new(cls, capacity: int, hidden_size: int) -> "KVCache":
        """Create a new KV cache with given capacity."""
        return cls(
            keys=[[0.0] * hidden_size for _ in range(capacity)],
            values=[[0.0] * hidden_size for _ in range(capacity)],
            seq_len=0,
            capacity=capacity,
        )

    def append(self, key: list[float], value: list[float]) -> None:
        """Append key-value pair to cache."""
        if self.seq_len >= self.capacity:
            raise ContextLengthExceededError(f"Cache full: {self.seq_len} >= {self.capacity}")
        self.keys[self.seq_len] = key
        self.values[self.seq_len] = value
        self.seq_len += 1

    def clear(self) -> None:
        """Clear the cache."""
        self.seq_len = 0

    def __len__(self) -> int:
        return self.seq_len

    def is_empty(self) -> bool:
        return self.seq_len == 0


# ============================================================================
# LLM Model
# ============================================================================


class FinishReason(Enum):
    """Reason for generation completion."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"


@dataclass
class GenerationOutput:
    """Output from text generation."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: FinishReason


class LlmModel:
    """Simulated LLM model for inference."""

    def __init__(self, metadata: GgufMetadata) -> None:
        self.metadata = metadata
        self.tokenizer = Tokenizer()
        self.kv_cache = KVCache.new(metadata.context_length, metadata.hidden_size)
        self._loaded = False

    def load(self) -> None:
        """Load the model (simulated)."""
        self._loaded = True

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationOutput:
        """Generate text from prompt."""
        if not self._loaded:
            raise ModelNotLoadedError(self.metadata.model_name)

        input_tokens = self.tokenizer.encode(prompt)
        prompt_len = len(input_tokens)

        # Check context length
        if prompt_len + config.max_new_tokens > self.metadata.context_length:
            raise ContextLengthExceededError(
                f"{prompt_len} + {config.max_new_tokens} > {self.metadata.context_length}"
            )

        # Simulate generation
        generated_text = self._simulate_generation(prompt)
        output_tokens = self.tokenizer.encode(generated_text)

        return GenerationOutput(
            text=generated_text,
            prompt_tokens=prompt_len,
            completion_tokens=len(output_tokens),
            finish_reason=FinishReason.STOP,
        )

    def _simulate_generation(self, prompt: str) -> str:
        """Simulate text generation based on prompt."""
        prompt_lower = prompt.lower()

        if "capital" in prompt_lower and "france" in prompt_lower:
            return "The capital of France is Paris."
        elif "hello" in prompt_lower:
            return "Hello! How can I help you today?"
        elif "what is ai" in prompt_lower or "artificial intelligence" in prompt_lower:
            return (
                "Artificial Intelligence (AI) is the simulation of human intelligence by machines."
            )
        elif "machine learning" in prompt_lower:
            return "Machine learning is a subset of AI that enables systems to learn from data."
        else:
            return "I understand your question. Let me provide a helpful response."

    def get_metadata(self) -> GgufMetadata:
        """Get model metadata."""
        return self.metadata

    def get_tokenizer(self) -> Tokenizer:
        """Get the tokenizer."""
        return self.tokenizer

    def clear_cache(self) -> None:
        """Clear the KV cache."""
        self.kv_cache.clear()


# ============================================================================
# Prompt Templates
# ============================================================================


class PromptTemplate:
    """Template for structured prompts."""

    def __init__(self, template: str) -> None:
        self.template = template
        self.variables = self._extract_variables(template)

    def _extract_variables(self, template: str) -> list[str]:
        """Extract variable names from template."""
        variables = []
        in_var = False
        var_name = ""

        for char in template:
            if char == "{":
                in_var = True
                var_name = ""
            elif char == "}" and in_var:
                variables.append(var_name)
                in_var = False
            elif in_var:
                var_name += char

        return variables

    def format(self, values: dict[str, str]) -> str:
        """Format template with provided values."""
        result = self.template
        for var in self.variables:
            if var not in values:
                raise GenerationError(f"Missing variable: {var}")
            result = result.replace(f"{{{var}}}", values[var])
        return result

    def get_variables(self) -> list[str]:
        """Get list of template variables."""
        return self.variables.copy()


# ============================================================================
# Demo
# ============================================================================


def main() -> None:
    """Run the LLM inference demo."""
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║     LLM Inference with Python - Course 4, Week 1              ║")
    print("║     GGUF Loading, Tokenization, Text Generation               ║")
    print("╚═══════════════════════════════════════════════════════════════╝")

    # Step 1: Model Metadata
    print("\n📋 Step 1: GGUF Model Metadata")
    metadata = GgufMetadata.new("llama-7b", QuantizationType.Q4_K_M)

    print(f"   Model: {metadata.model_name}")
    print(f"   Architecture: {metadata.architecture}")
    print(
        f"   Quantization: {metadata.quantization.value} "
        f"({metadata.quantization.bits_per_weight():.1f} bits/weight)"
    )
    print(f"   Context length: {metadata.context_length} tokens")
    print(f"   Vocab size: {metadata.vocab_size}")
    print(f"   Estimated memory: {metadata.estimated_memory_gb(7.0):.1f} GB")

    # Step 2: Tokenization
    print("\n📝 Step 2: Tokenization")
    tokenizer = Tokenizer()

    test_texts = [
        "Hello world",
        "What is the capital of France?",
        "Machine learning is AI",
    ]

    for text in test_texts:
        tokens = tokenizer.encode(text)
        print(f'   "{text}"')
        print(f"     Tokens: {tokens}")
        print(f"     Count: {len(tokens)}")

    # Step 3: Load Model
    print("\n🔧 Step 3: Model Loading")
    model = LlmModel(metadata)
    model.load()
    print(f"   Model loaded: {model.is_loaded()}")

    # Step 4: Generation
    print("\n🚀 Step 4: Text Generation")
    config = GenerationConfig().with_max_tokens(100).with_temperature(0.7)

    prompts = [
        "What is the capital of France?",
        "Hello, how are you?",
        "Explain machine learning briefly.",
    ]

    for prompt in prompts:
        try:
            output = model.generate(prompt, config)
            print(f'   Prompt: "{prompt}"')
            print(f'   Response: "{output.text}"')
            print(
                f"   Tokens: {output.prompt_tokens} prompt + {output.completion_tokens} completion"
            )
            print(f"   Finish: {output.finish_reason.value}\n")
        except LlmError as e:
            print(f"   Error: {e}\n")

    # Step 5: Prompt Templates
    print("📄 Step 5: Prompt Templates")
    template = PromptTemplate(
        "You are a {role}. Answer the following question:\n{question}\n\nAnswer:"
    )

    print(f"   Variables: {template.get_variables()}")

    values = {"role": "helpful assistant", "question": "What is AI?"}

    formatted = template.format(values)
    print("   Formatted prompt:")
    for line in formatted.split("\n"):
        print(f"     {line}")

    # Step 6: Quantization Comparison
    print("\n📊 Step 6: Quantization Comparison")
    quants = [
        QuantizationType.F16,
        QuantizationType.Q8_0,
        QuantizationType.Q4_K_M,
        QuantizationType.Q2_K,
    ]

    for quant in quants:
        meta = GgufMetadata.new("llama-7b", quant)
        print(
            f"   {quant.value}: {quant.bits_per_weight():.1f} bits, "
            f"~{meta.estimated_memory_gb(7.0):.1f} GB memory"
        )

    # Summary
    print("\n═══════════════════════════════════════════════════════════════")
    print("Demo Complete!")
    print()
    print("Key concepts demonstrated:")
    print("  • GGUF model metadata and quantization")
    print("  • BPE-style tokenization")
    print("  • Text generation with sampling parameters")
    print("  • KV cache for efficient inference")
    print("  • Prompt templates")
    print()
    print("Python equivalent of: Sovereign AI Stack realizar")
    print("Databricks equivalent: Foundation Model APIs")
    print("═══════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
