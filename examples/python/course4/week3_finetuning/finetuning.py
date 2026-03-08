"""
Fine-Tuning with Python (entrenar equivalent)

Demonstrates LLM fine-tuning patterns including:
- LoRA configuration (rank, alpha, target modules)
- QLoRA for memory-efficient training
- Training data formatting (Alpaca, ChatML)
- Learning rate scheduling (warmup + cosine decay)
- Adapter merging strategies

Course 4, Week 3: Fine-Tuning + Production + Capstone
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TrainingError(Exception):
    """Base exception for training operations."""

    pass


class ConfigError(TrainingError):
    """Configuration error."""

    pass


class DataError(TrainingError):
    """Data processing error."""

    pass


class CheckpointError(TrainingError):
    """Checkpoint save/load error."""

    pass


# ============================================================================
# LoRA Configuration
# ============================================================================


class BiasMode(Enum):
    """Bias handling mode for LoRA."""

    NONE = "none"
    ALL = "all"
    LORA_ONLY = "lora_only"


@dataclass
class LoraConfig:
    """Configuration for LoRA (Low-Rank Adaptation)."""

    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: BiasMode = BiasMode.NONE

    @classmethod
    def new(cls, rank: int, alpha: int) -> "LoraConfig":
        """Create new LoRA config with specified rank and alpha."""
        return cls(rank=rank, alpha=alpha)

    def with_dropout(self, dropout: float) -> "LoraConfig":
        """Set dropout and return self for chaining."""
        self.dropout = dropout
        return self

    def with_targets(self, modules: list[str]) -> "LoraConfig":
        """Set target modules and return self for chaining."""
        self.target_modules = modules
        return self

    def with_bias(self, bias: BiasMode) -> "LoraConfig":
        """Set bias mode and return self for chaining."""
        self.bias = bias
        return self

    def scaling_factor(self) -> float:
        """Calculate scaling factor (alpha / rank)."""
        return self.alpha / self.rank

    def trainable_params(self, hidden_dim: int) -> int:
        """Estimate number of trainable parameters."""
        # LoRA adds (rank * hidden_dim + hidden_dim * rank) per module
        return 2 * self.rank * hidden_dim * len(self.target_modules)

    def trainable_ratio(self, total_params: int, hidden_dim: int) -> float:
        """Calculate percentage of trainable parameters."""
        trainable = self.trainable_params(hidden_dim)
        return trainable / total_params * 100.0


# ============================================================================
# QLoRA Configuration
# ============================================================================


class QuantType(Enum):
    """Quantization type for QLoRA."""

    NF4 = "nf4"
    FP4 = "fp4"


class ComputeDtype(Enum):
    """Compute dtype for QLoRA."""

    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


@dataclass
class QloraConfig:
    """Configuration for QLoRA (Quantized LoRA)."""

    lora: LoraConfig
    bits: int = 4
    double_quant: bool = True
    quant_type: QuantType = QuantType.NF4
    compute_dtype: ComputeDtype = ComputeDtype.BFLOAT16

    @classmethod
    def new(cls, lora: LoraConfig, bits: int) -> "QloraConfig":
        """Create new QLoRA config."""
        return cls(lora=lora, bits=bits)

    def with_double_quant(self, enabled: bool) -> "QloraConfig":
        """Enable/disable double quantization and return self for chaining."""
        self.double_quant = enabled
        return self

    def with_compute_dtype(self, dtype: ComputeDtype) -> "QloraConfig":
        """Set compute dtype and return self for chaining."""
        self.compute_dtype = dtype
        return self

    def memory_reduction(self) -> float:
        """Calculate memory reduction factor."""
        base = self.bits / 16.0
        if self.double_quant:
            return base * 0.9  # Additional ~10% reduction
        return base

    def estimated_memory_gb(self, model_params_billions: float) -> float:
        """Estimate memory usage in GB."""
        return model_params_billions * 2.0 * self.memory_reduction()


# ============================================================================
# Training Data
# ============================================================================


@dataclass
class TrainingSample:
    """A training sample for instruction tuning."""

    instruction: str
    input: str
    output: str

    @classmethod
    def new(cls, instruction: str, input_text: str, output: str) -> "TrainingSample":
        """Create a new training sample."""
        return cls(instruction=instruction, input=input_text, output=output)

    def format_alpaca(self) -> str:
        """Format as Alpaca-style prompt."""
        if self.input:
            return (
                f"### Instruction:\n{self.instruction}\n\n"
                f"### Input:\n{self.input}\n\n"
                f"### Response:\n{self.output}"
            )
        else:
            return f"### Instruction:\n{self.instruction}\n\n### Response:\n{self.output}"

    def format_chatml(self) -> str:
        """Format as ChatML-style prompt."""
        if self.input:
            user_content = f"{self.instruction}\n\n{self.input}"
        else:
            user_content = self.instruction

        return (
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n{self.output}<|im_end|>"
        )

    def total_length(self) -> int:
        """Get total character length of the sample."""
        return len(self.instruction) + len(self.input) + len(self.output)


class DatasetFormat(Enum):
    """Dataset format for training."""

    ALPACA = "alpaca"
    CHATML = "chatml"
    SHAREGPT = "sharegpt"


class TrainingDataset:
    """Dataset for fine-tuning."""

    def __init__(self, format: DatasetFormat = DatasetFormat.ALPACA) -> None:
        self.samples: list[TrainingSample] = []
        self.format = format

    def add(self, sample: TrainingSample) -> None:
        """Add a training sample."""
        self.samples.append(sample)

    def add_many(self, samples: list[TrainingSample]) -> None:
        """Add multiple training samples."""
        self.samples.extend(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def is_empty(self) -> bool:
        return len(self.samples) == 0

    def format_all(self) -> list[str]:
        """Format all samples according to dataset format."""
        if self.format == DatasetFormat.ALPACA:
            return [s.format_alpaca() for s in self.samples]
        else:  # ChatML and ShareGPT use same format
            return [s.format_chatml() for s in self.samples]

    def split(self, train_ratio: float = 0.9) -> tuple["TrainingDataset", "TrainingDataset"]:
        """Split dataset into train and validation sets."""
        split_idx = int(len(self.samples) * train_ratio)

        train = TrainingDataset(self.format)
        train.samples = self.samples[:split_idx]

        val = TrainingDataset(self.format)
        val.samples = self.samples[split_idx:]

        return train, val

    def average_length(self) -> float:
        """Get average character length of samples."""
        if not self.samples:
            return 0.0
        return sum(s.total_length() for s in self.samples) / len(self.samples)


# ============================================================================
# Training Configuration
# ============================================================================


@dataclass
class TrainingConfig:
    """Configuration for training."""

    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    max_steps: Optional[int] = None
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_seq_length: int = 512
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100

    def effective_batch_size(self) -> int:
        """Calculate effective batch size with gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

    def steps_per_epoch(self, dataset_size: int) -> int:
        """Calculate training steps per epoch."""
        return (dataset_size + self.effective_batch_size() - 1) // self.effective_batch_size()

    def total_steps(self, dataset_size: int) -> int:
        """Calculate total training steps."""
        if self.max_steps is not None:
            return self.max_steps
        return self.steps_per_epoch(dataset_size) * self.epochs

    def warmup_steps(self, dataset_size: int) -> int:
        """Calculate number of warmup steps."""
        return int(self.total_steps(dataset_size) * self.warmup_ratio)


# ============================================================================
# Training Metrics
# ============================================================================


@dataclass
class TrainingMetrics:
    """Metrics for a training step."""

    step: int
    epoch: float
    loss: float
    learning_rate: float
    grad_norm: float
    samples_per_second: float


class TrainingHistory:
    """History of training metrics."""

    def __init__(self) -> None:
        self.metrics: list[TrainingMetrics] = []

    def add(self, metrics: TrainingMetrics) -> None:
        """Add metrics for a training step."""
        self.metrics.append(metrics)

    def __len__(self) -> int:
        return len(self.metrics)

    def is_empty(self) -> bool:
        return len(self.metrics) == 0

    def latest_loss(self) -> Optional[float]:
        """Get the most recent loss value."""
        if self.metrics:
            return self.metrics[-1].loss
        return None

    def average_loss(self) -> float:
        """Get average loss across all steps."""
        if not self.metrics:
            return 0.0
        return sum(m.loss for m in self.metrics) / len(self.metrics)

    def min_loss(self) -> Optional[float]:
        """Get minimum loss achieved."""
        if not self.metrics:
            return None
        return min(m.loss for m in self.metrics)


# ============================================================================
# Trainer
# ============================================================================


@dataclass
class TrainingResult:
    """Result of a training run."""

    final_loss: float
    best_loss: float
    total_steps: int
    epochs_completed: float


class Trainer:
    """Trainer for fine-tuning models."""

    def __init__(self, config: TrainingConfig, lora_config: LoraConfig) -> None:
        self.config = config
        self.lora_config = lora_config
        self.history = TrainingHistory()

    def train(self, dataset: TrainingDataset) -> TrainingResult:
        """Train the model on the dataset (simulated)."""
        if dataset.is_empty():
            raise DataError("Empty dataset")

        total_steps = self.config.total_steps(len(dataset))
        warmup_steps = self.config.warmup_steps(len(dataset))

        loss = 2.5
        for step in range(total_steps):
            # Simulate learning rate schedule
            lr = self._get_learning_rate(step, total_steps, warmup_steps)

            # Simulate loss decrease
            loss *= 0.997
            loss += abs(math.sin(step * 0.1) * 0.05)

            epoch = step / self.config.steps_per_epoch(len(dataset))

            self.history.add(
                TrainingMetrics(
                    step=step,
                    epoch=epoch,
                    loss=loss,
                    learning_rate=lr,
                    grad_norm=1.0 + math.cos(step * 0.01) * 0.3,
                    samples_per_second=10.0 + math.sin(step * 0.001) * 2.0,
                )
            )

        return TrainingResult(
            final_loss=self.history.latest_loss() or 0.0,
            best_loss=self.history.min_loss() or 0.0,
            total_steps=total_steps,
            epochs_completed=float(self.config.epochs),
        )

    def _get_learning_rate(self, step: int, total_steps: int, warmup_steps: int) -> float:
        """Calculate learning rate with warmup and cosine decay."""
        if step < warmup_steps:
            # Linear warmup
            return self.config.learning_rate * (step / warmup_steps)
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return self.config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    def get_history(self) -> TrainingHistory:
        """Get training history."""
        return self.history

    def get_config(self) -> TrainingConfig:
        """Get training configuration."""
        return self.config

    def get_lora_config(self) -> LoraConfig:
        """Get LoRA configuration."""
        return self.lora_config


# ============================================================================
# Model Merging
# ============================================================================


class MergeMethod(Enum):
    """Method for merging LoRA adapters."""

    LINEAR = "linear"
    SLERP = "slerp"
    TASK_ARITHMETIC = "task_arithmetic"


@dataclass
class LoraWeight:
    """A LoRA adapter with its merge weight."""

    name: str
    weight: float

    @classmethod
    def new(cls, name: str, weight: float) -> "LoraWeight":
        """Create a new LoRA weight."""
        return cls(name=name, weight=weight)


def merge_adapters(adapters: list[LoraWeight], method: MergeMethod) -> str:
    """Merge multiple LoRA adapters (simulated)."""
    if not adapters:
        raise ConfigError("No adapters to merge")

    # Normalize weights
    total_weight = sum(a.weight for a in adapters)
    if total_weight < 1e-6:
        raise ConfigError("Weights sum to zero")

    return f"Merged {len(adapters)} adapters using {method.value} method"


# ============================================================================
# Demo
# ============================================================================


def main() -> None:
    """Run the fine-tuning demo."""
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║     Fine-Tuning with Python - Course 4, Week 3                ║")
    print("║     LoRA, QLoRA, Training Configuration                       ║")
    print("╚═══════════════════════════════════════════════════════════════╝")

    # Step 1: LoRA Configuration
    print("\n🔧 Step 1: LoRA Configuration")
    lora = (
        LoraConfig.new(8, 16)
        .with_dropout(0.05)
        .with_targets(["q_proj", "v_proj", "k_proj", "o_proj"])
    )

    print(f"   Rank (r): {lora.rank}")
    print(f"   Alpha: {lora.alpha}")
    print(f"   Scaling factor: {lora.scaling_factor():.2f}")
    print(f"   Dropout: {lora.dropout}")
    print(f"   Target modules: {lora.target_modules}")

    hidden_dim = 4096
    total_params = 7_000_000_000
    print(f"   Trainable params: {lora.trainable_params(hidden_dim)}")
    print(f"   Trainable ratio: {lora.trainable_ratio(total_params, hidden_dim):.4f}%")

    # Step 2: QLoRA Configuration
    print("\n💾 Step 2: QLoRA Configuration")
    qlora = (
        QloraConfig.new(lora, 4).with_double_quant(True).with_compute_dtype(ComputeDtype.BFLOAT16)
    )

    print(f"   Bits: {qlora.bits}")
    print(f"   Quant type: {qlora.quant_type.value}")
    print(f"   Double quantization: {qlora.double_quant}")
    print(f"   Compute dtype: {qlora.compute_dtype.value}")
    print(f"   Memory reduction: {(1.0 - qlora.memory_reduction()) * 100:.0f}%")
    print(f"   Estimated memory (7B): {qlora.estimated_memory_gb(7.0):.1f} GB")

    # Step 3: Training Data
    print("\n📊 Step 3: Training Data")
    dataset = TrainingDataset(DatasetFormat.ALPACA)

    samples = [
        TrainingSample.new(
            "Summarize the text.",
            "Machine learning is a field of AI...",
            "ML enables systems to learn from data.",
        ),
        TrainingSample.new(
            "Translate to French.",
            "Hello, how are you?",
            "Bonjour, comment allez-vous?",
        ),
        TrainingSample.new(
            "Answer the question.",
            "What is the capital of France?",
            "Paris is the capital of France.",
        ),
        TrainingSample.new(
            "Classify sentiment.",
            "This product is amazing!",
            "Positive",
        ),
    ]

    for sample in samples:
        dataset.add(sample)

    # Add more samples for simulation
    for i in range(96):
        dataset.add(TrainingSample.new(f"Task {i}", f"Input {i}", f"Output {i}"))

    print(f"   Dataset size: {len(dataset)} samples")
    print(f"   Format: {dataset.format.value}")
    print(f"   Average length: {dataset.average_length():.0f} chars")

    train, val = dataset.split(0.9)
    print(f"   Train/Val split: {len(train)} / {len(val)}")

    # Step 4: Training Configuration
    print("\n⚙️  Step 4: Training Configuration")
    config = TrainingConfig(
        learning_rate=2e-4,
        batch_size=4,
        gradient_accumulation_steps=4,
        epochs=3,
        max_seq_length=512,
    )

    print(f"   Learning rate: {config.learning_rate:.0e}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"   Effective batch size: {config.effective_batch_size()}")
    print(f"   Epochs: {config.epochs}")
    print(f"   Steps per epoch: {config.steps_per_epoch(len(train))}")
    print(f"   Total steps: {config.total_steps(len(train))}")
    print(f"   Warmup steps: {config.warmup_steps(len(train))}")

    # Step 5: Training
    print("\n🚀 Step 5: Training Simulation")
    trainer = Trainer(config, lora)
    result = trainer.train(train)

    print(f"   Final loss: {result.final_loss:.4f}")
    print(f"   Best loss: {result.best_loss:.4f}")
    print(f"   Total steps: {result.total_steps}")
    print(f"   Epochs completed: {result.epochs_completed:.1f}")

    # Step 6: Model Merging
    print("\n🔀 Step 6: Adapter Merging")
    adapters = [
        LoraWeight.new("adapter_math", 0.5),
        LoraWeight.new("adapter_code", 0.3),
        LoraWeight.new("adapter_writing", 0.2),
    ]

    merge_result = merge_adapters(adapters, MergeMethod.LINEAR)
    print(f"   {merge_result}")

    # Summary
    print("\n═══════════════════════════════════════════════════════════════")
    print("Demo Complete!")
    print()
    print("Key concepts demonstrated:")
    print("  • LoRA configuration (rank, alpha, target modules)")
    print("  • QLoRA for memory-efficient training")
    print("  • Training data formatting (Alpaca, ChatML)")
    print("  • Learning rate scheduling (warmup + cosine decay)")
    print("  • Adapter merging strategies")
    print()
    print("Python equivalent of: Sovereign AI Stack entrenar")
    print("Databricks equivalent: Model Training, Fine-tuning APIs")
    print("═══════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
