"""
DET MLX Training Pipeline
=========================

LoRA fine-tuning on Apple Silicon using MLX for memory model retraining.

Phase 2.2 Implementation.
"""

import json
import os
import time
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from enum import Enum

# MLX imports (optional, graceful fallback if not available)
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load, generate
    from mlx_lm.tuner import TrainingArgs
    from mlx_lm.tuner.trainer import train as mlx_train
    from mlx_lm.tuner.utils import linear_to_lora_layers
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None


class TrainingStatus(Enum):
    """Status of a training job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingConfig:
    """Configuration for LoRA training."""
    # Model settings
    model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    # LoRA parameters
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_layers: int = 16  # Number of layers to apply LoRA to

    # Training parameters
    batch_size: int = 4
    learning_rate: float = 1e-4
    epochs: int = 3
    warmup_steps: int = 100
    max_seq_length: int = 512

    # Checkpointing
    save_every: int = 100
    eval_every: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_layers": self.lora_layers,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "warmup_steps": self.warmup_steps,
            "max_seq_length": self.max_seq_length,
        }


@dataclass
class TrainingExample:
    """A single training example."""
    instruction: str
    response: str
    domain: Optional[str] = None
    importance: float = 1.0

    def to_chat_format(self) -> List[Dict[str, str]]:
        """Convert to chat format for training."""
        return [
            {"role": "user", "content": self.instruction},
            {"role": "assistant", "content": self.response}
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "response": self.response,
            "domain": self.domain,
            "importance": self.importance,
        }


@dataclass
class TrainingJob:
    """Represents a training job."""
    job_id: str
    domain: str
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    num_examples: int = 0
    current_step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    error_message: Optional[str] = None
    adapter_path: Optional[Path] = None


class TrainingDataGenerator:
    """
    Generates training data from memories and context.

    Uses LLM to reformulate raw memories into instruction-response pairs
    suitable for LoRA fine-tuning.
    """

    # Templates for different domains
    DOMAIN_TEMPLATES = {
        "MATH": [
            ("Solve this math problem: {content}", "The solution is: {response}"),
            ("Calculate: {content}", "{response}"),
            ("What is the result of {content}?", "{response}"),
        ],
        "CODE": [
            ("Write code to {content}", "{response}"),
            ("How do I implement {content}?", "{response}"),
            ("Debug this code issue: {content}", "{response}"),
        ],
        "REASONING": [
            ("Explain why {content}", "{response}"),
            ("Analyze this: {content}", "{response}"),
            ("What are the implications of {content}?", "{response}"),
        ],
        "GENERAL": [
            ("Tell me about {content}", "{response}"),
            ("What do you know about {content}?", "{response}"),
            ("Explain {content}", "{response}"),
        ],
    }

    def __init__(self, llm_client=None):
        """
        Initialize the generator.

        Args:
            llm_client: Optional LLM client for enhanced generation.
        """
        self.llm_client = llm_client

    def generate_from_memory(
        self,
        content: str,
        domain: str = "GENERAL",
        use_llm: bool = False
    ) -> List[TrainingExample]:
        """
        Generate training examples from a memory entry.

        Args:
            content: The memory content.
            domain: The memory domain.
            use_llm: Whether to use LLM for enhanced generation.

        Returns:
            List of TrainingExample objects.
        """
        examples = []

        # Get templates for this domain
        templates = self.DOMAIN_TEMPLATES.get(domain, self.DOMAIN_TEMPLATES["GENERAL"])

        # Simple heuristic-based generation
        if "?" in content:
            # Q&A format
            parts = content.split("?", 1)
            if len(parts) == 2 and parts[1].strip():
                examples.append(TrainingExample(
                    instruction=parts[0].strip() + "?",
                    response=parts[1].strip(),
                    domain=domain,
                ))

        # Template-based generation
        if len(content) > 20:  # Meaningful content
            # Extract a topic/summary
            topic = content[:100].split(".")[0] if "." in content[:100] else content[:50]

            template = templates[0]  # Use first template
            examples.append(TrainingExample(
                instruction=template[0].format(content=topic),
                response=content,
                domain=domain,
            ))

        # Use LLM for enhanced generation if available
        if use_llm and self.llm_client and len(examples) < 3:
            llm_examples = self._generate_with_llm(content, domain)
            examples.extend(llm_examples)

        return examples

    def _generate_with_llm(
        self,
        content: str,
        domain: str
    ) -> List[TrainingExample]:
        """Generate training examples using LLM."""
        if not self.llm_client:
            return []

        prompt = f"""Given this knowledge content, generate 2 question-answer pairs that would help someone learn this information.

Content: {content}

Domain: {domain}

Format your response as JSON:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]"""

        try:
            response = self.llm_client.generate(prompt)
            # Parse JSON response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                pairs = json.loads(json_match.group())
                return [
                    TrainingExample(
                        instruction=pair["question"],
                        response=pair["answer"],
                        domain=domain,
                    )
                    for pair in pairs
                ]
        except Exception:
            pass  # Fall back to heuristic generation

        return []

    def generate_from_context(
        self,
        messages: List[Dict[str, str]],
        domain: str = "DIALOGUE"
    ) -> List[TrainingExample]:
        """
        Generate training examples from conversation context.

        Args:
            messages: List of conversation messages with 'role' and 'content'.
            domain: The target domain.

        Returns:
            List of TrainingExample objects.
        """
        examples = []

        # Extract user-assistant pairs
        for i in range(len(messages) - 1):
            if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                examples.append(TrainingExample(
                    instruction=messages[i]["content"],
                    response=messages[i + 1]["content"],
                    domain=domain,
                ))

        return examples

    def save_dataset(
        self,
        examples: List[TrainingExample],
        output_path: Path,
        format: str = "jsonl"
    ):
        """
        Save training examples to file.

        Args:
            examples: List of training examples.
            output_path: Output file path.
            format: Output format ('jsonl' or 'json').
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(output_path, "w") as f:
                for example in examples:
                    # MLX-LM expects chat format
                    data = {"messages": example.to_chat_format()}
                    f.write(json.dumps(data) + "\n")
        else:
            with open(output_path, "w") as f:
                json.dump([e.to_dict() for e in examples], f, indent=2)


class LoRATrainer:
    """
    LoRA fine-tuning using MLX.

    Provides efficient fine-tuning of language models on Apple Silicon
    using Low-Rank Adaptation (LoRA).
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        adapters_dir: Optional[Path] = None
    ):
        """
        Initialize the trainer.

        Args:
            config: Training configuration.
            adapters_dir: Directory for saving adapters.
        """
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX is not available. Install with: pip install mlx mlx-lm")

        self.config = config or TrainingConfig()
        self.adapters_dir = adapters_dir or Path.home() / ".det_agency" / "adapters"
        self.adapters_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._tokenizer = None
        self._current_job: Optional[TrainingJob] = None

    def load_model(self, model_name: Optional[str] = None):
        """
        Load the base model.

        Args:
            model_name: Model name/path (uses config default if None).
        """
        model_name = model_name or self.config.model_name
        self._model, self._tokenizer = load(model_name)

        # Apply LoRA layers
        self._model = self._apply_lora()

    def _apply_lora(self):
        """Apply LoRA layers to the model."""
        # Convert linear layers to LoRA
        linear_to_lora_layers(
            self._model,
            self.config.lora_layers,
            {
                "rank": self.config.lora_rank,
                "alpha": self.config.lora_alpha,
                "dropout": self.config.lora_dropout,
            }
        )
        return self._model

    def train(
        self,
        train_data: List[TrainingExample],
        domain: str,
        val_data: Optional[List[TrainingExample]] = None,
        callback: Optional[Callable[[TrainingJob], None]] = None
    ) -> TrainingJob:
        """
        Train LoRA adapter on the provided data.

        Args:
            train_data: Training examples.
            domain: Domain name for the adapter.
            val_data: Optional validation data.
            callback: Optional callback for progress updates.

        Returns:
            TrainingJob with results.
        """
        import uuid

        job = TrainingJob(
            job_id=str(uuid.uuid4())[:8],
            domain=domain,
            config=self.config,
            num_examples=len(train_data),
        )
        self._current_job = job

        try:
            job.status = TrainingStatus.RUNNING
            job.started_at = time.time()

            if callback:
                callback(job)

            # Ensure model is loaded
            if self._model is None:
                self.load_model()

            # Prepare data files
            data_dir = self.adapters_dir / "data" / job.job_id
            data_dir.mkdir(parents=True, exist_ok=True)

            train_file = data_dir / "train.jsonl"
            val_file = data_dir / "valid.jsonl" if val_data else None

            # Save training data
            generator = TrainingDataGenerator()
            generator.save_dataset(train_data, train_file)
            if val_data:
                generator.save_dataset(val_data, val_file)

            # Configure training
            adapter_path = self.adapters_dir / domain

            training_args = TrainingArgs(
                batch_size=self.config.batch_size,
                iters=self.config.epochs * (len(train_data) // self.config.batch_size + 1),
                val_batches=10 if val_data else 0,
                steps_per_report=self.config.eval_every,
                steps_per_eval=self.config.eval_every,
                save_every=self.config.save_every,
                adapter_path=str(adapter_path),
                max_seq_length=self.config.max_seq_length,
                grad_checkpoint=True,
            )

            job.total_steps = training_args.iters

            # Run training
            mlx_train(
                model=self._model,
                tokenizer=self._tokenizer,
                args=training_args,
                train_dataset=str(train_file),
                val_dataset=str(val_file) if val_file else None,
            )

            job.status = TrainingStatus.COMPLETED
            job.completed_at = time.time()
            job.adapter_path = adapter_path

            # Save job metadata
            self._save_job_metadata(job)

            # Cleanup data files
            shutil.rmtree(data_dir)

        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = time.time()

        finally:
            self._current_job = None
            if callback:
                callback(job)

        return job

    def _save_job_metadata(self, job: TrainingJob):
        """Save training job metadata."""
        if job.adapter_path:
            metadata_file = job.adapter_path / "training_metadata.json"
            metadata = {
                "job_id": job.job_id,
                "domain": job.domain,
                "config": job.config.to_dict(),
                "num_examples": job.num_examples,
                "created_at": job.created_at,
                "completed_at": job.completed_at,
                "status": job.status.value,
            }
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

    def load_adapter(self, domain: str) -> bool:
        """
        Load a trained adapter.

        Args:
            domain: Domain name of the adapter.

        Returns:
            True if adapter was loaded successfully.
        """
        adapter_path = self.adapters_dir / domain

        if not adapter_path.exists():
            return False

        try:
            # Load adapter weights
            adapter_file = adapter_path / "adapters.safetensors"
            if adapter_file.exists():
                self._model.load_weights(str(adapter_file), strict=False)
                return True
        except Exception:
            pass

        return False

    def list_adapters(self) -> List[Dict[str, Any]]:
        """
        List all available adapters.

        Returns:
            List of adapter metadata dictionaries.
        """
        adapters = []

        for adapter_dir in self.adapters_dir.iterdir():
            if adapter_dir.is_dir():
                metadata_file = adapter_dir / "training_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                        metadata["path"] = str(adapter_dir)
                        adapters.append(metadata)

        return adapters

    def delete_adapter(self, domain: str) -> bool:
        """
        Delete an adapter.

        Args:
            domain: Domain name of the adapter.

        Returns:
            True if adapter was deleted.
        """
        adapter_path = self.adapters_dir / domain

        if adapter_path.exists():
            shutil.rmtree(adapter_path)
            return True

        return False


class MemoryRetuner:
    """
    High-level interface for retraining memory models from session context.

    Coordinates between MemoryManager, TrainingDataGenerator, and LoRATrainer
    to enable continuous learning from conversations.
    """

    def __init__(
        self,
        core=None,
        memory_manager=None,
        trainer: Optional[LoRATrainer] = None,
        config: Optional[TrainingConfig] = None,
        llm_client=None
    ):
        """
        Initialize the retuner.

        Args:
            core: DETCore instance for approval checks.
            memory_manager: MemoryManager instance.
            trainer: LoRATrainer instance.
            config: Training configuration.
            llm_client: LLM client for data generation.
        """
        self.core = core
        self.memory_manager = memory_manager
        self.trainer = trainer or (LoRATrainer(config) if MLX_AVAILABLE else None)
        self.data_generator = TrainingDataGenerator(llm_client)

        self._training_history: List[TrainingJob] = []

    def can_retrain(self, domain: str, min_examples: int = 10) -> bool:
        """
        Check if retraining is possible and advisable.

        Args:
            domain: Target domain.
            min_examples: Minimum examples required.

        Returns:
            True if retraining should proceed.
        """
        if not MLX_AVAILABLE:
            return False

        if self.memory_manager is None:
            return False

        # Check DET core approval if available
        if self.core:
            # Check if we have enough resources and stable state
            valence, arousal, _ = self.core.get_self_affect()
            if valence < -0.3 or arousal > 0.8:
                return False  # Too strained or agitated

        # Check if we have enough data
        from .memory import MemoryDomain
        try:
            domain_enum = MemoryDomain[domain.upper()]
            memories = self.memory_manager.memories.get(domain_enum, [])
            if len(memories) < min_examples:
                return False
        except KeyError:
            return False

        return True

    def retrain_from_memories(
        self,
        domain: str,
        use_llm: bool = False,
        callback: Optional[Callable[[TrainingJob], None]] = None
    ) -> Optional[TrainingJob]:
        """
        Retrain a domain model from stored memories.

        Args:
            domain: Target domain.
            use_llm: Use LLM for enhanced data generation.
            callback: Progress callback.

        Returns:
            TrainingJob if training was started, None otherwise.
        """
        if not self.can_retrain(domain):
            return None

        # Get memories for this domain
        from .memory import MemoryDomain
        domain_enum = MemoryDomain[domain.upper()]
        memories = self.memory_manager.memories.get(domain_enum, [])

        # Generate training examples
        examples = []
        for memory in memories:
            memory_examples = self.data_generator.generate_from_memory(
                memory.content,
                domain,
                use_llm=use_llm
            )
            # Weight by importance
            for ex in memory_examples:
                ex.importance = memory.importance
            examples.extend(memory_examples)

        if not examples:
            return None

        # Split into train/val
        val_size = max(1, len(examples) // 10)
        val_data = examples[:val_size]
        train_data = examples[val_size:]

        # Run training
        job = self.trainer.train(
            train_data=train_data,
            domain=domain,
            val_data=val_data if val_data else None,
            callback=callback
        )

        self._training_history.append(job)

        # Update domain config
        if job.status == TrainingStatus.COMPLETED and self.memory_manager:
            domain_config = self.memory_manager.domains.get(domain_enum)
            if domain_config:
                domain_config.lora_adapter_path = job.adapter_path
                domain_config.training_samples = len(train_data)
                domain_config.last_training = time.time()

        return job

    def retrain_from_context(
        self,
        messages: List[Dict[str, str]],
        domain: str = "DIALOGUE",
        callback: Optional[Callable[[TrainingJob], None]] = None
    ) -> Optional[TrainingJob]:
        """
        Retrain from conversation context (for session consolidation).

        Args:
            messages: Conversation messages.
            domain: Target domain.
            callback: Progress callback.

        Returns:
            TrainingJob if training was started.
        """
        if not MLX_AVAILABLE or not self.trainer:
            return None

        # Check DET approval
        if self.core:
            valence, _, _ = self.core.get_self_affect()
            if valence < -0.3:
                return None

        # Generate examples from context
        examples = self.data_generator.generate_from_context(messages, domain)

        if len(examples) < 3:
            return None

        # Run training
        job = self.trainer.train(
            train_data=examples,
            domain=domain,
            callback=callback
        )

        self._training_history.append(job)

        return job

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return [
            {
                "job_id": job.job_id,
                "domain": job.domain,
                "status": job.status.value,
                "num_examples": job.num_examples,
                "created_at": job.created_at,
                "completed_at": job.completed_at,
            }
            for job in self._training_history
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get retuner status."""
        return {
            "mlx_available": MLX_AVAILABLE,
            "trainer_ready": self.trainer is not None,
            "adapters": self.trainer.list_adapters() if self.trainer else [],
            "training_history": self.get_training_history(),
            "has_core": self.core is not None,
            "has_memory_manager": self.memory_manager is not None,
        }


# Export MLX availability for checking
def is_mlx_available() -> bool:
    """Check if MLX is available."""
    return MLX_AVAILABLE
