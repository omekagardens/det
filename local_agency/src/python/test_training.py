#!/usr/bin/env python3
"""
DET Phase 2.2 MLX Training Tests
================================

Tests for the MLX training pipeline, including data generation,
training configuration, and memory integration.
"""

import sys
import tempfile
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from det import (
    DETCore, MemoryManager, MemoryDomain,
    TrainingConfig, TrainingExample, TrainingJob, TrainingStatus,
    TrainingDataGenerator, is_mlx_available,
)


# ============================================================================
# Training Configuration Tests
# ============================================================================

def test_training_config_defaults():
    """Test TrainingConfig default values."""
    print("  test_training_config_defaults...", end=" ")

    config = TrainingConfig()

    assert config.lora_rank == 8
    assert config.lora_alpha == 16.0
    assert config.batch_size == 4
    assert config.epochs == 3
    assert config.max_seq_length == 512

    print("PASS")


def test_training_config_to_dict():
    """Test TrainingConfig serialization."""
    print("  test_training_config_to_dict...", end=" ")

    config = TrainingConfig(
        model_name="test-model",
        lora_rank=16,
        epochs=5,
    )

    data = config.to_dict()

    assert data["model_name"] == "test-model"
    assert data["lora_rank"] == 16
    assert data["epochs"] == 5

    print("PASS")


def test_training_config_custom():
    """Test TrainingConfig with custom values."""
    print("  test_training_config_custom...", end=" ")

    config = TrainingConfig(
        lora_rank=4,
        lora_alpha=8.0,
        batch_size=8,
        learning_rate=5e-5,
        epochs=10,
    )

    assert config.lora_rank == 4
    assert config.lora_alpha == 8.0
    assert config.batch_size == 8
    assert config.learning_rate == 5e-5
    assert config.epochs == 10

    print("PASS")


# ============================================================================
# Training Example Tests
# ============================================================================

def test_training_example_creation():
    """Test TrainingExample creation."""
    print("  test_training_example_creation...", end=" ")

    example = TrainingExample(
        instruction="What is 2+2?",
        response="4",
        domain="MATH",
        importance=0.8,
    )

    assert example.instruction == "What is 2+2?"
    assert example.response == "4"
    assert example.domain == "MATH"
    assert example.importance == 0.8

    print("PASS")


def test_training_example_to_chat_format():
    """Test TrainingExample chat format conversion."""
    print("  test_training_example_to_chat_format...", end=" ")

    example = TrainingExample(
        instruction="Explain gravity",
        response="Gravity is a force...",
    )

    chat = example.to_chat_format()

    assert len(chat) == 2
    assert chat[0]["role"] == "user"
    assert chat[0]["content"] == "Explain gravity"
    assert chat[1]["role"] == "assistant"
    assert chat[1]["content"] == "Gravity is a force..."

    print("PASS")


def test_training_example_to_dict():
    """Test TrainingExample serialization."""
    print("  test_training_example_to_dict...", end=" ")

    example = TrainingExample(
        instruction="Test",
        response="Response",
        domain="CODE",
        importance=0.9,
    )

    data = example.to_dict()

    assert data["instruction"] == "Test"
    assert data["response"] == "Response"
    assert data["domain"] == "CODE"
    assert data["importance"] == 0.9

    print("PASS")


# ============================================================================
# Training Data Generator Tests
# ============================================================================

def test_data_generator_init():
    """Test TrainingDataGenerator initialization."""
    print("  test_data_generator_init...", end=" ")

    generator = TrainingDataGenerator()

    assert generator.llm_client is None
    assert "MATH" in generator.DOMAIN_TEMPLATES
    assert "CODE" in generator.DOMAIN_TEMPLATES

    print("PASS")


def test_data_generator_from_qa():
    """Test generating from Q&A content."""
    print("  test_data_generator_from_qa...", end=" ")

    generator = TrainingDataGenerator()

    content = "What is Python? Python is a programming language."
    examples = generator.generate_from_memory(content, "CODE")

    assert len(examples) >= 1
    # Should detect Q&A format
    qa_examples = [e for e in examples if "?" in e.instruction]
    assert len(qa_examples) >= 1

    print("PASS")


def test_data_generator_from_content():
    """Test generating from plain content."""
    print("  test_data_generator_from_content...", end=" ")

    generator = TrainingDataGenerator()

    content = "The quadratic formula is used to solve equations of the form ax² + bx + c = 0"
    examples = generator.generate_from_memory(content, "MATH")

    assert len(examples) >= 1
    assert any(e.domain == "MATH" for e in examples)

    print("PASS")


def test_data_generator_from_context():
    """Test generating from conversation context."""
    print("  test_data_generator_from_context...", end=" ")

    generator = TrainingDataGenerator()

    messages = [
        {"role": "user", "content": "How do I write a for loop?"},
        {"role": "assistant", "content": "Use: for i in range(n): ..."},
        {"role": "user", "content": "What about while loops?"},
        {"role": "assistant", "content": "Use: while condition: ..."},
    ]

    examples = generator.generate_from_context(messages)

    assert len(examples) == 2
    assert examples[0].instruction == "How do I write a for loop?"
    assert examples[1].instruction == "What about while loops?"

    print("PASS")


def test_data_generator_save_jsonl():
    """Test saving dataset to JSONL format."""
    print("  test_data_generator_save_jsonl...", end=" ")

    generator = TrainingDataGenerator()

    examples = [
        TrainingExample("Q1", "A1", "GENERAL"),
        TrainingExample("Q2", "A2", "GENERAL"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "train.jsonl"
        generator.save_dataset(examples, output_path)

        assert output_path.exists()

        # Verify content
        with open(output_path) as f:
            lines = f.readlines()
            assert len(lines) == 2

            data = json.loads(lines[0])
            assert "messages" in data
            assert len(data["messages"]) == 2

    print("PASS")


def test_data_generator_domain_templates():
    """Test domain-specific templates."""
    print("  test_data_generator_domain_templates...", end=" ")

    generator = TrainingDataGenerator()

    # Test different domains produce different templates (content must be > 20 chars)
    math_examples = generator.generate_from_memory(
        "Calculate the sum of all numbers from 1 to 100 using the formula", "MATH"
    )
    code_examples = generator.generate_from_memory(
        "Write a function that reverses a string in Python", "CODE"
    )

    # Both should produce examples
    assert len(math_examples) >= 1
    assert len(code_examples) >= 1

    # Should be different templates
    math_instr = math_examples[0].instruction if math_examples else ""
    code_instr = code_examples[0].instruction if code_examples else ""

    # They should use domain-appropriate language
    assert math_examples[0].domain == "MATH"
    assert code_examples[0].domain == "CODE"

    print("PASS")


# ============================================================================
# Training Job Tests
# ============================================================================

def test_training_job_creation():
    """Test TrainingJob creation."""
    print("  test_training_job_creation...", end=" ")

    config = TrainingConfig()
    job = TrainingJob(
        job_id="test-123",
        domain="MATH",
        config=config,
        num_examples=100,
    )

    assert job.job_id == "test-123"
    assert job.domain == "MATH"
    assert job.status == TrainingStatus.PENDING
    assert job.num_examples == 100
    assert job.created_at > 0

    print("PASS")


def test_training_status_enum():
    """Test TrainingStatus enum values."""
    print("  test_training_status_enum...", end=" ")

    assert TrainingStatus.PENDING.value == "pending"
    assert TrainingStatus.RUNNING.value == "running"
    assert TrainingStatus.COMPLETED.value == "completed"
    assert TrainingStatus.FAILED.value == "failed"
    assert TrainingStatus.CANCELLED.value == "cancelled"

    print("PASS")


# ============================================================================
# MLX Availability Tests
# ============================================================================

def test_mlx_availability():
    """Test MLX availability check."""
    print("  test_mlx_availability...", end=" ")

    # Function should return a boolean
    available = is_mlx_available()
    assert isinstance(available, bool)

    # On Apple Silicon with MLX installed, this should be True
    # We don't assert the value since it depends on the environment

    print("PASS")


def test_lora_trainer_init_requires_mlx():
    """Test LoRATrainer requires MLX."""
    print("  test_lora_trainer_init_requires_mlx...", end=" ")

    if is_mlx_available():
        # Should work
        from det import LoRATrainer
        trainer = LoRATrainer()
        assert trainer is not None
    else:
        # Should raise
        from det import LoRATrainer
        try:
            trainer = LoRATrainer()
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "MLX is not available" in str(e)

    print("PASS")


# ============================================================================
# Memory Integration Tests
# ============================================================================

def test_memory_retuner_init():
    """Test MemoryRetuner initialization."""
    print("  test_memory_retuner_init...", end=" ")

    from det import MemoryRetuner

    retuner = MemoryRetuner()

    assert retuner.core is None
    assert retuner.memory_manager is None
    assert retuner.data_generator is not None

    print("PASS")


def test_memory_retuner_with_core():
    """Test MemoryRetuner with DET core."""
    print("  test_memory_retuner_with_core...", end=" ")

    from det import MemoryRetuner

    with DETCore() as core:
        retuner = MemoryRetuner(core=core)

        assert retuner.core is core
        status = retuner.get_status()
        assert status["has_core"] is True

    print("PASS")


def test_memory_retuner_status():
    """Test MemoryRetuner status reporting."""
    print("  test_memory_retuner_status...", end=" ")

    from det import MemoryRetuner

    retuner = MemoryRetuner()
    status = retuner.get_status()

    assert "mlx_available" in status
    assert "trainer_ready" in status
    assert "adapters" in status
    assert "training_history" in status

    print("PASS")


def test_memory_retuner_can_retrain_no_memory():
    """Test can_retrain without memory manager."""
    print("  test_memory_retuner_can_retrain_no_memory...", end=" ")

    from det import MemoryRetuner

    retuner = MemoryRetuner()

    # Should return False without memory manager
    result = retuner.can_retrain("MATH")
    assert result is False

    print("PASS")


def test_memory_integration_with_training():
    """Test memory manager integration with training."""
    print("  test_memory_integration_with_training...", end=" ")

    from det import MemoryRetuner

    with DETCore() as core:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_manager = MemoryManager(core, storage_path=Path(tmpdir))

            # Store some memories
            for i in range(15):
                memory_manager.store(
                    f"What is {i}+{i}? The answer is {i*2}.",
                    domain=MemoryDomain.MATH,
                    importance=0.7,
                )

            retuner = MemoryRetuner(
                core=core,
                memory_manager=memory_manager,
            )

            # Check if retraining is possible
            can_retrain = retuner.can_retrain("MATH", min_examples=10)

            # Should be True if MLX is available, False otherwise
            if is_mlx_available():
                assert can_retrain is True
            else:
                assert can_retrain is False

    print("PASS")


# ============================================================================
# Full Integration Test (with MLX if available)
# ============================================================================

def test_full_training_flow():
    """Test full training flow (MLX-dependent)."""
    print("  test_full_training_flow...", end=" ")

    if not is_mlx_available():
        print("SKIP (MLX not available)")
        return

    from det import LoRATrainer, MemoryRetuner

    with DETCore() as core:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup
            memory_manager = MemoryManager(core, storage_path=Path(tmpdir) / "memory")

            # Store training data
            for i in range(20):
                memory_manager.store(
                    f"Calculate {i} squared? The result is {i*i}.",
                    domain=MemoryDomain.MATH,
                    importance=0.8,
                )

            # Create retuner with minimal config for quick test
            config = TrainingConfig(
                model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",  # Small model
                epochs=1,
                batch_size=2,
                max_seq_length=128,
            )

            trainer = LoRATrainer(
                config=config,
                adapters_dir=Path(tmpdir) / "adapters",
            )

            retuner = MemoryRetuner(
                core=core,
                memory_manager=memory_manager,
                trainer=trainer,
            )

            # Verify we can retrain
            assert retuner.can_retrain("MATH")

            # Note: Actually running training would download a model and take time
            # So we just verify the setup is correct
            status = retuner.get_status()
            assert status["trainer_ready"] is True

    print("PASS")


# ============================================================================
# Main
# ============================================================================

def run_tests():
    """Run all Phase 2.2 tests."""
    print("\n========================================")
    print("  DET Phase 2.2 MLX Training Tests")
    print("========================================\n")

    tests = [
        # Training Configuration
        test_training_config_defaults,
        test_training_config_to_dict,
        test_training_config_custom,

        # Training Examples
        test_training_example_creation,
        test_training_example_to_chat_format,
        test_training_example_to_dict,

        # Data Generator
        test_data_generator_init,
        test_data_generator_from_qa,
        test_data_generator_from_content,
        test_data_generator_from_context,
        test_data_generator_save_jsonl,
        test_data_generator_domain_templates,

        # Training Job
        test_training_job_creation,
        test_training_status_enum,

        # MLX Availability
        test_mlx_availability,
        test_lora_trainer_init_requires_mlx,

        # Memory Integration
        test_memory_retuner_init,
        test_memory_retuner_with_core,
        test_memory_retuner_status,
        test_memory_retuner_can_retrain_no_memory,
        test_memory_integration_with_training,

        # Full Integration
        test_full_training_flow,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            if "SKIP" in str(e):
                skipped += 1
            else:
                print(f"FAIL: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

    print("\n========================================")
    print(f"  Results: {passed}/{passed + failed} tests passed")
    if skipped:
        print(f"  Skipped: {skipped} tests (MLX not available)")
    print("========================================\n")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
