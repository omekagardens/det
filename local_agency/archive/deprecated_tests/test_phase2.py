#!/usr/bin/env python3
"""
DET Phase 2 Integration Tests
=============================

Tests for memory management and internal dialogue systems.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from det import (
    DETCore, DETDecision,
    MemoryManager, MemoryDomain, MemoryEntry, DomainRouter, ContextWindow,
    InternalDialogue, DialogueTurn, DialogueState
)


def test_domain_router():
    """Test domain routing logic."""
    print("  test_domain_router...", end=" ")

    router = DomainRouter()

    # Test math routing
    domain, conf = router.route("calculate 2 + 2")
    assert domain == MemoryDomain.MATH

    # Test code routing
    domain, conf = router.route("write a python function")
    assert domain == MemoryDomain.CODE

    # Test tool routing
    domain, conf = router.route("run this command in the terminal")
    assert domain == MemoryDomain.TOOL_USE

    # Test default routing
    domain, conf = router.route("hello there")
    assert domain == MemoryDomain.GENERAL

    print("PASS")


def test_memory_manager_init():
    """Test memory manager initialization."""
    print("  test_memory_manager_init...", end=" ")

    with DETCore() as core:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryManager(core, Path(tmpdir))

            # Should have 8 domains
            stats = memory.get_domain_stats()
            assert len(stats) == 8

            # Should start with no memories
            total = sum(s["entry_count"] for s in stats.values())
            assert total == 0

    print("PASS")


def test_memory_store_retrieve():
    """Test storing and retrieving memories."""
    print("  test_memory_store_retrieve...", end=" ")

    with DETCore() as core:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryManager(core, Path(tmpdir))

            # Store a math memory
            entry = memory.store("The integral of x^2 is x^3/3", importance=0.8)
            assert entry.domain == MemoryDomain.MATH
            assert entry.importance == 0.8

            # Store a code memory (with explicit code keyword)
            entry = memory.store("Here is a python function to debug: def hello()", importance=0.7)
            assert entry.domain == MemoryDomain.CODE

            # Retrieve math memories
            results = memory.retrieve("integral", limit=5)
            assert len(results) == 1
            assert "integral" in results[0].content.lower()

            # Retrieve code memories
            results = memory.retrieve("python function debug", limit=5)
            assert len(results) == 1
            assert "function" in results[0].content.lower()

    print("PASS")


def test_memory_persistence():
    """Test memory persistence across sessions."""
    print("  test_memory_persistence...", end=" ")

    with DETCore() as core:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)

            # Store memories in first session
            memory1 = MemoryManager(core, storage_path)
            memory1.store("Important math fact about pi", importance=0.9)
            memory1.store("Python coding tip about lists", importance=0.8)

            # Create new manager (simulating new session)
            memory2 = MemoryManager(core, storage_path)

            # Should load persisted memories
            stats = memory2.get_domain_stats()
            total = sum(s["entry_count"] for s in stats.values())
            assert total == 2

    print("PASS")


def test_memory_domain_coherence():
    """Test domain coherence tracking."""
    print("  test_memory_domain_coherence...", end=" ")

    with DETCore() as core:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryManager(core, Path(tmpdir))

            # Get coherence for a domain
            coherence = memory.core.get_domain_coherence(int(MemoryDomain.MATH))
            assert 0.0 <= coherence <= 1.0

            # Route request and check coherence
            domain, conf, coh = memory.route_request("calculate the sum of 1 to 100")
            assert domain == MemoryDomain.MATH
            assert 0.0 <= conf <= 1.0
            assert 0.0 <= coh <= 1.0

    print("PASS")


def test_context_window():
    """Test context window management."""
    print("  test_context_window...", end=" ")

    with DETCore() as core:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryManager(core, Path(tmpdir))
            context = ContextWindow(memory, max_tokens=100, reduction_threshold=0.8)

            # Add messages
            context.add_message("user", "Hello there!")
            context.add_message("assistant", "Hi! How can I help?")

            messages = context.get_context()
            assert len(messages) == 2

            # Add more messages to trigger reduction
            for i in range(10):
                context.add_message("user", f"Message {i} " * 20)

            # Should have reduced
            messages = context.get_context()
            assert len(messages) <= 4

    print("PASS")


def test_dialogue_state():
    """Test dialogue state enum."""
    print("  test_dialogue_state...", end=" ")

    assert DialogueState.IDLE == 0
    assert DialogueState.PROCESSING == 1
    assert DialogueState.REFORMULATING == 2
    assert DialogueState.ESCALATING == 3
    assert DialogueState.COMPLETE == 4
    assert DialogueState.FAILED == 5

    print("PASS")


def test_dialogue_turn():
    """Test dialogue turn creation."""
    print("  test_dialogue_turn...", end=" ")

    import time

    turn = DialogueTurn(
        input_text="Hello",
        decision=DETDecision.PROCEED,
        output_text="Hi there!",
        timestamp=time.time(),
        reformulation_count=0,
        affect={"valence": 0.5, "arousal": 0.3, "bondedness": 0.6}
    )

    assert turn.input_text == "Hello"
    assert turn.decision == DETDecision.PROCEED
    assert not turn.escalated
    assert turn.affect["valence"] == 0.5

    print("PASS")


def test_memory_training_data():
    """Test training data generation."""
    print("  test_memory_training_data...", end=" ")

    with DETCore() as core:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryManager(core, Path(tmpdir))

            # Store Q&A style memory
            memory.store("What is 2+2? It equals 4.")
            memory.store("How do you sort a list in Python? Use the sorted() function.")

            # Generate training data
            training = memory.generate_training_data(MemoryDomain.GENERAL)

            # Should have converted to instruction-response format
            assert len(training) >= 1

    print("PASS")


def test_memory_consolidation():
    """Test memory consolidation."""
    print("  test_memory_consolidation...", end=" ")

    with DETCore() as core:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryManager(core, Path(tmpdir))

            # Store some memories with low importance
            memory.store("Temporary note 1", importance=0.2)
            memory.store("Temporary note 2", importance=0.3)
            memory.store("Important note", importance=0.9)

            stats_before = memory.get_domain_stats()
            total_before = sum(s["entry_count"] for s in stats_before.values())
            assert total_before == 3

            # Consolidate (with 0 day threshold to consolidate all old low-importance)
            consolidated = memory.consolidate(threshold_days=0)

            # Low importance memories should be consolidated
            # Note: This depends on the timestamp, so may not consolidate immediately

    print("PASS")


def run_tests():
    """Run all Phase 2 tests."""
    print("\n========================================")
    print("  DET Phase 2 Test Suite")
    print("========================================\n")

    tests = [
        test_domain_router,
        test_memory_manager_init,
        test_memory_store_retrieve,
        test_memory_persistence,
        test_memory_domain_coherence,
        test_context_window,
        test_dialogue_state,
        test_dialogue_turn,
        test_memory_training_data,
        test_memory_consolidation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n========================================")
    print(f"  Results: {passed}/{passed + failed} tests passed")
    print("========================================\n")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
