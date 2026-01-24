#!/usr/bin/env python3
"""
Tests for DET-OS CLI integration.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from det.os.existence.bootstrap import DETOSBootstrap, BootConfig, BootState
from det.os.existence.runtime import CreatureState
from det.os.creatures.base import CreatureWrapper
from det.os.creatures.memory import MemoryCreature, spawn_memory_creature


def test_kernel_boot():
    """Test DET-OS kernel boots successfully."""
    print("  test_kernel_boot...", end=" ")

    config = BootConfig(total_F=100000.0, grace_pool=1000.0, debug=False)
    bootstrap = DETOSBootstrap(config=config)

    success = bootstrap.boot()
    assert success, "Kernel boot failed"
    assert bootstrap.state == BootState.RUNNING

    bootstrap.halt()
    assert bootstrap.state == BootState.HALTED

    print("PASS")


def test_kernel_ticks():
    """Test kernel tick execution."""
    print("  test_kernel_ticks...", end=" ")

    config = BootConfig(total_F=100000.0, tick_rate=50.0)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    cid = bootstrap.spawn("test", initial_f=100.0, initial_a=0.8)
    bootstrap.runtime.creatures[cid].state = CreatureState.RUNNING

    initial_tick = bootstrap.runtime.tick_count

    for _ in range(10):
        bootstrap.tick()

    assert bootstrap.runtime.tick_count == initial_tick + 10

    bootstrap.halt()
    print("PASS")


def test_creature_wrapper():
    """Test CreatureWrapper base class."""
    print("  test_creature_wrapper...", end=" ")

    config = BootConfig(total_F=100000.0)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    cid = bootstrap.spawn("test", initial_f=100.0, initial_a=0.8)
    bootstrap.runtime.creatures[cid].state = CreatureState.RUNNING

    wrapper = CreatureWrapper(bootstrap.runtime, cid)

    assert wrapper.F == 100.0
    assert wrapper.a == 0.8
    assert wrapper.is_alive
    assert wrapper.presence == 100.0 * 0.8 * 1.0  # F * a * C_self

    wrapper.inject_resource(50.0)
    assert wrapper.F == 150.0

    bootstrap.halt()
    print("PASS")


def test_memory_creature_spawn():
    """Test MemoryCreature spawns correctly."""
    print("  test_memory_creature_spawn...", end=" ")

    config = BootConfig(total_F=100000.0)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    memory = spawn_memory_creature(bootstrap.runtime, "memory", initial_f=50.0, initial_a=0.5)

    assert memory.cid > 0
    assert memory.F == 50.0
    assert memory.a == 0.5
    assert len(memory.memories) == 0

    bootstrap.halt()
    print("PASS")


def test_memory_store_direct():
    """Test direct memory storage."""
    print("  test_memory_store_direct...", end=" ")

    config = BootConfig(total_F=100000.0)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    memory = spawn_memory_creature(bootstrap.runtime, "memory", initial_f=50.0)

    success = memory.store("Hello world", source_cid=0)
    assert success
    assert len(memory.memories) == 1
    assert memory.memories[0].content == "Hello world"

    # Store costs F
    assert memory.F < 50.0

    bootstrap.halt()
    print("PASS")


def test_memory_recall_direct():
    """Test direct memory recall."""
    print("  test_memory_recall_direct...", end=" ")

    config = BootConfig(total_F=100000.0)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    memory = spawn_memory_creature(bootstrap.runtime, "memory", initial_f=50.0)

    memory.store("The capital of France is Paris")
    memory.store("Python is a programming language")
    memory.store("The Eiffel Tower is in Paris")

    results = memory.recall("Paris", limit=5)
    assert len(results) >= 2  # Should match France and Eiffel Tower

    bootstrap.halt()
    print("PASS")


def test_creature_bonding():
    """Test bond creation between creatures."""
    print("  test_creature_bonding...", end=" ")

    config = BootConfig(total_F=100000.0)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    # Create two creatures
    cid_a = bootstrap.spawn("creature_a", initial_f=50.0)
    cid_b = bootstrap.spawn("creature_b", initial_f=50.0)
    bootstrap.runtime.creatures[cid_a].state = CreatureState.RUNNING
    bootstrap.runtime.creatures[cid_b].state = CreatureState.RUNNING

    wrapper_a = CreatureWrapper(bootstrap.runtime, cid_a)
    wrapper_b = CreatureWrapper(bootstrap.runtime, cid_b)

    # Bond them
    channel_id = wrapper_a.bond_with(cid_b, coherence=0.8)
    wrapper_b.bonds[cid_a] = channel_id  # Register reverse

    assert channel_id in bootstrap.runtime.channels
    assert wrapper_a.get_bond_coherence(cid_b) == 0.8

    bootstrap.halt()
    print("PASS")


def test_bond_message_passing():
    """Test message passing through bonds."""
    print("  test_bond_message_passing...", end=" ")

    config = BootConfig(total_F=100000.0)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    # Create two creatures
    cid_a = bootstrap.spawn("sender", initial_f=50.0)
    cid_b = bootstrap.spawn("receiver", initial_f=50.0)
    bootstrap.runtime.creatures[cid_a].state = CreatureState.RUNNING
    bootstrap.runtime.creatures[cid_b].state = CreatureState.RUNNING

    sender = CreatureWrapper(bootstrap.runtime, cid_a)
    receiver = CreatureWrapper(bootstrap.runtime, cid_b)

    # Bond them with high coherence for reliable delivery
    channel_id = sender.bond_with(cid_b, coherence=1.0)
    receiver.bonds[cid_a] = channel_id

    # Send message
    success = sender.send_to(cid_b, {"type": "hello", "data": 42})
    assert success

    # Receive message
    msg = receiver.receive_from(cid_a)
    assert msg is not None
    assert msg["type"] == "hello"
    assert msg["data"] == 42

    bootstrap.halt()
    print("PASS")


def test_memory_via_bond():
    """Test memory operations through bond channel."""
    print("  test_memory_via_bond...", end=" ")

    config = BootConfig(total_F=100000.0)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    # Create LLM-like creature
    cid_llm = bootstrap.spawn("llm", initial_f=100.0, initial_a=0.8)
    bootstrap.runtime.creatures[cid_llm].state = CreatureState.RUNNING
    llm = CreatureWrapper(bootstrap.runtime, cid_llm)

    # Create memory creature
    memory = spawn_memory_creature(bootstrap.runtime, "memory", initial_f=50.0)

    # Bond them
    channel_id = llm.bond_with(memory.cid, coherence=0.9)
    memory.bonds[llm.cid] = channel_id

    # Send store request through bond
    llm.send_to(memory.cid, {"type": "store", "content": "Test memory"})

    # Memory processes
    memory.process_messages()

    # Check storage
    assert len(memory.memories) == 1

    # Check for ack
    responses = llm.receive_all_from(memory.cid)
    assert len(responses) == 1
    assert responses[0]["type"] == "store_ack"
    assert responses[0]["success"]

    bootstrap.halt()
    print("PASS")


def test_llm_creature():
    """Test LLMCreature class."""
    print("  test_llm_creature...", end=" ")

    from det_os_cli import LLMCreature

    config = BootConfig(total_F=100000.0)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    cid = bootstrap.spawn("llm_agent", initial_f=100.0, initial_a=0.8)
    bootstrap.runtime.creatures[cid].state = CreatureState.RUNNING

    llm = LLMCreature(bootstrap.runtime, cid, "http://localhost:11434", "test")

    assert llm.F == 100.0
    assert llm.a == 0.8
    assert llm.is_alive

    bootstrap.halt()
    print("PASS")


def test_llm_memory_integration():
    """Test LLMCreature with memory bonding."""
    print("  test_llm_memory_integration...", end=" ")

    from det_os_cli import LLMCreature

    config = BootConfig(total_F=100000.0)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    # Create LLM
    cid = bootstrap.spawn("llm_agent", initial_f=100.0, initial_a=0.8)
    bootstrap.runtime.creatures[cid].state = CreatureState.RUNNING
    llm = LLMCreature(bootstrap.runtime, cid, "http://localhost:11434", "test")

    # Create and bond memory
    memory = spawn_memory_creature(bootstrap.runtime, "memory", initial_f=50.0)
    llm.bond_to_memory(memory, coherence=0.9)

    # Store via LLM
    success = llm.store_memory("Important fact")
    assert success

    memory.process_messages()
    assert len(memory.memories) == 1

    # Recall via LLM
    success = llm.recall_memories("fact")
    assert success

    memory.process_messages()
    responses = llm.get_memory_responses()

    # Should have store_ack and recall response
    assert len(responses) >= 1

    bootstrap.halt()
    print("PASS")


def test_can_think_check():
    """Test the can_think resource check."""
    print("  test_can_think_check...", end=" ")

    from det_os_cli import LLMCreature

    config = BootConfig(total_F=100000.0)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    cid = bootstrap.spawn("llm_agent", initial_f=10.0, initial_a=0.8)
    bootstrap.runtime.creatures[cid].state = CreatureState.RUNNING

    llm = LLMCreature(bootstrap.runtime, cid, "http://localhost:11434", "test")

    # With F=10.0 and cost_per_output_token=0.05
    assert llm.can_think(100)  # 100 * 0.05 = 5.0 < 10.0
    assert not llm.can_think(300)  # 300 * 0.05 = 15.0 > 10.0

    bootstrap.halt()
    print("PASS")


def test_bond_coherence_stability():
    """Test that perfect bonds (coherence=1.0) don't decay over time."""
    print("  test_bond_coherence_stability...", end=" ")

    from det_os_cli import LLMCreature

    config = BootConfig(total_F=100000.0, tick_rate=100.0)  # Fast ticks
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    # Create LLM
    cid = bootstrap.spawn("llm_agent", initial_f=100.0, initial_a=0.8)
    bootstrap.runtime.creatures[cid].state = CreatureState.RUNNING
    llm = LLMCreature(bootstrap.runtime, cid, "http://localhost:11434", "test")

    # Create and bond memory with perfect coherence
    memory = spawn_memory_creature(bootstrap.runtime, "memory", initial_f=50.0)
    llm.bond_to_memory(memory, coherence=1.0)

    initial_coherence = llm.get_bond_coherence(memory.cid)
    assert initial_coherence == 1.0, f"Expected 1.0, got {initial_coherence}"

    # Run many ticks (simulating idle time)
    for _ in range(100):
        bootstrap.tick()

    # Coherence should remain stable (>= 0.99 for perfect bonds)
    final_coherence = llm.get_bond_coherence(memory.cid)
    assert final_coherence >= 0.99, f"Bond coherence decayed to {final_coherence}"

    # Bond should still work
    success = llm.store_memory("Test after many ticks")
    assert success, "Store failed after ticks"

    memory.process_messages()
    assert len(memory.memories) == 1, "Memory not stored"

    bootstrap.halt()
    print("PASS")


def run_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DET-OS CLI Integration Tests")
    print("=" * 60)

    print("\nKernel Tests:")
    test_kernel_boot()
    test_kernel_ticks()

    print("\nCreature Wrapper Tests:")
    test_creature_wrapper()

    print("\nMemory Creature Tests:")
    test_memory_creature_spawn()
    test_memory_store_direct()
    test_memory_recall_direct()

    print("\nBonding Tests:")
    test_creature_bonding()
    test_bond_message_passing()
    test_memory_via_bond()

    print("\nLLM Creature Tests:")
    test_llm_creature()
    test_llm_memory_integration()
    test_can_think_check()
    test_bond_coherence_stability()

    print("\n" + "=" * 60)
    print("All DET-OS CLI tests passed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_tests()
