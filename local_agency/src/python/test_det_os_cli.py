#!/usr/bin/env python3
"""
Tests for DET-OS CLI integration.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from det.os.existence.bootstrap import DETOSBootstrap, BootConfig, BootState
from det.os.existence.runtime import CreatureState


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


def test_llm_creature_spawn():
    """Test LLM creature spawns correctly."""
    print("  test_llm_creature_spawn...", end=" ")

    config = BootConfig(total_F=100000.0, grace_pool=1000.0)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    # Spawn LLM creature
    cid = bootstrap.spawn("llm_agent", initial_f=100.0, initial_a=0.8)

    assert cid > 0
    assert cid in bootstrap.runtime.creatures

    creature = bootstrap.runtime.creatures[cid]
    assert creature.name == "llm_agent"
    assert creature.F == 100.0
    assert creature.a == 0.8

    bootstrap.halt()
    print("PASS")


def test_creature_resource_depletion():
    """Test that creature F can be depleted."""
    print("  test_creature_resource_depletion...", end=" ")

    config = BootConfig(total_F=100000.0)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    cid = bootstrap.spawn("llm_agent", initial_f=100.0, initial_a=0.8)
    creature = bootstrap.runtime.creatures[cid]
    creature.state = CreatureState.RUNNING

    # Manually deplete resource (simulating token cost)
    initial_F = creature.F
    creature.F -= 10.0
    assert creature.F == initial_F - 10.0

    # Deplete more
    creature.F -= 50.0
    assert creature.F == 40.0

    bootstrap.halt()
    print("PASS")


def test_creature_state_tracking():
    """Test creature state transitions."""
    print("  test_creature_state_tracking...", end=" ")

    config = BootConfig(total_F=100000.0)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    cid = bootstrap.spawn("llm_agent", initial_f=100.0, initial_a=0.8)
    creature = bootstrap.runtime.creatures[cid]

    # Initially embryonic
    assert creature.state == CreatureState.EMBRYONIC

    # Activate
    creature.state = CreatureState.RUNNING
    assert creature.is_alive()

    # Check alive status
    creature.F = 0.0
    # Even with F=0, state doesn't auto-change until tick
    assert creature.state == CreatureState.RUNNING

    bootstrap.halt()
    print("PASS")


def test_multiple_creatures():
    """Test spawning multiple creatures."""
    print("  test_multiple_creatures...", end=" ")

    config = BootConfig(total_F=100000.0)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    # Spawn multiple creatures
    cid1 = bootstrap.spawn("llm_agent", initial_f=100.0, initial_a=0.8)
    cid2 = bootstrap.spawn("memory_creature", initial_f=50.0, initial_a=0.5)
    cid3 = bootstrap.spawn("tool_creature", initial_f=25.0, initial_a=0.3)

    assert len(bootstrap.runtime.creatures) == 4  # kernel + 3

    # Verify each
    assert bootstrap.runtime.creatures[cid1].name == "llm_agent"
    assert bootstrap.runtime.creatures[cid2].name == "memory_creature"
    assert bootstrap.runtime.creatures[cid3].name == "tool_creature"

    bootstrap.halt()
    print("PASS")


def test_kernel_ticks():
    """Test kernel tick execution."""
    print("  test_kernel_ticks...", end=" ")

    config = BootConfig(total_F=100000.0, tick_rate=50.0)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    cid = bootstrap.spawn("llm_agent", initial_f=100.0, initial_a=0.8)
    bootstrap.runtime.creatures[cid].state = CreatureState.RUNNING

    initial_tick = bootstrap.runtime.tick_count

    # Run some ticks
    for _ in range(10):
        bootstrap.tick()

    assert bootstrap.runtime.tick_count == initial_tick + 10

    bootstrap.halt()
    print("PASS")


def test_llm_creature_wrapper():
    """Test the LLMCreature wrapper class."""
    print("  test_llm_creature_wrapper...", end=" ")

    # Import the wrapper
    from det_os_cli import LLMCreature

    config = BootConfig(total_F=100000.0)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    cid = bootstrap.spawn("llm_agent", initial_f=100.0, initial_a=0.8)
    bootstrap.runtime.creatures[cid].state = CreatureState.RUNNING

    # Create wrapper
    llm = LLMCreature(bootstrap.runtime, cid, "http://localhost:11434", "test")

    # Test properties
    assert llm.F == 100.0
    assert llm.a == 0.8
    assert llm.presence == 80.0  # F * a
    assert llm.is_alive

    # Test resource injection
    llm.inject_resource(50.0)
    assert llm.F == 150.0

    # Test state dict
    state = llm.get_state()
    assert state["cid"] == cid
    assert state["F"] == 150.0
    assert state["a"] == 0.8

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
    # 100 tokens would cost 5.0 F
    assert llm.can_think(100)  # 100 * 0.05 = 5.0 < 10.0

    # 300 tokens would cost 15.0 F
    assert not llm.can_think(300)  # 300 * 0.05 = 15.0 > 10.0

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

    print("\nCreature Tests:")
    test_llm_creature_spawn()
    test_creature_resource_depletion()
    test_creature_state_tracking()
    test_multiple_creatures()

    print("\nLLM Creature Wrapper Tests:")
    test_llm_creature_wrapper()
    test_can_think_check()

    print("\n" + "=" * 60)
    print("All DET-OS CLI tests passed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_tests()
