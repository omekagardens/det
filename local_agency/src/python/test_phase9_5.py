#!/usr/bin/env python3
"""
DET Phase 9.5 Tests: DET-OS Kernel in Existence-Lang
====================================================

Phase 9.5: The kernel itself written in Existence-Lang.
This is true DET-first principles: the OS is expressed in agency-first semantics.

Architecture:
    Existence-Lang Kernel (kernel.ex)
          │
          ▼ compiles to / interpreted by
    Architecture VM (EIS VM + DET Core)
          │
          ▼ future
    DET-Native Hardware
"""

import sys
import time
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from det.os.existence.bootstrap import DETOSBootstrap, BootState, BootConfig
from det.os.existence.runtime import (
    ExistenceKernelRuntime, RuntimeCreature, RuntimeChannel,
    RuntimeCapability, CreatureState
)


# ============================================================================
# Runtime Tests
# ============================================================================

def test_runtime_creation():
    """Test runtime initialization."""
    print("  test_runtime_creation...", end=" ")

    from det.os.existence.bootstrap import BootConfig
    config = BootConfig(total_F=100000.0, grace_pool=1000.0)
    runtime = ExistenceKernelRuntime(config=config)

    assert runtime.total_F == 100000.0
    assert runtime.grace_pool == 1000.0
    assert len(runtime.creatures) == 0

    print("PASS")


def test_runtime_boot_kernel():
    """Test kernel creature boot."""
    print("  test_runtime_boot_kernel...", end=" ")

    config = BootConfig(total_F=100000.0, grace_pool=1000.0, max_creatures=100)
    runtime = ExistenceKernelRuntime(config=config)

    runtime.boot_kernel(total_F=100000.0, grace_pool=1000.0, max_creatures=100)

    assert runtime.kernel is not None
    assert runtime.kernel.cid == 0
    assert runtime.kernel.a == 1.0  # Maximum agency
    assert runtime.kernel.state == CreatureState.RUNNING
    assert 0 in runtime.creatures

    print("PASS")


def test_runtime_spawn():
    """Test creature spawning through runtime."""
    print("  test_runtime_spawn...", end=" ")

    config = BootConfig(total_F=100000.0, grace_pool=1000.0)
    runtime = ExistenceKernelRuntime(config=config)
    runtime.boot_kernel(total_F=100000.0, grace_pool=1000.0, max_creatures=100)

    # Spawn creature
    cid = runtime.spawn("test_creature", initial_f=100.0, initial_a=0.6)

    assert cid == 1
    assert cid in runtime.creatures
    creature = runtime.creatures[cid]
    assert creature.name == "test_creature"
    assert creature.F == 100.0
    assert creature.a == 0.6

    print("PASS")


def test_runtime_spawn_conservation():
    """Test F conservation during spawn."""
    print("  test_runtime_spawn_conservation...", end=" ")

    config = BootConfig(total_F=100000.0)
    runtime = ExistenceKernelRuntime(config=config)
    runtime.boot_kernel(total_F=100000.0, grace_pool=1000.0, max_creatures=100)

    kernel_F_before = runtime.kernel.F

    # Spawn costs F
    cid = runtime.spawn("child", initial_f=50.0, initial_a=0.5)

    # Kernel F should decrease by spawn cost (F + 10% overhead)
    expected_cost = 50.0 * 1.1
    assert runtime.kernel.F == kernel_F_before - expected_cost

    print("PASS")


def test_runtime_spawn_agency_inheritance():
    """Test child agency cannot exceed parent."""
    print("  test_runtime_spawn_agency_inheritance...", end=" ")

    config = BootConfig(total_F=100000.0)
    runtime = ExistenceKernelRuntime(config=config)
    runtime.boot_kernel(total_F=100000.0, grace_pool=1000.0, max_creatures=100)

    # Spawn with agency higher than parent (kernel a=1.0)
    cid = runtime.spawn("child", initial_f=10.0, initial_a=1.5)

    # Child agency should be capped at parent's
    assert runtime.creatures[cid].a <= runtime.kernel.a

    print("PASS")


def test_runtime_kill():
    """Test creature killing."""
    print("  test_runtime_kill...", end=" ")

    config = BootConfig(total_F=100000.0)
    runtime = ExistenceKernelRuntime(config=config)
    runtime.boot_kernel(total_F=100000.0, grace_pool=1000.0, max_creatures=100)

    cid = runtime.spawn("to_die", initial_f=10.0)
    assert runtime.creatures[cid].state == CreatureState.EMBRYONIC

    runtime.kill(cid, "test kill")
    assert runtime.creatures[cid].state == CreatureState.DYING

    print("PASS")


def test_runtime_schedule():
    """Test presence-based scheduling."""
    print("  test_runtime_schedule...", end=" ")

    config = BootConfig(total_F=100000.0)
    runtime = ExistenceKernelRuntime(config=config)
    runtime.boot_kernel(total_F=100000.0, grace_pool=1000.0, max_creatures=100)

    # Spawn creatures with different presence
    c1 = runtime.spawn("low", initial_f=10.0, initial_a=0.5)
    c2 = runtime.spawn("high", initial_f=100.0, initial_a=0.9)

    # Activate them
    runtime.creatures[c1].state = CreatureState.RUNNING
    runtime.creatures[c2].state = CreatureState.RUNNING

    scheduled = runtime.schedule()

    # Highest presence should be scheduled (c2 or kernel)
    # Kernel has highest F, so it might win
    assert scheduled is not None

    print("PASS")


def test_runtime_ipc():
    """Test IPC through bond channels."""
    print("  test_runtime_ipc...", end=" ")

    config = BootConfig(total_F=100000.0)
    runtime = ExistenceKernelRuntime(config=config)
    runtime.boot_kernel(total_F=100000.0, grace_pool=1000.0, max_creatures=100)

    c1 = runtime.spawn("sender", initial_f=100.0)
    c2 = runtime.spawn("receiver", initial_f=100.0)
    runtime.creatures[c1].state = CreatureState.RUNNING
    runtime.creatures[c2].state = CreatureState.RUNNING

    # Create channel with high coherence for reliable delivery
    ch_id = runtime.create_channel(c1, c2, initial_coherence=1.0)

    # Send
    success = runtime.send(c1, ch_id, "hello")
    assert success

    # Receive
    msg = runtime.receive(c2, ch_id)
    assert msg == "hello"

    print("PASS")


def test_runtime_grace():
    """Test grace injection."""
    print("  test_runtime_grace...", end=" ")

    config = BootConfig(total_F=100000.0, grace_pool=1000.0)
    runtime = ExistenceKernelRuntime(config=config)
    runtime.boot_kernel(total_F=100000.0, grace_pool=1000.0, max_creatures=100)

    cid = runtime.spawn("needy", initial_f=0.05)
    runtime.creatures[cid].state = CreatureState.RUNNING

    # Creature needs grace (F < 0.1)
    amount = runtime.inject_grace(cid, 5.0)

    assert amount > 0
    assert runtime.creatures[cid].F > 0.05

    print("PASS")


def test_runtime_tick():
    """Test kernel tick execution."""
    print("  test_runtime_tick...", end=" ")

    config = BootConfig(total_F=100000.0, grace_pool=1000.0)
    runtime = ExistenceKernelRuntime(config=config)
    runtime.boot_kernel(total_F=100000.0, grace_pool=1000.0, max_creatures=100)

    cid = runtime.spawn("worker", initial_f=100.0)

    initial_tick = runtime.tick_count

    runtime.tick(0.02)

    assert runtime.tick_count == initial_tick + 1

    print("PASS")


def test_runtime_creature_death():
    """Test natural death when F depletes."""
    print("  test_runtime_creature_death...", end=" ")

    config = BootConfig(total_F=100000.0)
    runtime = ExistenceKernelRuntime(config=config)
    runtime.boot_kernel(total_F=100000.0, grace_pool=1000.0, max_creatures=100)

    cid = runtime.spawn("mortal", initial_f=0.001)
    runtime.creatures[cid].state = CreatureState.RUNNING
    runtime.creatures[cid].F = 0.0  # Deplete

    runtime.tick(0.02)

    # Creature either died and was reaped, or is still dying
    if cid in runtime.creatures:
        assert runtime.creatures[cid].state in (CreatureState.DYING, CreatureState.DEAD)
    # If not in creatures, it was successfully reaped after death

    print("PASS")


def test_runtime_capabilities():
    """Test capability-based access control."""
    print("  test_runtime_capabilities...", end=" ")

    config = BootConfig(total_F=100000.0)
    runtime = ExistenceKernelRuntime(config=config)
    runtime.boot_kernel(total_F=100000.0, grace_pool=1000.0, max_creatures=100)

    # Spawn with moderate agency
    cid = runtime.spawn("user", initial_f=10.0, initial_a=0.6)

    # Should have read, write, execute, spawn, channel caps
    assert runtime.check_access(cid, "read", "/file")
    assert runtime.check_access(cid, "write", "/file")
    assert runtime.check_access(cid, "execute", "/program")

    print("PASS")


# ============================================================================
# Bootstrap Tests
# ============================================================================

def test_bootstrap_creation():
    """Test bootstrap initialization."""
    print("  test_bootstrap_creation...", end=" ")

    bootstrap = DETOSBootstrap()

    assert bootstrap.state == BootState.INIT
    assert bootstrap.config is not None

    print("PASS")


def test_bootstrap_config():
    """Test bootstrap configuration."""
    print("  test_bootstrap_config...", end=" ")

    config = BootConfig(
        total_F=50000.0,
        grace_pool=500.0,
        tick_rate=100.0,
        debug=True
    )
    bootstrap = DETOSBootstrap(config=config)

    assert bootstrap.config.total_F == 50000.0
    assert bootstrap.config.grace_pool == 500.0
    assert bootstrap.config.tick_rate == 100.0

    print("PASS")


def test_bootstrap_boot():
    """Test full boot sequence."""
    print("  test_bootstrap_boot...", end=" ")

    config = BootConfig(total_F=100000.0, grace_pool=1000.0, debug=False)
    bootstrap = DETOSBootstrap(config=config)

    success = bootstrap.boot()

    assert success
    assert bootstrap.state == BootState.RUNNING
    assert bootstrap.runtime is not None

    print("PASS")


def test_bootstrap_spawn():
    """Test spawning through bootstrap."""
    print("  test_bootstrap_spawn...", end=" ")

    config = BootConfig(total_F=100000.0, debug=False)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    cid = bootstrap.spawn("test_app", initial_f=10.0, initial_a=0.5)

    assert cid > 0
    assert cid in bootstrap.runtime.creatures

    print("PASS")


def test_bootstrap_tick():
    """Test tick through bootstrap."""
    print("  test_bootstrap_tick...", end=" ")

    config = BootConfig(total_F=100000.0, debug=False)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    bootstrap.spawn("worker", initial_f=100.0)

    state = bootstrap.tick()

    assert state.tick >= 1

    print("PASS")


def test_bootstrap_halt():
    """Test kernel halt."""
    print("  test_bootstrap_halt...", end=" ")

    config = BootConfig(total_F=100000.0, debug=False)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    bootstrap.spawn("worker1", initial_f=10.0)
    bootstrap.spawn("worker2", initial_f=10.0)

    bootstrap.halt()

    assert bootstrap.state == BootState.HALTED

    print("PASS")


def test_bootstrap_stats():
    """Test kernel statistics."""
    print("  test_bootstrap_stats...", end=" ")

    config = BootConfig(total_F=100000.0, debug=False)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    bootstrap.spawn("worker", initial_f=10.0)
    bootstrap.tick()

    stats = bootstrap.get_stats()

    assert "state" in stats
    assert "tick" in stats
    assert "num_creatures" in stats
    assert stats["state"] == "RUNNING"

    print("PASS")


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_lifecycle():
    """Test full creature lifecycle through Existence-Lang kernel."""
    print("  test_full_lifecycle...", end=" ")

    config = BootConfig(total_F=100000.0, grace_pool=1000.0, debug=False)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    # Spawn creature
    cid = bootstrap.spawn("lifecycle_test", initial_f=1.0)

    # Run some ticks
    for _ in range(5):
        bootstrap.tick()

    # Creature should still be alive
    creature = bootstrap.runtime.creatures.get(cid)
    assert creature is not None
    assert creature.is_alive()

    # Deplete resources
    creature.F = 0.0

    # Tick should trigger death
    bootstrap.tick()

    assert creature.state in (CreatureState.DYING, CreatureState.DEAD)

    print("PASS")


def test_parent_child_spawn():
    """Test parent spawning child (F conservation)."""
    print("  test_parent_child_spawn...", end=" ")

    config = BootConfig(total_F=100000.0, debug=False)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    # Spawn parent
    parent_cid = bootstrap.spawn("parent", initial_f=100.0, initial_a=0.8)
    parent = bootstrap.runtime.creatures[parent_cid]
    parent.state = CreatureState.RUNNING

    parent_F_before = parent.F

    # Parent spawns child (through runtime directly for now)
    child_cid = bootstrap.runtime.spawn("child", initial_f=20.0, initial_a=0.5,
                                        parent_cid=parent_cid)

    # Parent F should decrease
    assert parent.F < parent_F_before

    # Child exists
    child = bootstrap.runtime.creatures.get(child_cid)
    assert child is not None
    assert child.parent_cid == parent_cid

    print("PASS")


def test_ipc_between_creatures():
    """Test IPC between user creatures."""
    print("  test_ipc_between_creatures...", end=" ")

    config = BootConfig(total_F=100000.0, debug=False)
    bootstrap = DETOSBootstrap(config=config)
    bootstrap.boot()

    c1 = bootstrap.spawn("sender", initial_f=100.0)
    c2 = bootstrap.spawn("receiver", initial_f=100.0)

    bootstrap.runtime.creatures[c1].state = CreatureState.RUNNING
    bootstrap.runtime.creatures[c2].state = CreatureState.RUNNING

    # Create bond channel
    ch_id = bootstrap.runtime.create_channel(c1, c2, initial_coherence=1.0)

    # Send message
    bootstrap.runtime.send(c1, ch_id, {"type": "request", "data": 42})

    # Receive
    msg = bootstrap.runtime.receive(c2, ch_id)
    assert msg is not None
    assert msg["data"] == 42

    print("PASS")


def test_kernel_source_exists():
    """Test that kernel.ex source file exists."""
    print("  test_kernel_source_exists...", end=" ")

    kernel_path = Path(__file__).parent / "det" / "os" / "existence" / "kernel.ex"
    assert kernel_path.exists(), f"kernel.ex not found at {kernel_path}"

    with open(kernel_path, 'r') as f:
        source = f.read()

    # Check for key constructs
    assert "creature KernelCreature" in source
    assert "kernel Schedule" in source
    assert "kernel Allocate" in source
    assert "kernel Gate" in source
    assert "kernel Grace" in source
    assert "presence DET_OS" in source

    print("PASS")


# ============================================================================
# Main
# ============================================================================

def run_tests():
    """Run all Phase 9.5 tests."""
    print("\n" + "=" * 60)
    print("DET Phase 9.5 Tests: DET-OS Kernel in Existence-Lang")
    print("=" * 60)

    # Runtime tests
    print("\nRuntime Tests:")
    test_runtime_creation()
    test_runtime_boot_kernel()
    test_runtime_spawn()
    test_runtime_spawn_conservation()
    test_runtime_spawn_agency_inheritance()
    test_runtime_kill()
    test_runtime_schedule()
    test_runtime_ipc()
    test_runtime_grace()
    test_runtime_tick()
    test_runtime_creature_death()
    test_runtime_capabilities()

    # Bootstrap tests
    print("\nBootstrap Tests:")
    test_bootstrap_creation()
    test_bootstrap_config()
    test_bootstrap_boot()
    test_bootstrap_spawn()
    test_bootstrap_tick()
    test_bootstrap_halt()
    test_bootstrap_stats()

    # Integration tests
    print("\nIntegration Tests:")
    test_full_lifecycle()
    test_parent_child_spawn()
    test_ipc_between_creatures()
    test_kernel_source_exists()

    print("\n" + "=" * 60)
    print("All Phase 9.5 tests passed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_tests()
