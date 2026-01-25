#!/usr/bin/env python3
"""
DET Phase 9 Tests: DET-OS Kernel
================================

Phase 9: Native DET-OS kernel with agency-first design:
  - Creatures (processes)
  - Presence-based scheduling
  - Conservation-based allocation
  - Bond-based IPC
  - Agency-based security (Gatekeeper)
  - Kernel syscalls
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from det.os import (
    # Creature
    Creature, CreatureState, CreatureFlags, CreatureTable, CreatureHandle,
    # Scheduler
    PresenceScheduler, SchedulerPolicy, ScheduleResult, ScheduleStats,
    # Allocator
    ResourceAllocator, MemoryBlock, AllocationFlags, AllocationResult, ResourcePool,
    # IPC
    BondChannel, Message, MessageQueue, IPCResult, ChannelFlags, IPCManager,
    # Gatekeeper
    Gatekeeper, Permission, PermissionLevel, AccessResult, Capability,
    # Kernel
    DETOSKernel, KernelState, KernelConfig, SyscallResult, Syscall,
)


# ============================================================================
# Creature Tests
# ============================================================================

def test_creature_handle():
    """Test CreatureHandle hashing and equality."""
    print("  test_creature_handle...", end=" ")

    h1 = CreatureHandle(cid=1, generation=1)
    h2 = CreatureHandle(cid=1, generation=1)
    h3 = CreatureHandle(cid=1, generation=2)
    h4 = CreatureHandle(cid=2, generation=1)

    assert h1 == h2
    assert h1 != h3
    assert h1 != h4
    assert hash(h1) == hash(h2)

    print("PASS")


def test_creature_creation():
    """Test Creature dataclass creation."""
    print("  test_creature_creation...", end=" ")

    handle = CreatureHandle(cid=0, generation=1)
    creature = Creature(handle=handle, name="test", F=10.0, a=0.8)

    assert creature.F == 10.0
    assert creature.a == 0.8
    assert creature.q == 0.0
    assert creature.state == CreatureState.EMBRYONIC
    assert creature.is_alive()

    print("PASS")


def test_creature_presence():
    """Test presence computation P = F * C_self * a."""
    print("  test_creature_presence...", end=" ")

    handle = CreatureHandle(cid=0, generation=1)
    creature = Creature(handle=handle, name="test", F=10.0, a=0.5, C_self=0.8)

    P = creature.compute_presence()
    expected = 10.0 * 0.8 * 0.5  # 4.0

    assert abs(P - expected) < 0.001
    assert abs(creature.P - expected) < 0.001

    print("PASS")


def test_creature_resource_consumption():
    """Test resource consumption."""
    print("  test_creature_resource_consumption...", end=" ")

    handle = CreatureHandle(cid=0, generation=1)
    creature = Creature(handle=handle, name="test", F=5.0)

    # Normal consumption
    consumed = creature.consume_resource(2.0)
    assert consumed == 2.0
    assert creature.F == 3.0

    # Consumption exceeding available
    consumed = creature.consume_resource(10.0)
    assert consumed == 3.0
    assert creature.F == 0.0

    print("PASS")


def test_creature_grace_injection():
    """Test grace injection."""
    print("  test_creature_grace_injection...", end=" ")

    handle = CreatureHandle(cid=0, generation=1)
    creature = Creature(handle=handle, name="test", F=0.05, q=1.0)

    assert creature.needs_grace()

    creature.inject_grace(5.0)
    assert creature.grace_buffer == 5.0

    creature.process_grace()
    assert creature.F == 5.05
    assert creature.q < 1.0  # Grace reduces debt
    assert creature.grace_buffer == 0

    print("PASS")


def test_creature_table_spawn():
    """Test CreatureTable spawning."""
    print("  test_creature_table_spawn...", end=" ")

    table = CreatureTable(max_creatures=100)

    c1 = table.spawn("creature1", initial_f=10.0, initial_a=0.6)
    c2 = table.spawn("creature2", initial_f=5.0)

    assert c1.name == "creature1"
    assert c1.F == 10.0
    assert c1.a == 0.6
    assert c2.name == "creature2"

    assert table.num_total() == 2

    print("PASS")


def test_creature_table_get_by_handle():
    """Test getting creature by handle with generation check."""
    print("  test_creature_table_get_by_handle...", end=" ")

    table = CreatureTable()

    c1 = table.spawn("test")
    handle = c1.handle

    # Valid lookup
    found = table.get(handle)
    assert found is c1

    # Invalid generation
    bad_handle = CreatureHandle(cid=handle.cid, generation=handle.generation + 1)
    found = table.get(bad_handle)
    assert found is None

    print("PASS")


def test_creature_table_kill_and_reap():
    """Test creature death and reaping."""
    print("  test_creature_table_kill_and_reap...", end=" ")

    table = CreatureTable()

    c1 = table.spawn("test", initial_f=1.0)
    handle = c1.handle
    c1.state = CreatureState.RUNNING

    # Kill
    table.kill(handle, reason="test kill")
    assert c1.state == CreatureState.DYING
    assert c1.death_reason == "test kill"

    # Transition to dead
    c1.state = CreatureState.DEAD

    # Reap
    table.reap(handle)
    assert table.get(handle) is None

    print("PASS")


def test_creature_table_parent_child():
    """Test parent-child relationships."""
    print("  test_creature_table_parent_child...", end=" ")

    table = CreatureTable()

    parent = table.spawn("parent")
    child = table.spawn("child", parent=parent.handle)

    assert child.parent == parent.handle
    assert child.handle in parent.children

    print("PASS")


# ============================================================================
# Scheduler Tests
# ============================================================================

def test_scheduler_creation():
    """Test scheduler creation."""
    print("  test_scheduler_creation...", end=" ")

    table = CreatureTable()
    scheduler = PresenceScheduler(table)

    assert scheduler.creatures is table
    assert scheduler.policy == SchedulerPolicy.PRESENCE

    print("PASS")


def test_scheduler_no_schedulable():
    """Test scheduling with no creatures."""
    print("  test_scheduler_no_schedulable...", end=" ")

    table = CreatureTable()
    scheduler = PresenceScheduler(table)

    result = scheduler.schedule()

    assert result.creature is None
    assert "No schedulable" in result.reason

    print("PASS")


def test_scheduler_presence_based():
    """Test presence-based scheduling."""
    print("  test_scheduler_presence_based...", end=" ")

    table = CreatureTable()

    # Create creatures with different presence
    c1 = table.spawn("low", initial_f=1.0, initial_a=0.5)
    c2 = table.spawn("high", initial_f=10.0, initial_a=0.9)
    c1.state = CreatureState.RUNNING
    c2.state = CreatureState.RUNNING
    c1.compute_presence()
    c2.compute_presence()

    scheduler = PresenceScheduler(table)
    result = scheduler.schedule()

    # Higher presence creature should be scheduled
    assert result.creature is c2
    assert result.time_slice > 0

    print("PASS")


def test_scheduler_round_robin():
    """Test round-robin scheduling."""
    print("  test_scheduler_round_robin...", end=" ")

    table = CreatureTable()

    c1 = table.spawn("c1", initial_f=5.0)
    c2 = table.spawn("c2", initial_f=5.0)
    c1.state = CreatureState.RUNNING
    c2.state = CreatureState.RUNNING

    scheduler = PresenceScheduler(table, policy=SchedulerPolicy.ROUND_ROBIN)

    # First schedule
    r1 = scheduler.schedule()
    # Second schedule should pick different creature
    r2 = scheduler.schedule()

    assert r1.creature != r2.creature or table.num_alive() == 1

    print("PASS")


def test_scheduler_effective_presence_boosts():
    """Test presence boosts for special creatures."""
    print("  test_scheduler_effective_presence_boosts...", end=" ")

    table = CreatureTable()

    c1 = table.spawn("normal", initial_f=5.0, initial_a=0.5)
    c2 = table.spawn("kernel", initial_f=5.0, initial_a=0.5,
                     flags=CreatureFlags.KERNEL)
    c1.state = CreatureState.RUNNING
    c2.state = CreatureState.RUNNING
    c1.compute_presence()
    c2.compute_presence()

    scheduler = PresenceScheduler(table)

    # Kernel creature should have higher effective presence
    ep_normal = scheduler.compute_effective_presence(c1)
    ep_kernel = scheduler.compute_effective_presence(c2)

    assert ep_kernel > ep_normal

    print("PASS")


# ============================================================================
# Allocator Tests
# ============================================================================

def test_allocator_creation():
    """Test allocator creation."""
    print("  test_allocator_creation...", end=" ")

    allocator = ResourceAllocator(
        total_memory=1024 * 1024,
        total_F=100.0
    )

    assert allocator.pool.total_memory == 1024 * 1024
    assert allocator.pool.total_F == 100.0
    assert allocator.pool.available_F == 100.0

    print("PASS")


def test_allocator_allocate():
    """Test memory allocation."""
    print("  test_allocator_allocate...", end=" ")

    allocator = ResourceAllocator(total_F=100.0)

    result = allocator.allocate(
        creature_id=1,
        size=4096,
        creature_F=10.0
    )

    assert result.success
    assert result.block is not None
    assert result.block.size == 4096
    assert result.block.owner == 1
    assert result.F_consumed > 0

    print("PASS")


def test_allocator_insufficient_resource():
    """Test allocation with insufficient F."""
    print("  test_allocator_insufficient_resource...", end=" ")

    allocator = ResourceAllocator(total_F=100.0, F_per_page=1.0)

    result = allocator.allocate(
        creature_id=1,
        size=4096,
        creature_F=0.001  # Very low F
    )

    assert not result.success
    assert "Insufficient F" in result.reason

    print("PASS")


def test_allocator_free():
    """Test memory deallocation."""
    print("  test_allocator_free...", end=" ")

    allocator = ResourceAllocator(total_F=100.0)

    result = allocator.allocate(creature_id=1, size=4096, creature_F=10.0)
    block_id = result.block.block_id
    available_before = allocator.pool.available_F

    F_returned = allocator.free(block_id)

    assert F_returned > 0
    assert allocator.pool.available_F > available_before
    assert block_id not in allocator.blocks

    print("PASS")


def test_allocator_free_all_for_creature():
    """Test freeing all blocks for a creature."""
    print("  test_allocator_free_all_for_creature...", end=" ")

    allocator = ResourceAllocator(total_F=100.0)

    # Allocate multiple blocks
    allocator.allocate(creature_id=1, size=4096, creature_F=10.0)
    allocator.allocate(creature_id=1, size=8192, creature_F=10.0)
    allocator.allocate(creature_id=2, size=4096, creature_F=10.0)

    # Free all for creature 1
    F_returned = allocator.free_all_for_creature(1)

    assert F_returned > 0
    assert 1 not in allocator.owner_blocks or len(allocator.owner_blocks[1]) == 0
    assert 2 in allocator.owner_blocks and len(allocator.owner_blocks[2]) == 1

    print("PASS")


def test_allocator_share_block():
    """Test block sharing."""
    print("  test_allocator_share_block...", end=" ")

    allocator = ResourceAllocator(total_F=100.0)

    result = allocator.allocate(
        creature_id=1,
        size=4096,
        creature_F=10.0,
        flags=AllocationFlags.SHARED
    )

    success = allocator.share_block(result.block.block_id, with_creature=2)

    assert success
    assert 2 in result.block.shared_with

    print("PASS")


def test_allocator_defragment():
    """Test memory defragmentation."""
    print("  test_allocator_defragment...", end=" ")

    allocator = ResourceAllocator(total_F=100.0)

    # Allocate and free to create fragmentation
    r1 = allocator.allocate(creature_id=1, size=4096, creature_F=10.0)
    r2 = allocator.allocate(creature_id=1, size=4096, creature_F=10.0)
    r3 = allocator.allocate(creature_id=1, size=4096, creature_F=10.0)

    allocator.free(r1.block.block_id)
    allocator.free(r3.block.block_id)

    frag_before = len(allocator.free_list)
    debt_reduced = allocator.defragment()
    frag_after = len(allocator.free_list)

    # Defragmentation should merge adjacent free blocks
    assert frag_after <= frag_before

    print("PASS")


# ============================================================================
# IPC Tests
# ============================================================================

def test_message_queue():
    """Test MessageQueue operations."""
    print("  test_message_queue...", end=" ")

    queue = MessageQueue(capacity=4)

    msg = Message(msg_id=0, sender=1, receiver=2, payload="hello", F_cost=0.01)

    result = queue.push(msg)
    assert result == IPCResult.SUCCESS
    assert queue.size() == 1

    result, received = queue.pop()
    assert result == IPCResult.SUCCESS
    assert received.payload == "hello"
    assert queue.is_empty()

    print("PASS")


def test_message_queue_full():
    """Test MessageQueue capacity limit."""
    print("  test_message_queue_full...", end=" ")

    queue = MessageQueue(capacity=2)

    for i in range(2):
        msg = Message(msg_id=i, sender=1, receiver=2, payload=i, F_cost=0.01)
        result = queue.push(msg)
        assert result == IPCResult.SUCCESS

    # Queue is full
    msg = Message(msg_id=99, sender=1, receiver=2, payload="overflow", F_cost=0.01)
    result = queue.push(msg)
    assert result == IPCResult.QUEUE_FULL

    print("PASS")


def test_bond_channel_creation():
    """Test BondChannel creation."""
    print("  test_bond_channel_creation...", end=" ")

    channel = BondChannel(
        channel_id=0,
        creature_a=1,
        creature_b=2,
        coherence=0.8
    )

    assert channel.creature_a == 1
    assert channel.creature_b == 2
    assert channel.coherence == 0.8
    assert not channel.closed

    print("PASS")


def test_bond_channel_send_receive():
    """Test sending and receiving through bond channel."""
    print("  test_bond_channel_send_receive...", end=" ")

    channel = BondChannel(
        channel_id=0,
        creature_a=1,
        creature_b=2,
        coherence=1.0,  # Perfect coherence for reliable test
        flags=ChannelFlags.RELIABLE
    )

    # Send from creature 1 to creature 2
    result, F_cost = channel.send(1, "test message", F_available=10.0)
    assert result == IPCResult.SUCCESS
    assert F_cost > 0

    # Receive at creature 2
    result, msg = channel.receive(2)
    assert result == IPCResult.SUCCESS
    assert msg.payload == "test message"

    print("PASS")


def test_bond_channel_coherence_affects_reliability():
    """Test that low coherence can cause message loss."""
    print("  test_bond_channel_coherence_affects_reliability...", end=" ")

    channel = BondChannel(
        channel_id=0,
        creature_a=1,
        creature_b=2,
        coherence=0.05,  # Very low coherence
        flags=ChannelFlags.NONE  # Not reliable
    )

    # With very low coherence, channel should reject (below threshold)
    result, _ = channel.send(1, "test", F_available=10.0)
    assert result == IPCResult.NO_COHERENCE

    print("PASS")


def test_ipc_manager():
    """Test IPCManager channel management."""
    print("  test_ipc_manager...", end=" ")

    manager = IPCManager()

    c1 = manager.create_channel(1, 2)
    c2 = manager.create_channel(1, 3)

    assert len(manager.channels) == 2
    assert c1.channel_id != c2.channel_id

    # Find channel
    found = manager.find_channel(1, 2)
    assert found is c1

    # Close channel
    manager.close_channel(c1.channel_id)
    assert c1.channel_id not in manager.channels

    print("PASS")


def test_ipc_manager_close_all_for_creature():
    """Test closing all channels for a creature."""
    print("  test_ipc_manager_close_all_for_creature...", end=" ")

    manager = IPCManager()

    manager.create_channel(1, 2)
    manager.create_channel(1, 3)
    manager.create_channel(2, 3)

    # Close all for creature 1
    manager.close_all_for_creature(1)

    # Only channel 2-3 should remain
    assert len(manager.channels) == 1
    assert manager.find_channel(2, 3) is not None

    print("PASS")


# ============================================================================
# Gatekeeper Tests
# ============================================================================

def test_gatekeeper_create_capability():
    """Test capability creation."""
    print("  test_gatekeeper_create_capability...", end=" ")

    gk = Gatekeeper()

    perm = Permission(action="read", target="/home/*", level=PermissionLevel.READ)
    cap = gk.create_capability(
        name="home_read",
        owner=1,
        permissions=[perm]
    )

    assert cap.name == "home_read"
    assert cap.owner == 1
    assert len(cap.permissions) == 1
    assert cap.cap_id in gk.capabilities

    print("PASS")


def test_gatekeeper_check_access_granted():
    """Test access check that should be granted."""
    print("  test_gatekeeper_check_access_granted...", end=" ")

    gk = Gatekeeper()

    perm = Permission(action="read", target="*", level=PermissionLevel.READ)
    gk.create_capability(name="read_all", owner=1, permissions=[perm])

    result, reason = gk.check_access(
        creature_id=1,
        creature_agency=0.5,  # Above READ threshold (0.1)
        creature_coherence=0.5,
        creature_F=1.0,
        action="read",
        target="/some/file"
    )

    assert result == AccessResult.GRANTED

    print("PASS")


def test_gatekeeper_check_access_denied_no_capability():
    """Test access denied for missing capability."""
    print("  test_gatekeeper_check_access_denied_no_capability...", end=" ")

    gk = Gatekeeper()

    result, reason = gk.check_access(
        creature_id=99,
        creature_agency=1.0,
        creature_coherence=1.0,
        creature_F=100.0,
        action="write",
        target="/secret"
    )

    assert result == AccessResult.DENIED_NO_CAPABILITY

    print("PASS")


def test_gatekeeper_check_access_denied_insufficient_agency():
    """Test access denied for insufficient agency."""
    print("  test_gatekeeper_check_access_denied_insufficient_agency...", end=" ")

    gk = Gatekeeper()

    # Create admin permission requiring high agency
    perm = Permission(action="admin", target="*", level=PermissionLevel.ADMIN)
    gk.create_capability(name="admin", owner=1, permissions=[perm])

    result, reason = gk.check_access(
        creature_id=1,
        creature_agency=0.2,  # Below ADMIN threshold (0.8)
        creature_coherence=0.5,
        creature_F=10.0,
        action="admin",
        target="/system"
    )

    assert result == AccessResult.DENIED_NO_AGENCY

    print("PASS")


def test_gatekeeper_delegate_capability():
    """Test capability delegation."""
    print("  test_gatekeeper_delegate_capability...", end=" ")

    gk = Gatekeeper()

    perm = Permission(action="read", target="*", level=PermissionLevel.READ)
    cap = gk.create_capability(
        name="read_all",
        owner=1,
        permissions=[perm],
        transferable=True
    )

    # Delegate to creature 2
    delegated = gk.delegate_capability(cap.cap_id, from_creature=1, to_creature=2)

    assert delegated is not None
    assert delegated.owner == 2
    assert delegated.delegation_depth == 1
    assert 2 in gk.creature_caps

    print("PASS")


def test_gatekeeper_revoke_all():
    """Test revoking all capabilities for a creature."""
    print("  test_gatekeeper_revoke_all...", end=" ")

    gk = Gatekeeper()

    perm = Permission(action="read", target="*", level=PermissionLevel.READ)
    gk.create_capability(name="cap1", owner=1, permissions=[perm])
    gk.create_capability(name="cap2", owner=1, permissions=[perm])

    gk.revoke_all_for_creature(1)

    assert 1 not in gk.creature_caps or len(gk.creature_caps[1]) == 0

    print("PASS")


def test_gatekeeper_create_standard_capabilities():
    """Test creating standard capabilities for a permission level."""
    print("  test_gatekeeper_create_standard_capabilities...", end=" ")

    gk = Gatekeeper()

    gk.create_standard_capabilities(1, PermissionLevel.EXECUTE)

    # Should have read, write, execute capabilities
    assert 1 in gk.creature_caps
    assert len(gk.creature_caps[1]) >= 3

    print("PASS")


# ============================================================================
# Kernel Tests
# ============================================================================

def test_kernel_creation():
    """Test kernel creation and initialization."""
    print("  test_kernel_creation...", end=" ")

    kernel = DETOSKernel()

    assert kernel.state == KernelState.RUNNING
    assert kernel.tick == 0
    assert kernel.creatures.num_alive() >= 1  # Kernel creature

    print("PASS")


def test_kernel_spawn_creature():
    """Test spawning creatures through kernel."""
    print("  test_kernel_spawn_creature...", end=" ")

    kernel = DETOSKernel()

    creature = kernel.spawn("test_app", initial_f=10.0, initial_a=0.6)

    assert creature.name == "test_app"
    assert creature.F == 10.0
    assert creature.a == 0.6
    assert creature.parent == kernel.kernel_handle

    print("PASS")


def test_kernel_kill_creature():
    """Test killing creatures through kernel."""
    print("  test_kernel_kill_creature...", end=" ")

    kernel = DETOSKernel()

    creature = kernel.spawn("to_die")
    handle = creature.handle
    creature.state = CreatureState.RUNNING

    kernel.kill(handle, "test kill")

    assert creature.state == CreatureState.DYING

    print("PASS")


def test_kernel_syscall_spawn():
    """Test SPAWN syscall."""
    print("  test_kernel_syscall_spawn...", end=" ")

    kernel = DETOSKernel()

    parent = kernel.spawn("parent", initial_f=100.0)
    parent.state = CreatureState.RUNNING

    result = kernel.syscall(parent.handle, Syscall.SPAWN, "child", 5.0, 0.5)

    assert result.success
    assert result.value is not None  # Child handle

    child = kernel.get_creature(result.value)
    assert child is not None
    assert child.name == "child"

    print("PASS")


def test_kernel_syscall_alloc_free():
    """Test ALLOC and FREE syscalls."""
    print("  test_kernel_syscall_alloc_free...", end=" ")

    kernel = DETOSKernel()

    creature = kernel.spawn("allocator", initial_f=100.0)
    creature.state = CreatureState.RUNNING

    # Allocate
    result = kernel.syscall(creature.handle, Syscall.ALLOC, 4096)
    assert result.success
    block = result.value
    assert block is not None

    # Free
    result = kernel.syscall(creature.handle, Syscall.FREE, block.block_id)
    assert result.success

    print("PASS")


def test_kernel_syscall_channel_send_recv():
    """Test IPC syscalls."""
    print("  test_kernel_syscall_channel_send_recv...", end=" ")

    kernel = DETOSKernel()

    c1 = kernel.spawn("sender", initial_f=100.0)
    c2 = kernel.spawn("receiver", initial_f=100.0)
    c1.state = CreatureState.RUNNING
    c2.state = CreatureState.RUNNING

    # Create channel
    result = kernel.syscall(c1.handle, Syscall.CHANNEL_CREATE, c2.handle.cid)
    assert result.success
    channel = result.value

    # Send
    result = kernel.syscall(c1.handle, Syscall.SEND, channel.channel_id, "hello")
    assert result.success

    # Receive
    result = kernel.syscall(c2.handle, Syscall.RECV, channel.channel_id)
    assert result.success
    assert result.value.payload == "hello"

    print("PASS")


def test_kernel_syscall_getinfo():
    """Test GETINFO syscall."""
    print("  test_kernel_syscall_getinfo...", end=" ")

    kernel = DETOSKernel()

    creature = kernel.spawn("info_getter", initial_f=10.0)
    creature.state = CreatureState.RUNNING

    # Get tick
    result = kernel.syscall(creature.handle, Syscall.GETINFO, "tick")
    assert result.success
    assert result.value == 0

    # Get self info
    result = kernel.syscall(creature.handle, Syscall.GETINFO, "self")
    assert result.success
    assert "F" in result.value
    assert "a" in result.value

    print("PASS")


def test_kernel_tick():
    """Test kernel tick execution."""
    print("  test_kernel_tick...", end=" ")

    kernel = DETOSKernel()

    c1 = kernel.spawn("worker1", initial_f=10.0)
    c2 = kernel.spawn("worker2", initial_f=5.0)
    c1.state = CreatureState.RUNNING
    c2.state = CreatureState.RUNNING

    initial_tick = kernel.tick

    result = kernel.kernel_tick()

    assert kernel.tick == initial_tick + 1
    assert result.creature is not None  # Something was scheduled

    print("PASS")


def test_kernel_run_limited():
    """Test running kernel for limited ticks."""
    print("  test_kernel_run_limited...", end=" ")

    kernel = DETOSKernel()

    kernel.spawn("worker", initial_f=100.0)

    kernel.run(max_ticks=5)

    assert kernel.tick >= 5

    print("PASS")


def test_kernel_shutdown():
    """Test kernel shutdown."""
    print("  test_kernel_shutdown...", end=" ")

    kernel = DETOSKernel()

    kernel.spawn("worker1", initial_f=10.0)
    kernel.spawn("worker2", initial_f=10.0)

    kernel.shutdown()

    assert kernel.state == KernelState.HALTED
    # Only kernel creature should remain
    assert kernel.creatures.num_alive() == 1

    print("PASS")


def test_kernel_stats():
    """Test kernel statistics."""
    print("  test_kernel_stats...", end=" ")

    kernel = DETOSKernel()

    kernel.spawn("worker", initial_f=10.0)
    kernel.kernel_tick()

    stats = kernel.get_stats()

    assert "tick" in stats
    assert "uptime" in stats
    assert "creatures" in stats
    assert "scheduler" in stats
    assert "allocator" in stats
    assert "ipc" in stats

    print("PASS")


def test_kernel_grace_request():
    """Test grace request syscall."""
    print("  test_kernel_grace_request...", end=" ")

    kernel = DETOSKernel()

    creature = kernel.spawn("needy", initial_f=0.05)  # Low F
    creature.state = CreatureState.RUNNING

    result = kernel.syscall(creature.handle, Syscall.GRACE_REQUEST, 5.0)

    assert result.success
    assert result.value > 0  # Some grace was granted

    print("PASS")


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_lifecycle():
    """Test full creature lifecycle through kernel."""
    print("  test_full_lifecycle...", end=" ")

    kernel = DETOSKernel()

    # Spawn creature
    creature = kernel.spawn("lifecycle_test", initial_f=1.0)
    creature.state = CreatureState.RUNNING

    # Run some ticks - creature consumes resources
    for _ in range(10):
        kernel.kernel_tick()

    # Creature should still be alive (has resources)
    assert creature.is_alive()

    # Deplete resources manually
    creature.F = 0.0

    # Tick should trigger death
    kernel.kernel_tick()

    # Should be dying or dead
    assert creature.state in (CreatureState.DYING, CreatureState.DEAD)

    print("PASS")


def test_ipc_with_gatekeeper():
    """Test IPC access controlled by gatekeeper."""
    print("  test_ipc_with_gatekeeper...", end=" ")

    kernel = DETOSKernel()

    # Create two creatures
    c1 = kernel.spawn("sender", initial_f=100.0, initial_a=0.6)
    c2 = kernel.spawn("receiver", initial_f=100.0, initial_a=0.6)
    c1.state = CreatureState.RUNNING
    c2.state = CreatureState.RUNNING

    # Create channel - requires permission
    result = kernel.syscall(c1.handle, Syscall.CHANNEL_CREATE, c2.handle.cid)

    # Should succeed because creature has execute-level permissions
    assert result.success

    print("PASS")


def test_parent_child_spawn():
    """Test parent spawning child with resource inheritance."""
    print("  test_parent_child_spawn...", end=" ")

    kernel = DETOSKernel()

    parent = kernel.spawn("parent", initial_f=100.0, initial_a=0.8)
    parent.state = CreatureState.RUNNING

    # Parent spawns child
    result = kernel.syscall(parent.handle, Syscall.SPAWN, "child", 20.0, 0.5)
    assert result.success

    child = kernel.get_creature(result.value)
    assert child is not None

    # Parent's F should be reduced
    assert parent.F < 100.0

    # Child's agency should not exceed parent's
    assert child.a <= parent.a

    print("PASS")


# ============================================================================
# Main
# ============================================================================

def run_tests():
    """Run all Phase 9 tests."""
    print("\n" + "=" * 60)
    print("DET Phase 9 Tests: DET-OS Kernel")
    print("=" * 60)

    # Creature tests
    print("\nCreature Tests:")
    test_creature_handle()
    test_creature_creation()
    test_creature_presence()
    test_creature_resource_consumption()
    test_creature_grace_injection()
    test_creature_table_spawn()
    test_creature_table_get_by_handle()
    test_creature_table_kill_and_reap()
    test_creature_table_parent_child()

    # Scheduler tests
    print("\nScheduler Tests:")
    test_scheduler_creation()
    test_scheduler_no_schedulable()
    test_scheduler_presence_based()
    test_scheduler_round_robin()
    test_scheduler_effective_presence_boosts()

    # Allocator tests
    print("\nAllocator Tests:")
    test_allocator_creation()
    test_allocator_allocate()
    test_allocator_insufficient_resource()
    test_allocator_free()
    test_allocator_free_all_for_creature()
    test_allocator_share_block()
    test_allocator_defragment()

    # IPC tests
    print("\nIPC Tests:")
    test_message_queue()
    test_message_queue_full()
    test_bond_channel_creation()
    test_bond_channel_send_receive()
    test_bond_channel_coherence_affects_reliability()
    test_ipc_manager()
    test_ipc_manager_close_all_for_creature()

    # Gatekeeper tests
    print("\nGatekeeper Tests:")
    test_gatekeeper_create_capability()
    test_gatekeeper_check_access_granted()
    test_gatekeeper_check_access_denied_no_capability()
    test_gatekeeper_check_access_denied_insufficient_agency()
    test_gatekeeper_delegate_capability()
    test_gatekeeper_revoke_all()
    test_gatekeeper_create_standard_capabilities()

    # Kernel tests
    print("\nKernel Tests:")
    test_kernel_creation()
    test_kernel_spawn_creature()
    test_kernel_kill_creature()
    test_kernel_syscall_spawn()
    test_kernel_syscall_alloc_free()
    test_kernel_syscall_channel_send_recv()
    test_kernel_syscall_getinfo()
    test_kernel_tick()
    test_kernel_run_limited()
    test_kernel_shutdown()
    test_kernel_stats()
    test_kernel_grace_request()

    # Integration tests
    print("\nIntegration Tests:")
    test_full_lifecycle()
    test_ipc_with_gatekeeper()
    test_parent_child_spawn()

    print("\n" + "=" * 60)
    print("All Phase 9 tests passed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_tests()
