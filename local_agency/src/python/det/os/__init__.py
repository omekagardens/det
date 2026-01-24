"""
DET-OS Kernel - Agency-First Operating System
==============================================

An operating system where agency is the primitive and traditional OS concepts
emerge from DET physics:

    Process       → Creature (self-maintaining existence)
    Scheduler     → Presence dynamics (who appears in experience)
    Memory alloc  → Resource (F) distribution (conservation law)
    IPC           → Bond-mediated flux (coherent communication)
    Permissions   → Agency constraints (grace-gated access)
    Interrupts    → Somatic events (embodied urgency)

Core Philosophy:
    Agency creates distinction → distinction creates movement →
    movement leaves trace → trace becomes mathematics.

Usage:
    from det.os import DETOSKernel, Creature, CreatureState

    kernel = DETOSKernel()

    # Spawn a creature
    creature = kernel.spawn("sensor_daemon", initial_f=10.0, initial_a=0.8)

    # Run kernel tick
    kernel.tick()

    # Creature scheduling happens via presence dynamics
    scheduled = kernel.get_scheduled_creature()
"""

from .creature import (
    Creature, CreatureState, CreatureFlags,
    CreatureTable, CreatureHandle
)

from .scheduler import (
    PresenceScheduler, SchedulerPolicy,
    ScheduleResult, ScheduleStats
)

from .allocator import (
    ResourceAllocator, MemoryBlock, AllocationFlags,
    AllocationResult, ResourcePool
)

from .ipc import (
    BondChannel, Message, MessageQueue,
    IPCResult, ChannelFlags, IPCManager
)

from .gatekeeper import (
    Gatekeeper, Permission, PermissionLevel,
    AccessResult, Capability
)

from .kernel import (
    DETOSKernel, KernelState, KernelConfig,
    SyscallResult, Syscall
)

__version__ = "0.1.0"

__all__ = [
    # Creature management
    "Creature", "CreatureState", "CreatureFlags",
    "CreatureTable", "CreatureHandle",

    # Scheduling
    "PresenceScheduler", "SchedulerPolicy",
    "ScheduleResult", "ScheduleStats",

    # Resource allocation
    "ResourceAllocator", "MemoryBlock", "AllocationFlags",
    "AllocationResult", "ResourcePool",

    # IPC
    "BondChannel", "Message", "MessageQueue",
    "IPCResult", "ChannelFlags", "IPCManager",

    # Security
    "Gatekeeper", "Permission", "PermissionLevel",
    "AccessResult", "Capability",

    # Kernel
    "DETOSKernel", "KernelState", "KernelConfig",
    "SyscallResult", "Syscall",
]
