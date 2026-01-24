"""
DET-OS Kernel - Main Kernel Implementation
==========================================

The DET-OS kernel orchestrates:
    - Creature lifecycle (spawn, schedule, die)
    - Resource management (F conservation)
    - Inter-process communication (bond channels)
    - Security (agency-based access control)
    - System calls (interface to kernel services)

Design principles:
    - Everything is a creature (including kernel services)
    - All resources obey conservation laws
    - Agency gates all actions
    - Coherence enables communication
    - Grace provides recovery at boundaries

Kernel Architecture:
    - KernelCreature: The root creature (always alive)
    - SystemCreatures: Essential services (scheduler, allocator, etc.)
    - UserCreatures: Application creatures
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
import time

from .creature import (
    Creature, CreatureState, CreatureFlags, CreatureTable, CreatureHandle
)
from .scheduler import (
    PresenceScheduler, SchedulerPolicy, ScheduleResult
)
from .allocator import (
    ResourceAllocator, AllocationFlags, AllocationResult
)
from .ipc import (
    IPCManager, BondChannel, ChannelFlags, IPCResult, Message
)
from .gatekeeper import (
    Gatekeeper, Permission, PermissionLevel, AccessResult, Capability
)


class KernelState(Enum):
    """Kernel execution state."""
    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()
    SHUTTING_DOWN = auto()
    HALTED = auto()


class Syscall(Enum):
    """System call numbers."""
    # Creature management
    SPAWN = 1
    EXIT = 2
    KILL = 3
    WAIT = 4
    YIELD = 5

    # Memory
    ALLOC = 10
    FREE = 11
    SHARE = 12

    # IPC
    CHANNEL_CREATE = 20
    CHANNEL_CLOSE = 21
    SEND = 22
    RECV = 23

    # Permissions
    CAP_CREATE = 30
    CAP_REVOKE = 31
    CAP_DELEGATE = 32
    ACCESS_CHECK = 33

    # Grace
    GRACE_REQUEST = 40
    GRACE_INJECT = 41

    # Info
    GETINFO = 50
    TICK_COUNT = 51


@dataclass
class SyscallResult:
    """Result of a system call."""
    success: bool
    value: Any = None
    error: str = ""
    F_consumed: float = 0.0


@dataclass
class KernelConfig:
    """Kernel configuration."""
    max_creatures: int = 4096
    total_memory: int = 1024 * 1024 * 1024  # 1GB
    total_F: float = 1000.0
    tick_dt: float = 0.02                    # 50 Hz
    scheduler_policy: SchedulerPolicy = SchedulerPolicy.PRESENCE
    grace_pool: float = 100.0                # Grace reserve


class DETOSKernel:
    """
    DET-OS Kernel - Agency-First Operating System

    The kernel is itself a creature (the root creature) that manages
    all other creatures. It provides:
        - Creature spawn/kill/schedule
        - Resource allocation with conservation
        - Bond-based IPC
        - Agency-based security

    Usage:
        kernel = DETOSKernel()
        creature = kernel.spawn("my_app", initial_f=10.0)
        kernel.run()
    """

    def __init__(self, config: Optional[KernelConfig] = None):
        """Initialize the kernel."""
        self.config = config or KernelConfig()
        self.state = KernelState.INITIALIZING

        # Core subsystems
        self.creatures = CreatureTable(max_creatures=self.config.max_creatures)
        self.scheduler = PresenceScheduler(
            self.creatures,
            policy=self.config.scheduler_policy
        )
        self.allocator = ResourceAllocator(
            total_memory=self.config.total_memory,
            total_F=self.config.total_F
        )
        self.ipc = IPCManager()
        self.gatekeeper = Gatekeeper()

        # Kernel state
        self.tick = 0
        self.start_time = time.time()
        self.grace_pool = self.config.grace_pool

        # Create kernel creature
        self._init_kernel_creature()

        self.state = KernelState.RUNNING

    def _init_kernel_creature(self):
        """Initialize the kernel root creature."""
        kernel = self.creatures.spawn(
            name="kernel",
            parent=None,
            initial_f=100.0,
            initial_a=1.0,  # Maximum agency
            flags=CreatureFlags.KERNEL | CreatureFlags.IMMORTAL
        )
        kernel.state = CreatureState.RUNNING

        # Create root capability
        self.gatekeeper.create_standard_capabilities(
            kernel.handle.cid,
            PermissionLevel.ROOT
        )

        self.kernel_handle = kernel.handle

    # =========================================================================
    # CREATURE MANAGEMENT
    # =========================================================================

    def spawn(self,
              name: str,
              parent: Optional[CreatureHandle] = None,
              initial_f: float = 1.0,
              initial_a: float = 0.5,
              flags: CreatureFlags = CreatureFlags.NONE,
              program: Optional[bytes] = None) -> Creature:
        """
        Spawn a new creature.

        Args:
            name: Creature name
            parent: Parent creature (None for kernel-spawned)
            initial_f: Initial resource
            initial_a: Initial agency
            flags: Creature flags
            program: EIS bytecode

        Returns:
            The spawned creature
        """
        # Default parent to kernel
        if parent is None:
            parent = self.kernel_handle

        creature = self.creatures.spawn(
            name=name,
            parent=parent,
            initial_f=initial_f,
            initial_a=initial_a,
            flags=flags,
            program=program
        )

        # Create basic capabilities
        level = PermissionLevel.EXECUTE if initial_a >= 0.5 else PermissionLevel.WRITE
        self.gatekeeper.create_standard_capabilities(creature.handle.cid, level)

        return creature

    def kill(self, handle: CreatureHandle, reason: str = ""):
        """Kill a creature."""
        creature = self.creatures.get(handle)
        if not creature:
            return

        # Kernel creature cannot be killed
        if creature.flags & CreatureFlags.KERNEL:
            return

        self.creatures.kill(handle, reason)

        # Clean up resources
        self.allocator.free_all_for_creature(handle.cid)
        self.ipc.close_all_for_creature(handle.cid)
        self.gatekeeper.revoke_all_for_creature(handle.cid)

    def get_creature(self, handle: CreatureHandle) -> Optional[Creature]:
        """Get creature by handle."""
        return self.creatures.get(handle)

    def get_creature_by_name(self, name: str) -> Optional[Creature]:
        """Get creature by name."""
        return self.creatures.get_by_name(name)

    # =========================================================================
    # SYSTEM CALLS
    # =========================================================================

    def syscall(self,
                caller: CreatureHandle,
                call: Syscall,
                *args) -> SyscallResult:
        """
        Execute a system call.

        Args:
            caller: Calling creature
            call: System call number
            args: Call-specific arguments

        Returns:
            SyscallResult with success/failure and value
        """
        creature = self.creatures.get(caller)
        if not creature or not creature.is_alive():
            return SyscallResult(False, error="Invalid caller")

        # System call costs F
        syscall_cost = 0.001
        if creature.F < syscall_cost:
            return SyscallResult(False, error="Insufficient F for syscall")

        creature.consume_resource(syscall_cost)

        # Dispatch
        if call == Syscall.SPAWN:
            return self._sys_spawn(creature, *args)
        elif call == Syscall.EXIT:
            return self._sys_exit(creature, *args)
        elif call == Syscall.KILL:
            return self._sys_kill(creature, *args)
        elif call == Syscall.YIELD:
            return self._sys_yield(creature)
        elif call == Syscall.ALLOC:
            return self._sys_alloc(creature, *args)
        elif call == Syscall.FREE:
            return self._sys_free(creature, *args)
        elif call == Syscall.CHANNEL_CREATE:
            return self._sys_channel_create(creature, *args)
        elif call == Syscall.SEND:
            return self._sys_send(creature, *args)
        elif call == Syscall.RECV:
            return self._sys_recv(creature, *args)
        elif call == Syscall.GRACE_REQUEST:
            return self._sys_grace_request(creature, *args)
        elif call == Syscall.GETINFO:
            return self._sys_getinfo(creature, *args)
        else:
            return SyscallResult(False, error=f"Unknown syscall: {call}")

    def _sys_spawn(self, caller: Creature, name: str, initial_f: float = 1.0,
                   initial_a: float = 0.5) -> SyscallResult:
        """Spawn syscall."""
        # Check permission
        result, _ = self.gatekeeper.check_access(
            caller.handle.cid, caller.a, caller.C_self, caller.F,
            "spawn", name
        )
        if result != AccessResult.GRANTED:
            return SyscallResult(False, error="Permission denied")

        # Spawn costs F
        spawn_cost = initial_f * 1.1  # 10% overhead
        if caller.F < spawn_cost:
            return SyscallResult(False, error="Insufficient F to spawn")

        caller.consume_resource(spawn_cost)

        try:
            child = self.spawn(
                name=name,
                parent=caller.handle,
                initial_f=initial_f,
                initial_a=min(initial_a, caller.a)  # Can't exceed parent's agency
            )
            return SyscallResult(True, value=child.handle, F_consumed=spawn_cost)
        except Exception as e:
            return SyscallResult(False, error=str(e))

    def _sys_exit(self, caller: Creature, code: int = 0) -> SyscallResult:
        """Exit syscall."""
        self.creatures.kill(caller.handle, f"exit({code})")
        return SyscallResult(True, value=code)

    def _sys_kill(self, caller: Creature, target: CreatureHandle,
                  reason: str = "") -> SyscallResult:
        """Kill syscall."""
        target_creature = self.creatures.get(target)
        if not target_creature:
            return SyscallResult(False, error="Target not found")

        # Can only kill children or with admin permission
        is_parent = target_creature.parent == caller.handle
        if not is_parent:
            result, _ = self.gatekeeper.check_access(
                caller.handle.cid, caller.a, caller.C_self, caller.F,
                "kill", str(target.cid)
            )
            if result != AccessResult.GRANTED:
                return SyscallResult(False, error="Permission denied")

        self.kill(target, reason)
        return SyscallResult(True)

    def _sys_yield(self, caller: Creature) -> SyscallResult:
        """Yield syscall."""
        self.scheduler.yield_current()
        return SyscallResult(True)

    def _sys_alloc(self, caller: Creature, size: int,
                   flags: AllocationFlags = AllocationFlags.NONE) -> SyscallResult:
        """Alloc syscall."""
        result = self.allocator.allocate(
            caller.handle.cid, size, flags, caller.F
        )

        if result.success:
            caller.consume_resource(result.F_consumed)

        return SyscallResult(
            success=result.success,
            value=result.block,
            error=result.reason if not result.success else "",
            F_consumed=result.F_consumed
        )

    def _sys_free(self, caller: Creature, block_id: int) -> SyscallResult:
        """Free syscall."""
        block = self.allocator.blocks.get(block_id)
        if not block:
            return SyscallResult(False, error="Block not found")

        if block.owner != caller.handle.cid:
            return SyscallResult(False, error="Not owner")

        F_returned = self.allocator.free(block_id)
        caller.F += F_returned  # Return resource to creature

        return SyscallResult(True, value=F_returned)

    def _sys_channel_create(self, caller: Creature,
                            target_cid: int) -> SyscallResult:
        """Create channel syscall."""
        channel = self.ipc.create_channel(
            caller.handle.cid,
            target_cid,
            initial_coherence=caller.C_self
        )
        return SyscallResult(True, value=channel)

    def _sys_send(self, caller: Creature, channel_id: int,
                  payload: Any) -> SyscallResult:
        """Send message syscall."""
        channel = self.ipc.get_channel(channel_id)
        if not channel:
            return SyscallResult(False, error="Channel not found")

        result, F_cost = channel.send(caller.handle.cid, payload, caller.F)

        if result == IPCResult.SUCCESS:
            caller.consume_resource(F_cost)
            return SyscallResult(True, F_consumed=F_cost)
        else:
            return SyscallResult(False, error=result.name)

    def _sys_recv(self, caller: Creature, channel_id: int) -> SyscallResult:
        """Receive message syscall."""
        channel = self.ipc.get_channel(channel_id)
        if not channel:
            return SyscallResult(False, error="Channel not found")

        result, msg = channel.receive(caller.handle.cid)

        if result == IPCResult.SUCCESS:
            return SyscallResult(True, value=msg)
        else:
            return SyscallResult(False, error=result.name)

    def _sys_grace_request(self, caller: Creature,
                           amount: float) -> SyscallResult:
        """Request grace injection."""
        if not caller.needs_grace():
            return SyscallResult(False, error="Grace not needed")

        # Limit to what's available
        actual = min(amount, self.grace_pool)
        if actual <= 0:
            return SyscallResult(False, error="Grace pool empty")

        self.grace_pool -= actual
        caller.inject_grace(actual)

        return SyscallResult(True, value=actual)

    def _sys_getinfo(self, caller: Creature, info_type: str) -> SyscallResult:
        """Get system information."""
        if info_type == "tick":
            return SyscallResult(True, value=self.tick)
        elif info_type == "uptime":
            return SyscallResult(True, value=time.time() - self.start_time)
        elif info_type == "num_creatures":
            return SyscallResult(True, value=self.creatures.num_alive())
        elif info_type == "grace_pool":
            return SyscallResult(True, value=self.grace_pool)
        elif info_type == "self":
            return SyscallResult(True, value={
                "F": caller.F,
                "a": caller.a,
                "P": caller.P,
                "state": caller.state.name
            })
        else:
            return SyscallResult(False, error=f"Unknown info type: {info_type}")

    # =========================================================================
    # KERNEL TICK
    # =========================================================================

    def kernel_tick(self) -> ScheduleResult:
        """
        Execute one kernel tick.

        This is the main kernel loop iteration:
        1. Update creature table (grace, death)
        2. Update IPC channels (coherence decay)
        3. Schedule next creature
        4. Run scheduled creature
        5. Increment tick counter
        """
        dt = self.config.tick_dt

        # Update creatures
        self.creatures.tick(dt)

        # Update IPC
        self.ipc.tick(dt)

        # Schedule
        schedule = self.scheduler.schedule()

        # Run scheduled creature
        if schedule.creature:
            actual_time = self.scheduler.run_creature(
                schedule.creature,
                schedule.time_slice
            )

            # TODO: Execute EIS bytecode if creature has program

        # Replenish grace pool slowly
        self.grace_pool = min(
            self.config.grace_pool,
            self.grace_pool + dt * 0.1
        )

        self.tick += 1

        return schedule

    def run(self, max_ticks: Optional[int] = None):
        """
        Run the kernel main loop.

        Args:
            max_ticks: Maximum ticks to run (None for infinite)
        """
        self.state = KernelState.RUNNING
        ticks = 0

        while self.state == KernelState.RUNNING:
            self.kernel_tick()
            ticks += 1

            if max_ticks and ticks >= max_ticks:
                break

            # Small sleep to prevent CPU spin
            time.sleep(self.config.tick_dt)

    def pause(self):
        """Pause kernel execution."""
        self.state = KernelState.PAUSED

    def resume(self):
        """Resume kernel execution."""
        if self.state == KernelState.PAUSED:
            self.state = KernelState.RUNNING

    def shutdown(self):
        """Initiate kernel shutdown."""
        self.state = KernelState.SHUTTING_DOWN

        # Kill all non-kernel creatures
        for creature in self.creatures.get_all_alive():
            if not (creature.flags & CreatureFlags.KERNEL):
                self.kill(creature.handle, "shutdown")

        self.state = KernelState.HALTED

    # =========================================================================
    # KERNEL INFO
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get kernel statistics."""
        return {
            "tick": self.tick,
            "uptime": time.time() - self.start_time,
            "state": self.state.name,
            "creatures": {
                "alive": self.creatures.num_alive(),
                "total": self.creatures.num_total(),
            },
            "scheduler": self.scheduler.get_stats().__dict__,
            "allocator": self.allocator.get_stats(),
            "ipc": self.ipc.get_stats(),
            "grace_pool": self.grace_pool,
        }


__all__ = [
    'KernelState', 'Syscall', 'SyscallResult', 'KernelConfig',
    'DETOSKernel'
]
