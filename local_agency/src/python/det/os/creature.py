"""
DET-OS Creature Manager
=======================

Creatures are the DET-OS equivalent of processes. Unlike traditional processes,
creatures:
    - Have intrinsic agency (a) that cannot be created or destroyed
    - Require resource (F) to exist - when F depletes, creature dies
    - Participate in coherence dynamics with other creatures via bonds
    - Have presence (P = F·C·a) that determines scheduling priority
    - Can spawn child creatures (inheriting agency)
    - Die gracefully when resources are exhausted

Lifecycle:
    EMBRYONIC → RUNNING → (BLOCKED | WAITING) → DYING → DEAD

Key principle: A creature's existence is not guaranteed - it must maintain
itself through resource acquisition and coherent bonds.
"""

from enum import Enum, Flag, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Callable, Any
import time


class CreatureState(Enum):
    """Creature lifecycle states."""
    EMBRYONIC = auto()   # Just spawned, initializing
    RUNNING = auto()     # Active, can be scheduled
    BLOCKED = auto()     # Waiting for resource/bond
    WAITING = auto()     # Waiting for child/event
    DYING = auto()       # F critically low, entering death
    DEAD = auto()        # No longer exists


class CreatureFlags(Flag):
    """Creature attribute flags."""
    NONE = 0
    KERNEL = auto()      # Kernel creature (cannot die)
    DAEMON = auto()      # Background creature
    REALTIME = auto()    # Real-time scheduling
    PINNED = auto()      # Pinned to specific core
    IMMORTAL = auto()    # Grace injection on death
    SOMATIC = auto()     # Has physical I/O bindings
    LLM = auto()         # LLM-backed creature


@dataclass
class CreatureHandle:
    """Opaque handle to a creature."""
    cid: int             # Creature ID
    generation: int      # Generation counter (for ABA problem)

    def __hash__(self):
        return hash((self.cid, self.generation))

    def __eq__(self, other):
        if not isinstance(other, CreatureHandle):
            return False
        return self.cid == other.cid and self.generation == other.generation


@dataclass
class Creature:
    """
    A creature is a self-maintaining unit of existence in DET-OS.

    Properties:
        F (float): Resource level. When F → 0, creature dies.
        a (float): Agency [0, 1]. Intrinsic, cannot be created.
        q (float): Structural debt. Accumulated absence.
        theta (float): Phase angle for synchronization.
        sigma (float): Processing rate / conductivity.
        P (float): Presence = F · C_self · a (computed).

    Unlike Unix processes:
        - No PID wraparound - creatures have handles with generations
        - Death is natural when F depletes (not killed)
        - Parent-child is resource inheritance, not fork()
        - Scheduling is by presence, not priority numbers
    """

    # Identity
    handle: CreatureHandle
    name: str
    parent: Optional[CreatureHandle] = None

    # DET state
    F: float = 1.0           # Resource
    a: float = 0.5           # Agency (intrinsic)
    q: float = 0.0           # Structural debt
    theta: float = 0.0       # Phase
    sigma: float = 0.1       # Processing rate

    # Computed
    P: float = 0.0           # Presence (computed each tick)
    C_self: float = 1.0      # Self-coherence (from bonds)

    # Lifecycle
    state: CreatureState = CreatureState.EMBRYONIC
    flags: CreatureFlags = CreatureFlags.NONE

    # Timing
    spawn_time: float = field(default_factory=time.time)
    last_scheduled: float = 0.0
    total_runtime: float = 0.0
    tick_count: int = 0

    # Relationships
    children: Set[CreatureHandle] = field(default_factory=set)
    bonds: Set[int] = field(default_factory=set)  # Bond indices

    # Execution
    program: Optional[bytes] = None  # EIS bytecode
    pc: int = 0                      # Program counter
    registers: Dict[int, float] = field(default_factory=dict)

    # Grace and death
    grace_buffer: float = 0.0        # Pending grace injection
    death_reason: str = ""

    def compute_presence(self) -> float:
        """Compute presence P = F · C_self · a."""
        self.P = self.F * self.C_self * self.a
        return self.P

    def is_alive(self) -> bool:
        """Check if creature is still alive."""
        return self.state not in (CreatureState.DYING, CreatureState.DEAD)

    def is_schedulable(self) -> bool:
        """Check if creature can be scheduled."""
        return self.state == CreatureState.RUNNING and self.F > 0

    def needs_grace(self) -> bool:
        """Check if creature needs grace injection."""
        return self.F < 0.1 and self.state != CreatureState.DEAD

    def consume_resource(self, amount: float) -> float:
        """Consume resource, returning actual amount consumed."""
        actual = min(amount, self.F)
        self.F -= actual
        return actual

    def inject_grace(self, amount: float):
        """Inject grace (resource from boundary)."""
        self.grace_buffer += amount

    def process_grace(self):
        """Process pending grace injection."""
        if self.grace_buffer > 0:
            self.F += self.grace_buffer
            self.q = max(0, self.q - self.grace_buffer * 0.5)  # Grace reduces debt
            self.grace_buffer = 0


class CreatureTable:
    """
    Table of all creatures in the system.

    Uses generation counters to handle the ABA problem (creature IDs reused
    after death). Maintains parent-child relationships and provides efficient
    lookup by handle or name.
    """

    def __init__(self, max_creatures: int = 4096):
        self.max_creatures = max_creatures
        self.creatures: Dict[int, Creature] = {}
        self.generations: Dict[int, int] = {}  # cid -> current generation
        self.name_index: Dict[str, CreatureHandle] = {}
        self.free_ids: List[int] = list(range(max_creatures))
        self.next_id: int = 0

    def allocate_id(self) -> int:
        """Allocate a creature ID."""
        if self.free_ids:
            return self.free_ids.pop(0)
        if self.next_id < self.max_creatures:
            cid = self.next_id
            self.next_id += 1
            return cid
        raise RuntimeError("Creature table full")

    def free_id(self, cid: int):
        """Return a creature ID to the free pool."""
        self.free_ids.append(cid)

    def spawn(self, name: str, parent: Optional[CreatureHandle] = None,
              initial_f: float = 1.0, initial_a: float = 0.5,
              flags: CreatureFlags = CreatureFlags.NONE,
              program: Optional[bytes] = None) -> Creature:
        """
        Spawn a new creature.

        Args:
            name: Creature name (must be unique)
            parent: Parent creature handle (for resource inheritance)
            initial_f: Initial resource level
            initial_a: Initial agency (intrinsic)
            flags: Creature flags
            program: EIS bytecode to execute

        Returns:
            The newly spawned creature

        Raises:
            ValueError: If name already exists
            RuntimeError: If table is full
        """
        if name in self.name_index:
            raise ValueError(f"Creature '{name}' already exists")

        cid = self.allocate_id()
        gen = self.generations.get(cid, 0) + 1
        self.generations[cid] = gen

        handle = CreatureHandle(cid=cid, generation=gen)

        creature = Creature(
            handle=handle,
            name=name,
            parent=parent,
            F=initial_f,
            a=initial_a,
            flags=flags,
            program=program,
            state=CreatureState.EMBRYONIC
        )

        self.creatures[cid] = creature
        self.name_index[name] = handle

        # Register with parent
        if parent and parent.cid in self.creatures:
            self.creatures[parent.cid].children.add(handle)

        return creature

    def get(self, handle: CreatureHandle) -> Optional[Creature]:
        """Get creature by handle, checking generation."""
        creature = self.creatures.get(handle.cid)
        if creature and creature.handle.generation == handle.generation:
            return creature
        return None

    def get_by_name(self, name: str) -> Optional[Creature]:
        """Get creature by name."""
        handle = self.name_index.get(name)
        if handle:
            return self.get(handle)
        return None

    def kill(self, handle: CreatureHandle, reason: str = ""):
        """Kill a creature (begin death process)."""
        creature = self.get(handle)
        if creature and creature.is_alive():
            creature.state = CreatureState.DYING
            creature.death_reason = reason

    def reap(self, handle: CreatureHandle):
        """Reap a dead creature (remove from table)."""
        creature = self.get(handle)
        if creature and creature.state == CreatureState.DEAD:
            # Remove from parent's children
            if creature.parent:
                parent = self.get(creature.parent)
                if parent:
                    parent.children.discard(handle)

            # Remove from indices
            if creature.name in self.name_index:
                del self.name_index[creature.name]
            del self.creatures[handle.cid]
            self.free_id(handle.cid)

    def tick(self, dt: float = 0.02):
        """Process one tick for all creatures."""
        dead_creatures = []

        for cid, creature in self.creatures.items():
            if creature.state == CreatureState.DEAD:
                dead_creatures.append(creature.handle)
                continue

            # Process grace
            creature.process_grace()

            # Compute presence
            creature.compute_presence()

            # Check for death
            if creature.F <= 0 and not (creature.flags & CreatureFlags.IMMORTAL):
                if creature.state != CreatureState.DYING:
                    creature.state = CreatureState.DYING
                    creature.death_reason = "Resource depleted"

            # Process dying creatures
            if creature.state == CreatureState.DYING:
                creature.F = max(0, creature.F - dt * 0.1)  # Drain remaining
                if creature.F <= 0:
                    creature.state = CreatureState.DEAD

            # Activate embryonic creatures
            if creature.state == CreatureState.EMBRYONIC:
                creature.state = CreatureState.RUNNING

            creature.tick_count += 1

        # Reap dead creatures
        for handle in dead_creatures:
            self.reap(handle)

    def get_all_alive(self) -> List[Creature]:
        """Get all alive creatures."""
        return [c for c in self.creatures.values() if c.is_alive()]

    def get_schedulable(self) -> List[Creature]:
        """Get all schedulable creatures."""
        return [c for c in self.creatures.values() if c.is_schedulable()]

    def num_alive(self) -> int:
        """Count alive creatures."""
        return sum(1 for c in self.creatures.values() if c.is_alive())

    def num_total(self) -> int:
        """Count total creatures (including dead)."""
        return len(self.creatures)


__all__ = [
    'CreatureState', 'CreatureFlags', 'CreatureHandle',
    'Creature', 'CreatureTable'
]
