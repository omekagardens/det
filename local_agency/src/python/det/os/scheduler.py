"""
DET-OS Presence Scheduler
=========================

Scheduling in DET-OS is based on presence dynamics, not arbitrary priority
numbers. A creature's presence P = F·C·a determines its "right to appear"
in the computational experience.

Key differences from traditional schedulers:
    - No priority inversion (presence is physics-based)
    - No starvation (low-F creatures naturally die)
    - Coherence matters (bonded creatures sync)
    - Agency gates action (high-a creatures act first)

Scheduling Algorithm:
    1. Compute presence for all schedulable creatures
    2. Select creature with highest presence
    3. Run creature for time slice proportional to P
    4. Consume F proportional to computation done
    5. Update coherence based on bond interactions

Real-time Support:
    Creatures with REALTIME flag get minimum presence guarantee.
    Somatic creatures (physical I/O) get priority boost.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable
import time
import heapq

from .creature import Creature, CreatureHandle, CreatureTable, CreatureState, CreatureFlags


class SchedulerPolicy(Enum):
    """Scheduling policy."""
    PRESENCE = auto()      # Pure presence-based (default)
    ROUND_ROBIN = auto()   # Equal time slices
    FIFO = auto()          # First-in-first-out
    REALTIME = auto()      # Real-time guarantees
    COOPERATIVE = auto()   # Creatures yield explicitly


@dataclass
class ScheduleResult:
    """Result of scheduling decision."""
    creature: Optional[Creature]
    time_slice: float          # Allocated time in seconds
    reason: str                # Why this creature was chosen


@dataclass
class ScheduleStats:
    """Scheduler statistics."""
    total_schedules: int = 0
    total_runtime: float = 0.0
    context_switches: int = 0
    idle_time: float = 0.0
    presence_sum: float = 0.0
    creatures_scheduled: Dict[int, int] = field(default_factory=dict)


class PresenceScheduler:
    """
    Presence-based scheduler for DET-OS.

    Implements scheduling as a physics problem: creatures with higher
    presence (P = F·C·a) get more CPU time. This naturally handles
    priority, starvation, and fairness through DET dynamics.
    """

    def __init__(self,
                 creature_table: CreatureTable,
                 policy: SchedulerPolicy = SchedulerPolicy.PRESENCE,
                 base_time_slice: float = 0.01,
                 min_time_slice: float = 0.001,
                 max_time_slice: float = 0.1):
        """
        Initialize scheduler.

        Args:
            creature_table: Table of all creatures
            policy: Scheduling policy
            base_time_slice: Base time slice in seconds
            min_time_slice: Minimum time slice
            max_time_slice: Maximum time slice
        """
        self.creatures = creature_table
        self.policy = policy
        self.base_time_slice = base_time_slice
        self.min_time_slice = min_time_slice
        self.max_time_slice = max_time_slice

        self.current: Optional[CreatureHandle] = None
        self.stats = ScheduleStats()
        self.run_queue: List[CreatureHandle] = []
        self.last_schedule_time = time.time()

        # Presence boost for special creatures
        self.realtime_boost = 2.0
        self.somatic_boost = 1.5
        self.kernel_boost = 3.0

    def compute_effective_presence(self, creature: Creature) -> float:
        """
        Compute effective presence with policy adjustments.

        Effective presence = P × boosts × policy_modifier
        """
        base_p = creature.compute_presence()

        # Apply flag-based boosts
        boost = 1.0
        if creature.flags & CreatureFlags.KERNEL:
            boost *= self.kernel_boost
        if creature.flags & CreatureFlags.REALTIME:
            boost *= self.realtime_boost
        if creature.flags & CreatureFlags.SOMATIC:
            boost *= self.somatic_boost

        # Apply anti-starvation: creatures not scheduled recently get boost
        time_since_scheduled = time.time() - creature.last_scheduled
        if time_since_scheduled > 1.0:  # More than 1 second
            starvation_boost = min(2.0, 1.0 + time_since_scheduled * 0.1)
            boost *= starvation_boost

        return base_p * boost

    def compute_time_slice(self, creature: Creature, total_presence: float) -> float:
        """
        Compute time slice proportional to presence.

        Time slice = base × (P / total_P) × scale_factor
        """
        if total_presence <= 0:
            return self.min_time_slice

        effective_p = self.compute_effective_presence(creature)
        fraction = effective_p / total_presence

        # Scale time slice by presence fraction
        time_slice = self.base_time_slice * (1.0 + fraction * 10.0)

        # Clamp to bounds
        return max(self.min_time_slice, min(self.max_time_slice, time_slice))

    def schedule(self) -> ScheduleResult:
        """
        Select next creature to run.

        Returns:
            ScheduleResult with selected creature and time slice
        """
        schedulable = self.creatures.get_schedulable()

        if not schedulable:
            self.stats.idle_time += self.base_time_slice
            return ScheduleResult(
                creature=None,
                time_slice=self.base_time_slice,
                reason="No schedulable creatures"
            )

        if self.policy == SchedulerPolicy.PRESENCE:
            return self._schedule_by_presence(schedulable)
        elif self.policy == SchedulerPolicy.ROUND_ROBIN:
            return self._schedule_round_robin(schedulable)
        elif self.policy == SchedulerPolicy.FIFO:
            return self._schedule_fifo(schedulable)
        else:
            return self._schedule_by_presence(schedulable)

    def _schedule_by_presence(self, schedulable: List[Creature]) -> ScheduleResult:
        """Schedule by highest effective presence."""
        # Compute total presence
        total_p = sum(self.compute_effective_presence(c) for c in schedulable)
        self.stats.presence_sum = total_p

        # Find creature with highest effective presence
        best_creature = max(schedulable, key=self.compute_effective_presence)
        effective_p = self.compute_effective_presence(best_creature)

        time_slice = self.compute_time_slice(best_creature, total_p)

        # Update stats
        self._record_schedule(best_creature, time_slice)

        return ScheduleResult(
            creature=best_creature,
            time_slice=time_slice,
            reason=f"Highest presence: P={effective_p:.4f}"
        )

    def _schedule_round_robin(self, schedulable: List[Creature]) -> ScheduleResult:
        """Simple round-robin scheduling."""
        # Find next creature after current
        if self.current:
            current_idx = -1
            for i, c in enumerate(schedulable):
                if c.handle == self.current:
                    current_idx = i
                    break
            next_idx = (current_idx + 1) % len(schedulable)
        else:
            next_idx = 0

        creature = schedulable[next_idx]
        self._record_schedule(creature, self.base_time_slice)

        return ScheduleResult(
            creature=creature,
            time_slice=self.base_time_slice,
            reason="Round robin"
        )

    def _schedule_fifo(self, schedulable: List[Creature]) -> ScheduleResult:
        """First-in-first-out based on spawn time."""
        # Sort by spawn time (oldest first)
        oldest = min(schedulable, key=lambda c: c.spawn_time)
        self._record_schedule(oldest, self.base_time_slice)

        return ScheduleResult(
            creature=oldest,
            time_slice=self.base_time_slice,
            reason="FIFO (oldest)"
        )

    def _record_schedule(self, creature: Creature, time_slice: float):
        """Record scheduling decision in stats."""
        now = time.time()

        # Track context switch
        if self.current and self.current != creature.handle:
            self.stats.context_switches += 1

        self.current = creature.handle
        creature.last_scheduled = now
        creature.total_runtime += time_slice

        self.stats.total_schedules += 1
        self.stats.total_runtime += time_slice

        cid = creature.handle.cid
        self.stats.creatures_scheduled[cid] = \
            self.stats.creatures_scheduled.get(cid, 0) + 1

        self.last_schedule_time = now

    def run_creature(self, creature: Creature, time_slice: float) -> float:
        """
        Run a creature for the given time slice.

        In a real implementation, this would execute EIS bytecode.
        Returns actual time used (may be less if creature yields/blocks).
        """
        if not creature.is_schedulable():
            return 0.0

        # Consume resource proportional to time
        # Cost = time × sigma (processing rate)
        resource_cost = time_slice * creature.sigma
        actual_consumed = creature.consume_resource(resource_cost)

        # If we couldn't consume full cost, creature is resource-starved
        if actual_consumed < resource_cost:
            actual_time = time_slice * (actual_consumed / resource_cost)
        else:
            actual_time = time_slice

        return actual_time

    def yield_current(self):
        """Current creature yields voluntarily."""
        self.current = None

    def block_current(self, reason: str = ""):
        """Block current creature (waiting for resource/event)."""
        if self.current:
            creature = self.creatures.get(self.current)
            if creature:
                creature.state = CreatureState.BLOCKED
        self.current = None

    def unblock(self, handle: CreatureHandle):
        """Unblock a creature."""
        creature = self.creatures.get(handle)
        if creature and creature.state == CreatureState.BLOCKED:
            creature.state = CreatureState.RUNNING

    def get_stats(self) -> ScheduleStats:
        """Get scheduler statistics."""
        return self.stats

    def reset_stats(self):
        """Reset scheduler statistics."""
        self.stats = ScheduleStats()


__all__ = [
    'SchedulerPolicy', 'ScheduleResult', 'ScheduleStats',
    'PresenceScheduler'
]
