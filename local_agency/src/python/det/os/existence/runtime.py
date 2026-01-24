"""
Existence Kernel Runtime
========================

The runtime that executes the Existence-Lang kernel on the architecture VM.

This bridges:
    Existence-Lang Kernel → EIS VM → DET Core

In the future with DET-native hardware:
    Existence-Lang Kernel → DET Silicon (direct)
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set
import time


class CreatureState(Enum):
    """Creature lifecycle states."""
    EMBRYONIC = auto()
    RUNNING = auto()
    BLOCKED = auto()
    WAITING = auto()
    DYING = auto()
    DEAD = auto()


@dataclass
class RuntimeCreature:
    """Runtime representation of a creature."""
    cid: int
    name: str
    parent_cid: int = 0

    # DET state
    F: float = 1.0
    a: float = 0.5
    q: float = 0.0
    C_self: float = 1.0
    P: float = 0.0  # Presence (computed)

    # Lifecycle
    state: CreatureState = CreatureState.EMBRYONIC

    # Runtime
    program: Optional[bytes] = None
    pc: int = 0
    registers: Dict[int, float] = field(default_factory=dict)

    # Grace
    grace_buffer: float = 0.0
    death_reason: str = ""

    # Timing
    spawn_time: float = field(default_factory=time.time)
    last_scheduled: float = 0.0
    total_runtime: float = 0.0

    def compute_presence(self) -> float:
        """P = F · C · a"""
        self.P = self.F * self.C_self * self.a
        return self.P

    def is_alive(self) -> bool:
        return self.state not in (CreatureState.DYING, CreatureState.DEAD)

    def is_schedulable(self) -> bool:
        return self.state == CreatureState.RUNNING and self.F > 0


@dataclass
class RuntimeChannel:
    """Runtime representation of a bond channel."""
    channel_id: int
    creature_a: int
    creature_b: int
    coherence: float = 0.5
    queue_a_to_b: List[Any] = field(default_factory=list)
    queue_b_to_a: List[Any] = field(default_factory=list)
    closed: bool = False
    last_activity: float = field(default_factory=time.time)


@dataclass
class RuntimeCapability:
    """Runtime representation of a capability."""
    cap_id: int
    owner: int
    action: str
    target: str
    min_agency: float = 0.0
    min_coherence: float = 0.0


class ExistenceKernelRuntime:
    """
    Runtime for executing the Existence-Lang kernel.

    This is the "architecture VM" that the kernel runs on.
    It provides:
        - Creature management
        - Memory/resource management
        - IPC (bond channels)
        - Security (capabilities)
        - DET physics integration

    With DET-native hardware, this becomes silicon.
    """

    def __init__(self, det_core=None, eis_vm=None, config=None):
        """Initialize runtime."""
        self.det_core = det_core
        self.eis_vm = eis_vm
        self.config = config

        # Creature table
        self.creatures: Dict[int, RuntimeCreature] = {}
        self.next_cid: int = 1  # 0 is kernel

        # Kernel creature (cid=0)
        self.kernel: Optional[RuntimeCreature] = None

        # Memory
        self.total_memory: int = config.total_memory if config else 1024 * 1024 * 1024
        self.total_F: float = config.total_F if config else 1000000.0
        self.available_F: float = self.total_F

        # Grace
        self.grace_pool: float = config.grace_pool if config else 10000.0

        # IPC
        self.channels: Dict[int, RuntimeChannel] = {}
        self.next_channel_id: int = 0

        # Security
        self.capabilities: Dict[int, RuntimeCapability] = {}
        self.creature_caps: Dict[int, Set[int]] = {}
        self.next_cap_id: int = 0

        # Timing
        self.tick_count: int = 0
        self.start_time: float = time.time()

        # Current scheduled creature
        self.current_cid: int = 0

    def boot_kernel(self, total_F: float, grace_pool: float, max_creatures: int):
        """Boot the kernel creature (cid=0)."""
        self.total_F = total_F
        self.available_F = total_F
        self.grace_pool = grace_pool

        # Create kernel creature
        self.kernel = RuntimeCreature(
            cid=0,
            name="kernel",
            parent_cid=-1,  # No parent
            F=total_F * 0.1,  # Kernel gets 10% of total F
            a=1.0,            # Maximum agency
            q=0.0,
            C_self=1.0,
            state=CreatureState.RUNNING
        )
        self.creatures[0] = self.kernel

        # Create root capability (all permissions)
        self._create_root_capability()

    def _create_root_capability(self):
        """Create root capability for kernel."""
        cap = RuntimeCapability(
            cap_id=self.next_cap_id,
            owner=0,
            action="*",
            target="*",
            min_agency=0.0,
            min_coherence=0.0
        )
        self.capabilities[cap.cap_id] = cap
        self.creature_caps[0] = {cap.cap_id}
        self.next_cap_id += 1

    # =========================================================================
    # Creature Management (implements kernel.ex Spawn/Kill kernels)
    # =========================================================================

    def spawn(self, name: str, initial_f: float = 1.0,
              initial_a: float = 0.5, program: Optional[bytes] = None,
              parent_cid: int = 0) -> int:
        """
        Spawn a new creature.

        This implements the Spawn kernel from kernel.ex:
        - Parent provides F (conservation)
        - Child agency <= parent agency
        - Returns creature ID
        """
        parent = self.creatures.get(parent_cid)
        if not parent:
            raise ValueError(f"Parent creature {parent_cid} not found")

        # Spawn cost = F + 10% overhead
        spawn_cost = initial_f * 1.1

        if parent.F < spawn_cost:
            raise ValueError(f"Insufficient F: need {spawn_cost}, have {parent.F}")

        # Conservation: transfer F from parent
        parent.F -= spawn_cost
        actual_a = min(initial_a, parent.a)  # Can't exceed parent

        # Create creature
        cid = self.next_cid
        self.next_cid += 1

        creature = RuntimeCreature(
            cid=cid,
            name=name,
            parent_cid=parent_cid,
            F=initial_f,
            a=actual_a,
            q=0.0,
            C_self=1.0,
            state=CreatureState.EMBRYONIC,
            program=program
        )
        self.creatures[cid] = creature

        # Create standard capabilities
        self._create_standard_capabilities(cid, actual_a)

        return cid

    def kill(self, cid: int, reason: str = ""):
        """
        Kill a creature.

        This implements the Kill kernel from kernel.ex:
        - Creature enters DYING state
        - F drains, then creature becomes DEAD
        """
        creature = self.creatures.get(cid)
        if not creature:
            return

        if cid == 0:
            # Cannot kill kernel
            return

        creature.state = CreatureState.DYING
        creature.death_reason = reason

    def _reap_dead(self):
        """Reap dead creatures and return their F to the pool."""
        to_remove = []
        for cid, creature in self.creatures.items():
            if creature.state == CreatureState.DEAD and cid != 0:
                # Return remaining F to pool
                self.available_F += creature.F
                to_remove.append(cid)

        for cid in to_remove:
            # Clean up capabilities
            if cid in self.creature_caps:
                for cap_id in self.creature_caps[cid]:
                    if cap_id in self.capabilities:
                        del self.capabilities[cap_id]
                del self.creature_caps[cid]

            # Clean up channels
            channels_to_close = [
                ch_id for ch_id, ch in self.channels.items()
                if ch.creature_a == cid or ch.creature_b == cid
            ]
            for ch_id in channels_to_close:
                del self.channels[ch_id]

            del self.creatures[cid]

    # =========================================================================
    # Scheduling (implements kernel.ex Schedule kernel)
    # =========================================================================

    def schedule(self) -> Optional[RuntimeCreature]:
        """
        Select next creature to run based on presence.

        This implements the Schedule kernel from kernel.ex:
        P = F · C · a determines priority
        """
        schedulable = [c for c in self.creatures.values() if c.is_schedulable()]

        if not schedulable:
            return None

        # Compute presence for all
        for c in schedulable:
            c.compute_presence()

        # Select highest presence
        best = max(schedulable, key=lambda c: c.P)
        self.current_cid = best.cid

        return best

    def run_creature(self, creature: RuntimeCreature, time_slice: float) -> float:
        """
        Run a creature for the given time slice.

        Consumes F proportional to computation.
        """
        if not creature.is_schedulable():
            return 0.0

        # Consume F for computation
        # Cost = time × sigma (processing rate)
        sigma = 0.1  # Default processing rate
        cost = time_slice * sigma

        actual_consumed = min(cost, creature.F)
        creature.F -= actual_consumed
        creature.total_runtime += time_slice
        creature.last_scheduled = time.time()

        # If creature has EIS program, execute it
        if self.eis_vm and creature.program:
            # TODO: Execute EIS bytecode
            pass

        return actual_consumed

    # =========================================================================
    # IPC (implements kernel.ex Send/Receive kernels)
    # =========================================================================

    def create_channel(self, cid_a: int, cid_b: int,
                       initial_coherence: float = 0.5) -> int:
        """Create a bond channel between two creatures."""
        channel_id = self.next_channel_id
        self.next_channel_id += 1

        channel = RuntimeChannel(
            channel_id=channel_id,
            creature_a=cid_a,
            creature_b=cid_b,
            coherence=initial_coherence
        )
        self.channels[channel_id] = channel

        return channel_id

    def send(self, sender_cid: int, channel_id: int, payload: Any) -> bool:
        """
        Send message through channel.

        Implements the Send kernel:
        - Delivery probability = coherence
        - Message cost based on size
        """
        channel = self.channels.get(channel_id)
        if not channel or channel.closed:
            return False

        sender = self.creatures.get(sender_cid)
        if not sender:
            return False

        # Calculate cost
        import random
        msg_size = 64  # Base size
        coherence_factor = 1.0 / max(0.1, channel.coherence)
        f_cost = (msg_size / 1024) * 0.01 * coherence_factor

        if sender.F < f_cost:
            return False

        # Probabilistic delivery based on coherence
        if random.random() > channel.coherence:
            # Message lost
            sender.F -= f_cost * 0.5
            channel.coherence = max(0, channel.coherence - 0.05)
            return False

        # Successful send
        sender.F -= f_cost

        if sender_cid == channel.creature_a:
            channel.queue_a_to_b.append(payload)
        else:
            channel.queue_b_to_a.append(payload)

        channel.coherence = min(1.0, channel.coherence + 0.01)
        channel.last_activity = time.time()

        return True

    def receive(self, receiver_cid: int, channel_id: int) -> Optional[Any]:
        """Receive message from channel."""
        channel = self.channels.get(channel_id)
        if not channel or channel.closed:
            return None

        if receiver_cid == channel.creature_b:
            queue = channel.queue_a_to_b
        else:
            queue = channel.queue_b_to_a

        if not queue:
            return None

        msg = queue.pop(0)
        channel.coherence = min(1.0, channel.coherence + 0.01)
        channel.last_activity = time.time()

        return msg

    # =========================================================================
    # Security (implements kernel.ex Gate kernel)
    # =========================================================================

    def _create_standard_capabilities(self, cid: int, agency: float):
        """Create standard capabilities based on agency level."""
        self.creature_caps[cid] = set()

        # READ (a >= 0.1)
        if agency >= 0.1:
            cap = RuntimeCapability(
                cap_id=self.next_cap_id,
                owner=cid,
                action="read",
                target="*",
                min_agency=0.1
            )
            self.capabilities[cap.cap_id] = cap
            self.creature_caps[cid].add(cap.cap_id)
            self.next_cap_id += 1

        # WRITE (a >= 0.3)
        if agency >= 0.3:
            cap = RuntimeCapability(
                cap_id=self.next_cap_id,
                owner=cid,
                action="write",
                target="*",
                min_agency=0.3
            )
            self.capabilities[cap.cap_id] = cap
            self.creature_caps[cid].add(cap.cap_id)
            self.next_cap_id += 1

        # EXECUTE (a >= 0.5)
        if agency >= 0.5:
            for action in ["execute", "spawn", "channel"]:
                cap = RuntimeCapability(
                    cap_id=self.next_cap_id,
                    owner=cid,
                    action=action,
                    target="*",
                    min_agency=0.5
                )
                self.capabilities[cap.cap_id] = cap
                self.creature_caps[cid].add(cap.cap_id)
                self.next_cap_id += 1

    def check_access(self, cid: int, action: str, target: str) -> bool:
        """
        Check if creature can perform action.

        Implements the Gate kernel:
        - Must have capability
        - Must have sufficient agency
        """
        creature = self.creatures.get(cid)
        if not creature:
            return False

        caps = self.creature_caps.get(cid, set())
        for cap_id in caps:
            cap = self.capabilities.get(cap_id)
            if not cap:
                continue

            # Check action match
            if cap.action != "*" and cap.action != action:
                continue

            # Check target match
            if cap.target != "*" and cap.target != target:
                continue

            # Check agency threshold
            if creature.a >= cap.min_agency:
                return True

        return False

    # =========================================================================
    # Grace (implements kernel.ex Grace kernel)
    # =========================================================================

    def inject_grace(self, cid: int, amount: float) -> float:
        """
        Inject grace into a creature.

        Grace comes from outside the creature's own resources.
        """
        creature = self.creatures.get(cid)
        if not creature:
            return 0.0

        if creature.F >= 0.1:
            # Creature doesn't need grace
            return 0.0

        actual = min(amount, self.grace_pool)
        if actual <= 0:
            return 0.0

        self.grace_pool -= actual
        creature.F += actual
        creature.q = max(0, creature.q - actual * 0.5)

        return actual

    # =========================================================================
    # Main Tick (implements kernel.ex Tick kernel)
    # =========================================================================

    def tick(self, dt: float):
        """
        Execute one kernel tick.

        This implements the Tick kernel from kernel.ex:
        1. Process grace
        2. Update creature states
        3. Update IPC coherence
        4. Schedule
        5. Run scheduled creature
        6. Replenish grace pool
        """
        # 1. Process grace buffers
        for creature in self.creatures.values():
            if creature.grace_buffer > 0:
                creature.F += creature.grace_buffer
                creature.q = max(0, creature.q - creature.grace_buffer * 0.5)
                creature.grace_buffer = 0

        # 2. Update creature states
        for creature in self.creatures.values():
            # Compute presence
            creature.compute_presence()

            # Check for natural death
            if creature.F <= 0 and creature.state != CreatureState.DYING:
                if creature.cid != 0:  # Not kernel
                    creature.state = CreatureState.DYING
                    creature.death_reason = "Resource depleted"

            # Process dying
            if creature.state == CreatureState.DYING:
                creature.F = max(0, creature.F - dt * 0.1)
                if creature.F <= 0:
                    creature.state = CreatureState.DEAD

            # Activate embryonic
            if creature.state == CreatureState.EMBRYONIC:
                creature.state = CreatureState.RUNNING

        # 3. Update IPC coherence
        now = time.time()
        for channel in self.channels.values():
            if channel.closed:
                continue

            idle_time = now - channel.last_activity
            if idle_time > 1.0:
                decay = dt * 0.01 * idle_time
                channel.coherence = max(0, channel.coherence - decay)

            if channel.coherence <= 0:
                channel.closed = True

        # 4. Schedule
        scheduled = self.schedule()

        # 5. Run scheduled creature
        if scheduled:
            # Time slice based on presence
            total_p = sum(c.P for c in self.creatures.values() if c.is_schedulable())
            if total_p > 0:
                fraction = scheduled.P / total_p
                time_slice = 0.01 * (1.0 + fraction * 10.0)
            else:
                time_slice = 0.01

            self.run_creature(scheduled, time_slice)

        # 6. Replenish grace pool
        max_grace = self.config.grace_pool if self.config else 10000.0
        self.grace_pool = min(max_grace, self.grace_pool + dt * 0.1)

        # 7. Reap dead creatures
        self._reap_dead()

        self.tick_count += 1

    def halt(self):
        """Halt the kernel."""
        # Kill all non-kernel creatures
        for cid in list(self.creatures.keys()):
            if cid != 0:
                self.kill(cid, "kernel halt")

    def get_kernel_state(self):
        """Get current kernel state."""
        from .bootstrap import KernelState
        return KernelState(
            tick=self.tick_count,
            num_creatures=len(self.creatures),
            total_F=sum(c.F for c in self.creatures.values()),
            grace_pool=self.grace_pool,
            scheduled_cid=self.current_cid
        )


__all__ = ['ExistenceKernelRuntime', 'RuntimeCreature', 'RuntimeChannel',
           'RuntimeCapability', 'CreatureState']
