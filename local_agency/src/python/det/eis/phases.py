"""
EIS Execution Phases
====================

Canonical tick phases (bytecode-visible):

    Phase A — READ/MEASURE (Past-only reads)
        - Read trace fields into regs
        - Compute derived quantities
        - Produce measurement tokens

    Phase B — PROPOSE
        - Emit proposals with scores
        - Effects not applied yet (recorded in proposal buffers)

    Phase C — CHOOSE
        - Deterministically select one proposal per choice site
        - Uses only past trace + local seed

    Phase D — COMMIT
        - Apply chosen effects
        - All trace writes happen here
        - Emit witness tokens

This matches the semantics: no present-time branching, only staged resolution.
"""

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import List, Optional, Callable, Any


class Phase(IntEnum):
    """Execution phases within a tick."""
    IDLE = 0        # Between ticks
    READ = 1        # Phase A: Read trace, compute
    PROPOSE = 2     # Phase B: Emit proposals
    CHOOSE = 3      # Phase C: Select proposals
    COMMIT = 4      # Phase D: Apply effects, write witness


# Phase names for assembly
PHASE_NAMES = {
    Phase.IDLE: "IDLE",
    Phase.READ: "READ",
    Phase.PROPOSE: "PROPOSE",
    Phase.CHOOSE: "CHOOSE",
    Phase.COMMIT: "COMMIT",
}

PHASE_BY_NAME = {v: k for k, v in PHASE_NAMES.items()}


@dataclass
class PhaseViolation:
    """Record of a phase access violation."""
    phase: Phase
    attempted_op: str
    message: str


class PhaseController:
    """
    Controls phase transitions and enforces phase rules.

    Rules:
        - READ: Can read trace, can write to regs/scratch
        - PROPOSE: Can emit proposals with effects
        - CHOOSE: Can select from proposals
        - COMMIT: Can apply effects, write trace, emit witness

    Writes to trace are only allowed in COMMIT phase.
    """

    def __init__(self, strict: bool = True):
        self.current_phase: Phase = Phase.IDLE
        self.strict: bool = strict
        self.violations: List[PhaseViolation] = []
        self.tick: int = 0

        # Phase transition callbacks
        self._on_phase_enter: dict[Phase, List[Callable]] = {p: [] for p in Phase}
        self._on_phase_exit: dict[Phase, List[Callable]] = {p: [] for p in Phase}

    def begin_tick(self):
        """Start a new tick, enter READ phase."""
        self.tick += 1
        self._transition_to(Phase.READ)

    def end_tick(self):
        """End tick, return to IDLE."""
        self._transition_to(Phase.IDLE)

    def advance_phase(self) -> Phase:
        """Advance to next phase in sequence."""
        next_phases = {
            Phase.IDLE: Phase.READ,
            Phase.READ: Phase.PROPOSE,
            Phase.PROPOSE: Phase.CHOOSE,
            Phase.CHOOSE: Phase.COMMIT,
            Phase.COMMIT: Phase.IDLE,
        }
        next_phase = next_phases.get(self.current_phase, Phase.IDLE)
        self._transition_to(next_phase)
        return self.current_phase

    def set_phase(self, phase: Phase):
        """Explicitly set phase (for PHASE instruction)."""
        self._transition_to(phase)

    def _transition_to(self, phase: Phase):
        """Perform phase transition with callbacks."""
        old_phase = self.current_phase

        # Exit callbacks
        for callback in self._on_phase_exit[old_phase]:
            callback(old_phase)

        self.current_phase = phase

        # Enter callbacks
        for callback in self._on_phase_enter[phase]:
            callback(phase)

    # =========================================================================
    # Phase rule checks
    # =========================================================================

    def can_read_trace(self) -> bool:
        """Can read from trace in current phase?"""
        return self.current_phase in (Phase.READ, Phase.PROPOSE, Phase.CHOOSE, Phase.COMMIT)

    def can_write_trace(self) -> bool:
        """Can write to trace in current phase?"""
        return self.current_phase == Phase.COMMIT

    def can_emit_proposal(self) -> bool:
        """Can emit proposals in current phase?"""
        return self.current_phase == Phase.PROPOSE

    def can_choose(self) -> bool:
        """Can select proposals in current phase?"""
        return self.current_phase == Phase.CHOOSE

    def can_commit(self) -> bool:
        """Can commit effects in current phase?"""
        return self.current_phase == Phase.COMMIT

    # =========================================================================
    # Enforcement
    # =========================================================================

    def check_trace_read(self, op: str = "read") -> bool:
        """Check if trace read is allowed, record violation if not."""
        if self.can_read_trace():
            return True
        self._record_violation(op, "Trace read not allowed in IDLE phase")
        return not self.strict

    def check_trace_write(self, op: str = "write") -> bool:
        """Check if trace write is allowed, record violation if not."""
        if self.can_write_trace():
            return True
        self._record_violation(op, f"Trace write only allowed in COMMIT, current phase: {PHASE_NAMES[self.current_phase]}")
        return not self.strict

    def check_proposal(self, op: str = "propose") -> bool:
        """Check if proposal emission is allowed."""
        if self.can_emit_proposal():
            return True
        self._record_violation(op, f"Proposals only allowed in PROPOSE phase, current: {PHASE_NAMES[self.current_phase]}")
        return not self.strict

    def check_choose(self, op: str = "choose") -> bool:
        """Check if choice is allowed."""
        if self.can_choose():
            return True
        self._record_violation(op, f"Choose only allowed in CHOOSE phase, current: {PHASE_NAMES[self.current_phase]}")
        return not self.strict

    def check_commit(self, op: str = "commit") -> bool:
        """Check if commit is allowed."""
        if self.can_commit():
            return True
        self._record_violation(op, f"Commit only allowed in COMMIT phase, current: {PHASE_NAMES[self.current_phase]}")
        return not self.strict

    def _record_violation(self, op: str, message: str):
        """Record a phase violation."""
        self.violations.append(PhaseViolation(
            phase=self.current_phase,
            attempted_op=op,
            message=message
        ))

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_phase_enter(self, phase: Phase, callback: Callable):
        """Register callback for phase entry."""
        self._on_phase_enter[phase].append(callback)

    def on_phase_exit(self, phase: Phase, callback: Callable):
        """Register callback for phase exit."""
        self._on_phase_exit[phase].append(callback)

    # =========================================================================
    # State
    # =========================================================================

    def get_state(self) -> dict:
        """Get current phase controller state."""
        return {
            "tick": self.tick,
            "phase": PHASE_NAMES[self.current_phase],
            "strict": self.strict,
            "violations": len(self.violations),
        }

    def clear_violations(self):
        """Clear recorded violations."""
        self.violations = []


# ==============================================================================
# Phase sequence utilities
# ==============================================================================

def full_tick_sequence() -> List[Phase]:
    """Get the full sequence of phases for a tick."""
    return [Phase.READ, Phase.PROPOSE, Phase.CHOOSE, Phase.COMMIT]


def phase_from_name(name: str) -> Phase:
    """Get phase enum from name string."""
    return PHASE_BY_NAME.get(name.upper(), Phase.IDLE)


def phase_to_name(phase: Phase) -> str:
    """Get name string from phase enum."""
    return PHASE_NAMES.get(phase, "UNKNOWN")
