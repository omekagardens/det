"""
EIS Memory Model
================

Memory regions:
    1. Trace Store (authoritative, falsifiable)
       - Node arrays: F, q, a, θ, r, k, flags, token regs
       - Bond arrays: C, π, σ, bond flags
       - Token store (witness logs)

    2. Scratch / Working Memory (ephemeral)
       - Per-lane scratch
       - Per-bond scratch (offers, accumulators)

    3. Proposal Buffer (ephemeral → committed)
       - Per-lane list of proposals: {score, effect_id, effect_args}

    4. Boundary Buffers (readout-only by default)
       - Byte sinks (stdout)
       - Sensors (ingest must be explicit)

Access Model:
    - READ of local fields always allowed (doesn't coerce)
    - WRITE to trace only in COMMIT phase via verified effects
    - Agency gates proposal eligibility, not memory access
"""

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Tuple
import math

from .types import (
    NodeRef, BondRef, FieldRef, NodeField, BondField,
    WitnessToken, Tok32,
)


# ==============================================================================
# Effect Types (verified effects for COMMIT)
# ==============================================================================

class EffectType(IntEnum):
    """
    Verified effect types for COMMIT phase.

    Effects must be from this set to ensure conservation and consistency.
    """
    # Resource effects
    XFER_F = 0x01           # Antisymmetric F transfer
    ADD_F = 0x02            # Add F to node (injection)
    SUB_F = 0x03            # Subtract F from node (cost)

    # Token effects
    SET_TOKEN = 0x10        # Set token register value
    APPEND_TOKEN = 0x11     # Append to token tape

    # Node state effects
    SET_NODE_FIELD = 0x20   # Set node field value
    INC_NODE_FIELD = 0x21   # Increment node field

    # Bond state effects
    SET_BOND_FIELD = 0x30   # Set bond field value
    UPDATE_C_PI = 0x31      # Update coherence and momentum

    # Boundary effects
    APPEND_BYTE = 0x40      # Append byte to output buffer
    FLUSH_BUF = 0x41        # Flush buffer

    # Grace effects
    GRACE_XFER = 0x50       # Grace transfer (antisymmetric)

    # Special
    NOP = 0x00              # No operation
    WITNESS = 0xFF          # Emit witness token


@dataclass
class Effect:
    """
    A verified effect to be applied in COMMIT phase.

    Effects are data, not control flow. They describe
    what will happen when committed.
    """
    effect_type: EffectType
    args: Tuple[Any, ...] = ()

    # For antisymmetric effects
    src_node: Optional[NodeRef] = None
    dst_node: Optional[NodeRef] = None
    amount: float = 0.0

    # For field effects
    target_ref: Optional[Any] = None  # NodeRef or BondRef
    field_id: int = 0
    value: Any = None


@dataclass
class Proposal:
    """
    A proposal with score and effects.

    Proposals are data in buffers. CHOOSE selects one,
    COMMIT applies its effects.
    """
    index: int
    score: float = 0.0
    effects: List[Effect] = field(default_factory=list)
    name: str = ""
    committed: bool = False

    def add_effect(self, effect: Effect):
        """Add an effect to this proposal."""
        self.effects.append(effect)


# ==============================================================================
# Trace Store (authoritative truth)
# ==============================================================================

@dataclass
class NodeState:
    """State for a single node in trace."""
    F: float = 100.0        # Resource
    q: float = 0.0          # Structural debt
    a: float = 1.0          # Agency [0,1]
    theta: float = 0.0      # Phase [0, 2π]
    r: float = 0.0          # Auxiliary state
    k: int = 0              # Tick counter / complexity
    flags: int = 0          # Node flags

    sigma: float = 1.0      # Processing rate
    P: float = 1.0          # Presence (computed)
    tau: float = 0.0        # Proper time
    H: float = 0.0          # Coordination load

    # Token registers (8 slots)
    tokens: List[int] = field(default_factory=lambda: [0] * 8)

    def get_field(self, field_id: NodeField) -> Any:
        """Get field value by ID."""
        field_map = {
            NodeField.F: self.F,
            NodeField.Q: self.q,
            NodeField.A: self.a,
            NodeField.THETA: self.theta,
            NodeField.R: self.r,
            NodeField.K: self.k,
            NodeField.FLAGS: self.flags,
            NodeField.SIGMA: self.sigma,
            NodeField.P: self.P,
            NodeField.TAU: self.tau,
            NodeField.H: self.H,
        }
        if field_id in field_map:
            return field_map[field_id]
        if NodeField.TOK0 <= field_id <= NodeField.TOK7:
            return self.tokens[field_id - NodeField.TOK0]
        return 0.0

    def set_field(self, field_id: NodeField, value: Any):
        """Set field value by ID."""
        if field_id == NodeField.F:
            self.F = max(0.0, float(value))
        elif field_id == NodeField.Q:
            self.q = max(0.0, float(value))
        elif field_id == NodeField.A:
            self.a = max(0.0, min(1.0, float(value)))
        elif field_id == NodeField.THETA:
            self.theta = float(value) % (2 * math.pi)
        elif field_id == NodeField.R:
            self.r = float(value)
        elif field_id == NodeField.K:
            self.k = int(value)
        elif field_id == NodeField.FLAGS:
            self.flags = int(value)
        elif field_id == NodeField.SIGMA:
            self.sigma = max(0.0, float(value))
        elif field_id == NodeField.TAU:
            self.tau = float(value)
        elif field_id == NodeField.H:
            self.H = float(value)
        elif NodeField.TOK0 <= field_id <= NodeField.TOK7:
            self.tokens[field_id - NodeField.TOK0] = int(value)


@dataclass
class BondState:
    """State for a single bond in trace."""
    C: float = 0.5          # Coherence [0,1]
    pi: float = 0.0         # Momentum
    sigma: float = 1.0      # Conductivity
    flags: int = 0          # Bond flags

    alpha_C: float = 0.15   # Coherence charging rate
    lambda_C: float = 0.02  # Coherence decay rate
    slip_th: float = 0.3    # Phase slip threshold

    # Token slots
    tokens: List[int] = field(default_factory=lambda: [0] * 2)

    def get_field(self, field_id: BondField) -> Any:
        """Get field value by ID."""
        field_map = {
            BondField.C: self.C,
            BondField.PI: self.pi,
            BondField.SIGMA: self.sigma,
            BondField.FLAGS: self.flags,
            BondField.ALPHA_C: self.alpha_C,
            BondField.LAMBDA_C: self.lambda_C,
            BondField.SLIP_TH: self.slip_th,
        }
        if field_id in field_map:
            return field_map[field_id]
        if BondField.TOK0 <= field_id <= BondField.TOK1:
            return self.tokens[field_id - BondField.TOK0]
        return 0.0

    def set_field(self, field_id: BondField, value: Any):
        """Set field value by ID."""
        if field_id == BondField.C:
            self.C = max(0.0, min(1.0, float(value)))
        elif field_id == BondField.PI:
            self.pi = float(value)
        elif field_id == BondField.SIGMA:
            self.sigma = max(0.0, float(value))
        elif field_id == BondField.FLAGS:
            self.flags = int(value)
        elif BondField.TOK0 <= field_id <= BondField.TOK1:
            self.tokens[field_id - BondField.TOK0] = int(value)


class TraceStore:
    """
    Authoritative trace memory.

    Contains all falsifiable state: nodes, bonds, tokens.
    Writes only allowed in COMMIT phase via verified effects.
    """

    def __init__(self, num_nodes: int = 256, max_bonds: int = 1024):
        self.num_nodes = num_nodes
        self.max_bonds = max_bonds

        # Node state arrays
        self.nodes: List[NodeState] = [NodeState() for _ in range(num_nodes)]

        # Bond state (sparse representation)
        self.bonds: Dict[Tuple[int, int], BondState] = {}

        # Adjacency lists for locality enforcement
        self.adjacency: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}

        # Witness log (append-only)
        self.witness_log: List[Tuple[int, int, int]] = []  # (tick, node_id, token)

        # Current tick
        self.tick: int = 0

    def read_node(self, node_ref: NodeRef, field_id: NodeField) -> Any:
        """Read node field (always allowed)."""
        if 0 <= node_ref.node_id < self.num_nodes:
            return self.nodes[node_ref.node_id].get_field(field_id)
        return 0.0

    def read_bond(self, bond_ref: BondRef, field_id: BondField) -> Any:
        """Read bond field (always allowed)."""
        key = (bond_ref.node_i, bond_ref.node_j) if bond_ref.is_packed() else None
        if key and key in self.bonds:
            return self.bonds[key].get_field(field_id)
        return 0.0

    def write_node(self, node_ref: NodeRef, field_id: NodeField, value: Any):
        """Write node field (COMMIT only - caller must verify phase)."""
        if 0 <= node_ref.node_id < self.num_nodes:
            self.nodes[node_ref.node_id].set_field(field_id, value)

    def write_bond(self, bond_ref: BondRef, field_id: BondField, value: Any):
        """Write bond field (COMMIT only - caller must verify phase)."""
        if bond_ref.is_packed():
            key = (bond_ref.node_i, bond_ref.node_j)
            if key not in self.bonds:
                self.bonds[key] = BondState()
            self.bonds[key].set_field(field_id, value)

    def apply_xfer(self, src: NodeRef, dst: NodeRef, amount: float) -> Tuple[float, int]:
        """
        Apply antisymmetric transfer (conservation law).

        Returns (actual_amount, witness_token).
        """
        if not (0 <= src.node_id < self.num_nodes and 0 <= dst.node_id < self.num_nodes):
            return 0.0, WitnessToken.XFER_BLOCKED

        src_node = self.nodes[src.node_id]
        dst_node = self.nodes[dst.node_id]

        # Compute actual transfer (clamped to available)
        actual = min(amount, src_node.F)
        if actual <= 0:
            return 0.0, WitnessToken.XFER_BLOCKED

        # Apply antisymmetric update
        src_node.F -= actual
        dst_node.F += actual

        if actual < amount:
            return actual, WitnessToken.XFER_PARTIAL
        return actual, WitnessToken.XFER_OK

    def add_bond(self, node_i: int, node_j: int) -> BondRef:
        """Add a bond between two nodes."""
        key = (node_i, node_j)
        if key not in self.bonds:
            self.bonds[key] = BondState()
            self.adjacency[node_i].append(node_j)
            self.adjacency[node_j].append(node_i)
        return BondRef.pack(node_i, node_j)

    def get_neighbors(self, node_id: int) -> List[int]:
        """Get neighbor node IDs."""
        return self.adjacency.get(node_id, [])

    def emit_witness(self, node_id: int, token: int):
        """Append witness token to log."""
        self.witness_log.append((self.tick, node_id, token))

    def advance_tick(self):
        """Advance global tick counter."""
        self.tick += 1


# ==============================================================================
# Scratch Memory (ephemeral)
# ==============================================================================

class ScratchMemory:
    """
    Ephemeral working memory per lane.

    Cleared at the start of each tick.
    """

    def __init__(self, size: int = 256):
        self.size = size
        self.data: List[float] = [0.0] * size

    def read(self, addr: int) -> float:
        if 0 <= addr < self.size:
            return self.data[addr]
        return 0.0

    def write(self, addr: int, value: float):
        if 0 <= addr < self.size:
            self.data[addr] = value

    def clear(self):
        self.data = [0.0] * self.size


# ==============================================================================
# Proposal Buffer
# ==============================================================================

class ProposalBuffer:
    """
    Buffer for proposals within a tick.

    Proposals are added during PROPOSE phase,
    selected during CHOOSE phase,
    and applied during COMMIT phase.
    """

    def __init__(self, max_proposals: int = 32):
        self.max_proposals = max_proposals
        self.proposals: List[Proposal] = []
        self.choices: Dict[int, int] = {}  # choice_id -> proposal_index

    def begin_proposal(self, name: str = "") -> Proposal:
        """Begin a new proposal."""
        if len(self.proposals) >= self.max_proposals:
            raise RuntimeError("Proposal buffer full")
        prop = Proposal(index=len(self.proposals), name=name)
        self.proposals.append(prop)
        return prop

    def get_proposal(self, index: int) -> Optional[Proposal]:
        """Get proposal by index."""
        if 0 <= index < len(self.proposals):
            return self.proposals[index]
        return None

    def choose(self, choice_id: int, proposal_indices: List[int],
               decisiveness: float = 1.0, seed: int = 0) -> int:
        """
        Choose from proposals based on scores.

        Higher decisiveness = more likely to pick highest score.
        """
        if not proposal_indices:
            return -1

        # Get scores
        scores = []
        for idx in proposal_indices:
            prop = self.get_proposal(idx)
            scores.append(prop.score if prop else 0.0)

        # Deterministic selection based on decisiveness and seed
        if decisiveness >= 1.0:
            # Always pick highest score
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
        else:
            # Weighted selection
            import random
            rng = random.Random(seed)
            weights = [s ** decisiveness for s in scores]
            total = sum(weights)
            if total > 0:
                r = rng.random() * total
                cumsum = 0.0
                best_idx = len(weights) - 1
                for i, w in enumerate(weights):
                    cumsum += w
                    if r <= cumsum:
                        best_idx = i
                        break
            else:
                best_idx = 0

        selected = proposal_indices[best_idx]
        self.choices[choice_id] = selected
        return selected

    def get_chosen(self, choice_id: int) -> Optional[Proposal]:
        """Get the chosen proposal for a choice ID."""
        if choice_id in self.choices:
            return self.get_proposal(self.choices[choice_id])
        return None

    def clear(self):
        """Clear all proposals (start of tick)."""
        self.proposals = []
        self.choices = {}


# ==============================================================================
# Boundary Buffer (I/O)
# ==============================================================================

class BoundaryBuffer:
    """
    Boundary buffer for I/O operations.

    Output buffers: append bytes during COMMIT
    Input buffers: read-only, populated externally
    """

    def __init__(self, buffer_id: int, size: int = 4096, is_output: bool = True):
        self.buffer_id = buffer_id
        self.size = size
        self.is_output = is_output
        self.data: bytearray = bytearray(size)
        self.write_pos: int = 0
        self.read_pos: int = 0

    def append(self, byte_val: int):
        """Append byte to output buffer."""
        if self.is_output and self.write_pos < self.size:
            self.data[self.write_pos] = byte_val & 0xFF
            self.write_pos += 1

    def read_byte(self) -> int:
        """Read byte from input buffer."""
        if not self.is_output and self.read_pos < self.write_pos:
            val = self.data[self.read_pos]
            self.read_pos += 1
            return val
        return 0

    def flush(self) -> bytes:
        """Flush output buffer, return contents."""
        if self.is_output:
            result = bytes(self.data[:self.write_pos])
            self.write_pos = 0
            return result
        return b''

    def fill(self, data: bytes):
        """Fill input buffer with data."""
        if not self.is_output:
            n = min(len(data), self.size)
            self.data[:n] = data[:n]
            self.write_pos = n
            self.read_pos = 0
