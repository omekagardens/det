"""
EIS Type System
===============

Fundamental types for the Existence Instruction Set.

Numeric Types:
    - F32: 32-bit float for DET fields (F, q, a, θ, etc.)
    - I32: 32-bit integer for indices, loop counts
    - Tok32: 32-bit token ID (enums, witness codes)

Reference Types (handles/capabilities):
    - NodeRef: 32-bit node ID
    - BondRef: 32-bit bond ID or packed (i, j, layer)
    - FieldRef: Packed descriptor {kind, field_id, access_flags}
    - PropRef: Index into proposal buffer
    - ChoiceRef: Chosen proposal index
    - BufRef: Boundary buffer handle

Register Types:
    - ScalarReg: R0-R15 (numeric working values)
    - RefReg: H0-H7 (handles to trace memory)
    - TokenReg: T0-T7 (witness token intermediates)
"""

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Union, NewType
import struct

# ==============================================================================
# Numeric Types
# ==============================================================================

# Type aliases for clarity
F32 = NewType('F32', float)
I32 = NewType('I32', int)
Tok32 = NewType('Tok32', int)


@dataclass(frozen=True)
class TypedValue:
    """A value with its type tag."""
    value: Union[float, int]
    type_tag: 'ValueType'

    def as_f32(self) -> float:
        return float(self.value)

    def as_i32(self) -> int:
        return int(self.value)

    def as_tok32(self) -> int:
        return int(self.value) & 0xFFFFFFFF


class ValueType(IntEnum):
    """Type tags for values in registers."""
    F32 = 0
    I32 = 1
    TOK32 = 2
    NODE_REF = 3
    BOND_REF = 4
    FIELD_REF = 5
    PROP_REF = 6
    BUF_REF = 7
    CHOICE_REF = 8
    VOID = 15


# ==============================================================================
# Reference Types (Handles/Capabilities)
# ==============================================================================

@dataclass(frozen=True)
class NodeRef:
    """
    Reference to a node in the trace store.

    32-bit node ID. Provides capability to access node fields.
    """
    node_id: int

    def __int__(self) -> int:
        return self.node_id & 0xFFFFFFFF

    def is_valid(self) -> bool:
        return self.node_id >= 0

    @staticmethod
    def invalid() -> 'NodeRef':
        return NodeRef(-1)


@dataclass(frozen=True)
class BondRef:
    """
    Reference to a bond in the trace store.

    Can be either:
    - Simple 32-bit bond ID
    - Packed (node_i:12, node_j:12, layer:8) for direct addressing
    """
    bond_id: int
    # Optional unpacked form
    node_i: int = -1
    node_j: int = -1
    layer: int = 0

    def __int__(self) -> int:
        return self.bond_id & 0xFFFFFFFF

    def is_packed(self) -> bool:
        return self.node_i >= 0 and self.node_j >= 0

    @staticmethod
    def pack(node_i: int, node_j: int, layer: int = 0) -> 'BondRef':
        """Pack node indices into bond ID."""
        bond_id = ((node_i & 0xFFF) << 20) | ((node_j & 0xFFF) << 8) | (layer & 0xFF)
        return BondRef(bond_id, node_i, node_j, layer)

    @staticmethod
    def unpack(bond_id: int) -> 'BondRef':
        """Unpack bond ID into node indices."""
        node_i = (bond_id >> 20) & 0xFFF
        node_j = (bond_id >> 8) & 0xFFF
        layer = bond_id & 0xFF
        return BondRef(bond_id, node_i, node_j, layer)


class FieldKind(IntEnum):
    """Kind of field being referenced."""
    NODE = 0
    BOND = 1


class AccessFlags(IntEnum):
    """Access permission flags for field references."""
    READ = 1
    WRITE = 2
    READ_WRITE = 3
    COMMIT_ONLY = 4  # Write only in COMMIT phase


@dataclass(frozen=True)
class FieldRef:
    """
    Reference to a specific field in trace memory.

    Packed descriptor: {kind:2, field_id:14, access:4, reserved:12}
    """
    kind: FieldKind
    field_id: int
    access: AccessFlags

    def pack(self) -> int:
        """Pack into 32-bit descriptor."""
        return ((self.kind & 0x3) << 30) | \
               ((self.field_id & 0x3FFF) << 16) | \
               ((self.access & 0xF) << 12)

    @staticmethod
    def unpack(packed: int) -> 'FieldRef':
        """Unpack from 32-bit descriptor."""
        kind = FieldKind((packed >> 30) & 0x3)
        field_id = (packed >> 16) & 0x3FFF
        access = AccessFlags((packed >> 12) & 0xF)
        return FieldRef(kind, field_id, access)


@dataclass(frozen=True)
class PropRef:
    """Reference to a proposal in the proposal buffer."""
    index: int

    def __int__(self) -> int:
        return self.index & 0xFFFF


@dataclass(frozen=True)
class ChoiceRef:
    """Reference to a chosen proposal."""
    choice_index: int
    proposal_index: int  # Which proposal was selected

    def __int__(self) -> int:
        return ((self.choice_index & 0xFFFF) << 16) | (self.proposal_index & 0xFFFF)


@dataclass(frozen=True)
class BufRef:
    """Reference to a boundary buffer (for I/O)."""
    buffer_id: int
    offset: int = 0

    def __int__(self) -> int:
        return ((self.buffer_id & 0xFFFF) << 16) | (self.offset & 0xFFFF)


# ==============================================================================
# Node and Bond Field Identifiers
# ==============================================================================

class NodeField(IntEnum):
    """Field identifiers for node state in trace."""
    F = 0       # Resource
    Q = 1       # Structural debt
    A = 2       # Agency [0,1]
    THETA = 3   # Phase [0, 2π]
    R = 4       # Auxiliary state
    K = 5       # Tick counter / complexity
    FLAGS = 6   # Node flags
    # Token registers (8 slots)
    TOK0 = 8
    TOK1 = 9
    TOK2 = 10
    TOK3 = 11
    TOK4 = 12
    TOK5 = 13
    TOK6 = 14
    TOK7 = 15
    # Extended state
    SIGMA = 16  # Processing rate
    P = 17      # Presence (computed)
    TAU = 18    # Proper time
    H = 19      # Coordination load


class BondField(IntEnum):
    """Field identifiers for bond state in trace."""
    C = 0       # Coherence [0,1]
    PI = 1      # Momentum
    SIGMA = 2   # Conductivity
    FLAGS = 3   # Bond flags
    # Bond token slots
    TOK0 = 8
    TOK1 = 9
    # Extended
    ALPHA_C = 16    # Coherence charging rate
    LAMBDA_C = 17   # Coherence decay rate
    SLIP_TH = 18    # Phase slip threshold


# ==============================================================================
# Register Types
# ==============================================================================

class ScalarReg(IntEnum):
    """
    Scalar registers R0-R15.

    Hold intermediate F32/I32 values during tick phases.
    These are working values, not truth.
    """
    R0 = 0
    R1 = 1
    R2 = 2
    R3 = 3
    R4 = 4
    R5 = 5
    R6 = 6
    R7 = 7
    R8 = 8
    R9 = 9
    R10 = 10
    R11 = 11
    R12 = 12
    R13 = 13
    R14 = 14
    R15 = 15


class RefReg(IntEnum):
    """
    Reference registers H0-H7.

    Hold handles to trace memory (NodeRef, BondRef, FieldRef, etc.)
    """
    H0 = 0
    H1 = 1
    H2 = 2
    H3 = 3
    H4 = 4
    H5 = 5
    H6 = 6
    H7 = 7


class TokenReg(IntEnum):
    """
    Token registers T0-T7.

    Hold witness tokens (Tok32) as intermediates.
    """
    T0 = 0
    T1 = 1
    T2 = 2
    T3 = 3
    T4 = 4
    T5 = 5
    T6 = 6
    T7 = 7


# ==============================================================================
# Witness Token Values (Predefined)
# ==============================================================================

class WitnessToken(IntEnum):
    """Predefined witness token values."""
    # Comparison results
    LT = 0x0001
    EQ = 0x0002
    GT = 0x0003

    # Reconciliation results
    EQ_OK = 0x0010
    EQ_FAIL = 0x0011
    EQ_REFUSE = 0x0012

    # Transfer results
    XFER_OK = 0x0020
    XFER_PARTIAL = 0x0021
    XFER_BLOCKED = 0x0022

    # Write results
    WRITE_OK = 0x0030
    WRITE_REFUSED = 0x0031
    WRITE_PARTIAL = 0x0032

    # Proposal results
    PROP_ACCEPTED = 0x0040
    PROP_REJECTED = 0x0041

    # Grace results
    GRACE_OFFERED = 0x0050
    GRACE_ACCEPTED = 0x0051
    GRACE_DECLINED = 0x0052

    # Phase markers
    PHASE_READ = 0x0100
    PHASE_PROPOSE = 0x0101
    PHASE_CHOOSE = 0x0102
    PHASE_COMMIT = 0x0103

    # Special
    VOID = 0x0000
    TRUE = 0xFFFE
    FALSE = 0xFFFF
