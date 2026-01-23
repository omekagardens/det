"""
EIS Register File
=================

Two-tier register file per executing lane:
    - Scalar registers (R0-R15): Fast, transient working values
    - Reference registers (H0-H7): Handles to trace memory
    - Token registers (T0-T7): Witness token intermediates

Register encoding in instructions (5 bits):
    0-15:  Scalar registers R0-R15
    16-23: Reference registers H0-H7
    24-31: Token registers T0-T7
"""

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, Optional, Union
import math

from .types import (
    F32, I32, Tok32, ValueType,
    NodeRef, BondRef, FieldRef, PropRef, BufRef, ChoiceRef,
    ScalarReg, RefReg, TokenReg, WitnessToken,
)


class LaneType(IntEnum):
    """Type of execution lane."""
    NODE = 0    # Creature kernel (local node state)
    BOND = 1    # Bond kernel (flux, grace, coherence)


@dataclass
class RegisterFile:
    """
    Register file for one execution lane.

    Layout:
        - 16 scalar registers (R0-R15): F32/I32 values
        - 8 reference registers (H0-H7): handles
        - 8 token registers (T0-T7): Tok32 values

    Unified 5-bit addressing:
        0-15:  R0-R15 (scalar)
        16-23: H0-H7 (reference)
        24-31: T0-T7 (token)
    """

    lane_type: LaneType = LaneType.NODE
    lane_id: int = 0

    # Scalar registers (working values)
    _scalars: list = field(default_factory=lambda: [0.0] * 16)
    _scalar_types: list = field(default_factory=lambda: [ValueType.F32] * 16)

    # Reference registers (handles)
    _refs: list = field(default_factory=lambda: [None] * 8)
    _ref_types: list = field(default_factory=lambda: [ValueType.VOID] * 8)

    # Token registers
    _tokens: list = field(default_factory=lambda: [WitnessToken.VOID] * 8)

    # Self reference (set at lane creation)
    self_node: Optional[NodeRef] = None
    self_bond: Optional[BondRef] = None

    def reset(self):
        """Reset all registers to default values."""
        self._scalars = [0.0] * 16
        self._scalar_types = [ValueType.F32] * 16
        self._refs = [None] * 8
        self._ref_types = [ValueType.VOID] * 8
        self._tokens = [WitnessToken.VOID] * 8

    # =========================================================================
    # Unified register access (5-bit addressing)
    # =========================================================================

    def read(self, reg: int) -> Any:
        """Read from unified register address (0-31)."""
        if 0 <= reg < 16:
            return self._scalars[reg]
        elif 16 <= reg < 24:
            return self._refs[reg - 16]
        elif 24 <= reg < 32:
            return self._tokens[reg - 24]
        else:
            raise ValueError(f"Invalid register address: {reg}")

    def write(self, reg: int, value: Any, type_tag: Optional[ValueType] = None):
        """Write to unified register address (0-31)."""
        if 0 <= reg < 16:
            self._scalars[reg] = float(value) if isinstance(value, (int, float)) else 0.0
            self._scalar_types[reg] = type_tag or ValueType.F32
        elif 16 <= reg < 24:
            self._refs[reg - 16] = value
            self._ref_types[reg - 16] = type_tag or self._infer_ref_type(value)
        elif 24 <= reg < 32:
            self._tokens[reg - 24] = int(value) if isinstance(value, int) else value
        else:
            raise ValueError(f"Invalid register address: {reg}")

    def read_type(self, reg: int) -> ValueType:
        """Get type tag for register."""
        if 0 <= reg < 16:
            return self._scalar_types[reg]
        elif 16 <= reg < 24:
            return self._ref_types[reg - 16]
        elif 24 <= reg < 32:
            return ValueType.TOK32
        else:
            return ValueType.VOID

    def _infer_ref_type(self, value: Any) -> ValueType:
        """Infer reference type from value."""
        if isinstance(value, NodeRef):
            return ValueType.NODE_REF
        elif isinstance(value, BondRef):
            return ValueType.BOND_REF
        elif isinstance(value, FieldRef):
            return ValueType.FIELD_REF
        elif isinstance(value, PropRef):
            return ValueType.PROP_REF
        elif isinstance(value, BufRef):
            return ValueType.BUF_REF
        elif isinstance(value, ChoiceRef):
            return ValueType.CHOICE_REF
        else:
            return ValueType.VOID

    # =========================================================================
    # Typed scalar access
    # =========================================================================

    def read_scalar(self, reg: ScalarReg) -> float:
        """Read scalar register as F32."""
        return self._scalars[reg.value]

    def write_scalar(self, reg: ScalarReg, value: float, is_int: bool = False):
        """Write scalar register."""
        self._scalars[reg.value] = value
        self._scalar_types[reg.value] = ValueType.I32 if is_int else ValueType.F32

    def read_i32(self, reg: ScalarReg) -> int:
        """Read scalar register as I32."""
        return int(self._scalars[reg.value])

    def write_i32(self, reg: ScalarReg, value: int):
        """Write scalar register as I32."""
        self._scalars[reg.value] = float(value)
        self._scalar_types[reg.value] = ValueType.I32

    # =========================================================================
    # Reference register access
    # =========================================================================

    def read_ref(self, reg: RefReg) -> Any:
        """Read reference register."""
        return self._refs[reg.value]

    def write_ref(self, reg: RefReg, value: Any):
        """Write reference register."""
        self._refs[reg.value] = value
        self._ref_types[reg.value] = self._infer_ref_type(value)

    def read_node_ref(self, reg: RefReg) -> Optional[NodeRef]:
        """Read reference register as NodeRef."""
        val = self._refs[reg.value]
        return val if isinstance(val, NodeRef) else None

    def read_bond_ref(self, reg: RefReg) -> Optional[BondRef]:
        """Read reference register as BondRef."""
        val = self._refs[reg.value]
        return val if isinstance(val, BondRef) else None

    # =========================================================================
    # Token register access
    # =========================================================================

    def read_token(self, reg: TokenReg) -> int:
        """Read token register."""
        return self._tokens[reg.value]

    def write_token(self, reg: TokenReg, value: int):
        """Write token register."""
        self._tokens[reg.value] = value

    def read_token_as(self, reg: TokenReg, token_enum: type) -> Any:
        """Read token register as specific enum type."""
        val = self._tokens[reg.value]
        try:
            return token_enum(val)
        except ValueError:
            return val

    # =========================================================================
    # Special registers
    # =========================================================================

    def get_self_node(self) -> Optional[NodeRef]:
        """Get self node reference (for node lanes)."""
        return self.self_node

    def get_self_bond(self) -> Optional[BondRef]:
        """Get self bond reference (for bond lanes)."""
        return self.self_bond

    # =========================================================================
    # Debug/inspection
    # =========================================================================

    def dump(self) -> Dict[str, Any]:
        """Dump register state for debugging."""
        return {
            "lane_type": self.lane_type.name,
            "lane_id": self.lane_id,
            "scalars": {
                f"R{i}": (self._scalars[i], self._scalar_types[i].name)
                for i in range(16) if self._scalars[i] != 0.0
            },
            "refs": {
                f"H{i}": (self._refs[i], self._ref_types[i].name)
                for i in range(8) if self._refs[i] is not None
            },
            "tokens": {
                f"T{i}": self._tokens[i]
                for i in range(8) if self._tokens[i] != WitnessToken.VOID
            },
            "self_node": self.self_node,
            "self_bond": self.self_bond,
        }

    def __repr__(self) -> str:
        active_scalars = sum(1 for v in self._scalars if v != 0.0)
        active_refs = sum(1 for v in self._refs if v is not None)
        active_tokens = sum(1 for v in self._tokens if v != WitnessToken.VOID)
        return f"RegisterFile({self.lane_type.name}, scalars={active_scalars}, refs={active_refs}, tokens={active_tokens})"


# ==============================================================================
# Register name utilities
# ==============================================================================

def reg_name(reg: int) -> str:
    """Get register name from unified address."""
    if 0 <= reg < 16:
        return f"R{reg}"
    elif 16 <= reg < 24:
        return f"H{reg - 16}"
    elif 24 <= reg < 32:
        return f"T{reg - 24}"
    else:
        return f"?{reg}"


def parse_reg_name(name: str) -> int:
    """Parse register name to unified address."""
    name = name.upper().strip()
    if name.startswith('R') and name[1:].isdigit():
        n = int(name[1:])
        if 0 <= n < 16:
            return n
    elif name.startswith('H') and name[1:].isdigit():
        n = int(name[1:])
        if 0 <= n < 8:
            return 16 + n
    elif name.startswith('T') and name[1:].isdigit():
        n = int(name[1:])
        if 0 <= n < 8:
            return 24 + n
    raise ValueError(f"Invalid register name: {name}")
