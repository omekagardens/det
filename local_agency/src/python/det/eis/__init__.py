"""
Existence Instruction Set (EIS)
===============================

Low-level instruction set that preserves agency semantics.
Designed for DET-OS where agency is the execution model.

Architecture:
    - Two-tier register file: scalar regs + trace regs
    - Fixed 32-bit instruction encoding (RISC-like)
    - Explicit tick phases: READ → PROPOSE → CHOOSE → COMMIT
    - Agency gates proposal eligibility, not memory access
    - Conservation enforced by verified effects

Register Model:
    - 16 scalar registers (R0-R15) - working values
    - 8 ref registers (H0-H7) - handles to trace memory
    - 8 token registers (T0-T7) - witness tokens

Instruction Encoding (32-bit):
    [opcode:8][dst:5][src0:5][src1:5][imm:9]

Phases:
    - READ: Load trace fields, compute derived quantities
    - PROPOSE: Emit proposals with scores (effects not applied)
    - CHOOSE: Select proposals deterministically
    - COMMIT: Apply effects, write witness tokens
"""

from .types import (
    # Numeric types
    F32, I32, Tok32, ValueType,
    # Reference types
    NodeRef, BondRef, FieldRef, PropRef, BufRef, ChoiceRef,
    FieldKind, AccessFlags,
    # Register types
    ScalarReg, RefReg, TokenReg,
    # Field identifiers
    NodeField, BondField,
    # Token values
    WitnessToken,
)

from .encoding import (
    Opcode, Instruction, encode_instruction, decode_instruction,
    InstructionFormat,
)

from .registers import (
    RegisterFile, LaneType,
)

from .memory import (
    TraceStore, ScratchMemory, ProposalBuffer, BoundaryBuffer,
    Proposal, Effect, EffectType,
)

from .phases import (
    Phase, PhaseController,
)

from .vm import (
    EISVM, Lane, ExecutionState,
)

from .assembler import (
    Assembler, assemble, disassemble,
)

from .creature_runner import (
    CreatureRunner, CreatureInstance, CreatureState,
    CompiledCreatureData, Channel, compile_creature_for_runner,
)

from .primitives import (
    PrimitiveRegistry, PrimitiveSpec, PrimitiveCall, PrimitiveResult,
    get_registry, call_primitive, list_primitives,
)

__version__ = "0.1.0"

__all__ = [
    # Types
    "F32", "I32", "Tok32", "ValueType",
    "NodeRef", "BondRef", "FieldRef", "PropRef", "BufRef", "ChoiceRef",
    "FieldKind", "AccessFlags",
    "ScalarReg", "RefReg", "TokenReg",
    "NodeField", "BondField",
    "WitnessToken",
    # Encoding
    "Opcode", "Instruction", "encode_instruction", "decode_instruction",
    "InstructionFormat",
    # Registers
    "RegisterFile", "LaneType",
    # Memory
    "TraceStore", "ScratchMemory", "ProposalBuffer", "BoundaryBuffer",
    "Proposal", "Effect", "EffectType",
    # Phases
    "Phase", "PhaseController",
    # VM
    "EISVM", "Lane", "ExecutionState",
    # Assembler
    "Assembler", "assemble", "disassemble",
    # Creature Runner
    "CreatureRunner", "CreatureInstance", "CreatureState",
    "CompiledCreatureData", "Channel", "compile_creature_for_runner",
    # Primitives
    "PrimitiveRegistry", "PrimitiveSpec", "PrimitiveCall", "PrimitiveResult",
    "get_registry", "call_primitive", "list_primitives",
]
