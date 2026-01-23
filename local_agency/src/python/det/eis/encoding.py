"""
EIS Instruction Encoding
========================

Fixed 32-bit instruction encoding (RISC-like).

Format: [opcode:8][dst:5][src0:5][src1:5][imm:9]

    - opcode: 8 bits (256 possible opcodes)
    - dst: 5 bits (32 register slots)
    - src0: 5 bits (32 register slots)
    - src1: 5 bits (32 register slots)
    - imm: 9 bits (small immediate or extension indicator)

Register encoding (5 bits):
    - 0-15: Scalar registers R0-R15
    - 16-23: Reference registers H0-H7
    - 24-31: Token registers T0-T7

Extension words (32-bit) follow instruction for:
    - Large immediates (>9 bits)
    - Field descriptors
    - Extended addresses
"""

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Optional, List, Tuple
import struct


# ==============================================================================
# Opcode Definitions
# ==============================================================================

class Opcode(IntEnum):
    """
    EIS Instruction Opcodes (8-bit).

    Organized by category:
    - 0x00-0x0F: Phase control and flow
    - 0x10-0x1F: Load operations
    - 0x20-0x2F: Store operations (staged)
    - 0x30-0x4F: Arithmetic/math
    - 0x50-0x5F: Comparison and token ops
    - 0x60-0x6F: Proposal operations
    - 0x70-0x7F: Choose/commit
    - 0x80-0x8F: Reference/handle ops
    - 0x90-0x9F: Conservation primitives
    - 0xA0-0xAF: Grace protocol
    - 0xF0-0xFF: System/debug
    """

    # === Phase Control (0x00-0x0F) ===
    NOP = 0x00
    PHASE = 0x01        # Set current phase: PHASE imm
    HALT = 0x02         # Stop execution
    YIELD = 0x03        # Yield to scheduler
    FENCE = 0x04        # Memory fence
    TICK = 0x05         # Advance tick counter

    # === Load Operations (0x10-0x1F) ===
    LDI = 0x10          # Load immediate: LDI dst, imm
    LDI_EXT = 0x11      # Load extended immediate (next word)
    LDN = 0x12          # Load node field: LDN dst, nodeRef, fieldId
    LDB = 0x13          # Load bond field: LDB dst, bondRef, fieldId
    LDNB = 0x14         # Load neighbor field: LDNB dst, nodeRef, neighborIdx, fieldId
    LDT = 0x15          # Load token: LDT dst, tokRef
    LDR = 0x16          # Load from ref: LDR dst, refReg
    LDBUF = 0x17        # Load from buffer: LDBUF dst, bufRef, offset

    # === Store Operations (0x20-0x2F) - Staged for COMMIT ===
    ST_TOK = 0x20       # Stage token write: ST_TOK tokRef, src
    ST_NODE = 0x21      # Stage node field write: ST_NODE nodeRef, fieldId, src
    ST_BOND = 0x22      # Stage bond field write: ST_BOND bondRef, fieldId, src
    ST_BUF = 0x23       # Stage buffer write: ST_BUF bufRef, offset, src

    # === Arithmetic (0x30-0x3F) ===
    ADD = 0x30          # ADD dst, src0, src1
    SUB = 0x31          # SUB dst, src0, src1
    MUL = 0x32          # MUL dst, src0, src1
    DIV = 0x33          # DIV dst, src0, src1
    MAD = 0x34          # Multiply-add: MAD dst, src0, src1, src2 (needs ext)
    NEG = 0x35          # NEG dst, src
    ABS = 0x36          # ABS dst, src
    MOD = 0x37          # MOD dst, src0, src1

    # === Math Functions (0x40-0x4F) ===
    SQRT = 0x40         # SQRT dst, src
    EXP = 0x41          # EXP dst, src
    LOG = 0x42          # LOG dst, src
    SIN = 0x43          # SIN dst, src
    COS = 0x44          # COS dst, src
    MIN = 0x45          # MIN dst, src0, src1
    MAX = 0x46          # MAX dst, src0, src1
    CLAMP = 0x47        # CLAMP dst, src, lo, hi (needs ext)
    RELU = 0x48         # RELU dst, src (max(0, src))
    SIGMOID = 0x49      # SIGMOID dst, src

    # === Comparison and Token Ops (0x50-0x5F) ===
    CMP = 0x50          # Compare: CMP dstTok, src0, src1 → LT/EQ/GT
    CMP_EPS = 0x51      # Compare with epsilon: CMP_EPS dstTok, src0, src1, eps
    TEQ = 0x52          # Token equality: TEQ dstTok, tok0, tok1
    TNE = 0x53          # Token not equal: TNE dstTok, tok0, tok1
    TMOV = 0x54         # Token move: TMOV dstTok, srcTok
    TSET = 0x55         # Set token: TSET dstTok, imm
    TGET = 0x56         # Get token value to scalar: TGET dst, srcTok

    # === Proposal Operations (0x60-0x6F) ===
    PROP_BEGIN = 0x60   # Begin proposal: PROP_BEGIN propRef
    PROP_SCORE = 0x61   # Set score: PROP_SCORE propRef, scoreReg
    PROP_EFFECT = 0x62  # Add effect: PROP_EFFECT propRef, effectId, args...
    PROP_END = 0x63     # End proposal: PROP_END propRef
    PROP_LIST = 0x64    # Get proposal list ref: PROP_LIST dst

    # === Choose/Commit (0x70-0x7F) ===
    CHOOSE = 0x70       # Choose: CHOOSE choiceRef, propListRef, decisiveness, seed
    COMMIT = 0x71       # Commit: COMMIT choiceRef
    COMMIT_ALL = 0x72   # Commit all pending: COMMIT_ALL
    WITNESS = 0x73      # Write witness: WITNESS tokRef, value
    ABORT = 0x74        # Abort proposals: ABORT

    # === Reference/Handle Ops (0x80-0x8F) ===
    MKNODE = 0x80       # Make node ref: MKNODE dst, nodeId
    MKBOND = 0x81       # Make bond ref: MKBOND dst, nodeI, nodeJ
    MKFIELD = 0x82      # Make field ref: MKFIELD dst, kind, fieldId
    MKPROP = 0x83       # Make prop ref: MKPROP dst, index
    MKBUF = 0x84        # Make buffer ref: MKBUF dst, bufId
    GETSELF = 0x85      # Get self node ref: GETSELF dst
    GETBOND = 0x86      # Get current bond ref: GETBOND dst
    NEIGHBOR = 0x87     # Get neighbor: NEIGHBOR dst, nodeRef, neighborIdx

    # === Conservation Primitives (0x90-0x9F) ===
    XFER = 0x90         # Transfer: XFER srcNode, dstNode, amount → antisymmetric
    DIFFUSE = 0x91      # Diffuse: DIFFUSE nodeA, nodeB, sigma
    DISTINCT = 0x92     # Create distinct: DISTINCT dst0, dst1
    COALESCE = 0x93     # Merge: COALESCE src0, src1, dst

    # === Grace Protocol (0xA0-0xAF) ===
    GRACE_NEED = 0xA0   # Compute need: GRACE_NEED dst, nodeRef
    GRACE_EXCESS = 0xA1 # Compute excess: GRACE_EXCESS dst, nodeRef
    GRACE_OFFER = 0xA2  # Stage offer: GRACE_OFFER bondRef, amount
    GRACE_ACCEPT = 0xA3 # Accept offer: GRACE_ACCEPT bondRef, scale
    GRACE_COMMIT = 0xA4 # Commit grace: GRACE_COMMIT bondRef

    # === Branching (limited - past-token only) ===
    BR_TOK = 0xB0       # Branch on token: BR_TOK tokReg, target, expectedTok
    BR_PHASE = 0xB1     # Branch on phase: BR_PHASE target, expectedPhase
    CALL = 0xB2         # Call kernel: CALL kernelId
    RET = 0xB3          # Return from kernel

    # === System/Debug (0xF0-0xFF) ===
    DEBUG = 0xF0        # Debug breakpoint
    TRACE = 0xF1        # Emit trace event
    ASSERT = 0xF2       # Assert condition
    PRINT = 0xF3        # Debug print
    STATS = 0xF4        # Emit statistics
    INVALID = 0xFF      # Invalid opcode (trap)


# ==============================================================================
# Instruction Format
# ==============================================================================

class InstructionFormat(IntEnum):
    """Instruction format types based on operand needs."""
    R = 0       # Register only: op dst, src0, src1
    RI = 1      # Register + immediate: op dst, src0, imm
    I = 2       # Immediate only: op imm
    RR = 3      # Two registers: op dst, src
    RRI = 4     # Two regs + immediate: op dst, src, imm
    EXT = 5     # Extended (uses next word)
    NONE = 6    # No operands


# Opcode to format mapping
OPCODE_FORMAT = {
    # Phase control
    Opcode.NOP: InstructionFormat.NONE,
    Opcode.PHASE: InstructionFormat.I,
    Opcode.HALT: InstructionFormat.NONE,
    Opcode.YIELD: InstructionFormat.NONE,
    Opcode.FENCE: InstructionFormat.NONE,
    Opcode.TICK: InstructionFormat.NONE,

    # Loads
    Opcode.LDI: InstructionFormat.RI,
    Opcode.LDI_EXT: InstructionFormat.EXT,
    Opcode.LDN: InstructionFormat.R,
    Opcode.LDB: InstructionFormat.R,
    Opcode.LDNB: InstructionFormat.EXT,
    Opcode.LDT: InstructionFormat.RR,
    Opcode.LDR: InstructionFormat.RR,

    # Stores
    Opcode.ST_TOK: InstructionFormat.RR,
    Opcode.ST_NODE: InstructionFormat.R,
    Opcode.ST_BOND: InstructionFormat.R,

    # Arithmetic
    Opcode.ADD: InstructionFormat.R,
    Opcode.SUB: InstructionFormat.R,
    Opcode.MUL: InstructionFormat.R,
    Opcode.DIV: InstructionFormat.R,
    Opcode.NEG: InstructionFormat.RR,
    Opcode.ABS: InstructionFormat.RR,

    # Math
    Opcode.SQRT: InstructionFormat.RR,
    Opcode.MIN: InstructionFormat.R,
    Opcode.MAX: InstructionFormat.R,
    Opcode.RELU: InstructionFormat.RR,

    # Comparison
    Opcode.CMP: InstructionFormat.R,
    Opcode.CMP_EPS: InstructionFormat.EXT,
    Opcode.TMOV: InstructionFormat.RR,
    Opcode.TSET: InstructionFormat.RI,

    # Proposals
    Opcode.PROP_BEGIN: InstructionFormat.RR,
    Opcode.PROP_SCORE: InstructionFormat.RR,
    Opcode.PROP_EFFECT: InstructionFormat.EXT,
    Opcode.PROP_END: InstructionFormat.RR,

    # Choose/Commit
    Opcode.CHOOSE: InstructionFormat.EXT,
    Opcode.COMMIT: InstructionFormat.RR,
    Opcode.COMMIT_ALL: InstructionFormat.NONE,
    Opcode.WITNESS: InstructionFormat.RR,

    # References
    Opcode.MKNODE: InstructionFormat.RI,
    Opcode.MKBOND: InstructionFormat.R,
    Opcode.GETSELF: InstructionFormat.RR,

    # Conservation
    Opcode.XFER: InstructionFormat.R,
    Opcode.DIFFUSE: InstructionFormat.R,
    Opcode.DISTINCT: InstructionFormat.RR,

    # Grace
    Opcode.GRACE_OFFER: InstructionFormat.RR,
    Opcode.GRACE_COMMIT: InstructionFormat.RR,

    # Branch
    Opcode.BR_TOK: InstructionFormat.EXT,
    Opcode.CALL: InstructionFormat.I,
    Opcode.RET: InstructionFormat.NONE,

    # Debug
    Opcode.DEBUG: InstructionFormat.NONE,
    Opcode.TRACE: InstructionFormat.I,
}


# ==============================================================================
# Instruction Data Structure
# ==============================================================================

@dataclass
class Instruction:
    """
    Decoded EIS instruction.

    Fields:
        opcode: 8-bit opcode
        dst: 5-bit destination register
        src0: 5-bit source register 0
        src1: 5-bit source register 1
        imm: 9-bit immediate value
        ext: Optional 32-bit extension word
    """
    opcode: Opcode
    dst: int = 0
    src0: int = 0
    src1: int = 0
    imm: int = 0
    ext: Optional[int] = None

    def __post_init__(self):
        # Validate ranges
        assert 0 <= self.dst < 32, f"dst out of range: {self.dst}"
        assert 0 <= self.src0 < 32, f"src0 out of range: {self.src0}"
        assert 0 <= self.src1 < 32, f"src1 out of range: {self.src1}"
        assert -256 <= self.imm < 512, f"imm out of range: {self.imm}"

    @property
    def format(self) -> InstructionFormat:
        return OPCODE_FORMAT.get(self.opcode, InstructionFormat.R)

    def has_extension(self) -> bool:
        return self.ext is not None

    def size_words(self) -> int:
        return 2 if self.has_extension() else 1


# ==============================================================================
# Encoding/Decoding
# ==============================================================================

def encode_instruction(instr: Instruction) -> bytes:
    """
    Encode instruction to bytes.

    Format: [opcode:8][dst:5][src0:5][src1:5][imm:9]

    Returns 4 bytes (or 8 if extension present).
    """
    # Pack main word
    imm_bits = instr.imm & 0x1FF  # 9 bits, handle negative
    word = ((instr.opcode & 0xFF) << 24) | \
           ((instr.dst & 0x1F) << 19) | \
           ((instr.src0 & 0x1F) << 14) | \
           ((instr.src1 & 0x1F) << 9) | \
           (imm_bits & 0x1FF)

    result = struct.pack('>I', word)

    # Add extension if present
    if instr.ext is not None:
        result += struct.pack('>I', instr.ext & 0xFFFFFFFF)

    return result


def decode_instruction(data: bytes, offset: int = 0) -> Tuple[Instruction, int]:
    """
    Decode instruction from bytes.

    Returns (instruction, bytes_consumed).
    """
    word = struct.unpack_from('>I', data, offset)[0]

    opcode = Opcode((word >> 24) & 0xFF)
    dst = (word >> 19) & 0x1F
    src0 = (word >> 14) & 0x1F
    src1 = (word >> 9) & 0x1F
    imm = word & 0x1FF

    # Sign-extend immediate if negative (bit 8 set)
    if imm & 0x100:
        imm = imm - 0x200

    # Check if extension needed
    fmt = OPCODE_FORMAT.get(opcode, InstructionFormat.R)
    ext = None
    consumed = 4

    if fmt == InstructionFormat.EXT and len(data) >= offset + 8:
        ext = struct.unpack_from('>I', data, offset + 4)[0]
        consumed = 8

    return Instruction(opcode, dst, src0, src1, imm, ext), consumed


def encode_program(instructions: List[Instruction]) -> bytes:
    """Encode a list of instructions to bytes."""
    result = b''
    for instr in instructions:
        result += encode_instruction(instr)
    return result


def decode_program(data: bytes) -> List[Instruction]:
    """Decode bytes to list of instructions."""
    instructions = []
    offset = 0
    while offset < len(data):
        instr, consumed = decode_instruction(data, offset)
        instructions.append(instr)
        offset += consumed
    return instructions
