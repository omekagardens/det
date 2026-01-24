"""
EIS Native Compiler - JIT Compilation to Native Machine Code
============================================================

Compiles EIS bytecode to native machine code (ARM64 or x86_64).
Provides significant speedup for performance-critical kernels.

Architecture:
    1. EIS bytecode → IR (Intermediate Representation)
    2. IR optimization passes
    3. IR → Native code generation
    4. JIT execution via mmap + execute

Usage:
    from det.lang.eis_native import EISNativeCompiler, Target

    compiler = EISNativeCompiler(target=Target.ARM64)
    native_fn = compiler.compile(bytecode)
    result = native_fn.execute(inputs)
"""

import struct
import mmap
import ctypes
import platform
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any

# Import EIS types - handle both package and direct import
try:
    from ..eis import (
        Opcode, EISInstruction, decode_instruction,
        RegisterFile, EIS_MAX_SCALAR_REGS, EIS_MAX_REF_REGS, EIS_MAX_TOKEN_REGS
    )
except ImportError:
    # Direct import for testing
    import sys
    import os
    import importlib.util

    def _load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    _base = os.path.join(os.path.dirname(__file__), "..", "eis")
    _encoding = _load_module("det.eis.encoding", os.path.join(_base, "encoding.py"))

    Opcode = _encoding.Opcode
    EISInstruction = _encoding.Instruction
    decode_instruction = _encoding.decode_instruction

    # Constants
    EIS_MAX_SCALAR_REGS = 16
    EIS_MAX_REF_REGS = 8
    EIS_MAX_TOKEN_REGS = 8

    class RegisterFile:
        """Minimal RegisterFile stub for native compilation."""
        pass


# =============================================================================
# TARGET ARCHITECTURE
# =============================================================================

class Target(Enum):
    """Target architecture for native compilation."""
    ARM64 = auto()    # Apple Silicon, ARM servers
    X86_64 = auto()   # Intel/AMD
    AUTO = auto()     # Detect from platform


def detect_target() -> Target:
    """Detect the current platform's target architecture."""
    machine = platform.machine().lower()
    if machine in ('arm64', 'aarch64'):
        return Target.ARM64
    elif machine in ('x86_64', 'amd64'):
        return Target.X86_64
    else:
        raise RuntimeError(f"Unsupported architecture: {machine}")


# =============================================================================
# INTERMEDIATE REPRESENTATION
# =============================================================================

class IROp(Enum):
    """IR operation codes."""
    # Data movement
    LOAD_IMM = auto()      # Load immediate to virtual reg
    LOAD_REG = auto()      # Load from EIS register
    STORE_REG = auto()     # Store to EIS register
    MOVE = auto()          # Move between virtual regs

    # Arithmetic
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    NEG = auto()
    ABS = auto()

    # Math functions
    SQRT = auto()
    SIN = auto()
    COS = auto()
    MIN = auto()
    MAX = auto()

    # Comparison
    CMP_LT = auto()
    CMP_EQ = auto()
    CMP_GT = auto()

    # Control flow
    LABEL = auto()
    JUMP = auto()
    JUMP_IF = auto()
    CALL = auto()
    RET = auto()

    # Phase/special
    PHASE = auto()
    HALT = auto()
    NOP = auto()


@dataclass
class IRInstr:
    """IR instruction."""
    op: IROp
    dst: int = 0           # Destination virtual register
    src1: int = 0          # Source 1 virtual register
    src2: int = 0          # Source 2 virtual register
    imm: float = 0.0       # Immediate value
    label: str = ""        # Label name (for jumps)

    def __repr__(self):
        if self.op == IROp.LOAD_IMM:
            return f"v{self.dst} = {self.imm}"
        elif self.op == IROp.LOAD_REG:
            return f"v{self.dst} = R[{self.src1}]"
        elif self.op == IROp.STORE_REG:
            return f"R[{self.dst}] = v{self.src1}"
        elif self.op == IROp.ADD:
            return f"v{self.dst} = v{self.src1} + v{self.src2}"
        elif self.op == IROp.SUB:
            return f"v{self.dst} = v{self.src1} - v{self.src2}"
        elif self.op == IROp.MUL:
            return f"v{self.dst} = v{self.src1} * v{self.src2}"
        elif self.op == IROp.LABEL:
            return f"{self.label}:"
        elif self.op == IROp.JUMP:
            return f"jump {self.label}"
        elif self.op == IROp.JUMP_IF:
            return f"jump_if v{self.src1} -> {self.label}"
        elif self.op == IROp.RET:
            return "ret"
        else:
            return f"{self.op.name} v{self.dst}, v{self.src1}, v{self.src2}"


@dataclass
class IRProgram:
    """IR program (list of IR instructions)."""
    instructions: List[IRInstr] = field(default_factory=list)
    num_virtual_regs: int = 0
    labels: Dict[str, int] = field(default_factory=dict)

    def add(self, instr: IRInstr):
        if instr.op == IROp.LABEL:
            self.labels[instr.label] = len(self.instructions)
        self.instructions.append(instr)

    def alloc_vreg(self) -> int:
        """Allocate a new virtual register."""
        vreg = self.num_virtual_regs
        self.num_virtual_regs += 1
        return vreg

    def dump(self) -> str:
        """Dump IR to string."""
        lines = []
        for i, instr in enumerate(self.instructions):
            lines.append(f"{i:4d}: {instr}")
        return "\n".join(lines)


# =============================================================================
# EIS TO IR LOWERING
# =============================================================================

class EISToIR:
    """Lower EIS bytecode to IR."""

    def __init__(self):
        self.ir = IRProgram()
        self.reg_map: Dict[int, int] = {}  # EIS reg -> virtual reg

    def get_vreg(self, eis_reg: int) -> int:
        """Get or create virtual register for EIS register."""
        if eis_reg not in self.reg_map:
            vreg = self.ir.alloc_vreg()
            self.reg_map[eis_reg] = vreg
            # Load initial value from EIS register file
            self.ir.add(IRInstr(IROp.LOAD_REG, dst=vreg, src1=eis_reg))
        return self.reg_map[eis_reg]

    def lower(self, bytecode: bytes) -> IRProgram:
        """Lower EIS bytecode to IR."""
        self.ir = IRProgram()
        self.reg_map = {}

        offset = 0
        while offset < len(bytecode):
            instr, consumed = decode_instruction(bytecode, offset)
            self._lower_instruction(instr)
            offset += consumed

        # Ensure we have a return
        if not self.ir.instructions or self.ir.instructions[-1].op != IROp.RET:
            self.ir.add(IRInstr(IROp.RET))

        return self.ir

    def _lower_instruction(self, instr: EISInstruction):
        """Lower single EIS instruction to IR."""
        op = instr.opcode

        if op == Opcode.NOP:
            self.ir.add(IRInstr(IROp.NOP))

        elif op == Opcode.HALT:
            self.ir.add(IRInstr(IROp.HALT))
            self.ir.add(IRInstr(IROp.RET))

        elif op == Opcode.LDI:
            dst = self.ir.alloc_vreg()
            self.reg_map[instr.dst] = dst
            self.ir.add(IRInstr(IROp.LOAD_IMM, dst=dst, imm=float(instr.imm)))

        elif op == Opcode.LDI_EXT:
            dst = self.ir.alloc_vreg()
            self.reg_map[instr.dst] = dst
            # Convert extension word to float
            float_val = struct.unpack('f', struct.pack('I', instr.ext))[0]
            self.ir.add(IRInstr(IROp.LOAD_IMM, dst=dst, imm=float_val))

        elif op == Opcode.ADD:
            src1 = self.get_vreg(instr.src0)
            src2 = self.get_vreg(instr.src1)
            dst = self.ir.alloc_vreg()
            self.reg_map[instr.dst] = dst
            self.ir.add(IRInstr(IROp.ADD, dst=dst, src1=src1, src2=src2))

        elif op == Opcode.SUB:
            src1 = self.get_vreg(instr.src0)
            src2 = self.get_vreg(instr.src1)
            dst = self.ir.alloc_vreg()
            self.reg_map[instr.dst] = dst
            self.ir.add(IRInstr(IROp.SUB, dst=dst, src1=src1, src2=src2))

        elif op == Opcode.MUL:
            src1 = self.get_vreg(instr.src0)
            src2 = self.get_vreg(instr.src1)
            dst = self.ir.alloc_vreg()
            self.reg_map[instr.dst] = dst
            self.ir.add(IRInstr(IROp.MUL, dst=dst, src1=src1, src2=src2))

        elif op == Opcode.DIV:
            src1 = self.get_vreg(instr.src0)
            src2 = self.get_vreg(instr.src1)
            dst = self.ir.alloc_vreg()
            self.reg_map[instr.dst] = dst
            self.ir.add(IRInstr(IROp.DIV, dst=dst, src1=src1, src2=src2))

        elif op == Opcode.NEG:
            src1 = self.get_vreg(instr.src0)
            dst = self.ir.alloc_vreg()
            self.reg_map[instr.dst] = dst
            self.ir.add(IRInstr(IROp.NEG, dst=dst, src1=src1))

        elif op == Opcode.ABS:
            src1 = self.get_vreg(instr.src0)
            dst = self.ir.alloc_vreg()
            self.reg_map[instr.dst] = dst
            self.ir.add(IRInstr(IROp.ABS, dst=dst, src1=src1))

        elif op == Opcode.SQRT:
            src1 = self.get_vreg(instr.src0)
            dst = self.ir.alloc_vreg()
            self.reg_map[instr.dst] = dst
            self.ir.add(IRInstr(IROp.SQRT, dst=dst, src1=src1))

        elif op == Opcode.SIN:
            src1 = self.get_vreg(instr.src0)
            dst = self.ir.alloc_vreg()
            self.reg_map[instr.dst] = dst
            self.ir.add(IRInstr(IROp.SIN, dst=dst, src1=src1))

        elif op == Opcode.COS:
            src1 = self.get_vreg(instr.src0)
            dst = self.ir.alloc_vreg()
            self.reg_map[instr.dst] = dst
            self.ir.add(IRInstr(IROp.COS, dst=dst, src1=src1))

        elif op == Opcode.MIN:
            src1 = self.get_vreg(instr.src0)
            src2 = self.get_vreg(instr.src1)
            dst = self.ir.alloc_vreg()
            self.reg_map[instr.dst] = dst
            self.ir.add(IRInstr(IROp.MIN, dst=dst, src1=src1, src2=src2))

        elif op == Opcode.MAX:
            src1 = self.get_vreg(instr.src0)
            src2 = self.get_vreg(instr.src1)
            dst = self.ir.alloc_vreg()
            self.reg_map[instr.dst] = dst
            self.ir.add(IRInstr(IROp.MAX, dst=dst, src1=src1, src2=src2))

        elif op == Opcode.CMP:
            src1 = self.get_vreg(instr.src0)
            src2 = self.get_vreg(instr.src1)
            dst = self.ir.alloc_vreg()
            self.reg_map[instr.dst] = dst
            # CMP produces token: LT=1, EQ=2, GT=3
            self.ir.add(IRInstr(IROp.CMP_LT, dst=dst, src1=src1, src2=src2))

        elif op == Opcode.PHASE:
            self.ir.add(IRInstr(IROp.PHASE, imm=float(instr.imm)))

        else:
            # Unsupported opcode - emit NOP
            self.ir.add(IRInstr(IROp.NOP))


# =============================================================================
# NATIVE CODE GENERATION - ARM64
# =============================================================================

class ARM64CodeGen:
    """Generate ARM64 machine code from IR."""

    # ARM64 registers
    # x0-x7: arguments and return values
    # x8: indirect result
    # x9-x15: caller-saved temporaries
    # x16-x17: intra-procedure-call scratch
    # x18: platform register
    # x19-x28: callee-saved
    # x29: frame pointer
    # x30: link register (return address)
    # sp: stack pointer

    # Floating point: d0-d31 (64-bit) or s0-s31 (32-bit)

    def __init__(self):
        self.code: bytearray = bytearray()
        self.vreg_to_freg: Dict[int, int] = {}  # Virtual reg -> FP reg
        self.next_freg: int = 0  # Next available FP register (s0-s15)
        self.labels: Dict[str, int] = {}
        self.fixups: List[Tuple[int, str]] = []  # (offset, label) for branch fixups

    def alloc_freg(self, vreg: int) -> int:
        """Allocate floating-point register for virtual register."""
        if vreg in self.vreg_to_freg:
            return self.vreg_to_freg[vreg]

        freg = self.next_freg
        self.next_freg += 1
        if self.next_freg > 15:
            # Register spilling would be needed - for now, wrap around
            self.next_freg = 8  # Use s8-s15 as spill range

        self.vreg_to_freg[vreg] = freg
        return freg

    def emit32(self, value: int):
        """Emit 32-bit instruction."""
        self.code.extend(struct.pack('<I', value & 0xFFFFFFFF))

    def emit_nop(self):
        """NOP instruction."""
        self.emit32(0xD503201F)

    def emit_ret(self):
        """RET instruction (return via x30)."""
        self.emit32(0xD65F03C0)

    def emit_fmov_imm(self, sd: int, imm: float):
        """Load floating-point immediate to register.
        For simplicity, we'll load from a constant pool approach using
        integer registers and fmov."""
        # Convert float to bits
        bits = struct.unpack('<I', struct.pack('<f', imm))[0]

        # Load bits into w9 (using MOVZ + MOVK if needed)
        lo16 = bits & 0xFFFF
        hi16 = (bits >> 16) & 0xFFFF

        # MOVZ w9, #lo16
        self.emit32(0x52800009 | (lo16 << 5))

        if hi16 != 0:
            # MOVK w9, #hi16, LSL #16
            self.emit32(0x72A00009 | (hi16 << 5))

        # FMOV sd, w9
        self.emit32(0x1E270120 | sd | (9 << 5))

    def emit_fadd(self, sd: int, sn: int, sm: int):
        """FADD Sd, Sn, Sm (single precision)."""
        self.emit32(0x1E202800 | sd | (sn << 5) | (sm << 16))

    def emit_fsub(self, sd: int, sn: int, sm: int):
        """FSUB Sd, Sn, Sm (single precision)."""
        self.emit32(0x1E203800 | sd | (sn << 5) | (sm << 16))

    def emit_fmul(self, sd: int, sn: int, sm: int):
        """FMUL Sd, Sn, Sm (single precision)."""
        self.emit32(0x1E200800 | sd | (sn << 5) | (sm << 16))

    def emit_fdiv(self, sd: int, sn: int, sm: int):
        """FDIV Sd, Sn, Sm (single precision)."""
        self.emit32(0x1E201800 | sd | (sn << 5) | (sm << 16))

    def emit_fneg(self, sd: int, sn: int):
        """FNEG Sd, Sn (single precision)."""
        self.emit32(0x1E214000 | sd | (sn << 5))

    def emit_fabs(self, sd: int, sn: int):
        """FABS Sd, Sn (single precision)."""
        self.emit32(0x1E20C000 | sd | (sn << 5))

    def emit_fsqrt(self, sd: int, sn: int):
        """FSQRT Sd, Sn (single precision)."""
        self.emit32(0x1E21C000 | sd | (sn << 5))

    def emit_fmov(self, sd: int, sn: int):
        """FMOV Sd, Sn (register to register)."""
        self.emit32(0x1E204000 | sd | (sn << 5))

    def emit_fcmp(self, sn: int, sm: int):
        """FCMP Sn, Sm (sets condition flags)."""
        self.emit32(0x1E202000 | (sn << 5) | (sm << 16))

    def emit_fmin(self, sd: int, sn: int, sm: int):
        """FMIN Sd, Sn, Sm (single precision)."""
        self.emit32(0x1E205800 | sd | (sn << 5) | (sm << 16))

    def emit_fmax(self, sd: int, sn: int, sm: int):
        """FMAX Sd, Sn, Sm (single precision)."""
        self.emit32(0x1E204800 | sd | (sn << 5) | (sm << 16))

    def generate(self, ir: IRProgram) -> bytes:
        """Generate ARM64 code from IR."""
        self.code = bytearray()
        self.vreg_to_freg = {}
        self.next_freg = 0
        self.labels = {}
        self.fixups = []

        # Prologue (minimal for leaf function)
        # STP x29, x30, [sp, #-16]!
        self.emit32(0xA9BF7BFD)
        # MOV x29, sp
        self.emit32(0x910003FD)

        for instr in ir.instructions:
            self._emit_instruction(instr)

        # Apply branch fixups
        for offset, label in self.fixups:
            if label in self.labels:
                target = self.labels[label]
                delta = (target - offset) // 4
                # Patch the branch instruction
                old = struct.unpack('<I', self.code[offset:offset+4])[0]
                new = (old & 0xFC000000) | (delta & 0x03FFFFFF)
                self.code[offset:offset+4] = struct.pack('<I', new)

        return bytes(self.code)

    def _emit_instruction(self, instr: IRInstr):
        """Emit ARM64 code for IR instruction."""
        op = instr.op

        if op == IROp.NOP:
            self.emit_nop()

        elif op == IROp.LOAD_IMM:
            freg = self.alloc_freg(instr.dst)
            self.emit_fmov_imm(freg, instr.imm)

        elif op == IROp.MOVE:
            src = self.alloc_freg(instr.src1)
            dst = self.alloc_freg(instr.dst)
            self.emit_fmov(dst, src)

        elif op == IROp.ADD:
            src1 = self.alloc_freg(instr.src1)
            src2 = self.alloc_freg(instr.src2)
            dst = self.alloc_freg(instr.dst)
            self.emit_fadd(dst, src1, src2)

        elif op == IROp.SUB:
            src1 = self.alloc_freg(instr.src1)
            src2 = self.alloc_freg(instr.src2)
            dst = self.alloc_freg(instr.dst)
            self.emit_fsub(dst, src1, src2)

        elif op == IROp.MUL:
            src1 = self.alloc_freg(instr.src1)
            src2 = self.alloc_freg(instr.src2)
            dst = self.alloc_freg(instr.dst)
            self.emit_fmul(dst, src1, src2)

        elif op == IROp.DIV:
            src1 = self.alloc_freg(instr.src1)
            src2 = self.alloc_freg(instr.src2)
            dst = self.alloc_freg(instr.dst)
            self.emit_fdiv(dst, src1, src2)

        elif op == IROp.NEG:
            src1 = self.alloc_freg(instr.src1)
            dst = self.alloc_freg(instr.dst)
            self.emit_fneg(dst, src1)

        elif op == IROp.ABS:
            src1 = self.alloc_freg(instr.src1)
            dst = self.alloc_freg(instr.dst)
            self.emit_fabs(dst, src1)

        elif op == IROp.SQRT:
            src1 = self.alloc_freg(instr.src1)
            dst = self.alloc_freg(instr.dst)
            self.emit_fsqrt(dst, src1)

        elif op == IROp.MIN:
            src1 = self.alloc_freg(instr.src1)
            src2 = self.alloc_freg(instr.src2)
            dst = self.alloc_freg(instr.dst)
            self.emit_fmin(dst, src1, src2)

        elif op == IROp.MAX:
            src1 = self.alloc_freg(instr.src1)
            src2 = self.alloc_freg(instr.src2)
            dst = self.alloc_freg(instr.dst)
            self.emit_fmax(dst, src1, src2)

        elif op == IROp.LABEL:
            self.labels[instr.label] = len(self.code)

        elif op == IROp.RET or op == IROp.HALT:
            # Move result to s0 if we have virtual reg 0
            if 0 in self.vreg_to_freg and self.vreg_to_freg[0] != 0:
                self.emit_fmov(0, self.vreg_to_freg[0])

            # Epilogue
            # LDP x29, x30, [sp], #16
            self.emit32(0xA8C17BFD)
            # RET
            self.emit_ret()

        elif op == IROp.PHASE:
            # Phase changes don't generate code (handled by VM)
            self.emit_nop()

        else:
            # Unknown op - emit NOP
            self.emit_nop()


# =============================================================================
# NATIVE CODE GENERATION - X86_64
# =============================================================================

class X86_64CodeGen:
    """Generate x86_64 machine code from IR."""

    # x86_64 calling convention (System V AMD64 ABI):
    # Arguments: rdi, rsi, rdx, rcx, r8, r9 (then stack)
    # Return: rax (integer), xmm0 (float)
    # Caller-saved: rax, rcx, rdx, rsi, rdi, r8-r11, xmm0-xmm15
    # Callee-saved: rbx, rbp, r12-r15

    # SSE registers: xmm0-xmm15

    def __init__(self):
        self.code: bytearray = bytearray()
        self.vreg_to_xmm: Dict[int, int] = {}
        self.next_xmm: int = 0
        self.labels: Dict[str, int] = {}
        self.fixups: List[Tuple[int, str]] = []
        self.constants: List[float] = []
        self.const_offset: int = 0

    def alloc_xmm(self, vreg: int) -> int:
        """Allocate XMM register for virtual register."""
        if vreg in self.vreg_to_xmm:
            return self.vreg_to_xmm[vreg]

        xmm = self.next_xmm
        self.next_xmm += 1
        if self.next_xmm > 15:
            self.next_xmm = 8  # Wrap around

        self.vreg_to_xmm[vreg] = xmm
        return xmm

    def emit(self, *bytes_data):
        """Emit bytes."""
        self.code.extend(bytes_data)

    def emit_movss_imm(self, xmm: int, imm: float):
        """Load float immediate to XMM register via constant pool."""
        # Store constant and emit RIP-relative load
        const_idx = len(self.constants)
        self.constants.append(imm)

        # MOVSS xmm, [rip + offset] (will be patched later)
        # REX prefix if xmm >= 8
        if xmm >= 8:
            self.emit(0x44)  # REX.R
            xmm -= 8

        # F3 0F 10 /r - MOVSS xmm, m32
        self.emit(0xF3, 0x0F, 0x10)
        # ModRM: mod=00, reg=xmm, rm=101 (RIP-relative)
        modrm = (xmm << 3) | 0x05
        self.emit(modrm)
        # Displacement placeholder (will be patched)
        self.fixups.append((len(self.code), f"const_{const_idx}"))
        self.emit(0x00, 0x00, 0x00, 0x00)

    def emit_addss(self, dst: int, src: int):
        """ADDSS xmm, xmm."""
        self._emit_sse_rr(0xF3, 0x0F, 0x58, dst, src)

    def emit_subss(self, dst: int, src: int):
        """SUBSS xmm, xmm."""
        self._emit_sse_rr(0xF3, 0x0F, 0x5C, dst, src)

    def emit_mulss(self, dst: int, src: int):
        """MULSS xmm, xmm."""
        self._emit_sse_rr(0xF3, 0x0F, 0x59, dst, src)

    def emit_divss(self, dst: int, src: int):
        """DIVSS xmm, xmm."""
        self._emit_sse_rr(0xF3, 0x0F, 0x5E, dst, src)

    def emit_sqrtss(self, dst: int, src: int):
        """SQRTSS xmm, xmm."""
        self._emit_sse_rr(0xF3, 0x0F, 0x51, dst, src)

    def emit_minss(self, dst: int, src: int):
        """MINSS xmm, xmm."""
        self._emit_sse_rr(0xF3, 0x0F, 0x5D, dst, src)

    def emit_maxss(self, dst: int, src: int):
        """MAXSS xmm, xmm."""
        self._emit_sse_rr(0xF3, 0x0F, 0x5F, dst, src)

    def emit_movss_rr(self, dst: int, src: int):
        """MOVSS xmm, xmm."""
        self._emit_sse_rr(0xF3, 0x0F, 0x10, dst=dst, src=src)

    def emit_xorps(self, dst: int, src: int):
        """XORPS xmm, xmm (for negation)."""
        self._emit_sse_rr(0x0F, 0x57, 0x00, dst, src, no_prefix=True)

    def _emit_sse_rr(self, *prefix_and_opcode, dst: int, src: int, no_prefix: bool = False):
        """Emit SSE instruction with register-register operands."""
        prefix = list(prefix_and_opcode[:-2]) if not no_prefix else list(prefix_and_opcode[:-1])
        opcode = prefix_and_opcode[-2] if not no_prefix else prefix_and_opcode[-1]

        # REX prefix if high registers
        rex = 0
        if dst >= 8:
            rex |= 0x44  # REX.R
            dst -= 8
        if src >= 8:
            rex |= 0x41  # REX.B
            src -= 8

        if rex:
            self.emit(rex)

        for b in prefix:
            self.emit(b)
        self.emit(opcode)

        # ModRM: mod=11 (register), reg=dst, rm=src
        modrm = 0xC0 | (dst << 3) | src
        self.emit(modrm)

    def emit_ret(self):
        """RET instruction."""
        self.emit(0xC3)

    def emit_nop(self):
        """NOP instruction."""
        self.emit(0x90)

    def generate(self, ir: IRProgram) -> bytes:
        """Generate x86_64 code from IR."""
        self.code = bytearray()
        self.vreg_to_xmm = {}
        self.next_xmm = 1  # Reserve xmm0 for return value
        self.labels = {}
        self.fixups = []
        self.constants = []

        # Prologue
        # push rbp
        self.emit(0x55)
        # mov rbp, rsp
        self.emit(0x48, 0x89, 0xE5)

        for instr in ir.instructions:
            self._emit_instruction(instr)

        # Append constant pool
        self.const_offset = len(self.code)
        for const in self.constants:
            self.code.extend(struct.pack('<f', const))

        # Patch constant pool references
        for offset, label in self.fixups:
            if label.startswith("const_"):
                const_idx = int(label.split("_")[1])
                # RIP-relative offset: target - (rip after instruction)
                target = self.const_offset + const_idx * 4
                rip = offset + 4  # RIP points to next instruction
                delta = target - rip
                self.code[offset:offset+4] = struct.pack('<i', delta)

        return bytes(self.code)

    def _emit_instruction(self, instr: IRInstr):
        """Emit x86_64 code for IR instruction."""
        op = instr.op

        if op == IROp.NOP:
            self.emit_nop()

        elif op == IROp.LOAD_IMM:
            xmm = self.alloc_xmm(instr.dst)
            self.emit_movss_imm(xmm, instr.imm)

        elif op == IROp.MOVE:
            src = self.alloc_xmm(instr.src1)
            dst = self.alloc_xmm(instr.dst)
            self.emit_movss_rr(dst, src)

        elif op == IROp.ADD:
            src1 = self.alloc_xmm(instr.src1)
            src2 = self.alloc_xmm(instr.src2)
            dst = self.alloc_xmm(instr.dst)
            if dst != src1:
                self.emit_movss_rr(dst, src1)
            self.emit_addss(dst, src2)

        elif op == IROp.SUB:
            src1 = self.alloc_xmm(instr.src1)
            src2 = self.alloc_xmm(instr.src2)
            dst = self.alloc_xmm(instr.dst)
            if dst != src1:
                self.emit_movss_rr(dst, src1)
            self.emit_subss(dst, src2)

        elif op == IROp.MUL:
            src1 = self.alloc_xmm(instr.src1)
            src2 = self.alloc_xmm(instr.src2)
            dst = self.alloc_xmm(instr.dst)
            if dst != src1:
                self.emit_movss_rr(dst, src1)
            self.emit_mulss(dst, src2)

        elif op == IROp.DIV:
            src1 = self.alloc_xmm(instr.src1)
            src2 = self.alloc_xmm(instr.src2)
            dst = self.alloc_xmm(instr.dst)
            if dst != src1:
                self.emit_movss_rr(dst, src1)
            self.emit_divss(dst, src2)

        elif op == IROp.SQRT:
            src1 = self.alloc_xmm(instr.src1)
            dst = self.alloc_xmm(instr.dst)
            self.emit_sqrtss(dst, src1)

        elif op == IROp.MIN:
            src1 = self.alloc_xmm(instr.src1)
            src2 = self.alloc_xmm(instr.src2)
            dst = self.alloc_xmm(instr.dst)
            if dst != src1:
                self.emit_movss_rr(dst, src1)
            self.emit_minss(dst, src2)

        elif op == IROp.MAX:
            src1 = self.alloc_xmm(instr.src1)
            src2 = self.alloc_xmm(instr.src2)
            dst = self.alloc_xmm(instr.dst)
            if dst != src1:
                self.emit_movss_rr(dst, src1)
            self.emit_maxss(dst, src2)

        elif op == IROp.LABEL:
            self.labels[instr.label] = len(self.code)

        elif op == IROp.RET or op == IROp.HALT:
            # Move result to xmm0 if needed
            if 0 in self.vreg_to_xmm and self.vreg_to_xmm[0] != 0:
                self.emit_movss_rr(0, self.vreg_to_xmm[0])

            # Epilogue
            # pop rbp
            self.emit(0x5D)
            # ret
            self.emit_ret()

        elif op == IROp.PHASE:
            self.emit_nop()

        else:
            self.emit_nop()


# =============================================================================
# JIT EXECUTION
# =============================================================================

@dataclass
class NativeFunction:
    """Compiled native function."""
    code: bytes
    target: Target
    _mmap: Optional[mmap.mmap] = field(default=None, repr=False)
    _func: Optional[Callable] = field(default=None, repr=False)

    def __post_init__(self):
        self._compile()

    def _compile(self):
        """Map code to executable memory and create function pointer."""
        import sys

        # Allocate executable memory
        size = len(self.code)
        if size == 0:
            return

        # Round up to page size
        page_size = 4096
        alloc_size = ((size + page_size - 1) // page_size) * page_size

        # Create executable mapping
        if sys.platform == 'darwin':
            # macOS: MAP_JIT requires special handling
            self._mmap = mmap.mmap(-1, alloc_size,
                                    prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
                                    flags=mmap.MAP_PRIVATE | mmap.MAP_ANON)
        else:
            # Linux
            self._mmap = mmap.mmap(-1, alloc_size,
                                    prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
                                    flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)

        # Copy code
        self._mmap.write(self.code)
        self._mmap.seek(0)

        # Get function pointer
        buf_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._mmap))

        # Create callable
        # Signature: float func(void) - returns result in xmm0/s0
        FUNC_TYPE = ctypes.CFUNCTYPE(ctypes.c_float)
        self._func = FUNC_TYPE(buf_addr)

    def execute(self) -> float:
        """Execute the native function."""
        if self._func is None:
            raise RuntimeError("Function not compiled")
        return self._func()

    def __del__(self):
        """Clean up memory mapping."""
        if self._mmap:
            self._mmap.close()


# =============================================================================
# EIS NATIVE COMPILER
# =============================================================================

class EISNativeCompiler:
    """
    JIT compiler for EIS bytecode.

    Compiles EIS bytecode to native machine code for fast execution.

    Usage:
        compiler = EISNativeCompiler(target=Target.ARM64)
        native_fn = compiler.compile(bytecode)
        result = native_fn.execute()
    """

    def __init__(self, target: Target = Target.AUTO):
        if target == Target.AUTO:
            target = detect_target()
        self.target = target

        if target == Target.ARM64:
            self.codegen = ARM64CodeGen()
        elif target == Target.X86_64:
            self.codegen = X86_64CodeGen()
        else:
            raise ValueError(f"Unsupported target: {target}")

    def compile(self, bytecode: bytes) -> NativeFunction:
        """Compile EIS bytecode to native function."""
        # Lower to IR
        lowerer = EISToIR()
        ir = lowerer.lower(bytecode)

        # Generate native code
        native_code = self.codegen.generate(ir)

        return NativeFunction(code=native_code, target=self.target)

    def compile_ir(self, ir: IRProgram) -> NativeFunction:
        """Compile IR directly to native function."""
        native_code = self.codegen.generate(ir)
        return NativeFunction(code=native_code, target=self.target)

    def get_ir(self, bytecode: bytes) -> IRProgram:
        """Get IR from bytecode (for debugging)."""
        lowerer = EISToIR()
        return lowerer.lower(bytecode)

    def disassemble_ir(self, bytecode: bytes) -> str:
        """Disassemble bytecode to IR text."""
        ir = self.get_ir(bytecode)
        return ir.dump()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compile_native(bytecode: bytes, target: Target = Target.AUTO) -> NativeFunction:
    """Compile EIS bytecode to native function."""
    compiler = EISNativeCompiler(target=target)
    return compiler.compile(bytecode)


def jit_execute(bytecode: bytes) -> float:
    """JIT compile and execute EIS bytecode, returning result."""
    fn = compile_native(bytecode)
    return fn.execute()


__all__ = [
    # Enums
    'Target', 'IROp',
    # IR
    'IRInstr', 'IRProgram', 'EISToIR',
    # Code generators
    'ARM64CodeGen', 'X86_64CodeGen',
    # Native function
    'NativeFunction',
    # Compiler
    'EISNativeCompiler',
    # Convenience
    'compile_native', 'jit_execute', 'detect_target',
]
