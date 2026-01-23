"""
EIS Assembler
=============

Assembles EIS assembly text into bytecode.

Assembly syntax:
    OPCODE dst, src0, src1      ; R-format
    OPCODE dst, src0, #imm      ; RI-format
    OPCODE #imm                 ; I-format
    OPCODE dst, src             ; RR-format

    ; Comments start with semicolon
    label:                      ; Labels for branches

Register names:
    R0-R15  : Scalar registers
    H0-H7   : Reference registers
    T0-T7   : Token registers

Examples:
    LDI R0, #100        ; Load immediate 100 into R0
    ADD R2, R0, R1      ; R2 = R0 + R1
    LDN R3, H0, #0      ; Load node field F (0) from H0 into R3
    CMP T0, R0, R1      ; Compare R0, R1, store result in T0
    PROP_BEGIN H1       ; Begin proposal
    COMMIT H2           ; Commit choice
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re

from .encoding import Opcode, Instruction, encode_instruction, decode_instruction
from .registers import parse_reg_name, reg_name
from .types import NodeField, BondField, WitnessToken


@dataclass
class AsmLine:
    """A parsed assembly line."""
    line_num: int
    label: Optional[str]
    opcode: Optional[str]
    operands: List[str]
    comment: Optional[str]
    raw: str


class AssemblerError(Exception):
    """Assembler error with line info."""
    def __init__(self, message: str, line_num: int = 0):
        self.line_num = line_num
        super().__init__(f"Line {line_num}: {message}")


class Assembler:
    """
    EIS Assembler.

    Converts assembly text to bytecode.
    """

    def __init__(self):
        self.labels: Dict[str, int] = {}
        self.instructions: List[Instruction] = []
        self.errors: List[str] = []

        # Build opcode lookup
        self.opcodes = {op.name: op for op in Opcode}

        # Field name lookup
        self.node_fields = {f.name: f.value for f in NodeField}
        self.bond_fields = {f.name: f.value for f in BondField}
        self.tokens = {t.name: t.value for t in WitnessToken}

    def assemble(self, source: str) -> bytes:
        """Assemble source text to bytecode."""
        self.labels = {}
        self.instructions = []
        self.errors = []

        lines = source.split('\n')
        parsed_lines = [self._parse_line(i + 1, line) for i, line in enumerate(lines)]

        # First pass: collect labels
        byte_offset = 0
        for parsed in parsed_lines:
            if parsed.label:
                self.labels[parsed.label] = byte_offset
            if parsed.opcode:
                byte_offset += 4  # Each instruction is 4 bytes (may need ext)

        # Second pass: assemble instructions
        for parsed in parsed_lines:
            if parsed.opcode:
                try:
                    instr = self._assemble_instruction(parsed)
                    self.instructions.append(instr)
                except AssemblerError as e:
                    self.errors.append(str(e))
                except Exception as e:
                    self.errors.append(f"Line {parsed.line_num}: {e}")

        if self.errors:
            raise AssemblerError("\n".join(self.errors))

        # Encode to bytes
        result = b''
        for instr in self.instructions:
            result += encode_instruction(instr)

        return result

    def _parse_line(self, line_num: int, line: str) -> AsmLine:
        """Parse a single assembly line."""
        raw = line
        label = None
        opcode = None
        operands = []
        comment = None

        # Strip comment
        if ';' in line:
            idx = line.index(';')
            comment = line[idx + 1:].strip()
            line = line[:idx]

        line = line.strip()
        if not line:
            return AsmLine(line_num, label, opcode, operands, comment, raw)

        # Check for label
        if ':' in line:
            idx = line.index(':')
            label = line[:idx].strip()
            line = line[idx + 1:].strip()

        if not line:
            return AsmLine(line_num, label, opcode, operands, comment, raw)

        # Parse opcode and operands
        parts = line.split(None, 1)
        opcode = parts[0].upper()

        if len(parts) > 1:
            # Parse operands (comma-separated)
            ops_str = parts[1]
            operands = [op.strip() for op in ops_str.split(',')]

        return AsmLine(line_num, label, opcode, operands, comment, raw)

    def _assemble_instruction(self, line: AsmLine) -> Instruction:
        """Assemble a single instruction."""
        if line.opcode not in self.opcodes:
            raise AssemblerError(f"Unknown opcode: {line.opcode}", line.line_num)

        opcode = self.opcodes[line.opcode]
        dst, src0, src1, imm, ext = 0, 0, 0, 0, None

        # Parse operands based on opcode
        ops = line.operands

        if opcode in (Opcode.NOP, Opcode.HALT, Opcode.YIELD, Opcode.FENCE,
                      Opcode.TICK, Opcode.COMMIT_ALL, Opcode.RET, Opcode.DEBUG):
            # No operands
            pass

        elif opcode == Opcode.PHASE:
            # PHASE #imm
            imm = self._parse_immediate(ops[0] if ops else "0", line.line_num)

        elif opcode in (Opcode.LDI,):
            # LDI dst, #imm
            dst = self._parse_register(ops[0], line.line_num)
            imm = self._parse_immediate(ops[1] if len(ops) > 1 else "0", line.line_num)

        elif opcode in (Opcode.LDI_EXT,):
            # LDI_EXT dst, #imm32
            dst = self._parse_register(ops[0], line.line_num)
            ext = self._parse_immediate(ops[1] if len(ops) > 1 else "0", line.line_num)

        elif opcode in (Opcode.LDN, Opcode.LDB):
            # LDN dst, refReg, #fieldId
            dst = self._parse_register(ops[0], line.line_num)
            src0 = self._parse_register(ops[1] if len(ops) > 1 else "H0", line.line_num)
            imm = self._parse_field_id(ops[2] if len(ops) > 2 else "0", line.line_num)

        elif opcode in (Opcode.LDT, Opcode.LDR, Opcode.NEG, Opcode.ABS,
                        Opcode.SQRT, Opcode.RELU, Opcode.SIN, Opcode.COS):
            # Two-register format: OP dst, src
            dst = self._parse_register(ops[0], line.line_num)
            src0 = self._parse_register(ops[1] if len(ops) > 1 else "R0", line.line_num)

        elif opcode in (Opcode.ST_TOK, Opcode.ST_NODE, Opcode.ST_BOND):
            # Store format
            if opcode == Opcode.ST_TOK:
                dst = self._parse_register(ops[0], line.line_num)
                src0 = self._parse_register(ops[1] if len(ops) > 1 else "R0", line.line_num)
            else:
                src0 = self._parse_register(ops[0], line.line_num)
                imm = self._parse_field_id(ops[1] if len(ops) > 1 else "0", line.line_num)

        elif opcode in (Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV,
                        Opcode.MIN, Opcode.MAX, Opcode.MOD):
            # Three-register format: OP dst, src0, src1
            dst = self._parse_register(ops[0], line.line_num)
            src0 = self._parse_register(ops[1] if len(ops) > 1 else "R0", line.line_num)
            src1 = self._parse_register(ops[2] if len(ops) > 2 else "R0", line.line_num)

        elif opcode == Opcode.CMP:
            # CMP dstTok, src0, src1
            dst = self._parse_register(ops[0], line.line_num)
            src0 = self._parse_register(ops[1] if len(ops) > 1 else "R0", line.line_num)
            src1 = self._parse_register(ops[2] if len(ops) > 2 else "R0", line.line_num)

        elif opcode == Opcode.TSET:
            # TSET dstTok, #value
            dst = self._parse_register(ops[0], line.line_num)
            imm = self._parse_token_value(ops[1] if len(ops) > 1 else "0", line.line_num)

        elif opcode == Opcode.TMOV:
            # TMOV dstTok, srcTok
            dst = self._parse_register(ops[0], line.line_num)
            src0 = self._parse_register(ops[1] if len(ops) > 1 else "T0", line.line_num)

        elif opcode in (Opcode.PROP_BEGIN, Opcode.PROP_END):
            # PROP_BEGIN propRef
            dst = self._parse_register(ops[0] if ops else "H0", line.line_num)

        elif opcode == Opcode.PROP_SCORE:
            # PROP_SCORE propRef, scoreReg
            dst = self._parse_register(ops[0] if ops else "H0", line.line_num)
            src0 = self._parse_register(ops[1] if len(ops) > 1 else "R0", line.line_num)

        elif opcode == Opcode.CHOOSE:
            # CHOOSE choiceRef, decisiveness, seed
            dst = self._parse_register(ops[0] if ops else "H0", line.line_num)
            src0 = self._parse_register(ops[1] if len(ops) > 1 else "R0", line.line_num)
            src1 = self._parse_register(ops[2] if len(ops) > 2 else "R0", line.line_num)

        elif opcode == Opcode.COMMIT:
            # COMMIT choiceRef
            src0 = self._parse_register(ops[0] if ops else "H0", line.line_num)

        elif opcode == Opcode.WITNESS:
            # WITNESS tokRef, value
            dst = self._parse_register(ops[0] if ops else "T0", line.line_num)
            src0 = self._parse_register(ops[1] if len(ops) > 1 else "R0", line.line_num)

        elif opcode == Opcode.MKNODE:
            # MKNODE dst, #nodeId
            dst = self._parse_register(ops[0], line.line_num)
            imm = self._parse_immediate(ops[1] if len(ops) > 1 else "0", line.line_num)

        elif opcode == Opcode.GETSELF:
            # GETSELF dst
            dst = self._parse_register(ops[0] if ops else "H0", line.line_num)

        elif opcode == Opcode.XFER:
            # XFER amountReg, srcRef, dstRef
            dst = self._parse_register(ops[0] if ops else "R0", line.line_num)
            src0 = self._parse_register(ops[1] if len(ops) > 1 else "H0", line.line_num)
            src1 = self._parse_register(ops[2] if len(ops) > 2 else "H1", line.line_num)

        elif opcode == Opcode.DIFFUSE:
            # DIFFUSE sigmaReg, nodeARef, nodeBRef
            dst = self._parse_register(ops[0] if ops else "R0", line.line_num)
            src0 = self._parse_register(ops[1] if len(ops) > 1 else "H0", line.line_num)
            src1 = self._parse_register(ops[2] if len(ops) > 2 else "H1", line.line_num)

        elif opcode == Opcode.CALL:
            # CALL #kernelId
            imm = self._parse_immediate(ops[0] if ops else "0", line.line_num)

        elif opcode == Opcode.TRACE:
            # TRACE #value
            imm = self._parse_immediate(ops[0] if ops else "0", line.line_num)

        return Instruction(opcode, dst, src0, src1, imm, ext)

    def _parse_register(self, s: str, line_num: int) -> int:
        """Parse register name to unified address."""
        s = s.strip().upper()
        try:
            return parse_reg_name(s)
        except ValueError:
            raise AssemblerError(f"Invalid register: {s}", line_num)

    def _parse_immediate(self, s: str, line_num: int) -> int:
        """Parse immediate value."""
        s = s.strip()
        if s.startswith('#'):
            s = s[1:]

        try:
            if s.startswith('0x') or s.startswith('0X'):
                return int(s, 16)
            elif s.startswith('0b') or s.startswith('0B'):
                return int(s, 2)
            else:
                return int(s)
        except ValueError:
            raise AssemblerError(f"Invalid immediate: {s}", line_num)

    def _parse_field_id(self, s: str, line_num: int) -> int:
        """Parse field ID (name or number)."""
        s = s.strip().upper()
        if s.startswith('#'):
            s = s[1:]

        # Try as field name
        if s in self.node_fields:
            return self.node_fields[s]
        if s in self.bond_fields:
            return self.bond_fields[s]

        # Try as number
        try:
            return int(s)
        except ValueError:
            raise AssemblerError(f"Invalid field ID: {s}", line_num)

    def _parse_token_value(self, s: str, line_num: int) -> int:
        """Parse token value (name or number)."""
        s = s.strip().upper()
        if s.startswith('#'):
            s = s[1:]

        # Try as token name
        if s in self.tokens:
            return self.tokens[s]

        # Try as number
        try:
            if s.startswith('0x') or s.startswith('0X'):
                return int(s, 16)
            return int(s)
        except ValueError:
            raise AssemblerError(f"Invalid token value: {s}", line_num)


def assemble(source: str) -> bytes:
    """Convenience function to assemble source."""
    asm = Assembler()
    return asm.assemble(source)


def disassemble(bytecode: bytes) -> str:
    """Disassemble bytecode to assembly text."""
    lines = []
    offset = 0

    while offset < len(bytecode):
        try:
            instr, consumed = decode_instruction(bytecode, offset)
            line = _format_instruction(instr, offset)
            lines.append(f"{offset:04x}:  {line}")
            offset += consumed
        except Exception as e:
            lines.append(f"{offset:04x}:  ??? (decode error: {e})")
            offset += 4

    return '\n'.join(lines)


def _format_instruction(instr: Instruction, offset: int = 0) -> str:
    """Format instruction as assembly text."""
    op = instr.opcode.name

    # Format based on opcode
    if instr.opcode in (Opcode.NOP, Opcode.HALT, Opcode.YIELD, Opcode.FENCE,
                        Opcode.TICK, Opcode.COMMIT_ALL, Opcode.RET, Opcode.DEBUG):
        return op

    elif instr.opcode == Opcode.PHASE:
        return f"{op} #{instr.imm}"

    elif instr.opcode in (Opcode.LDI,):
        return f"{op} {reg_name(instr.dst)}, #{instr.imm}"

    elif instr.opcode in (Opcode.LDN, Opcode.LDB):
        return f"{op} {reg_name(instr.dst)}, {reg_name(instr.src0)}, #{instr.imm}"

    elif instr.opcode in (Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV,
                          Opcode.MIN, Opcode.MAX, Opcode.CMP):
        return f"{op} {reg_name(instr.dst)}, {reg_name(instr.src0)}, {reg_name(instr.src1)}"

    elif instr.opcode in (Opcode.NEG, Opcode.ABS, Opcode.SQRT, Opcode.RELU,
                          Opcode.LDT, Opcode.LDR, Opcode.TMOV):
        return f"{op} {reg_name(instr.dst)}, {reg_name(instr.src0)}"

    elif instr.opcode == Opcode.TSET:
        return f"{op} {reg_name(instr.dst)}, #{instr.imm}"

    elif instr.opcode in (Opcode.PROP_BEGIN, Opcode.PROP_END, Opcode.GETSELF):
        return f"{op} {reg_name(instr.dst)}"

    elif instr.opcode == Opcode.COMMIT:
        return f"{op} {reg_name(instr.src0)}"

    elif instr.opcode == Opcode.CALL:
        return f"{op} #{instr.imm}"

    else:
        # Generic format
        parts = [op]
        if instr.dst:
            parts.append(reg_name(instr.dst))
        if instr.src0:
            parts.append(reg_name(instr.src0))
        if instr.src1:
            parts.append(reg_name(instr.src1))
        if instr.imm:
            parts.append(f"#{instr.imm}")
        return ' '.join(parts)
