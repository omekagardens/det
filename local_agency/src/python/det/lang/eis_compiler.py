"""
Existence-Lang to EIS Compiler
==============================

Compiles Existence-Lang AST to EIS (Existence Instruction Set) bytecode.

Compilation Strategy:
    1. Creatures compile to node-lane programs
    2. Kernels compile to callable bytecode sequences
    3. Expressions compile to arithmetic instructions
    4. Control flow compiles to branch-on-token instructions
    5. Proposals compile to PROP_BEGIN/SCORE/END sequences
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Set, TYPE_CHECKING
from enum import IntEnum
import sys
import os
import importlib.util

# Direct module loading to avoid det/__init__.py import issues
def _load_module_direct(name: str, path: str):
    """Load a module directly from path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

_BASE_PATH = os.path.dirname(os.path.abspath(__file__))
_EIS_PATH = os.path.join(os.path.dirname(_BASE_PATH), "eis")

# Load AST nodes directly
if "det.lang.ast_nodes" not in sys.modules:
    _ast_nodes = _load_module_direct("det.lang.ast_nodes", os.path.join(_BASE_PATH, "ast_nodes.py"))
else:
    _ast_nodes = sys.modules["det.lang.ast_nodes"]

# Import AST types
Program = _ast_nodes.Program
Declaration = _ast_nodes.Declaration
Creature = _ast_nodes.Creature
Kernel = _ast_nodes.Kernel
Bond = _ast_nodes.Bond
Presence = _ast_nodes.Presence
Boundary = _ast_nodes.Boundary
Statement = _ast_nodes.Statement
Expression = _ast_nodes.Expression
Block = _ast_nodes.Block
VarDecl = _ast_nodes.VarDecl
WitnessDecl = _ast_nodes.WitnessDecl
Assignment = _ast_nodes.Assignment
ExpressionStmt = _ast_nodes.ExpressionStmt
IfPast = _ast_nodes.IfPast
RepeatPast = _ast_nodes.RepeatPast
WhilePast = _ast_nodes.WhilePast
Return = _ast_nodes.Return
KernelCall = _ast_nodes.KernelCall
InjectF = _ast_nodes.InjectF
RequestGrace = _ast_nodes.RequestGrace
Literal = _ast_nodes.Literal
Identifier = _ast_nodes.Identifier
BinaryOp = _ast_nodes.BinaryOp
UnaryOp = _ast_nodes.UnaryOp
Call = _ast_nodes.Call
FieldAccess = _ast_nodes.FieldAccess
IndexAccess = _ast_nodes.IndexAccess
Distinct = _ast_nodes.Distinct
This = _ast_nodes.This
Compare = _ast_nodes.Compare
Choose = _ast_nodes.Choose
TupleExpr = _ast_nodes.TupleExpr
TypeKind = _ast_nodes.TypeKind
PortDirection = _ast_nodes.PortDirection
SensorDecl = _ast_nodes.SensorDecl
ActuatorDecl = _ast_nodes.ActuatorDecl
ParticipateBlock = _ast_nodes.ParticipateBlock
AgencyBlock = _ast_nodes.AgencyBlock
GraceBlock = _ast_nodes.GraceBlock
Proposal = _ast_nodes.Proposal
PhaseBlock = _ast_nodes.PhaseBlock
Law = _ast_nodes.Law
ASTVisitor = _ast_nodes.ASTVisitor
PrimitiveCallExpr = _ast_nodes.PrimitiveCallExpr

# Load EIS modules directly
if "det.eis.types" not in sys.modules:
    _eis_types = _load_module_direct("det.eis.types", os.path.join(_EIS_PATH, "types.py"))
else:
    _eis_types = sys.modules["det.eis.types"]

if "det.eis.encoding" not in sys.modules:
    _eis_encoding = _load_module_direct("det.eis.encoding", os.path.join(_EIS_PATH, "encoding.py"))
else:
    _eis_encoding = sys.modules["det.eis.encoding"]

# Import EIS types
Opcode = _eis_encoding.Opcode
Instruction = _eis_encoding.Instruction
encode_instruction = _eis_encoding.encode_instruction
NodeField = _eis_types.NodeField
BondField = _eis_types.BondField
EISWitnessToken = _eis_types.WitnessToken

# Import V2 opcodes for phase-aware substrate compilation
V2_PHASE_R = getattr(_eis_encoding, 'V2_PHASE_R', 0x04)
V2_PHASE_P = getattr(_eis_encoding, 'V2_PHASE_P', 0x05)
V2_PHASE_C = getattr(_eis_encoding, 'V2_PHASE_C', 0x06)
V2_PHASE_X = getattr(_eis_encoding, 'V2_PHASE_X', 0x07)
V2_PROP_NEW = getattr(_eis_encoding, 'V2_PROP_NEW', 0x50)
V2_PROP_SCORE = getattr(_eis_encoding, 'V2_PROP_SCORE', 0x51)
V2_PROP_EFFECT = getattr(_eis_encoding, 'V2_PROP_EFFECT', 0x52)
V2_PROP_ARG = getattr(_eis_encoding, 'V2_PROP_ARG', 0x53)
V2_PROP_END = getattr(_eis_encoding, 'V2_PROP_END', 0x54)
V2_CHOOSE = getattr(_eis_encoding, 'V2_CHOOSE', 0x60)
V2_COMMIT = getattr(_eis_encoding, 'V2_COMMIT', 0x61)
V2_WITNESS = getattr(_eis_encoding, 'V2_WITNESS', 0x62)

# Phase name to opcode mapping
V2_PHASE_OPCODES = {
    'read': V2_PHASE_R,
    'propose': V2_PHASE_P,
    'choose': V2_PHASE_C,
    'commit': V2_PHASE_X,
    'READ': V2_PHASE_R,
    'PROPOSE': V2_PHASE_P,
    'CHOOSE': V2_PHASE_C,
    'COMMIT': V2_PHASE_X,
}


# ==============================================================================
# Register Allocator (Enhanced with scoping and spilling)
# ==============================================================================

# Register file sizes (must match substrate_types.h)
# Note: Instruction encoding uses 5-bit fields (0-31), so we cap at 32 per class
# even though the substrate supports more. This is sufficient for most kernels.
NUM_SCALAR_REGS = 32   # R0-R31 (instruction encoding limit)
NUM_REF_REGS = 32      # H0-H31
NUM_TOKEN_REGS = 32    # T0-T31

# Spill memory configuration
SPILL_BASE_ADDR = 0x1000  # Base address for spill slots


class RegClass(IntEnum):
    """Register classes."""
    SCALAR = 0   # R0-R31
    REF = 1      # H0-H31
    TOKEN = 2    # T0-T31


@dataclass
class SpillSlot:
    """Spill slot for register overflow."""
    slot_id: int
    reg_class: RegClass
    var_name: str
    addr: int  # Memory address for spilled value


@dataclass
class RegScope:
    """Scope for register lifetime tracking."""
    name: str
    allocated: Dict[str, int] = field(default_factory=dict)  # var -> reg
    reg_classes: Dict[str, RegClass] = field(default_factory=dict)  # var -> class


@dataclass
class RegAlloc:
    """Enhanced register allocation with scoping and spilling."""
    # Free register pools (expanded sizes)
    free_scalars: List[int] = field(default_factory=lambda: list(range(NUM_SCALAR_REGS - 1, -1, -1)))
    free_refs: List[int] = field(default_factory=lambda: list(range(NUM_REF_REGS - 1, -1, -1)))
    free_tokens: List[int] = field(default_factory=lambda: list(range(NUM_TOKEN_REGS - 1, -1, -1)))

    # Variable to register mapping (global)
    var_regs: Dict[str, int] = field(default_factory=dict)
    var_classes: Dict[str, RegClass] = field(default_factory=dict)

    # Scope stack for lifetime tracking
    scopes: List[RegScope] = field(default_factory=list)

    # Spill slots for overflow
    spill_slots: List[SpillSlot] = field(default_factory=list)
    next_spill_addr: int = SPILL_BASE_ADDR

    # Temporary register tracking (can be freed immediately after use)
    temp_regs: Set[int] = field(default_factory=set)

    def push_scope(self, name: str = ""):
        """Push a new scope for register lifetime tracking."""
        self.scopes.append(RegScope(name=name))

    def pop_scope(self) -> List[int]:
        """Pop scope and free all registers allocated in it."""
        if not self.scopes:
            return []
        scope = self.scopes.pop()
        freed = []
        for var_name, reg in scope.allocated.items():
            reg_class = scope.reg_classes.get(var_name, RegClass.SCALAR)
            self._free_reg(reg, reg_class)
            if var_name in self.var_regs:
                del self.var_regs[var_name]
            if var_name in self.var_classes:
                del self.var_classes[var_name]
            freed.append(reg)
        return freed

    def _free_reg(self, reg: int, reg_class: RegClass):
        """Return register to appropriate pool (class-local index)."""
        if reg_class == RegClass.SCALAR:
            if 0 <= reg < NUM_SCALAR_REGS and reg not in self.free_scalars:
                self.free_scalars.append(reg)
        elif reg_class == RegClass.REF:
            if 0 <= reg < NUM_REF_REGS and reg not in self.free_refs:
                self.free_refs.append(reg)
        elif reg_class == RegClass.TOKEN:
            if 0 <= reg < NUM_TOKEN_REGS and reg not in self.free_tokens:
                self.free_tokens.append(reg)

    def alloc_scalar(self, name: str = "", is_temp: bool = False) -> int:
        """Allocate a scalar register (returns class-local index 0-31)."""
        if not self.free_scalars:
            # Try to spill an existing register
            return self._spill_and_alloc(RegClass.SCALAR, name)
        reg = self.free_scalars.pop()
        if name:
            self.var_regs[name] = reg
            self.var_classes[name] = RegClass.SCALAR
            if self.scopes:
                self.scopes[-1].allocated[name] = reg
                self.scopes[-1].reg_classes[name] = RegClass.SCALAR
        if is_temp:
            self.temp_regs.add((RegClass.SCALAR, reg))
        return reg

    def alloc_ref(self, name: str = "", is_temp: bool = False) -> int:
        """Allocate a reference register (returns class-local index 0-31)."""
        if not self.free_refs:
            return self._spill_and_alloc(RegClass.REF, name)
        reg = self.free_refs.pop()
        if name:
            self.var_regs[name] = reg
            self.var_classes[name] = RegClass.REF
            if self.scopes:
                self.scopes[-1].allocated[name] = reg
                self.scopes[-1].reg_classes[name] = RegClass.REF
        if is_temp:
            self.temp_regs.add((RegClass.REF, reg))
        return reg

    def alloc_token(self, name: str = "", is_temp: bool = False) -> int:
        """Allocate a token register (returns class-local index 0-31)."""
        if not self.free_tokens:
            return self._spill_and_alloc(RegClass.TOKEN, name)
        reg = self.free_tokens.pop()
        if name:
            self.var_regs[name] = reg
            self.var_classes[name] = RegClass.TOKEN
            if self.scopes:
                self.scopes[-1].allocated[name] = reg
                self.scopes[-1].reg_classes[name] = RegClass.TOKEN
        if is_temp:
            self.temp_regs.add((RegClass.TOKEN, reg))
        return reg

    def _spill_and_alloc(self, reg_class: RegClass, name: str) -> int:
        """Spill oldest register and allocate new one."""
        # Find a non-temp register to spill
        target_vars = [v for v, c in self.var_classes.items()
                       if c == reg_class and (c, self.var_regs.get(v)) not in self.temp_regs]
        if not target_vars:
            raise RuntimeError(f"Out of {reg_class.name} registers (all are temps)")

        # Spill the first (oldest) variable
        spill_var = target_vars[0]
        spill_reg = self.var_regs[spill_var]

        # Create spill slot
        slot = SpillSlot(
            slot_id=len(self.spill_slots),
            reg_class=reg_class,
            var_name=spill_var,
            addr=self.next_spill_addr
        )
        self.spill_slots.append(slot)
        self.next_spill_addr += 8  # 8 bytes per slot

        # Free the register
        del self.var_regs[spill_var]
        del self.var_classes[spill_var]
        self._free_reg(spill_reg, reg_class)

        # Allocate for new variable
        if reg_class == RegClass.SCALAR:
            return self.alloc_scalar(name)
        elif reg_class == RegClass.REF:
            return self.alloc_ref(name)
        else:
            return self.alloc_token(name)

    def free_temp(self, reg: int, reg_class: RegClass = RegClass.SCALAR):
        """Free a temporary register."""
        key = (reg_class, reg)
        if key in self.temp_regs:
            self.temp_regs.discard(key)
            self._free_reg(reg, reg_class)

    def free_scalar(self, reg: int):
        """Return scalar register to pool."""
        self._free_reg(reg, RegClass.SCALAR)

    def free_ref(self, reg: int):
        """Return ref register to pool."""
        self._free_reg(reg, RegClass.REF)

    def free_token(self, reg: int):
        """Return token register to pool."""
        self._free_reg(reg, RegClass.TOKEN)

    def free_var(self, name: str):
        """Free register associated with a variable."""
        if name in self.var_regs:
            reg = self.var_regs[name]
            reg_class = self.var_classes.get(name, RegClass.SCALAR)
            del self.var_regs[name]
            if name in self.var_classes:
                del self.var_classes[name]
            self._free_reg(reg, reg_class)

    def lookup(self, name: str) -> Optional[int]:
        """Look up variable's register (class-local index)."""
        return self.var_regs.get(name)

    def lookup_class(self, name: str) -> Optional[RegClass]:
        """Look up variable's register class."""
        return self.var_classes.get(name)

    def lookup_both(self, name: str) -> Tuple[Optional[int], Optional[RegClass]]:
        """Look up variable's register and class."""
        return self.var_regs.get(name), self.var_classes.get(name)

    def reset(self):
        """Reset allocation state."""
        self.free_scalars = list(range(NUM_SCALAR_REGS - 1, -1, -1))
        self.free_refs = list(range(NUM_REF_REGS - 1, -1, -1))
        self.free_tokens = list(range(NUM_TOKEN_REGS - 1, -1, -1))
        self.var_regs.clear()
        self.var_classes.clear()
        self.scopes.clear()
        self.spill_slots.clear()
        self.next_spill_addr = SPILL_BASE_ADDR
        self.temp_regs.clear()

    def stats(self) -> dict:
        """Get allocation statistics."""
        return {
            'scalars_used': NUM_SCALAR_REGS - len(self.free_scalars),
            'scalars_free': len(self.free_scalars),
            'refs_used': NUM_REF_REGS - len(self.free_refs),
            'refs_free': len(self.free_refs),
            'tokens_used': NUM_TOKEN_REGS - len(self.free_tokens),
            'tokens_free': len(self.free_tokens),
            'spill_slots': len(self.spill_slots),
            'scope_depth': len(self.scopes),
        }


# ==============================================================================
# Compiled Program
# ==============================================================================

@dataclass
class CompiledCreature:
    """Compiled creature bytecode."""
    name: str
    agency_code: bytes = b''
    grace_code: bytes = b''
    participate_code: bytes = b''
    init_code: bytes = b''
    kernels: Dict[str, 'CompiledKernel'] = field(default_factory=dict)


@dataclass
class CompiledPort:
    """A compiled kernel port with register assignment."""
    name: str
    direction: str  # "in" or "out"
    port_type: str  # "Register" or "TokenReg"
    reg_index: int  # Allocated register index


@dataclass
class CompiledKernel:
    """Compiled kernel bytecode."""
    name: str
    code: bytes = b''
    port_count: int = 0
    ports: List[CompiledPort] = field(default_factory=list)


@dataclass
class CompiledProgram:
    """Complete compiled program."""
    creatures: Dict[str, CompiledCreature] = field(default_factory=dict)
    kernels: Dict[str, CompiledKernel] = field(default_factory=dict)
    presences: Dict[str, bytes] = field(default_factory=dict)
    entry_point: str = ""


# ==============================================================================
# EIS Compiler
# ==============================================================================

class EISCompiler(ASTVisitor):
    """
    Compiles Existence-Lang AST to EIS bytecode.

    Compilation approach:
        - Each creature becomes a set of bytecode programs
        - Each kernel becomes a callable bytecode sequence
        - Variables map to registers
        - Expressions compile to arithmetic sequences
        - Witness bindings use token registers

    V2 Mode:
        When use_v2=True, emits phase-aware substrate v2 opcodes:
        - V2_PHASE_R/P/C/X for phase transitions
        - V2_PROP_NEW/SCORE/EFFECT/END for proposals
        - V2_CHOOSE/COMMIT/WITNESS for selection
    """

    def __init__(self, use_v2: bool = False):
        self.regs = RegAlloc()
        self.instructions: List[Instruction] = []
        self.labels: Dict[str, int] = {}
        self.pending_labels: Dict[str, List[Tuple[int, str]]] = {}  # label -> [(instr_idx, field)]

        # Symbol tables
        self.creatures: Dict[str, Creature] = {}
        self.kernels: Dict[str, Kernel] = {}

        # Current context
        self.current_creature: Optional[str] = None
        self.current_kernel: Optional[str] = None

        # V2 substrate mode
        self.use_v2 = use_v2

        # Compiled output
        self.output = CompiledProgram()

    def compile(self, program: Program) -> CompiledProgram:
        """Compile a complete program to EIS bytecode."""
        # First pass: collect declarations
        for decl in program.declarations:
            if isinstance(decl, Creature):
                self.creatures[decl.name] = decl
            elif isinstance(decl, Kernel):
                self.kernels[decl.name] = decl

        # Second pass: compile
        for decl in program.declarations:
            if isinstance(decl, Creature):
                self._compile_creature(decl)
            elif isinstance(decl, Kernel):
                self._compile_kernel(decl)
            elif isinstance(decl, Presence):
                self._compile_presence(decl)

        return self.output

    # ==========================================================================
    # Creature Compilation
    # ==========================================================================

    def _compile_creature(self, creature: Creature):
        """Compile a creature to bytecode."""
        self.current_creature = creature.name
        compiled = CompiledCreature(name=creature.name)

        # Compile variable initializations
        self._reset_compilation()
        for var in creature.variables:
            self._compile_var_decl(var)
        if self.instructions:
            compiled.init_code = self._finalize_bytecode()

        # Compile agency block
        if creature.agency:
            self._reset_compilation()
            self._compile_block(creature.agency.body)
            compiled.agency_code = self._finalize_bytecode()

        # Compile grace block
        if creature.grace:
            self._reset_compilation()
            self._compile_block(creature.grace.body)
            compiled.grace_code = self._finalize_bytecode()

        # Compile participate block
        if creature.participate:
            self._reset_compilation()
            self._compile_block(creature.participate.body)
            compiled.participate_code = self._finalize_bytecode()

        # Compile nested kernels
        if hasattr(creature, 'nested_kernels'):
            for kernel in creature.nested_kernels:
                self._compile_kernel_for_creature(kernel, compiled)

        self.output.creatures[creature.name] = compiled
        self.current_creature = None

    def _compile_kernel_for_creature(self, kernel: Kernel, compiled_creature: CompiledCreature):
        """Compile a kernel and add it to the creature's kernel dict."""
        self.current_kernel = kernel.name
        self._reset_compilation()

        # Push kernel-level scope for ports (preserved across phases)
        self.regs.push_scope(f"kernel:{kernel.name}")

        # Allocate registers for ports and record port info
        compiled_ports = []
        for port in kernel.ports:
            if port.type_annotation:
                # Get direction as string (in/out/inout)
                if hasattr(port, 'direction') and port.direction:
                    direction = port.direction.name.lower()  # IN -> "in", OUT -> "out"
                else:
                    direction = "in"

                if port.type_annotation.kind == TypeKind.REGISTER:
                    reg_idx = self.regs.alloc_scalar(port.name)
                    compiled_ports.append(CompiledPort(
                        name=port.name,
                        direction=direction,
                        port_type="Register",
                        reg_index=reg_idx
                    ))
                elif port.type_annotation.kind == TypeKind.TOKEN_REG:
                    reg_idx = self.regs.alloc_token(port.name)
                    compiled_ports.append(CompiledPort(
                        name=port.name,
                        direction=direction,
                        port_type="TokenReg",
                        reg_index=reg_idx
                    ))

        # Compile kernel phases
        for phase_block in kernel.phases:
            self._compile_phase_block(phase_block)

        # RET instruction
        self._emit(Instruction(Opcode.RET))

        # Pop kernel scope
        self.regs.pop_scope()

        compiled_kernel = CompiledKernel(
            name=kernel.name,
            code=self._finalize_bytecode(),
            port_count=len(kernel.ports),
            ports=compiled_ports
        )

        # Add to creature's kernels
        compiled_creature.kernels[kernel.name] = compiled_kernel

        # Also add to global kernels for backward compatibility
        full_name = f"{self.current_creature}.{kernel.name}"
        self.output.kernels[full_name] = compiled_kernel

        self.current_kernel = None

    # ==========================================================================
    # Kernel Compilation
    # ==========================================================================

    def _compile_kernel(self, kernel: Kernel):
        """Compile a kernel to bytecode."""
        self.current_kernel = kernel.name
        self._reset_compilation()

        # Push kernel-level scope for ports (preserved across phases)
        self.regs.push_scope(f"kernel:{kernel.name}")

        # Allocate registers for ports and record port info
        compiled_ports = []
        for port in kernel.ports:
            if port.type_annotation:
                # Get direction as string (in/out/inout)
                if hasattr(port, 'direction') and port.direction:
                    direction = port.direction.name.lower()  # IN -> "in", OUT -> "out"
                else:
                    direction = "in"

                if port.type_annotation.kind == TypeKind.REGISTER:
                    reg_idx = self.regs.alloc_scalar(port.name)
                    compiled_ports.append(CompiledPort(
                        name=port.name,
                        direction=direction,
                        port_type="Register",
                        reg_index=reg_idx
                    ))
                elif port.type_annotation.kind == TypeKind.TOKEN_REG:
                    reg_idx = self.regs.alloc_token(port.name)
                    compiled_ports.append(CompiledPort(
                        name=port.name,
                        direction=direction,
                        port_type="TokenReg",
                        reg_index=reg_idx
                    ))

        # Compile kernel phases
        for phase_block in kernel.phases:
            self._compile_phase_block(phase_block)

        # RET instruction
        self._emit(Instruction(Opcode.RET))

        # Pop kernel scope
        self.regs.pop_scope()

        compiled = CompiledKernel(
            name=kernel.name,
            code=self._finalize_bytecode(),
            port_count=len(kernel.ports),
            ports=compiled_ports
        )
        self.output.kernels[kernel.name] = compiled
        self.current_kernel = None

    def _compile_phase_block(self, phase_block: PhaseBlock):
        """
        Compile a phase block with proper phase transitions.

        In v2 mode, emits:
        - V2_PHASE_R before READ phase
        - V2_PHASE_P before PROPOSE phase
        - V2_PHASE_C before CHOOSE
        - V2_PHASE_X before COMMIT
        """
        phase_name = getattr(phase_block, 'phase_name', '').lower()

        # Push phase scope for local variables
        self.regs.push_scope(f"phase:{phase_name}")

        # Emit phase transition opcode in v2 mode
        if self.use_v2 and phase_name in V2_PHASE_OPCODES:
            phase_opcode = V2_PHASE_OPCODES[phase_name]
            self._emit_raw(phase_opcode, 0, 0, 0, 0)

        # Compile proposals in PROPOSE phase
        if hasattr(phase_block, 'proposals') and phase_block.proposals:
            for proposal in phase_block.proposals:
                self._compile_proposal(proposal)

        # Compile choice in CHOOSE phase
        if hasattr(phase_block, 'choice') and phase_block.choice:
            self._compile_choice(phase_block.choice)

        # Compile commit in COMMIT phase
        if hasattr(phase_block, 'commit') and phase_block.commit:
            self._compile_commit(phase_block.commit)

        # Compile witness write
        if hasattr(phase_block, 'witness_write') and phase_block.witness_write:
            self._compile_witness_decl(phase_block.witness_write)

        # Compile general statements
        if hasattr(phase_block, 'statements') and phase_block.statements:
            for stmt in phase_block.statements:
                self._compile_statement(stmt)

        # Legacy: compile body if present
        if hasattr(phase_block, 'body') and phase_block.body:
            self._compile_block(phase_block.body)

        # Pop phase scope - frees local variables
        self.regs.pop_scope()

    def _compile_choice(self, choice):
        """
        Compile choice declaration (CHOOSE phase).

        Emits V2_CHOOSE in v2 mode.
        """
        # Allocate ref register for chosen proposal
        choice_ref = self.regs.alloc_ref()

        if self.use_v2:
            # V2: CHOOSE opcode - H[dst] = chosen proposal index
            # The choose expression provides the decisiveness parameter
            if hasattr(choice, 'choose_expr') and choice.choose_expr:
                # Compile decisiveness if provided
                if hasattr(choice.choose_expr, 'decisiveness') and choice.choose_expr.decisiveness:
                    dec_reg = self._compile_expression(choice.choose_expr.decisiveness)
                    self._emit_raw(V2_CHOOSE, choice_ref, dec_reg, 0, 0)
                    self.regs.free_scalar(dec_reg)
                else:
                    # Default decisiveness = 1.0 (100% highest score)
                    self._emit_raw(V2_CHOOSE, choice_ref, 0, 0, 1)
            else:
                self._emit_raw(V2_CHOOSE, choice_ref, 0, 0, 1)
        else:
            # V1: Use CHOOSE opcode
            self._emit(Instruction(Opcode.CHOOSE, dst=choice_ref))

        # Store choice result in variable if named
        if hasattr(choice, 'name') and choice.name:
            self.regs.var_regs[choice.name] = choice_ref

    def _compile_commit(self, commit_target):
        """
        Compile commit statement (COMMIT phase).

        Emits V2_COMMIT in v2 mode.
        """
        if self.use_v2:
            # V2: COMMIT opcode - applies chosen effect
            self._emit_raw(V2_COMMIT, 0, 0, 0, 0)
        else:
            # V1: Use COMMIT opcode
            self._emit(Instruction(Opcode.COMMIT))

    # ==========================================================================
    # Presence Compilation
    # ==========================================================================

    def _compile_presence(self, presence: Presence):
        """Compile a presence configuration."""
        self._reset_compilation()

        # Compile init block
        if presence.init_block:
            self._compile_block(presence.init_block.body)

        self._emit(Instruction(Opcode.HALT))

        self.output.presences[presence.name] = self._finalize_bytecode()
        if not self.output.entry_point:
            self.output.entry_point = presence.name

    # ==========================================================================
    # Statement Compilation
    # ==========================================================================

    def _compile_block(self, block: Block):
        """Compile a block of statements."""
        for stmt in block.statements:
            self._compile_statement(stmt)

    def _compile_statement(self, stmt: Statement):
        """Compile a statement."""
        if isinstance(stmt, VarDecl):
            self._compile_var_decl(stmt)
        elif isinstance(stmt, WitnessDecl):
            self._compile_witness_decl(stmt)
        elif isinstance(stmt, Assignment):
            self._compile_assignment(stmt)
        elif isinstance(stmt, ExpressionStmt):
            self._compile_expr_stmt(stmt)
        elif isinstance(stmt, IfPast):
            self._compile_if_past(stmt)
        elif isinstance(stmt, RepeatPast):
            self._compile_repeat_past(stmt)
        elif isinstance(stmt, WhilePast):
            self._compile_while_past(stmt)
        elif isinstance(stmt, Return):
            self._compile_return(stmt)
        elif isinstance(stmt, KernelCall):
            self._compile_kernel_call(stmt)
        elif isinstance(stmt, InjectF):
            self._compile_inject_f(stmt)
        elif isinstance(stmt, Proposal):
            self._compile_proposal(stmt)

    def _compile_var_decl(self, var: VarDecl):
        """Compile variable declaration."""
        # Allocate register
        reg = self.regs.alloc_scalar(var.name)

        # Initialize if value provided
        if var.initializer:
            val_reg = self._compile_expression(var.initializer)
            if val_reg != reg:
                # Move value to allocated register
                self._emit(Instruction(Opcode.ADD, dst=reg, src0=val_reg, src1=0))  # ADD with 0
            self.regs.free_scalar(val_reg)

    def _compile_witness_decl(self, witness: WitnessDecl):
        """Compile witness binding (::=)."""
        # Allocate token register
        tok_reg = self.regs.alloc_token(witness.name)

        # Compile expression (should produce witness token)
        expr_reg = self._compile_expression(witness.expression)

        # Store result as token
        self._emit(Instruction(Opcode.ST_TOK, dst=tok_reg, src0=expr_reg))

        self.regs.free_scalar(expr_reg)

    def _compile_assignment(self, assign: Assignment):
        """Compile assignment statement."""
        target_reg = self.regs.lookup(assign.target.name if isinstance(assign.target, Identifier) else "")

        if target_reg is None:
            # New variable
            target_reg = self.regs.alloc_scalar(assign.target.name if isinstance(assign.target, Identifier) else "")

        val_reg = self._compile_expression(assign.value)

        if val_reg != target_reg:
            self._emit(Instruction(Opcode.ADD, dst=target_reg, src0=val_reg, src1=0))

        if val_reg != target_reg:
            self.regs.free_scalar(val_reg)

    def _compile_expr_stmt(self, stmt: ExpressionStmt):
        """Compile expression statement (for side effects)."""
        reg = self._compile_expression(stmt.expression)
        self.regs.free_scalar(reg)

    def _compile_if_past(self, if_stmt: IfPast):
        """
        Compile if_past statement.

        if_past uses past tokens for branching - this is the only
        allowed form of conditional in Existence-Lang.
        """
        # Generate labels
        else_label = f"_else_{len(self.labels)}"
        end_label = f"_endif_{len(self.labels)}"

        # Compile condition (should be token comparison)
        cond_reg = self._compile_expression(if_stmt.condition)

        # Branch to else if condition is false/zero
        # Use BR_TOK to branch on token value
        self._emit(Instruction(Opcode.BR_TOK, dst=cond_reg, imm=0))  # Placeholder
        branch_idx = len(self.instructions) - 1
        self._record_pending_label(else_label, branch_idx, 'imm')

        # Compile then block
        self._compile_block(if_stmt.then_block)

        # Jump to end
        self._emit(Instruction(Opcode.BR_TOK, imm=0))  # Unconditional jump
        jump_idx = len(self.instructions) - 1
        self._record_pending_label(end_label, jump_idx, 'imm')

        # Else label
        self._define_label(else_label)

        # Compile else block if present
        if if_stmt.else_block:
            self._compile_block(if_stmt.else_block)

        # End label
        self._define_label(end_label)

        self.regs.free_scalar(cond_reg)

    def _compile_repeat_past(self, repeat: RepeatPast):
        """
        Compile repeat_past statement.

        Loop count must come from a past token.
        """
        # Compile count expression
        count_reg = self._compile_expression(repeat.count)

        # Allocate loop counter
        counter_reg = self.regs.alloc_scalar()

        # Initialize counter to 0
        self._emit(Instruction(Opcode.LDI, dst=counter_reg, imm=0))

        # Loop label
        loop_label = f"_loop_{len(self.labels)}"
        end_label = f"_endloop_{len(self.labels)}"

        self._define_label(loop_label)

        # Compare counter < count
        cmp_tok = self.regs.alloc_token()
        self._emit(Instruction(Opcode.CMP, dst=cmp_tok, src0=counter_reg, src1=count_reg))

        # Branch to end if counter >= count (token != LT)
        self._emit(Instruction(Opcode.BR_TOK, dst=cmp_tok, imm=0))
        branch_idx = len(self.instructions) - 1
        self._record_pending_label(end_label, branch_idx, 'imm')

        # Compile body
        self._compile_block(repeat.body)

        # Increment counter
        one_reg = self.regs.alloc_scalar()
        self._emit(Instruction(Opcode.LDI, dst=one_reg, imm=1))
        self._emit(Instruction(Opcode.ADD, dst=counter_reg, src0=counter_reg, src1=one_reg))
        self.regs.free_scalar(one_reg)

        # Jump back to loop
        self._emit(Instruction(Opcode.BR_TOK, imm=0))  # Unconditional
        jump_idx = len(self.instructions) - 1
        self._record_pending_label(loop_label, jump_idx, 'imm')

        # End label
        self._define_label(end_label)

        self.regs.free_scalar(counter_reg)
        self.regs.free_token(cmp_tok)
        self.regs.free_scalar(count_reg)

    def _compile_while_past(self, while_stmt: WhilePast):
        """Compile while_past loop."""
        loop_label = f"_while_{len(self.labels)}"
        end_label = f"_endwhile_{len(self.labels)}"

        self._define_label(loop_label)

        # Compile condition
        cond_reg = self._compile_expression(while_stmt.condition)

        # Branch to end if false
        self._emit(Instruction(Opcode.BR_TOK, dst=cond_reg, imm=0))
        branch_idx = len(self.instructions) - 1
        self._record_pending_label(end_label, branch_idx, 'imm')

        # Compile body
        self._compile_block(while_stmt.body)

        # Jump back to loop
        self._emit(Instruction(Opcode.BR_TOK, imm=0))
        jump_idx = len(self.instructions) - 1
        self._record_pending_label(loop_label, jump_idx, 'imm')

        self._define_label(end_label)
        self.regs.free_scalar(cond_reg)

    def _compile_return(self, ret: Return):
        """Compile return statement."""
        if ret.value:
            val_reg = self._compile_expression(ret.value)
            # Move to R0 (return register)
            if val_reg != 0:
                self._emit(Instruction(Opcode.ADD, dst=0, src0=val_reg, src1=0))
            self.regs.free_scalar(val_reg)
        self._emit(Instruction(Opcode.RET))

    def _compile_kernel_call(self, call: KernelCall):
        """Compile kernel call."""
        # Look up kernel ID
        kernel_id = list(self.kernels.keys()).index(call.kernel_name) if call.kernel_name in self.kernels else 0

        # Compile arguments and place in registers
        for i, arg in enumerate(call.arguments):
            arg_reg = self._compile_expression(arg)
            if arg_reg != i:
                self._emit(Instruction(Opcode.ADD, dst=i, src0=arg_reg, src1=0))
            if arg_reg != i:
                self.regs.free_scalar(arg_reg)

        # CALL instruction
        self._emit(Instruction(Opcode.CALL, imm=kernel_id))

    def _compile_inject_f(self, inject: InjectF):
        """Compile inject_F statement."""
        # Compile amount
        amount_reg = self._compile_expression(inject.amount)

        # Get node reference (self node)
        self._emit(Instruction(Opcode.GETSELF, dst=16))  # H0

        # Add F to node
        self._emit(Instruction(Opcode.LDN, dst=amount_reg + 1, src0=16, imm=NodeField.F))
        self._emit(Instruction(Opcode.ADD, dst=amount_reg + 1, src0=amount_reg + 1, src1=amount_reg))
        self._emit(Instruction(Opcode.ST_NODE, src0=amount_reg + 1, imm=NodeField.F))

        self.regs.free_scalar(amount_reg)

    def _compile_proposal(self, proposal: Proposal):
        """
        Compile proposal statement.

        V2 mode emits:
        - V2_PROP_NEW: Begin new proposal
        - V2_PROP_SCORE: Set proposal score
        - V2_PROP_EFFECT: Set proposal effect
        - V2_PROP_ARG: Add effect arguments
        - V2_PROP_END: Finalize proposal
        """
        # Push scope for proposal-local variables
        self.regs.push_scope(f"proposal:{proposal.name}")

        # Compile any local statements in proposal (computed values)
        if hasattr(proposal, 'statements') and proposal.statements:
            for stmt in proposal.statements:
                self._compile_statement(stmt)

        # Allocate ref register for proposal handle
        prop_ref = self.regs.alloc_ref()
        prop_idx = prop_ref  # H register index (now class-local)

        if self.use_v2:
            # V2: PROP_NEW - H[dst] = new proposal
            self._emit_raw(V2_PROP_NEW, prop_idx, 0, 0, 0)

            # V2: Compile and set score
            score_reg = self._compile_expression(proposal.score)
            self._emit_raw(V2_PROP_SCORE, prop_idx, score_reg, 0, 0)
            self.regs.free_scalar(score_reg)

            # V2: Compile effect block - each statement becomes V2_PROP_EFFECT + V2_PROP_ARG
            if proposal.effect and hasattr(proposal.effect, 'statements'):
                for stmt in proposal.effect.statements:
                    self._compile_proposal_effect_v2(prop_idx, stmt)

            # V2: PROP_END - finalize proposal
            self._emit_raw(V2_PROP_END, prop_idx, 0, 0, 0)
        else:
            # V1: Original implementation
            self._emit(Instruction(Opcode.PROP_BEGIN, dst=prop_ref))

            # Compile score expression
            score_reg = self._compile_expression(proposal.score)
            self._emit(Instruction(Opcode.PROP_SCORE, dst=prop_ref, src0=score_reg))
            self.regs.free_scalar(score_reg)

            # Compile effect block
            if proposal.effect and hasattr(proposal.effect, 'statements'):
                for stmt in proposal.effect.statements:
                    self._compile_statement(stmt)

            # End proposal
            self._emit(Instruction(Opcode.PROP_END, dst=prop_ref))

        self.regs.free_ref(prop_ref)

        # Pop proposal scope - frees local variables
        self.regs.pop_scope()

    def _compile_proposal_effect_v2(self, prop_idx: int, effect: Statement):
        """
        Compile a proposal effect for v2 substrate.

        Effects are encoded as:
        - V2_PROP_EFFECT: effect_id (transfer=1, store=2, etc.)
        - V2_PROP_ARG: argument values
        """
        # Determine effect type from statement
        if isinstance(effect, Assignment):
            # Store effect
            effect_id = 2  # EFFECT_STORE
            self._emit_raw(V2_PROP_EFFECT, prop_idx, 0, 0, effect_id)

            # Compile target and value as arguments
            if isinstance(effect.target, Identifier):
                target_reg = self.regs.lookup(effect.target.name)
                if target_reg is not None:
                    self._emit_raw(V2_PROP_ARG, prop_idx, target_reg, 0, 0)

            val_reg = self._compile_expression(effect.value)
            self._emit_raw(V2_PROP_ARG, prop_idx, val_reg, 0, 1)
            self.regs.free_scalar(val_reg)

        elif isinstance(effect, KernelCall):
            if effect.kernel_name == 'transfer':
                effect_id = 1  # EFFECT_TRANSFER
                self._emit_raw(V2_PROP_EFFECT, prop_idx, 0, 0, effect_id)

                # Compile transfer arguments
                for i, arg in enumerate(effect.arguments):
                    arg_reg = self._compile_expression(arg)
                    self._emit_raw(V2_PROP_ARG, prop_idx, arg_reg, 0, i)
                    self.regs.free_scalar(arg_reg)
            else:
                # Generic kernel call effect
                kernel_id = list(self.kernels.keys()).index(effect.kernel_name) if effect.kernel_name in self.kernels else 0
                effect_id = 0x10 + kernel_id  # EFFECT_KERNEL_BASE + id
                self._emit_raw(V2_PROP_EFFECT, prop_idx, 0, 0, effect_id)

                for i, arg in enumerate(effect.arguments):
                    arg_reg = self._compile_expression(arg)
                    self._emit_raw(V2_PROP_ARG, prop_idx, arg_reg, 0, i)
                    self.regs.free_scalar(arg_reg)
        else:
            # Generic statement - compile as nested
            self._compile_statement(effect)

    # ==========================================================================
    # Expression Compilation
    # ==========================================================================

    def _compile_expression(self, expr: Expression) -> int:
        """
        Compile expression to instructions.

        Returns the register containing the result.
        """
        if isinstance(expr, Literal):
            return self._compile_literal(expr)
        elif isinstance(expr, Identifier):
            return self._compile_identifier(expr)
        elif isinstance(expr, BinaryOp):
            return self._compile_binary_op(expr)
        elif isinstance(expr, UnaryOp):
            return self._compile_unary_op(expr)
        elif isinstance(expr, Call):
            return self._compile_call(expr)
        elif isinstance(expr, FieldAccess):
            return self._compile_field_access(expr)
        elif isinstance(expr, Compare):
            return self._compile_compare(expr)
        elif isinstance(expr, Distinct):
            return self._compile_distinct(expr)
        elif isinstance(expr, This):
            return self._compile_this(expr)
        elif isinstance(expr, TupleExpr):
            # For tuples, compile first element
            if expr.elements:
                return self._compile_expression(expr.elements[0])
            return self.regs.alloc_scalar()
        elif isinstance(expr, PrimitiveCallExpr):
            return self._compile_primitive_call(expr)
        else:
            # Unknown expression type
            return self.regs.alloc_scalar()

    def _compile_literal(self, lit: Literal) -> int:
        """Compile literal value."""
        reg = self.regs.alloc_scalar()

        if isinstance(lit.value, (int, float)):
            val = int(lit.value)
            if -256 <= val < 256:
                self._emit(Instruction(Opcode.LDI, dst=reg, imm=val))
            else:
                # Extended immediate
                self._emit(Instruction(Opcode.LDI_EXT, dst=reg, ext=val))
        elif isinstance(lit.value, bool):
            self._emit(Instruction(Opcode.LDI, dst=reg, imm=1 if lit.value else 0))
        else:
            # String or other - default to 0
            self._emit(Instruction(Opcode.LDI, dst=reg, imm=0))

        return reg

    def _compile_identifier(self, ident: Identifier) -> int:
        """Compile identifier reference."""
        reg = self.regs.lookup(ident.name)
        if reg is not None:
            return reg

        # Unknown identifier - allocate new register
        return self.regs.alloc_scalar(ident.name)

    def _compile_binary_op(self, binop: BinaryOp) -> int:
        """Compile binary operation."""
        left_reg = self._compile_expression(binop.left)
        right_reg = self._compile_expression(binop.right)
        result_reg = self.regs.alloc_scalar()

        op_map = {
            '+': Opcode.ADD,
            '-': Opcode.SUB,
            '*': Opcode.MUL,
            '/': Opcode.DIV,
        }

        if binop.operator in op_map:
            self._emit(Instruction(op_map[binop.operator], dst=result_reg, src0=left_reg, src1=right_reg))
        elif binop.operator in ('<', '<=', '>', '>=', '==', '!='):
            # Comparison - produce token
            tok_reg = self.regs.alloc_token()
            self._emit(Instruction(Opcode.CMP, dst=tok_reg, src0=left_reg, src1=right_reg))
            # Convert token to scalar for boolean result
            self._emit(Instruction(Opcode.LDT, dst=result_reg, src0=tok_reg))
            self.regs.free_token(tok_reg)
        else:
            # Default: add
            self._emit(Instruction(Opcode.ADD, dst=result_reg, src0=left_reg, src1=right_reg))

        self.regs.free_scalar(left_reg)
        self.regs.free_scalar(right_reg)
        return result_reg

    def _compile_unary_op(self, unop: UnaryOp) -> int:
        """Compile unary operation."""
        operand_reg = self._compile_expression(unop.operand)
        result_reg = self.regs.alloc_scalar()

        if unop.operator == '-':
            self._emit(Instruction(Opcode.NEG, dst=result_reg, src0=operand_reg))
        elif unop.operator == '!':
            # Logical not - compare with 0
            zero_reg = self.regs.alloc_scalar()
            self._emit(Instruction(Opcode.LDI, dst=zero_reg, imm=0))
            tok_reg = self.regs.alloc_token()
            self._emit(Instruction(Opcode.CMP, dst=tok_reg, src0=operand_reg, src1=zero_reg))
            self._emit(Instruction(Opcode.LDT, dst=result_reg, src0=tok_reg))
            self.regs.free_scalar(zero_reg)
            self.regs.free_token(tok_reg)
        else:
            # Default: copy
            self._emit(Instruction(Opcode.ADD, dst=result_reg, src0=operand_reg, src1=0))

        self.regs.free_scalar(operand_reg)
        return result_reg

    def _compile_call(self, call: Call) -> int:
        """Compile function/kernel call."""
        result_reg = self.regs.alloc_scalar()

        # Check for built-in functions
        if isinstance(call.callee, Identifier):
            name = call.callee.name

            if name == 'transfer':
                return self._compile_transfer(call)
            elif name == 'diffuse':
                return self._compile_diffuse(call)
            elif name == 'compare':
                return self._compile_compare_call(call)
            elif name == 'distinct':
                return self._compile_distinct_call(call)
            elif name == 'sqrt':
                if call.arguments:
                    arg_reg = self._compile_expression(call.arguments[0])
                    self._emit(Instruction(Opcode.SQRT, dst=result_reg, src0=arg_reg))
                    self.regs.free_scalar(arg_reg)
                return result_reg
            elif name == 'abs':
                if call.arguments:
                    arg_reg = self._compile_expression(call.arguments[0])
                    self._emit(Instruction(Opcode.ABS, dst=result_reg, src0=arg_reg))
                    self.regs.free_scalar(arg_reg)
                return result_reg
            elif name == 'min':
                if len(call.arguments) >= 2:
                    a_reg = self._compile_expression(call.arguments[0])
                    b_reg = self._compile_expression(call.arguments[1])
                    self._emit(Instruction(Opcode.MIN, dst=result_reg, src0=a_reg, src1=b_reg))
                    self.regs.free_scalar(a_reg)
                    self.regs.free_scalar(b_reg)
                return result_reg
            elif name == 'max':
                if len(call.arguments) >= 2:
                    a_reg = self._compile_expression(call.arguments[0])
                    b_reg = self._compile_expression(call.arguments[1])
                    self._emit(Instruction(Opcode.MAX, dst=result_reg, src0=a_reg, src1=b_reg))
                    self.regs.free_scalar(a_reg)
                    self.regs.free_scalar(b_reg)
                return result_reg

        # Generic kernel call
        if isinstance(call.callee, Identifier) and call.callee.name in self.kernels:
            kernel_id = list(self.kernels.keys()).index(call.callee.name)
            for i, arg in enumerate(call.arguments):
                arg_reg = self._compile_expression(arg)
                if arg_reg != i:
                    self._emit(Instruction(Opcode.ADD, dst=i, src0=arg_reg, src1=0))
                if arg_reg != i:
                    self.regs.free_scalar(arg_reg)
            self._emit(Instruction(Opcode.CALL, imm=kernel_id))
            # Result in R0
            if result_reg != 0:
                self._emit(Instruction(Opcode.ADD, dst=result_reg, src0=0, src1=0))

        return result_reg

    def _compile_transfer(self, call: Call) -> int:
        """Compile transfer() call to XFER instruction."""
        result_reg = self.regs.alloc_scalar()

        if len(call.arguments) >= 3:
            # transfer(src, dst, amount)
            src_reg = self._compile_expression(call.arguments[0])
            dst_reg = self._compile_expression(call.arguments[1])
            amount_reg = self._compile_expression(call.arguments[2])

            # Store refs in H registers
            src_ref = self.regs.alloc_ref()
            dst_ref = self.regs.alloc_ref()
            self._emit(Instruction(Opcode.MKNODE, dst=src_ref, imm=0))  # Placeholder
            self._emit(Instruction(Opcode.MKNODE, dst=dst_ref, imm=0))

            # XFER instruction
            self._emit(Instruction(Opcode.XFER, dst=amount_reg, src0=src_ref, src1=dst_ref))

            # Result token in T0
            self._emit(Instruction(Opcode.LDT, dst=result_reg, src0=24))  # T0

            self.regs.free_scalar(src_reg)
            self.regs.free_scalar(dst_reg)
            self.regs.free_scalar(amount_reg)
            self.regs.free_ref(src_ref)
            self.regs.free_ref(dst_ref)

        return result_reg

    def _compile_diffuse(self, call: Call) -> int:
        """Compile diffuse() call to DIFFUSE instruction."""
        result_reg = self.regs.alloc_scalar()

        if len(call.arguments) >= 2:
            a_reg = self._compile_expression(call.arguments[0])
            b_reg = self._compile_expression(call.arguments[1])

            # Allocate refs
            a_ref = self.regs.alloc_ref()
            b_ref = self.regs.alloc_ref()

            # Get sigma from third arg or default
            sigma_reg = self.regs.alloc_scalar()
            if len(call.arguments) >= 3:
                self.regs.free_scalar(sigma_reg)
                sigma_reg = self._compile_expression(call.arguments[2])
            else:
                self._emit(Instruction(Opcode.LDI, dst=sigma_reg, imm=1))

            # DIFFUSE instruction
            self._emit(Instruction(Opcode.DIFFUSE, dst=sigma_reg, src0=a_ref, src1=b_ref))

            # Return flux magnitude
            self._emit(Instruction(Opcode.ADD, dst=result_reg, src0=sigma_reg, src1=0))

            self.regs.free_scalar(a_reg)
            self.regs.free_scalar(b_reg)
            self.regs.free_ref(a_ref)
            self.regs.free_ref(b_ref)
            self.regs.free_scalar(sigma_reg)

        return result_reg

    def _compile_compare_call(self, call: Call) -> int:
        """Compile compare() call."""
        tok_reg = self.regs.alloc_token()

        if len(call.arguments) >= 2:
            a_reg = self._compile_expression(call.arguments[0])
            b_reg = self._compile_expression(call.arguments[1])
            self._emit(Instruction(Opcode.CMP, dst=tok_reg, src0=a_reg, src1=b_reg))
            self.regs.free_scalar(a_reg)
            self.regs.free_scalar(b_reg)

        return tok_reg

    def _compile_distinct_call(self, call: Call) -> int:
        """Compile distinct() call."""
        result_reg = self.regs.alloc_scalar()
        result2_reg = self.regs.alloc_scalar()

        self._emit(Instruction(Opcode.DISTINCT, dst=result_reg, src0=result2_reg))

        self.regs.free_scalar(result2_reg)
        return result_reg

    def _compile_field_access(self, access: FieldAccess) -> int:
        """Compile field access expression."""
        result_reg = self.regs.alloc_scalar()

        # Compile object
        obj_reg = self._compile_expression(access.object)

        # Map field name to field ID
        field_map = {
            'F': NodeField.F,
            'q': NodeField.Q,
            'a': NodeField.A,
            'theta': NodeField.THETA,
            'r': NodeField.R,
            'k': NodeField.K,
            'sigma': NodeField.SIGMA,
            'P': NodeField.P,
            'tau': NodeField.TAU,
            'H': NodeField.H,
        }

        if access.field in field_map:
            # Object should be a node reference
            ref_reg = self.regs.alloc_ref()
            self._emit(Instruction(Opcode.ADD, dst=ref_reg, src0=obj_reg, src1=0))
            self._emit(Instruction(Opcode.LDN, dst=result_reg, src0=ref_reg, imm=field_map[access.field]))
            self.regs.free_ref(ref_reg)
        else:
            # Unknown field - copy object
            self._emit(Instruction(Opcode.ADD, dst=result_reg, src0=obj_reg, src1=0))

        self.regs.free_scalar(obj_reg)
        return result_reg

    def _compile_compare(self, compare: Compare) -> int:
        """Compile Compare expression."""
        tok_reg = self.regs.alloc_token()

        left_reg = self._compile_expression(compare.left)
        right_reg = self._compile_expression(compare.right)

        if compare.epsilon:
            # CMP_EPS with epsilon
            eps_reg = self._compile_expression(compare.epsilon)
            self._emit(Instruction(Opcode.CMP_EPS, dst=tok_reg, src0=left_reg, src1=right_reg, ext=eps_reg))
            self.regs.free_scalar(eps_reg)
        else:
            self._emit(Instruction(Opcode.CMP, dst=tok_reg, src0=left_reg, src1=right_reg))

        self.regs.free_scalar(left_reg)
        self.regs.free_scalar(right_reg)
        return tok_reg

    def _compile_distinct(self, distinct: Distinct) -> int:
        """Compile distinct() expression."""
        return self._compile_distinct_call(Call(callee=Identifier(name="distinct"), arguments=[]))

    def _compile_this(self, this: This) -> int:
        """Compile 'this' reference."""
        ref_reg = self.regs.alloc_ref()
        self._emit(Instruction(Opcode.GETSELF, dst=ref_reg))
        return ref_reg

    def _compile_primitive_call(self, prim: 'PrimitiveCallExpr') -> int:
        """
        Compile primitive("name", args...) expression.

        Primitives are external I/O functions provided by the substrate.
        At runtime, the VM will dispatch to the primitive registry.

        Encoding:
        - V2_PRIM opcode with primitive name index in ext field
        - Arguments passed in R0-R7
        - Result returned in dst register
        - imm = arg_count
        """
        result_reg = self.regs.alloc_scalar()

        # Map primitive name to index for compact encoding
        # Well-known primitives get stable indices
        PRIM_NAMES = {
            'llm_call': 0, 'llm_chat': 1,
            'exec': 2, 'exec_safe': 3,
            'file_read': 4, 'file_write': 5, 'file_exists': 6, 'file_list': 7,
            'now': 8, 'now_iso': 9, 'sleep': 10,
            'random': 11, 'random_int': 12, 'random_seed': 13,
            'print': 14, 'log': 15,
            'hash_sha256': 16,
            # Terminal primitives (Phase 19)
            'terminal_read': 17, 'terminal_write': 18, 'terminal_prompt': 19,
            'terminal_clear': 20, 'terminal_color': 21,
        }
        name_id = PRIM_NAMES.get(prim.primitive_name, hash(prim.primitive_name) & 0xFFFF)

        # Compile arguments and place in registers R0-R7
        for i, arg in enumerate(prim.arguments[:8]):  # Max 8 args
            arg_reg = self._compile_expression(arg)
            if arg_reg != i:
                # Move to argument register
                self._emit(Instruction(Opcode.ADD, dst=i, src0=arg_reg, src1=0))
                self.regs.free_scalar(arg_reg)

        # Emit V2_PRIM instruction
        # Format: V2_PRIM dst, arg_count, ext=name_id
        arg_count = len(prim.arguments)
        self._emit(Instruction(Opcode.V2_PRIM, dst=result_reg, imm=arg_count, ext=name_id))

        return result_reg

    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    def _emit(self, instr: Instruction):
        """Emit an instruction."""
        self.instructions.append(instr)

    def _emit_raw(self, opcode: int, dst: int = 0, src0: int = 0, src1: int = 0, imm: int = 0):
        """
        Emit a raw instruction by opcode value.

        Used for v2 opcodes that aren't in the Opcode enum.
        Creates a pseudo-instruction that will be encoded directly.
        """
        # Create instruction with raw opcode
        # We use a wrapper that stores the raw opcode value
        instr = Instruction(
            opcode=opcode,  # This will be the raw value
            dst=dst,
            src0=src0,
            src1=src1,
            imm=imm
        )
        self.instructions.append(instr)

    def _define_label(self, name: str):
        """Define a label at current position."""
        self.labels[name] = len(self.instructions)

        # Resolve pending references
        if name in self.pending_labels:
            for idx, field_name in self.pending_labels[name]:
                # Patch the instruction
                offset = self.labels[name] - idx
                instr = self.instructions[idx]
                # Create new instruction with patched imm
                self.instructions[idx] = Instruction(
                    instr.opcode, instr.dst, instr.src0, instr.src1,
                    offset, instr.ext
                )
            del self.pending_labels[name]

    def _record_pending_label(self, name: str, instr_idx: int, field: str):
        """Record a pending label reference to be patched."""
        if name not in self.pending_labels:
            self.pending_labels[name] = []
        self.pending_labels[name].append((instr_idx, field))

    def _reset_compilation(self):
        """Reset compilation state for new code section."""
        self.regs.reset()
        self.instructions = []
        self.labels = {}
        self.pending_labels = {}

    def _finalize_bytecode(self) -> bytes:
        """Finalize and return bytecode."""
        # Resolve any remaining labels
        for name, refs in self.pending_labels.items():
            if name in self.labels:
                for idx, field_name in refs:
                    offset = self.labels[name] - idx
                    instr = self.instructions[idx]
                    self.instructions[idx] = Instruction(
                        instr.opcode, instr.dst, instr.src0, instr.src1,
                        offset, instr.ext
                    )

        # Encode all instructions
        result = b''
        for instr in self.instructions:
            result += encode_instruction(instr)

        return result


# ==============================================================================
# Public API
# ==============================================================================

def compile_to_eis(program: Program, use_v2: bool = False) -> CompiledProgram:
    """
    Compile an Existence-Lang program to EIS bytecode.

    Args:
        program: Parsed AST program
        use_v2: If True, emit v2 substrate opcodes with phase transitions

    Returns:
        Compiled program with bytecode for each creature/kernel
    """
    compiler = EISCompiler(use_v2=use_v2)
    return compiler.compile(program)


def compile_source(source: str, use_v2: bool = False) -> CompiledProgram:
    """
    Compile Existence-Lang source code to EIS bytecode.

    Args:
        source: Existence-Lang source code
        use_v2: If True, emit v2 substrate opcodes with phase transitions

    Returns:
        Compiled program with bytecode for each creature/kernel
    """
    # Load parser and lexer directly to avoid import issues
    if "det.lang.tokens" not in sys.modules:
        _tokens = _load_module_direct("det.lang.tokens", os.path.join(_BASE_PATH, "tokens.py"))
    else:
        _tokens = sys.modules["det.lang.tokens"]

    if "det.lang.parser" not in sys.modules:
        _parser = _load_module_direct("det.lang.parser", os.path.join(_BASE_PATH, "parser.py"))
    else:
        _parser = sys.modules["det.lang.parser"]

    Lexer = _tokens.Lexer
    Parser = _parser.Parser

    # Parser creates its own lexer, so pass source directly
    parser = Parser(source)
    program = parser.parse()

    return compile_to_eis(program, use_v2=use_v2)
