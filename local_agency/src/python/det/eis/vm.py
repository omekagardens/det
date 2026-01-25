"""
EIS Virtual Machine
===================

Interpreter for EIS bytecode.

Execution model:
    - Lanes: node-lanes (creature kernels) and bond-lanes (flux/grace)
    - Phases: READ → PROPOSE → CHOOSE → COMMIT
    - Proposals are data; CHOOSE selects; COMMIT applies effects
    - Conservation enforced by verified effects
"""

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Tuple, Callable
import math

from .types import (
    NodeRef, BondRef, FieldRef, PropRef, ChoiceRef, BufRef,
    NodeField, BondField, WitnessToken, ValueType,
)
from .encoding import Opcode, Instruction, decode_instruction
from .registers import RegisterFile, LaneType, reg_name
from .memory import (
    TraceStore, ScratchMemory, ProposalBuffer, BoundaryBuffer,
    Effect, EffectType, Proposal,
)
from .phases import Phase, PhaseController, phase_to_name
from .primitives import get_registry, PrimitiveResult


class ExecutionState(IntEnum):
    """VM execution state."""
    RUNNING = 0
    HALTED = 1
    YIELDED = 2
    ERROR = 3
    BREAKPOINT = 4


@dataclass
class Lane:
    """
    An execution lane (node or bond).

    Each lane has its own register file and proposal buffer.
    """
    lane_id: int
    lane_type: LaneType
    registers: RegisterFile
    proposals: ProposalBuffer
    scratch: ScratchMemory

    # Program state
    pc: int = 0
    program: bytes = b''

    # Execution state
    state: ExecutionState = ExecutionState.RUNNING
    error_msg: str = ""

    # Current proposal being built
    current_proposal: Optional[Proposal] = None


class EISVM:
    """
    EIS Virtual Machine.

    Executes EIS bytecode with proper phase semantics.
    """

    def __init__(self, num_nodes: int = 256, max_bonds: int = 1024):
        # Memory
        self.trace = TraceStore(num_nodes, max_bonds)
        self.boundary_buffers: Dict[int, BoundaryBuffer] = {}

        # Phase control
        self.phases = PhaseController(strict=True)

        # Lanes
        self.node_lanes: Dict[int, Lane] = {}
        self.bond_lanes: Dict[Tuple[int, int], Lane] = {}

        # Global state
        self.tick: int = 0
        self.halted: bool = False

        # Debug
        self.trace_execution: bool = False
        self.breakpoints: set = set()

        # Kernel library (callable from CALL instruction)
        self.kernels: Dict[int, Callable] = {}

    # =========================================================================
    # Lane management
    # =========================================================================

    def create_node_lane(self, node_id: int, program: bytes) -> Lane:
        """Create a node lane for executing a creature kernel."""
        regs = RegisterFile(lane_type=LaneType.NODE, lane_id=node_id)
        regs.self_node = NodeRef(node_id)

        lane = Lane(
            lane_id=node_id,
            lane_type=LaneType.NODE,
            registers=regs,
            proposals=ProposalBuffer(),
            scratch=ScratchMemory(),
            program=program,
        )
        self.node_lanes[node_id] = lane
        return lane

    def create_bond_lane(self, node_i: int, node_j: int, program: bytes) -> Lane:
        """Create a bond lane for executing a bond kernel."""
        bond_ref = BondRef.pack(node_i, node_j)
        lane_id = int(bond_ref)

        regs = RegisterFile(lane_type=LaneType.BOND, lane_id=lane_id)
        regs.self_bond = bond_ref

        lane = Lane(
            lane_id=lane_id,
            lane_type=LaneType.BOND,
            registers=regs,
            proposals=ProposalBuffer(),
            scratch=ScratchMemory(),
            program=program,
        )
        self.bond_lanes[(node_i, node_j)] = lane
        return lane

    # =========================================================================
    # Execution
    # =========================================================================

    def step_lane(self, lane: Lane) -> ExecutionState:
        """Execute one instruction in a lane."""
        if lane.state != ExecutionState.RUNNING:
            return lane.state

        if lane.pc >= len(lane.program):
            lane.state = ExecutionState.HALTED
            return lane.state

        # Check breakpoint
        if lane.pc in self.breakpoints:
            lane.state = ExecutionState.BREAKPOINT
            return lane.state

        # Decode instruction
        try:
            instr, consumed = decode_instruction(lane.program, lane.pc)
        except Exception as e:
            lane.state = ExecutionState.ERROR
            lane.error_msg = f"Decode error at PC={lane.pc}: {e}"
            return lane.state

        # Trace
        if self.trace_execution:
            print(f"[{lane.lane_id}] PC={lane.pc:04x} {instr.opcode.name}")

        # Execute
        try:
            self._execute_instruction(lane, instr)
            lane.pc += consumed
        except Exception as e:
            lane.state = ExecutionState.ERROR
            lane.error_msg = f"Execution error at PC={lane.pc}: {e}"

        return lane.state

    def run_lane(self, lane: Lane, max_steps: int = 10000) -> ExecutionState:
        """Run a lane until halt, yield, or max steps."""
        steps = 0
        while steps < max_steps:
            state = self.step_lane(lane)
            if state != ExecutionState.RUNNING:
                return state
            steps += 1
        return lane.state

    def run_tick(self):
        """Run one complete tick across all lanes."""
        self.phases.begin_tick()
        self.tick += 1

        # Clear scratch and proposals
        for lane in self.node_lanes.values():
            lane.scratch.clear()
            lane.proposals.clear()
            lane.pc = 0
            lane.state = ExecutionState.RUNNING

        for lane in self.bond_lanes.values():
            lane.scratch.clear()
            lane.proposals.clear()
            lane.pc = 0
            lane.state = ExecutionState.RUNNING

        # Run through phases
        for phase in [Phase.READ, Phase.PROPOSE, Phase.CHOOSE, Phase.COMMIT]:
            self.phases.set_phase(phase)

            # Run node lanes
            for lane in self.node_lanes.values():
                self.run_lane(lane)
                lane.pc = 0
                lane.state = ExecutionState.RUNNING

            # Run bond lanes
            for lane in self.bond_lanes.values():
                self.run_lane(lane)
                lane.pc = 0
                lane.state = ExecutionState.RUNNING

        self.phases.end_tick()
        self.trace.advance_tick()

    # =========================================================================
    # Instruction execution
    # =========================================================================

    def _execute_instruction(self, lane: Lane, instr: Instruction):
        """Execute a single instruction."""
        op = instr.opcode
        regs = lane.registers

        # === Phase Control ===
        if op == Opcode.NOP:
            pass

        elif op == Opcode.PHASE:
            # Set phase (usually handled by run_tick)
            self.phases.set_phase(Phase(instr.imm))

        elif op == Opcode.HALT:
            lane.state = ExecutionState.HALTED

        elif op == Opcode.YIELD:
            lane.state = ExecutionState.YIELDED

        elif op == Opcode.TICK:
            # Increment local tick counter
            pass

        # === Load Operations ===
        elif op == Opcode.LDI:
            # Load immediate
            regs.write(instr.dst, float(instr.imm))

        elif op == Opcode.LDI_EXT:
            # Load extended immediate (32-bit from ext word)
            if instr.ext is not None:
                regs.write(instr.dst, float(instr.ext))

        elif op == Opcode.LDN:
            # Load node field: LDN dst, nodeRefReg, fieldId
            if self.phases.check_trace_read("LDN"):
                # Get node ref from ref register (H0-H7) or create from scalar
                if 16 <= instr.src0 < 24:
                    ref_idx = instr.src0 - 16
                    node_ref = regs._refs[ref_idx]
                    if not isinstance(node_ref, NodeRef):
                        node_ref = None
                else:
                    node_ref = NodeRef(int(regs.read(instr.src0)))
                if node_ref:
                    field_id = NodeField(instr.imm)
                    value = self.trace.read_node(node_ref, field_id)
                    regs.write(instr.dst, value)

        elif op == Opcode.LDB:
            # Load bond field: LDB dst, bondRefReg, fieldId
            if self.phases.check_trace_read("LDB"):
                # Get bond ref from ref register (H0-H7)
                if 16 <= instr.src0 < 24:
                    ref_idx = instr.src0 - 16
                    bond_ref = regs._refs[ref_idx]
                    if not isinstance(bond_ref, BondRef):
                        bond_ref = None
                else:
                    bond_ref = None
                if bond_ref:
                    field_id = BondField(instr.imm)
                    value = self.trace.read_bond(bond_ref, field_id)
                    regs.write(instr.dst, value)

        elif op == Opcode.LDT:
            # Load token: LDT dst, tokReg
            tok_idx = instr.src0 - 24 if instr.src0 >= 24 else 0
            value = regs._tokens[tok_idx]
            regs.write(instr.dst, float(value))

        # === Store Operations (staged) ===
        elif op == Opcode.ST_TOK:
            # Stage token write
            if self.phases.check_commit("ST_TOK"):
                tok_idx = instr.dst - 24 if instr.dst >= 24 else 0
                value = int(regs.read(instr.src0))
                regs._tokens[tok_idx] = value

        elif op == Opcode.ST_NODE:
            # Stage node field write
            if self.phases.check_commit("ST_NODE"):
                node_ref = regs.get_self_node() or NodeRef(0)
                field_id = NodeField(instr.imm)
                value = regs.read(instr.src0)
                self.trace.write_node(node_ref, field_id, value)

        elif op == Opcode.ST_BOND:
            # Stage bond field write
            if self.phases.check_commit("ST_BOND"):
                bond_ref = regs.get_self_bond()
                if bond_ref:
                    field_id = BondField(instr.imm)
                    value = regs.read(instr.src0)
                    self.trace.write_bond(bond_ref, field_id, value)

        # === Arithmetic ===
        elif op == Opcode.ADD:
            a = regs.read(instr.src0)
            b = regs.read(instr.src1)
            regs.write(instr.dst, a + b)

        elif op == Opcode.SUB:
            a = regs.read(instr.src0)
            b = regs.read(instr.src1)
            regs.write(instr.dst, a - b)

        elif op == Opcode.MUL:
            a = regs.read(instr.src0)
            b = regs.read(instr.src1)
            regs.write(instr.dst, a * b)

        elif op == Opcode.DIV:
            a = regs.read(instr.src0)
            b = regs.read(instr.src1)
            regs.write(instr.dst, a / b if b != 0 else 0.0)

        elif op == Opcode.NEG:
            a = regs.read(instr.src0)
            regs.write(instr.dst, -a)

        elif op == Opcode.ABS:
            a = regs.read(instr.src0)
            regs.write(instr.dst, abs(a))

        # === Math Functions ===
        elif op == Opcode.SQRT:
            a = regs.read(instr.src0)
            regs.write(instr.dst, math.sqrt(max(0, a)))

        elif op == Opcode.MIN:
            a = regs.read(instr.src0)
            b = regs.read(instr.src1)
            regs.write(instr.dst, min(a, b))

        elif op == Opcode.MAX:
            a = regs.read(instr.src0)
            b = regs.read(instr.src1)
            regs.write(instr.dst, max(a, b))

        elif op == Opcode.RELU:
            a = regs.read(instr.src0)
            regs.write(instr.dst, max(0, a))

        elif op == Opcode.SIN:
            a = regs.read(instr.src0)
            regs.write(instr.dst, math.sin(a))

        elif op == Opcode.COS:
            a = regs.read(instr.src0)
            regs.write(instr.dst, math.cos(a))

        # === Comparison ===
        elif op == Opcode.CMP:
            a = regs.read(instr.src0)
            b = regs.read(instr.src1)
            tok_idx = instr.dst - 24 if instr.dst >= 24 else 0
            if a < b:
                regs._tokens[tok_idx] = WitnessToken.LT
            elif a > b:
                regs._tokens[tok_idx] = WitnessToken.GT
            else:
                regs._tokens[tok_idx] = WitnessToken.EQ

        elif op == Opcode.TSET:
            tok_idx = instr.dst - 24 if instr.dst >= 24 else 0
            regs._tokens[tok_idx] = instr.imm

        elif op == Opcode.TMOV:
            src_idx = instr.src0 - 24 if instr.src0 >= 24 else 0
            dst_idx = instr.dst - 24 if instr.dst >= 24 else 0
            regs._tokens[dst_idx] = regs._tokens[src_idx]

        # === Proposal Operations ===
        elif op == Opcode.PROP_BEGIN:
            if self.phases.check_proposal("PROP_BEGIN"):
                lane.current_proposal = lane.proposals.begin_proposal()

        elif op == Opcode.PROP_SCORE:
            if lane.current_proposal:
                score = regs.read(instr.src0)
                lane.current_proposal.score = score

        elif op == Opcode.PROP_EFFECT:
            if lane.current_proposal and instr.ext:
                effect_type = EffectType(instr.imm)
                effect = Effect(effect_type=effect_type)
                lane.current_proposal.add_effect(effect)

        elif op == Opcode.PROP_END:
            lane.current_proposal = None

        # === Choose/Commit ===
        elif op == Opcode.CHOOSE:
            if self.phases.check_choose("CHOOSE"):
                # Choose from proposals
                choice_id = instr.dst
                prop_count = len(lane.proposals.proposals)
                if prop_count > 0:
                    indices = list(range(prop_count))
                    decisiveness = regs.read(instr.src0)
                    seed = int(regs.read(instr.src1))
                    lane.proposals.choose(choice_id, indices, decisiveness, seed)

        elif op == Opcode.COMMIT:
            if self.phases.check_commit("COMMIT"):
                choice_id = instr.src0
                chosen = lane.proposals.get_chosen(choice_id)
                if chosen:
                    self._apply_effects(lane, chosen.effects)
                    chosen.committed = True

        elif op == Opcode.WITNESS:
            if self.phases.check_commit("WITNESS"):
                tok_idx = instr.dst - 24 if instr.dst >= 24 else 0
                value = int(regs.read(instr.src0))
                regs._tokens[tok_idx] = value
                # Log witness
                node_ref = regs.get_self_node()
                if node_ref:
                    self.trace.emit_witness(node_ref.node_id, value)

        # === Reference Operations ===
        elif op == Opcode.MKNODE:
            node_id = instr.imm
            regs.write(instr.dst, NodeRef(node_id), ValueType.NODE_REF)

        elif op == Opcode.GETSELF:
            if lane.lane_type == LaneType.NODE:
                regs.write(instr.dst, regs.self_node, ValueType.NODE_REF)
            else:
                regs.write(instr.dst, regs.self_bond, ValueType.BOND_REF)

        # === Conservation Primitives ===
        elif op == Opcode.XFER:
            if self.phases.check_commit("XFER"):
                src_ref = regs.read(instr.src0)
                dst_ref = regs.read(instr.src1)
                amount = regs.read(instr.dst)  # Using dst as amount reg
                if isinstance(src_ref, NodeRef) and isinstance(dst_ref, NodeRef):
                    actual, token = self.trace.apply_xfer(src_ref, dst_ref, amount)
                    # Store result token
                    regs._tokens[0] = token

        elif op == Opcode.DIFFUSE:
            if self.phases.check_commit("DIFFUSE"):
                node_a = regs.read(instr.src0)
                node_b = regs.read(instr.src1)
                sigma = regs.read(instr.dst)
                if isinstance(node_a, NodeRef) and isinstance(node_b, NodeRef):
                    # Read F values
                    f_a = self.trace.read_node(node_a, NodeField.F)
                    f_b = self.trace.read_node(node_b, NodeField.F)
                    # Compute flux
                    flux = sigma * (f_a - f_b) * 0.5
                    # Apply antisymmetric
                    self.trace.write_node(node_a, NodeField.F, f_a - flux)
                    self.trace.write_node(node_b, NodeField.F, f_b + flux)

        # === Branching ===
        elif op == Opcode.CALL:
            kernel_id = instr.imm
            if kernel_id in self.kernels:
                self.kernels[kernel_id](self, lane)

        elif op == Opcode.RET:
            lane.state = ExecutionState.HALTED

        # === Debug ===
        elif op == Opcode.DEBUG:
            lane.state = ExecutionState.BREAKPOINT

        elif op == Opcode.TRACE:
            if self.trace_execution:
                print(f"TRACE[{lane.lane_id}]: imm={instr.imm}")

        # === V2 Primitive Call ===
        elif op == Opcode.V2_PRIM:
            # V2_PRIM dst, arg_count, ext=name_id
            # Execute substrate primitive and store result
            self._execute_primitive(lane, regs, instr)

        # === V2 Phase Control (pass-through for now) ===
        elif op in (Opcode.V2_PHASE_R, Opcode.V2_PHASE_P,
                    Opcode.V2_PHASE_C, Opcode.V2_PHASE_X):
            # Phase transitions handled by phase controller
            phase_map = {
                Opcode.V2_PHASE_R: Phase.READ,
                Opcode.V2_PHASE_P: Phase.PROPOSE,
                Opcode.V2_PHASE_C: Phase.CHOOSE,
                Opcode.V2_PHASE_X: Phase.COMMIT,
            }
            self.phases.set_phase(phase_map.get(op, Phase.READ))

        # === V2 Proposal Operations (stub) ===
        elif op == Opcode.V2_PROP_NEW:
            # Create new proposal
            if lane.current_proposal is None:
                lane.current_proposal = Proposal(proposal_id=len(lane.proposals.proposals))
            lane.proposals.add_proposal(lane.current_proposal)
            # Store ref in dst
            if 16 <= instr.dst < 24:
                lane.registers._refs[instr.dst - 16] = PropRef(lane.current_proposal.proposal_id)

        elif op == Opcode.V2_PROP_SCORE:
            if lane.current_proposal:
                lane.current_proposal.score = regs.read(instr.src0)

        elif op == Opcode.V2_PROP_END:
            lane.current_proposal = None

        elif op == Opcode.V2_CHOOSE:
            # Select best proposal
            lane.proposals.select()

        elif op == Opcode.V2_COMMIT:
            # Apply selected proposal's effects
            selected = lane.proposals.get_selected()
            if selected:
                self._apply_effects(lane, selected.effects)

        else:
            # Unknown opcode - check if it's a raw integer (for forward compatibility)
            if isinstance(op, int):
                if self.trace_execution:
                    print(f"Unknown opcode: 0x{op:02X}")
            else:
                raise RuntimeError(f"Unimplemented opcode: {op.name}")

    def _execute_primitive(self, lane: Lane, regs: RegisterFile, instr: Instruction):
        """
        Execute a substrate primitive call.

        V2_PRIM format:
        - dst: result register
        - imm: argument count
        - ext: primitive name ID

        Arguments are passed in R0-R7.
        Result is stored in dst register.
        """
        # Primitive name lookup (reverse of compiler's PRIM_NAMES)
        PRIM_ID_TO_NAME = {
            0: 'llm_call', 1: 'llm_chat',
            2: 'exec', 3: 'exec_safe',
            4: 'file_read', 5: 'file_write', 6: 'file_exists', 7: 'file_list',
            8: 'now', 9: 'now_iso', 10: 'sleep',
            11: 'random', 12: 'random_int', 13: 'random_seed',
            14: 'print', 15: 'log',
            16: 'hash_sha256',
            # Terminal primitives (Phase 19)
            17: 'terminal_read', 18: 'terminal_write', 19: 'terminal_prompt',
            20: 'terminal_clear', 21: 'terminal_color',
        }

        name_id = instr.ext if instr.ext is not None else 0
        arg_count = instr.imm
        dst_reg = instr.dst

        # Get primitive name
        prim_name = PRIM_ID_TO_NAME.get(name_id)
        if prim_name is None:
            # Unknown primitive ID - store error result
            regs.write(dst_reg, 0.0)
            return

        # Collect arguments from R0-R7
        args = []
        for i in range(min(arg_count, 8)):
            args.append(regs.read(i))

        # Get creature's F and agency from lane's self node
        node_ref = regs.get_self_node()
        if node_ref:
            available_f = self.trace.read_node(node_ref, NodeField.F)
            agency = self.trace.read_node(node_ref, NodeField.A)
        else:
            available_f = 10.0
            agency = 0.5

        # Call primitive via registry
        registry = get_registry()
        call = registry.call(prim_name, args, available_f, agency)

        # Store result
        if call.result_code == PrimitiveResult.OK:
            # Deduct cost from F
            if node_ref:
                new_f = available_f - call.cost
                self.trace.write_node(node_ref, NodeField.F, new_f)

            # Store result based on type
            if isinstance(call.result, (int, float)):
                regs.write(dst_reg, float(call.result))
            elif isinstance(call.result, bool):
                regs.write(dst_reg, 1.0 if call.result else 0.0)
            elif isinstance(call.result, str):
                # String results stored as hash/token for now
                # Full string handling requires TokenReg extension
                regs.write(dst_reg, float(len(call.result)))
            else:
                regs.write(dst_reg, 1.0)  # Success indicator
        else:
            # Error - store 0
            regs.write(dst_reg, 0.0)

        if self.trace_execution:
            print(f"PRIM[{lane.lane_id}]: {prim_name}({args}) -> {call.result_code.name}")

    def _apply_effects(self, lane: Lane, effects: List[Effect]):
        """Apply a list of effects during COMMIT."""
        for effect in effects:
            if effect.effect_type == EffectType.XFER_F:
                if effect.src_node and effect.dst_node:
                    self.trace.apply_xfer(effect.src_node, effect.dst_node, effect.amount)

            elif effect.effect_type == EffectType.SET_NODE_FIELD:
                if effect.target_ref and isinstance(effect.target_ref, NodeRef):
                    self.trace.write_node(effect.target_ref, NodeField(effect.field_id), effect.value)

            elif effect.effect_type == EffectType.SET_BOND_FIELD:
                if effect.target_ref and isinstance(effect.target_ref, BondRef):
                    self.trace.write_bond(effect.target_ref, BondField(effect.field_id), effect.value)

            elif effect.effect_type == EffectType.SET_TOKEN:
                tok_idx = effect.field_id
                if 0 <= tok_idx < 8:
                    lane.registers._tokens[tok_idx] = effect.value

            elif effect.effect_type == EffectType.WITNESS:
                node_ref = lane.registers.get_self_node()
                if node_ref:
                    self.trace.emit_witness(node_ref.node_id, effect.value)

    # =========================================================================
    # Debug / Inspection
    # =========================================================================

    def get_state(self) -> dict:
        """Get VM state for debugging."""
        return {
            "tick": self.tick,
            "phase": phase_to_name(self.phases.current_phase),
            "trace_tick": self.trace.tick,
            "node_lanes": len(self.node_lanes),
            "bond_lanes": len(self.bond_lanes),
            "halted": self.halted,
            "violations": len(self.phases.violations),
        }

    def dump_lane(self, lane: Lane) -> dict:
        """Dump lane state."""
        return {
            "lane_id": lane.lane_id,
            "lane_type": lane.lane_type.name,
            "pc": lane.pc,
            "state": lane.state.name,
            "error": lane.error_msg,
            "registers": lane.registers.dump(),
            "proposals": len(lane.proposals.proposals),
        }
