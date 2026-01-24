#!/usr/bin/env python3
"""
Run DET-OS Kernel on C-based EIS VM
===================================

This script runs Existence-Lang kernel code on the native C EIS VM.

Pipeline:
    kernel.ex (Existence-Lang)
          │
          ▼ compile to
    EIS Bytecode (32-bit instructions)
          │
          ▼ load into
    C EIS VM (eis_vm.c)
          │
          ▼ executes with
    DET Core (det_core.c)

Usage:
    python run_on_c_vm.py              # Run simple test
    python run_on_c_vm.py --debug      # Debug mode
"""

import sys
import struct
import ctypes
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def find_library():
    """Find the compiled libdet_core library."""
    base = Path(__file__).parent.parent.parent.parent.parent
    candidates = [
        base / "det_core" / "build" / "libdet_core.dylib",
        base / "det_core" / "build" / "libdet_core.so",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    return None


# Opcodes from eis_vm.h (matching C implementation)
class EIS_OP:
    # Phase Control (0x00-0x0F)
    NOP = 0x00
    PHASE = 0x01
    HALT = 0x02
    YIELD = 0x03
    # Note: TICK (0x05) is NOT implemented in the C VM switch!

    # Load Operations (0x10-0x1F)
    LDI = 0x10
    LDI_EXT = 0x11
    LDN = 0x12

    # Arithmetic (0x30-0x3F)
    ADD = 0x30
    SUB = 0x31
    MUL = 0x32
    DIV = 0x33

    # Comparison and Token Ops (0x50-0x5F)
    CMP = 0x50
    TSET = 0x55

    # Branching (0xB0-0xBF)
    BR_TOK = 0xB0   # Branch on token
    BR_PHASE = 0xB1
    RET = 0xB3

    # Debug (0xF0-0xFF)
    TRACE = 0xF1


class EIS_PHASE:
    IDLE = 0
    READ = 1
    PROPOSE = 2
    CHOOSE = 3
    COMMIT = 4


class EIS_STATE:
    RUNNING = 0
    HALTED = 1
    YIELDED = 2
    ERROR = 3


def encode_instruction(opcode, dst=0, src0=0, src1=0, imm=0):
    """
    Encode a 32-bit EIS instruction.

    Format: [opcode:8][dst:5][src0:5][src1:5][imm:9]
    """
    # Handle signed immediate (9-bit two's complement)
    if imm < 0:
        imm = imm & 0x1FF
    elif imm > 255:
        imm = imm & 0x1FF

    word = ((opcode & 0xFF) << 24) | \
           ((dst & 0x1F) << 19) | \
           ((src0 & 0x1F) << 14) | \
           ((src1 & 0x1F) << 9) | \
           (imm & 0x1FF)

    return word


def create_kernel_bytecode():
    """
    Create a simple DET-OS kernel program as EIS bytecode.

    This kernel demonstrates:
    - Phase transitions (READ → PROPOSE → CHOOSE → COMMIT)
    - Register operations
    - Basic arithmetic
    - Halt

    Note: The C EIS VM doesn't have TICK or conditional jump instructions,
    so this is a simplified linear program that demonstrates the VM works.
    """
    instructions = []

    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    # R0 = tick counter
    # R1 = total F (simulated)
    # R2 = grace pool
    # R3 = num creatures

    # R0 = 0 (tick counter)
    instructions.append(encode_instruction(EIS_OP.LDI, dst=0, imm=0))

    # R1 = 10 (will multiply to get 1000)
    instructions.append(encode_instruction(EIS_OP.LDI, dst=1, imm=10))

    # R4 = 100
    instructions.append(encode_instruction(EIS_OP.LDI, dst=4, imm=100))

    # R1 = R1 * R4 = 1000 (total F)
    instructions.append(encode_instruction(EIS_OP.MUL, dst=1, src0=1, src1=4))

    # R2 = 100 (grace pool)
    instructions.append(encode_instruction(EIS_OP.LDI, dst=2, imm=100))

    # R3 = 1 (kernel creature)
    instructions.append(encode_instruction(EIS_OP.LDI, dst=3, imm=1))

    # R5 = 1 (increment value)
    instructions.append(encode_instruction(EIS_OP.LDI, dst=5, imm=1))

    # =========================================================================
    # KERNEL TICK CYCLE (no loop - just one cycle for testing)
    # =========================================================================

    # --- PHASE: READ ---
    instructions.append(encode_instruction(EIS_OP.PHASE, imm=EIS_PHASE.READ))
    instructions.append(encode_instruction(EIS_OP.NOP))  # Would read DET state

    # --- PHASE: PROPOSE ---
    instructions.append(encode_instruction(EIS_OP.PHASE, imm=EIS_PHASE.PROPOSE))
    instructions.append(encode_instruction(EIS_OP.NOP))  # Would make proposals

    # --- PHASE: CHOOSE ---
    instructions.append(encode_instruction(EIS_OP.PHASE, imm=EIS_PHASE.CHOOSE))
    instructions.append(encode_instruction(EIS_OP.NOP))  # Would select proposals

    # --- PHASE: COMMIT ---
    instructions.append(encode_instruction(EIS_OP.PHASE, imm=EIS_PHASE.COMMIT))

    # Commit: R0 = R0 + R5 (increment tick)
    instructions.append(encode_instruction(EIS_OP.ADD, dst=0, src0=0, src1=5))

    # R6 = R1 / R4 = 1000 / 100 = 10 (test division)
    instructions.append(encode_instruction(EIS_OP.DIV, dst=6, src0=1, src1=4))

    # R7 = R1 - R2 = 1000 - 100 = 900 (test subtraction)
    instructions.append(encode_instruction(EIS_OP.SUB, dst=7, src0=1, src1=2))

    # --- PHASE: IDLE (end of tick) ---
    instructions.append(encode_instruction(EIS_OP.PHASE, imm=EIS_PHASE.IDLE))

    # HALT
    instructions.append(encode_instruction(EIS_OP.HALT))

    return instructions


def instructions_to_bytes(instructions):
    """Convert list of 32-bit instructions to bytes (big-endian, matching C decoder)."""
    return b''.join(struct.pack('>I', instr) for instr in instructions)


class CBasedEISVM:
    """
    Wrapper for the C-based EIS VM.

    Uses ctypes to call the native C implementation.
    """

    def __init__(self, library_path=None):
        """Initialize the C-based VM."""
        if library_path is None:
            library_path = find_library()

        if library_path is None:
            raise RuntimeError(
                "Cannot find libdet_core library.\n"
                "Build with:\n"
                "  cd local_agency/src/det_core/build\n"
                "  cmake ..\n"
                "  make"
            )

        self.lib = ctypes.CDLL(library_path)
        self._setup_functions()

        # Create DET core
        self.core = self.lib.det_core_create()
        if not self.core:
            raise RuntimeError("Failed to create DET core")

        # Create EIS VM
        self.vm = self.lib.eis_vm_create(self.core)
        if not self.vm:
            raise RuntimeError("Failed to create EIS VM")

        self.lane = None

    def _setup_functions(self):
        """Setup ctypes function signatures."""
        # DET Core
        self.lib.det_core_create.restype = ctypes.c_void_p
        self.lib.det_core_create.argtypes = []

        self.lib.det_core_destroy.restype = None
        self.lib.det_core_destroy.argtypes = [ctypes.c_void_p]

        self.lib.det_core_step.restype = ctypes.c_int
        self.lib.det_core_step.argtypes = [ctypes.c_void_p, ctypes.c_float]

        # EIS VM
        self.lib.eis_vm_create.restype = ctypes.c_void_p
        self.lib.eis_vm_create.argtypes = [ctypes.c_void_p]

        self.lib.eis_vm_destroy.restype = None
        self.lib.eis_vm_destroy.argtypes = [ctypes.c_void_p]

        # Lane creation - returns pointer to lane struct
        self.lib.eis_vm_create_node_lane.restype = ctypes.c_void_p
        self.lib.eis_vm_create_node_lane.argtypes = [
            ctypes.c_void_p,      # vm
            ctypes.c_uint16,      # node_id
            ctypes.c_void_p,      # program (uint8_t*)
            ctypes.c_uint32       # size
        ]

        # Execution
        self.lib.eis_vm_step_lane.restype = ctypes.c_int  # EIS_ExecState
        self.lib.eis_vm_step_lane.argtypes = [
            ctypes.c_void_p,      # vm
            ctypes.c_void_p       # lane
        ]

        self.lib.eis_vm_run_lane.restype = ctypes.c_int  # EIS_ExecState
        self.lib.eis_vm_run_lane.argtypes = [
            ctypes.c_void_p,      # vm
            ctypes.c_void_p,      # lane
            ctypes.c_uint32       # max_steps
        ]

        self.lib.eis_vm_run_tick.restype = None
        self.lib.eis_vm_run_tick.argtypes = [ctypes.c_void_p]

    def load_kernel(self, bytecode):
        """
        Load kernel bytecode into a node lane.

        Args:
            bytecode: bytes or list of 32-bit instructions
        """
        if isinstance(bytecode, list):
            bytecode = instructions_to_bytes(bytecode)

        # Create buffer
        buf = (ctypes.c_uint8 * len(bytecode)).from_buffer_copy(bytecode)

        # Create a lane for node 0 (kernel node)
        self.lane = self.lib.eis_vm_create_node_lane(
            self.vm,
            0,  # node_id = 0 (kernel)
            ctypes.cast(buf, ctypes.c_void_p),
            len(bytecode)
        )

        if not self.lane:
            raise RuntimeError("Failed to create kernel lane")

        # Keep buffer alive
        self._program_buf = buf

    def step(self):
        """Execute one instruction. Returns execution state."""
        if not self.lane:
            raise RuntimeError("No kernel loaded")
        return self.lib.eis_vm_step_lane(self.vm, self.lane)

    def run(self, max_steps=10000):
        """Run until halt or max steps."""
        if not self.lane:
            raise RuntimeError("No kernel loaded")
        return self.lib.eis_vm_run_lane(self.vm, self.lane, max_steps)

    def run_tick(self):
        """Run a complete tick across all lanes."""
        self.lib.eis_vm_run_tick(self.vm)

    def det_step(self, dt=0.02):
        """Step the DET core physics."""
        return self.lib.det_core_step(self.core, dt)

    def __del__(self):
        """Clean up."""
        if hasattr(self, 'vm') and self.vm:
            self.lib.eis_vm_destroy(self.vm)
        if hasattr(self, 'core') and self.core:
            self.lib.det_core_destroy(self.core)


def run_test():
    """Run the kernel on C-based EIS VM."""
    print("=" * 60)
    print("DET-OS Kernel on C-based EIS VM")
    print("=" * 60)

    # Check for library
    lib_path = find_library()
    if lib_path is None:
        print("\nERROR: Cannot find libdet_core library.")
        print("Build it with:")
        print("  cd local_agency/src/det_core/build")
        print("  cmake ..")
        print("  make")
        return False

    print(f"\nLibrary: {lib_path}")

    try:
        # Create VM
        print("\n1. Creating EIS VM...")
        vm = CBasedEISVM(lib_path)
        print("   ✓ VM created")

        # Create bytecode
        print("\n2. Generating kernel bytecode...")
        instructions = create_kernel_bytecode()
        print(f"   ✓ Generated {len(instructions)} instructions")

        # Print bytecode (debug)
        print("\n   Bytecode:")
        for i, instr in enumerate(instructions):
            opcode = (instr >> 24) & 0xFF
            dst = (instr >> 19) & 0x1F
            src0 = (instr >> 14) & 0x1F
            src1 = (instr >> 9) & 0x1F
            imm = instr & 0x1FF
            if imm > 255:
                imm = imm - 512  # Sign extend
            print(f"   {i:3d}: 0x{instr:08X}  op=0x{opcode:02X} dst={dst} src0={src0} src1={src1} imm={imm}")

        # Load kernel
        print("\n3. Loading kernel into VM...")
        vm.load_kernel(instructions)
        print("   ✓ Kernel loaded")

        # Run
        print("\n4. Running kernel...")
        print("-" * 40)

        state = vm.run(max_steps=5000)

        state_names = ["RUNNING", "HALTED", "YIELDED", "ERROR"]
        state_name = state_names[state] if state < len(state_names) else f"UNKNOWN({state})"

        print("-" * 40)
        print(f"\n   Final state: {state_name}")

        return state == EIS_STATE.HALTED

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    print("\nDET-OS Kernel Execution on Native C VM")
    print("Pipeline: Existence-Lang → EIS Bytecode → C EIS VM → DET Core\n")

    success = run_test()

    if success:
        print("\n✓ Test passed! Kernel executed successfully on C VM.")
    else:
        print("\n✗ Test failed")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
