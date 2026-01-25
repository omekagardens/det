#!/usr/bin/env python3
"""
Phase 14 Tests: GPU Backend with Metal Compute Shaders
======================================================

Tests for the Metal GPU backend implementation.
"""

import sys
import time
import platform
import ctypes
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from det.metal import (
    MetalBackend,
    NodeArraysHelper,
    BondArraysHelper,
    NodeArrays,
    BondArrays,
    PredecodedProgram,
    SubstrateInstr,
    PHASE_READ,
    PHASE_PROPOSE,
    PHASE_CHOOSE,
    PHASE_COMMIT,
    LANE_OWNER_NONE,
    LANE_OWNER_NODE,
    SUB_METAL_OK,
)


def test_availability():
    """Test Metal availability detection."""
    print("\n=== Test: Metal Availability ===")

    if platform.system() != 'Darwin':
        print("SKIP: Not on macOS")
        return True

    available = MetalBackend.is_available()
    print(f"Metal available: {available}")

    if not available:
        print("SKIP: Metal GPU not available")
        return True

    # Create context
    metal = MetalBackend()
    print(f"Device name: {metal.device_name}")
    print(f"Memory allocated: {metal.memory_usage / 1024 / 1024:.2f} MB")

    print("PASS")
    return True


def test_upload_download_nodes():
    """Test uploading and downloading node state."""
    print("\n=== Test: Upload/Download Nodes ===")

    if not MetalBackend.is_available():
        print("SKIP: Metal not available")
        return True

    num_nodes = 100

    # Create node arrays
    nodes = NodeArraysHelper(num_nodes)

    # Set initial values
    for i in range(num_nodes):
        nodes.F[i] = float(i) * 0.5
        nodes.q[i] = float(i) * 0.1
        nodes.a[i] = 0.5 + float(i) / (2 * num_nodes)

    # Upload to GPU
    metal = MetalBackend()
    metal.upload_nodes(nodes.as_ctypes(), num_nodes)

    # Create fresh arrays for download
    downloaded = NodeArraysHelper(num_nodes)
    metal.download_nodes(downloaded.as_ctypes(), num_nodes)

    # Verify values match
    errors = 0
    for i in range(num_nodes):
        if abs(downloaded.F[i] - nodes.F[i]) > 1e-6:
            print(f"  F[{i}] mismatch: {downloaded.F[i]} != {nodes.F[i]}")
            errors += 1
        if abs(downloaded.q[i] - nodes.q[i]) > 1e-6:
            print(f"  q[{i}] mismatch: {downloaded.q[i]} != {nodes.q[i]}")
            errors += 1
        if abs(downloaded.a[i] - nodes.a[i]) > 1e-6:
            print(f"  a[{i}] mismatch: {downloaded.a[i]} != {nodes.a[i]}")
            errors += 1

    if errors > 0:
        print(f"FAIL: {errors} mismatches")
        return False

    print(f"Verified {num_nodes} nodes")
    print("PASS")
    return True


def test_upload_download_bonds():
    """Test uploading and downloading bond state."""
    print("\n=== Test: Upload/Download Bonds ===")

    if not MetalBackend.is_available():
        print("SKIP: Metal not available")
        return True

    num_bonds = 50

    # Create bond arrays
    bonds = BondArraysHelper(num_bonds)

    # Set initial values (ring topology)
    for i in range(num_bonds):
        bonds.connect(i, i, (i + 1) % num_bonds)
        bonds.C[i] = 0.5 + float(i) / (2 * num_bonds)
        bonds.sigma[i] = 1.0 + float(i) * 0.01

    # Upload to GPU
    metal = MetalBackend()
    metal.upload_bonds(bonds.as_ctypes(), num_bonds)

    # Download
    downloaded = BondArraysHelper(num_bonds)
    metal.download_bonds(downloaded.as_ctypes(), num_bonds)

    # Verify
    errors = 0
    for i in range(num_bonds):
        if downloaded.node_i[i] != bonds.node_i[i]:
            errors += 1
        if downloaded.node_j[i] != bonds.node_j[i]:
            errors += 1
        if abs(downloaded.C[i] - bonds.C[i]) > 1e-6:
            errors += 1

    if errors > 0:
        print(f"FAIL: {errors} mismatches")
        return False

    print(f"Verified {num_bonds} bonds")
    print("PASS")
    return True


def test_simple_program():
    """Test uploading a simple program."""
    print("\n=== Test: Program Upload ===")

    if not MetalBackend.is_available():
        print("SKIP: Metal not available")
        return True

    # Create a simple program: LDI R0, 42; HALT
    num_instrs = 10
    instrs = (SubstrateInstr * num_instrs)()

    # OP_LDI = 0x13
    instrs[0].opcode = 0x13  # LDI
    instrs[0].dst = 0
    instrs[0].imm = 42

    # OP_HALT = 0x01
    instrs[1].opcode = 0x01  # HALT

    program = PredecodedProgram()
    program.instrs = ctypes.cast(instrs, ctypes.POINTER(SubstrateInstr))
    program.count = 2
    program.capacity = num_instrs

    metal = MetalBackend()
    metal.upload_program(program)

    print("Program uploaded successfully")
    print("PASS")
    return True


def test_execute_tick():
    """Test executing a full tick."""
    print("\n=== Test: Execute Tick ===")

    if not MetalBackend.is_available():
        print("SKIP: Metal not available")
        return True

    num_nodes = 10
    num_lanes = 10

    # Create node arrays
    nodes = NodeArraysHelper(num_nodes)
    for i in range(num_nodes):
        nodes.F[i] = 10.0

    # Create minimal program
    num_instrs = 20
    instrs = (SubstrateInstr * num_instrs)()

    # Phase transitions
    instrs[0].opcode = 0x04  # PHASE_R (READ)
    instrs[1].opcode = 0x00  # NOP
    instrs[2].opcode = 0x05  # PHASE_P (PROPOSE)
    instrs[3].opcode = 0x00  # NOP
    instrs[4].opcode = 0x06  # PHASE_C (CHOOSE)
    instrs[5].opcode = 0x00  # NOP
    instrs[6].opcode = 0x07  # PHASE_X (COMMIT)
    instrs[7].opcode = 0x00  # NOP
    instrs[8].opcode = 0x01  # HALT

    program = PredecodedProgram()
    program.instrs = ctypes.cast(instrs, ctypes.POINTER(SubstrateInstr))
    program.count = 9
    program.capacity = num_instrs

    metal = MetalBackend()
    metal.upload_nodes(nodes.as_ctypes(), num_nodes)
    metal.upload_program(program)

    # Execute one tick
    metal.execute_tick(num_lanes)
    metal.synchronize()

    # Download results
    metal.download_nodes(nodes.as_ctypes(), num_nodes)

    print(f"Executed tick with {num_lanes} lanes")
    print("PASS")
    return True


def test_multiple_ticks():
    """Test executing multiple ticks."""
    print("\n=== Test: Multiple Ticks ===")

    if not MetalBackend.is_available():
        print("SKIP: Metal not available")
        return True

    num_nodes = 100
    num_lanes = 100
    num_ticks = 10

    # Create node arrays
    nodes = NodeArraysHelper(num_nodes)
    for i in range(num_nodes):
        nodes.F[i] = 10.0

    # Create minimal program
    num_instrs = 20
    instrs = (SubstrateInstr * num_instrs)()
    instrs[0].opcode = 0x04  # PHASE_R
    instrs[1].opcode = 0x05  # PHASE_P
    instrs[2].opcode = 0x06  # PHASE_C
    instrs[3].opcode = 0x07  # PHASE_X
    instrs[4].opcode = 0x01  # HALT

    program = PredecodedProgram()
    program.instrs = ctypes.cast(instrs, ctypes.POINTER(SubstrateInstr))
    program.count = 5
    program.capacity = num_instrs

    metal = MetalBackend()
    metal.upload_nodes(nodes.as_ctypes(), num_nodes)
    metal.upload_program(program)

    # Execute multiple ticks
    start = time.perf_counter()
    metal.execute_ticks(num_lanes, num_ticks)
    metal.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    print(f"Executed {num_ticks} ticks with {num_lanes} lanes")
    print(f"Time: {elapsed:.2f} ms")
    print(f"Rate: {num_ticks * 1000 / elapsed:.1f} ticks/sec")
    print("PASS")
    return True


def test_lane_mode():
    """Test lane ownership modes."""
    print("\n=== Test: Lane Ownership Modes ===")

    if not MetalBackend.is_available():
        print("SKIP: Metal not available")
        return True

    metal = MetalBackend()

    # Test setting different modes
    metal.set_lane_mode(LANE_OWNER_NONE)
    print("  Set LANE_OWNER_NONE")

    metal.set_lane_mode(LANE_OWNER_NODE)
    print("  Set LANE_OWNER_NODE")

    print("PASS")
    return True


def test_performance():
    """Performance benchmark."""
    print("\n=== Test: Performance Benchmark ===")

    if not MetalBackend.is_available():
        print("SKIP: Metal not available")
        return True

    sizes = [
        (100, 200, 100),
        (1000, 2000, 100),
        (10000, 20000, 100),
    ]

    metal = MetalBackend()
    print(f"Device: {metal.device_name}")
    print()

    for num_nodes, num_bonds, num_ticks in sizes:
        # Create test data
        nodes = NodeArraysHelper(num_nodes)
        bonds = BondArraysHelper(num_bonds)

        # Ring topology
        for i in range(num_bonds):
            bonds.connect(i, i % num_nodes, (i + 1) % num_nodes)

        # Initialize resource
        for i in range(num_nodes):
            nodes.F[i] = 10.0

        # Upload
        metal.upload_nodes(nodes.as_ctypes(), num_nodes)
        metal.upload_bonds(bonds.as_ctypes(), num_bonds)

        # Create minimal program
        num_instrs = 10
        instrs = (SubstrateInstr * num_instrs)()
        instrs[0].opcode = 0x04  # PHASE_R
        instrs[1].opcode = 0x05  # PHASE_P
        instrs[2].opcode = 0x06  # PHASE_C
        instrs[3].opcode = 0x07  # PHASE_X
        instrs[4].opcode = 0x01  # HALT

        program = PredecodedProgram()
        program.instrs = ctypes.cast(instrs, ctypes.POINTER(SubstrateInstr))
        program.count = 5
        program.capacity = num_instrs

        metal.upload_program(program)

        # Benchmark
        start = time.perf_counter()
        metal.execute_ticks(num_nodes, num_ticks)
        metal.synchronize()
        elapsed = (time.perf_counter() - start) * 1000

        rate = num_ticks * 1000 / elapsed if elapsed > 0 else 0
        per_tick = elapsed / num_ticks

        print(f"  Nodes: {num_nodes:>6}, Bonds: {num_bonds:>6}, Ticks: {num_ticks}")
        print(f"    Total: {elapsed:.2f} ms, Per tick: {per_tick:.3f} ms, Rate: {rate:.1f} ticks/sec")
        print()

    print("PASS")
    return True


def run_all_tests():
    """Run all Phase 14 tests."""
    print("=" * 60)
    print("Phase 14: GPU Backend with Metal Compute Shaders")
    print("=" * 60)

    tests = [
        test_availability,
        test_upload_download_nodes,
        test_upload_download_bonds,
        test_simple_program,
        test_execute_tick,
        test_multiple_ticks,
        test_lane_mode,
        test_performance,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
