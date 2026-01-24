"""
EIS Substrate v2 - Metal GPU Backend Python Bindings
=====================================================

GPU acceleration for the EIS Substrate using Metal compute shaders on macOS.

Usage:
    from det.metal import MetalBackend

    if MetalBackend.is_available():
        metal = MetalBackend()
        metal.upload_state(nodes, bonds)
        metal.upload_program(program)
        metal.execute_tick(num_lanes=1000)
        metal.download_state(nodes, bonds)
"""

import ctypes
import platform
from pathlib import Path
from typing import Optional, Tuple


# =============================================================================
# CTYPES STRUCTURES (match C headers)
# =============================================================================

class NodeArrays(ctypes.Structure):
    """Node state arrays (SoA layout for GPU efficiency)."""
    _fields_ = [
        ('F', ctypes.POINTER(ctypes.c_float)),           # Resource
        ('q', ctypes.POINTER(ctypes.c_float)),           # Structural debt
        ('a', ctypes.POINTER(ctypes.c_float)),           # Agency [0,1]
        ('sigma', ctypes.POINTER(ctypes.c_float)),       # Processing rate
        ('P', ctypes.POINTER(ctypes.c_float)),           # Presence (computed)
        ('tau', ctypes.POINTER(ctypes.c_float)),         # Proper time
        ('cos_theta', ctypes.POINTER(ctypes.c_float)),   # cos(θ)
        ('sin_theta', ctypes.POINTER(ctypes.c_float)),   # sin(θ)
        ('k', ctypes.POINTER(ctypes.c_uint32)),          # Event count
        ('r', ctypes.POINTER(ctypes.c_uint32)),          # Reconciliation seed
        ('flags', ctypes.POINTER(ctypes.c_uint32)),      # Status flags
        ('num_nodes', ctypes.c_uint32),
        ('capacity', ctypes.c_uint32),
    ]


class BondArrays(ctypes.Structure):
    """Bond state arrays (SoA layout for GPU efficiency)."""
    _fields_ = [
        ('node_i', ctypes.POINTER(ctypes.c_uint32)),     # First node
        ('node_j', ctypes.POINTER(ctypes.c_uint32)),     # Second node
        ('C', ctypes.POINTER(ctypes.c_float)),           # Coherence [0,1]
        ('pi', ctypes.POINTER(ctypes.c_float)),          # Momentum
        ('sigma', ctypes.POINTER(ctypes.c_float)),       # Conductivity
        ('flags', ctypes.POINTER(ctypes.c_uint32)),      # Status flags
        ('num_bonds', ctypes.c_uint32),
        ('capacity', ctypes.c_uint32),
    ]


class SubstrateInstr(ctypes.Structure):
    """Decoded instruction."""
    _fields_ = [
        ('opcode', ctypes.c_uint8),
        ('dst', ctypes.c_uint8),
        ('src0', ctypes.c_uint8),
        ('src1', ctypes.c_uint8),
        ('imm', ctypes.c_int16),
        ('ext', ctypes.c_uint32),
        ('has_ext', ctypes.c_bool),
    ]


class PredecodedProgram(ctypes.Structure):
    """Predecoded program for fast dispatch."""
    _fields_ = [
        ('instrs', ctypes.POINTER(SubstrateInstr)),
        ('count', ctypes.c_uint32),
        ('capacity', ctypes.c_uint32),
    ]


class SubstrateMetalConfig(ctypes.Structure):
    """Metal backend configuration."""
    _fields_ = [
        ('max_nodes', ctypes.c_uint32),
        ('max_bonds', ctypes.c_uint32),
        ('max_lanes', ctypes.c_uint32),
        ('max_proposals', ctypes.c_uint32),
        ('max_instructions', ctypes.c_uint32),
        ('enable_timestamps', ctypes.c_bool),
        ('prefer_discrete_gpu', ctypes.c_bool),
    ]


# Execution phases
PHASE_READ = 0
PHASE_PROPOSE = 1
PHASE_CHOOSE = 2
PHASE_COMMIT = 3

# Lane ownership modes
LANE_OWNER_NONE = 0
LANE_OWNER_NODE = 1
LANE_OWNER_BOND = 2

# Error codes
SUB_METAL_OK = 0
SUB_METAL_ERR_NO_DEVICE = -1
SUB_METAL_ERR_ALLOC = -2
SUB_METAL_ERR_COMPILE = -3
SUB_METAL_ERR_UPLOAD = -4
SUB_METAL_ERR_EXECUTE = -5
SUB_METAL_ERR_DOWNLOAD = -6
SUB_METAL_ERR_INVALID = -7
SUB_METAL_ERR_OVERFLOW = -8


def _find_metal_library() -> Optional[Path]:
    """Find the Metal backend shared library."""
    search_paths = [
        # Relative to this module (in source tree)
        Path(__file__).parent / 'libsubstrate_metal.dylib',
        Path(__file__).parent / '../../../substrate/build/metal/libsubstrate_metal.dylib',
        Path(__file__).parent / '../../../../substrate/build/metal/libsubstrate_metal.dylib',
        Path(__file__).parent / '../../../substrate/metal/libsubstrate_metal.dylib',
        # Absolute path to build directory
        Path('/Volumes/AI_DATA/development/det_local_agency/det/local_agency/src/substrate/build/metal/libsubstrate_metal.dylib'),
        # System paths
        Path('/usr/local/lib/libsubstrate_metal.dylib'),
        Path.home() / '.local/lib/libsubstrate_metal.dylib',
    ]

    # Also check DYLD_LIBRARY_PATH
    import os
    dyld_path = os.environ.get('DYLD_LIBRARY_PATH', '')
    for p in dyld_path.split(':'):
        if p:
            search_paths.insert(0, Path(p) / 'libsubstrate_metal.dylib')

    for path in search_paths:
        if path.exists():
            return path

    return None


class MetalBackend:
    """GPU-accelerated substrate execution via Metal.

    Provides high-performance GPU execution of the EIS Substrate
    using Metal compute shaders on macOS.

    Example:
        metal = MetalBackend()

        # Allocate and initialize node state
        nodes = NodeArraysHelper(1000)
        bonds = BondArraysHelper(2000)

        # Upload to GPU
        metal.upload_nodes(nodes.as_ctypes(), 1000)
        metal.upload_bonds(bonds.as_ctypes(), 2000)
        metal.upload_program(program)

        # Execute
        for _ in range(100):
            metal.execute_tick(num_lanes=1000)

        # Download results
        metal.download_nodes(nodes.as_ctypes(), 1000)
    """

    _lib: Optional[ctypes.CDLL] = None
    _lib_loaded: bool = False

    @classmethod
    def _load_library(cls) -> bool:
        """Load the Metal backend library."""
        if cls._lib_loaded:
            return cls._lib is not None

        cls._lib_loaded = True

        if platform.system() != 'Darwin':
            cls._lib = None
            return False

        lib_path = _find_metal_library()
        if not lib_path:
            cls._lib = None
            return False

        try:
            cls._lib = ctypes.CDLL(str(lib_path))
            cls._setup_bindings()
            return True
        except OSError:
            cls._lib = None
            return False

    @classmethod
    def _setup_bindings(cls):
        """Set up ctypes function signatures."""
        lib = cls._lib
        if not lib:
            return

        # Availability check
        lib.sub_metal_is_available.argtypes = []
        lib.sub_metal_is_available.restype = ctypes.c_int

        # Lifecycle
        lib.sub_metal_create.argtypes = []
        lib.sub_metal_create.restype = ctypes.c_void_p

        lib.sub_metal_create_with_config.argtypes = [ctypes.POINTER(SubstrateMetalConfig)]
        lib.sub_metal_create_with_config.restype = ctypes.c_void_p

        lib.sub_metal_destroy.argtypes = [ctypes.c_void_p]
        lib.sub_metal_destroy.restype = None

        # State transfer
        lib.sub_metal_upload_nodes.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(NodeArrays), ctypes.c_uint32
        ]
        lib.sub_metal_upload_nodes.restype = ctypes.c_int

        lib.sub_metal_upload_bonds.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(BondArrays), ctypes.c_uint32
        ]
        lib.sub_metal_upload_bonds.restype = ctypes.c_int

        lib.sub_metal_upload_program.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(PredecodedProgram)
        ]
        lib.sub_metal_upload_program.restype = ctypes.c_int

        lib.sub_metal_download_nodes.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(NodeArrays), ctypes.c_uint32
        ]
        lib.sub_metal_download_nodes.restype = ctypes.c_int

        lib.sub_metal_download_bonds.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(BondArrays), ctypes.c_uint32
        ]
        lib.sub_metal_download_bonds.restype = ctypes.c_int

        # Execution
        lib.sub_metal_execute_phase.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32
        ]
        lib.sub_metal_execute_phase.restype = ctypes.c_int

        lib.sub_metal_execute_tick.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.sub_metal_execute_tick.restype = ctypes.c_int

        lib.sub_metal_execute_ticks.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32
        ]
        lib.sub_metal_execute_ticks.restype = ctypes.c_int

        lib.sub_metal_synchronize.argtypes = [ctypes.c_void_p]
        lib.sub_metal_synchronize.restype = None

        # Configuration
        lib.sub_metal_set_lane_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.sub_metal_set_lane_mode.restype = None

        lib.sub_metal_set_seed.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.sub_metal_set_seed.restype = None

        lib.sub_metal_set_tick.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.sub_metal_set_tick.restype = None

        # Query
        lib.sub_metal_device_name.argtypes = [ctypes.c_void_p]
        lib.sub_metal_device_name.restype = ctypes.c_char_p

        lib.sub_metal_memory_usage.argtypes = [ctypes.c_void_p]
        lib.sub_metal_memory_usage.restype = ctypes.c_size_t

        lib.sub_metal_get_error.argtypes = [ctypes.c_void_p]
        lib.sub_metal_get_error.restype = ctypes.c_char_p

    @classmethod
    def is_available(cls) -> bool:
        """Check if Metal GPU is available."""
        if platform.system() != 'Darwin':
            return False

        if not cls._load_library():
            return False

        return cls._lib.sub_metal_is_available() != 0

    def __init__(self, config: Optional[SubstrateMetalConfig] = None):
        """Create a Metal backend context.

        Args:
            config: Optional configuration. If None, uses defaults.

        Raises:
            RuntimeError: If Metal is not available or context creation fails.
        """
        if not self._load_library():
            raise RuntimeError("Metal backend library not found")

        if not self._lib.sub_metal_is_available():
            raise RuntimeError("Metal GPU not available")

        if config:
            self._handle = self._lib.sub_metal_create_with_config(ctypes.byref(config))
        else:
            self._handle = self._lib.sub_metal_create()

        if not self._handle:
            raise RuntimeError("Failed to create Metal context")

    def __del__(self):
        """Destroy the Metal context."""
        if hasattr(self, '_handle') and self._handle and self._lib:
            self._lib.sub_metal_destroy(self._handle)
            self._handle = None

    def upload_nodes(self, nodes: NodeArrays, num_nodes: int) -> None:
        """Upload node state to GPU.

        Args:
            nodes: NodeArrays structure with allocated arrays.
            num_nodes: Number of nodes to upload.

        Raises:
            RuntimeError: If upload fails.
        """
        result = self._lib.sub_metal_upload_nodes(
            self._handle, ctypes.byref(nodes), num_nodes
        )
        if result != SUB_METAL_OK:
            raise RuntimeError(f"Failed to upload nodes: {self.get_error()}")

    def upload_bonds(self, bonds: BondArrays, num_bonds: int) -> None:
        """Upload bond state to GPU.

        Args:
            bonds: BondArrays structure with allocated arrays.
            num_bonds: Number of bonds to upload.

        Raises:
            RuntimeError: If upload fails.
        """
        result = self._lib.sub_metal_upload_bonds(
            self._handle, ctypes.byref(bonds), num_bonds
        )
        if result != SUB_METAL_OK:
            raise RuntimeError(f"Failed to upload bonds: {self.get_error()}")

    def upload_program(self, program: PredecodedProgram) -> None:
        """Upload compiled EIS program to GPU.

        Args:
            program: Predecoded program structure.

        Raises:
            RuntimeError: If upload fails.
        """
        result = self._lib.sub_metal_upload_program(
            self._handle, ctypes.byref(program)
        )
        if result != SUB_METAL_OK:
            raise RuntimeError(f"Failed to upload program: {self.get_error()}")

    def download_nodes(self, nodes: NodeArrays, num_nodes: int) -> None:
        """Download node state from GPU.

        Args:
            nodes: NodeArrays structure to fill (must be pre-allocated).
            num_nodes: Number of nodes to download.

        Raises:
            RuntimeError: If download fails.
        """
        result = self._lib.sub_metal_download_nodes(
            self._handle, ctypes.byref(nodes), num_nodes
        )
        if result != SUB_METAL_OK:
            raise RuntimeError(f"Failed to download nodes: {self.get_error()}")

    def download_bonds(self, bonds: BondArrays, num_bonds: int) -> None:
        """Download bond state from GPU.

        Args:
            bonds: BondArrays structure to fill (must be pre-allocated).
            num_bonds: Number of bonds to download.

        Raises:
            RuntimeError: If download fails.
        """
        result = self._lib.sub_metal_download_bonds(
            self._handle, ctypes.byref(bonds), num_bonds
        )
        if result != SUB_METAL_OK:
            raise RuntimeError(f"Failed to download bonds: {self.get_error()}")

    def execute_phase(self, phase: int, num_lanes: int) -> None:
        """Execute a single phase on the GPU.

        Args:
            phase: Phase to execute (PHASE_READ, PHASE_PROPOSE, PHASE_CHOOSE, PHASE_COMMIT).
            num_lanes: Number of parallel lanes.

        Raises:
            RuntimeError: If execution fails.
        """
        result = self._lib.sub_metal_execute_phase(self._handle, phase, num_lanes)
        if result != SUB_METAL_OK:
            raise RuntimeError(f"Failed to execute phase: {self.get_error()}")

    def execute_tick(self, num_lanes: int) -> None:
        """Execute one full tick (all 4 phases).

        Args:
            num_lanes: Number of parallel lanes.

        Raises:
            RuntimeError: If execution fails.
        """
        result = self._lib.sub_metal_execute_tick(self._handle, num_lanes)
        if result != SUB_METAL_OK:
            raise RuntimeError(f"Failed to execute tick: {self.get_error()}")

    def execute_ticks(self, num_lanes: int, num_ticks: int) -> None:
        """Execute multiple ticks.

        Args:
            num_lanes: Number of parallel lanes.
            num_ticks: Number of ticks to execute.

        Raises:
            RuntimeError: If execution fails.
        """
        result = self._lib.sub_metal_execute_ticks(self._handle, num_lanes, num_ticks)
        if result != SUB_METAL_OK:
            raise RuntimeError(f"Failed to execute ticks: {self.get_error()}")

    def synchronize(self) -> None:
        """Wait for all GPU operations to complete."""
        self._lib.sub_metal_synchronize(self._handle)

    def set_lane_mode(self, mode: int) -> None:
        """Set lane ownership mode.

        Args:
            mode: LANE_OWNER_NONE, LANE_OWNER_NODE, or LANE_OWNER_BOND.
        """
        self._lib.sub_metal_set_lane_mode(self._handle, mode)

    def set_seed(self, seed: int) -> None:
        """Set base random seed.

        Args:
            seed: Base seed (each lane derives its own).
        """
        self._lib.sub_metal_set_seed(self._handle, seed)

    def set_tick(self, tick: int) -> None:
        """Set current tick number.

        Args:
            tick: Tick number.
        """
        self._lib.sub_metal_set_tick(self._handle, tick)

    @property
    def device_name(self) -> str:
        """Get the Metal device name."""
        name = self._lib.sub_metal_device_name(self._handle)
        return name.decode() if name else "Unknown"

    @property
    def memory_usage(self) -> int:
        """Get GPU memory usage in bytes."""
        return self._lib.sub_metal_memory_usage(self._handle)

    def get_error(self) -> str:
        """Get the last error message."""
        error = self._lib.sub_metal_get_error(self._handle)
        return error.decode() if error else ""


class NodeArraysHelper:
    """Helper class to create and manage NodeArrays."""

    def __init__(self, num_nodes: int):
        """Create node arrays for the given number of nodes.

        Args:
            num_nodes: Number of nodes to allocate.
        """
        self.num_nodes = num_nodes

        # Allocate arrays
        self.F = (ctypes.c_float * num_nodes)()
        self.q = (ctypes.c_float * num_nodes)()
        self.a = (ctypes.c_float * num_nodes)()
        self.sigma = (ctypes.c_float * num_nodes)()
        self.P = (ctypes.c_float * num_nodes)()
        self.tau = (ctypes.c_float * num_nodes)()
        self.cos_theta = (ctypes.c_float * num_nodes)()
        self.sin_theta = (ctypes.c_float * num_nodes)()
        self.k = (ctypes.c_uint32 * num_nodes)()
        self.r = (ctypes.c_uint32 * num_nodes)()
        self.flags = (ctypes.c_uint32 * num_nodes)()

        # Initialize defaults
        for i in range(num_nodes):
            self.a[i] = 1.0  # Full agency
            self.sigma[i] = 1.0  # Default processing rate
            self.cos_theta[i] = 1.0  # Phase = 0
            self.sin_theta[i] = 0.0

    def as_ctypes(self) -> NodeArrays:
        """Get as ctypes NodeArrays structure."""
        nodes = NodeArrays()
        nodes.F = ctypes.cast(self.F, ctypes.POINTER(ctypes.c_float))
        nodes.q = ctypes.cast(self.q, ctypes.POINTER(ctypes.c_float))
        nodes.a = ctypes.cast(self.a, ctypes.POINTER(ctypes.c_float))
        nodes.sigma = ctypes.cast(self.sigma, ctypes.POINTER(ctypes.c_float))
        nodes.P = ctypes.cast(self.P, ctypes.POINTER(ctypes.c_float))
        nodes.tau = ctypes.cast(self.tau, ctypes.POINTER(ctypes.c_float))
        nodes.cos_theta = ctypes.cast(self.cos_theta, ctypes.POINTER(ctypes.c_float))
        nodes.sin_theta = ctypes.cast(self.sin_theta, ctypes.POINTER(ctypes.c_float))
        nodes.k = ctypes.cast(self.k, ctypes.POINTER(ctypes.c_uint32))
        nodes.r = ctypes.cast(self.r, ctypes.POINTER(ctypes.c_uint32))
        nodes.flags = ctypes.cast(self.flags, ctypes.POINTER(ctypes.c_uint32))
        nodes.num_nodes = self.num_nodes
        nodes.capacity = self.num_nodes
        return nodes


class BondArraysHelper:
    """Helper class to create and manage BondArrays."""

    def __init__(self, num_bonds: int):
        """Create bond arrays for the given number of bonds.

        Args:
            num_bonds: Number of bonds to allocate.
        """
        self.num_bonds = num_bonds

        # Allocate arrays
        self.node_i = (ctypes.c_uint32 * num_bonds)()
        self.node_j = (ctypes.c_uint32 * num_bonds)()
        self.C = (ctypes.c_float * num_bonds)()
        self.pi = (ctypes.c_float * num_bonds)()
        self.sigma = (ctypes.c_float * num_bonds)()
        self.flags = (ctypes.c_uint32 * num_bonds)()

        # Initialize defaults
        for i in range(num_bonds):
            self.C[i] = 0.5  # Default coherence
            self.sigma[i] = 1.0  # Default conductivity

    def as_ctypes(self) -> BondArrays:
        """Get as ctypes BondArrays structure."""
        bonds = BondArrays()
        bonds.node_i = ctypes.cast(self.node_i, ctypes.POINTER(ctypes.c_uint32))
        bonds.node_j = ctypes.cast(self.node_j, ctypes.POINTER(ctypes.c_uint32))
        bonds.C = ctypes.cast(self.C, ctypes.POINTER(ctypes.c_float))
        bonds.pi = ctypes.cast(self.pi, ctypes.POINTER(ctypes.c_float))
        bonds.sigma = ctypes.cast(self.sigma, ctypes.POINTER(ctypes.c_float))
        bonds.flags = ctypes.cast(self.flags, ctypes.POINTER(ctypes.c_uint32))
        bonds.num_bonds = self.num_bonds
        bonds.capacity = self.num_bonds
        return bonds

    def connect(self, bond_id: int, node_i: int, node_j: int) -> None:
        """Set bond endpoints.

        Args:
            bond_id: Bond index.
            node_i: First node ID.
            node_j: Second node ID.
        """
        self.node_i[bond_id] = node_i
        self.node_j[bond_id] = node_j


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def benchmark_cpu_vs_gpu(num_nodes: int = 1000, num_bonds: int = 2000, num_ticks: int = 100):
    """Benchmark CPU vs GPU execution.

    Args:
        num_nodes: Number of nodes.
        num_bonds: Number of bonds.
        num_ticks: Number of ticks to execute.

    Returns:
        Tuple of (cpu_time_ms, gpu_time_ms, speedup).
    """
    import time

    if not MetalBackend.is_available():
        print("Metal GPU not available for benchmarking")
        return None

    # Create test data
    nodes = NodeArraysHelper(num_nodes)
    bonds = BondArraysHelper(num_bonds)

    # Simple ring topology
    for i in range(num_bonds):
        bonds.connect(i, i % num_nodes, (i + 1) % num_nodes)

    # Initialize with some resource
    for i in range(num_nodes):
        nodes.F[i] = 10.0

    # GPU benchmark
    metal = MetalBackend()
    metal.upload_nodes(nodes.as_ctypes(), num_nodes)
    metal.upload_bonds(bonds.as_ctypes(), num_bonds)

    start = time.perf_counter()
    metal.execute_ticks(num_nodes, num_ticks)
    metal.synchronize()
    gpu_time = (time.perf_counter() - start) * 1000

    metal.download_nodes(nodes.as_ctypes(), num_nodes)

    print(f"Nodes: {num_nodes}, Bonds: {num_bonds}, Ticks: {num_ticks}")
    print(f"GPU time: {gpu_time:.2f}ms")
    print(f"Device: {metal.device_name}")
    print(f"GPU memory: {metal.memory_usage / 1024 / 1024:.1f} MB")

    return gpu_time


if __name__ == '__main__':
    print("Metal Backend Status")
    print("=" * 40)
    print(f"Platform: {platform.system()}")
    print(f"Metal available: {MetalBackend.is_available()}")

    if MetalBackend.is_available():
        metal = MetalBackend()
        print(f"Device: {metal.device_name}")
        print(f"Memory allocated: {metal.memory_usage / 1024 / 1024:.1f} MB")
        print()
        benchmark_cpu_vs_gpu()
