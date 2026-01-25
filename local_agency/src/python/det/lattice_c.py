"""
EIS Substrate v2 - C Lattice Backend Python Bindings
=====================================================

Python bindings for the C lattice substrate implementation.
The C substrate provides native DET v6.3 physics execution.

Usage:
    from det.lattice_c import CLattice, CLatticeConfig

    # Create a 1D lattice with 200 nodes
    lattice = CLattice(dim=1, size=200)

    # Add a resource packet
    lattice.add_packet(center=[100], mass=10.0, width=5.0, momentum=[0.1], q=0.3)

    # Run physics
    lattice.step(100)

    # Get stats
    stats = lattice.get_stats()
    print(f"Total mass: {stats.total_mass}")
"""

import ctypes
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass


# =============================================================================
# CTYPES STRUCTURES (match C headers)
# =============================================================================

LATTICE_MAX_DIM = 3


class LatticeConfig(ctypes.Structure):
    """Lattice configuration (matches substrate_lattice.h)."""
    _fields_ = [
        ('dim', ctypes.c_uint32),
        ('shape', ctypes.c_uint32 * LATTICE_MAX_DIM),
        ('boundary', ctypes.c_int),
        ('dx', ctypes.c_float),
        ('dt', ctypes.c_float),
    ]


class LatticePhysicsParams(ctypes.Structure):
    """DET v6.3 physics parameters."""
    _fields_ = [
        # Flow
        ('sigma', ctypes.c_float),
        ('outflow_limit', ctypes.c_float),
        ('F_floor', ctypes.c_float),
        # Gravity
        ('kappa_grav', ctypes.c_float),
        ('mu_grav', ctypes.c_float),
        ('beta_g', ctypes.c_float),
        # Momentum
        ('alpha_pi', ctypes.c_float),
        ('lambda_pi', ctypes.c_float),
        # Coherence
        ('alpha_C', ctypes.c_float),
        ('lambda_C', ctypes.c_float),
        ('C_init', ctypes.c_float),
        # Structure
        ('alpha_q', ctypes.c_float),
        ('gamma_q', ctypes.c_float),
        # Grace
        ('grace_enabled', ctypes.c_bool),
        ('F_MIN_grace', ctypes.c_float),
        # Features
        ('gravity_enabled', ctypes.c_bool),
        ('momentum_enabled', ctypes.c_bool),
        ('variable_dt', ctypes.c_bool),
    ]


class NodeArrays(ctypes.Structure):
    """Node state arrays (SoA layout)."""
    _fields_ = [
        ('F', ctypes.POINTER(ctypes.c_float)),
        ('q', ctypes.POINTER(ctypes.c_float)),
        ('a', ctypes.POINTER(ctypes.c_float)),
        ('sigma', ctypes.POINTER(ctypes.c_float)),
        ('P', ctypes.POINTER(ctypes.c_float)),
        ('tau', ctypes.POINTER(ctypes.c_float)),
        ('cos_theta', ctypes.POINTER(ctypes.c_float)),
        ('sin_theta', ctypes.POINTER(ctypes.c_float)),
        ('k', ctypes.POINTER(ctypes.c_uint32)),
        ('r', ctypes.POINTER(ctypes.c_uint32)),
        ('flags', ctypes.POINTER(ctypes.c_uint32)),
        ('num_nodes', ctypes.c_uint32),
        ('capacity', ctypes.c_uint32),
    ]


class BondArrays(ctypes.Structure):
    """Bond state arrays (SoA layout)."""
    _fields_ = [
        ('node_i', ctypes.POINTER(ctypes.c_uint32)),
        ('node_j', ctypes.POINTER(ctypes.c_uint32)),
        ('C', ctypes.POINTER(ctypes.c_float)),
        ('pi', ctypes.POINTER(ctypes.c_float)),
        ('sigma', ctypes.POINTER(ctypes.c_float)),
        ('flags', ctypes.POINTER(ctypes.c_uint32)),
        ('num_bonds', ctypes.c_uint32),
        ('capacity', ctypes.c_uint32),
    ]


class LatticeStats(ctypes.Structure):
    """Lattice statistics."""
    _fields_ = [
        ('total_mass', ctypes.c_float),
        ('total_structure', ctypes.c_float),
        ('total_momentum', ctypes.c_float),
        ('kinetic_energy', ctypes.c_float),
        ('potential_energy', ctypes.c_float),
        ('separation', ctypes.c_float),
        ('com', ctypes.c_float * LATTICE_MAX_DIM),
        ('step_count', ctypes.c_uint64),
    ]


# =============================================================================
# LIBRARY LOADING
# =============================================================================

def _find_substrate_library() -> Optional[Path]:
    """Find the substrate shared library."""
    search_paths = [
        # Relative to this module (in source tree)
        Path(__file__).parent / 'libeis_substrate_v2.dylib',
        Path(__file__).parent / '../../../substrate/build/libeis_substrate_v2.dylib',
        Path(__file__).parent / '../../../../substrate/build/libeis_substrate_v2.dylib',
        # Absolute path to build directory
        Path('/Volumes/AI_DATA/development/det_local_agency/det/local_agency/src/substrate/build/libeis_substrate_v2.dylib'),
        # System paths
        Path('/usr/local/lib/libeis_substrate_v2.dylib'),
        Path.home() / '.local/lib/libeis_substrate_v2.dylib',
    ]

    import os
    dyld_path = os.environ.get('DYLD_LIBRARY_PATH', '')
    for p in dyld_path.split(':'):
        if p:
            search_paths.insert(0, Path(p) / 'libeis_substrate_v2.dylib')

    for path in search_paths:
        if path.exists():
            return path

    return None


# Global library handle
_lib = None


def _get_lib():
    """Get or load the substrate library."""
    global _lib
    if _lib is None:
        lib_path = _find_substrate_library()
        if lib_path is None:
            raise RuntimeError(
                "Could not find libeis_substrate_v2.dylib. "
                "Build the substrate first: cd substrate/build && make"
            )
        _lib = ctypes.CDLL(str(lib_path))
        _setup_bindings(_lib)
    return _lib


def _setup_bindings(lib):
    """Set up ctypes function signatures."""
    # Lattice lifecycle
    lib.lattice_create_default.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    lib.lattice_create_default.restype = ctypes.c_void_p

    lib.lattice_destroy.argtypes = [ctypes.c_void_p]
    lib.lattice_destroy.restype = None

    lib.lattice_reset.argtypes = [ctypes.c_void_p]
    lib.lattice_reset.restype = None

    # Packet injection
    lib.lattice_add_packet.argtypes = [
        ctypes.c_void_p,                          # lattice
        ctypes.POINTER(ctypes.c_float),           # center
        ctypes.c_float,                           # mass
        ctypes.c_float,                           # width
        ctypes.POINTER(ctypes.c_float),           # momentum (can be NULL)
        ctypes.c_float,                           # initial_q
    ]
    lib.lattice_add_packet.restype = None

    # Physics
    lib.lattice_step.argtypes = [ctypes.c_void_p]
    lib.lattice_step.restype = None

    lib.lattice_step_n.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.lattice_step_n.restype = None

    # Statistics
    lib.lattice_total_mass.argtypes = [ctypes.c_void_p]
    lib.lattice_total_mass.restype = ctypes.c_float

    lib.lattice_separation.argtypes = [ctypes.c_void_p]
    lib.lattice_separation.restype = ctypes.c_float

    lib.lattice_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(LatticeStats)]
    lib.lattice_get_stats.restype = None

    # Rendering
    lib.lattice_render.argtypes = [
        ctypes.c_void_p,                          # lattice
        ctypes.c_int,                             # field
        ctypes.c_uint32,                          # width
        ctypes.c_char_p,                          # out
        ctypes.c_uint32,                          # out_size
    ]
    lib.lattice_render.restype = ctypes.c_uint32

    # Parameter control
    lib.lattice_set_param.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float]
    lib.lattice_set_param.restype = ctypes.c_bool

    lib.lattice_get_param.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.lattice_get_param.restype = ctypes.c_float

    # Registry
    lib.lattice_registry_init.argtypes = []
    lib.lattice_registry_init.restype = None

    lib.lattice_registry_get.argtypes = [ctypes.c_uint32]
    lib.lattice_registry_get.restype = ctypes.c_void_p


# =============================================================================
# RENDER FIELD ENUM
# =============================================================================

RENDER_FIELD_F = 0      # Resource
RENDER_FIELD_Q = 1      # Structure
RENDER_FIELD_A = 2      # Agency
RENDER_FIELD_P = 3      # Presence
RENDER_FIELD_PHI = 4    # Gravity potential


# =============================================================================
# CLATTICE CLASS
# =============================================================================

@dataclass
class CLatticeStats:
    """Python-friendly lattice statistics."""
    total_mass: float
    total_structure: float
    total_momentum: float
    kinetic_energy: float
    potential_energy: float
    separation: float
    center_of_mass: List[float]
    step_count: int


class CLattice:
    """Python wrapper for C lattice substrate.

    This provides a native DET v6.3 physics implementation using the C
    substrate, which is much faster than the pure Python implementation.

    For GPU acceleration, use CLatticeGPU instead.
    """

    def __init__(self, dim: int = 1, size: int = 100, **kwargs):
        """Create a new lattice.

        Args:
            dim: Dimensionality (1, 2, or 3)
            size: Grid size per dimension
            **kwargs: Physics parameters (e.g., beta_g=10.0, gravity_enabled=True)
        """
        if dim < 1 or dim > 3:
            raise ValueError(f"dim must be 1, 2, or 3, got {dim}")

        lib = _get_lib()

        # Initialize registry
        lib.lattice_registry_init()

        # Create lattice
        self._ptr = lib.lattice_create_default(dim, size)
        if not self._ptr:
            raise RuntimeError("Failed to create lattice")

        self.dim = dim
        self.size = size
        self._lib = lib

        # Apply any custom physics parameters
        for key, value in kwargs.items():
            self.set_param(key, value)

    def __del__(self):
        """Destroy the lattice."""
        if hasattr(self, '_ptr') and self._ptr:
            self._lib.lattice_destroy(self._ptr)
            self._ptr = None

    def reset(self):
        """Reset lattice to initial state (keep topology)."""
        self._lib.lattice_reset(self._ptr)

    def add_packet(self,
                   center: List[float],
                   mass: float,
                   width: float,
                   momentum: Optional[List[float]] = None,
                   q: float = 0.0):
        """Add a Gaussian resource packet.

        Args:
            center: Center position [x] or [x,y] or [x,y,z]
            mass: Total resource to inject
            width: Gaussian width (sigma)
            momentum: Momentum per direction (optional)
            q: Initial structure value [0,1]
        """
        # Convert to C arrays
        center_arr = (ctypes.c_float * len(center))(*center)

        if momentum is not None:
            momentum_arr = (ctypes.c_float * len(momentum))(*momentum)
            momentum_ptr = ctypes.cast(momentum_arr, ctypes.POINTER(ctypes.c_float))
        else:
            momentum_ptr = None

        self._lib.lattice_add_packet(
            self._ptr,
            center_arr,
            mass,
            width,
            momentum_ptr,
            q
        )

    def step(self, n: int = 1):
        """Execute n physics timesteps."""
        if n == 1:
            self._lib.lattice_step(self._ptr)
        else:
            self._lib.lattice_step_n(self._ptr, n)

    def total_mass(self) -> float:
        """Get total mass (sum of F)."""
        return self._lib.lattice_total_mass(self._ptr)

    def separation(self) -> float:
        """Get separation between two largest mass concentrations."""
        return self._lib.lattice_separation(self._ptr)

    def get_stats(self) -> CLatticeStats:
        """Get comprehensive statistics."""
        stats = LatticeStats()
        self._lib.lattice_get_stats(self._ptr, ctypes.byref(stats))

        return CLatticeStats(
            total_mass=stats.total_mass,
            total_structure=stats.total_structure,
            total_momentum=stats.total_momentum,
            kinetic_energy=stats.kinetic_energy,
            potential_energy=stats.potential_energy,
            separation=stats.separation,
            center_of_mass=list(stats.com[:self.dim]),
            step_count=stats.step_count,
        )

    def render(self, field: int = RENDER_FIELD_F, width: int = 60) -> str:
        """Render field to ASCII art.

        Args:
            field: Field to render (RENDER_FIELD_F, RENDER_FIELD_Q, etc.)
            width: Output width in characters

        Returns:
            ASCII art string
        """
        buf_size = width * 100  # Enough for 2D
        buf = ctypes.create_string_buffer(buf_size)

        length = self._lib.lattice_render(self._ptr, field, width, buf, buf_size)
        return buf.value[:length].decode('ascii')

    def set_param(self, name: str, value: float) -> bool:
        """Set a physics parameter.

        Args:
            name: Parameter name (e.g., "beta_g", "gravity_enabled")
            value: New value

        Returns:
            True if parameter was set
        """
        return self._lib.lattice_set_param(self._ptr, name.encode(), value)

    def get_param(self, name: str) -> float:
        """Get a physics parameter value."""
        return self._lib.lattice_get_param(self._ptr, name.encode())


# =============================================================================
# CONVENIENCE FUNCTIONS (match lattice.py API)
# =============================================================================

def is_available() -> bool:
    """Check if C lattice substrate is available."""
    try:
        _get_lib()
        return True
    except RuntimeError:
        return False


def create_lattice(dim: int, size: int, **kwargs) -> CLattice:
    """Create a new C lattice.

    This is a convenience function matching the lattice.py API.
    """
    return CLattice(dim=dim, size=size, **kwargs)


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("Testing C Lattice bindings...")

    if not is_available():
        print("C lattice not available - build the substrate first")
        exit(1)

    # Create lattice
    L = CLattice(dim=1, size=200)
    print(f"Created 1D lattice with {L.size} nodes")

    # Set strong gravity for visible binding
    L.set_param("beta_g", 30.0)
    L.set_param("mu_grav", 3.0)

    # Add two packets
    L.add_packet(center=[70], mass=10.0, width=5.0, momentum=[0.15], q=0.5)
    L.add_packet(center=[130], mass=10.0, width=5.0, momentum=[-0.15], q=0.5)

    initial_mass = L.total_mass()
    initial_sep = L.separation()
    print(f"Initial: mass={initial_mass:.4f}, separation={initial_sep:.2f}")

    # Run physics
    L.step(300)

    final_mass = L.total_mass()
    final_sep = L.separation()
    print(f"Final:   mass={final_mass:.4f}, separation={final_sep:.2f}")

    # Render
    print("\nResource field:")
    print(L.render(RENDER_FIELD_F, width=60))

    # Stats
    stats = L.get_stats()
    print(f"\nStats: {stats}")

    print("\nC Lattice bindings test passed!")
