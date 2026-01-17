"""
DET Core - Python ctypes bindings
=================================

Low-level Python interface to the DET C kernel.
"""

import ctypes
from ctypes import (
    Structure, POINTER, c_float, c_uint8, c_uint16, c_uint32, c_uint64,
    c_int32, c_bool, c_char, c_void_p, byref
)
from enum import IntEnum
from pathlib import Path
from typing import Optional, Tuple, List
import platform


# Constants matching det_core.h
DET_MAX_NODES = 4096
DET_MAX_BONDS = 16384
DET_MAX_PORTS = 64
DET_MAX_DOMAINS = 16
DET_P_LAYER_SIZE = 16
DET_A_LAYER_SIZE = 256
DET_DORMANT_SIZE = 3760


class DETLayer(IntEnum):
    """Node layer classification."""
    DORMANT = 0
    A = 1
    P = 2
    PORT = 3


class DETDecision(IntEnum):
    """Gatekeeper decision outcomes."""
    PROCEED = 0
    RETRY = 1
    STOP = 2
    ESCALATE = 3


class DETEmotion(IntEnum):
    """Emotional state interpretation."""
    NEUTRAL = 0
    FLOW = 1
    CONTENTMENT = 2
    STRESS = 3
    OVERWHELM = 4
    APATHY = 5
    BOREDOM = 6
    PEACE = 7


# C structure definitions
class DETParams(Structure):
    """DET physics parameters."""
    _fields_ = [
        ("tau_base", c_float),
        ("sigma_base", c_float),
        ("lambda_base", c_float),
        ("mu_base", c_float),
        ("kappa_base", c_float),
        ("C_0", c_float),
        ("lambda_a", c_float),
        ("phi_L", c_float),
        ("pi_max", c_float),
        ("alpha_AA", c_float),
        ("lambda_AA", c_float),
        ("slip_AA", c_float),
        ("alpha_PP", c_float),
        ("lambda_PP", c_float),
        ("slip_PP", c_float),
        ("alpha_PA", c_float),
        ("lambda_PA", c_float),
        ("slip_PA", c_float),
    ]

    def __repr__(self):
        return f"DETParams(tau={self.tau_base}, sigma={self.sigma_base}, C_0={self.C_0})"


class DETAffect(Structure):
    """Per-node affect state."""
    _fields_ = [
        ("v", c_float),
        ("r", c_float),
        ("b", c_float),
        ("ema_throughput", c_float),
        ("ema_surprise", c_float),
        ("ema_fragmentation", c_float),
        ("ema_debt", c_float),
        ("ema_bonding", c_float),
    ]


class DETDualEMA(Structure):
    """Per-node dual EMA state."""
    _fields_ = [
        ("flux_short", c_float),
        ("flux_long", c_float),
        ("coherence_short", c_float),
        ("coherence_long", c_float),
        ("debt_short", c_float),
        ("debt_long", c_float),
    ]


class DETCadence(Structure):
    """Per-node cadence state."""
    _fields_ = [
        ("last_active_tick", c_uint32),
        ("quiet_ticks", c_uint32),
        ("membrane_window", c_uint16),
        ("window_counter", c_uint16),
    ]


class DETNode(Structure):
    """Per-node state."""
    _fields_ = [
        ("F", c_float),
        ("q", c_float),
        ("a", c_float),
        ("theta", c_float),
        ("sigma", c_float),
        ("P", c_float),
        ("tau", c_float),
        # Phase 4: Extended dynamics
        ("L", c_float),           # Angular momentum
        ("dtheta_dt", c_float),   # Phase velocity
        ("grace_buffer", c_float),# Grace injection buffer
        # Classification
        ("layer", c_uint32),  # DETLayer enum
        ("domain", c_uint8),
        ("active", c_bool),
        ("affect", DETAffect),
        ("ema", DETDualEMA),
        ("cadence", DETCadence),
        ("novelty_score", c_float),
        ("escalation_pending", c_bool),
    ]


class DETBond(Structure):
    """Per-bond state."""
    _fields_ = [
        ("i", c_uint16),
        ("j", c_uint16),
        ("C", c_float),
        ("pi", c_float),
        ("sigma", c_float),
        ("flux_ema", c_float),
        ("phase_align_ema", c_float),
        ("stability_ema", c_float),
        ("lambda_decay", c_float),
        ("lambda_slip", c_float),
        ("is_temporary", c_bool),
        ("is_cross_layer", c_bool),
    ]


class DETPort(Structure):
    """Port node for LLM interface."""
    _fields_ = [
        ("node_id", c_uint16),
        ("port_type", c_uint8),
        ("name", c_char * 32),
        ("target_domain", c_uint8),
    ]


class DETDomain(Structure):
    """Memory domain linkage."""
    _fields_ = [
        ("name", c_char * 32),
        ("coherence_to_core", c_float),
        ("activation_level", c_float),
        ("model_handle", c_void_p),
    ]


class DETSelf(Structure):
    """Self-cluster identification result."""
    _fields_ = [
        ("nodes", POINTER(c_uint16)),
        ("num_nodes", c_uint32),
        ("cluster_agency", c_float),
        ("continuity", c_float),
        ("valence", c_float),
        ("arousal", c_float),
        ("bondedness", c_float),
    ]


class DETCoreStruct(Structure):
    """Main DET core state - matches C struct."""
    _fields_ = [
        ("params", DETParams),
        ("nodes", DETNode * DET_MAX_NODES),
        ("num_nodes", c_uint32),
        ("num_active", c_uint32),
        ("bonds", DETBond * DET_MAX_BONDS),
        ("num_bonds", c_uint32),
        ("ports", DETPort * DET_MAX_PORTS),
        ("num_ports", c_uint32),
        ("domains", DETDomain * DET_MAX_DOMAINS),
        ("num_domains", c_uint32),
        ("self", DETSelf),
        ("self_nodes_storage", c_uint16 * DET_MAX_NODES),
        ("aggregate_presence", c_float),
        ("aggregate_coherence", c_float),
        ("aggregate_resource", c_float),
        ("aggregate_debt", c_float),
        ("tick", c_uint64),
        ("emotion", c_uint32),  # DETEmotion enum
    ]


def _find_library() -> str:
    """Find the det_core shared library."""
    # Look for library relative to this file
    this_dir = Path(__file__).parent
    project_root = this_dir.parent.parent

    # Platform-specific library names
    system = platform.system()
    if system == "Darwin":
        lib_names = ["libdet_core.dylib"]
    elif system == "Linux":
        lib_names = ["libdet_core.so"]
    elif system == "Windows":
        lib_names = ["det_core.dll", "libdet_core.dll"]
    else:
        lib_names = ["libdet_core.so", "libdet_core.dylib"]

    # Search paths
    search_paths = [
        project_root / "det_core" / "build",
        project_root.parent / "det_core" / "build",
        Path(__file__).parent.parent.parent / "det_core" / "build",
        Path.cwd() / "build",
        Path.cwd(),
    ]

    for search_path in search_paths:
        for lib_name in lib_names:
            lib_path = search_path / lib_name
            if lib_path.exists():
                return str(lib_path)

    raise FileNotFoundError(
        f"Could not find det_core library. Searched: {search_paths}"
    )


class DETCore:
    """
    Python wrapper for the DET C kernel.

    Provides a Pythonic interface to the Deep Existence Theory core,
    including state inspection, simulation stepping, and LLM interface.
    """

    def __init__(self, params: Optional[DETParams] = None, lib_path: Optional[str] = None):
        """
        Create a new DET core instance.

        Args:
            params: Optional custom parameters. Uses defaults if None.
            lib_path: Optional path to the shared library.
        """
        # Load the library
        if lib_path is None:
            lib_path = _find_library()

        self._lib = ctypes.CDLL(lib_path)
        self._setup_functions()

        # Create the core
        if params is not None:
            self._core = self._lib.det_core_create_with_params(byref(params))
        else:
            self._core = self._lib.det_core_create()

        if not self._core:
            raise RuntimeError("Failed to create DET core")

    def _setup_functions(self):
        """Set up ctypes function signatures."""
        lib = self._lib

        # Lifecycle
        lib.det_core_create.restype = POINTER(DETCoreStruct)
        lib.det_core_create.argtypes = []

        lib.det_core_create_with_params.restype = POINTER(DETCoreStruct)
        lib.det_core_create_with_params.argtypes = [POINTER(DETParams)]

        lib.det_core_destroy.restype = None
        lib.det_core_destroy.argtypes = [POINTER(DETCoreStruct)]

        lib.det_core_reset.restype = None
        lib.det_core_reset.argtypes = [POINTER(DETCoreStruct)]

        # Simulation
        lib.det_core_step.restype = None
        lib.det_core_step.argtypes = [POINTER(DETCoreStruct), c_float]

        lib.det_core_update_presence.restype = None
        lib.det_core_update_presence.argtypes = [POINTER(DETCoreStruct)]

        lib.det_core_update_coherence.restype = None
        lib.det_core_update_coherence.argtypes = [POINTER(DETCoreStruct), c_float]

        lib.det_core_update_agency.restype = None
        lib.det_core_update_agency.argtypes = [POINTER(DETCoreStruct), c_float]

        lib.det_core_update_affect.restype = None
        lib.det_core_update_affect.argtypes = [POINTER(DETCoreStruct), c_float]

        lib.det_core_identify_self.restype = None
        lib.det_core_identify_self.argtypes = [POINTER(DETCoreStruct)]

        # Gatekeeper
        lib.det_core_evaluate_request.restype = c_int32
        lib.det_core_evaluate_request.argtypes = [
            POINTER(DETCoreStruct),
            POINTER(c_uint32),
            c_uint32,
            c_uint8,
            c_uint32
        ]

        # Port Interface
        lib.det_core_init_ports.restype = None
        lib.det_core_init_ports.argtypes = [POINTER(DETCoreStruct)]

        lib.det_core_inject_stimulus.restype = None
        lib.det_core_inject_stimulus.argtypes = [
            POINTER(DETCoreStruct),
            POINTER(c_uint8),
            POINTER(c_float),
            c_uint32
        ]

        lib.det_core_create_interface_bonds.restype = None
        lib.det_core_create_interface_bonds.argtypes = [
            POINTER(DETCoreStruct),
            c_uint8,
            c_float
        ]

        lib.det_core_cleanup_interface_bonds.restype = None
        lib.det_core_cleanup_interface_bonds.argtypes = [POINTER(DETCoreStruct)]

        # Memory Domains
        lib.det_core_register_domain.restype = c_bool
        lib.det_core_register_domain.argtypes = [
            POINTER(DETCoreStruct),
            ctypes.c_char_p,
            c_void_p
        ]

        lib.det_core_get_domain_coherence.restype = c_float
        lib.det_core_get_domain_coherence.argtypes = [POINTER(DETCoreStruct), c_uint8]

        # Queries
        lib.det_core_get_emotion.restype = c_int32
        lib.det_core_get_emotion.argtypes = [POINTER(DETCoreStruct)]

        lib.det_core_emotion_string.restype = ctypes.c_char_p
        lib.det_core_emotion_string.argtypes = [c_int32]

        lib.det_core_get_self_affect.restype = None
        lib.det_core_get_self_affect.argtypes = [
            POINTER(DETCoreStruct),
            POINTER(c_float),
            POINTER(c_float),
            POINTER(c_float)
        ]

        lib.det_core_get_aggregates.restype = None
        lib.det_core_get_aggregates.argtypes = [
            POINTER(DETCoreStruct),
            POINTER(c_float),
            POINTER(c_float),
            POINTER(c_float),
            POINTER(c_float)
        ]

        # Node/Bond Management
        lib.det_core_recruit_node.restype = c_int32
        lib.det_core_recruit_node.argtypes = [POINTER(DETCoreStruct), c_int32]

        lib.det_core_retire_node.restype = None
        lib.det_core_retire_node.argtypes = [POINTER(DETCoreStruct), c_uint16]

        lib.det_core_create_bond.restype = c_int32
        lib.det_core_create_bond.argtypes = [POINTER(DETCoreStruct), c_uint16, c_uint16]

        lib.det_core_find_bond.restype = c_int32
        lib.det_core_find_bond.argtypes = [POINTER(DETCoreStruct), c_uint16, c_uint16]

        # Default Parameters
        lib.det_default_params.restype = DETParams
        lib.det_default_params.argtypes = []

        # Phase 4: Extended Dynamics
        lib.det_core_update_momentum.restype = None
        lib.det_core_update_momentum.argtypes = [POINTER(DETCoreStruct), c_float]

        lib.det_core_update_angular_momentum.restype = None
        lib.det_core_update_angular_momentum.argtypes = [POINTER(DETCoreStruct), c_float]

        lib.det_core_update_debt.restype = None
        lib.det_core_update_debt.argtypes = [POINTER(DETCoreStruct), c_float]

        lib.det_core_inject_grace.restype = None
        lib.det_core_inject_grace.argtypes = [POINTER(DETCoreStruct), c_uint16, c_float]

        lib.det_core_process_grace.restype = None
        lib.det_core_process_grace.argtypes = [POINTER(DETCoreStruct), c_float]

        lib.det_core_needs_grace.restype = c_bool
        lib.det_core_needs_grace.argtypes = [POINTER(DETCoreStruct), c_uint16]

        lib.det_core_total_grace_needed.restype = c_float
        lib.det_core_total_grace_needed.argtypes = [POINTER(DETCoreStruct)]

        # Phase 4: Learning via Recruitment
        lib.det_core_can_learn.restype = c_bool
        lib.det_core_can_learn.argtypes = [POINTER(DETCoreStruct), c_float, c_uint8]

        lib.det_core_activate_domain.restype = c_bool
        lib.det_core_activate_domain.argtypes = [
            POINTER(DETCoreStruct),
            ctypes.c_char_p,
            c_uint32,
            c_float
        ]

        lib.det_core_transfer_pattern.restype = c_bool
        lib.det_core_transfer_pattern.argtypes = [
            POINTER(DETCoreStruct),
            c_uint8,
            c_uint8,
            c_float
        ]

        lib.det_core_learning_capacity.restype = c_float
        lib.det_core_learning_capacity.argtypes = [POINTER(DETCoreStruct)]

        # Phase 4: Multi-Session Support
        lib.det_core_state_size.restype = ctypes.c_size_t
        lib.det_core_state_size.argtypes = [POINTER(DETCoreStruct)]

        lib.det_core_save_state.restype = ctypes.c_size_t
        lib.det_core_save_state.argtypes = [POINTER(DETCoreStruct), c_void_p, ctypes.c_size_t]

        lib.det_core_load_state.restype = c_bool
        lib.det_core_load_state.argtypes = [POINTER(DETCoreStruct), c_void_p, ctypes.c_size_t]

    def __del__(self):
        """Clean up the core on deletion."""
        if hasattr(self, '_core') and self._core:
            self._lib.det_core_destroy(self._core)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._core:
            self._lib.det_core_destroy(self._core)
            self._core = None
        return False

    # Properties for direct state access
    @property
    def tick(self) -> int:
        """Current simulation tick."""
        return self._core.contents.tick

    @property
    def num_nodes(self) -> int:
        """Total number of nodes."""
        return self._core.contents.num_nodes

    @property
    def num_active(self) -> int:
        """Number of active nodes."""
        return self._core.contents.num_active

    @property
    def num_bonds(self) -> int:
        """Number of bonds."""
        return self._core.contents.num_bonds

    @property
    def num_ports(self) -> int:
        """Number of port nodes."""
        return self._core.contents.num_ports

    @property
    def params(self) -> DETParams:
        """Current parameters."""
        return self._core.contents.params

    # Simulation methods
    def step(self, dt: float = 0.1):
        """
        Advance the simulation by one timestep.

        Args:
            dt: Time delta for this step.
        """
        self._lib.det_core_step(self._core, c_float(dt))

    def reset(self):
        """Reset the core to initial state."""
        self._lib.det_core_reset(self._core)

    def update_presence(self):
        """Update presence for all nodes."""
        self._lib.det_core_update_presence(self._core)

    def update_coherence(self, dt: float = 0.1):
        """Update coherence for all bonds."""
        self._lib.det_core_update_coherence(self._core, c_float(dt))

    def identify_self(self):
        """Identify the Self cluster."""
        self._lib.det_core_identify_self(self._core)

    # Gatekeeper
    def evaluate_request(
        self,
        tokens: List[int],
        target_domain: int = 0,
        retry_count: int = 0
    ) -> DETDecision:
        """
        Evaluate a request through the gatekeeper.

        Args:
            tokens: Token IDs representing the request.
            target_domain: Target memory domain.
            retry_count: Number of retries so far.

        Returns:
            Decision: PROCEED, RETRY, STOP, or ESCALATE.
        """
        token_array = (c_uint32 * len(tokens))(*tokens)
        result = self._lib.det_core_evaluate_request(
            self._core,
            token_array,
            c_uint32(len(tokens)),
            c_uint8(target_domain),
            c_uint32(retry_count)
        )
        return DETDecision(result)

    # Port Interface
    def inject_stimulus(self, port_indices: List[int], activations: List[float]):
        """
        Inject stimulus through port nodes.

        Args:
            port_indices: Port indices to activate.
            activations: Activation values for each port.
        """
        if len(port_indices) != len(activations):
            raise ValueError("port_indices and activations must have same length")

        indices = (c_uint8 * len(port_indices))(*port_indices)
        acts = (c_float * len(activations))(*activations)
        self._lib.det_core_inject_stimulus(
            self._core,
            indices,
            acts,
            c_uint32(len(port_indices))
        )

    def create_interface_bonds(self, target_domain: int, initial_c: float = 0.3):
        """Create temporary interface bonds to a domain."""
        self._lib.det_core_create_interface_bonds(
            self._core,
            c_uint8(target_domain),
            c_float(initial_c)
        )

    def cleanup_interface_bonds(self):
        """Clean up temporary interface bonds."""
        self._lib.det_core_cleanup_interface_bonds(self._core)

    # Memory Domains
    def register_domain(self, name: str) -> bool:
        """Register a memory domain."""
        return self._lib.det_core_register_domain(
            self._core,
            name.encode('utf-8'),
            None
        )

    def get_domain_coherence(self, domain: int) -> float:
        """Get coherence for a domain."""
        return self._lib.det_core_get_domain_coherence(self._core, c_uint8(domain))

    # Queries
    def get_emotion(self) -> DETEmotion:
        """Get current emotional state."""
        result = self._lib.det_core_get_emotion(self._core)
        return DETEmotion(result)

    def get_emotion_string(self) -> str:
        """Get emotional state as string."""
        emotion = self.get_emotion()
        return self._lib.det_core_emotion_string(c_int32(emotion)).decode('utf-8')

    def get_self_affect(self) -> Tuple[float, float, float]:
        """
        Get Self cluster affect.

        Returns:
            Tuple of (valence, arousal, bondedness).
        """
        valence = c_float()
        arousal = c_float()
        bondedness = c_float()
        self._lib.det_core_get_self_affect(
            self._core,
            byref(valence),
            byref(arousal),
            byref(bondedness)
        )
        return (valence.value, arousal.value, bondedness.value)

    def get_aggregates(self) -> Tuple[float, float, float, float]:
        """
        Get aggregate metrics.

        Returns:
            Tuple of (presence, coherence, resource, debt).
        """
        presence = c_float()
        coherence = c_float()
        resource = c_float()
        debt = c_float()
        self._lib.det_core_get_aggregates(
            self._core,
            byref(presence),
            byref(coherence),
            byref(resource),
            byref(debt)
        )
        return (presence.value, coherence.value, resource.value, debt.value)

    # Node/Bond access
    def get_node(self, index: int) -> DETNode:
        """Get node state by index."""
        if index < 0 or index >= self.num_nodes:
            raise IndexError(f"Node index {index} out of range")
        return self._core.contents.nodes[index]

    def get_bond(self, index: int) -> DETBond:
        """Get bond state by index."""
        if index < 0 or index >= self.num_bonds:
            raise IndexError(f"Bond index {index} out of range")
        return self._core.contents.bonds[index]

    def get_port(self, index: int) -> DETPort:
        """Get port info by index."""
        if index < 0 or index >= self.num_ports:
            raise IndexError(f"Port index {index} out of range")
        return self._core.contents.ports[index]

    def recruit_node(self, layer: DETLayer) -> int:
        """
        Recruit a node from the dormant pool.

        Args:
            layer: Target layer for the node.

        Returns:
            Node index, or -1 if no nodes available.
        """
        return self._lib.det_core_recruit_node(self._core, c_int32(layer))

    def retire_node(self, node_id: int):
        """Return a node to the dormant pool."""
        self._lib.det_core_retire_node(self._core, c_uint16(node_id))

    def create_bond(self, i: int, j: int) -> int:
        """
        Create a bond between two nodes.

        Returns:
            Bond index, or -1 if failed.
        """
        return self._lib.det_core_create_bond(self._core, c_uint16(i), c_uint16(j))

    def find_bond(self, i: int, j: int) -> int:
        """
        Find bond between two nodes.

        Returns:
            Bond index, or -1 if not found.
        """
        return self._lib.det_core_find_bond(self._core, c_uint16(i), c_uint16(j))

    # Phase 4: Extended Dynamics
    def inject_grace(self, node_id: int, amount: float):
        """
        Inject grace to a node for boundary recovery.

        Grace replenishes resource (F), reduces structural debt (q),
        and boosts coherence with neighbors.

        Args:
            node_id: Target node index.
            amount: Grace amount to inject.
        """
        self._lib.det_core_inject_grace(self._core, c_uint16(node_id), c_float(amount))

    def needs_grace(self, node_id: int) -> bool:
        """Check if a node needs grace injection."""
        return self._lib.det_core_needs_grace(self._core, c_uint16(node_id))

    def total_grace_needed(self) -> float:
        """Get total grace needed across all nodes."""
        return self._lib.det_core_total_grace_needed(self._core)

    # Phase 4: Learning via Recruitment
    def can_learn(self, complexity: float, domain: int = 0) -> bool:
        """
        Check if learning/recruitment is possible.

        Evaluates:
        - Self-cluster has sufficient agency
        - Resource is available
        - Not in prison regime
        - Dormant nodes available for recruitment

        Args:
            complexity: Complexity of the learning task.
            domain: Target domain for learning.

        Returns:
            True if learning is possible.
        """
        return self._lib.det_core_can_learn(
            self._core, c_float(complexity), c_uint8(domain)
        )

    def activate_domain(
        self,
        name: str,
        num_nodes: int = 8,
        initial_coherence: float = 0.4
    ) -> bool:
        """
        Activate a new domain with recruited nodes.

        Args:
            name: Domain name.
            num_nodes: Number of nodes to recruit for the domain.
            initial_coherence: Initial coherence for domain bonds.

        Returns:
            True if activation succeeded.
        """
        return self._lib.det_core_activate_domain(
            self._core,
            name.encode('utf-8'),
            c_uint32(num_nodes),
            c_float(initial_coherence)
        )

    def transfer_pattern(
        self,
        source_domain: int,
        target_domain: int,
        transfer_strength: float = 0.5
    ) -> bool:
        """
        Transfer pattern template from source to target domain.

        Creates bonds between domains and copies coherence structure.

        Args:
            source_domain: Source domain index.
            target_domain: Target domain index.
            transfer_strength: Strength of transfer.

        Returns:
            True if transfer succeeded.
        """
        return self._lib.det_core_transfer_pattern(
            self._core,
            c_uint8(source_domain),
            c_uint8(target_domain),
            c_float(transfer_strength)
        )

    def learning_capacity(self) -> float:
        """
        Get current learning capacity.

        Based on available dormant agency, resources, and cluster state.
        """
        return self._lib.det_core_learning_capacity(self._core)

    # Phase 4: Multi-Session Support
    def save_state(self) -> bytes:
        """
        Save core state to bytes for persistence.

        Returns:
            Serialized state as bytes.
        """
        size = self._lib.det_core_state_size(self._core)
        buffer = ctypes.create_string_buffer(size)
        written = self._lib.det_core_save_state(
            self._core,
            buffer,
            ctypes.c_size_t(size)
        )
        if written == 0:
            raise RuntimeError("Failed to save state")
        return buffer.raw[:written]

    def load_state(self, data: bytes) -> bool:
        """
        Load core state from bytes.

        Args:
            data: Serialized state bytes.

        Returns:
            True if load succeeded.
        """
        buffer = ctypes.create_string_buffer(data)
        return self._lib.det_core_load_state(
            self._core,
            buffer,
            ctypes.c_size_t(len(data))
        )

    def save_to_file(self, filepath: str):
        """Save core state to file."""
        data = self.save_state()
        with open(filepath, 'wb') as f:
            f.write(data)

    def load_from_file(self, filepath: str) -> bool:
        """Load core state from file."""
        with open(filepath, 'rb') as f:
            data = f.read()
        return self.load_state(data)

    # Inspection
    def inspect(self) -> dict:
        """
        Get a snapshot of the core state for inspection.

        Returns:
            Dictionary with key metrics and state.
        """
        presence, coherence, resource, debt = self.get_aggregates()
        valence, arousal, bondedness = self.get_self_affect()

        # Count nodes by layer
        layer_counts = {layer: 0 for layer in DETLayer}
        for i in range(self.num_nodes):
            node = self.get_node(i)
            layer = DETLayer(node.layer)
            layer_counts[layer] += 1

        return {
            "tick": self.tick,
            "emotion": self.get_emotion_string(),
            "aggregates": {
                "presence": presence,
                "coherence": coherence,
                "resource": resource,
                "debt": debt,
            },
            "self_affect": {
                "valence": valence,
                "arousal": arousal,
                "bondedness": bondedness,
            },
            "counts": {
                "nodes": self.num_nodes,
                "active": self.num_active,
                "bonds": self.num_bonds,
                "ports": self.num_ports,
            },
            "layers": {
                "P": layer_counts[DETLayer.P],
                "A": layer_counts[DETLayer.A],
                "dormant": layer_counts[DETLayer.DORMANT],
                "port": layer_counts[DETLayer.PORT],
            }
        }

    def __repr__(self):
        return (
            f"DETCore(tick={self.tick}, active={self.num_active}, "
            f"bonds={self.num_bonds}, emotion={self.get_emotion_string()})"
        )


def get_default_params() -> DETParams:
    """Get default DET parameters."""
    lib_path = _find_library()
    lib = ctypes.CDLL(lib_path)
    lib.det_default_params.restype = DETParams
    lib.det_default_params.argtypes = []
    return lib.det_default_params()
