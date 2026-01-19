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
DET_MAX_SOMATIC = 64
DET_P_LAYER_SIZE = 16
DET_A_LAYER_SIZE = 256
DET_DORMANT_SIZE = 3760


class DETLayer(IntEnum):
    """Node layer classification."""
    DORMANT = 0
    A = 1
    P = 2
    PORT = 3
    SOMATIC = 4


class SomaticType(IntEnum):
    """Somatic node type classification."""
    # Afferent (sensor -> mind)
    TEMPERATURE = 0
    HUMIDITY = 1
    LIGHT = 2
    MOTION = 3
    SOUND = 4
    TOUCH = 5
    DISTANCE = 6
    VOLTAGE = 7
    GENERIC_SENSOR = 15

    # Efferent (mind -> actuator)
    SWITCH = 16
    MOTOR = 17
    LED = 18
    SPEAKER = 19
    SERVO = 20
    RELAY = 21
    PWM = 22
    GENERIC_ACTUATOR = 31

    # Proprioceptive (internal state)
    BATTERY = 32
    SIGNAL_STRENGTH = 33
    ERROR_STATE = 34
    HEARTBEAT = 35

    @classmethod
    def is_sensor(cls, t: 'SomaticType') -> bool:
        return t < 16

    @classmethod
    def is_actuator(cls, t: 'SomaticType') -> bool:
        return 16 <= t < 32


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
        # v6.4 Agency dynamics: Two-component model
        ("beta_a", c_float),         # Agency relaxation rate
        ("gamma_a_max", c_float),    # Max relational drive strength
        ("gamma_a_power", c_float),  # Coherence gating exponent (n >= 2)
        # Momentum dynamics
        ("alpha_pi", c_float),       # Momentum charging gain
        ("lambda_pi", c_float),      # Momentum decay rate
        ("beta_g", c_float),         # Gravity-momentum coupling
        # Layer-specific bond parameters
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


class SomaticNode(Structure):
    """Somatic node for physical I/O (sensors/actuators).

    Somatic nodes are first-class members of the mind, not peripherals.
    They bond to P-layer and A-layer, participate in coherence dynamics,
    and have their own agency for autonomous behavior (reflexes).
    """
    _fields_ = [
        ("node_id", c_uint16),
        ("remote_id", c_uint8),
        ("channel", c_uint8),
        ("type", c_uint32),  # SomaticType enum
        ("name", c_char * 32),

        # Current state
        ("value", c_float),
        ("raw_value", c_float),
        ("min_range", c_float),
        ("max_range", c_float),
        ("target", c_float),

        # Timing
        ("last_update_ms", c_uint32),
        ("sample_rate_ms", c_uint32),

        # Status
        ("online", c_bool),
        ("is_virtual", c_bool),
        ("error_count", c_uint8),

        # Simulation parameters
        ("noise_level", c_float),
        ("drift_rate", c_float),
        ("response_time", c_float),
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
        ("somatic", SomaticNode * DET_MAX_SOMATIC),
        ("num_somatic", c_uint32),
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

        # Somatic (Physical I/O)
        lib.det_core_create_somatic.restype = c_int32
        lib.det_core_create_somatic.argtypes = [
            POINTER(DETCoreStruct),
            c_uint32,  # SomaticType
            ctypes.c_char_p,  # name
            c_bool,  # is_virtual
            c_uint8,  # remote_id
            c_uint8,  # channel
        ]

        lib.det_core_remove_somatic.restype = c_bool
        lib.det_core_remove_somatic.argtypes = [POINTER(DETCoreStruct), c_uint32]

        lib.det_core_update_somatic_value.restype = None
        lib.det_core_update_somatic_value.argtypes = [
            POINTER(DETCoreStruct),
            c_uint32,  # somatic_idx
            c_float,  # value
            c_float,  # raw_value
        ]

        lib.det_core_set_somatic_target.restype = None
        lib.det_core_set_somatic_target.argtypes = [
            POINTER(DETCoreStruct),
            c_uint32,  # somatic_idx
            c_float,  # target
        ]

        lib.det_core_get_somatic.restype = POINTER(SomaticNode)
        lib.det_core_get_somatic.argtypes = [POINTER(DETCoreStruct), c_uint32]

        lib.det_core_find_somatic.restype = c_int32
        lib.det_core_find_somatic.argtypes = [POINTER(DETCoreStruct), ctypes.c_char_p]

        lib.det_core_simulate_somatic.restype = None
        lib.det_core_simulate_somatic.argtypes = [POINTER(DETCoreStruct), c_float]

        lib.det_core_get_somatic_output.restype = c_float
        lib.det_core_get_somatic_output.argtypes = [POINTER(DETCoreStruct), c_uint32]

        lib.det_somatic_is_sensor.restype = c_bool
        lib.det_somatic_is_sensor.argtypes = [c_uint32]

        lib.det_somatic_is_actuator.restype = c_bool
        lib.det_somatic_is_actuator.argtypes = [c_uint32]

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
        """Reset the core to initial state.

        This performs a full reset including reinitializing all layers,
        bonds, and ports - equivalent to creating a fresh core.
        """
        self._lib.det_core_reset(self._core)

    def init_ports(self):
        """Initialize the port nodes for LLM interface.

        This creates the initial port layout:
        - 0-4: Intent ports (answer, plan, execute, learn, debug)
        - 5-8: Domain ports (math, language, tool_use, science)
        - 9-16: Boundary ports (evaluative + relational signals)
        """
        self._lib.det_core_init_ports(self._core)

    def warmup(self, steps: int = 50, dt: float = 0.1):
        """
        Run warmup simulation steps to stabilize the system.

        This should be called after creation and before processing requests.
        It allows the DET core to compute initial aggregates and stabilize
        coherence, presence, and affect values.

        Args:
            steps: Number of warmup steps (default: 50).
            dt: Time delta per step (default: 0.1).
        """
        for _ in range(steps):
            self.step(dt)
        # Ensure self-cluster is identified
        self.identify_self()

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
    def inspect(self, detailed: bool = False) -> dict:
        """
        Get a snapshot of the core state for inspection.

        Args:
            detailed: If True, include per-node and per-bond details.

        Returns:
            Dictionary with key metrics and state.
        """
        presence, coherence, resource, debt = self.get_aggregates()
        valence, arousal, bondedness = self.get_self_affect()

        # Count nodes by layer and compute layer stats
        layer_counts = {layer: 0 for layer in DETLayer}
        layer_F_sum = {layer: 0.0 for layer in DETLayer}
        layer_q_sum = {layer: 0.0 for layer in DETLayer}

        # Domain stats (for A-layer)
        domain_stats = {i: {"count": 0, "F_sum": 0.0, "coherence": 0.0}
                        for i in range(4)}

        # Grace metrics
        nodes_needing_grace = 0
        total_grace_needed = self.total_grace_needed()

        for i in range(self.num_nodes):
            node = self.get_node(i)
            layer = DETLayer(node.layer)
            layer_counts[layer] += 1
            layer_F_sum[layer] += node.F
            layer_q_sum[layer] += node.q

            if self.needs_grace(i):
                nodes_needing_grace += 1

            # Domain stats for A-layer
            if layer == DETLayer.A and i >= 16:
                domain = node.domain
                if 0 <= domain < 4:
                    domain_stats[domain]["count"] += 1
                    domain_stats[domain]["F_sum"] += node.F

        # Compute averages
        layer_avg_F = {}
        layer_avg_q = {}
        for layer in DETLayer:
            count = layer_counts[layer]
            if count > 0:
                layer_avg_F[layer.name] = layer_F_sum[layer] / count
                layer_avg_q[layer.name] = layer_q_sum[layer] / count

        # Domain coherence
        domain_names = ["math", "language", "tool_use", "science"]
        for i, name in enumerate(domain_names):
            domain_stats[i]["coherence"] = self.get_domain_coherence(i)
            domain_stats[i]["name"] = name
            if domain_stats[i]["count"] > 0:
                domain_stats[i]["avg_F"] = domain_stats[i]["F_sum"] / domain_stats[i]["count"]
            else:
                domain_stats[i]["avg_F"] = 0.0

        # Self-cluster info
        self._lib.det_core_identify_self(self._core)
        self_struct = self._core.contents.self
        self_cluster = []
        for i in range(self_struct.num_nodes):
            self_cluster.append(self_struct.nodes[i])

        result = {
            "tick": self.tick,
            "emotion": self.get_emotion_string(),
            "aggregates": {
                "presence": round(presence, 4),
                "coherence": round(coherence, 4),
                "resource": round(resource, 4),
                "debt": round(debt, 4),
            },
            "self_affect": {
                "valence": round(valence, 4),
                "arousal": round(arousal, 4),
                "bondedness": round(bondedness, 4),
            },
            "grace": {
                "nodes_needing": nodes_needing_grace,
                "total_needed": round(total_grace_needed, 4),
            },
            "counts": {
                "nodes": self.num_nodes,
                "active": self.num_active,
                "bonds": self.num_bonds,
                "ports": self.num_ports,
                "self_cluster": len(self_cluster),
            },
            "layers": {
                "P": layer_counts[DETLayer.P],
                "A": layer_counts[DETLayer.A],
                "dormant": layer_counts[DETLayer.DORMANT],
                "port": layer_counts[DETLayer.PORT],
            },
            "layer_health": {
                "avg_F": {k: round(v, 4) for k, v in layer_avg_F.items()},
                "avg_q": {k: round(v, 4) for k, v in layer_avg_q.items()},
            },
            "domains": {
                domain_stats[i]["name"]: {
                    "count": domain_stats[i]["count"],
                    "coherence": round(domain_stats[i]["coherence"], 4),
                    "avg_F": round(domain_stats[i]["avg_F"], 4),
                }
                for i in range(4)
            },
            "self_cluster": self_cluster[:20] if not detailed else self_cluster,
        }

        # Detailed node/bond info if requested
        if detailed:
            result["nodes"] = []
            for i in range(min(self.num_active, 100)):
                node = self.get_node(i)
                result["nodes"].append({
                    "id": i,
                    "layer": DETLayer(node.layer).name,
                    "a": round(node.a, 4),
                    "F": round(node.F, 4),
                    "q": round(node.q, 4),
                    "P": round(node.P, 4),
                    "theta": round(node.theta, 4),
                    "affect": {
                        "v": round(node.affect.v, 4),
                        "r": round(node.affect.r, 4),
                        "b": round(node.affect.b, 4),
                    },
                    "in_self": i in self_cluster,
                    "needs_grace": self.needs_grace(i),
                })

            result["bonds"] = []
            for b in range(min(self.num_bonds, 200)):
                bond = self.get_bond(b)
                if bond.C > 0.01:
                    result["bonds"].append({
                        "i": bond.i,
                        "j": bond.j,
                        "C": round(bond.C, 4),
                        "pi": round(bond.pi, 4),
                        "flux_ema": round(bond.flux_ema, 4),
                        "cross_layer": bool(bond.is_cross_layer),
                    })

        return result

    # Somatic (Physical I/O) Methods

    @property
    def num_somatic(self) -> int:
        """Number of somatic nodes."""
        return self._core.contents.num_somatic

    def create_somatic(
        self,
        somatic_type: SomaticType,
        name: str,
        is_virtual: bool = True,
        remote_id: int = 0,
        channel: int = 0
    ) -> int:
        """Create a somatic node (virtual or physical).

        Args:
            somatic_type: Type of somatic node (sensor or actuator).
            name: Human-readable name for the node.
            is_virtual: True for simulated, False for physical hardware.
            remote_id: Physical device ID (e.g., ESP32 address).
            channel: Pin/channel on the remote device.

        Returns:
            Somatic index, or -1 on error.
        """
        return self._lib.det_core_create_somatic(
            self._core,
            c_uint32(int(somatic_type)),
            name.encode('utf-8'),
            c_bool(is_virtual),
            c_uint8(remote_id),
            c_uint8(channel)
        )

    def remove_somatic(self, somatic_idx: int) -> bool:
        """Remove a somatic node.

        Args:
            somatic_idx: Index of somatic node to remove.

        Returns:
            True if removed successfully.
        """
        return self._lib.det_core_remove_somatic(self._core, c_uint32(somatic_idx))

    def update_somatic_value(self, somatic_idx: int, value: float, raw_value: float = None):
        """Update somatic node value (from sensor reading or simulation).

        Args:
            somatic_idx: Index of somatic node.
            value: Normalized value (0-1 or -1 to 1).
            raw_value: Raw sensor reading (physical units). Defaults to value.
        """
        if raw_value is None:
            raw_value = value
        self._lib.det_core_update_somatic_value(
            self._core,
            c_uint32(somatic_idx),
            c_float(value),
            c_float(raw_value)
        )

    def set_somatic_target(self, somatic_idx: int, target: float):
        """Set somatic actuator target (for output nodes).

        Args:
            somatic_idx: Index of somatic node.
            target: Target value (0-1).
        """
        self._lib.det_core_set_somatic_target(
            self._core,
            c_uint32(somatic_idx),
            c_float(target)
        )

    def get_somatic(self, somatic_idx: int) -> Optional[SomaticNode]:
        """Get somatic node by index.

        Args:
            somatic_idx: Index of somatic node.

        Returns:
            SomaticNode or None if not found.
        """
        ptr = self._lib.det_core_get_somatic(self._core, c_uint32(somatic_idx))
        return ptr.contents if ptr else None

    def find_somatic(self, name: str) -> int:
        """Find somatic node by name.

        Args:
            name: Name of somatic node.

        Returns:
            Somatic index, or -1 if not found.
        """
        return self._lib.det_core_find_somatic(self._core, name.encode('utf-8'))

    def simulate_somatic(self, dt: float = 0.1):
        """Simulate virtual somatic nodes for one timestep.

        Args:
            dt: Time delta for simulation step.
        """
        self._lib.det_core_simulate_somatic(self._core, c_float(dt))

    def get_somatic_output(self, somatic_idx: int) -> float:
        """Get somatic output (for actuators, modulated by agency).

        Args:
            somatic_idx: Index of somatic node.

        Returns:
            Output value (target * agency_gate).
        """
        return self._lib.det_core_get_somatic_output(self._core, c_uint32(somatic_idx))

    def get_all_somatic(self) -> List[dict]:
        """Get all somatic nodes as a list of dictionaries.

        Returns:
            List of somatic node information.
        """
        result = []
        for i in range(self.num_somatic):
            node = self.get_somatic(i)
            if node:
                result.append({
                    "idx": i,
                    "node_id": node.node_id,
                    "name": node.name.decode('utf-8'),
                    "type": SomaticType(node.type),
                    "type_name": SomaticType(node.type).name,
                    "is_sensor": SomaticType.is_sensor(node.type),
                    "is_actuator": SomaticType.is_actuator(node.type),
                    "value": round(node.value, 4),
                    "raw_value": round(node.raw_value, 4),
                    "target": round(node.target, 4),
                    "is_virtual": node.is_virtual,
                    "online": node.online,
                    "remote_id": node.remote_id,
                    "channel": node.channel,
                })
        return result

    def __repr__(self):
        return (
            f"DETCore(tick={self.tick}, active={self.num_active}, "
            f"bonds={self.num_bonds}, somatic={self.num_somatic}, "
            f"emotion={self.get_emotion_string()})"
        )


def get_default_params() -> DETParams:
    """Get default DET parameters."""
    lib_path = _find_library()
    lib = ctypes.CDLL(lib_path)
    lib.det_default_params.restype = DETParams
    lib.det_default_params.argtypes = []
    return lib.det_default_params()
