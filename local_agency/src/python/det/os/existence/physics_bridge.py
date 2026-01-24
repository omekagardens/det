"""
Physics Bridge - Connect physics.ex to Substrate v2
====================================================

This module bridges Existence-Lang physics kernels to the C substrate v2.
It translates kernel invocations to substrate effects.

Hierarchy:
    physics.ex (Existence-Lang)
      → physics_bridge.py (this file)
        → libeis_substrate_v2 (C library)
          → DET-native hardware (future)
"""

import ctypes
from ctypes import c_float, c_uint32, c_uint8, c_bool, c_int32, POINTER
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Tuple
import os

# =============================================================================
# EFFECT IDS (from effect_table.h)
# =============================================================================

class EffectId(IntEnum):
    """Effect IDs matching substrate v2."""
    NONE = 0x00
    XFER_F = 0x01
    DIFFUSE = 0x02
    SET_F = 0x03
    ADD_F = 0x04
    SET_Q = 0x05
    ADD_Q = 0x06
    SET_A = 0x07
    SET_SIGMA = 0x08
    SET_P = 0x09
    SET_THETA = 0x0A
    INC_K = 0x0B
    INC_TAU = 0x0C
    SET_C = 0x10
    ADD_C = 0x11
    SET_PI = 0x12
    ADD_PI = 0x13
    SET_BOND_SIGMA = 0x14
    EMIT_TOK = 0x20
    EMIT_BYTE = 0x30
    EMIT_FLOAT = 0x31
    SET_SEED = 0xF0


class TokenValue(IntEnum):
    """Token values matching substrate_types.h."""
    VOID = 0x0000
    ERR = 0xFFFF
    FALSE = 0x0001
    TRUE = 0x0002
    LT = 0x0010
    EQ = 0x0011
    GT = 0x0012
    OK = 0x0100
    FAIL = 0x0101
    REFUSE = 0x0102
    PARTIAL = 0x0103
    XFER_OK = 0x0200
    XFER_REFUSED = 0x0201
    XFER_PARTIAL = 0x0202
    DIFFUSE_OK = 0x0210
    GRACE_OK = 0x0220
    GRACE_NONE = 0x0221
    COV_ALIGNED = 0x0230
    COV_DRIFT = 0x0231
    COV_BROKEN = 0x0232


class NodeFieldId(IntEnum):
    """Node field IDs."""
    F = 0x00
    Q = 0x01
    A = 0x02
    SIGMA = 0x03
    P = 0x04
    TAU = 0x05
    COS_THETA = 0x06
    SIN_THETA = 0x07
    K = 0x08
    R = 0x09
    FLAGS = 0x0A


class BondFieldId(IntEnum):
    """Bond field IDs."""
    NODE_I = 0x00
    NODE_J = 0x01
    C = 0x02
    PI = 0x03
    SIGMA = 0x04
    FLAGS = 0x05


# =============================================================================
# WITNESS TOKENS (from physics.ex)
# =============================================================================

@dataclass
class WitnessToken:
    """Immutable record of an operation outcome."""
    token: int
    value: float = 0.0
    extra: Optional[any] = None

    @property
    def is_ok(self) -> bool:
        return self.token in (TokenValue.OK, TokenValue.XFER_OK,
                               TokenValue.DIFFUSE_OK, TokenValue.GRACE_OK)

    @property
    def is_fail(self) -> bool:
        return self.token in (TokenValue.FAIL, TokenValue.REFUSE, TokenValue.ERR)


# =============================================================================
# SUBSTRATE INTERFACE
# =============================================================================

class SubstrateInterface:
    """
    Interface to the C substrate v2 library.

    This wraps the ctypes calls and provides a Pythonic API.
    """

    def __init__(self, lib_path: Optional[str] = None):
        """Load the substrate library."""
        if lib_path is None:
            # Try to find the library
            base = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(base, '..', '..', '..', '..', 'substrate', 'build', 'libeis_substrate_v2.dylib'),
                os.path.join(base, '..', '..', '..', '..', 'substrate', 'build', 'libeis_substrate_v2.so'),
                '/usr/local/lib/libeis_substrate_v2.dylib',
                '/usr/local/lib/libeis_substrate_v2.so',
            ]
            for path in candidates:
                if os.path.exists(path):
                    lib_path = path
                    break

        self._lib = None
        self._vm = None

        if lib_path and os.path.exists(lib_path):
            try:
                self._lib = ctypes.CDLL(lib_path)
                self._setup_functions()
            except OSError as e:
                print(f"Warning: Could not load substrate library: {e}")
                self._lib = None

    def _setup_functions(self):
        """Set up ctypes function signatures."""
        if not self._lib:
            return

        # Lifecycle
        self._lib.substrate_create.restype = ctypes.c_void_p
        self._lib.substrate_create.argtypes = []

        self._lib.substrate_destroy.restype = None
        self._lib.substrate_destroy.argtypes = [ctypes.c_void_p]

        self._lib.substrate_reset.restype = None
        self._lib.substrate_reset.argtypes = [ctypes.c_void_p]

        # Memory
        self._lib.substrate_alloc_nodes.restype = c_bool
        self._lib.substrate_alloc_nodes.argtypes = [ctypes.c_void_p, c_uint32]

        self._lib.substrate_alloc_bonds.restype = c_bool
        self._lib.substrate_alloc_bonds.argtypes = [ctypes.c_void_p, c_uint32]

        # Node access
        self._lib.substrate_node_get_f.restype = c_float
        self._lib.substrate_node_get_f.argtypes = [ctypes.c_void_p, c_uint32, c_uint32]

        self._lib.substrate_node_set_f.restype = None
        self._lib.substrate_node_set_f.argtypes = [ctypes.c_void_p, c_uint32, c_uint32, c_float]

        self._lib.substrate_node_get_i.restype = c_uint32
        self._lib.substrate_node_get_i.argtypes = [ctypes.c_void_p, c_uint32, c_uint32]

        self._lib.substrate_node_set_i.restype = None
        self._lib.substrate_node_set_i.argtypes = [ctypes.c_void_p, c_uint32, c_uint32, c_uint32]

        # Bond access
        self._lib.substrate_bond_get_f.restype = c_float
        self._lib.substrate_bond_get_f.argtypes = [ctypes.c_void_p, c_uint32, c_uint32]

        self._lib.substrate_bond_set_f.restype = None
        self._lib.substrate_bond_set_f.argtypes = [ctypes.c_void_p, c_uint32, c_uint32, c_float]

        self._lib.substrate_bond_get_i.restype = c_uint32
        self._lib.substrate_bond_get_i.argtypes = [ctypes.c_void_p, c_uint32, c_uint32]

        self._lib.substrate_bond_set_i.restype = None
        self._lib.substrate_bond_set_i.argtypes = [ctypes.c_void_p, c_uint32, c_uint32, c_uint32]

        # Proposals
        self._lib.substrate_prop_new.restype = c_uint32
        self._lib.substrate_prop_new.argtypes = [ctypes.c_void_p]

        self._lib.substrate_prop_score.restype = None
        self._lib.substrate_prop_score.argtypes = [ctypes.c_void_p, c_uint32, c_float]

        self._lib.substrate_choose.restype = c_uint32
        self._lib.substrate_choose.argtypes = [ctypes.c_void_p, c_float]

        self._lib.substrate_commit.restype = c_uint32
        self._lib.substrate_commit.argtypes = [ctypes.c_void_p]

    def create_vm(self, num_nodes: int = 1024, num_bonds: int = 2048) -> bool:
        """Create and initialize a substrate VM."""
        if not self._lib:
            return False

        self._vm = self._lib.substrate_create()
        if not self._vm:
            return False

        self._lib.substrate_alloc_nodes(self._vm, num_nodes)
        self._lib.substrate_alloc_bonds(self._vm, num_bonds)
        return True

    def destroy_vm(self):
        """Destroy the VM."""
        if self._lib and self._vm:
            self._lib.substrate_destroy(self._vm)
            self._vm = None

    # Node operations
    def node_get_F(self, node_id: int) -> float:
        if not self._vm:
            return 0.0
        return self._lib.substrate_node_get_f(self._vm, node_id, NodeFieldId.F)

    def node_set_F(self, node_id: int, value: float):
        if self._vm:
            self._lib.substrate_node_set_f(self._vm, node_id, NodeFieldId.F, value)

    def node_get_a(self, node_id: int) -> float:
        if not self._vm:
            return 0.0
        return self._lib.substrate_node_get_f(self._vm, node_id, NodeFieldId.A)

    def node_set_a(self, node_id: int, value: float):
        if self._vm:
            self._lib.substrate_node_set_f(self._vm, node_id, NodeFieldId.A, value)

    def node_get_k(self, node_id: int) -> int:
        if not self._vm:
            return 0
        return self._lib.substrate_node_get_i(self._vm, node_id, NodeFieldId.K)

    def node_inc_k(self, node_id: int):
        if self._vm:
            k = self.node_get_k(node_id)
            self._lib.substrate_node_set_i(self._vm, node_id, NodeFieldId.K, k + 1)

    # Bond operations
    def bond_get_C(self, bond_id: int) -> float:
        if not self._vm:
            return 0.0
        return self._lib.substrate_bond_get_f(self._vm, bond_id, BondFieldId.C)

    def bond_set_C(self, bond_id: int, value: float):
        if self._vm:
            self._lib.substrate_bond_set_f(self._vm, bond_id, BondFieldId.C, value)

    def bond_get_nodes(self, bond_id: int) -> Tuple[int, int]:
        if not self._vm:
            return (0, 0)
        i = self._lib.substrate_bond_get_i(self._vm, bond_id, BondFieldId.NODE_I)
        j = self._lib.substrate_bond_get_i(self._vm, bond_id, BondFieldId.NODE_J)
        return (i, j)

    def bond_set_nodes(self, bond_id: int, node_i: int, node_j: int):
        if self._vm:
            self._lib.substrate_bond_set_i(self._vm, bond_id, BondFieldId.NODE_I, node_i)
            self._lib.substrate_bond_set_i(self._vm, bond_id, BondFieldId.NODE_J, node_j)


# =============================================================================
# PHYSICS KERNELS (Python implementations)
# =============================================================================

class PhysicsKernels:
    """
    Python implementations of physics.ex kernels.

    These can execute on the substrate (when available) or in pure Python.
    """

    def __init__(self, substrate: Optional[SubstrateInterface] = None):
        self.substrate = substrate or SubstrateInterface()
        self._use_substrate = self.substrate._vm is not None

    def transfer(self, src_id: int, dst_id: int, amount: float) -> WitnessToken:
        """
        Transfer kernel: antisymmetric resource movement.

        Maps to EFFECT_XFER_F in substrate v2.
        """
        if self._use_substrate:
            src_F = self.substrate.node_get_F(src_id)
            src_a = self.substrate.node_get_a(src_id)
        else:
            # Pure Python fallback would need a separate state store
            return WitnessToken(TokenValue.FAIL, 0.0, "No substrate")

        # Calculate available
        available = min(amount, src_F)
        willing = available * src_a

        if willing <= 0 or available <= 0:
            return WitnessToken(TokenValue.REFUSE, 0.0)

        # Determine actual transfer amount
        if available >= amount:
            actual = amount
            token = TokenValue.XFER_OK
        else:
            actual = available
            token = TokenValue.XFER_PARTIAL

        # Execute transfer (antisymmetric)
        dst_F = self.substrate.node_get_F(dst_id)
        self.substrate.node_set_F(src_id, src_F - actual)
        self.substrate.node_set_F(dst_id, dst_F + actual)

        # Increment event counter
        self.substrate.node_inc_k(src_id)

        return WitnessToken(token, actual)

    def diffuse(self, bond_id: int, sigma: float = 0.0) -> WitnessToken:
        """
        Diffuse kernel: symmetric flux exchange.

        Maps to EFFECT_DIFFUSE in substrate v2.
        """
        if not self._use_substrate:
            return WitnessToken(TokenValue.FAIL, 0.0, "No substrate")

        # Get bond endpoints
        node_i, node_j = self.substrate.bond_get_nodes(bond_id)

        # Read state
        F_i = self.substrate.node_get_F(node_i)
        F_j = self.substrate.node_get_F(node_j)
        a_i = self.substrate.node_get_a(node_i)
        a_j = self.substrate.node_get_a(node_j)
        C = self.substrate.bond_get_C(bond_id)

        # Calculate gradient
        gradient = F_i - F_j
        if abs(gradient) < 0.0001:
            return WitnessToken(0x0211, 0.0)  # DIFFUSE_NONE

        # Coherence gates flow
        if C < 0.01:
            return WitnessToken(0x0212, C)  # DIFFUSE_BLOCKED

        # Effective conductivity
        effective_sigma = sigma if sigma > 0 else 1.0
        flow_rate = effective_sigma * C

        # Agency weights
        import math
        agency_weight = math.sqrt(a_i * a_j)

        # Calculate flux
        raw_flux = gradient * flow_rate * agency_weight * 0.1

        # Clamp to available
        if raw_flux > 0:
            actual_flux = min(raw_flux, F_i)
        else:
            actual_flux = max(raw_flux, -F_j)

        # Apply symmetric update
        self.substrate.node_set_F(node_i, F_i - actual_flux)
        self.substrate.node_set_F(node_j, F_j + actual_flux)

        return WitnessToken(TokenValue.DIFFUSE_OK, abs(actual_flux))

    def compare(self, a_id: int, b_id: int, epsilon: float = 0.0001) -> WitnessToken:
        """
        Compare kernel: measurement of past traces.

        Returns LT, EQ, or GT token.
        """
        if not self._use_substrate:
            return WitnessToken(TokenValue.FAIL, 0.0, "No substrate")

        val_a = self.substrate.node_get_F(a_id)
        val_b = self.substrate.node_get_F(b_id)
        diff = abs(val_a - val_b)

        # Increment event counter
        self.substrate.node_inc_k(a_id)

        if diff < epsilon:
            return WitnessToken(TokenValue.EQ, diff)
        elif val_a < val_b:
            return WitnessToken(TokenValue.LT, diff)
        else:
            return WitnessToken(TokenValue.GT, diff)

    def compute_presence(self, node_id: int) -> float:
        """
        ComputePresence kernel: P = F · C · a
        """
        if not self._use_substrate:
            return 0.0

        F = self.substrate.node_get_F(node_id)
        a = self.substrate.node_get_a(node_id)
        # C_self is stored in node sigma for now
        C = 1.0  # Default self-coherence

        P = F * C * a
        return P

    def grace_flow(self, bond_id: int, threshold: float = 0.5) -> WitnessToken:
        """
        GraceFlow kernel: complete grace protocol on bond.
        """
        if not self._use_substrate:
            return WitnessToken(TokenValue.FAIL, 0.0, "No substrate")

        import math

        # Get bond endpoints
        node_i, node_j = self.substrate.bond_get_nodes(bond_id)

        # Read state
        F_i = self.substrate.node_get_F(node_i)
        F_j = self.substrate.node_get_F(node_j)
        a_i = self.substrate.node_get_a(node_i)
        a_j = self.substrate.node_get_a(node_j)
        C = self.substrate.bond_get_C(bond_id)

        # Local mean and threshold
        local_mean = (F_i + F_j) / 2
        F_thresh = threshold * local_mean

        # Helper function
        def relu(x):
            return max(0.0, x)

        # Excess and need
        excess_i = relu(F_i - F_thresh)
        need_j = relu(F_thresh - F_j)
        excess_j = relu(F_j - F_thresh)
        need_i = relu(F_thresh - F_i)

        # Quantum gating
        Q = relu(1.0 - math.sqrt(C) / 0.9)
        w = math.sqrt(a_i * a_j) * Q

        # Offers
        if need_j > 0.001:
            offer_ij = 0.1 * excess_i * w * need_j / max(need_j + 0.001, 1.0)
        else:
            offer_ij = 0.0

        if need_i > 0.001:
            offer_ji = 0.1 * excess_j * w * need_i / max(need_i + 0.001, 1.0)
        else:
            offer_ji = 0.0

        # Acceptance
        accepted_ij = min(offer_ij, need_j)
        accepted_ji = min(offer_ji, need_i)

        # Net flow
        G = accepted_ij - accepted_ji

        if abs(G) <= 0.0001:
            return WitnessToken(0x0237, 0.0)  # GRACE_FLOW_NONE

        # Antisymmetric transfer
        self.substrate.node_set_F(node_i, F_i - G)
        self.substrate.node_set_F(node_j, F_j + G)

        return WitnessToken(0x0236, G)  # GRACE_FLOW_OK


# =============================================================================
# CONVENIENCE
# =============================================================================

def create_physics_runtime(num_nodes: int = 1024,
                           num_bonds: int = 2048) -> PhysicsKernels:
    """Create a physics runtime with substrate support."""
    substrate = SubstrateInterface()
    if substrate.create_vm(num_nodes, num_bonds):
        print(f"Physics runtime using substrate v2 ({num_nodes} nodes, {num_bonds} bonds)")
    else:
        print("Physics runtime using pure Python fallback")
    return PhysicsKernels(substrate)


__all__ = [
    'EffectId', 'TokenValue', 'NodeFieldId', 'BondFieldId',
    'WitnessToken', 'SubstrateInterface', 'PhysicsKernels',
    'create_physics_runtime'
]
