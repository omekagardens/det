"""
Existence-Lang Primitive Kernels
================================

Fundamental operations: Transfer, Diffuse, Distinct, Compare.
These are the Tier-0 physics primitives that all other operations build upon.
"""

from ..runtime import KernelBase, Register, TokenReg, CompareResult, TransferResult
import math


class Transfer(KernelBase):
    """
    Antisymmetric resource transfer kernel.

    Moves resource from source to destination.
    Cost: k += 1

    Ports:
        in  src: Register - source of resource
        in  dst: Register - destination
        in  amount: Register - amount to transfer (optional, defaults to all)
        out witness: TokenReg - TRANSFER_OK, TRANSFER_PARTIAL, TRANSFER_BLOCKED
    """

    def __init__(self):
        super().__init__()
        self.src = Register()
        self.dst = Register()
        self.amount = Register()
        self.witness = TokenReg()
        self._params["J_quanta"] = 0.01  # Minimum transfer quantum

    def phase_COMMIT(self):
        """Execute transfer."""
        j_quanta = self._params.get("J_quanta", 0.01)

        # Determine amount to transfer
        if self.amount.F > 0:
            desired = self.amount.F
        else:
            desired = self.src.F

        # Clamp to available
        actual = min(desired, self.src.F)

        # Check if blocked
        if actual < j_quanta:
            self.witness.token = TransferResult.TRANSFER_BLOCKED
            return

        # Execute transfer (antisymmetric)
        self.src.F -= actual
        self.dst.F += actual

        # Set witness
        if actual < desired:
            self.witness.token = TransferResult.TRANSFER_PARTIAL
        else:
            self.witness.token = TransferResult.TRANSFER_OK


class Diffuse(KernelBase):
    """
    Symmetric flux exchange kernel.

    Exchanges resource between two nodes based on gradient.
    Cost: k += 1

    Ports:
        inout a: Register - first register
        inout b: Register - second register
        out flux: Register - magnitude of flux
        out witness: TokenReg
    """

    def __init__(self):
        super().__init__()
        self.a = Register()
        self.b = Register()
        self.flux = Register()
        self.witness = TokenReg()
        self._params["sigma"] = 1.0  # Conductivity

    def phase_COMMIT(self):
        """Execute diffusion."""
        sigma = self._params.get("sigma", 1.0)

        # Compute gradient
        diff = self.a.F - self.b.F

        # Compute flux (proportional to gradient and conductivity)
        j = sigma * diff * 0.5

        # Apply antisymmetric update
        self.a.F -= j
        self.b.F += j

        # Record flux magnitude
        self.flux.F = abs(j)

        self.witness.token = "OK"


class Distinct(KernelBase):
    """
    Create two distinct identities kernel.

    The most fundamental act of agency - differentiation.
    Cost: k += 1

    Ports:
        out id_a: Register - first distinct identity
        out id_b: Register - second distinct identity
        out witness: TokenReg
    """

    _next_id = 1  # Class-level identity counter

    def __init__(self):
        super().__init__()
        self.id_a = Register()
        self.id_b = Register()
        self.witness = TokenReg()

    def phase_COMMIT(self):
        """Create two distinct identities."""
        # Assign unique identities via F values
        self.id_a.F = Distinct._next_id
        self.id_b.F = Distinct._next_id + 1
        Distinct._next_id += 2

        self.witness.token = "OK"


class Compare(KernelBase):
    """
    Measure past traces kernel.

    Compares two values and produces a comparison token.
    This is information: fossilized agency.
    Cost: k += 1

    Ports:
        in  src_a: Register - first value
        in  src_b: Register - second value
        out token: TokenReg - LT, EQ, GT
    """

    def __init__(self):
        super().__init__()
        self.src_a = Register()
        self.src_b = Register()
        self.token = TokenReg()
        self._params["epsilon"] = 1e-6  # Equality tolerance

    def phase_COMMIT(self):
        """Execute comparison."""
        epsilon = self._params.get("epsilon", 1e-6)

        a_val = self.src_a.F
        b_val = self.src_b.F

        diff = a_val - b_val

        if abs(diff) < epsilon:
            self.token.token = CompareResult.EQ
        elif diff < 0:
            self.token.token = CompareResult.LT
        else:
            self.token.token = CompareResult.GT


class Pump(KernelBase):
    """
    Directional resource pump kernel.

    Moves resource in one direction only (no backflow).
    Used in kernel effects.

    Ports:
        in  src: Register
        out dst: Register
        out witness: TokenReg
    """

    def __init__(self, j_step: float = 0.01):
        super().__init__()
        self.src = Register()
        self.dst = Register()
        self.witness = TokenReg()
        self._params["J_step"] = j_step

    def move(self, src: Register, dst: Register):
        """Move resource from src to dst."""
        self.src = src
        self.dst = dst
        self.phase_COMMIT()

    def phase_COMMIT(self):
        """Execute pump."""
        j_step = self._params.get("J_step", 0.01)

        # Transfer up to j_step
        actual = min(j_step, self.src.F)
        self.src.F -= actual
        self.dst.F += actual

        self.witness.token = TransferResult.TRANSFER_OK if actual > 0 else TransferResult.TRANSFER_BLOCKED


class Measure(KernelBase):
    """
    Trace equality measurement kernel (==).

    Compares past traces with epsilon tolerance.
    Produces EQ_TRUE, EQ_FALSE, or EQ_APPROX token.

    Ports:
        in  src_a: Register
        in  src_b: Register
        in  epsilon: Register (optional)
        out token: TokenReg
    """

    def __init__(self):
        super().__init__()
        self.src_a = Register()
        self.src_b = Register()
        self.epsilon = Register()
        self.epsilon.F = 1e-6
        self.token = TokenReg()

    def phase_COMMIT(self):
        """Execute measurement."""
        eps = self.epsilon.F

        diff = abs(self.src_a.F - self.src_b.F)

        if diff < eps:
            self.token.token = "EQ_TRUE"
        elif diff < eps * 10:
            self.token.token = "EQ_APPROX"
        else:
            self.token.token = "EQ_FALSE"


class Covenant(KernelBase):
    """
    Bond truth measurement kernel (≡).

    Checks bondedness/coherence between two nodes.
    May update coherence if misaligned.

    Ports:
        in  node_i: Register (with coherence tracking)
        in  node_j: Register
        out witness: TokenReg - COV_ALIGNED, COV_DRIFT, COV_BROKEN
    """

    def __init__(self):
        super().__init__()
        self.node_i = Register()
        self.node_j = Register()
        self.coherence = Register()
        self.coherence.F = 1.0
        self.witness = TokenReg()

    def phase_COMMIT(self):
        """Check covenant/bond truth."""
        c = self.coherence.F

        if c > 0.8:
            self.witness.token = "COV_ALIGNED"
        elif c > 0.3:
            self.witness.token = "COV_DRIFT"
        else:
            self.witness.token = "COV_BROKEN"
