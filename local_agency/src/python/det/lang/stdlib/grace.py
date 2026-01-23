"""
Existence-Lang Grace Kernels
============================

Grace semantics: agency-gated, not agency-spending.
Grace exists to ensure movement does not itself cost agency.
"""

from ..runtime import KernelBase, Register, TokenReg
import math


class GraceOffer(KernelBase):
    """
    Grace offer kernel.

    Proposes grace offer from donor to receiver.
    Agency gates but is NOT spent.

    Properties:
    - Agency weights participation, never decremented
    - High-coherence bonds block grace (quantum gate)
    - Offer is bounded by donor capacity

    Ports:
        in  src_node: Register - donor node (has F, a)
        in  dst_node: Register - receiver node
        in  coherence: Register - bond coherence C_ij
        in  amount: Register - requested amount
        out offer: Register - actual offer amount
        out w: TokenReg
    """

    def __init__(self):
        super().__init__()
        self.src_node = Register()
        self.dst_node = Register()
        self.coherence = Register()
        self.amount = Register()
        self.offer = Register()
        self.w = TokenReg()

        # Parameters
        self._params["beta_g"] = 0.5       # Grace threshold fraction
        self._params["eta_g"] = 0.1        # Grace rate limiter
        self._params["C_quantum"] = 0.8    # Coherence quantum threshold

    def phase_COMMIT(self):
        """Compute grace offer."""
        beta_g = self._params.get("beta_g", 0.5)
        eta_g = self._params.get("eta_g", 0.1)
        C_quantum = self._params.get("C_quantum", 0.8)

        # Get values
        F_i = self.src_node.F
        F_j = self.dst_node.F
        C_ij = self.coherence.F

        # Estimate local mean (simplified: just use average)
        F_local = (F_i + F_j) / 2
        F_thresh = beta_g * F_local

        # Donor excess
        excess = max(0, F_i - F_thresh)

        # Receiver need
        need = max(0, F_thresh - F_j)

        # Agency gating (using F as proxy for agency)
        a_i = min(1.0, F_i / 100.0)  # Normalized agency
        donor_cap = a_i * excess     # Gate, not spend

        # Quantum gating for high coherence
        Q_ij = max(0, 1 - math.sqrt(C_ij) / C_quantum)
        w_ij = math.sqrt(a_i) * Q_ij

        # Compute offer (bounded)
        offer_raw = eta_g * donor_cap * w_ij
        self.offer.F = min(offer_raw, self.amount.F)

        self.w.token = "OFFER_OK" if self.offer.F > 0 else "OFFER_NONE"


class GraceAccept(KernelBase):
    """
    Grace acceptance kernel.

    Receiver accepts offered grace (agency-gated).
    No overfilling allowed.

    Ports:
        in  offer: Register - offered amount
        in  recv_cap: Register - receiver capacity
        out accepted: Register - accepted amount
        out w: TokenReg
    """

    def __init__(self):
        super().__init__()
        self.offer = Register()
        self.recv_cap = Register()
        self.accepted = Register()
        self.w = TokenReg()

    def phase_COMMIT(self):
        """Accept grace offer."""
        # Acceptance bounded by receiver capacity
        self.accepted.F = min(self.offer.F, self.recv_cap.F)

        self.w.token = "ACCEPT_OK" if self.accepted.F > 0 else "ACCEPT_NONE"


class GraceFlow(KernelBase):
    """
    Full grace flow kernel for a bond.

    Executes complete grace protocol:
    1. Compute local need and excess
    2. Agency-gated offer
    3. Quantum-gated by coherence
    4. Bounded acceptance
    5. Antisymmetric flux update

    Ports:
        inout node_i: Register - first node (F, a proxy)
        inout node_j: Register - second node
        in  coherence: Register - bond coherence C_ij
        out flux: Register - net grace flux (positive = i→j)
        out w: TokenReg
    """

    def __init__(self):
        super().__init__()
        self.node_i = Register()
        self.node_j = Register()
        self.coherence = Register()
        self.flux = Register()
        self.w = TokenReg()

        self._params["beta_g"] = 0.5
        self._params["eta_g"] = 0.1
        self._params["C_quantum"] = 0.8
        self._params["epsilon"] = 1e-6

    def phase_COMMIT(self):
        """Execute full grace protocol on bond."""
        beta_g = self._params.get("beta_g", 0.5)
        eta_g = self._params.get("eta_g", 0.1)
        C_quantum = self._params.get("C_quantum", 0.8)
        epsilon = self._params.get("epsilon", 1e-6)

        F_i = self.node_i.F
        F_j = self.node_j.F
        C_ij = self.coherence.F

        # Local mean
        F_local = (F_i + F_j) / 2
        F_thresh = beta_g * F_local

        # Needs and excesses
        need_i = max(0, F_thresh - F_i)
        need_j = max(0, F_thresh - F_j)
        excess_i = max(0, F_i - F_thresh)
        excess_j = max(0, F_j - F_thresh)

        # Agency (use F as proxy, normalized)
        a_i = min(1.0, F_i / 100.0)
        a_j = min(1.0, F_j / 100.0)

        # Donor/receiver capacities (agency-gated)
        donor_cap_i = a_i * excess_i
        donor_cap_j = a_j * excess_j
        recv_cap_i = a_i * need_i
        recv_cap_j = a_j * need_j

        # Quantum gating
        Q_ij = max(0, 1 - math.sqrt(C_ij) / C_quantum)
        w_ij = math.sqrt(a_i * a_j) * Q_ij

        # Offers in both directions
        offer_i_to_j = eta_g * donor_cap_i * (w_ij * recv_cap_j / (w_ij * recv_cap_j + epsilon))
        offer_j_to_i = eta_g * donor_cap_j * (w_ij * recv_cap_i / (w_ij * recv_cap_i + epsilon))

        # Acceptance
        accept_i_to_j = min(offer_i_to_j, recv_cap_j)
        accept_j_to_i = min(offer_j_to_i, recv_cap_i)

        # Net antisymmetric flux
        G_ij = accept_i_to_j - accept_j_to_i

        # Apply flux
        self.node_i.F -= G_ij
        self.node_j.F += G_ij

        self.flux.F = G_ij

        self.w.token = "GRACE_OK"


class GraceInject(KernelBase):
    """
    External grace injection kernel.

    Injects grace from boundary into a node.
    This is how external resource enters the system.

    Ports:
        inout node: Register - target node
        in  amount: Register - injection amount
        out w: TokenReg
    """

    def __init__(self):
        super().__init__()
        self.node = Register()
        self.amount = Register()
        self.w = TokenReg()

    def phase_COMMIT(self):
        """Inject external grace."""
        self.node.F += self.amount.F
        self.w.token = "INJECT_OK"


class GraceNeed(KernelBase):
    """
    Grace need assessment kernel.

    Computes how much grace a node needs based on local state.

    Ports:
        in  node: Register - node to assess
        in  threshold: Register - F threshold
        out need: Register - grace needed
        out w: TokenReg
    """

    def __init__(self):
        super().__init__()
        self.node = Register()
        self.threshold = Register()
        self.need = Register()
        self.w = TokenReg()

    def phase_COMMIT(self):
        """Compute grace need."""
        F = self.node.F
        thresh = self.threshold.F

        self.need.F = max(0, thresh - F)

        if self.need.F > 0:
            self.w.token = "NEEDS_GRACE"
        else:
            self.w.token = "NO_NEED"
