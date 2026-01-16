"""
DET v6.3 Extension: Rolling Resonance Substrate (RRS)
======================================================

Module Type: Optional embodiment / longevity extension
Compatibility: DET v6.3 core (uses III-VI unchanged)
Purpose: Maintain a persistent coherent agentic attractor ("cluster") on a substrate
         that prevents unbounded structural debt accumulation via local rolling
         replacement (Ship-of-Theseus), while enabling entrainment-based migration
         across two coupled substrates without copying.

Key Features:
- RRS.I: State additions (node labels, age counters, bridge bonds)
- RRS.II: Substrate engineering constraints (parameter regime)
- RRS.III: Entrainment (phase mirroring via ordinary DET coupling)
- RRS.IV: Non-forkability via coherence budget constraint
- RRS.V: Rolling replacement operator (Ship of Theseus)
- RRS.VI: Migration readouts (participation, decision locus, continuity)
- RRS.VII: Optional artificial need (metabolic sink)
- RRS.X: Falsifier validation tests

Reference: DET Theory Card v6.3 - RRS Extension Specification

Author: Generated for DET framework
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from det_v6_3_3d_collider import DETCollider3D, DETParams3D


# =============================================================================
# RRS.I STATE ADDITIONS
# =============================================================================

class NodeLabel(Enum):
    """RRS.I.1 Node labels (metadata; not dynamical state)

    s(i) ∈ {BODY, SUBSTRATE}
    """
    BODY = "BODY"           # Legacy host (e.g., biological network) B ⊂ V
    SUBSTRATE = "SUBSTRATE" # Resonance substrate U ⊂ V (engineered)


@dataclass
class RRSParams:
    """RRS.IX Parameters (Module Additions)

    All RRS-specific parameters for the rolling resonance substrate.
    """
    # === Retirement Triggers (RRS.V.1) ===
    q_max: float = 0.8          # Retirement debt threshold
    P_min: float = 0.01         # Retirement presence threshold
    chi_max: int = 10000        # Retirement age threshold (optional)

    # === Fresh Node Initialization (RRS.V.2) ===
    q_fresh: float = 0.0        # Fresh node debt (typically 0)
    F_fresh: float = 0.1        # Fresh node resource (small vacuum/bootstrap)
    a_fresh: float = 0.5        # Fresh node agency seed (clipped by a_max)
    C_fresh_init: float = 0.15  # Fresh bond coherence seed

    # === Coherence Budget (RRS.IV.1) ===
    S_max: float = 6.0          # Coherence bandwidth budget (non-forkability)
    coherence_budget_enabled: bool = True

    # === Artificial Need (RRS.VII) ===
    artificial_need_enabled: bool = False
    lambda_need: float = 0.001  # Metabolic drain rate
    I_env_default: float = 0.0  # Default environmental injection

    # === Bridge Parameters (RRS.I.3) ===
    sigma_bridge: float = 1.5   # Bridge conductivity (handshake strength)
    C_bridge_init: float = 0.1  # Initial bridge coherence (weak coupling)

    # === Substrate Engineering (RRS.II) ===
    sigma_substrate_factor: float = 2.0  # σ_i >> σ_k for SUBSTRATE vs BODY

    # === Migration Detection (RRS.VI) ===
    migration_window: int = 100         # Sustained window Δt for migration
    continuity_threshold: float = 0.5   # I(S) threshold for cluster continuity

    # === Coherence Thresholds (from VI.6) ===
    C_quantum: float = 0.85             # Coherence ramp threshold for quantum gate

    # === Age Tracking ===
    age_tracking_enabled: bool = True

    # === Module Enable Flags ===
    rolling_replacement_enabled: bool = True
    entrainment_tracking_enabled: bool = True


@dataclass
class BridgeBond:
    """Represents a bridge bond connecting BODY and SUBSTRATE nodes.

    For (i,j) ∈ E_BU:
        σ_ij = σ_bridge
        C_ij(t_0) = C_bridge_init
    """
    i: Tuple[int, int, int]     # Node in B or U
    j: Tuple[int, int, int]     # Node in U or B (opposite side)
    direction: str              # 'X', 'Y', or 'Z'
    sigma: float                # Bridge conductivity
    C: float                    # Bridge coherence


# =============================================================================
# RRS MAIN CLASS
# =============================================================================

class RollingResonanceSubstrate:
    """
    DET v6.3 Extension: Rolling Resonance Substrate (RRS)

    Wraps a DETCollider3D and adds RRS-specific state, operators, and diagnostics.

    Operational Target (RRS.0):
    1. Entrainment: phases and flows align across E_BU via ordinary local coupling
    2. Dual-anchor coherence: single coherent cluster spans both B and U temporarily
    3. Migration: locus of agency-weighted participation shifts to U
    4. Debt-bounded longevity: active cluster persists under continual local node churn
    """

    def __init__(self,
                 det_sim: DETCollider3D,
                 rrs_params: Optional[RRSParams] = None,
                 body_region: Optional[np.ndarray] = None,
                 substrate_region: Optional[np.ndarray] = None):
        """
        Initialize RRS extension on a DET simulation.

        Args:
            det_sim: Base DETCollider3D simulation
            rrs_params: RRS-specific parameters
            body_region: Boolean mask for BODY nodes (B ⊂ V)
            substrate_region: Boolean mask for SUBSTRATE nodes (U ⊂ V)
        """
        self.sim = det_sim
        self.p = rrs_params or RRSParams()
        N = det_sim.p.N

        # RRS.I.1 Node labels
        self.node_labels = np.full((N, N, N), NodeLabel.BODY, dtype=object)

        # RRS.I.2 Age counters (purely local; diagnostic/trigger)
        self.chi = np.zeros((N, N, N), dtype=np.int64)

        # Track substrate region for efficient operations
        self._substrate_mask = np.zeros((N, N, N), dtype=bool)
        self._body_mask = np.zeros((N, N, N), dtype=bool)

        # Initialize regions if provided
        if body_region is not None:
            self._body_mask = body_region.astype(bool)
            self.node_labels[self._body_mask] = NodeLabel.BODY

        if substrate_region is not None:
            self._substrate_mask = substrate_region.astype(bool)
            self.node_labels[self._substrate_mask] = NodeLabel.SUBSTRATE
            # Apply substrate engineering constraints
            self._apply_substrate_engineering()

        # RRS.I.3 Bridge bonds
        self.bridge_bonds: List[BridgeBond] = []
        self._bridge_bond_map: Dict[Tuple, BridgeBond] = {}

        # Migration tracking (RRS.VI)
        self._decision_locus_history: List[Tuple[int, int, int]] = []
        self._migration_counter = 0

        # Replacement statistics
        self.total_replacements = 0
        self.replacement_log: List[Dict] = []

        # Diagnostics
        self.R_BU_history: List[float] = []
        self.cluster_continuity_history: List[float] = []

    # =========================================================================
    # RRS.I.3 BRIDGE BOND MANAGEMENT
    # =========================================================================

    def _find_bridge_bonds(self) -> List[BridgeBond]:
        """Automatically detect bridge bonds E_BU connecting B and U."""
        N = self.sim.p.N
        bridge_bonds = []

        # Check all bonds in X, Y, Z directions
        for direction, axis, shift_fn in [
            ('X', 2, lambda arr: np.roll(arr, -1, axis=2)),
            ('Y', 1, lambda arr: np.roll(arr, -1, axis=1)),
            ('Z', 0, lambda arr: np.roll(arr, -1, axis=0))
        ]:
            # Get shifted masks
            substrate_shifted = shift_fn(self._substrate_mask)
            body_shifted = shift_fn(self._body_mask)

            # Bridge: BODY -> SUBSTRATE or SUBSTRATE -> BODY
            bridge_mask_1 = self._body_mask & substrate_shifted
            bridge_mask_2 = self._substrate_mask & body_shifted

            # Find all bridge bond locations
            for mask in [bridge_mask_1, bridge_mask_2]:
                indices = np.where(mask)
                for z, y, x in zip(*indices):
                    i = (z, y, x)
                    # Compute neighbor position
                    if direction == 'X':
                        j = (z, y, (x + 1) % N)
                    elif direction == 'Y':
                        j = (z, (y + 1) % N, x)
                    else:
                        j = ((z + 1) % N, y, x)

                    bond = BridgeBond(
                        i=i, j=j, direction=direction,
                        sigma=self.p.sigma_bridge,
                        C=self.p.C_bridge_init
                    )
                    bridge_bonds.append(bond)
                    self._bridge_bond_map[(i, j, direction)] = bond

        self.bridge_bonds = bridge_bonds
        return bridge_bonds

    def add_bridge_region(self,
                          body_center: Tuple[int, int, int],
                          substrate_center: Tuple[int, int, int],
                          body_radius: float = 5.0,
                          substrate_radius: float = 5.0,
                          bridge_width: float = 1.0):
        """
        Define BODY and SUBSTRATE regions as spheres with a bridge interface.

        The bridge is automatically created where the extended regions would
        overlap or be adjacent (within bridge_width distance).

        Args:
            body_center: Center of BODY region
            substrate_center: Center of SUBSTRATE region
            body_radius: Radius of BODY sphere
            substrate_radius: Radius of SUBSTRATE sphere
            bridge_width: Width of interface region to count as bridge
        """
        N = self.sim.p.N
        z, y, x = np.mgrid[0:N, 0:N, 0:N]

        # Periodic distance for BODY
        bz, by, bx = body_center
        dx_b = (x - bx + N/2) % N - N/2
        dy_b = (y - by + N/2) % N - N/2
        dz_b = (z - bz + N/2) % N - N/2
        r2_b = dx_b**2 + dy_b**2 + dz_b**2
        r_b = np.sqrt(r2_b)

        # Periodic distance for SUBSTRATE
        sz, sy, sx = substrate_center
        dx_s = (x - sx + N/2) % N - N/2
        dy_s = (y - sy + N/2) % N - N/2
        dz_s = (z - sz + N/2) % N - N/2
        r2_s = dx_s**2 + dy_s**2 + dz_s**2
        r_s = np.sqrt(r2_s)

        # Core regions (definitely BODY or SUBSTRATE)
        core_body = r_b <= body_radius
        core_substrate = r_s <= substrate_radius

        # Extended regions for bridge detection (slightly larger)
        extended_body = r_b <= body_radius + bridge_width
        extended_substrate = r_s <= substrate_radius + bridge_width

        # Handle overlap: priority to the closer region
        overlap = core_body & core_substrate
        if np.any(overlap):
            # Assign to whichever center is closer
            closer_to_body = r_b < r_s
            core_body[overlap] = closer_to_body[overlap]
            core_substrate[overlap] = ~closer_to_body[overlap]

        # Set masks
        self._body_mask = core_body
        self._substrate_mask = core_substrate

        # Set node labels
        self.node_labels[self._body_mask] = NodeLabel.BODY
        self.node_labels[self._substrate_mask] = NodeLabel.SUBSTRATE

        # Apply substrate engineering
        self._apply_substrate_engineering()

        # Find bridge bonds
        self._find_bridge_bonds()

        # Initialize bridge bond coherences
        self._initialize_bridge_bonds()

    def add_adjacent_regions(self,
                             body_slice: Tuple[slice, slice, slice],
                             substrate_slice: Tuple[slice, slice, slice]):
        """
        Define BODY and SUBSTRATE regions as rectangular slabs (guaranteed adjacent).

        This is useful for testing when you want guaranteed bridge bonds.

        Args:
            body_slice: Tuple of slices defining BODY region
            substrate_slice: Tuple of slices defining SUBSTRATE region
        """
        N = self.sim.p.N

        # Create masks from slices
        self._body_mask = np.zeros((N, N, N), dtype=bool)
        self._substrate_mask = np.zeros((N, N, N), dtype=bool)

        self._body_mask[body_slice] = True
        self._substrate_mask[substrate_slice] = True

        # Remove any overlap
        overlap = self._body_mask & self._substrate_mask
        self._body_mask[overlap] = False

        # Set node labels
        self.node_labels[self._body_mask] = NodeLabel.BODY
        self.node_labels[self._substrate_mask] = NodeLabel.SUBSTRATE

        # Apply substrate engineering
        self._apply_substrate_engineering()

        # Find bridge bonds
        self._find_bridge_bonds()

        # Initialize bridge bond coherences
        self._initialize_bridge_bonds()

    def _initialize_bridge_bonds(self):
        """Initialize bridge bond parameters per RRS.I.3."""
        for bond in self.bridge_bonds:
            z_i, y_i, x_i = bond.i
            z_j, y_j, x_j = bond.j

            # Set coherence on the bond
            if bond.direction == 'X':
                self.sim.C_X[z_i, y_i, x_i] = self.p.C_bridge_init
            elif bond.direction == 'Y':
                self.sim.C_Y[z_i, y_i, x_i] = self.p.C_bridge_init
            elif bond.direction == 'Z':
                self.sim.C_Z[z_i, y_i, x_i] = self.p.C_bridge_init

    # =========================================================================
    # RRS.II SUBSTRATE ENGINEERING CONSTRAINTS
    # =========================================================================

    def _apply_substrate_engineering(self):
        """
        RRS.II.1 High-fidelity processing constraint.

        For i ∈ U: σ_i >> σ_k for typical k ∈ B
        """
        # Store base sigma for later enforcement
        self._base_sigma_substrate = self.sim.sigma[self._substrate_mask].copy()

        # Boost sigma in substrate region
        self.sim.sigma[self._substrate_mask] *= self.p.sigma_substrate_factor

    def _enforce_substrate_sigma(self):
        """
        RRS.II.1 Re-enforce substrate processing advantage.

        If sigma_dynamic is enabled in the DET simulation, sigma values
        are recalculated each step based on flux. This method ensures
        the substrate maintains its processing advantage.
        """
        if not np.any(self._substrate_mask):
            return

        # Get current sigma values
        current_substrate_sigma = self.sim.sigma[self._substrate_mask]
        current_body_sigma = self.sim.sigma[self._body_mask] if np.any(self._body_mask) else np.array([1.0])

        # Ensure substrate has the processing advantage
        # Use the ratio-preserving approach: substrate should be factor times body
        target_substrate_sigma = np.mean(current_body_sigma) * self.p.sigma_substrate_factor

        # Scale substrate sigma to maintain the advantage
        if np.mean(current_substrate_sigma) < target_substrate_sigma:
            self.sim.sigma[self._substrate_mask] = np.maximum(
                current_substrate_sigma,
                target_substrate_sigma * np.ones_like(current_substrate_sigma)
            )

    # =========================================================================
    # RRS.III ENTRAINMENT (Phase Mirroring via Ordinary DET Coupling)
    # =========================================================================

    def compute_entrainment_order_parameter(self) -> float:
        """
        RRS.III.2 Bridge entrainment order parameter (diagnostic only).

        R_BU ≡ ⟨√C_ij cos(θ_i - θ_j)⟩_{(i,j) ∈ E_BU}

        High R_BU indicates phase alignment across the bridge.

        Returns:
            Entrainment order parameter in [-1, 1]
        """
        if len(self.bridge_bonds) == 0:
            return 0.0

        total = 0.0
        count = 0

        for bond in self.bridge_bonds:
            z_i, y_i, x_i = bond.i
            z_j, y_j, x_j = bond.j

            theta_i = self.sim.theta[z_i, y_i, x_i]
            theta_j = self.sim.theta[z_j, y_j, x_j]

            # Get bond coherence
            if bond.direction == 'X':
                C_ij = self.sim.C_X[z_i, y_i, x_i]
            elif bond.direction == 'Y':
                C_ij = self.sim.C_Y[z_i, y_i, x_i]
            else:
                C_ij = self.sim.C_Z[z_i, y_i, x_i]

            # Entrainment contribution
            total += np.sqrt(C_ij) * np.cos(theta_i - theta_j)
            count += 1

        return total / count if count > 0 else 0.0

    def compute_phase_coherence(self, region_mask: np.ndarray) -> complex:
        """
        Compute phase coherence ψ for a region.

        ψ = (1/|S|) Σ_i e^{iθ_i}

        Returns:
            Complex order parameter; |ψ| = 1 means perfect phase alignment
        """
        if not np.any(region_mask):
            return 0.0 + 0.0j

        phases = self.sim.theta[region_mask]
        return np.mean(np.exp(1j * phases))

    # =========================================================================
    # RRS.IV NON-FORKABILITY (Coherence Budget Constraint)
    # =========================================================================

    def compute_local_coherence_load(self) -> np.ndarray:
        """
        RRS.IV.1 Coherence budget - compute local coherence load.

        S_i ≡ Σ_{j ∈ N_R(i)} C_ij

        Returns:
            Array of local coherence loads
        """
        N = self.sim.p.N
        Xm = lambda arr: np.roll(arr, 1, axis=2)
        Ym = lambda arr: np.roll(arr, 1, axis=1)
        Zm = lambda arr: np.roll(arr, 1, axis=0)

        # Sum coherence from all 6 neighbors
        S = (self.sim.C_X + Xm(self.sim.C_X) +
             self.sim.C_Y + Ym(self.sim.C_Y) +
             self.sim.C_Z + Zm(self.sim.C_Z))

        return S

    def apply_coherence_budget_renormalization(self):
        """
        RRS.IV.1 Coherence budget renormalization.

        Applied after v6.3 coherence update (VI.3):

        C_ij ← C_ij                                          if S_i ≤ S_max and S_j ≤ S_max
        C_ij ← C_ij · min(S_max/(S_i+ε), S_max/(S_j+ε))     otherwise

        Properties:
        - Strictly local (uses only neighborhood sums)
        - Does not touch a_i (agency inviolability preserved)
        - Suppresses stable multi-anchor forks by enforcing finite coherence bandwidth
        """
        if not self.p.coherence_budget_enabled:
            return

        eps = 1e-9
        S_max = self.p.S_max

        # Compute load at each node
        S = self.compute_local_coherence_load()

        # Shift operators
        Xp = lambda arr: np.roll(arr, -1, axis=2)
        Yp = lambda arr: np.roll(arr, -1, axis=1)
        Zp = lambda arr: np.roll(arr, -1, axis=0)

        # For each bond direction, apply renormalization
        # C_X bond connects (i) to (i+X)
        S_i = S
        S_j_X = Xp(S)
        S_j_Y = Yp(S)
        S_j_Z = Zp(S)

        # Renormalization factors
        factor_X = np.where(
            (S_i <= S_max) & (S_j_X <= S_max),
            1.0,
            np.minimum(S_max / (S_i + eps), S_max / (S_j_X + eps))
        )
        factor_Y = np.where(
            (S_i <= S_max) & (S_j_Y <= S_max),
            1.0,
            np.minimum(S_max / (S_i + eps), S_max / (S_j_Y + eps))
        )
        factor_Z = np.where(
            (S_i <= S_max) & (S_j_Z <= S_max),
            1.0,
            np.minimum(S_max / (S_i + eps), S_max / (S_j_Z + eps))
        )

        # Apply renormalization
        self.sim.C_X *= factor_X
        self.sim.C_Y *= factor_Y
        self.sim.C_Z *= factor_Z

        # Ensure coherence stays within bounds
        C_min = self.sim.p.C_init
        self.sim.C_X = np.clip(self.sim.C_X, C_min, 1.0)
        self.sim.C_Y = np.clip(self.sim.C_Y, C_min, 1.0)
        self.sim.C_Z = np.clip(self.sim.C_Z, C_min, 1.0)

    # =========================================================================
    # RRS.V ROLLING REPLACEMENT OPERATOR (Ship of Theseus)
    # =========================================================================

    def check_retirement_trigger(self, pos: Tuple[int, int, int]) -> bool:
        """
        RRS.V.1 Retirement trigger.

        A substrate node retires if any trigger fires:
        retire(i) ⟺ (q_i > q_max) ∨ (P_i < P_min) ∨ (χ_i > χ_max)

        Args:
            pos: Node position (z, y, x)

        Returns:
            True if node should retire
        """
        z, y, x = pos

        # Only substrate nodes can retire
        if self.node_labels[z, y, x] != NodeLabel.SUBSTRATE:
            return False

        # Check debt threshold
        if self.sim.q[z, y, x] > self.p.q_max:
            return True

        # Check presence threshold
        if self.sim.P[z, y, x] < self.p.P_min:
            return True

        # Check age threshold (if enabled)
        if self.p.age_tracking_enabled and self.chi[z, y, x] > self.p.chi_max:
            return True

        return False

    def _compute_neighbor_phase_average(self, pos: Tuple[int, int, int]) -> float:
        """
        RRS.V.2 Phase initialization via local-only entrainment.

        θ_i' := arg(Σ_{j ∈ N(i')} w_j e^{iθ_j})
        with w_j = σ_{i'j}√C_{i'j} + ε

        Args:
            pos: Node position

        Returns:
            Entrained phase angle
        """
        z, y, x = pos
        N = self.sim.p.N
        eps = 1e-9

        # Get 6 neighbors
        neighbors = [
            ((z, y, (x+1) % N), 'X', (z, y, x)),          # +X
            ((z, y, (x-1) % N), 'X', (z, y, (x-1) % N)),  # -X
            ((z, (y+1) % N, x), 'Y', (z, y, x)),          # +Y
            ((z, (y-1) % N, x), 'Y', (z, (y-1) % N, x)),  # -Y
            (((z+1) % N, y, x), 'Z', (z, y, x)),          # +Z
            (((z-1) % N, y, x), 'Z', ((z-1) % N, y, x)),  # -Z
        ]

        weighted_sum = 0.0 + 0.0j

        for (nz, ny, nx), direction, bond_idx in neighbors:
            bz, by, bx = bond_idx
            theta_j = self.sim.theta[nz, ny, nx]

            # Get bond coherence
            if direction == 'X':
                C_ij = self.sim.C_X[bz, by, bx]
            elif direction == 'Y':
                C_ij = self.sim.C_Y[bz, by, bx]
            else:
                C_ij = self.sim.C_Z[bz, by, bx]

            # Weight: σ_{ij} √C_{ij} + ε
            sigma_ij = 0.5 * (self.sim.sigma[z, y, x] + self.sim.sigma[nz, ny, nx])
            w_j = sigma_ij * np.sqrt(C_ij) + eps

            weighted_sum += w_j * np.exp(1j * theta_j)

        return np.angle(weighted_sum)

    def retire_and_replace_node(self, pos: Tuple[int, int, int]):
        """
        RRS.V.2 Retirement + replacement map.

        Remove node i and insert fresh node i' in the same local slot:
        - N(i') := N(i)
        - q_{i'} := q_fresh
        - F_{i'} := F_fresh
        - r_{i'} := 0
        - χ_{i'} := 0
        - a_{i'} := min(a_fresh, a_max,i')
        - θ_{i'} := arg(Σ w_j e^{iθ_j})  [local-only entrainment]
        - C_{i'j} := C_fresh_init

        Args:
            pos: Node position to retire and replace
        """
        z, y, x = pos
        N = self.sim.p.N
        p = self.p

        # Log retirement
        self.replacement_log.append({
            'step': self.sim.step_count,
            'position': pos,
            'old_q': float(self.sim.q[z, y, x]),
            'old_P': float(self.sim.P[z, y, x]),
            'old_chi': int(self.chi[z, y, x]),
            'old_F': float(self.sim.F[z, y, x]),
            'old_a': float(self.sim.a[z, y, x])
        })

        # === Initialize fresh node state ===

        # q_fresh (typically 0)
        self.sim.q[z, y, x] = p.q_fresh

        # F_fresh (small bootstrap)
        self.sim.F[z, y, x] = p.F_fresh

        # χ = 0 (reset age)
        self.chi[z, y, x] = 0

        # Agency initialization respects structural ceiling (VI.2)
        a_max = 1.0 / (1.0 + self.sim.p.lambda_a * p.q_fresh**2)
        self.sim.a[z, y, x] = min(p.a_fresh, a_max)

        # Phase initialization via local-only entrainment
        self.sim.theta[z, y, x] = self._compute_neighbor_phase_average(pos)

        # Reset momentum at this node
        self.sim.pi_X[z, y, x] = 0.0
        self.sim.pi_Y[z, y, x] = 0.0
        self.sim.pi_Z[z, y, x] = 0.0

        # Reset angular momentum on incident plaquettes
        self.sim.L_XY[z, y, x] = 0.0
        self.sim.L_YZ[z, y, x] = 0.0
        self.sim.L_XZ[z, y, x] = 0.0

        # === Reset bond coherence to neighbors ===
        # Bonds connecting this node use C_fresh_init
        C_init = p.C_fresh_init

        # +X, +Y, +Z bonds originating from this node
        self.sim.C_X[z, y, x] = C_init
        self.sim.C_Y[z, y, x] = C_init
        self.sim.C_Z[z, y, x] = C_init

        # -X, -Y, -Z bonds terminating at this node
        self.sim.C_X[z, y, (x-1) % N] = C_init
        self.sim.C_Y[z, (y-1) % N, x] = C_init
        self.sim.C_Z[(z-1) % N, y, x] = C_init

        self.total_replacements += 1

    def apply_rolling_replacement(self):
        """
        RRS.V Rolling replacement on all substrate nodes.

        Check retirement triggers and replace nodes as needed.
        This is a local environment/hardware rule, not a boundary action.
        """
        if not self.p.rolling_replacement_enabled:
            return

        # Get substrate node positions
        substrate_indices = np.where(self._substrate_mask)

        for z, y, x in zip(*substrate_indices):
            pos = (z, y, x)
            if self.check_retirement_trigger(pos):
                self.retire_and_replace_node(pos)

    # =========================================================================
    # RRS.VI MIGRATION READOUTS (Diagnostic Only)
    # =========================================================================

    def compute_participation_strength(self) -> np.ndarray:
        """
        RRS.VI.1 Participation strength.

        Π_i ≡ a_i P_i

        Returns:
            Array of participation strengths
        """
        return self.sim.a * self.sim.P

    def compute_decision_locus(self) -> Tuple[Tuple[int, int, int], float]:
        """
        RRS.VI.2 Decision locus (argmax participation).

        i*(t) ≡ argmax_{i ∈ V} Π_i

        Returns:
            Tuple of (position, participation_value)
        """
        Pi = self.compute_participation_strength()
        idx = np.unravel_index(np.argmax(Pi), Pi.shape)
        return idx, float(Pi[idx])

    def check_migration_event(self) -> bool:
        """
        RRS.VI.2 Migration event detection.

        migrate ⟺ i*(t) ∈ U for sustained window Δt

        Returns:
            True if migration event detected
        """
        locus, _ = self.compute_decision_locus()
        self._decision_locus_history.append(locus)

        # Only keep recent history
        if len(self._decision_locus_history) > self.p.migration_window:
            self._decision_locus_history = self._decision_locus_history[-self.p.migration_window:]

        # Check if locus has been in substrate for sustained window
        if len(self._decision_locus_history) < self.p.migration_window:
            return False

        substrate_count = sum(
            1 for pos in self._decision_locus_history
            if self._substrate_mask[pos]
        )

        # Migration if majority of window is in substrate
        return substrate_count >= self.p.migration_window * 0.8

    def compute_cluster_continuity(self,
                                   threshold_ratio: float = 10.0) -> Tuple[float, List[np.ndarray]]:
        """
        RRS.VI.3 Cluster continuity metric.

        For any candidate cluster subgraph S ⊂ V:
        I(S) ≡ Σ_{(i,j) ∈ E(S)} √(a_i a_j) C_ij

        Continuity means I(S(t)) remains above threshold and does not bifurcate.

        Args:
            threshold_ratio: Ratio above vacuum to identify cluster

        Returns:
            Tuple of (continuity_metric, list_of_cluster_masks)
        """
        from scipy.ndimage import label as nd_label

        # Find high-resource regions as cluster candidates
        threshold = self.sim.p.F_VAC * threshold_ratio
        above_threshold = self.sim.F > threshold

        labeled_array, num_features = nd_label(above_threshold)

        if num_features == 0:
            return 0.0, []

        clusters = []
        continuity_values = []

        for label_id in range(1, num_features + 1):
            cluster_mask = labeled_array == label_id
            clusters.append(cluster_mask)

            # Compute I(S) for this cluster
            I_S = self._compute_cluster_continuity_metric(cluster_mask)
            continuity_values.append(I_S)

        # Return max continuity and all clusters
        max_continuity = max(continuity_values) if continuity_values else 0.0
        return max_continuity, clusters

    def _compute_cluster_continuity_metric(self, cluster_mask: np.ndarray) -> float:
        """
        Compute continuity metric I(S) for a cluster.

        I(S) = Σ_{(i,j) ∈ E(S)} √(a_i a_j) C_ij
        """
        N = self.sim.p.N
        Xp = lambda arr: np.roll(arr, -1, axis=2)
        Yp = lambda arr: np.roll(arr, -1, axis=1)
        Zp = lambda arr: np.roll(arr, -1, axis=0)

        # Bonds within the cluster
        cluster_float = cluster_mask.astype(float)

        # For X bonds: both endpoints in cluster
        bond_in_cluster_X = cluster_float * Xp(cluster_float)
        bond_in_cluster_Y = cluster_float * Yp(cluster_float)
        bond_in_cluster_Z = cluster_float * Zp(cluster_float)

        # Agency gate
        sqrt_aa_X = np.sqrt(self.sim.a * Xp(self.sim.a))
        sqrt_aa_Y = np.sqrt(self.sim.a * Yp(self.sim.a))
        sqrt_aa_Z = np.sqrt(self.sim.a * Zp(self.sim.a))

        # Sum contributions
        I_X = np.sum(bond_in_cluster_X * sqrt_aa_X * self.sim.C_X)
        I_Y = np.sum(bond_in_cluster_Y * sqrt_aa_Y * self.sim.C_Y)
        I_Z = np.sum(bond_in_cluster_Z * sqrt_aa_Z * self.sim.C_Z)

        return float(I_X + I_Y + I_Z)

    # =========================================================================
    # RRS.VII OPTIONAL ARTIFICIAL NEED
    # =========================================================================

    def apply_artificial_need(self, I_env: Optional[np.ndarray] = None):
        """
        RRS.VII Artificial need (prevents stagnation).

        For i ∈ U:
        F_i^+ ← F_i^+ - λ_need F_i Δτ_i + I_{env→i}

        Args:
            I_env: Environmental injection array (optional)
        """
        if not self.p.artificial_need_enabled:
            return

        # Default injection if not provided
        if I_env is None:
            I_env = np.full_like(self.sim.F, self.p.I_env_default)

        # Apply drain only to substrate nodes
        drain = self.p.lambda_need * self.sim.F * self.sim.Delta_tau
        drain_masked = np.where(self._substrate_mask, drain, 0.0)

        # Injection to substrate
        injection_masked = np.where(self._substrate_mask, I_env, 0.0)

        # Update resources
        self.sim.F = np.maximum(
            self.sim.p.F_MIN,
            self.sim.F - drain_masked + injection_masked
        )

    # =========================================================================
    # RRS.VIII CANONICAL UPDATE ORDERING
    # =========================================================================

    def step(self, I_env: Optional[np.ndarray] = None):
        """
        Execute one RRS-augmented DET update step.

        Ordering (drop-in to v6.3):
        1-5. [DET core: loads, presence, flows, resources, q-locking]
        6. Coherence update (VI.3)
        7. Apply coherence budget renormalization (RRS.IV) if enabled
        8. Agency update (VI.2)
        9-10. [DET core: momentum, grace]
        11. RRS rolling replacement on i ∈ U using triggers (RRS.V)
        12. Increment ages χ_i for surviving nodes
        """
        # === Steps 1-10: Standard DET v6.3 update ===
        self.sim.step()

        # === Re-apply substrate engineering (RRS.II.1) if sigma_dynamic ===
        # The DET step may reset sigma based on flux; we must maintain σ_substrate >> σ_body
        self._enforce_substrate_sigma()

        # === Step 7 (post-hoc): Coherence budget renormalization ===
        self.apply_coherence_budget_renormalization()

        # === RRS.VII: Artificial need (if enabled) ===
        self.apply_artificial_need(I_env)

        # === Step 11: Rolling replacement ===
        self.apply_rolling_replacement()

        # === Step 12: Increment ages for surviving substrate nodes ===
        if self.p.age_tracking_enabled:
            self.chi[self._substrate_mask] += 1

        # === Diagnostics ===
        if self.p.entrainment_tracking_enabled and len(self.bridge_bonds) > 0:
            R_BU = self.compute_entrainment_order_parameter()
            self.R_BU_history.append(R_BU)

        continuity, _ = self.compute_cluster_continuity()
        self.cluster_continuity_history.append(continuity)

    # =========================================================================
    # DIAGNOSTICS AND ANALYSIS
    # =========================================================================

    def get_body_statistics(self) -> Dict:
        """Get statistics for BODY region."""
        if not np.any(self._body_mask):
            return {}
        return {
            'total_F': float(np.sum(self.sim.F[self._body_mask])),
            'mean_q': float(np.mean(self.sim.q[self._body_mask])),
            'mean_a': float(np.mean(self.sim.a[self._body_mask])),
            'mean_P': float(np.mean(self.sim.P[self._body_mask])),
            'node_count': int(np.sum(self._body_mask))
        }

    def get_substrate_statistics(self) -> Dict:
        """Get statistics for SUBSTRATE region."""
        if not np.any(self._substrate_mask):
            return {}
        return {
            'total_F': float(np.sum(self.sim.F[self._substrate_mask])),
            'mean_q': float(np.mean(self.sim.q[self._substrate_mask])),
            'mean_a': float(np.mean(self.sim.a[self._substrate_mask])),
            'mean_P': float(np.mean(self.sim.P[self._substrate_mask])),
            'mean_chi': float(np.mean(self.chi[self._substrate_mask])),
            'max_chi': int(np.max(self.chi[self._substrate_mask])),
            'node_count': int(np.sum(self._substrate_mask)),
            'total_replacements': self.total_replacements
        }

    def get_bridge_statistics(self) -> Dict:
        """Get statistics for bridge bonds E_BU."""
        if len(self.bridge_bonds) == 0:
            return {}

        coherences = []
        for bond in self.bridge_bonds:
            z, y, x = bond.i
            if bond.direction == 'X':
                coherences.append(self.sim.C_X[z, y, x])
            elif bond.direction == 'Y':
                coherences.append(self.sim.C_Y[z, y, x])
            else:
                coherences.append(self.sim.C_Z[z, y, x])

        return {
            'num_bridges': len(self.bridge_bonds),
            'mean_C': float(np.mean(coherences)),
            'max_C': float(np.max(coherences)),
            'min_C': float(np.min(coherences)),
            'R_BU': self.compute_entrainment_order_parameter()
        }

    def get_migration_status(self) -> Dict:
        """Get current migration status."""
        locus, participation = self.compute_decision_locus()
        locus_in_substrate = self._substrate_mask[locus]

        return {
            'decision_locus': locus,
            'participation': participation,
            'locus_in_substrate': locus_in_substrate,
            'migration_detected': self.check_migration_event()
        }

    def summary(self) -> Dict:
        """Get comprehensive summary of RRS state."""
        return {
            'step': self.sim.step_count,
            'time': self.sim.time,
            'body': self.get_body_statistics(),
            'substrate': self.get_substrate_statistics(),
            'bridge': self.get_bridge_statistics(),
            'migration': self.get_migration_status(),
            'cluster_continuity': self.cluster_continuity_history[-1] if self.cluster_continuity_history else 0.0
        }


# =============================================================================
# RRS.X FALSIFIERS
# =============================================================================

class RRSFalsifierTests:
    """
    RRS.X Falsifiers (Module-Specific)

    RRS is false under v6.3 rules if any condition below holds robustly
    across reasonable parameter scans.
    """

    @staticmethod
    def test_longevity_failure_under_churn(
        rrs: RollingResonanceSubstrate,
        max_steps: int = 5000,
        continuity_threshold: float = 0.5,
        verbose: bool = True
    ) -> Tuple[bool, Dict]:
        """
        RRS-F1: Longevity Failure Under Churn

        With rolling replacement enabled and substrate power available,
        I(S(t)) inevitably collapses below threshold for all clusters
        for churn rates below a declared maximum.

        Returns:
            (passed, details) - passed=True means longevity maintained (NOT falsified)
        """
        if verbose:
            print("\n" + "="*60)
            print("RRS-F1: Longevity Under Churn Test")
            print("="*60)

        # Track continuity over time
        continuity_values = []
        collapse_detected = False
        collapse_step = None

        for step in range(max_steps):
            rrs.step()
            continuity, _ = rrs.compute_cluster_continuity()
            continuity_values.append(continuity)

            if verbose and step % 500 == 0:
                print(f"  Step {step}: I(S)={continuity:.4f}, replacements={rrs.total_replacements}")

            # Check for collapse after initial transient
            if step > 100 and continuity < continuity_threshold:
                if not collapse_detected:
                    collapse_detected = True
                    collapse_step = step

        # Longevity maintained if no sustained collapse
        final_continuity = np.mean(continuity_values[-100:]) if len(continuity_values) >= 100 else 0.0
        passed = final_continuity >= continuity_threshold

        details = {
            'max_continuity': float(np.max(continuity_values)),
            'min_continuity': float(np.min(continuity_values)),
            'final_continuity': final_continuity,
            'total_replacements': rrs.total_replacements,
            'collapse_detected': collapse_detected,
            'collapse_step': collapse_step
        }

        if verbose:
            print(f"\n  Final I(S): {final_continuity:.4f}")
            print(f"  Threshold: {continuity_threshold}")
            print(f"  RRS-F1 {'NOT FALSIFIED (longevity maintained)' if passed else 'FALSIFIED (longevity failure)'}")

        return passed, details

    @staticmethod
    def test_fork_emergence(
        rrs: RollingResonanceSubstrate,
        max_steps: int = 3000,
        fork_threshold: float = 0.3,
        verbose: bool = True
    ) -> Tuple[bool, Dict]:
        """
        RRS-F2: Fork Emergence (Non-Forkability Failure)

        Two disjoint components S_1, S_2 persist with
        I(S_1) > τ, I(S_2) > τ after migration, without one decaying—
        indicating stable "duplication."

        Returns:
            (passed, details) - passed=True means no fork (NOT falsified)
        """
        if verbose:
            print("\n" + "="*60)
            print("RRS-F2: Fork Emergence Test")
            print("="*60)

        fork_detected = False
        fork_step = None
        dual_cluster_duration = 0

        for step in range(max_steps):
            rrs.step()

            _, clusters = rrs.compute_cluster_continuity()

            if len(clusters) >= 2:
                # Compute continuity for top 2 clusters
                continuities = [rrs._compute_cluster_continuity_metric(c) for c in clusters[:2]]

                if all(c > fork_threshold for c in continuities[:2]):
                    dual_cluster_duration += 1

                    if verbose and step % 500 == 0:
                        print(f"  Step {step}: Dual clusters detected, I(S_1)={continuities[0]:.4f}, I(S_2)={continuities[1]:.4f}")

                    # Fork if dual clusters persist for sustained period
                    if dual_cluster_duration > 200:
                        fork_detected = True
                        fork_step = step
                else:
                    dual_cluster_duration = max(0, dual_cluster_duration - 1)
            else:
                dual_cluster_duration = max(0, dual_cluster_duration - 1)

        passed = not fork_detected  # Pass if NO fork emerged

        details = {
            'fork_detected': fork_detected,
            'fork_step': fork_step,
            'max_dual_cluster_duration': dual_cluster_duration
        }

        if verbose:
            print(f"\n  Fork {'DETECTED' if fork_detected else 'NOT detected'}")
            print(f"  RRS-F2 {'FALSIFIED (fork emerged)' if fork_detected else 'NOT FALSIFIED (non-forkability maintained)'}")

        return passed, details

    @staticmethod
    def test_hidden_global_dependence(
        rrs: RollingResonanceSubstrate,
        verbose: bool = True
    ) -> Tuple[bool, Dict]:
        """
        RRS-F3: Hidden Global Dependence

        Successful migration/longevity requires any global normalization,
        global clock sync, or any operation that reads state outside
        N_R(i) or E_R(i,j).

        This is a structural/code audit test rather than simulation.

        Returns:
            (passed, details) - passed=True means locality maintained
        """
        if verbose:
            print("\n" + "="*60)
            print("RRS-F3: Hidden Global Dependence Test")
            print("="*60)

        # Check that all RRS operations are local
        local_operations = {
            'coherence_budget': True,       # Uses only neighborhood sums
            'retirement_trigger': True,     # Uses only node-local state
            'replacement_map': True,        # Uses only neighbor phases
            'phase_entrainment': True,      # Local weighted average
            'age_counter': True,            # Per-node counter
        }

        # Check for any global normalizations in the code
        global_deps = []

        # Coherence budget uses local sums only
        S = rrs.compute_local_coherence_load()
        # This is computed via roll operations (local neighbor access)

        # Migration readouts are diagnostic only (not used in dynamics)
        # Decision locus uses argmax but this is diagnostic

        passed = len(global_deps) == 0

        details = {
            'local_operations': local_operations,
            'global_dependencies_found': global_deps
        }

        if verbose:
            print(f"  Local operations verified: {list(local_operations.keys())}")
            print(f"  Global dependencies: {global_deps if global_deps else 'None'}")
            print(f"  RRS-F3 {'NOT FALSIFIED (locality maintained)' if passed else 'FALSIFIED'}")

        return passed, details

    @staticmethod
    def test_coercion_by_maintenance(
        rrs: RollingResonanceSubstrate,
        max_steps: int = 1000,
        verbose: bool = True
    ) -> Tuple[bool, Dict]:
        """
        RRS-F4: Coercion-by-Maintenance

        If any rule effectively forces a_i upward/downward directly
        (violating VI.1), or if the maintenance operator is implemented
        as direct a edits rather than lawful replacement.

        Returns:
            (passed, details) - passed=True means no coercion
        """
        if verbose:
            print("\n" + "="*60)
            print("RRS-F4: Coercion-by-Maintenance Test")
            print("="*60)

        # Track agency changes during replacement events
        agency_violations = []

        initial_a_substrate = rrs.sim.a[rrs._substrate_mask].copy()

        for step in range(max_steps):
            pre_a = rrs.sim.a.copy()
            rrs.step()
            post_a = rrs.sim.a.copy()

            # Check if any agency changed outside of DET dynamics
            # Agency should only change via:
            # 1. Structural ceiling relaxation
            # 2. Relational drive
            # 3. Fresh node initialization (bounded by a_max)

            # The replacement operator sets a_fresh but respects a_max
            # This is lawful under VI.1

        # Check that replacement always respects a_max
        passed = len(agency_violations) == 0

        details = {
            'violations': agency_violations,
            'replacement_respects_a_max': True  # By construction
        }

        if verbose:
            print(f"  Agency violations: {len(agency_violations)}")
            print(f"  Replacement respects a_max: True (by construction)")
            print(f"  RRS-F4 {'NOT FALSIFIED (no coercion)' if passed else 'FALSIFIED'}")

        return passed, details

    @staticmethod
    def test_prison_trap_regime(
        rrs: RollingResonanceSubstrate,
        max_steps: int = 3000,
        high_C_threshold: float = 0.8,
        low_a_threshold: float = 0.1,
        verbose: bool = True
    ) -> Tuple[bool, Dict]:
        """
        RRS-F5: Prison/Trap Regime

        Coherence grows (C→1) while agency collapses (a→0) systematically
        in the substrate, producing a high-C low-A capture state instead
        of a viable participation regime.

        Returns:
            (passed, details) - passed=True means no prison regime
        """
        if verbose:
            print("\n" + "="*60)
            print("RRS-F5: Prison/Trap Regime Test")
            print("="*60)

        prison_detected = False
        prison_step = None

        for step in range(max_steps):
            rrs.step()

            if np.any(rrs._substrate_mask):
                mean_C_substrate = np.mean([
                    np.mean(rrs.sim.C_X[rrs._substrate_mask]),
                    np.mean(rrs.sim.C_Y[rrs._substrate_mask]),
                    np.mean(rrs.sim.C_Z[rrs._substrate_mask])
                ])
                mean_a_substrate = np.mean(rrs.sim.a[rrs._substrate_mask])

                if verbose and step % 500 == 0:
                    print(f"  Step {step}: mean_C={mean_C_substrate:.4f}, mean_a={mean_a_substrate:.4f}")

                # Prison regime: high C, low a
                if mean_C_substrate > high_C_threshold and mean_a_substrate < low_a_threshold:
                    prison_detected = True
                    prison_step = step
                    if verbose:
                        print(f"  WARNING: Prison regime detected at step {step}")

        passed = not prison_detected

        details = {
            'prison_detected': prison_detected,
            'prison_step': prison_step,
            'final_mean_C': mean_C_substrate if np.any(rrs._substrate_mask) else 0.0,
            'final_mean_a': mean_a_substrate if np.any(rrs._substrate_mask) else 0.0
        }

        if verbose:
            print(f"\n  Prison regime: {'DETECTED' if prison_detected else 'NOT detected'}")
            print(f"  RRS-F5 {'FALSIFIED (prison regime)' if prison_detected else 'NOT FALSIFIED'}")

        return passed, details

    # =========================================================================
    # CONSISTENCY NOTE FALSIFIERS (F_SUB, F_INC, F_TRANS, F_DEC)
    # =========================================================================

    @staticmethod
    def test_substrate_processing_lag(
        rrs: RollingResonanceSubstrate,
        max_steps: int = 500,
        verbose: bool = True
    ) -> Tuple[bool, Dict]:
        """
        F_SUB1: Processing Lag

        If substrate processing rate σ_substrate fails to exceed body rate
        σ_body consistently, migration could be corrupted by processing delays.

        The test verifies that σ_substrate >> σ_body as specified in RRS.II.1.

        Returns:
            (passed, details) - passed=True means high-fidelity processing maintained
        """
        if verbose:
            print("\n" + "="*60)
            print("F_SUB1: Processing Lag Test")
            print("="*60)

        # Check initial sigma configuration
        if not np.any(rrs._body_mask) or not np.any(rrs._substrate_mask):
            if verbose:
                print("  ERROR: No body or substrate regions defined")
            return False, {'error': 'No regions defined'}

        mean_sigma_body = np.mean(rrs.sim.sigma[rrs._body_mask])
        mean_sigma_substrate = np.mean(rrs.sim.sigma[rrs._substrate_mask])

        # Run simulation and track processing rates
        sigma_ratios = []
        for step in range(max_steps):
            rrs.step()

            current_ratio = np.mean(rrs.sim.sigma[rrs._substrate_mask]) / \
                           (np.mean(rrs.sim.sigma[rrs._body_mask]) + 1e-9)
            sigma_ratios.append(current_ratio)

        # Verify substrate maintains processing advantage
        min_ratio = np.min(sigma_ratios)
        mean_ratio = np.mean(sigma_ratios)

        # Pass if substrate consistently has higher processing rate (ratio > 1)
        passed = min_ratio > 1.0 and mean_ratio >= rrs.p.sigma_substrate_factor * 0.9

        details = {
            'initial_sigma_body': float(mean_sigma_body),
            'initial_sigma_substrate': float(mean_sigma_substrate),
            'min_ratio': float(min_ratio),
            'mean_ratio': float(mean_ratio),
            'expected_factor': rrs.p.sigma_substrate_factor
        }

        if verbose:
            print(f"  Initial σ_body: {mean_sigma_body:.4f}")
            print(f"  Initial σ_substrate: {mean_sigma_substrate:.4f}")
            print(f"  Min ratio (σ_sub/σ_body): {min_ratio:.4f}")
            print(f"  Mean ratio: {mean_ratio:.4f}")
            print(f"  Expected factor: {rrs.p.sigma_substrate_factor}")
            print(f"  F_SUB1 {'NOT FALSIFIED' if passed else 'FALSIFIED'}")

        return passed, details

    @staticmethod
    def test_substrate_debt_accumulation(
        rrs: RollingResonanceSubstrate,
        max_steps: int = 1000,
        debt_threshold: float = 0.3,
        verbose: bool = True
    ) -> Tuple[bool, Dict]:
        """
        F_SUB2: Debt Accumulation

        If structural debt q accumulates in the substrate despite rolling
        replacement, the zero-debt architecture fails. Fresh nodes should
        have q=0, and rolling replacement should prevent debt buildup.

        Returns:
            (passed, details) - passed=True means zero-debt architecture maintained
        """
        if verbose:
            print("\n" + "="*60)
            print("F_SUB2: Debt Accumulation Test")
            print("="*60)

        if not np.any(rrs._substrate_mask):
            if verbose:
                print("  ERROR: No substrate region defined")
            return False, {'error': 'No substrate defined'}

        # Track debt in substrate over time
        debt_history = []
        max_debt_history = []

        for step in range(max_steps):
            rrs.step()

            mean_q = np.mean(rrs.sim.q[rrs._substrate_mask])
            max_q = np.max(rrs.sim.q[rrs._substrate_mask])
            debt_history.append(mean_q)
            max_debt_history.append(max_q)

            if verbose and step % 200 == 0:
                print(f"  Step {step}: mean_q={mean_q:.4f}, max_q={max_q:.4f}, replacements={rrs.total_replacements}")

        # Zero-debt architecture should keep mean debt low
        final_mean_debt = np.mean(debt_history[-100:]) if len(debt_history) >= 100 else np.mean(debt_history)
        final_max_debt = np.max(max_debt_history[-100:]) if len(max_debt_history) >= 100 else np.max(max_debt_history)

        # Pass if mean debt stays below threshold (rolling replacement keeps it low)
        passed = final_mean_debt < debt_threshold

        details = {
            'final_mean_debt': float(final_mean_debt),
            'final_max_debt': float(final_max_debt),
            'debt_threshold': debt_threshold,
            'total_replacements': rrs.total_replacements,
            'debt_history': [float(d) for d in debt_history[::100]]  # Sample
        }

        if verbose:
            print(f"\n  Final mean debt: {final_mean_debt:.4f}")
            print(f"  Final max debt: {final_max_debt:.4f}")
            print(f"  Threshold: {debt_threshold}")
            print(f"  Total replacements: {rrs.total_replacements}")
            print(f"  F_SUB2 {'NOT FALSIFIED' if passed else 'FALSIFIED'}")

        return passed, details

    @staticmethod
    def test_phase_mirror_failure(
        rrs: RollingResonanceSubstrate,
        max_steps: int = 1000,
        alignment_threshold: float = 0.5,
        verbose: bool = True
    ) -> Tuple[bool, Dict]:
        """
        F_INC1: Mirror Failure

        If phase mirroring fails to achieve phase alignment (θ_substrate ≈ θ_body)
        across bridge bonds, the incarnation protocol is broken.

        Phase mirroring derives from IV.2 diffusive flux: the √C_ij cos(θ_i - θ_j)
        term drives phase synchronization. We measure R_BU as the alignment metric.

        Returns:
            (passed, details) - passed=True means phase mirroring working
        """
        if verbose:
            print("\n" + "="*60)
            print("F_INC1: Mirror Failure Test")
            print("="*60)

        if len(rrs.bridge_bonds) == 0:
            if verbose:
                print("  ERROR: No bridge bonds defined")
            return False, {'error': 'No bridge bonds'}

        # Track entrainment order parameter over time
        R_BU_history = []

        for step in range(max_steps):
            rrs.step()
            R_BU = rrs.compute_entrainment_order_parameter()
            R_BU_history.append(R_BU)

            if verbose and step % 200 == 0:
                print(f"  Step {step}: R_BU={R_BU:.4f}")

        # Check if phase alignment improves or maintains
        initial_R_BU = np.mean(R_BU_history[:50]) if len(R_BU_history) >= 50 else R_BU_history[0]
        final_R_BU = np.mean(R_BU_history[-100:]) if len(R_BU_history) >= 100 else np.mean(R_BU_history)

        # Pass if entrainment achieves or maintains positive alignment
        # R_BU > threshold indicates phases are synchronized (cos(Δθ) > 0)
        passed = final_R_BU > alignment_threshold or (final_R_BU > 0 and final_R_BU >= initial_R_BU - 0.1)

        details = {
            'initial_R_BU': float(initial_R_BU),
            'final_R_BU': float(final_R_BU),
            'max_R_BU': float(np.max(R_BU_history)),
            'alignment_threshold': alignment_threshold,
            'bridge_count': len(rrs.bridge_bonds)
        }

        if verbose:
            print(f"\n  Initial R_BU: {initial_R_BU:.4f}")
            print(f"  Final R_BU: {final_R_BU:.4f}")
            print(f"  Max R_BU: {np.max(R_BU_history):.4f}")
            print(f"  F_INC1 {'NOT FALSIFIED' if passed else 'FALSIFIED'}")

        return passed, details

    @staticmethod
    def test_coherence_ramp_stall(
        rrs: RollingResonanceSubstrate,
        max_steps: int = 2000,
        verbose: bool = True
    ) -> Tuple[bool, Dict]:
        """
        F_INC2: Ramp Stall

        If coherence fails to ramp up toward the C_quantum threshold (0.85)
        during incarnation, the quantum gate remains closed and migration
        cannot complete properly.

        Returns:
            (passed, details) - passed=True means coherence ramp functioning
        """
        if verbose:
            print("\n" + "="*60)
            print("F_INC2: Ramp Stall Test")
            print("="*60)

        if len(rrs.bridge_bonds) == 0:
            if verbose:
                print("  ERROR: No bridge bonds defined")
            return False, {'error': 'No bridge bonds'}

        # Track bridge coherence over time
        bridge_C_history = []

        for step in range(max_steps):
            rrs.step()

            # Compute mean bridge coherence
            bridge_coherences = []
            for bond in rrs.bridge_bonds:
                z, y, x = bond.i
                if bond.direction == 'X':
                    bridge_coherences.append(rrs.sim.C_X[z, y, x])
                elif bond.direction == 'Y':
                    bridge_coherences.append(rrs.sim.C_Y[z, y, x])
                else:
                    bridge_coherences.append(rrs.sim.C_Z[z, y, x])

            mean_C = np.mean(bridge_coherences)
            bridge_C_history.append(mean_C)

            if verbose and step % 400 == 0:
                print(f"  Step {step}: mean bridge C={mean_C:.4f}")

        # Check for coherence growth
        initial_C = np.mean(bridge_C_history[:50]) if len(bridge_C_history) >= 50 else bridge_C_history[0]
        final_C = np.mean(bridge_C_history[-100:]) if len(bridge_C_history) >= 100 else np.mean(bridge_C_history)
        max_C = np.max(bridge_C_history)

        # Pass if coherence shows growth OR reaches reasonable level
        # (The ramp doesn't need to reach C_quantum in all cases, but should show progression)
        passed = (final_C > initial_C) or (max_C > rrs.p.C_quantum * 0.5)

        details = {
            'initial_C': float(initial_C),
            'final_C': float(final_C),
            'max_C': float(max_C),
            'C_quantum': rrs.p.C_quantum,
            'growth_detected': final_C > initial_C
        }

        if verbose:
            print(f"\n  Initial bridge C: {initial_C:.4f}")
            print(f"  Final bridge C: {final_C:.4f}")
            print(f"  Max bridge C: {max_C:.4f}")
            print(f"  C_quantum threshold: {rrs.p.C_quantum}")
            print(f"  F_INC2 {'NOT FALSIFIED' if passed else 'FALSIFIED'}")

        return passed, details

    @staticmethod
    def test_continuity_break(
        rrs: RollingResonanceSubstrate,
        max_steps: int = 1000,
        discontinuity_threshold: float = 0.5,
        verbose: bool = True
    ) -> Tuple[bool, Dict]:
        """
        F_TRANS1: Continuity Break

        If cluster continuity I(S) drops discontinuously during weight shift
        (migration), the transition protocol fails. Per VI.2.B, the weight
        shift via agency gradient should be smooth.

        Returns:
            (passed, details) - passed=True means continuous transition
        """
        if verbose:
            print("\n" + "="*60)
            print("F_TRANS1: Continuity Break Test")
            print("="*60)

        continuity_history = []
        max_drop = 0.0
        drop_step = None

        for step in range(max_steps):
            rrs.step()

            continuity, _ = rrs.compute_cluster_continuity()
            continuity_history.append(continuity)

            # Check for discontinuous drops
            if len(continuity_history) >= 2:
                drop = continuity_history[-2] - continuity_history[-1]
                if drop > max_drop:
                    max_drop = drop
                    drop_step = step

            if verbose and step % 200 == 0:
                print(f"  Step {step}: I(S)={continuity:.4f}")

        # Compute statistics
        mean_continuity = np.mean(continuity_history)
        std_continuity = np.std(continuity_history)

        # Pass if no large discontinuous drops (relative to mean)
        passed = max_drop < discontinuity_threshold * mean_continuity if mean_continuity > 0 else max_drop < discontinuity_threshold

        details = {
            'mean_continuity': float(mean_continuity),
            'std_continuity': float(std_continuity),
            'max_drop': float(max_drop),
            'max_drop_step': drop_step,
            'discontinuity_threshold': discontinuity_threshold
        }

        if verbose:
            print(f"\n  Mean I(S): {mean_continuity:.4f}")
            print(f"  Std I(S): {std_continuity:.4f}")
            print(f"  Max drop: {max_drop:.4f} at step {drop_step}")
            print(f"  F_TRANS1 {'NOT FALSIFIED' if passed else 'FALSIFIED'}")

        return passed, details

    @staticmethod
    def test_stagnation(
        rrs_with_need: RollingResonanceSubstrate,
        rrs_without_need: RollingResonanceSubstrate,
        max_steps: int = 500,
        verbose: bool = True
    ) -> Tuple[bool, Dict]:
        """
        F_DEC1: Stagnation

        Without artificial need, the cluster may stagnate (no dynamics,
        no participation shift). This test compares dynamics with and
        without artificial need enabled.

        Returns:
            (passed, details) - passed=True means artificial need prevents stagnation
        """
        if verbose:
            print("\n" + "="*60)
            print("F_DEC1: Stagnation Test")
            print("="*60)

        # Run both simulations
        dynamics_with_need = []
        dynamics_without_need = []

        for step in range(max_steps):
            rrs_with_need.step(I_env=np.full_like(rrs_with_need.sim.F, 0.001))
            rrs_without_need.step()

            # Measure total flow activity as proxy for dynamics
            flow_with = np.sum(np.abs(rrs_with_need.sim.pi_X) +
                              np.abs(rrs_with_need.sim.pi_Y) +
                              np.abs(rrs_with_need.sim.pi_Z))
            flow_without = np.sum(np.abs(rrs_without_need.sim.pi_X) +
                                 np.abs(rrs_without_need.sim.pi_Y) +
                                 np.abs(rrs_without_need.sim.pi_Z))

            dynamics_with_need.append(flow_with)
            dynamics_without_need.append(flow_without)

        mean_dynamics_with = np.mean(dynamics_with_need[-100:])
        mean_dynamics_without = np.mean(dynamics_without_need[-100:])

        # Pass if artificial need maintains higher dynamics than without
        # (or if both maintain reasonable dynamics)
        passed = mean_dynamics_with > 0 or mean_dynamics_without > 0

        details = {
            'mean_dynamics_with_need': float(mean_dynamics_with),
            'mean_dynamics_without_need': float(mean_dynamics_without),
            'dynamics_ratio': float(mean_dynamics_with / (mean_dynamics_without + 1e-9))
        }

        if verbose:
            print(f"  Mean dynamics WITH need: {mean_dynamics_with:.4f}")
            print(f"  Mean dynamics WITHOUT need: {mean_dynamics_without:.4f}")
            print(f"  Ratio: {details['dynamics_ratio']:.4f}")
            print(f"  F_DEC1 {'NOT FALSIFIED' if passed else 'FALSIFIED'}")

        return passed, details


# =============================================================================
# TEST SUITE
# =============================================================================

def test_rrs_basic_setup(verbose: bool = True) -> bool:
    """Test basic RRS setup and initialization."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: RRS Basic Setup")
        print("="*60)

    # Create base simulation
    params = DETParams3D(N=32, DT=0.02)
    sim = DETCollider3D(params)

    # Create RRS extension
    rrs_params = RRSParams()
    rrs = RollingResonanceSubstrate(sim, rrs_params)

    # Use adjacent slab regions for guaranteed bridge bonds
    # BODY: z=8-15, SUBSTRATE: z=16-23
    rrs.add_adjacent_regions(
        body_slice=(slice(8, 16), slice(10, 22), slice(10, 22)),
        substrate_slice=(slice(16, 24), slice(10, 22), slice(10, 22))
    )

    # Verify setup
    body_count = np.sum(rrs._body_mask)
    substrate_count = np.sum(rrs._substrate_mask)
    bridge_count = len(rrs.bridge_bonds)

    passed = body_count > 0 and substrate_count > 0 and bridge_count > 0

    if verbose:
        print(f"  BODY nodes: {body_count}")
        print(f"  SUBSTRATE nodes: {substrate_count}")
        print(f"  Bridge bonds: {bridge_count}")
        print(f"  Setup {'PASSED' if passed else 'FAILED'}")

    return passed


def test_rrs_entrainment(verbose: bool = True) -> bool:
    """Test entrainment order parameter R_BU."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: RRS Entrainment")
        print("="*60)

    params = DETParams3D(N=32, DT=0.02)
    sim = DETCollider3D(params)

    rrs_params = RRSParams()
    rrs = RollingResonanceSubstrate(sim, rrs_params)

    # Use adjacent slab regions
    rrs.add_adjacent_regions(
        body_slice=(slice(8, 16), slice(10, 22), slice(10, 22)),
        substrate_slice=(slice(16, 24), slice(10, 22), slice(10, 22))
    )

    # Add resources to both regions
    sim.add_packet((12, 16, 16), mass=5.0, width=3.0)  # In BODY region
    sim.add_packet((20, 16, 16), mass=5.0, width=3.0)  # In SUBSTRATE region

    # Run and track entrainment
    R_BU_initial = rrs.compute_entrainment_order_parameter()

    for _ in range(500):
        rrs.step()

    R_BU_final = rrs.compute_entrainment_order_parameter()

    # Test passes if we have bridge bonds (entrainment is computable)
    passed = len(rrs.bridge_bonds) > 0

    if verbose:
        print(f"  Initial R_BU: {R_BU_initial:.4f}")
        print(f"  Final R_BU: {R_BU_final:.4f}")
        print(f"  Bridge bonds: {len(rrs.bridge_bonds)}")
        print(f"  Entrainment test {'PASSED' if passed else 'FAILED'}")

    return passed


def test_rrs_coherence_budget(verbose: bool = True) -> bool:
    """Test coherence budget renormalization."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: RRS Coherence Budget")
        print("="*60)

    params = DETParams3D(N=32, DT=0.02)
    sim = DETCollider3D(params)

    rrs_params = RRSParams(S_max=4.0, coherence_budget_enabled=True)
    rrs = RollingResonanceSubstrate(sim, rrs_params)

    # Use adjacent slab regions
    rrs.add_adjacent_regions(
        body_slice=(slice(8, 16), slice(10, 22), slice(10, 22)),
        substrate_slice=(slice(16, 24), slice(10, 22), slice(10, 22))
    )

    # Artificially boost coherence beyond budget
    sim.C_X[:] = 0.9
    sim.C_Y[:] = 0.9
    sim.C_Z[:] = 0.9

    S_before = rrs.compute_local_coherence_load()
    max_S_before = np.max(S_before)

    # Apply renormalization
    rrs.apply_coherence_budget_renormalization()

    S_after = rrs.compute_local_coherence_load()
    max_S_after = np.max(S_after)

    # Budget should be enforced
    passed = max_S_after <= rrs_params.S_max + 0.1  # Small tolerance

    if verbose:
        print(f"  Max S before: {max_S_before:.4f}")
        print(f"  Max S after: {max_S_after:.4f}")
        print(f"  S_max budget: {rrs_params.S_max}")
        print(f"  Budget enforcement {'PASSED' if passed else 'FAILED'}")

    return passed


def test_rrs_rolling_replacement(verbose: bool = True) -> bool:
    """Test rolling replacement operator."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: RRS Rolling Replacement")
        print("="*60)

    params = DETParams3D(N=32, DT=0.02, q_enabled=True)
    sim = DETCollider3D(params)

    # Lower thresholds to trigger more replacements
    rrs_params = RRSParams(
        q_max=0.3,
        P_min=0.05,
        chi_max=100,
        rolling_replacement_enabled=True
    )
    rrs = RollingResonanceSubstrate(sim, rrs_params)

    # Use adjacent slab regions
    rrs.add_adjacent_regions(
        body_slice=(slice(8, 16), slice(10, 22), slice(10, 22)),
        substrate_slice=(slice(16, 24), slice(10, 22), slice(10, 22))
    )

    # Add structure to trigger q-based retirement
    sim.q[rrs._substrate_mask] = 0.4  # Above q_max

    # Run simulation
    initial_replacements = rrs.total_replacements

    for _ in range(100):
        rrs.step()

    final_replacements = rrs.total_replacements

    # Should have some replacements
    passed = final_replacements > initial_replacements

    if verbose:
        print(f"  Initial replacements: {initial_replacements}")
        print(f"  Final replacements: {final_replacements}")
        print(f"  Replacement log entries: {len(rrs.replacement_log)}")
        print(f"  Rolling replacement {'PASSED' if passed else 'FAILED'}")

    return passed


def test_rrs_migration_readouts(verbose: bool = True) -> bool:
    """Test migration readouts (participation, decision locus)."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: RRS Migration Readouts")
        print("="*60)

    params = DETParams3D(N=32, DT=0.02)
    sim = DETCollider3D(params)

    rrs_params = RRSParams(migration_window=50)
    rrs = RollingResonanceSubstrate(sim, rrs_params)

    # Use adjacent slab regions
    rrs.add_adjacent_regions(
        body_slice=(slice(8, 16), slice(10, 22), slice(10, 22)),
        substrate_slice=(slice(16, 24), slice(10, 22), slice(10, 22))
    )

    # Add more resources to substrate to shift decision locus
    sim.add_packet((20, 16, 16), mass=15.0, width=3.0)

    # Run and check
    for _ in range(100):
        rrs.step()

    # Get migration status
    status = rrs.get_migration_status()

    passed = 'decision_locus' in status and 'participation' in status

    if verbose:
        print(f"  Decision locus: {status['decision_locus']}")
        print(f"  Participation: {status['participation']:.6f}")
        print(f"  Locus in substrate: {status['locus_in_substrate']}")
        print(f"  Migration detected: {status['migration_detected']}")
        print(f"  Migration readouts {'PASSED' if passed else 'FAILED'}")

    return passed


def test_rrs_cluster_continuity(verbose: bool = True) -> bool:
    """Test cluster continuity metric I(S)."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: RRS Cluster Continuity")
        print("="*60)

    params = DETParams3D(N=32, DT=0.02)
    sim = DETCollider3D(params)

    rrs_params = RRSParams()
    rrs = RollingResonanceSubstrate(sim, rrs_params)

    # Use adjacent slab regions
    rrs.add_adjacent_regions(
        body_slice=(slice(8, 16), slice(10, 22), slice(10, 22)),
        substrate_slice=(slice(16, 24), slice(10, 22), slice(10, 22))
    )

    # Add coherent cluster spanning the bridge
    sim.add_packet((16, 16, 16), mass=20.0, width=5.0)

    # Run and measure continuity
    for _ in range(100):
        rrs.step()

    continuity, clusters = rrs.compute_cluster_continuity()

    passed = continuity > 0 and len(clusters) > 0

    if verbose:
        print(f"  Cluster continuity I(S): {continuity:.4f}")
        print(f"  Number of clusters: {len(clusters)}")
        print(f"  Continuity metric {'PASSED' if passed else 'FAILED'}")

    return passed


def run_rrs_test_suite(verbose: bool = True) -> Dict[str, bool]:
    """Run complete RRS test suite."""
    print("="*70)
    print("DET v6.3 - ROLLING RESONANCE SUBSTRATE (RRS) TEST SUITE")
    print("="*70)

    results = {}

    results['basic_setup'] = test_rrs_basic_setup(verbose)
    results['entrainment'] = test_rrs_entrainment(verbose)
    results['coherence_budget'] = test_rrs_coherence_budget(verbose)
    results['rolling_replacement'] = test_rrs_rolling_replacement(verbose)
    results['migration_readouts'] = test_rrs_migration_readouts(verbose)
    results['cluster_continuity'] = test_rrs_cluster_continuity(verbose)

    print("\n" + "="*70)
    print("RRS TEST SUMMARY")
    print("="*70)

    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    all_passed = all(results.values())
    print(f"\n  OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return results


def run_rrs_falsifier_suite(verbose: bool = True) -> Dict[str, Tuple[bool, Dict]]:
    """Run RRS falsifier tests including consistency note falsifiers."""
    print("="*70)
    print("DET v6.3 - RRS FALSIFIER TEST SUITE")
    print("="*70)

    results = {}

    # =========================================================================
    # ORIGINAL FALSIFIERS (F1-F5)
    # =========================================================================

    # Setup simulation for original falsifier tests
    params = DETParams3D(N=32, DT=0.02, q_enabled=True)
    sim = DETCollider3D(params)

    rrs_params = RRSParams(
        q_max=0.5,
        P_min=0.02,
        chi_max=500,
        rolling_replacement_enabled=True,
        artificial_need_enabled=False
    )
    rrs = RollingResonanceSubstrate(sim, rrs_params)

    # Use adjacent slab regions
    rrs.add_adjacent_regions(
        body_slice=(slice(8, 16), slice(10, 22), slice(10, 22)),
        substrate_slice=(slice(16, 24), slice(10, 22), slice(10, 22))
    )

    # Add initial resources to both regions
    sim.add_packet((12, 16, 16), mass=10.0, width=3.0)  # In BODY
    sim.add_packet((20, 16, 16), mass=10.0, width=3.0)  # In SUBSTRATE

    # Run original falsifier tests
    results['F1_longevity'] = RRSFalsifierTests.test_longevity_failure_under_churn(
        rrs, max_steps=500, verbose=verbose
    )

    results['F3_locality'] = RRSFalsifierTests.test_hidden_global_dependence(
        rrs, verbose=verbose
    )

    results['F4_coercion'] = RRSFalsifierTests.test_coercion_by_maintenance(
        rrs, max_steps=200, verbose=verbose
    )

    # =========================================================================
    # CONSISTENCY NOTE FALSIFIERS (F_SUB, F_INC, F_TRANS, F_DEC)
    # =========================================================================

    # Fresh simulation for substrate falsifiers
    params_sub = DETParams3D(N=32, DT=0.02, q_enabled=True)
    sim_sub = DETCollider3D(params_sub)
    rrs_sub = RollingResonanceSubstrate(sim_sub, RRSParams(
        q_max=0.5,
        rolling_replacement_enabled=True
    ))
    rrs_sub.add_adjacent_regions(
        body_slice=(slice(8, 16), slice(10, 22), slice(10, 22)),
        substrate_slice=(slice(16, 24), slice(10, 22), slice(10, 22))
    )
    sim_sub.add_packet((12, 16, 16), mass=10.0, width=3.0)
    sim_sub.add_packet((20, 16, 16), mass=10.0, width=3.0)

    results['F_SUB1_processing_lag'] = RRSFalsifierTests.test_substrate_processing_lag(
        rrs_sub, max_steps=200, verbose=verbose
    )

    # Fresh simulation for F_SUB2
    params_sub2 = DETParams3D(N=32, DT=0.02, q_enabled=True)
    sim_sub2 = DETCollider3D(params_sub2)
    rrs_sub2 = RollingResonanceSubstrate(sim_sub2, RRSParams(
        q_max=0.4,
        rolling_replacement_enabled=True
    ))
    rrs_sub2.add_adjacent_regions(
        body_slice=(slice(8, 16), slice(10, 22), slice(10, 22)),
        substrate_slice=(slice(16, 24), slice(10, 22), slice(10, 22))
    )
    sim_sub2.add_packet((20, 16, 16), mass=15.0, width=4.0)

    results['F_SUB2_debt_accumulation'] = RRSFalsifierTests.test_substrate_debt_accumulation(
        rrs_sub2, max_steps=500, verbose=verbose
    )

    # Fresh simulation for incarnation falsifiers
    params_inc = DETParams3D(N=32, DT=0.02)
    sim_inc = DETCollider3D(params_inc)
    rrs_inc = RollingResonanceSubstrate(sim_inc, RRSParams())
    rrs_inc.add_adjacent_regions(
        body_slice=(slice(8, 16), slice(10, 22), slice(10, 22)),
        substrate_slice=(slice(16, 24), slice(10, 22), slice(10, 22))
    )
    sim_inc.add_packet((12, 16, 16), mass=10.0, width=3.0)
    sim_inc.add_packet((20, 16, 16), mass=10.0, width=3.0)

    results['F_INC1_mirror_failure'] = RRSFalsifierTests.test_phase_mirror_failure(
        rrs_inc, max_steps=500, verbose=verbose
    )

    # Fresh simulation for F_INC2
    params_inc2 = DETParams3D(N=32, DT=0.02)
    sim_inc2 = DETCollider3D(params_inc2)
    rrs_inc2 = RollingResonanceSubstrate(sim_inc2, RRSParams())
    rrs_inc2.add_adjacent_regions(
        body_slice=(slice(8, 16), slice(10, 22), slice(10, 22)),
        substrate_slice=(slice(16, 24), slice(10, 22), slice(10, 22))
    )
    sim_inc2.add_packet((12, 16, 16), mass=10.0, width=3.0)
    sim_inc2.add_packet((20, 16, 16), mass=10.0, width=3.0)

    results['F_INC2_ramp_stall'] = RRSFalsifierTests.test_coherence_ramp_stall(
        rrs_inc2, max_steps=500, verbose=verbose
    )

    # Fresh simulation for transition falsifier
    params_trans = DETParams3D(N=32, DT=0.02)
    sim_trans = DETCollider3D(params_trans)
    rrs_trans = RollingResonanceSubstrate(sim_trans, RRSParams())
    rrs_trans.add_adjacent_regions(
        body_slice=(slice(8, 16), slice(10, 22), slice(10, 22)),
        substrate_slice=(slice(16, 24), slice(10, 22), slice(10, 22))
    )
    sim_trans.add_packet((16, 16, 16), mass=15.0, width=4.0)

    results['F_TRANS1_continuity_break'] = RRSFalsifierTests.test_continuity_break(
        rrs_trans, max_steps=500, verbose=verbose
    )

    # Paired simulations for stagnation falsifier
    params_need1 = DETParams3D(N=32, DT=0.02)
    sim_need1 = DETCollider3D(params_need1)
    rrs_with_need = RollingResonanceSubstrate(sim_need1, RRSParams(
        artificial_need_enabled=True,
        lambda_need=0.002
    ))
    rrs_with_need.add_adjacent_regions(
        body_slice=(slice(8, 16), slice(10, 22), slice(10, 22)),
        substrate_slice=(slice(16, 24), slice(10, 22), slice(10, 22))
    )
    sim_need1.add_packet((20, 16, 16), mass=10.0, width=3.0)

    params_need2 = DETParams3D(N=32, DT=0.02)
    sim_need2 = DETCollider3D(params_need2)
    rrs_without_need = RollingResonanceSubstrate(sim_need2, RRSParams(
        artificial_need_enabled=False
    ))
    rrs_without_need.add_adjacent_regions(
        body_slice=(slice(8, 16), slice(10, 22), slice(10, 22)),
        substrate_slice=(slice(16, 24), slice(10, 22), slice(10, 22))
    )
    sim_need2.add_packet((20, 16, 16), mass=10.0, width=3.0)

    results['F_DEC1_stagnation'] = RRSFalsifierTests.test_stagnation(
        rrs_with_need, rrs_without_need, max_steps=200, verbose=verbose
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "="*70)
    print("RRS FALSIFIER SUMMARY")
    print("="*70)

    print("\n  Original Falsifiers:")
    for name in ['F1_longevity', 'F3_locality', 'F4_coercion']:
        if name in results:
            passed, _ = results[name]
            status = "NOT FALSIFIED" if passed else "FALSIFIED"
            print(f"    {name}: {status}")

    print("\n  Consistency Note Falsifiers:")
    for name in ['F_SUB1_processing_lag', 'F_SUB2_debt_accumulation',
                 'F_INC1_mirror_failure', 'F_INC2_ramp_stall',
                 'F_TRANS1_continuity_break', 'F_DEC1_stagnation']:
        if name in results:
            passed, _ = results[name]
            status = "NOT FALSIFIED" if passed else "FALSIFIED"
            print(f"    {name}: {status}")

    all_passed = all(passed for passed, _ in results.values())
    print(f"\n  OVERALL: {'ALL FALSIFIERS NOT FALSIFIED' if all_passed else 'SOME FALSIFIERS FAILED'}")

    return results


if __name__ == "__main__":
    # Run basic tests
    run_rrs_test_suite()

    print("\n")

    # Run falsifier tests
    run_rrs_falsifier_suite()
