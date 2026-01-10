"""
DET v6 3D Collider Suite v3 - Plaquette Angular Momentum (IV.4b)
================================================================

REVISION: Addresses critic's review with the following fixes:
1. Added diff_enabled toggle to isolate rotational flux tests
2. Removed hard vacuum clamp (now clamps to 0, not F_VAC) for conservation tests
3. Added sigma_dynamic, coherence_dynamic, agency_dynamic toggles
4. Fixed axis convention comments
5. Added ablation matrix for F_L3

Key features per specification:
- L is defined on plaquettes (faces), not nodes
- Charging law: L charged by discrete curl of π (IV.4b.1)
- Rotational flux: J_rot is perpendicular gradient of L (IV.4b.2)
  - Divergence-free by construction
  - F_avg factor prevents vacuum push
- Placement: After linear momentum update (step 5b)

Falsifiers:
- F_L1: Rotational flux conservation (with proper isolation)
- F_L2: Vacuum spin does not transport (with F_vac sweep)
- F_L3: Orbital capture (with ablation matrix)

Reference: DET Theory Card 5.0 + IV.4b Patch
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PARAMETERS
# =============================================================================

@dataclass
class DETParams3D:
    """
    DET simulation parameters for 3D with plaquette angular momentum.
    
    NOTE: The presence formula P = a*σ/(1+F)/(1+H) is a simplified lattice
    realization, not the full bond-sum form from the theory card. This is
    labeled as an ablation choice for sandbox testing.
    """
    N: int = 32
    DT: float = 0.02
    F_VAC: float = 0.01           # Initial vacuum level (NOT a hard clamp)
    F_MIN: float = 0.0            # True minimum for F (0 for conservation tests)
    R: int = 2                    # Neighborhood radius (unused in lattice PDE realization)

    # Coherence
    C_init: float = 0.15

    # === FLUX COMPONENT TOGGLES ===
    # Diffusive flux (IV.2 simplified - classical gradient only)
    diff_enabled: bool = True
    
    # Linear Momentum (IV.4)
    momentum_enabled: bool = True
    alpha_pi: float = 0.12        # Momentum accumulation
    lambda_pi: float = 0.008      # Momentum decay
    mu_pi: float = 0.35           # Momentum-to-flux coupling
    pi_max: float = 3.0

    # Plaquette Angular Momentum (IV.4b)
    angular_momentum_enabled: bool = True
    alpha_L: float = 0.06         # Spin charging rate
    lambda_L: float = 0.005       # Spin decay/friction
    mu_L: float = 0.18            # Spin-to-flux coupling
    L_max: float = 5.0            # Clip for numerical stability

    # Floor repulsion (IV.3a)
    floor_enabled: bool = True
    eta_floor: float = 0.12
    F_core: float = 5.0
    floor_power: float = 2.0

    # === DYNAMICS TOGGLES (for test isolation) ===
    # Structure (q-locking)
    q_enabled: bool = True
    alpha_q: float = 0.012

    # Agency dynamics
    agency_dynamic: bool = True
    a_coupling: float = 30.0
    a_rate: float = 0.2

    # Sigma dynamics
    sigma_dynamic: bool = True
    
    # Coherence dynamics
    coherence_dynamic: bool = True

    # Numerical stability
    outflow_limit: float = 0.2

    # Phase dynamics
    phase_enabled: bool = False

    def summary(self) -> str:
        toggles = []
        if self.diff_enabled: toggles.append("diff")
        if self.momentum_enabled: toggles.append("mom")
        if self.angular_momentum_enabled: toggles.append("ang")
        if self.floor_enabled: toggles.append("floor")
        return (f"N={self.N}, DT={self.DT}, toggles=[{','.join(toggles)}]\n"
                f"Linear Mom: α_π={self.alpha_pi}, λ_π={self.lambda_pi}, μ_π={self.mu_pi}\n"
                f"Angular Mom (IV.4b): α_L={self.alpha_L}, λ_L={self.lambda_L}, μ_L={self.mu_L}")


# =============================================================================
# MAIN COLLIDER CLASS
# =============================================================================

class DETCollider3D:
    """
    DET v6 3D Collider with Plaquette Angular Momentum (IV.4b)
    
    State variables:
    - F: Resource field (per node)
    - q: Structure (per node)
    - a: Agency (per node)
    - pi_X, pi_Y, pi_Z: Linear bond momentum (per bond, directed +X/+Y/+Z)
    - L_XY, L_YZ, L_XZ: Plaquette angular momentum (per face)
    
    Indexing convention:
    - Arrays are indexed as [z, y, x] (standard numpy mgrid order)
    - axis=0 is Z, axis=1 is Y, axis=2 is X
    - pi_X[z,y,x] = momentum on bond from (z,y,x) to (z,y,x+1) in +X direction
    - L_XY[z,y,x] = angular momentum on XY-face with corner at (z,y,x)
    
    NOTE: This is a lattice PDE realization. The presence formula is simplified
    (not the full bond-sum form) - labeled as ablation for sandbox testing.
    """

    def __init__(self, params: Optional[DETParams3D] = None):
        self.p = params or DETParams3D()
        N = self.p.N

        # Per-node state
        self.F = np.ones((N, N, N), dtype=np.float64) * self.p.F_VAC
        self.q = np.zeros((N, N, N), dtype=np.float64)
        self.theta = np.random.uniform(0, 2*np.pi, (N, N, N)).astype(np.float64)
        self.a = np.ones((N, N, N), dtype=np.float64)

        # Per-bond linear momentum (directed: +X, +Y, +Z)
        self.pi_X = np.zeros((N, N, N), dtype=np.float64)
        self.pi_Y = np.zeros((N, N, N), dtype=np.float64)
        self.pi_Z = np.zeros((N, N, N), dtype=np.float64)

        # Per-plaquette angular momentum (IV.4b)
        self.L_XY = np.zeros((N, N, N), dtype=np.float64)
        self.L_YZ = np.zeros((N, N, N), dtype=np.float64)
        self.L_XZ = np.zeros((N, N, N), dtype=np.float64)

        # Coherence (per bond)
        self.C_X = np.ones((N, N, N), dtype=np.float64) * self.p.C_init
        self.C_Y = np.ones((N, N, N), dtype=np.float64) * self.p.C_init
        self.C_Z = np.ones((N, N, N), dtype=np.float64) * self.p.C_init

        self.sigma = np.ones((N, N, N), dtype=np.float64)

        # Diagnostics
        self.time = 0.0
        self.step_count = 0
        self.P = np.ones((N, N, N), dtype=np.float64)
        self.Delta_tau = np.ones((N, N, N), dtype=np.float64) * self.p.DT

    def add_packet(self, center: Tuple[int, int, int], mass: float = 10.0,
                   width: float = 3.0, momentum: Tuple[float, float, float] = (0, 0, 0)):
        """Add a 3D Gaussian resource packet with optional initial momentum."""
        N = self.p.N
        z, y, x = np.mgrid[0:N, 0:N, 0:N]
        cz, cy, cx = center
        
        # Periodic distance
        dx = (x - cx + N/2) % N - N/2
        dy = (y - cy + N/2) % N - N/2
        dz = (z - cz + N/2) % N - N/2
        
        r2 = dx**2 + dy**2 + dz**2
        envelope = np.exp(-0.5 * r2 / width**2)

        self.F += mass * envelope
        self.C_X = np.clip(self.C_X + 0.5 * envelope, self.p.C_init, 1.0)
        self.C_Y = np.clip(self.C_Y + 0.5 * envelope, self.p.C_init, 1.0)
        self.C_Z = np.clip(self.C_Z + 0.5 * envelope, self.p.C_init, 1.0)

        px, py, pz = momentum
        if px != 0 or py != 0 or pz != 0:
            self.pi_X += px * envelope
            self.pi_Y += py * envelope
            self.pi_Z += pz * envelope

        self._clip()

    def add_spin(self, center: Tuple[int, int, int], spin: float = 1.0, width: float = 4.0):
        """Add initial angular momentum (for testing)."""
        N = self.p.N
        z, y, x = np.mgrid[0:N, 0:N, 0:N]
        cz, cy, cx = center
        
        dx = (x - cx + N/2) % N - N/2
        dy = (y - cy + N/2) % N - N/2
        dz = (z - cz + N/2) % N - N/2
        
        r2 = dx**2 + dy**2 + dz**2
        envelope = np.exp(-0.5 * r2 / width**2)
        
        self.L_XY += spin * envelope
        self._clip()

    def _clip(self):
        """
        Enforce physical bounds.
        
        IMPORTANT: F is clipped to F_MIN (default 0), NOT F_VAC.
        This preserves conservation in tests. F_VAC is only for initialization.
        """
        p = self.p
        self.F = np.clip(self.F, p.F_MIN, 1000)
        self.q = np.clip(self.q, 0, 1)
        self.a = np.clip(self.a, 0, 1)
        self.pi_X = np.clip(self.pi_X, -p.pi_max, p.pi_max)
        self.pi_Y = np.clip(self.pi_Y, -p.pi_max, p.pi_max)
        self.pi_Z = np.clip(self.pi_Z, -p.pi_max, p.pi_max)
        self.L_XY = np.clip(self.L_XY, -p.L_max, p.L_max)
        self.L_YZ = np.clip(self.L_YZ, -p.L_max, p.L_max)
        self.L_XZ = np.clip(self.L_XZ, -p.L_max, p.L_max)

    def step(self):
        """
        Execute one canonical DET update step per IV.4b specification.
        
        Order:
        1. Compute P_i, Δτ_i
        2. Compute J components (diff, mom, floor, rot) - each gated by toggle
        3. Dissipation (via limiter)
        4. Update F using sender-clocked transport
        5. Update π (linear momentum)
        5b. Update L (plaquette angular momentum) ← IV.4b
        6. Update q (if q_enabled)
        7. Update a (if agency_dynamic)
        8. Update σ, C (if sigma_dynamic, coherence_dynamic)
        """
        p = self.p
        dk = p.DT

        # Neighbor access operators (periodic)
        # axis=2 is X, axis=1 is Y, axis=0 is Z
        Xp = lambda arr: np.roll(arr, -1, axis=2)  # +X neighbor
        Xm = lambda arr: np.roll(arr, 1, axis=2)   # -X neighbor
        Yp = lambda arr: np.roll(arr, -1, axis=1)  # +Y neighbor
        Ym = lambda arr: np.roll(arr, 1, axis=1)   # -Y neighbor
        Zp = lambda arr: np.roll(arr, -1, axis=0)  # +Z neighbor
        Zm = lambda arr: np.roll(arr, 1, axis=0)   # -Z neighbor

        # ================================================================
        # STEP 1: Presence and proper time (simplified lattice form)
        # NOTE: This is NOT the full bond-sum form - labeled as ablation
        # ================================================================
        H = self.sigma
        self.P = self.a * self.sigma / (1.0 + self.F) / (1.0 + H)
        self.Delta_tau = self.P * dk

        # Bond-averaged proper time
        Delta_tau_Xp = 0.5 * (self.Delta_tau + Xp(self.Delta_tau))
        Delta_tau_Yp = 0.5 * (self.Delta_tau + Yp(self.Delta_tau))
        Delta_tau_Zp = 0.5 * (self.Delta_tau + Zp(self.Delta_tau))

        # Plaquette-averaged proper time (IV.4b.1)
        Delta_tau_XY = 0.25 * (self.Delta_tau + Xp(self.Delta_tau) + 
                               Yp(self.Delta_tau) + Xp(Yp(self.Delta_tau)))
        Delta_tau_YZ = 0.25 * (self.Delta_tau + Yp(self.Delta_tau) + 
                               Zp(self.Delta_tau) + Yp(Zp(self.Delta_tau)))
        Delta_tau_XZ = 0.25 * (self.Delta_tau + Xp(self.Delta_tau) + 
                               Zp(self.Delta_tau) + Xp(Zp(self.Delta_tau)))

        # ================================================================
        # STEP 2: Compute flux components
        # ================================================================
        
        # Initialize flux arrays to zero
        J_Xp = np.zeros_like(self.F)
        J_Xm = np.zeros_like(self.F)
        J_Yp = np.zeros_like(self.F)
        J_Ym = np.zeros_like(self.F)
        J_Zp = np.zeros_like(self.F)
        J_Zm = np.zeros_like(self.F)
        
        # Track diffusive flux separately for momentum charging
        J_diff_Xp = np.zeros_like(self.F)
        J_diff_Yp = np.zeros_like(self.F)
        J_diff_Zp = np.zeros_like(self.F)

        # --- J^diff: Diffusive flux (IV.2 simplified) ---
        if p.diff_enabled:
            # Classical gradient drive
            classical_Xp = self.F - Xp(self.F)
            classical_Xm = self.F - Xm(self.F)
            classical_Yp = self.F - Yp(self.F)
            classical_Ym = self.F - Ym(self.F)
            classical_Zp = self.F - Zp(self.F)
            classical_Zm = self.F - Zm(self.F)

            # Agency gating
            g_Xp = np.sqrt(self.a * Xp(self.a))
            g_Xm = np.sqrt(self.a * Xm(self.a))
            g_Yp = np.sqrt(self.a * Yp(self.a))
            g_Ym = np.sqrt(self.a * Ym(self.a))
            g_Zp = np.sqrt(self.a * Zp(self.a))
            g_Zm = np.sqrt(self.a * Zm(self.a))

            # Conductivity
            cond_Xp = self.sigma * (self.C_X + 1e-4)
            cond_Xm = self.sigma * (Xm(self.C_X) + 1e-4)
            cond_Yp = self.sigma * (self.C_Y + 1e-4)
            cond_Ym = self.sigma * (Ym(self.C_Y) + 1e-4)
            cond_Zp = self.sigma * (self.C_Z + 1e-4)
            cond_Zm = self.sigma * (Zm(self.C_Z) + 1e-4)

            J_diff_Xp = g_Xp * cond_Xp * classical_Xp
            J_diff_Xm = g_Xm * cond_Xm * classical_Xm
            J_diff_Yp = g_Yp * cond_Yp * classical_Yp
            J_diff_Ym = g_Ym * cond_Ym * classical_Ym
            J_diff_Zp = g_Zp * cond_Zp * classical_Zp
            J_diff_Zm = g_Zm * cond_Zm * classical_Zm

            J_Xp += J_diff_Xp
            J_Xm += J_diff_Xm
            J_Yp += J_diff_Yp
            J_Ym += J_diff_Ym
            J_Zp += J_diff_Zp
            J_Zm += J_diff_Zm

        # --- J^mom: Linear momentum flux (IV.4) ---
        if p.momentum_enabled:
            F_avg_Xp = 0.5 * (self.F + Xp(self.F))
            F_avg_Xm = 0.5 * (self.F + Xm(self.F))
            F_avg_Yp = 0.5 * (self.F + Yp(self.F))
            F_avg_Ym = 0.5 * (self.F + Ym(self.F))
            F_avg_Zp = 0.5 * (self.F + Zp(self.F))
            F_avg_Zm = 0.5 * (self.F + Zm(self.F))

            J_Xp += p.mu_pi * self.sigma * self.pi_X * F_avg_Xp
            J_Xm -= p.mu_pi * self.sigma * Xm(self.pi_X) * F_avg_Xm
            J_Yp += p.mu_pi * self.sigma * self.pi_Y * F_avg_Yp
            J_Ym -= p.mu_pi * self.sigma * Ym(self.pi_Y) * F_avg_Ym
            J_Zp += p.mu_pi * self.sigma * self.pi_Z * F_avg_Zp
            J_Zm -= p.mu_pi * self.sigma * Zm(self.pi_Z) * F_avg_Zm

        # --- J^floor: Floor repulsion (IV.3a) ---
        if p.floor_enabled:
            # Need classical gradients if not already computed
            if not p.diff_enabled:
                classical_Xp = self.F - Xp(self.F)
                classical_Xm = self.F - Xm(self.F)
                classical_Yp = self.F - Yp(self.F)
                classical_Ym = self.F - Ym(self.F)
                classical_Zp = self.F - Zp(self.F)
                classical_Zm = self.F - Zm(self.F)
            
            s = np.maximum(0, (self.F - p.F_core) / p.F_core)**p.floor_power
            J_Xp += p.eta_floor * self.sigma * (s + Xp(s)) * classical_Xp
            J_Xm += p.eta_floor * self.sigma * (s + Xm(s)) * classical_Xm
            J_Yp += p.eta_floor * self.sigma * (s + Yp(s)) * classical_Yp
            J_Ym += p.eta_floor * self.sigma * (s + Ym(s)) * classical_Ym
            J_Zp += p.eta_floor * self.sigma * (s + Zp(s)) * classical_Zp
            J_Zm += p.eta_floor * self.sigma * (s + Zm(s)) * classical_Zm

        # --- J^rot: Rotational flux from plaquette angular momentum (IV.4b.2) ---
        if p.angular_momentum_enabled:
            # Edge-averaged F for rotational flux
            F_avg_Xp = 0.5 * (self.F + Xp(self.F))
            F_avg_Yp = 0.5 * (self.F + Yp(self.F))
            F_avg_Zp = 0.5 * (self.F + Zp(self.F))
            
            # L_XY affects X and Y edges (circulation in XY plane)
            # J^rot_X from L_XY: perpendicular gradient in Y direction
            J_rot_Xp_from_XY = p.mu_L * self.sigma * F_avg_Xp * (self.L_XY - Ym(self.L_XY))
            # J^rot_Y from L_XY: perpendicular gradient in X direction (negative)
            J_rot_Yp_from_XY = -p.mu_L * self.sigma * F_avg_Yp * (self.L_XY - Xm(self.L_XY))
            
            # L_YZ affects Y and Z edges (circulation in YZ plane)
            J_rot_Yp_from_YZ = p.mu_L * self.sigma * F_avg_Yp * (self.L_YZ - Zm(self.L_YZ))
            J_rot_Zp_from_YZ = -p.mu_L * self.sigma * F_avg_Zp * (self.L_YZ - Ym(self.L_YZ))
            
            # L_XZ affects X and Z edges (circulation in XZ plane)
            J_rot_Xp_from_XZ = p.mu_L * self.sigma * F_avg_Xp * (self.L_XZ - Zm(self.L_XZ))
            J_rot_Zp_from_XZ = -p.mu_L * self.sigma * F_avg_Zp * (self.L_XZ - Xm(self.L_XZ))
            
            # Total rotational flux
            J_rot_Xp = J_rot_Xp_from_XY + J_rot_Xp_from_XZ
            J_rot_Yp = J_rot_Yp_from_XY + J_rot_Yp_from_YZ
            J_rot_Zp = J_rot_Zp_from_YZ + J_rot_Zp_from_XZ
            
            J_Xp += J_rot_Xp
            J_Xm -= Xm(J_rot_Xp)
            J_Yp += J_rot_Yp
            J_Ym -= Ym(J_rot_Yp)
            J_Zp += J_rot_Zp
            J_Zm -= Zm(J_rot_Zp)

        # ================================================================
        # STEP 3: Dissipation (conservative limiter)
        # ================================================================
        total_outflow = (np.maximum(0, J_Xp) + np.maximum(0, J_Xm) +
                         np.maximum(0, J_Yp) + np.maximum(0, J_Ym) +
                         np.maximum(0, J_Zp) + np.maximum(0, J_Zm))
        max_total_out = p.outflow_limit * self.F / (self.Delta_tau + 1e-9)
        scale = np.minimum(1.0, max_total_out / (total_outflow + 1e-9))

        J_Xp_lim = np.where(J_Xp > 0, J_Xp * scale, J_Xp)
        J_Xm_lim = np.where(J_Xm > 0, J_Xm * scale, J_Xm)
        J_Yp_lim = np.where(J_Yp > 0, J_Yp * scale, J_Yp)
        J_Ym_lim = np.where(J_Ym > 0, J_Ym * scale, J_Ym)
        J_Zp_lim = np.where(J_Zp > 0, J_Zp * scale, J_Zp)
        J_Zm_lim = np.where(J_Zm > 0, J_Zm * scale, J_Zm)

        J_diff_Xp_scaled = np.where(J_diff_Xp > 0, J_diff_Xp * scale, J_diff_Xp) if p.diff_enabled else np.zeros_like(self.F)
        J_diff_Yp_scaled = np.where(J_diff_Yp > 0, J_diff_Yp * scale, J_diff_Yp) if p.diff_enabled else np.zeros_like(self.F)
        J_diff_Zp_scaled = np.where(J_diff_Zp > 0, J_diff_Zp * scale, J_diff_Zp) if p.diff_enabled else np.zeros_like(self.F)

        # ================================================================
        # STEP 4: Update F using sender-clocked transport
        # ================================================================
        transfer_Xp = J_Xp_lim * self.Delta_tau
        transfer_Xm = J_Xm_lim * self.Delta_tau
        transfer_Yp = J_Yp_lim * self.Delta_tau
        transfer_Ym = J_Ym_lim * self.Delta_tau
        transfer_Zp = J_Zp_lim * self.Delta_tau
        transfer_Zm = J_Zm_lim * self.Delta_tau

        outflow = transfer_Xp + transfer_Xm + transfer_Yp + transfer_Ym + transfer_Zp + transfer_Zm
        inflow = (Xm(transfer_Xp) + Xp(transfer_Xm) +
                  Ym(transfer_Yp) + Yp(transfer_Ym) +
                  Zm(transfer_Zp) + Zp(transfer_Zm))

        dF = inflow - outflow
        # NOTE: Clamp to F_MIN (default 0), NOT F_VAC, to preserve conservation
        self.F = np.clip(self.F + dF, p.F_MIN, 1000)

        # ================================================================
        # STEP 5: Update π (linear momentum)
        # ================================================================
        if p.momentum_enabled:
            decay_X = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_Xp)
            decay_Y = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_Yp)
            decay_Z = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_Zp)
            self.pi_X = decay_X * self.pi_X + p.alpha_pi * J_diff_Xp_scaled * Delta_tau_Xp
            self.pi_Y = decay_Y * self.pi_Y + p.alpha_pi * J_diff_Yp_scaled * Delta_tau_Yp
            self.pi_Z = decay_Z * self.pi_Z + p.alpha_pi * J_diff_Zp_scaled * Delta_tau_Zp

        # ================================================================
        # STEP 5b: Update L (plaquette angular momentum) - IV.4b.1
        # ================================================================
        if p.angular_momentum_enabled:
            # Discrete curl of π for each plaquette orientation
            # XY plaquette: curl of (π_X, π_Y)
            curl_XY = self.pi_X + Xp(self.pi_Y) - Yp(self.pi_X) - self.pi_Y
            # YZ plaquette: curl of (π_Y, π_Z)
            curl_YZ = self.pi_Y + Yp(self.pi_Z) - Zp(self.pi_Y) - self.pi_Z
            # XZ plaquette: curl of (π_X, π_Z)
            curl_XZ = self.pi_Z + Zp(self.pi_X) - Xp(self.pi_Z) - self.pi_X

            # Presence-clocked update (IV.4b.1)
            decay_XY = np.maximum(0.0, 1.0 - p.lambda_L * Delta_tau_XY)
            decay_YZ = np.maximum(0.0, 1.0 - p.lambda_L * Delta_tau_YZ)
            decay_XZ = np.maximum(0.0, 1.0 - p.lambda_L * Delta_tau_XZ)
            
            self.L_XY = decay_XY * self.L_XY + p.alpha_L * curl_XY * Delta_tau_XY
            self.L_YZ = decay_YZ * self.L_YZ + p.alpha_L * curl_YZ * Delta_tau_YZ
            self.L_XZ = decay_XZ * self.L_XZ + p.alpha_L * curl_XZ * Delta_tau_XZ

        # ================================================================
        # STEP 6: Update q (q-locking)
        # ================================================================
        if p.q_enabled:
            self.q = np.clip(self.q + p.alpha_q * np.maximum(0, -dF), 0, 1)

        # ================================================================
        # STEP 7: Update a (agency) - GATED BY agency_dynamic
        # ================================================================
        if p.agency_dynamic:
            a_target = 1.0 / (1.0 + p.a_coupling * self.q**2)
            self.a = self.a + p.a_rate * (a_target - self.a)

        # ================================================================
        # STEP 8: Update σ, C - GATED BY sigma_dynamic, coherence_dynamic
        # ================================================================
        J_mag = (np.abs(J_Xp_lim) + np.abs(J_Yp_lim) + np.abs(J_Zp_lim)) / 3.0
        
        if p.coherence_dynamic:
            self.C_X = np.clip(self.C_X + 0.04 * np.abs(J_Xp_lim) * self.Delta_tau
                              - 0.002 * self.C_X * self.Delta_tau, p.C_init, 1.0)
            self.C_Y = np.clip(self.C_Y + 0.04 * np.abs(J_Yp_lim) * self.Delta_tau
                              - 0.002 * self.C_Y * self.Delta_tau, p.C_init, 1.0)
            self.C_Z = np.clip(self.C_Z + 0.04 * np.abs(J_Zp_lim) * self.Delta_tau
                              - 0.002 * self.C_Z * self.Delta_tau, p.C_init, 1.0)

        if p.sigma_dynamic:
            self.sigma = 1.0 + 0.1 * np.log(1.0 + J_mag)

        self._clip()
        self.time += dk
        self.step_count += 1

    # ================================================================
    # DIAGNOSTICS
    # ================================================================

    def total_F(self) -> float:
        return float(np.sum(self.F))

    def total_linear_momentum(self) -> Tuple[float, float, float]:
        return float(np.sum(self.pi_X)), float(np.sum(self.pi_Y)), float(np.sum(self.pi_Z))

    def total_angular_momentum(self) -> Tuple[float, float, float]:
        """Total angular momentum (L_x from YZ, L_y from XZ, L_z from XY)."""
        return float(np.sum(self.L_YZ)), float(np.sum(self.L_XZ)), float(np.sum(self.L_XY))

    def center_of_mass(self) -> Tuple[float, float, float]:
        N = self.p.N
        z, y, x = np.mgrid[0:N, 0:N, 0:N]
        total = np.sum(self.F) + 1e-9
        return (float(np.sum(x * self.F) / total), 
                float(np.sum(y * self.F) / total), 
                float(np.sum(z * self.F) / total))

    def find_blobs(self, threshold_ratio: float = 50.0) -> List[Dict]:
        """Find distinct blobs using connected components."""
        from scipy.ndimage import label
        threshold = self.p.F_VAC * threshold_ratio
        above = self.F > threshold
        labeled, num = label(above)
        N = self.p.N
        z, y, x = np.mgrid[0:N, 0:N, 0:N]

        blobs = []
        for i in range(1, num + 1):
            mask = labeled == i
            if np.sum(mask) == 0:
                continue
            weights = self.F[mask]
            total_mass = float(np.sum(weights))
            if total_mass < 0.5:
                continue
            blobs.append({
                'x': float(np.sum(x[mask] * weights) / total_mass),
                'y': float(np.sum(y[mask] * weights) / total_mass),
                'z': float(np.sum(z[mask] * weights) / total_mass),
                'mass': total_mass,
                'size': int(np.sum(mask))
            })
        blobs.sort(key=lambda c: -c['mass'])
        return blobs

    def separation(self) -> Tuple[float, int]:
        blobs = self.find_blobs()
        if len(blobs) < 2:
            return 0.0, len(blobs)
        N = self.p.N
        dx = blobs[1]['x'] - blobs[0]['x']
        dy = blobs[1]['y'] - blobs[0]['y']
        dz = blobs[1]['z'] - blobs[0]['z']
        dx = dx - N if dx > N/2 else (dx + N if dx < -N/2 else dx)
        dy = dy - N if dy > N/2 else (dy + N if dy < -N/2 else dy)
        dz = dz - N if dz > N/2 else (dz + N if dz < -N/2 else dz)
        return float(np.sqrt(dx**2 + dy**2 + dz**2)), len(blobs)

    def pi_energy_proxy(self) -> float:
        """Proxy for linear momentum energy: sum of |π|²."""
        return float(np.sum(self.pi_X**2 + self.pi_Y**2 + self.pi_Z**2))

    def L_energy_proxy(self) -> float:
        """Proxy for angular momentum energy: sum of |L|²."""
        return float(np.sum(self.L_XY**2 + self.L_YZ**2 + self.L_XZ**2))

    def rot_flux_magnitude(self) -> float:
        """Magnitude of rotational flux (for diagnostics)."""
        if not self.p.angular_momentum_enabled:
            return 0.0
        Xm = lambda arr: np.roll(arr, 1, axis=2)
        Ym = lambda arr: np.roll(arr, 1, axis=1)
        Zm = lambda arr: np.roll(arr, 1, axis=0)
        Xp = lambda arr: np.roll(arr, -1, axis=2)
        Yp = lambda arr: np.roll(arr, -1, axis=1)
        
        F_avg_Xp = 0.5 * (self.F + Xp(self.F))
        F_avg_Yp = 0.5 * (self.F + Yp(self.F))
        
        J_rot_Xp = self.p.mu_L * self.sigma * F_avg_Xp * (self.L_XY - Ym(self.L_XY))
        J_rot_Yp = -self.p.mu_L * self.sigma * F_avg_Yp * (self.L_XY - Xm(self.L_XY))
        
        return float(np.sum(np.abs(J_rot_Xp) + np.abs(J_rot_Yp)))


# =============================================================================
# FALSIFIER TESTS (IV.4b) - WITH PROPER ISOLATION
# =============================================================================

def test_F_L1_rotational_flux_conservation(verbose: bool = True) -> Dict:
    """
    F_L1: Rotational Flux Conservation (PROPERLY ISOLATED)
    
    With J^rot enabled and ALL OTHER dynamics disabled, verify:
    - Total F is conserved to machine precision
    - COM develops only pure circulation (no net drift)
    
    Isolation:
    - diff_enabled=False
    - momentum_enabled=False
    - floor_enabled=False
    - q_enabled=False
    - agency_dynamic=False
    - sigma_dynamic=False
    - coherence_dynamic=False
    - F_MIN=0 (no vacuum clamp)
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F_L1: Rotational Flux Conservation (ISOLATED)")
        print("="*60)

    # Full isolation
    params = DETParams3D(
        N=24,
        F_MIN=0.0,                    # No vacuum clamp
        diff_enabled=False,           # No diffusion
        momentum_enabled=False,       # No linear momentum flux
        floor_enabled=False,          # No floor
        q_enabled=False,              # No q-locking
        agency_dynamic=False,         # Freeze agency
        sigma_dynamic=False,          # Freeze sigma
        coherence_dynamic=False,      # Freeze coherence
        angular_momentum_enabled=True # Only rotational flux
    )
    sim = DETCollider3D(params)
    center = params.N // 2

    # Add mass and spin
    sim.add_packet((center, center, center), mass=15.0, width=4.0)
    sim.add_spin((center, center, center), spin=2.0, width=5.0)

    initial_F = sim.total_F()
    initial_com = sim.center_of_mass()
    
    mass_history = []
    com_drift = []
    
    for t in range(500):
        sim.step()
        mass_history.append(sim.total_F())
        com = sim.center_of_mass()
        drift = np.sqrt((com[0] - initial_com[0])**2 + 
                       (com[1] - initial_com[1])**2 + 
                       (com[2] - initial_com[2])**2)
        com_drift.append(drift)

    final_F = sim.total_F()
    mass_err = abs(final_F - initial_F) / initial_F
    max_drift = max(com_drift)
    
    # Pass criteria: mass conserved to machine precision (<1e-10), no COM drift (<0.1 cell)
    passed = mass_err < 1e-10 and max_drift < 0.1

    result = {
        'passed': passed,
        'initial_F': initial_F,
        'final_F': final_F,
        'mass_err': mass_err,
        'max_drift': max_drift,
        'mass_history': mass_history,
        'com_drift': com_drift
    }

    if verbose:
        print(f"  Initial mass: {initial_F:.10f}")
        print(f"  Final mass: {final_F:.10f}")
        print(f"  Mass error: {mass_err:.2e} (threshold: 1e-10)")
        print(f"  Max COM drift: {max_drift:.6f} cells (threshold: 0.1)")
        print(f"  F_L1 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F_L2_vacuum_spin_no_transport(verbose: bool = True) -> Dict:
    """
    F_L2: Vacuum Spin Does Not Transport (WITH F_VAC SWEEP)
    
    Initialize L≠0 but F=F_vac everywhere. Verify:
    - max|J^rot| scales linearly with F_vac
    - Net change in F is machine-noise small
    
    Isolation: Same as F_L1
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F_L2: Vacuum Spin Does Not Transport")
        print("="*60)

    results = []
    F_vac_values = [0.001, 0.01, 0.1]
    
    for F_vac in F_vac_values:
        params = DETParams3D(
            N=20,
            F_VAC=F_vac,
            F_MIN=0.0,
            diff_enabled=False,
            momentum_enabled=False,
            floor_enabled=False,
            q_enabled=False,
            agency_dynamic=False,
            sigma_dynamic=False,
            coherence_dynamic=False,
            angular_momentum_enabled=True
        )
        sim = DETCollider3D(params)
        
        # Pure vacuum with nonzero spin
        sim.F = np.ones_like(sim.F) * F_vac
        center = params.N // 2
        sim.add_spin((center, center, center), spin=2.0, width=5.0)

        initial_F = sim.total_F()
        max_J_rot = 0
        
        for t in range(200):
            max_J_rot = max(max_J_rot, sim.rot_flux_magnitude())
            sim.step()

        final_F = sim.total_F()
        mass_change = abs(final_F - initial_F)
        
        results.append({
            'F_vac': F_vac,
            'max_J_rot': max_J_rot,
            'mass_change': mass_change
        })
        
        if verbose:
            print(f"  F_vac={F_vac}: max|J_rot|={max_J_rot:.6f}, ΔF={mass_change:.2e}")

    # Check linear scaling: J_rot should scale with F_vac
    # Ratio of J_rot values should match ratio of F_vac values
    ratio_J = results[2]['max_J_rot'] / (results[0]['max_J_rot'] + 1e-15)
    ratio_F = F_vac_values[2] / F_vac_values[0]
    scaling_ok = 0.5 < ratio_J / ratio_F < 2.0  # Within factor of 2
    
    # Mass change should be negligible
    mass_ok = all(r['mass_change'] < 1e-10 for r in results)
    
    passed = scaling_ok and mass_ok

    if verbose:
        print(f"\n  J_rot scaling ratio: {ratio_J:.2f} (expected ~{ratio_F:.0f})")
        print(f"  Scaling OK: {scaling_ok}")
        print(f"  Mass conservation OK: {mass_ok}")
        print(f"  F_L2 {'PASSED' if passed else 'FAILED'}")

    return {'passed': passed, 'results': results, 'scaling_ok': scaling_ok, 'mass_ok': mass_ok}


def test_F_L3_ablation_matrix(verbose: bool = True) -> Dict:
    """
    F_L3: Orbital Capture - ABLATION MATRIX
    
    Run 4 configurations to verify angular momentum is mediating the orbit:
    1. momentum ON, angular OFF - baseline (should NOT orbit stably)
    2. momentum ON, angular ON - full system (should orbit)
    3. momentum OFF, angular ON - should fail (no π to charge L)
    4. momentum ON, angular ON, floor OFF - test floor contribution
    
    Track: separation, L_z, π energy, rotational flux magnitude
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F_L3: Orbital Capture - ABLATION MATRIX")
        print("="*60)

    configs = [
        {'name': 'mom_only', 'momentum_enabled': True, 'angular_momentum_enabled': False, 'floor_enabled': True},
        {'name': 'full', 'momentum_enabled': True, 'angular_momentum_enabled': True, 'floor_enabled': True},
        {'name': 'ang_only', 'momentum_enabled': False, 'angular_momentum_enabled': True, 'floor_enabled': True},
        {'name': 'no_floor', 'momentum_enabled': True, 'angular_momentum_enabled': True, 'floor_enabled': False},
    ]
    
    results = {}
    steps = 1500
    
    for cfg in configs:
        if verbose:
            print(f"\n  Config: {cfg['name']}")
        
        params = DETParams3D(
            N=50,
            momentum_enabled=cfg['momentum_enabled'],
            angular_momentum_enabled=cfg['angular_momentum_enabled'],
            floor_enabled=cfg['floor_enabled']
        )
        sim = DETCollider3D(params)
        center = params.N // 2
        sep_init = 15
        b = 3

        sim.add_packet((center - sep_init, center - b, center), mass=10.0, width=2.5, momentum=(2.0, 0, 0))
        sim.add_packet((center + sep_init, center + b, center), mass=10.0, width=2.5, momentum=(-2.0, 0, 0))

        rec = {'t': [], 'sep': [], 'blobs': [], 'L_z': [], 'pi_energy': [], 'rot_flux': [], 'angle': []}
        prev_angle = 0
        total_angle = 0

        for t in range(steps):
            sep, num = sim.separation()
            L = sim.total_angular_momentum()
            
            blobs = sim.find_blobs()
            if len(blobs) >= 2:
                dx = blobs[1]['x'] - blobs[0]['x']
                dy = blobs[1]['y'] - blobs[0]['y']
                angle = np.arctan2(dy, dx)
                d_angle = angle - prev_angle
                if d_angle > np.pi: d_angle -= 2*np.pi
                elif d_angle < -np.pi: d_angle += 2*np.pi
                total_angle += d_angle
                prev_angle = angle

            rec['t'].append(t)
            rec['sep'].append(sep)
            rec['blobs'].append(num)
            rec['L_z'].append(L[2])
            rec['pi_energy'].append(sim.pi_energy_proxy())
            rec['rot_flux'].append(sim.rot_flux_magnitude())
            rec['angle'].append(total_angle)

            sim.step()

        # Analyze
        sep_array = np.array(rec['sep'])
        valid_seps = sep_array[sep_array > 0]
        
        if len(valid_seps) > 10:
            second_half = valid_seps[len(valid_seps)//2:]
            sep_mean = np.mean(second_half)
            sep_std = np.std(second_half)
            sep_max = np.max(second_half)
            total_revolutions = abs(total_angle) / (2 * np.pi)
            bounded = sep_max < sep_init * 2.5
            orbital_capture = bounded and (sep_std > 0.3 or total_revolutions > 0.1)
        else:
            sep_mean = sep_std = sep_max = total_revolutions = 0
            orbital_capture = False

        rec['sep_mean'] = sep_mean
        rec['sep_std'] = sep_std
        rec['sep_max'] = sep_max
        rec['total_revolutions'] = total_revolutions
        rec['orbital_capture'] = orbital_capture
        rec['final_L_z'] = rec['L_z'][-1]
        rec['final_pi_energy'] = rec['pi_energy'][-1]
        rec['final_rot_flux'] = rec['rot_flux'][-1]
        
        results[cfg['name']] = rec

        if verbose:
            print(f"    Orbital capture: {orbital_capture}")
            print(f"    Revolutions: {total_revolutions:.2f}")
            print(f"    Final L_z: {rec['final_L_z']:.4f}")
            print(f"    Final π energy: {rec['final_pi_energy']:.2f}")
            print(f"    Final rot flux: {rec['final_rot_flux']:.4f}")

    # Summary analysis
    if verbose:
        print("\n" + "-"*60)
        print("ABLATION SUMMARY")
        print("-"*60)
        print(f"  mom_only (baseline): orbit={results['mom_only']['orbital_capture']}")
        print(f"  full (mom+ang): orbit={results['full']['orbital_capture']}")
        print(f"  ang_only (no mom): orbit={results['ang_only']['orbital_capture']}")
        print(f"  no_floor: orbit={results['no_floor']['orbital_capture']}")
        
        # Key insight: full should orbit, mom_only should NOT (or less stable)
        ang_helps = results['full']['total_revolutions'] > results['mom_only']['total_revolutions']
        print(f"\n  Angular momentum helps orbit: {ang_helps}")
        print(f"    full revs: {results['full']['total_revolutions']:.2f}")
        print(f"    mom_only revs: {results['mom_only']['total_revolutions']:.2f}")

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_ablation_results(results: Dict, filename: str = 'det_3d_ablation.png'):
    """Visualize ablation matrix results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    configs = ['mom_only', 'full', 'ang_only', 'no_floor']
    titles = ['Momentum Only', 'Full (Mom+Ang)', 'Angular Only', 'No Floor']
    colors = ['blue', 'green', 'orange', 'red']
    
    # Separation over time
    ax = axes[0, 0]
    for cfg, title, color in zip(configs, titles, colors):
        if cfg in results:
            ax.plot(results[cfg]['t'], results[cfg]['sep'], color=color, label=title, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Separation')
    ax.set_title('Inter-body Separation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Angular momentum over time
    ax = axes[0, 1]
    for cfg, title, color in zip(configs, titles, colors):
        if cfg in results:
            ax.plot(results[cfg]['t'], results[cfg]['L_z'], color=color, label=title, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('L_z')
    ax.set_title('Z-Angular Momentum')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Angle winding
    ax = axes[1, 0]
    for cfg, title, color in zip(configs, titles, colors):
        if cfg in results:
            ax.plot(results[cfg]['t'], np.array(results[cfg]['angle']) / (2*np.pi), 
                   color=color, label=title, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Revolutions')
    ax.set_title('Relative Angle Winding')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Summary table
    ax = axes[1, 1]
    ax.axis('off')
    summary = "ABLATION MATRIX SUMMARY\n" + "="*40 + "\n\n"
    for cfg, title in zip(configs, titles):
        if cfg in results:
            r = results[cfg]
            summary += f"{title}:\n"
            summary += f"  Orbit: {'YES' if r['orbital_capture'] else 'NO'}\n"
            summary += f"  Revs: {r['total_revolutions']:.2f}\n"
            summary += f"  L_z: {r['final_L_z']:.4f}\n\n"
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('DET v6 3D Collider - F_L3 Ablation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# =============================================================================
# MAIN TEST SUITE
# =============================================================================

def run_full_test_suite():
    """Run the complete IV.4b angular momentum test suite with fixes."""
    print("="*70)
    print("DET v6 3D COLLIDER v3 - PLAQUETTE ANGULAR MOMENTUM (IV.4b)")
    print("WITH CRITIC'S FIXES: Proper isolation, no vacuum clamp, ablation matrix")
    print("="*70)

    results = {}

    # F_L1: Rotational flux conservation (isolated)
    results['F_L1'] = test_F_L1_rotational_flux_conservation(verbose=True)

    # F_L2: Vacuum spin doesn't transport (with sweep)
    results['F_L2'] = test_F_L2_vacuum_spin_no_transport(verbose=True)

    # F_L3: Orbital capture (ablation matrix)
    results['F_L3'] = test_F_L3_ablation_matrix(verbose=True)

    # Summary
    print("\n" + "="*70)
    print("SUITE SUMMARY")
    print("="*70)
    print(f"  F_L1 (Rotational flux conservation): {'PASS' if results['F_L1']['passed'] else 'FAIL'}")
    print(f"  F_L2 (Vacuum spin no transport): {'PASS' if results['F_L2']['passed'] else 'FAIL'}")
    print(f"  F_L3 (Orbital capture - full config): {'PASS' if results['F_L3']['full']['orbital_capture'] else 'FAIL'}")

    return results


if __name__ == "__main__":
    start = time.time()

    results = run_full_test_suite()

    # Visualization
    if 'F_L3' in results:
        visualize_ablation_results(results['F_L3'], './det_3d_ablation_v3.png')

    elapsed = time.time() - start
    print(f"\nTotal runtime: {elapsed:.1f}s")
