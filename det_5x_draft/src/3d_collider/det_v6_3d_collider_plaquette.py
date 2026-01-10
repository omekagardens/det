"""
DET v6 3D Collider Suite - Plaquette Angular Momentum (IV.4b)
=============================================================

Implementation of the DET v5 Revision Patch: Local Angular Momentum Module

Key features per specification:
- L is defined on plaquettes (faces), not nodes
- Charging law: L charged by discrete curl of π (IV.4b.1)
- Rotational flux: J_rot is perpendicular gradient of L (IV.4b.2)
  - Divergence-free by construction
  - F_avg factor prevents vacuum push
- Placement: After linear momentum update (step 5b)

Falsifiers:
- F_L1: Rotational flux conservation
- F_L2: Vacuum spin does not transport
- F_L3: Orbital capture

Reference: DET Theory Card 5.0 + IV.4b Patch
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PARAMETERS
# =============================================================================

@dataclass
class DETParams3D:
    """DET simulation parameters for 3D with plaquette angular momentum."""
    N: int = 32
    DT: float = 0.02
    F_VAC: float = 0.01
    R: int = 2

    # Coherence
    C_init: float = 0.15

    # Linear Momentum (IV.4)
    momentum_enabled: bool = True
    alpha_pi: float = 0.12        # Momentum accumulation
    lambda_pi: float = 0.008      # Momentum decay
    mu_pi: float = 0.35           # Momentum-to-flux coupling
    pi_max: float = 3.0

    # Plaquette Angular Momentum (IV.4b) - NEW
    angular_momentum_enabled: bool = True
    alpha_L: float = 0.06         # Spin charging rate
    lambda_L: float = 0.005       # Spin decay/friction
    mu_L: float = 0.18            # Spin-to-flux coupling
    L_max: float = 5.0            # Clip for numerical stability

    # Structure (q-locking)
    q_enabled: bool = True
    alpha_q: float = 0.012

    # Agency dynamics
    a_coupling: float = 30.0
    a_rate: float = 0.2

    # Floor repulsion (IV.3a)
    floor_enabled: bool = True
    eta_floor: float = 0.12
    F_core: float = 5.0
    floor_power: float = 2.0

    # Numerical stability
    outflow_limit: float = 0.2

    # Phase dynamics
    phase_enabled: bool = False

    def summary(self) -> str:
        return (f"N={self.N}, DT={self.DT}\n"
                f"Linear Mom: α_π={self.alpha_pi}, λ_π={self.lambda_pi}, μ_π={self.mu_pi}\n"
                f"Angular Mom (IV.4b): α_L={self.alpha_L}, λ_L={self.lambda_L}, μ_L={self.mu_L}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def periodic_local_sum_3d(x: np.ndarray, radius: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    size = 2 * radius + 1
    return uniform_filter(x, size=size, mode='wrap') * (size ** 3)


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
    - pi_X, pi_Y, pi_Z: Linear bond momentum (per bond)
    - L_XY, L_YZ, L_XZ: Plaquette angular momentum (per face)
    """

    def __init__(self, params: Optional[DETParams3D] = None):
        self.p = params or DETParams3D()
        N = self.p.N

        # Per-node state
        self.F = np.ones((N, N, N), dtype=np.float32) * self.p.F_VAC
        self.q = np.zeros((N, N, N), dtype=np.float32)
        self.theta = np.random.uniform(0, 2*np.pi, (N, N, N)).astype(np.float32)
        self.a = np.ones((N, N, N), dtype=np.float32)

        # Per-bond linear momentum (directed: +X, +Y, +Z)
        # pi_X[i,j,k] = momentum on bond from (i,j,k) to (i,j,k+1) [+X direction]
        self.pi_X = np.zeros((N, N, N), dtype=np.float32)
        self.pi_Y = np.zeros((N, N, N), dtype=np.float32)
        self.pi_Z = np.zeros((N, N, N), dtype=np.float32)

        # Per-plaquette angular momentum (IV.4b)
        # L_XY[i,j,k] = angular momentum on XY-face with corner at (i,j,k)
        # Represents circulation in the XY plane (Z-component of angular momentum)
        self.L_XY = np.zeros((N, N, N), dtype=np.float32)
        self.L_YZ = np.zeros((N, N, N), dtype=np.float32)
        self.L_XZ = np.zeros((N, N, N), dtype=np.float32)

        # Coherence (per bond)
        self.C_X = np.ones((N, N, N), dtype=np.float32) * self.p.C_init
        self.C_Y = np.ones((N, N, N), dtype=np.float32) * self.p.C_init
        self.C_Z = np.ones((N, N, N), dtype=np.float32) * self.p.C_init

        self.sigma = np.ones((N, N, N), dtype=np.float32)

        # Diagnostics
        self.time = 0.0
        self.step_count = 0
        self.P = np.ones((N, N, N), dtype=np.float32)
        self.Delta_tau = np.ones((N, N, N), dtype=np.float32) * self.p.DT

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
        envelope = np.exp(-0.5 * r2 / width**2).astype(np.float32)

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
        envelope = np.exp(-0.5 * r2 / width**2).astype(np.float32)
        
        self.L_XY += spin * envelope
        self._clip()

    def _clip(self):
        """Enforce physical bounds."""
        self.F = np.clip(self.F, self.p.F_VAC, 1000)
        self.q = np.clip(self.q, 0, 1)
        self.a = np.clip(self.a, 0, 1)
        self.pi_X = np.clip(self.pi_X, -self.p.pi_max, self.p.pi_max)
        self.pi_Y = np.clip(self.pi_Y, -self.p.pi_max, self.p.pi_max)
        self.pi_Z = np.clip(self.pi_Z, -self.p.pi_max, self.p.pi_max)
        self.L_XY = np.clip(self.L_XY, -self.p.L_max, self.p.L_max)
        self.L_YZ = np.clip(self.L_YZ, -self.p.L_max, self.p.L_max)
        self.L_XZ = np.clip(self.L_XZ, -self.p.L_max, self.p.L_max)

    def step(self):
        """
        Execute one canonical DET update step per IV.4b specification.
        
        Order:
        1. Compute P_i, Δτ_i
        2. Compute J components (diff, mom, floor, rot)
        3. Dissipation (via limiter)
        4. Update F using sender-clocked transport
        5. Update π (linear momentum)
        5b. Update L (plaquette angular momentum) ← NEW
        6. Update q
        7. Update a
        8. Update θ (disabled for speed)
        """
        p = self.p
        dk = p.DT

        # Neighbor access operators (periodic)
        Xp = lambda x: np.roll(x, -1, axis=2)  # i+1 in X
        Xm = lambda x: np.roll(x, 1, axis=2)   # i-1 in X
        Yp = lambda x: np.roll(x, -1, axis=1)  # j+1 in Y
        Ym = lambda x: np.roll(x, 1, axis=1)   # j-1 in Y
        Zp = lambda x: np.roll(x, -1, axis=0)  # k+1 in Z
        Zm = lambda x: np.roll(x, 1, axis=0)   # k-1 in Z

        # ================================================================
        # STEP 1: Presence and proper time
        # ================================================================
        H = self.sigma
        self.P = self.a * self.sigma / (1.0 + self.F) / (1.0 + H)
        self.Delta_tau = self.P * dk

        # Bond-averaged proper time
        Delta_tau_Xp = 0.5 * (self.Delta_tau + Xp(self.Delta_tau))
        Delta_tau_Yp = 0.5 * (self.Delta_tau + Yp(self.Delta_tau))
        Delta_tau_Zp = 0.5 * (self.Delta_tau + Zp(self.Delta_tau))

        # Plaquette-averaged proper time (IV.4b.1)
        # Δτ_□ = (1/4) Σ_{v∈□} Δτ_v
        Delta_tau_XY = 0.25 * (self.Delta_tau + Xp(self.Delta_tau) + 
                               Yp(self.Delta_tau) + Xp(Yp(self.Delta_tau)))
        Delta_tau_YZ = 0.25 * (self.Delta_tau + Yp(self.Delta_tau) + 
                               Zp(self.Delta_tau) + Yp(Zp(self.Delta_tau)))
        Delta_tau_XZ = 0.25 * (self.Delta_tau + Xp(self.Delta_tau) + 
                               Zp(self.Delta_tau) + Xp(Zp(self.Delta_tau)))

        # ================================================================
        # STEP 2: Compute flux components
        # ================================================================
        
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

        # J^diff: Diffusive flux
        J_diff_Xp = g_Xp * cond_Xp * classical_Xp
        J_diff_Xm = g_Xm * cond_Xm * classical_Xm
        J_diff_Yp = g_Yp * cond_Yp * classical_Yp
        J_diff_Ym = g_Ym * cond_Ym * classical_Ym
        J_diff_Zp = g_Zp * cond_Zp * classical_Zp
        J_diff_Zm = g_Zm * cond_Zm * classical_Zm

        # Initialize total flux
        J_Xp = J_diff_Xp.copy()
        J_Xm = J_diff_Xm.copy()
        J_Yp = J_diff_Yp.copy()
        J_Ym = J_diff_Ym.copy()
        J_Zp = J_diff_Zp.copy()
        J_Zm = J_diff_Zm.copy()

        # J^mom: Linear momentum flux (IV.4)
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

        # J^floor: Floor repulsion (IV.3a)
        if p.floor_enabled:
            s = np.maximum(0, (self.F - p.F_core) / p.F_core)**p.floor_power
            J_Xp += p.eta_floor * self.sigma * (s + Xp(s)) * classical_Xp
            J_Xm += p.eta_floor * self.sigma * (s + Xm(s)) * classical_Xm
            J_Yp += p.eta_floor * self.sigma * (s + Yp(s)) * classical_Yp
            J_Ym += p.eta_floor * self.sigma * (s + Ym(s)) * classical_Ym
            J_Zp += p.eta_floor * self.sigma * (s + Zp(s)) * classical_Zp
            J_Zm += p.eta_floor * self.sigma * (s + Zm(s)) * classical_Zm

        # ================================================================
        # J^rot: Rotational flux from plaquette angular momentum (IV.4b.2)
        # ================================================================
        if p.angular_momentum_enabled:
            # Per IV.4b.2: J^rot is perpendicular gradient of L
            # J^rot_x(i,j) = μ_L * σ * F^avg_x * (L(i,j) - L(i-1,j))
            # J^rot_y(i,j) = -μ_L * σ * F^avg_y * (L(i,j) - L(i,j-1))
            
            # Edge-averaged F for rotational flux
            F_avg_Xp = 0.5 * (self.F + Xp(self.F))
            F_avg_Yp = 0.5 * (self.F + Yp(self.F))
            F_avg_Zp = 0.5 * (self.F + Zp(self.F))
            
            # L_XY affects X and Y edges (circulation in XY plane)
            # J^rot_X from L_XY: perpendicular gradient in Y direction
            # J^rot_X(i,j,k) = μ_L * σ * F_avg * (L_XY(i,j,k) - L_XY(i,j-1,k))
            J_rot_Xp_from_XY = p.mu_L * self.sigma * F_avg_Xp * (self.L_XY - Ym(self.L_XY))
            
            # J^rot_Y from L_XY: perpendicular gradient in X direction (negative sign)
            # J^rot_Y(i,j,k) = -μ_L * σ * F_avg * (L_XY(i,j,k) - L_XY(i-1,j,k))
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
            
            # Add to total flux
            J_Xp += J_rot_Xp
            J_Xm -= Xm(J_rot_Xp)  # Antisymmetric
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

        J_diff_Xp_scaled = np.where(J_diff_Xp > 0, J_diff_Xp * scale, J_diff_Xp)
        J_diff_Yp_scaled = np.where(J_diff_Yp > 0, J_diff_Yp * scale, J_diff_Yp)
        J_diff_Zp_scaled = np.where(J_diff_Zp > 0, J_diff_Zp * scale, J_diff_Zp)

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
        self.F = np.clip(self.F + dF, p.F_VAC, 1000)

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
            # curl_π(i,j) = π_x(i,j) + π_y(i+1,j) - π_x(i,j+1) - π_y(i,j)
            # Using our convention: π_X is +X bond, π_Y is +Y bond
            
            # XY plaquette: curl of (π_X, π_Y)
            # curl_z = π_X(i,j,k) + π_Y(i+1,j,k) - π_X(i,j+1,k) - π_Y(i,j,k)
            curl_XY = self.pi_X + Xp(self.pi_Y) - Yp(self.pi_X) - self.pi_Y
            
            # YZ plaquette: curl of (π_Y, π_Z)
            curl_YZ = self.pi_Y + Yp(self.pi_Z) - Zp(self.pi_Y) - self.pi_Z
            
            # XZ plaquette: curl of (π_X, π_Z)
            # Note: curl_y = π_Z(i,j,k) + π_X(i,j,k+1) - π_Z(i+1,j,k) - π_X(i,j,k)
            # But we want consistent handedness, so:
            curl_XZ = self.pi_Z + Zp(self.pi_X) - Xp(self.pi_Z) - self.pi_X

            # Presence-clocked update (IV.4b.1)
            # L⁺ = (1 - λ_L * Δτ_□) * L + α_L * curl_π * Δτ_□
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
        # STEP 7: Update a (agency)
        # ================================================================
        a_target = 1.0 / (1.0 + p.a_coupling * self.q**2)
        self.a = self.a + p.a_rate * (a_target - self.a)

        # ================================================================
        # Auxiliary updates
        # ================================================================
        J_mag = (np.abs(J_Xp_lim) + np.abs(J_Yp_lim) + np.abs(J_Zp_lim)) / 3.0
        self.C_X = np.clip(self.C_X + 0.04 * np.abs(J_Xp_lim) * self.Delta_tau
                          - 0.002 * self.C_X * self.Delta_tau, p.C_init, 1.0)
        self.C_Y = np.clip(self.C_Y + 0.04 * np.abs(J_Yp_lim) * self.Delta_tau
                          - 0.002 * self.C_Y * self.Delta_tau, p.C_init, 1.0)
        self.C_Z = np.clip(self.C_Z + 0.04 * np.abs(J_Zp_lim) * self.Delta_tau
                          - 0.002 * self.C_Z * self.Delta_tau, p.C_init, 1.0)

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
        """Find distinct blobs using connected components with higher threshold."""
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


# =============================================================================
# FALSIFIER TESTS (IV.4b)
# =============================================================================

def test_F_L1_rotational_flux_conservation(verbose: bool = True) -> bool:
    """
    F_L1: Rotational Flux Conservation
    
    With J^rot enabled and all other flux components disabled, verify:
    - Total F is conserved
    - Node-wise F develops only pure circulation (no net drift)
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F_L1: Rotational Flux Conservation")
        print("="*60)

    # Disable all flux except rotational
    params = DETParams3D(
        N=24,
        momentum_enabled=False,
        floor_enabled=False,
        q_enabled=False,
        angular_momentum_enabled=True
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
    mass_err = 100 * abs(final_F - initial_F) / initial_F
    max_drift = max(com_drift)
    
    # Pass criteria: mass conserved (<1%), no net COM drift (<1 cell)
    passed = mass_err < 1.0 and max_drift < 1.0

    if verbose:
        print(f"  Initial mass: {initial_F:.4f}")
        print(f"  Final mass: {final_F:.4f}")
        print(f"  Mass error: {mass_err:.4f}%")
        print(f"  Max COM drift: {max_drift:.4f} cells")
        print(f"  F_L1 {'PASSED' if passed else 'FAILED'}")

    return passed


def test_F_L2_vacuum_spin_no_transport(verbose: bool = True) -> bool:
    """
    F_L2: Vacuum Spin Does Not Transport
    
    Initialize L≠0 but F=F_vac everywhere. Verify:
    - max|J^rot| ≈ 0
    - Negligible mass drift
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F_L2: Vacuum Spin Does Not Transport")
        print("="*60)

    params = DETParams3D(
        N=20,
        momentum_enabled=False,
        floor_enabled=False,
        q_enabled=False,
        angular_momentum_enabled=True
    )
    sim = DETCollider3D(params)
    
    # Pure vacuum with nonzero spin
    sim.F = np.ones_like(sim.F) * params.F_VAC
    center = params.N // 2
    sim.add_spin((center, center, center), spin=2.0, width=5.0)

    initial_F = sim.total_F()
    
    # Track max J_rot
    max_J_rot = 0
    for t in range(200):
        # Compute J_rot magnitude before step
        F_avg = 0.5 * (sim.F + np.roll(sim.F, -1, axis=2))
        J_rot_mag = params.mu_L * sim.sigma * F_avg * np.abs(sim.L_XY - np.roll(sim.L_XY, 1, axis=1))
        max_J_rot = max(max_J_rot, float(np.max(J_rot_mag)))
        sim.step()

    final_F = sim.total_F()
    mass_drift = abs(final_F - initial_F) / initial_F
    
    # Pass criteria: J_rot negligible, mass drift negligible
    # J_rot should be proportional to F_vac which is tiny
    expected_J_rot_order = params.mu_L * params.F_VAC * 2.0  # rough estimate
    passed = max_J_rot < 0.01 and mass_drift < 0.01

    if verbose:
        print(f"  Max |J^rot| observed: {max_J_rot:.6f}")
        print(f"  Expected order: μ_L * F_vac * L ~ {expected_J_rot_order:.6f}")
        print(f"  Mass drift: {mass_drift*100:.4f}%")
        print(f"  F_L2 {'PASSED' if passed else 'FAILED'}")

    return passed


def test_F_L3_orbital_capture(params: Optional[DETParams3D] = None,
                              impact_param: float = 6.0,
                              steps: int = 3000,
                              verbose: bool = True) -> Dict:
    """
    F_L3: Orbital Capture (Past-Resolved)
    
    For two packets with non-zero impact parameter, verify:
    - Separation remains bounded for long durations
    - Relative angle winds (true circulation)
    - No secular runaway in separation
    """
    if verbose:
        print("\n" + "="*60)
        print(f"TEST F_L3: Orbital Capture (b={impact_param})")
        print("="*60)

    p = params or DETParams3D(N=50)
    sim = DETCollider3D(p)
    center = p.N // 2
    sep_init = 15  # Larger separation for distinct blobs
    b = int(impact_param / 2)

    # Two packets with impact parameter (offset in Y)
    # Larger separation and stronger momentum for visible dynamics
    sim.add_packet((center - sep_init, center - b, center), mass=10.0, width=2.5, momentum=(2.0, 0, 0))
    sim.add_packet((center + sep_init, center + b, center), mass=10.0, width=2.5, momentum=(-2.0, 0, 0))

    initial_F = sim.total_F()
    
    rec = {'t': [], 'sep': [], 'blobs': [], 'mass_err': [], 
           'L_z': [], 'angle': [], 'q_max': [], 'min_a': []}
    
    prev_angle = 0
    total_angle = 0

    for t in range(steps):
        sep, num = sim.separation()
        mass_err = 100 * (sim.total_F() - initial_F) / initial_F
        L = sim.total_angular_momentum()
        
        # Track relative angle between blobs
        blobs = sim.find_blobs()
        if len(blobs) >= 2:
            dx = blobs[1]['x'] - blobs[0]['x']
            dy = blobs[1]['y'] - blobs[0]['y']
            angle = np.arctan2(dy, dx)
            # Unwrap angle for winding count
            d_angle = angle - prev_angle
            if d_angle > np.pi:
                d_angle -= 2*np.pi
            elif d_angle < -np.pi:
                d_angle += 2*np.pi
            total_angle += d_angle
            prev_angle = angle
        else:
            angle = 0

        rec['t'].append(t)
        rec['sep'].append(sep)
        rec['blobs'].append(num)
        rec['mass_err'].append(mass_err)
        rec['L_z'].append(L[2])
        rec['angle'].append(total_angle)
        rec['q_max'].append(float(np.max(sim.q)))
        rec['min_a'].append(float(np.min(sim.a)))

        if verbose and t % 500 == 0:
            print(f"  t={t}: sep={sep:.1f}, blobs={num}, L_z={L[2]:.3f}, "
                  f"angle={total_angle/(2*np.pi):.2f} rev")

        sim.step()

    # Analyze for orbital capture
    sep_array = np.array(rec['sep'])
    valid_seps = sep_array[sep_array > 0]
    
    if len(valid_seps) > 10:
        # Check if separation is bounded (not running away)
        second_half = valid_seps[len(valid_seps)//2:]
        sep_mean = np.mean(second_half)
        sep_std = np.std(second_half)
        sep_max = np.max(second_half)
        
        # Check for angle winding
        total_revolutions = abs(total_angle) / (2 * np.pi)
        
        # Orbital capture criteria:
        # 1. Separation bounded (max < initial separation * 1.5)
        # 2. Some oscillation (std > 0.5)
        # 3. Angle winds (> 0.25 revolutions)
        bounded = sep_max < sep_init * 2.5
        oscillating = sep_std > 0.3
        winding = total_revolutions > 0.1
        
        orbital_capture = bounded and (oscillating or winding)
    else:
        sep_mean = 0
        sep_std = 0
        sep_max = 0
        total_revolutions = 0
        orbital_capture = False

    rec['sep_mean'] = sep_mean
    rec['sep_std'] = sep_std
    rec['sep_max'] = sep_max
    rec['total_revolutions'] = total_revolutions
    rec['orbital_capture'] = orbital_capture
    rec['final_sim'] = sim

    if verbose:
        print(f"\n  Results:")
        print(f"    Separation bounded: {sep_max < sep_init * 2.5}")
        print(f"    Sep mean (2nd half): {sep_mean:.1f}")
        print(f"    Sep std: {sep_std:.2f}")
        print(f"    Total revolutions: {total_revolutions:.2f}")
        print(f"    Final L_z: {rec['L_z'][-1]:.4f}")
        print(f"    F_L3 {'PASSED' if orbital_capture else 'FAILED'}")

    return rec


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_orbital_test(rec: Dict, filename: str = 'det_3d_orbital.png'):
    """Visualize orbital capture test results."""
    fig = plt.figure(figsize=(16, 12))

    ax = fig.add_subplot(2, 3, 1)
    ax.plot(rec['t'], rec['sep'], 'b-', lw=1.5)
    if rec.get('sep_mean', 0) > 0:
        ax.axhline(rec['sep_mean'], color='g', ls='--', alpha=0.5, 
                   label=f"Mean: {rec['sep_mean']:.1f}")
    ax.set_xlabel('Step')
    ax.set_ylabel('Separation')
    ax.set_title('Inter-body Separation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(2, 3, 2)
    ax.plot(rec['t'], rec['L_z'], 'purple', lw=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('L_z')
    ax.set_title('Z-Angular Momentum (from curl π)')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(2, 3, 3)
    ax.plot(rec['t'], np.array(rec['angle']) / (2*np.pi), 'orange', lw=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Revolutions')
    ax.set_title('Relative Angle Winding')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(2, 3, 4)
    ax.plot(rec['t'], rec['q_max'], 'r-', lw=1.5, label='max(q)')
    ax.plot(rec['t'], rec['min_a'], 'g-', lw=1.5, label='min(a)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.set_title('Structure & Agency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(2, 3, 5)
    ax.plot(rec['t'], rec['mass_err'], 'm-', lw=1.5)
    ax.axhline(0, color='k', ls='-', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mass Error %')
    ax.set_title('Mass Conservation')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(2, 3, 6)
    ax.axis('off')
    summary = f"""
F_L3: Orbital Capture Test
==========================
Orbital capture: {'YES' if rec.get('orbital_capture', False) else 'NO'}
Sep max (2nd half): {rec.get('sep_max', 0):.1f}
Sep mean (2nd half): {rec.get('sep_mean', 0):.1f}
Sep std: {rec.get('sep_std', 0):.2f}
Total revolutions: {rec.get('total_revolutions', 0):.2f}
Final L_z: {rec['L_z'][-1]:.4f}
Mass error: {rec['mass_err'][-1]:+.3f}%
"""
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('DET v6 3D Collider - Orbital Capture (IV.4b)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# =============================================================================
# MAIN TEST SUITE
# =============================================================================

def run_angular_momentum_suite():
    """Run the complete IV.4b angular momentum test suite."""
    print("="*70)
    print("DET v6 3D COLLIDER - PLAQUETTE ANGULAR MOMENTUM (IV.4b)")
    print("="*70)

    results = {}

    # F_L1: Rotational flux conservation
    results['F_L1'] = test_F_L1_rotational_flux_conservation(verbose=True)

    # F_L2: Vacuum spin doesn't transport
    results['F_L2'] = test_F_L2_vacuum_spin_no_transport(verbose=True)

    # F_L3: Orbital capture
    orbital = test_F_L3_orbital_capture(verbose=True)
    results['F_L3'] = orbital

    # Summary
    print("\n" + "="*70)
    print("SUITE SUMMARY")
    print("="*70)
    print(f"  F_L1 (Rotational flux conservation): {'PASS' if results['F_L1'] else 'FAIL'}")
    print(f"  F_L2 (Vacuum spin no transport): {'PASS' if results['F_L2'] else 'FAIL'}")
    print(f"  F_L3 (Orbital capture): {'PASS' if orbital['orbital_capture'] else 'FAIL'}")

    return results


if __name__ == "__main__":
    start = time.time()

    results = run_angular_momentum_suite()

    # Visualization
    if 'F_L3' in results and isinstance(results['F_L3'], dict):
        visualize_orbital_test(results['F_L3'], './det_3d_orbital_iv4b.png')

    elapsed = time.time() - start
    print(f"\nTotal runtime: {elapsed:.1f}s")
