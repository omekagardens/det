"""
DET v6.2 3D Collider with Boundary Operators
============================================

This implementation adds the DET boundary operator module (Section VI) to the 3D collider:
- Grace injection (VI.5): Agency-gated resource injection to needy nodes
- Bond healing (optional): Agency-gated coherence recovery

Key Features:
- Strictly local operations (3D neighborhood)
- Agency inviolability (VI.1): Boundary operators never modify a_i directly
- F2 Falsifier: a=0 blocks all boundary action
- F3 Falsifier: Boundary on/off produces qualitatively different outcomes

Reference: DET Theory Card v6.2, Section VI
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from scipy.fft import fftn, ifftn


@dataclass
class DETParams3D:
    """DET 3D simulation parameters with boundary operators."""
    N: int = 24                     # Grid size (N x N x N)
    DT: float = 0.02                # Global step size (dk)
    F_VAC: float = 0.01             # Vacuum resource level (initial)
    F_MIN: float = 0.0              # True minimum for F (0 for conservation)
    
    # Coherence
    C_init: float = 0.3             # Initial bond coherence
    
    # Momentum module (IV.4)
    momentum_enabled: bool = True
    alpha_pi: float = 0.10
    lambda_pi: float = 0.02
    mu_pi: float = 0.30
    pi_max: float = 3.0
    
    # Structure (q-locking)
    q_enabled: bool = True
    alpha_q: float = 0.015
    
    # Agency dynamics (VI.2B)
    a_coupling: float = 30.0
    a_rate: float = 0.2
    
    # Floor repulsion (IV.6)
    floor_enabled: bool = True
    eta_floor: float = 0.15
    F_core: float = 5.0
    floor_power: float = 2.0
    
    # ========== BOUNDARY OPERATORS (VI) ==========
    boundary_enabled: bool = True    # Master toggle for all boundary ops
    
    # Grace injection (VI.5)
    grace_enabled: bool = True
    F_MIN_grace: float = 0.05       # Threshold for "need"
    
    # Bond healing (optional extension)
    healing_enabled: bool = False   # Default off until tested
    eta_heal: float = 0.03          # Healing rate
    
    # Local neighborhood radius for boundary ops
    R_boundary: int = 2             # Radius for local normalization (smaller for 3D)
    
    # Numerical stability
    outflow_limit: float = 0.25


def periodic_local_sum_3d(x: np.ndarray, radius: int) -> np.ndarray:
    """Compute local sum within radius (periodic BCs) in 3D."""
    result = np.zeros_like(x)
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            for dk in range(-radius, radius + 1):
                result += np.roll(np.roll(np.roll(x, di, axis=0), dj, axis=1), dk, axis=2)
    return result


class DETCollider3DBoundary:
    """
    DET v6.2 3D Collider with Boundary Operators
    
    Implements:
    - Agency-gated diffusion (IV.2)
    - Presence-clocked transport (III.1)
    - Momentum dynamics (IV.4)
    - Floor repulsion (IV.6)
    - **Grace injection (VI.5)** - NEW
    - **Bond healing (optional)** - NEW
    - Target-tracking agency update (VI.2B)
    
    Boundary Operator Discipline:
    - All boundary ops are local
    - All boundary ops are agency-gated (a=0 → no action)
    - Boundary ops never modify a directly (VI.1 inviolability)
    """
    
    def __init__(self, params: Optional[DETParams3D] = None):
        self.p = params or DETParams3D()
        N = self.p.N
        shape = (N, N, N)
        
        # Per-node state (II.1) - 3D arrays
        self.F = np.ones(shape) * self.p.F_VAC
        self.q = np.zeros(shape)
        self.a = np.ones(shape)
        
        # Per-bond state (II.2) - 6 directions: +X, -X, +Y, -Y, +Z, -Z
        # We store 3 positive directions; negative are accessed via neighbor rolls
        self.pi_X = np.zeros(shape)  # +X directed momentum
        self.pi_Y = np.zeros(shape)  # +Y directed momentum
        self.pi_Z = np.zeros(shape)  # +Z directed momentum
        self.C_X = np.ones(shape) * self.p.C_init  # +X bond coherence
        self.C_Y = np.ones(shape) * self.p.C_init  # +Y bond coherence
        self.C_Z = np.ones(shape) * self.p.C_init  # +Z bond coherence
        self.sigma = np.ones(shape)  # Conductivity (isotropic)
        
        # Diagnostics
        self.time = 0.0
        self.step_count = 0
        self.P = np.ones(shape)
        self.Delta_tau = np.ones(shape) * self.p.DT
        
        # Boundary operator diagnostics
        self.last_grace_injection = np.zeros(shape)
        self.total_grace_injected = 0.0
    
    def add_packet(self, center: Tuple[int, int, int], mass: float = 5.0, 
                   width: float = 3.0, momentum: Tuple[float, float, float] = (0, 0, 0)):
        """Add a Gaussian resource packet with optional initial momentum."""
        N = self.p.N
        ci, cj, ck = center
        i_arr, j_arr, k_arr = np.ogrid[:N, :N, :N]
        
        # Distance from center (with periodic wrapping)
        di = np.minimum(np.abs(i_arr - ci), N - np.abs(i_arr - ci))
        dj = np.minimum(np.abs(j_arr - cj), N - np.abs(j_arr - cj))
        dk = np.minimum(np.abs(k_arr - ck), N - np.abs(k_arr - ck))
        r2 = di**2 + dj**2 + dk**2
        
        envelope = np.exp(-0.5 * r2 / width**2)
        
        self.F += mass * envelope
        self.C_X = np.clip(self.C_X + 0.5 * envelope, self.p.C_init, 1.0)
        self.C_Y = np.clip(self.C_Y + 0.5 * envelope, self.p.C_init, 1.0)
        self.C_Z = np.clip(self.C_Z + 0.5 * envelope, self.p.C_init, 1.0)
        
        # Initial momentum
        mom_x, mom_y, mom_z = momentum
        if mom_x != 0 or mom_y != 0 or mom_z != 0:
            mom_env = np.exp(-0.5 * r2 / (width * 2)**2)
            self.pi_X += mom_x * mom_env
            self.pi_Y += mom_y * mom_env
            self.pi_Z += mom_z * mom_env
        
        self._clip()
    
    def _clip(self):
        """Enforce physical bounds on state variables."""
        self.F = np.clip(self.F, self.p.F_MIN, 1000)
        self.q = np.clip(self.q, 0, 1)
        self.a = np.clip(self.a, 0, 1)
        self.pi_X = np.clip(self.pi_X, -self.p.pi_max, self.p.pi_max)
        self.pi_Y = np.clip(self.pi_Y, -self.p.pi_max, self.p.pi_max)
        self.pi_Z = np.clip(self.pi_Z, -self.p.pi_max, self.p.pi_max)
    
    def compute_grace_injection(self, D: np.ndarray) -> np.ndarray:
        """
        Grace Injection per DET VI.5 (3D version)
        
        CRITICAL: If a_i = 0, then w_i = 0, so I_{g→i} = 0
        This is the F2 (Coercion) guarantee.
        """
        p = self.p
        
        # Need: how far below threshold
        n = np.maximum(0, p.F_MIN_grace - self.F)
        
        # Weight: AGENCY-GATED need
        w = self.a * n
        
        # Local normalization (within 3D neighborhood)
        w_sum = periodic_local_sum_3d(w, p.R_boundary) + 1e-12
        
        # Injection amount
        I_g = D * w / w_sum
        
        return I_g
    
    def step(self):
        """
        Execute one canonical DET update step with boundary operators (3D).
        """
        p = self.p
        N = p.N
        dk = p.DT
        
        # Neighbor access operators (periodic BCs)
        Xp = lambda x: np.roll(x, -1, axis=0)  # +X neighbor
        Xm = lambda x: np.roll(x, 1, axis=0)   # -X neighbor
        Yp = lambda x: np.roll(x, -1, axis=1)  # +Y neighbor
        Ym = lambda x: np.roll(x, 1, axis=1)   # -Y neighbor
        Zp = lambda x: np.roll(x, -1, axis=2)  # +Z neighbor
        Zm = lambda x: np.roll(x, 1, axis=2)   # -Z neighbor
        
        # ============================================================
        # STEP 1: Presence and proper time (III.1)
        # ============================================================
        H = self.sigma
        self.P = self.a * self.sigma / (1.0 + self.F) / (1.0 + H)
        self.Delta_tau = self.P * dk
        
        Delta_tau_X = 0.5 * (self.Delta_tau + Xp(self.Delta_tau))
        Delta_tau_Y = 0.5 * (self.Delta_tau + Yp(self.Delta_tau))
        Delta_tau_Z = 0.5 * (self.Delta_tau + Zp(self.Delta_tau))
        
        # ============================================================
        # STEP 2: Flow computation (6 directions)
        # ============================================================
        
        # Classical (pressure) contributions
        classical_Xp = self.F - Xp(self.F)
        classical_Xm = self.F - Xm(self.F)
        classical_Yp = self.F - Yp(self.F)
        classical_Ym = self.F - Ym(self.F)
        classical_Zp = self.F - Zp(self.F)
        classical_Zm = self.F - Zm(self.F)
        
        # Coherence interpolation
        sqrt_C_Xp = np.sqrt(self.C_X)
        sqrt_C_Xm = np.sqrt(Xm(self.C_X))
        sqrt_C_Yp = np.sqrt(self.C_Y)
        sqrt_C_Ym = np.sqrt(Ym(self.C_Y))
        sqrt_C_Zp = np.sqrt(self.C_Z)
        sqrt_C_Zm = np.sqrt(Zm(self.C_Z))
        
        # Combined drive
        drive_Xp = (1 - sqrt_C_Xp) * classical_Xp
        drive_Xm = (1 - sqrt_C_Xm) * classical_Xm
        drive_Yp = (1 - sqrt_C_Yp) * classical_Yp
        drive_Ym = (1 - sqrt_C_Ym) * classical_Ym
        drive_Zp = (1 - sqrt_C_Zp) * classical_Zp
        drive_Zm = (1 - sqrt_C_Zm) * classical_Zm
        
        # Agency-gated diffusion (IV.2)
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
        
        # Agency-gated diffusive flux
        J_diff_Xp = g_Xp * cond_Xp * drive_Xp
        J_diff_Xm = g_Xm * cond_Xm * drive_Xm
        J_diff_Yp = g_Yp * cond_Yp * drive_Yp
        J_diff_Ym = g_Ym * cond_Ym * drive_Ym
        J_diff_Zp = g_Zp * cond_Zp * drive_Zp
        J_diff_Zm = g_Zm * cond_Zm * drive_Zm
        
        # Momentum-driven flux (IV.4)
        if p.momentum_enabled:
            F_avg_Xp = 0.5 * (self.F + Xp(self.F))
            F_avg_Xm = 0.5 * (self.F + Xm(self.F))
            F_avg_Yp = 0.5 * (self.F + Yp(self.F))
            F_avg_Ym = 0.5 * (self.F + Ym(self.F))
            F_avg_Zp = 0.5 * (self.F + Zp(self.F))
            F_avg_Zm = 0.5 * (self.F + Zm(self.F))
            
            J_mom_Xp = p.mu_pi * self.sigma * self.pi_X * F_avg_Xp
            J_mom_Xm = -p.mu_pi * self.sigma * Xm(self.pi_X) * F_avg_Xm
            J_mom_Yp = p.mu_pi * self.sigma * self.pi_Y * F_avg_Yp
            J_mom_Ym = -p.mu_pi * self.sigma * Ym(self.pi_Y) * F_avg_Ym
            J_mom_Zp = p.mu_pi * self.sigma * self.pi_Z * F_avg_Zp
            J_mom_Zm = -p.mu_pi * self.sigma * Zm(self.pi_Z) * F_avg_Zm
        else:
            J_mom_Xp = J_mom_Xm = J_mom_Yp = J_mom_Ym = J_mom_Zp = J_mom_Zm = 0
        
        # Floor repulsion flux (IV.6)
        if p.floor_enabled:
            s = np.maximum(0, (self.F - p.F_core) / p.F_core)**p.floor_power
            J_floor_Xp = p.eta_floor * self.sigma * (s + Xp(s)) * classical_Xp
            J_floor_Xm = p.eta_floor * self.sigma * (s + Xm(s)) * classical_Xm
            J_floor_Yp = p.eta_floor * self.sigma * (s + Yp(s)) * classical_Yp
            J_floor_Ym = p.eta_floor * self.sigma * (s + Ym(s)) * classical_Ym
            J_floor_Zp = p.eta_floor * self.sigma * (s + Zp(s)) * classical_Zp
            J_floor_Zm = p.eta_floor * self.sigma * (s + Zm(s)) * classical_Zm
        else:
            J_floor_Xp = J_floor_Xm = J_floor_Yp = J_floor_Ym = J_floor_Zp = J_floor_Zm = 0
        
        # Total flux per direction
        J_Xp = J_diff_Xp + J_mom_Xp + J_floor_Xp
        J_Xm = J_diff_Xm + J_mom_Xm + J_floor_Xm
        J_Yp = J_diff_Yp + J_mom_Yp + J_floor_Yp
        J_Ym = J_diff_Ym + J_mom_Ym + J_floor_Ym
        J_Zp = J_diff_Zp + J_mom_Zp + J_floor_Zp
        J_Zm = J_diff_Zm + J_mom_Zm + J_floor_Zm
        
        # ============================================================
        # STEP 3: Dissipation and limiter
        # ============================================================
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
        
        # Dissipation (for grace injection)
        D = (np.abs(J_Xp_lim) + np.abs(J_Xm_lim) + 
             np.abs(J_Yp_lim) + np.abs(J_Ym_lim) +
             np.abs(J_Zp_lim) + np.abs(J_Zm_lim)) * self.Delta_tau
        
        # ============================================================
        # STEP 4: Resource update (IV.7)
        # ============================================================
        transfer_Xp = J_Xp_lim * self.Delta_tau
        transfer_Xm = J_Xm_lim * self.Delta_tau
        transfer_Yp = J_Yp_lim * self.Delta_tau
        transfer_Ym = J_Ym_lim * self.Delta_tau
        transfer_Zp = J_Zp_lim * self.Delta_tau
        transfer_Zm = J_Zm_lim * self.Delta_tau
        
        outflow = (transfer_Xp + transfer_Xm + transfer_Yp + 
                   transfer_Ym + transfer_Zp + transfer_Zm)
        inflow = (Xm(transfer_Xp) + Xp(transfer_Xm) + 
                  Ym(transfer_Yp) + Yp(transfer_Ym) +
                  Zm(transfer_Zp) + Zp(transfer_Zm))
        
        dF = inflow - outflow
        self.F = np.clip(self.F + dF, p.F_MIN, 1000)
        
        # ============================================================
        # STEP 5: Grace Injection (VI.5) - BOUNDARY OPERATOR
        # ============================================================
        if p.boundary_enabled and p.grace_enabled:
            I_g = self.compute_grace_injection(D)
            self.F = self.F + I_g
            
            self.last_grace_injection = I_g.copy()
            self.total_grace_injected += np.sum(I_g)
        else:
            self.last_grace_injection = np.zeros_like(self.F)
        
        # ============================================================
        # STEP 6: Momentum update (IV.4)
        # ============================================================
        if p.momentum_enabled:
            decay_X = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_X)
            decay_Y = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_Y)
            decay_Z = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_Z)
            self.pi_X = decay_X * self.pi_X + p.alpha_pi * J_diff_Xp_scaled * Delta_tau_X
            self.pi_Y = decay_Y * self.pi_Y + p.alpha_pi * J_diff_Yp_scaled * Delta_tau_Y
            self.pi_Z = decay_Z * self.pi_Z + p.alpha_pi * J_diff_Zp_scaled * Delta_tau_Z
        
        # ============================================================
        # STEP 7: Structural update (canonical q-locking)
        # ============================================================
        if p.q_enabled:
            self.q = np.clip(self.q + p.alpha_q * np.maximum(0, -dF), 0, 1)
        
        # ============================================================
        # STEP 8: Agency update (VI.2B)
        # ============================================================
        a_target = 1.0 / (1.0 + p.a_coupling * self.q**2)
        self.a = self.a + p.a_rate * (a_target - self.a)
        
        self._clip()
        self.time += dk
        self.step_count += 1
    
    def total_mass(self) -> float:
        return float(np.sum(self.F))
    
    def center_of_mass(self) -> Tuple[float, float, float]:
        N = self.p.N
        i_arr, j_arr, k_arr = np.ogrid[:N, :N, :N]
        total = np.sum(self.F) + 1e-9
        com_x = float(np.sum(i_arr * self.F) / total)
        com_y = float(np.sum(j_arr * self.F) / total)
        com_z = float(np.sum(k_arr * self.F) / total)
        return com_x, com_y, com_z


# ======================================================================
# F2/F3 FALSIFIER TESTS FOR 3D
# ======================================================================

def test_f2_grace_coercion_3d():
    """
    F2 Test A (3D): Hard-zero agency sentinel
    """
    print("=" * 70)
    print("F2 COERCION TEST A (3D): Hard-Zero Agency Sentinel (Grace)")
    print("=" * 70)
    
    params = DETParams3D(
        N=16, 
        boundary_enabled=True, 
        grace_enabled=True,
        F_MIN_grace=0.15,
        a_rate=0.0,
    )
    sim = DETCollider3DBoundary(params)
    
    # Create collision scenario
    sim.add_packet((8, 8, 4), mass=2.0, width=2.0, momentum=(0, 0, 0.4))
    sim.add_packet((8, 8, 12), mass=2.0, width=2.0, momentum=(0, 0, -0.4))
    
    # Set sentinel at center
    sentinel = (8, 8, 8)
    sim.a[sentinel] = 0.0
    sim.F[sentinel] = 0.01
    
    for _ in range(100):
        sim.step()
    
    sentinel_grace = sim.last_grace_injection[sentinel]
    
    print(f"\n  Sentinel (a=0) at {sentinel}")
    print(f"  Sentinel a (should be 0): {sim.a[sentinel]:.4f}")
    print(f"  Sentinel F = {sim.F[sentinel]:.4f} (needy)")
    print(f"  Grace received by sentinel: {sentinel_grace:.2e}")
    print(f"  Total grace injected: {sim.total_grace_injected:.4f}")
    
    passed = sentinel_grace == 0.0
    print(f"\n  F2 Grace Test (3D): {'PASSED ✓' if passed else 'FAILED ✗'}")
    if passed:
        print("  → Agency gate correctly blocks grace to a=0 node")
    return passed


def test_f3_scarcity_recovery_3d():
    """
    F3 Test D (3D): Scarcity collapse vs recovery
    """
    print("\n" + "=" * 70)
    print("F3 REDUNDANCY TEST D (3D): Scarcity Collapse vs Recovery")
    print("=" * 70)
    
    def run_scenario(boundary_on: bool):
        params = DETParams3D(
            N=16,
            F_VAC=0.02,
            boundary_enabled=boundary_on,
            grace_enabled=True,
            F_MIN_grace=0.15,
        )
        sim = DETCollider3DBoundary(params)
        
        # Head-on collision
        sim.add_packet((8, 8, 4), mass=1.5, width=2.0, momentum=(0, 0, 0.3))
        sim.add_packet((8, 8, 12), mass=1.5, width=2.0, momentum=(0, 0, -0.3))
        
        for _ in range(150):
            sim.step()
        
        # Measure collision zone
        zone = sim.F[6:10, 6:10, 6:10]
        return np.mean(zone), sim.total_grace_injected
    
    F_off, grace_off = run_scenario(False)
    F_on, grace_on = run_scenario(True)
    
    print(f"\n  Boundary OFF:")
    print(f"    Final <F> in collision zone: {F_off:.4f}")
    print(f"    Total grace received: {grace_off:.4f}")
    
    print(f"\n  Boundary ON:")
    print(f"    Final <F> in collision zone: {F_on:.4f}")
    print(f"    Total grace received: {grace_on:.4f}")
    
    grace_diff = grace_on - grace_off
    qualitative = grace_diff > 0.001
    
    print(f"\n  Boundary ON vs OFF comparison:")
    print(f"    Grace difference: {grace_diff:.4f}")
    print(f"    Qualitative difference: {qualitative}")
    
    passed = qualitative
    print(f"\n  F3 Scarcity Test (3D): {'PASSED ✓' if passed else 'FAILED ✗'}")
    if passed:
        print("  → Boundary ON produces measurably different outcome")
    return passed


def test_f3_locality_3d():
    """
    F3 Test E (3D): Local crisis, local response
    """
    print("\n" + "=" * 70)
    print("F3 LOCALITY TEST E (3D): Local Crisis, Local Response")
    print("=" * 70)
    
    params = DETParams3D(
        N=16,
        boundary_enabled=True,
        grace_enabled=True,
        F_MIN_grace=0.15,
    )
    sim = DETCollider3DBoundary(params)
    
    # Zone A (low z): Crisis - low F
    sim.F[:, :, :8] = 0.03
    
    # Zone B (high z): Stable - high F
    sim.F[:, :, 8:] = 0.5
    
    # Add flow in Zone A
    sim.add_packet((8, 8, 3), mass=0.8, width=1.5, momentum=(0.2, 0, 0))
    
    grace_A_total = 0.0
    grace_B_total = 0.0
    
    for _ in range(100):
        sim.step()
        grace_A_total += np.sum(sim.last_grace_injection[:, :, :8])
        grace_B_total += np.sum(sim.last_grace_injection[:, :, 8:])
    
    print(f"\n  Zone A (crisis - z<8):")
    print(f"    Initial <F>: 0.03 (below 0.15)")
    print(f"    Final <F>: {np.mean(sim.F[:, :, :8]):.4f}")
    print(f"    Total grace received: {grace_A_total:.6f}")
    
    print(f"\n  Zone B (stable - z≥8):")
    print(f"    Initial <F>: 0.5 (above 0.15)")
    print(f"    Final <F>: {np.mean(sim.F[:, :, 8:]):.4f}")
    print(f"    Total grace received: {grace_B_total:.6f}")
    
    passed = grace_A_total > grace_B_total * 5
    print(f"\n  F3 Locality Test (3D): {'PASSED ✓' if passed else 'FAILED ✗'}")
    if passed:
        print("  → Grace went preferentially to needy zone A")
    return passed


def run_all_3d_tests():
    """Run all F2/F3 tests for the 3D collider."""
    print("=" * 70)
    print("DET v6.2 3D BOUNDARY OPERATOR TEST SUITE")
    print("=" * 70)
    print("\nThis tests boundary operators in 3D:")
    print("  1. Strictly local (3D neighborhood)")
    print("  2. Agency-gated (a=0 → no action)")
    print("  3. Qualitatively different outcomes when enabled")
    
    results = {}
    
    # F2 Tests
    print("\n" + "=" * 70)
    print("F2 COERCION TEST SUITE (3D)")
    print("=" * 70)
    
    results['f2_grace'] = test_f2_grace_coercion_3d()
    
    # F3 Tests
    print("\n" + "=" * 70)
    print("F3 BOUNDARY REDUNDANCY TEST SUITE (3D)")
    print("=" * 70)
    
    results['f3_scarcity'] = test_f3_scarcity_recovery_3d()
    results['f3_locality'] = test_f3_locality_3d()
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY (3D)")
    print("=" * 70)
    
    f2_pass = results['f2_grace']
    f3_pass = results['f3_scarcity'] and results['f3_locality']
    
    print(f"  F2 (Coercion): {'PASSED ✓' if f2_pass else 'FAILED ✗'}")
    print(f"  F3 (Boundary Redundancy): {'PASSED ✓' if f3_pass else 'FAILED ✗'}")
    print(f"\n  OVERALL: {'ALL TESTS PASSED ✓' if f2_pass and f3_pass else 'SOME TESTS FAILED ✗'}")
    
    return f2_pass and f3_pass


if __name__ == "__main__":
    run_all_3d_tests()
