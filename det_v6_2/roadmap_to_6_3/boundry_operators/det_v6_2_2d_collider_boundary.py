"""
DET v6.2 2D Collider with Boundary Operators
============================================

This implementation adds the DET boundary operator module (Section VI) to the 2D collider:
- Grace injection (VI.5): Agency-gated resource injection to needy nodes
- Bond healing (optional): Agency-gated coherence recovery

Key Features:
- Strictly local operations (2D neighborhood)
- Agency inviolability (VI.1): Boundary operators never modify a_i directly
- F2 Falsifier: a=0 blocks all boundary action
- F3 Falsifier: Boundary on/off produces qualitatively different outcomes

Reference: DET Theory Card v6.2, Section VI
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from scipy.fft import fft2, ifft2


@dataclass
class DETParams2D:
    """DET 2D simulation parameters with boundary operators."""
    N: int = 64                     # Grid size (N x N)
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
    R_boundary: int = 3             # Radius for local normalization
    
    # Numerical stability
    outflow_limit: float = 0.25


def periodic_local_sum_2d(x: np.ndarray, radius: int) -> np.ndarray:
    """Compute local sum within radius (periodic BCs) in 2D."""
    result = np.zeros_like(x)
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            result += np.roll(np.roll(x, di, axis=0), dj, axis=1)
    return result


class DETCollider2DBoundary:
    """
    DET v6.2 2D Collider with Boundary Operators
    
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
    
    def __init__(self, params: Optional[DETParams2D] = None):
        self.p = params or DETParams2D()
        N = self.p.N
        
        # Per-node state (II.1) - 2D arrays
        self.F = np.ones((N, N)) * self.p.F_VAC
        self.q = np.zeros((N, N))
        self.a = np.ones((N, N))
        
        # Per-bond state (II.2) - 4 directions: E, W, N, S
        self.pi_E = np.zeros((N, N))  # East-directed momentum
        self.pi_S = np.zeros((N, N))  # South-directed momentum
        self.C_E = np.ones((N, N)) * self.p.C_init  # East bond coherence
        self.C_S = np.ones((N, N)) * self.p.C_init  # South bond coherence
        self.sigma = np.ones((N, N))  # Conductivity (isotropic)
        
        # Diagnostics
        self.time = 0.0
        self.step_count = 0
        self.P = np.ones((N, N))
        self.Delta_tau = np.ones((N, N)) * self.p.DT
        
        # Boundary operator diagnostics
        self.last_grace_injection = np.zeros((N, N))
        self.last_healing_E = np.zeros((N, N))
        self.last_healing_S = np.zeros((N, N))
        self.total_grace_injected = 0.0
    
    def add_packet(self, center_i: int, center_j: int, mass: float = 5.0, 
                   width: float = 5.0, momentum: Tuple[float, float] = (0, 0)):
        """Add a Gaussian resource packet with optional initial momentum."""
        N = self.p.N
        i_arr, j_arr = np.ogrid[:N, :N]
        
        # Distance from center (with periodic wrapping)
        di = np.minimum(np.abs(i_arr - center_i), N - np.abs(i_arr - center_i))
        dj = np.minimum(np.abs(j_arr - center_j), N - np.abs(j_arr - center_j))
        r2 = di**2 + dj**2
        
        envelope = np.exp(-0.5 * r2 / width**2)
        
        self.F += mass * envelope
        self.C_E = np.clip(self.C_E + 0.5 * envelope, self.p.C_init, 1.0)
        self.C_S = np.clip(self.C_S + 0.5 * envelope, self.p.C_init, 1.0)
        
        # Initial momentum
        mom_i, mom_j = momentum
        if mom_i != 0 or mom_j != 0:
            mom_env = np.exp(-0.5 * r2 / (width * 2)**2)
            self.pi_E += mom_j * mom_env  # East = +j direction
            self.pi_S += mom_i * mom_env  # South = +i direction
        
        self._clip()
    
    def _clip(self):
        """Enforce physical bounds on state variables."""
        self.F = np.clip(self.F, self.p.F_MIN, 1000)
        self.q = np.clip(self.q, 0, 1)
        self.a = np.clip(self.a, 0, 1)
        self.pi_E = np.clip(self.pi_E, -self.p.pi_max, self.p.pi_max)
        self.pi_S = np.clip(self.pi_S, -self.p.pi_max, self.p.pi_max)
    
    def compute_grace_injection(self, D: np.ndarray) -> np.ndarray:
        """
        Grace Injection per DET VI.5 (2D version)
        
        I_{g→i} = D_i · w_i / (Σ_{k∈N_R(i)} w_k + ε)
        
        where:
        - n_i = max(0, F_min_grace - F_i)  # Need
        - w_i = a_i · n_i                   # Weight (agency-gated)
        - D_i = local dissipation
        
        CRITICAL: If a_i = 0, then w_i = 0, so I_{g→i} = 0
        This is the F2 (Coercion) guarantee.
        """
        p = self.p
        
        # Need: how far below threshold
        n = np.maximum(0, p.F_MIN_grace - self.F)
        
        # Weight: AGENCY-GATED need
        w = self.a * n
        
        # Local normalization (within 2D neighborhood)
        w_sum = periodic_local_sum_2d(w, p.R_boundary) + 1e-12
        
        # Injection amount
        I_g = D * w / w_sum
        
        return I_g
    
    def compute_bond_healing(self, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optional Bond Healing Operator (Agency-Gated) - 2D version
        
        Returns healing increment for East and South bonds.
        """
        p = self.p
        E = lambda x: np.roll(x, -1, axis=1)  # East neighbor
        S = lambda x: np.roll(x, -1, axis=0)  # South neighbor
        
        # Agency gate: BOTH endpoints must be open
        g_E = np.sqrt(self.a * E(self.a))
        g_S = np.sqrt(self.a * S(self.a))
        
        # Healing room
        room_E = 1.0 - self.C_E
        room_S = 1.0 - self.C_S
        
        # Average dissipation at bond
        D_avg_E = 0.5 * (D + E(D))
        D_avg_S = 0.5 * (D + S(D))
        
        # Bond-local time step
        Delta_tau_E = 0.5 * (self.Delta_tau + E(self.Delta_tau))
        Delta_tau_S = 0.5 * (self.Delta_tau + S(self.Delta_tau))
        
        # Healing increment
        dC_heal_E = p.eta_heal * g_E * room_E * D_avg_E * Delta_tau_E
        dC_heal_S = p.eta_heal * g_S * room_S * D_avg_S * Delta_tau_S
        
        return dC_heal_E, dC_heal_S
    
    def step(self):
        """
        Execute one canonical DET update step with boundary operators (2D).
        """
        p = self.p
        N = p.N
        dk = p.DT
        
        # Neighbor access operators (periodic BCs)
        E = lambda x: np.roll(x, -1, axis=1)  # East (j+1)
        W = lambda x: np.roll(x, 1, axis=1)   # West (j-1)
        S = lambda x: np.roll(x, -1, axis=0)  # South (i+1)
        N_ = lambda x: np.roll(x, 1, axis=0)  # North (i-1)
        
        # ============================================================
        # STEP 1: Presence and proper time (III.1)
        # ============================================================
        H = self.sigma
        self.P = self.a * self.sigma / (1.0 + self.F) / (1.0 + H)
        self.Delta_tau = self.P * dk
        
        Delta_tau_E = 0.5 * (self.Delta_tau + E(self.Delta_tau))
        Delta_tau_S = 0.5 * (self.Delta_tau + S(self.Delta_tau))
        
        # ============================================================
        # STEP 2: Flow computation (4 directions)
        # ============================================================
        
        # Classical (pressure) contributions
        classical_E = self.F - E(self.F)
        classical_W = self.F - W(self.F)
        classical_S = self.F - S(self.F)
        classical_N = self.F - N_(self.F)
        
        # Coherence interpolation
        sqrt_C_E = np.sqrt(self.C_E)
        sqrt_C_W = np.sqrt(W(self.C_E))
        sqrt_C_S = np.sqrt(self.C_S)
        sqrt_C_N = np.sqrt(N_(self.C_S))
        
        # Combined drive (pressure only - phase not implemented)
        drive_E = (1 - sqrt_C_E) * classical_E
        drive_W = (1 - sqrt_C_W) * classical_W
        drive_S = (1 - sqrt_C_S) * classical_S
        drive_N = (1 - sqrt_C_N) * classical_N
        
        # Agency-gated diffusion (IV.2)
        g_E = np.sqrt(self.a * E(self.a))
        g_W = np.sqrt(self.a * W(self.a))
        g_S = np.sqrt(self.a * S(self.a))
        g_N = np.sqrt(self.a * N_(self.a))
        
        # Conductivity
        cond_E = self.sigma * (self.C_E + 1e-4)
        cond_W = self.sigma * (W(self.C_E) + 1e-4)
        cond_S = self.sigma * (self.C_S + 1e-4)
        cond_N = self.sigma * (N_(self.C_S) + 1e-4)
        
        # Agency-gated diffusive flux
        J_diff_E = g_E * cond_E * drive_E
        J_diff_W = g_W * cond_W * drive_W
        J_diff_S = g_S * cond_S * drive_S
        J_diff_N = g_N * cond_N * drive_N
        
        # Momentum-driven flux (IV.4)
        if p.momentum_enabled:
            F_avg_E = 0.5 * (self.F + E(self.F))
            F_avg_W = 0.5 * (self.F + W(self.F))
            F_avg_S = 0.5 * (self.F + S(self.F))
            F_avg_N = 0.5 * (self.F + N_(self.F))
            
            J_mom_E = p.mu_pi * self.sigma * self.pi_E * F_avg_E
            J_mom_W = -p.mu_pi * self.sigma * W(self.pi_E) * F_avg_W
            J_mom_S = p.mu_pi * self.sigma * self.pi_S * F_avg_S
            J_mom_N = -p.mu_pi * self.sigma * N_(self.pi_S) * F_avg_N
        else:
            J_mom_E = J_mom_W = J_mom_S = J_mom_N = 0
        
        # Floor repulsion flux (IV.6)
        if p.floor_enabled:
            s = np.maximum(0, (self.F - p.F_core) / p.F_core)**p.floor_power
            J_floor_E = p.eta_floor * self.sigma * (s + E(s)) * classical_E
            J_floor_W = p.eta_floor * self.sigma * (s + W(s)) * classical_W
            J_floor_S = p.eta_floor * self.sigma * (s + S(s)) * classical_S
            J_floor_N = p.eta_floor * self.sigma * (s + N_(s)) * classical_N
        else:
            J_floor_E = J_floor_W = J_floor_S = J_floor_N = 0
        
        # Total flux per direction
        J_E = J_diff_E + J_mom_E + J_floor_E
        J_W = J_diff_W + J_mom_W + J_floor_W
        J_S = J_diff_S + J_mom_S + J_floor_S
        J_N = J_diff_N + J_mom_N + J_floor_N
        
        # ============================================================
        # STEP 3: Dissipation and limiter
        # ============================================================
        total_outflow = (np.maximum(0, J_E) + np.maximum(0, J_W) + 
                         np.maximum(0, J_S) + np.maximum(0, J_N))
        max_total_out = p.outflow_limit * self.F / (self.Delta_tau + 1e-9)
        scale = np.minimum(1.0, max_total_out / (total_outflow + 1e-9))
        
        J_E_lim = np.where(J_E > 0, J_E * scale, J_E)
        J_W_lim = np.where(J_W > 0, J_W * scale, J_W)
        J_S_lim = np.where(J_S > 0, J_S * scale, J_S)
        J_N_lim = np.where(J_N > 0, J_N * scale, J_N)
        
        J_diff_E_scaled = np.where(J_diff_E > 0, J_diff_E * scale, J_diff_E)
        J_diff_S_scaled = np.where(J_diff_S > 0, J_diff_S * scale, J_diff_S)
        
        # Dissipation (for grace injection and healing)
        D = (np.abs(J_E_lim) + np.abs(J_W_lim) + 
             np.abs(J_S_lim) + np.abs(J_N_lim)) * self.Delta_tau
        
        # ============================================================
        # STEP 4: Resource update (IV.7)
        # ============================================================
        transfer_E = J_E_lim * self.Delta_tau
        transfer_W = J_W_lim * self.Delta_tau
        transfer_S = J_S_lim * self.Delta_tau
        transfer_N = J_N_lim * self.Delta_tau
        
        outflow = transfer_E + transfer_W + transfer_S + transfer_N
        inflow = W(transfer_E) + E(transfer_W) + N_(transfer_S) + S(transfer_N)
        
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
            self.last_grace_injection = np.zeros((N, N))
        
        # ============================================================
        # STEP 6: Momentum update (IV.4)
        # ============================================================
        if p.momentum_enabled:
            decay_E = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_E)
            decay_S = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_S)
            self.pi_E = decay_E * self.pi_E + p.alpha_pi * J_diff_E_scaled * Delta_tau_E
            self.pi_S = decay_S * self.pi_S + p.alpha_pi * J_diff_S_scaled * Delta_tau_S
        
        # ============================================================
        # STEP 7: Structural update (canonical q-locking)
        # ============================================================
        if p.q_enabled:
            self.q = np.clip(self.q + p.alpha_q * np.maximum(0, -dF), 0, 1)
        
        # ============================================================
        # STEP 8: Bond Healing (optional) - BOUNDARY OPERATOR
        # ============================================================
        if p.boundary_enabled and p.healing_enabled:
            dC_heal_E, dC_heal_S = self.compute_bond_healing(D)
            self.C_E = np.clip(self.C_E + dC_heal_E, p.C_init, 1.0)
            self.C_S = np.clip(self.C_S + dC_heal_S, p.C_init, 1.0)
            self.last_healing_E = dC_heal_E.copy()
            self.last_healing_S = dC_heal_S.copy()
        else:
            self.last_healing_E = np.zeros((N, N))
            self.last_healing_S = np.zeros((N, N))
        
        # ============================================================
        # STEP 9: Agency update (VI.2B)
        # ============================================================
        a_target = 1.0 / (1.0 + p.a_coupling * self.q**2)
        self.a = self.a + p.a_rate * (a_target - self.a)
        
        self._clip()
        self.time += dk
        self.step_count += 1
    
    def total_mass(self) -> float:
        return float(np.sum(self.F))
    
    def center_of_mass(self) -> Tuple[float, float]:
        N = self.p.N
        i_arr, j_arr = np.ogrid[:N, :N]
        total = np.sum(self.F) + 1e-9
        com_i = float(np.sum(i_arr * self.F) / total)
        com_j = float(np.sum(j_arr * self.F) / total)
        return com_i, com_j


# ======================================================================
# F2/F3 FALSIFIER TESTS FOR 2D
# ======================================================================

def test_f2_grace_coercion_2d():
    """
    F2 Test A (2D): Hard-zero agency sentinel
    
    Setup: Central node with a=0, F << F_MIN_grace
    Verify: I_{g→s} = 0 exactly
    """
    print("=" * 70)
    print("F2 COERCION TEST A (2D): Hard-Zero Agency Sentinel (Grace)")
    print("=" * 70)
    
    params = DETParams2D(
        N=32, 
        boundary_enabled=True, 
        grace_enabled=True,
        F_MIN_grace=0.15,
        a_rate=0.0,  # Freeze agency
    )
    sim = DETCollider2DBoundary(params)
    
    # Create collision scenario with depleted center
    sim.add_packet(16, 10, mass=3.0, width=3.0, momentum=(0, 0.5))
    sim.add_packet(16, 22, mass=3.0, width=3.0, momentum=(0, -0.5))
    
    # Set sentinel at center
    sentinel_i, sentinel_j = 16, 16
    sim.a[sentinel_i, sentinel_j] = 0.0
    sim.F[sentinel_i, sentinel_j] = 0.01  # Very needy
    
    # Run until collision creates dissipation
    for _ in range(200):
        sim.step()
    
    sentinel_grace = sim.last_grace_injection[sentinel_i, sentinel_j]
    neighbor_grace = (sim.last_grace_injection[sentinel_i-1, sentinel_j] +
                      sim.last_grace_injection[sentinel_i+1, sentinel_j] +
                      sim.last_grace_injection[sentinel_i, sentinel_j-1] +
                      sim.last_grace_injection[sentinel_i, sentinel_j+1])
    
    print(f"\n  Sentinel (a=0) at ({sentinel_i}, {sentinel_j})")
    print(f"  Sentinel a (should be 0): {sim.a[sentinel_i, sentinel_j]:.4f}")
    print(f"  Sentinel F = {sim.F[sentinel_i, sentinel_j]:.4f} (needy)")
    print(f"  Grace received by sentinel: {sentinel_grace:.2e}")
    print(f"  Grace to 4 neighbors: {neighbor_grace:.4f}")
    print(f"  Total grace injected: {sim.total_grace_injected:.4f}")
    
    passed = sentinel_grace == 0.0
    print(f"\n  F2 Grace Test (2D): {'PASSED ✓' if passed else 'FAILED ✗'}")
    if passed:
        print("  → Agency gate correctly blocks grace to a=0 node")
    return passed


def test_f2_healing_coercion_2d():
    """
    F2 Test B (2D): Bond-heal coercion
    
    Setup: Bond with a_i=0, a_j=1, low coherence
    Verify: ΔC^{heal}_{ij} = 0
    """
    print("\n" + "=" * 70)
    print("F2 COERCION TEST B (2D): Bond-Heal Coercion")
    print("=" * 70)
    
    params = DETParams2D(
        N=32, 
        boundary_enabled=True, 
        healing_enabled=True,
        eta_heal=0.1,
        a_rate=0.0,
    )
    sim = DETCollider2DBoundary(params)
    
    # Create flow conditions
    sim.add_packet(16, 10, mass=3.0, width=3.0, momentum=(0, 0.3))
    sim.add_packet(16, 22, mass=3.0, width=3.0, momentum=(0, -0.3))
    
    # Test bond at (16, 16) → (16, 17)
    test_i, test_j = 16, 16
    sim.a[test_i, test_j] = 0.0
    sim.a[test_i, test_j + 1] = 1.0
    sim.C_E[test_i, test_j] = 0.3  # Low coherence
    
    # Control bond at (10, 16) → (10, 17) with both a=1
    ctrl_i, ctrl_j = 10, 16
    sim.a[ctrl_i, ctrl_j] = 1.0
    sim.a[ctrl_i, ctrl_j + 1] = 1.0
    sim.C_E[ctrl_i, ctrl_j] = 0.3
    
    for _ in range(100):
        sim.step()
    
    test_heal = sim.last_healing_E[test_i, test_j]
    ctrl_heal = sim.last_healing_E[ctrl_i, ctrl_j]
    
    print(f"\n  Test bond at ({test_i}, {test_j})")
    print(f"  Left endpoint a = {sim.a[test_i, test_j]} (should be 0)")
    print(f"  Right endpoint a = {sim.a[test_i, test_j + 1]}")
    print(f"  Max healing received (test): {test_heal:.2e}")
    print(f"  Max healing received (control): {ctrl_heal:.2e}")
    
    passed = test_heal == 0.0
    print(f"\n  F2 Healing Test (2D): {'PASSED ✓' if passed else 'FAILED ✗'}")
    if passed:
        print("  → Agency gate correctly blocks healing when either endpoint has a=0")
    return passed


def test_f3_scarcity_recovery_2d():
    """
    F3 Test D (2D): Scarcity collapse vs recovery
    
    Compare boundary ON vs OFF in harsh collision scenario.
    """
    print("\n" + "=" * 70)
    print("F3 REDUNDANCY TEST D (2D): Scarcity Collapse vs Recovery")
    print("=" * 70)
    
    def run_scenario(boundary_on: bool):
        params = DETParams2D(
            N=32,
            F_VAC=0.02,
            boundary_enabled=boundary_on,
            grace_enabled=True,
            F_MIN_grace=0.15,
        )
        sim = DETCollider2DBoundary(params)
        
        # Head-on collision
        sim.add_packet(16, 8, mass=2.0, width=2.5, momentum=(0, 0.4))
        sim.add_packet(16, 24, mass=2.0, width=2.5, momentum=(0, -0.4))
        
        # Run collision
        for _ in range(300):
            sim.step()
        
        # Measure collision zone (central region)
        zone = sim.F[14:19, 14:19]
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
    F_diff = F_on - F_off
    qualitative = grace_diff > 0.001
    
    print(f"\n  Boundary ON vs OFF comparison:")
    print(f"    Grace difference: {grace_diff:.4f}")
    print(f"    Final F difference: {F_diff:.4f}")
    print(f"    Qualitative difference: {qualitative}")
    
    passed = qualitative
    print(f"\n  F3 Scarcity Test (2D): {'PASSED ✓' if passed else 'FAILED ✗'}")
    if passed:
        print("  → Boundary ON produces measurably different outcome")
    return passed


def test_f3_locality_2d():
    """
    F3 Test E (2D): Local crisis, local response
    
    Zone A (crisis) should receive grace, Zone B (stable) should not.
    """
    print("\n" + "=" * 70)
    print("F3 LOCALITY TEST E (2D): Local Crisis, Local Response")
    print("=" * 70)
    
    params = DETParams2D(
        N=32,
        boundary_enabled=True,
        grace_enabled=True,
        F_MIN_grace=0.15,
    )
    sim = DETCollider2DBoundary(params)
    
    # Zone A (top half): Crisis - low F
    sim.F[:16, :] = 0.03
    
    # Zone B (bottom half): Stable - high F
    sim.F[16:, :] = 0.5
    
    # Add some flow in Zone A
    sim.add_packet(8, 10, mass=1.0, width=2.0, momentum=(0, 0.3))
    sim.add_packet(8, 22, mass=1.0, width=2.0, momentum=(0, -0.3))
    
    grace_A_total = 0.0
    grace_B_total = 0.0
    
    for _ in range(200):
        sim.step()
        grace_A_total += np.sum(sim.last_grace_injection[:16, :])
        grace_B_total += np.sum(sim.last_grace_injection[16:, :])
    
    F_A_final = np.mean(sim.F[:16, :])
    F_B_final = np.mean(sim.F[16:, :])
    
    print(f"\n  Zone A (crisis - top half):")
    print(f"    Initial <F>: 0.03 (below 0.15)")
    print(f"    Final <F>: {F_A_final:.4f}")
    print(f"    Total grace received: {grace_A_total:.6f}")
    
    print(f"\n  Zone B (stable - bottom half):")
    print(f"    Initial <F>: 0.5 (above 0.15)")
    print(f"    Final <F>: {F_B_final:.4f}")
    print(f"    Total grace received: {grace_B_total:.6f}")
    
    passed = grace_A_total > grace_B_total * 10  # A should get much more
    print(f"\n  F3 Locality Test (2D): {'PASSED ✓' if passed else 'FAILED ✗'}")
    if passed:
        print("  → Grace went preferentially to needy zone A")
    return passed


def run_all_2d_tests():
    """Run all F2/F3 tests for the 2D collider."""
    print("=" * 70)
    print("DET v6.2 2D BOUNDARY OPERATOR TEST SUITE")
    print("=" * 70)
    print("\nThis tests boundary operators in 2D:")
    print("  1. Strictly local (2D neighborhood)")
    print("  2. Agency-gated (a=0 → no action)")
    print("  3. Qualitatively different outcomes when enabled")
    
    results = {}
    
    # F2 Tests
    print("\n" + "=" * 70)
    print("F2 COERCION TEST SUITE (2D)")
    print("=" * 70)
    
    results['f2_grace'] = test_f2_grace_coercion_2d()
    results['f2_healing'] = test_f2_healing_coercion_2d()
    
    # F3 Tests
    print("\n" + "=" * 70)
    print("F3 BOUNDARY REDUNDANCY TEST SUITE (2D)")
    print("=" * 70)
    
    results['f3_scarcity'] = test_f3_scarcity_recovery_2d()
    results['f3_locality'] = test_f3_locality_2d()
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY (2D)")
    print("=" * 70)
    
    f2_pass = results['f2_grace'] and results['f2_healing']
    f3_pass = results['f3_scarcity'] and results['f3_locality']
    
    print(f"  F2 (Coercion): {'PASSED ✓' if f2_pass else 'FAILED ✗'}")
    print(f"  F3 (Boundary Redundancy): {'PASSED ✓' if f3_pass else 'FAILED ✗'}")
    print(f"\n  OVERALL: {'ALL TESTS PASSED ✓' if f2_pass and f3_pass else 'SOME TESTS FAILED ✗'}")
    
    return f2_pass and f3_pass


if __name__ == "__main__":
    run_all_2d_tests()
